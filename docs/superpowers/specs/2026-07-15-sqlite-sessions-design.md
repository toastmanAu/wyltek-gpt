# SQLite server-side sessions — design spec

**Date:** 2026-07-15
**Status:** Approved (brainstorm), pending implementation plan
**Gap reference:** `docs/superpowers/research/2026-07-15-opencode-harness-gap-analysis.md` — Gap #2 (server-side session persistence), the enabling foundation for Gap #1 (conversation compaction).

## Why

Chat history currently lives client-side: the browser POSTs the full `messages` array to `/api/chat` every turn, and `app.py chat()` rebuilds the model context from that incoming list. There is no server-side history, so there is:

- no resume or cross-device continuity (a real loss for a PWA installed on phone *and* desktop),
- nowhere authoritative for a compaction summary to live, and
- unbounded growth — the window fills with stale turns until the model truncates mid-stream (the code even bumps `_CHAT_DEFAULT_NUM_CTX` to 24576 to paper over it).

This spec covers **Phase 1: server-side sessions as the source of truth**. It is deliberately scoped so that **Phase 2 (compaction/auto-summarization)** becomes a thin layer on top — a `summarize()` call plus one `UPDATE`. Phase 2 has its own spec.

## Scope

**In scope (Phase 1):**
- A `SessionStore` persistence module backed by SQLite.
- Session CRUD endpoints.
- `/api/chat` becomes dual-mode: server-backed **session mode** and the existing stateless **legacy mode** (kept permanently as a fallback).
- Frontend sessions sidebar: list / new / open / resume / delete.
- Config-gated auto-title.

**Out of scope (Phase 2, documented as hooks here):**
- Summarization/compaction logic and its trigger.
- Dynamic per-model `num_ctx` (separate coupled task; the compaction trigger will be a % of it).

## Architecture

New module **`backend/sessions.py`** exposing a `SessionStore` class, following the existing single-uvicorn-worker patterns of `CapabilityCache` and `html_demo.DemoStore` (cross-process writes are not a concern). It owns a SQLite database at `data/sessions.db`, opened in **WAL mode** with a short `busy_timeout`.

**Module boundaries:**
- `SessionStore` — pure persistence, no HTTP, no SQL leaks upward. Unit-testable against an in-memory SQLite.
- `app.py` — thin endpoint wrappers + `/api/chat` wiring. No SQL in the endpoint layer.

### Data model

```sql
CREATE TABLE sessions (
  id               TEXT PRIMARY KEY,
  title            TEXT,
  summary          TEXT,             -- Phase 2: compaction summary (NULL in Phase 1)
  summary_upto_seq INTEGER NOT NULL DEFAULT 0,  -- Phase 2 watermark
  model            TEXT,
  created          REAL NOT NULL,
  updated          REAL NOT NULL
);

CREATE TABLE messages (
  session_id TEXT NOT NULL,
  seq        INTEGER NOT NULL,       -- monotonic per session, 1-based
  role       TEXT NOT NULL,          -- 'user' | 'assistant'
  content    TEXT NOT NULL,          -- final visible text (no thinking/tool steps)
  tokens     INTEGER,                -- eval_count for assistant turns, NULL/estimate for user
  created    REAL NOT NULL,
  PRIMARY KEY (session_id, seq)
);
CREATE INDEX idx_messages_session ON messages(session_id, seq);
```

### The `summary_upto_seq` watermark

The single mechanism that makes Phase 2 cheap. The model context for a session is **always** computed as:

```
[system prompt]
  + ([summary] as one synthetic message, if sessions.summary is set)
  + messages WHERE seq > summary_upto_seq   (ordered by seq)
```

Compaction (Phase 2) never deletes or rewrites message rows. It writes `summary` and bumps `summary_upto_seq`. Old rows remain in the DB — the **full transcript is preserved** for resume/audit/full-view — but they drop out of the model's window. In Phase 1, `summary` is always NULL and `summary_upto_seq` is 0, so `load_context()` returns the whole transcript.

### `SessionStore` interface

```
create_session(model, title=None) -> id
append_message(session_id, role, content, tokens=None) -> seq
load_context(session_id) -> list[{role, content}]     # the replay list above (no system prompt; app.py prepends it)
list_sessions() -> list[{id, title, updated, model}]  # newest first
get_full(session_id) -> {title, summary, messages[]}  # entire transcript, for open/resume
rename(session_id, title)
delete(session_id)
set_summary(session_id, summary, upto_seq)            # Phase 2 only; present but unused in Phase 1
```

All methods wrap SQLite access; failures raise a `SessionStoreError` that the endpoint/chat layer catches and degrades on (see Error handling).

## API

```
POST   /api/sessions            -> {id}                              (create; body: {model, title?})
GET    /api/sessions            -> [{id, title, updated, model}]     (sidebar list, newest first)
GET    /api/sessions/{id}       -> {title, summary, messages[]}      (open / resume; 404 if unknown)
PATCH  /api/sessions/{id}       -> {ok}                              (rename; body: {title})
DELETE /api/sessions/{id}       -> {ok}
```

### `/api/chat` dual-mode

**Session mode** — request includes `session_id`:
1. `append_message(session_id, 'user', last_user_content)`.
2. Build context = `[system]` + `load_context(session_id)`; run existing injection (html_demo/cellc/text_files/images) on it exactly as today.
3. Stream to Ollama through the existing `relay()` machinery (tool loops unchanged).
4. **Accumulate the streamed assistant text server-side** as chunks flow; on the `done` event, `append_message(session_id, 'assistant', accumulated, tokens=eval_count)`.
5. Only the final visible assistant content is persisted — thinking channels and `__demo_step__`/`__cellc_step__` sentinels stay ephemeral (Ollama never needs prior thinking replayed).

**Legacy mode** — no `session_id`: exactly today's behaviour. Use incoming `messages`, persist nothing. Retained permanently as a stateless fallback for non-PWA callers.

## Frontend (`app.js`)

- A sessions **sidebar**: list (from `GET /api/sessions`), **new**, **open** (loads transcript via `GET /api/sessions/{id}` and renders it), **delete**, rename.
- Chat send switches to posting `session_id` + the new user message instead of the full array. A "New chat" creates a session lazily on first send.
- Existing localStorage transcripts are left as-is — **no migration UI** (YAGNI). Server-backed sessions start fresh from first use. Legacy-mode chat still works for anything not yet migrated.

## Auto-title

After the first assistant turn completes, fire-and-forget a short title via `auto_router.summary_model` (fallback order: `summary_model` -> `captioner_model` -> truncated first user message). Never blocks the chat stream. Config-gated: with no model configured, the truncated-first-message fallback is used. Users can `PATCH` to rename.

Config (`config.yaml`, extending the existing `auto_router` block):
```yaml
auto_router:
  captioner_model: qwen3-omni-captioner:30b-a3b-q4
  summary_model: ""        # used for titles (Phase 1) and compaction (Phase 2); empty disables
```

## Error handling

Follows the "never block on the feature" pattern already used by `html_demo`, converters, and the output mirror:

- Every `SessionStore` op is wrapped. A persistence failure **logs and lets the chat stream continue** — that turn simply isn't saved; the user still gets their answer.
- SQLite opened in WAL mode with a short `busy_timeout` to tolerate the rare overlapping write on the single worker.
- Unknown `session_id` on a session endpoint -> 404. Unknown `session_id` on `/api/chat` -> treat as a fresh session (create it) rather than erroring.
- Missing or unreadable DB file -> recreate schema on open; the app boots regardless (a failed store init disables session mode, leaving legacy mode working).

## Testing (TDD, 80%+ coverage)

**`SessionStore` unit tests** (in-memory SQLite, no HTTP):
- create/append/list ordering and `updated` bumping.
- `load_context` returns full transcript when no summary.
- **Watermark replay (headline):** after `set_summary(id, "S", upto=8)`, `load_context` returns `[{assistant,"S"}] + messages where seq > 8`, in order.
- `get_full` returns the entire transcript including pre-watermark rows.
- delete cascades messages; rename updates title.

**Endpoint integration tests** (FastAPI `TestClient`):
- `/api/sessions` CRUD happy paths + 404s.
- `/api/chat` **session mode**: a turn persists a user row then an assistant row with `tokens`, and streams as before.
- `/api/chat` **legacy mode**: no `session_id` -> nothing persisted, identical stream to today.
- Store-failure degradation: with the store forced to raise, session-mode chat still streams a complete answer.

## Phase 2 hooks (designed-in, not built here)

- `load_context()` already returns the compaction-ready replay — Phase 2 touches no read path.
- Trigger, computed after persisting the assistant turn: `fullness = prompt_eval_count / dynamic_num_ctx`; if `> compaction.threshold` (config, default ~0.8) run `maybe_compact(session_id)` asynchronously after the stream — summarize messages up to a keep-last-K window via `summary_model`, then `set_summary`.
- **Threshold is a percentage of the *dynamic* `num_ctx`** (the separate coupled task), so a tight 27B window compacts sooner than a roomy 9B window automatically, from the same model metadata.
- `summary_model` is the same config knob introduced for auto-title.

## Open questions / risks

- **Assistant-content accumulation** must capture exactly the visible text the model produced (post-thinking, post-tool-loop). The `relay()` generator already distinguishes `thinking`/`chunk`/tool events, so accumulation hooks the `chunk` path only. Verify the accumulation matches what the frontend renders.
- **Lazy session creation** on first send must be race-free on the single worker (create-then-append in one store call path).
- Frontend migration is intentionally omitted; if users have long localStorage chats they care about, a one-shot import is a later, separate nicety.

## Known limitations (Phase 1, accepted)

These are deliberate Phase-1 tradeoffs surfaced during the whole-branch review, documented so Phase 2 inherits them explicitly rather than by surprise:

- **History split-brain on mid-conversation store failure.** In session mode the client sends only the new user turn; the server loads prior history. If a turn's persistence fails (a transient `sqlite3` error caught and logged, or the assistant append failing after the stream), the server transcript diverges from what the browser shows. Chat keeps working (never breaks), but a reload can silently drop the un-persisted turn. Acceptable for a single-user local app; a stronger fix (reconcile on resume, or re-post full history on a signalled failure) is deferred.
- **No per-session ownership / auth.** Sessions and the store are global and unscoped. Fine for the single-user local deployment, but since "cross-device continuity" is a goal, the DB may become reachable from other LAN devices — any client can list/open/delete any session. Add scoping if multi-user ever matters.
- **Phase-2 trap — double system message.** `SUMMARY_ROLE = "system"` and `app.py` unconditionally prepends its own system prompt, so once Phase 2 sets a summary, `load_context` yields two back-to-back `system` messages. Inert in Phase 1 (summary always NULL); Phase 2 must use a non-system role or dedupe (marked with a comment at `backend/sessions.py`).
- **Synchronous SQLite on the event loop.** `SESSIONS.*` calls run directly in the async handlers (not via `asyncio.to_thread`), briefly blocking the loop. Negligible at single-user scale; revisit if concurrency grows.
