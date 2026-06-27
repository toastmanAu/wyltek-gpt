# cellc ↔ wyltek-gpt — Phase 3 (loop tools, prompt grounding, confirmed save)

**Date:** 2026-06-27
**Status:** Approved design, pre-implementation
**Repo:** `~/local-chatbot` (wyltek-gpt), branch `feat/cellc-phase3`, driveThree
**Builds on:** Phase 2 (the agentic loop in `/api/chat`, the cellc bridge with
`tool_schemas()`/`dispatch()`). Reuses both; adds two loop tools, a system-prompt
grounding block, and a confirmed save operation.

## Purpose

Three additive enhancements to the Phase 2 agentic loop:
- **A.** Give the model two more read-only loop tools (`cellc_metadata`,
  `cellc_list_examples`) so it can inspect a contract's semantics or browse
  examples while iterating.
- **B.** Inject the CellScript language reference into the system prompt on
  CellScript-related chats so the model writes valid syntax first try (fewer
  failed checks / loop iterations).
- **C.** Let the model save a checked contract to the session workspace via the
  existing Y/N/E confirm card — a real `.cell` artifact, re-verified before write.

## Core design line: read = auto-loop, write = confirm card

The Phase 2 cellc tools auto-run because they are read-only and the stream
flows. A **write must be human-gated**, so `cellc_save` is NOT a cellc auto-loop
tool — it is a registered OPERATIONS op that flows through the existing
`__tool_calls__` → confirm-card → `/api/operations/run` path. This avoids the
hardest streaming problem (pausing a stream mid-loop for a click): the model
proposes the save as its turn's tool call, the loop does not execute it, and the
established op confirm flow takes over.

## Component A — `cellc_metadata` + `cellc_list_examples` loop tools

`backend/bridges/cellc.py`:
- `CELLC_TOOL_NAMES` grows from 4 to 6 (add `cellc_metadata`, `cellc_list_examples`).
- `tool_schemas()` adds two schemas:
  - `cellc_metadata` — params `{source: string (required), target_profile?: enum[ckb]}`
  - `cellc_list_examples` — params `{}`
- `dispatch()` keeps the existing `cellc_metadata` arm (currently dead — becomes
  live) and adds a `cellc_list_examples` arm → `{"examples": list_examples()}`.
  Remove the dead-code ambiguity: `cellc_metadata` validation matches `cellc_check`.
- `backend/app.py` `_summarize_cellc_step`: add summaries — `cellc_metadata` →
  `metadata: <N> resources, <M> actions` (from the summary dict's count keys);
  `cellc_list_examples` → `<N> examples`.

No frontend change (rendered by the existing `__cellc_step__` path). These are
read-only → they stay in the auto-loop.

## Component B — language reference in the system prompt (cellc-intent gated)

`_full_system_prompt()` is also used by `/api/config` without message context, so
the full reference must NOT go there unconditionally (token bloat on every chat).

- `_full_system_prompt()` gains a **one-line hint** appended when
  `cellc_bridge.available()`: e.g. *"CellScript (.cell) tooling is available:
  call cellc_check to verify contracts and cellc_language_reference for syntax."*
- `/api/chat` prepends the **full** language reference to the system message ONLY
  when `cellc_bridge.available()` AND the latest user message matches cellc-intent.
  - `_is_cellc_intent(text) -> bool`: case-insensitive match on any of
    `cellscript`, `.cell`, `cell contract`, `ckb contract`, `nervos contract`,
    `resource ...has`, `cellc`. Keep the regex small and in one place.
  - The reference text comes from `cellc_bridge.language_reference()`.
  - Prepended as an additional system block (after the base system prompt),
    so the existing host-context/operations blocks are unchanged.

The full reference's token cost is paid only on CellScript chats; everyone else
gets the one-liner (or nothing when cellc is unavailable).

## Component C — `cellc_save` confirmed workspace write

### Tool exposure + routing
- A synthetic `cellc_save` Operation is registered at startup via
  `OPERATIONS.add(...)` ONLY when `cellc_bridge.available()`. Because it is a
  registered op, it appears in `OPERATIONS.tool_schemas()` automatically, so the
  model sees it. It is NOT added to `CELLC_TOOL_NAMES`, so the Phase 2 partition
  routes a `cellc_save` call to `op_calls` → the existing `__tool_calls__` →
  confirm-card path (NOT auto-run).

### The synthetic Operation
- `id = "cellc_save"`, `kind = "cellc_save"` (new kind),
  `description` = "Save a checked CellScript contract to the workspace as
  <name>.cell. Only call after cellc_check passes.",
  `params`:
  - `name` — `TextValidator(max_len=64)`, required (the file stem; further
    constrained to `^[A-Za-z0-9_-]+$` in the handler).
  - `source` — `TextValidator(max_len=200_000)`, required (the `.cell` content).
  Built with the same `Param`/validator constructs the converter op uses;
  `to_tool_schema()` produces the Ollama schema automatically.

### Execution handler
- `/api/operations/run` gains an arm: `elif op.kind == "cellc_save": out = await
  _run_cellc_save_op(op, session_dir, validated)`.
- `_run_cellc_save_op(op, session_dir, validated) -> Path`:
  1. `name` must match `^[A-Za-z0-9_-]+$` (else 400) — defense-in-depth over the
     TextValidator.
  2. **Re-check before write**: `result = cellc_bridge.check(validated["source"])`.
     If `not result.get("ok")` → `HTTPException(400, "refusing to save: cellc_check
     failed: <first diagnostics>")`. A failing contract is never written.
  3. Write `validated["source"]` (UTF-8) to `session_dir / f"{name}.cell"`
     (path-validated via the existing `_resolve_in_workspace` discipline /
     `_assert_inside`).
  4. Return the Path (mirrored to OUTPUT_DIR like other ops; `/api/operations/run`
     returns the existing `{name, url, size, via, kind}` shape).
- This makes `cellc_save` a file-producing op exactly like the others — it fits
  `/api/operations/run`'s "returns a Path" contract.

### Frontend
The existing `presentOpConfirm`/`buildOpCard` renders any registered op generically
from its metadata, so `cellc_save` should surface in the confirm card with no JS
change. If the `source` text param needs a larger input affordance than the card
provides, add a minimal tweak — but treat that as optional and verify via the
manual smoke first (no speculative JS).

## Error handling

- A/B/C all gate on `cellc_bridge.available()` — unavailable cellc means the two
  new tools aren't advertised, the prompt block is omitted, and `cellc_save` is
  not registered. Graceful no-op, app boots normally.
- `dispatch` still never raises (Phase 2 contract preserved for the two new arms).
- `cellc_save` re-check failure returns a clear 400 through the existing op error
  path; the confirm card shows the reason. A bad `name` → 400. Path traversal is
  blocked by the name regex + `_assert_inside`.
- `cellc_save` writing is the only write in the whole cellc integration; it is
  gated by the confirm card by construction (never auto-run).

## Testing

- **A:** extend `tests/test_cellc_dispatch.py` — `tool_schemas()` now returns 6
  named tools; `dispatch("cellc_metadata", {source})` and
  `dispatch("cellc_list_examples", {})` forward correctly; `CELLC_TOOL_NAMES` has 6.
- **B:** `tests/test_cellc_prompt.py` — `_is_cellc_intent` true/false cases;
  `/api/chat` (mock `_stream_one`, mock `available()` True) prepends the reference
  when the user message is cellc-intent and omits it otherwise; `_full_system_prompt`
  contains the one-line hint when available.
- **C:** `tests/test_cellc_save.py` — `_run_cellc_save_op` writes `<name>.cell` on a
  passing check (mock `cellc_bridge.check` → ok), refuses (400) on a failing check,
  rejects a bad `name`; the synthetic op is registered only when available; a
  `cellc_save` tool call routes to `op_calls` (not the auto-loop) via the Phase 2
  partition.
- **Integration** (`needs_cellc`, gated): `_run_cellc_save_op` with the REAL binary
  — a good contract writes the file; a bad contract is refused.
- **No regressions:** full suite stays green; existing op/confirm/loop behavior
  unchanged.

## Git safety

`feat/cellc-phase3` off `main`. `main` still carries the user's uncommitted WIP
(`.gitignore`, `data/capabilities.json`, `frontend/index.html`,
`frontend/style.css`) — NEVER `git add -A`; stage only Phase-3 files per task.
`backend/app.py`, `backend/bridges/cellc.py`, `backend/operations.py`,
`frontend/app.js` are clean of that WIP (committed earlier).

## Non-Goals

- No downstream cellc workspace op yet (deploy/convert the saved `.cell`) — that is
  future work; Component C just lands the artifact.
- No change to the read=auto / write=confirm split; no auto-write path.
- No new confirm-card mechanism — reuse the existing op flow.
