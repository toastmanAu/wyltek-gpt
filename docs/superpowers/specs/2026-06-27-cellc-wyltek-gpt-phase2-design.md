# cellc ↔ wyltek-gpt — Phase 2 (agentic tool-result loop)

**Date:** 2026-06-27
**Status:** Approved design, pre-implementation
**Repo:** `~/local-chatbot` (wyltek-gpt), branch `feat/cellc-phase2`, driveThree
**Builds on:** Phase 1 (`backend/bridges/cellc.py`, `/api/cellc/*`). Reuses the bridge unchanged except for two additions (tool schemas + dispatch).

## Purpose

Let a capable local model (qwen) self-iterate **write → check → fix** on
CellScript inside normal chat: the model emits a `cellc_check` tool call, the
backend auto-runs it (read-only, no confirm card), feeds the result back, and
the model continues — looping until it stops calling cellc tools or a cap is
hit. Each step is shown inline so the user sees the compiler actually ran.

This is the headline agentic capability. Phase 1 gave a human-driven endpoint;
Phase 2 closes the loop so the model drives it.

## Scope decisions (locked)

- Loop lives **inside the existing `/api/chat`** stream, gated to fire ONLY on
  cellc tool calls. File-op (`OPERATIONS`) tool calls keep their existing
  frontend confirm-card path, untouched.
- Tool steps are **transparent**: each cellc call + result summary renders
  inline as the model iterates.
- cellc tool calls **auto-run** (read-only; no Y/N/E confirm card).
- Loop tools exposed to the model: **4** — `cellc_check`, `cellc_explain`,
  `cellc_get_example`, `cellc_language_reference`. (`metadata`/`list_examples`
  deferred — fewer tools = less wander for a 27B.)

## Non-Goals

- No change to the file-op (`OPERATIONS`) confirm-card flow.
- No new chat endpoint or "mode" — one `/api/chat` path.
- No persistence of tool results beyond the single chat response.
- No exposure of cellc_mcp's `path=` parameter (only `source` text), preserving
  Phase 1's read-only posture.

## Architecture

The loop is added to the `stream()`/`relay()` async generators in `/api/chat`.
Today `relay` forwards a `tool_calls` event to the frontend and stops. Phase 2:

```
relay(source, messages, iterations):
  for kind, value in source:
    ... thinking/chunk/stats/error/done as today ...
    on "tool_calls":
      cellc_calls, op_calls = partition(value)            # by tool name
      if op_calls:
        yield __tool_calls__(op_calls)                    # existing path, unchanged
      if cellc_calls and iterations < MAX_CELLC_ITERS:
        assistant_msg = {role: assistant, content: "", tool_calls: cellc_calls}
        tool_msgs = []
        for c in cellc_calls:
          result = await to_thread(cellc_bridge.dispatch, c.name, c.args)
          yield __cellc_step__({tool: c.name, summary: summarize(c.name, result)})
          tool_msgs.append({role: tool, tool_name: c.name, content: json(result)})
        new_messages = messages + [assistant_msg, *tool_msgs]
        # re-invoke Ollama with the extended history; continue the SAME relay
        async for wire in relay(_stream_one(body(new_messages)), new_messages, iterations+1):
          yield wire
        return
      if cellc_calls and iterations >= MAX_CELLC_ITERS:
        yield inline note "(reached cellc check limit)"
        # do not re-invoke; let the model's current text stand
```

`MAX_CELLC_ITERS = 5` (module constant) bounds loop depth + cost. `messages` is
threaded through the recursion as the growing Ollama history.

### Tool exposure

`body_with_tools["tools"]` becomes `OPERATIONS.tool_schemas() +
cellc_bridge.tool_schemas()` — but the cellc schemas are appended **only when
`cellc_bridge.available()`**. If cellc is unavailable, no cellc schemas are
sent, the model can't call them, and the loop never triggers (graceful no-op).

The existing tools→no-tools fallback (model rejects function-calling) is
preserved unchanged.

## Component changes

### `backend/bridges/cellc.py` (two additions)

- `tool_schemas() -> list[dict]` — Ollama function-tool definitions matching the
  shape `OPERATIONS` already emits: `{"type": "function", "function": {"name",
  "description", "parameters": <JSON Schema>}}`. Four tools:
  - `cellc_check` — params `{source: string (required), target_profile?: enum[ckb]}`
  - `cellc_explain` — params `{code: string (required)}`
  - `cellc_get_example` — params `{name: string (required)}`
  - `cellc_language_reference` — params `{}` (no args)
- `dispatch(name: str, arguments: dict) -> dict` — validates the model-supplied
  arguments (same boundary rules as the Phase 1 endpoints: source non-empty +
  ≤ `_CELLC_MAX_SOURCE`, profile ∈ {ckb}, code regex, name regex) then calls the
  matching wrapper. On a validation failure it returns a `tool_error`-shaped
  dict (NOT raise) so the bad-args result is fed back to the model to retry.
  Unknown tool name → `tool_error` dict.

The Phase 1 wrappers (`check`/`explain`/`get_example`/`language_reference`) are
reused unchanged. Validation constants are shared with `app.py` (lift them into
the bridge or a small shared module so both the endpoints and `dispatch` use one
definition — avoid duplicating `_CELLC_MAX_SOURCE`/regexes).

### `backend/app.py` (the loop)

- Partition helper: cellc tool calls = those whose `function.name` is in the set
  of cellc tool names from `cellc_bridge.tool_schemas()`.
- Restructure `relay` to accept and thread `messages` + `iterations`, and to
  re-invoke `_stream_one` after executing cellc calls (per the pseudocode).
- New module constant `MAX_CELLC_ITERS = 5`.
- Merge cellc schemas into `body_with_tools["tools"]` when available.

### Ollama message formats (VERIFY at implementation time — flagged unknown)

Ollama 0.30.5. The re-invocation history appends:
- assistant turn: `{"role": "assistant", "content": "", "tool_calls": [<the cellc calls as received>]}`
- per result: `{"role": "tool", "tool_name": "<name>", "content": "<json string>"}`

The exact tool-result key (`tool_name` vs `name`) and whether `content` must be a
string vs object will be confirmed against the installed Ollama in the FIRST
implementation step (a tiny scripted round-trip), before building the loop on top.

### Transparent steps — wire format + frontend

- Backend emits, interleaved with model text:
  `"\n" + json.dumps({"__cellc_step__": {"tool": "cellc_check", "summary": "✗ 2 errors (L12, L20)"}})`
  — same newline-delimited-JSON-sentinel convention as `__tool_calls__`/`__stats__`.
- `summarize(name, result)` produces a short human string: check → `✓ passed` or
  `✗ N error(s) (L.., L..)`; explain → `<ecode>: <title>`; others → `ok`/`error`.
- **Frontend nuance:** unlike `__stats__`/`__tool_calls__` (one, at end, found via
  `lastIndexOf`), `__cellc_step__` lines appear **multiple times, interleaved**
  with text. The stream reader must extract EVERY `{"__cellc_step__": …}` line as
  it arrives, strip it from displayed text, and render each as an inline trace
  element (a small "🔧 tool → summary" line) in order. Acceptance is: all steps
  visible, in order, within the assistant message — exact pixel positioning
  relative to surrounding text is not required.

## Error handling

- `dispatch` never raises (cellc_mcp contract + arg validation returns
  `tool_error`); a `tool_error` result is fed to the model as the tool result so
  it can react, and its step summary shows e.g. `cellc error: …`.
- `available()` False → no cellc schemas sent → loop never triggers.
- Ollama error during a re-invocation → existing error path emits the inline
  `[backend error: …]` and stops; no hang.
- Cap reached → inline note, loop stops, model's current text stands.
- A malformed/again-cellc-calling response at the cap does not re-invoke.

## Testing

- **Unit** (mock `_stream_one` + `cellc_bridge`): 
  - partition separates cellc vs op calls;
  - a cellc call executes, appends correctly-shaped assistant+tool messages, and
    re-invokes (assert the second `_stream_one` body carries the tool result);
  - op calls pass through as `__tool_calls__` and do not loop;
  - the loop stops at `MAX_CELLC_ITERS`;
  - `dispatch` arg-validation returns `tool_error` for bad source/profile/code/name
    and for an unknown tool name.
- **Integration** (`needs_cellc`, gated): a scripted fake-Ollama generator that
  emits one `cellc_check` tool call for `_BAD` then text — drive `/api/chat`
  through `TestClient`, assert a `__cellc_step__` with an error summary appears
  and a second model turn occurred. (A real-model end-to-end is a manual check.)
- **No regressions:** existing non-cellc chat flow (plain text, file-op
  tool_calls, thinking, stats, tools→no-tools fallback) keeps its behavior;
  add/keep a test that a response with only op tool_calls is unchanged.

## Git safety

`feat/cellc-phase2` off `main`. `main` still carries the user's pre-existing
uncommitted WIP (`frontend/index.html`, `frontend/style.css`,
`data/capabilities.json`, `.gitignore`) — NEVER `git add -A`; stage only Phase-2
files per task. `backend/app.py` and `frontend/app.js` no longer carry that WIP
(it was committed in Phase 1), so Phase-2 commits to them are clean.

## Phase 3 carry-forward (not built here)

Optional later: expose `metadata`/`list_examples` as loop tools; inject the
language reference into the system prompt for cellc-tagged chats; let the model
write the checked contract to the workspace via an explicit (confirmed) file op.
