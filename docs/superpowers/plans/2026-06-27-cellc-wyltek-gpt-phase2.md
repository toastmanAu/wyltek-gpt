# cellc ↔ wyltek-gpt Phase 2 — Agentic Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a capable local model self-iterate write→check→fix on CellScript inside `/api/chat`: model emits a `cellc_check` tool call → backend auto-runs it → feeds the result back → model continues, looping (cap 5) with each step shown inline.

**Architecture:** Add cellc tool schemas + a `dispatch()` to the Phase-1 bridge. In `/api/chat`'s `relay()` generator, intercept cellc tool calls, execute them via the bridge, append `role:tool` results to the Ollama message history, and re-invoke — gated so file-op tool calls and all existing behavior are untouched. A `__cellc_step__` stream sentinel renders each step inline in the frontend.

**Tech Stack:** Python/FastAPI async generators, Ollama 0.30.5 tool calling, pytest + TestClient, the Phase-1 `cellc_mcp` bridge.

## Global Constraints

- Repo `~/local-chatbot` on branch `feat/cellc-phase2`. `main` carries pre-existing user WIP (`.gitignore`, `data/capabilities.json`, `frontend/index.html`, `frontend/style.css`). NEVER `git add -A`; stage ONLY each task's files. `backend/app.py` and `frontend/app.js` are clean of that WIP now (committed in Phase 1).
- The loop fires ONLY on cellc tool calls; file-op (`OPERATIONS`) tool calls keep their existing `__tool_calls__` → confirm-card path unchanged. No new chat endpoint/mode.
- cellc tool calls auto-run (read-only; no confirm card). Only `source` text is exposed, never cellc_mcp's `path=`.
- `MAX_CELLC_ITERS = 5`. Loop tools (exactly 4): `cellc_check`, `cellc_explain`, `cellc_get_example`, `cellc_language_reference`.
- cellc schemas are added to the Ollama `tools` array ONLY when `cellc_bridge.available()`.
- `dispatch()` never raises — bad args / unknown tool → a `tool_error`-shaped dict fed back to the model.
- Validation rules (shared with Phase 1, do not re-define divergently): `_CELLC_MAX_SOURCE = 200_000`; `target_profile ∈ {"ckb"}`; code `^[A-Za-z0-9_]+$`; name `^[A-Za-z0-9_-]+$`.
- Existing `tool_schemas()` shape (match it): `{"type": "function", "function": {"name", "description", "parameters": <JSON Schema>}}`.
- Stream sentinel convention: a line `"\n" + json.dumps({"__cellc_step__": {...}})`, same family as `__tool_calls__`/`__stats__`.

---

## File Structure

- `backend/bridges/cellc.py` — ADD `tool_schemas()`, `dispatch()`, and shared validation constants/helpers.
- `backend/app.py` — restructure `relay()` to thread `messages`+`iterations` and run the loop; merge cellc schemas into `body_with_tools`; add `MAX_CELLC_ITERS`, a partition helper, and a `summarize_cellc_step()` helper.
- `frontend/app.js` — extract + render interleaved `__cellc_step__` sentinels.
- `tests/test_cellc_dispatch.py` — `dispatch()` + `tool_schemas()` unit tests.
- `tests/test_chat_cellc_loop.py` — loop unit tests (mock `_stream_one` + bridge) + a gated integration test.

---

### Task 1: Verify Ollama tool-result format + add bridge `tool_schemas`/`dispatch`

**Files:**
- Modify: `backend/bridges/cellc.py`
- Create: `tests/test_cellc_dispatch.py`

**Interfaces:**
- Consumes: the Phase-1 wrappers in the same module.
- Produces:
  - `CELLC_TOOL_NAMES: frozenset[str]`
  - `tool_schemas() -> list[dict]`
  - `dispatch(name: str, arguments: dict) -> dict`
  - shared `MAX_SOURCE`, `PROFILES`, `CODE_RE`, `NAME_RE` (lifted so both endpoints and dispatch can share — Task 2 will point app.py at them; for now define them in the bridge).

- [ ] **Step 1: Verify the Ollama tool-message contract (do this FIRST, record the result)**

Run a tiny round-trip against the installed Ollama (0.30.5) to confirm the exact tool-result message shape before building on it. Use any installed tool-calling model (`ollama list`); if none, note it and proceed with the documented shape.
```bash
cd ~/local-chatbot
.venv/bin/python - <<'PY'
import httpx, json
OLLAMA="http://localhost:11434"
model=httpx.get(f"{OLLAMA}/api/tags").json()["models"][0]["name"]
tools=[{"type":"function","function":{"name":"ping","description":"ping","parameters":{"type":"object","properties":{},"required":[]}}}]
# 1) ask in a way that elicits a tool call
r=httpx.post(f"{OLLAMA}/api/chat",json={"model":model,"messages":[{"role":"user","content":"call the ping tool"}],"tools":tools,"stream":False},timeout=120)
msg=r.json()["message"]; print("ASSISTANT MSG:",json.dumps(msg)[:400])
# 2) feed a tool result back and confirm Ollama accepts the shape
msgs=[{"role":"user","content":"call the ping tool"},msg,{"role":"tool","tool_name":"ping","content":"pong"}]
r2=httpx.post(f"{OLLAMA}/api/chat",json={"model":model,"messages":msgs,"stream":False},timeout=120)
print("AFTER TOOL RESULT ok:",r2.status_code, json.dumps(r2.json().get("message",{}))[:200])
PY
```
Record in the report: the assistant tool_call shape and whether `{"role":"tool","tool_name":...,"content":...}` is accepted (HTTP 200). If Ollama wants `name` instead of `tool_name`, or a string-only content, use the accepted shape in Step 3 and note the deviation. If no model is installed, proceed with `{"role":"tool","tool_name":name,"content":json_str}` and flag it UNVERIFIED.

- [ ] **Step 2: Write the failing tests**

`tests/test_cellc_dispatch.py`:
```python
from unittest import mock

from backend.bridges import cellc


def test_tool_schemas_shape():
    schemas = cellc.tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"cellc_check", "cellc_explain", "cellc_get_example", "cellc_language_reference"}
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]
    # cellc_check requires source
    check = next(s for s in schemas if s["function"]["name"] == "cellc_check")
    assert "source" in check["function"]["parameters"]["required"]


def test_dispatch_check_validates_and_forwards():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_check.return_value = {"ok": True, "diagnostics": []}
        out = cellc.dispatch("cellc_check", {"source": "module x", "target_profile": "ckb"})
    srv.cellc_check.assert_called_once_with(source="module x", target_profile="ckb", full=False)
    assert out["ok"] is True


def test_dispatch_empty_source_is_tool_error_not_raise():
    out = cellc.dispatch("cellc_check", {"source": ""})
    assert out["tool_error"] is True
    assert "source" in out["stderr"].lower()


def test_dispatch_bad_profile_tool_error():
    out = cellc.dispatch("cellc_check", {"source": "module x", "target_profile": "evm"})
    assert out["tool_error"] is True


def test_dispatch_bad_code_tool_error():
    out = cellc.dispatch("cellc_explain", {"code": "E14; rm -rf"})
    assert out["tool_error"] is True


def test_dispatch_unknown_tool_error():
    out = cellc.dispatch("cellc_nope", {})
    assert out["tool_error"] is True


def test_dispatch_language_reference_no_args():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_language_reference.return_value = "REF"
        out = cellc.dispatch("cellc_language_reference", {})
    assert out == {"reference": "REF"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_dispatch.py -v`
Expected: FAIL (`AttributeError: module 'backend.bridges.cellc' has no attribute 'tool_schemas'`).

- [ ] **Step 4: Write minimal implementation**

Add to `backend/bridges/cellc.py` (after the existing wrappers):
```python
import re

MAX_SOURCE = 200_000
PROFILES = {"ckb"}
CODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

CELLC_TOOL_NAMES = frozenset(
    {"cellc_check", "cellc_explain", "cellc_get_example", "cellc_language_reference"}
)


def tool_schemas() -> list[dict]:
    def fn(name, desc, props, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }
    return [
        fn("cellc_check",
           "Type-check a CellScript .cell contract. Returns ok + diagnostics (line/column/code/message).",
           {"source": {"type": "string", "description": "full .cell source"},
            "target_profile": {"type": "string", "enum": ["ckb"], "description": "target profile (default ckb)"}},
           ["source"]),
        fn("cellc_explain",
           "Explain a CellScript error code (e.g. E0014) with description and fix hint.",
           {"code": {"type": "string", "description": "error code or name, e.g. E0014"}},
           ["code"]),
        fn("cellc_get_example",
           "Return a bundled example .cell contract's source by name (e.g. token, nft).",
           {"name": {"type": "string", "description": "example name"}},
           ["name"]),
        fn("cellc_language_reference",
           "Return the full CellScript language surface (keywords, effects, a worked example).",
           {}, []),
    ]


def _tool_err(msg: str) -> dict:
    return {"ok": False, "tool_error": True, "exit_code": -1, "stderr": msg}


def dispatch(name: str, arguments: dict) -> dict:
    arguments = arguments or {}
    if name == "cellc_check" or name == "cellc_metadata":
        source = arguments.get("source")
        if not isinstance(source, str) or not source.strip():
            return _tool_err("missing or empty 'source'")
        if len(source) > MAX_SOURCE:
            return _tool_err(f"source too long ({len(source)} > {MAX_SOURCE})")
        profile = arguments.get("target_profile", "ckb")
        if profile not in PROFILES:
            return _tool_err(f"invalid target_profile {profile!r}")
        full = bool(arguments.get("full"))
        if name == "cellc_check":
            return check(source, target_profile=profile, full=full)
        return metadata(source, target_profile=profile, full=full)
    if name == "cellc_explain":
        code = arguments.get("code")
        if not isinstance(code, str) or not CODE_RE.match(code):
            return _tool_err("invalid 'code' (expected an error code/name like E0014)")
        return explain(code)
    if name == "cellc_get_example":
        ex = arguments.get("name")
        if not isinstance(ex, str) or not NAME_RE.match(ex):
            return _tool_err("invalid example 'name'")
        return get_example(ex)
    if name == "cellc_language_reference":
        return {"reference": language_reference()}
    return _tool_err(f"unknown cellc tool {name!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_dispatch.py -v`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add backend/bridges/cellc.py tests/test_cellc_dispatch.py
git commit -m "feat: cellc tool schemas + dispatch for the agentic loop"
```

---

### Task 2: The loop in `/api/chat` `relay()`

**Files:**
- Modify: `backend/app.py`
- Create: `tests/test_chat_cellc_loop.py`

**Interfaces:**
- Consumes: `cellc_bridge.tool_schemas()`, `cellc_bridge.dispatch()`, `cellc_bridge.CELLC_TOOL_NAMES`, `cellc_bridge.available()`, the existing `_stream_one`.
- Produces: a looping `relay`; `MAX_CELLC_ITERS`; `_partition_cellc_calls(tool_calls)`; `_summarize_cellc_step(name, result)`.

**Note for the implementer:** read the current `/api/chat` handler (`backend/app.py`, the `stream()`/`relay()`/`_stream_one` region, ~310–410) before editing. Preserve the thinking/chunk/stats/error/done handling and the tools→no-tools fallback exactly; only ADD the cellc-call branch and thread `messages`+`iterations`. Use the Ollama tool-result shape VERIFIED in Task 1 Step 1.

- [ ] **Step 1: Write the failing tests**

`tests/test_chat_cellc_loop.py`:
```python
import json
from unittest import mock

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _events(*evts):
    async def gen(body):
        for e in evts:
            yield e
    return gen


def _read(resp):
    return resp.text


def test_partition_separates_cellc_and_ops():
    calls = [
        {"function": {"name": "cellc_check", "arguments": {"source": "x"}}},
        {"function": {"name": "trim_video", "arguments": {}}},
    ]
    cellc_calls, op_calls = app_module._partition_cellc_calls(calls)
    assert len(cellc_calls) == 1 and cellc_calls[0]["function"]["name"] == "cellc_check"
    assert len(op_calls) == 1 and op_calls[0]["function"]["name"] == "trim_video"


def test_summarize_step_check_failed():
    s = app_module._summarize_cellc_step(
        "cellc_check",
        {"ok": False, "error_count": 2, "diagnostics": [{"line": 12}, {"line": 20}]},
    )
    assert "2" in s and "12" in s


def test_loop_executes_cellc_and_reinvokes(monkeypatch):
    # First model turn emits a cellc_check tool call; second turn emits text.
    turn1 = _events(
        ("tool_calls", [{"function": {"name": "cellc_check", "arguments": {"source": "module x"}}}]),
        ("done", None),
    )
    turn2 = _events(("chunk", "fixed it"), ("done", None))
    bodies = []
    calls = iter([turn1, turn2])

    def fake_stream_one(body):
        bodies.append(body)
        return next(calls)(body)

    monkeypatch.setattr(app_module, "_stream_one_factoryless", None, raising=False)
    monkeypatch.setattr(app_module, "cellc_bridge", mock.Mock(
        available=mock.Mock(return_value=True),
        CELLC_TOOL_NAMES=frozenset({"cellc_check"}),
        tool_schemas=mock.Mock(return_value=[]),
        dispatch=mock.Mock(return_value={"ok": False, "error_count": 1, "diagnostics": [{"line": 3}]}),
    ))
    # Patch _stream_one used inside the handler:
    monkeypatch.setattr(app_module, "_stream_one", fake_stream_one, raising=False)

    resp = client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "write a token"}]})
    body = _read(resp)
    assert "__cellc_step__" in body          # a transparent step was emitted
    assert "fixed it" in body                # the model's second turn streamed
    # the second _stream_one body carried a tool-result message
    assert len(bodies) == 2
    assert any(m.get("role") == "tool" for m in bodies[1]["messages"])


def test_loop_stops_at_cap(monkeypatch):
    # Every turn re-emits a cellc call; ensure we don't exceed MAX_CELLC_ITERS re-invokes.
    def always_cellc(body):
        async def gen(_b):
            yield ("tool_calls", [{"function": {"name": "cellc_check", "arguments": {"source": "x"}}}])
            yield ("done", None)
        return gen(body)
    count = {"n": 0}
    def counting(body):
        count["n"] += 1
        return always_cellc(body)
    monkeypatch.setattr(app_module, "cellc_bridge", mock.Mock(
        available=mock.Mock(return_value=True),
        CELLC_TOOL_NAMES=frozenset({"cellc_check"}),
        tool_schemas=mock.Mock(return_value=[]),
        dispatch=mock.Mock(return_value={"ok": False, "error_count": 1, "diagnostics": []}),
    ))
    monkeypatch.setattr(app_module, "_stream_one", counting, raising=False)
    client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    # initial call + at most MAX_CELLC_ITERS re-invokes
    assert count["n"] <= app_module.MAX_CELLC_ITERS + 1
```
(If the existing `_stream_one` is a closure inside `chat()` rather than a module function, the implementer must lift it to a module-level function — e.g. `_stream_one(body)` — so it is patchable; do this as part of the restructure and keep behavior identical.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_chat_cellc_loop.py -v`
Expected: FAIL (`AttributeError` on `_partition_cellc_calls`/`_summarize_cellc_step`/`MAX_CELLC_ITERS`).

- [ ] **Step 3: Implement the loop**

In `backend/app.py`:
1. Add module constant `MAX_CELLC_ITERS = 5`.
2. Lift `_stream_one` to a module-level function if it is currently a closure (keep its body identical; it takes `body` and yields the same typed events). The `chat()` handler calls the module-level one so tests can patch it.
3. Add helpers:
```python
def _partition_cellc_calls(tool_calls):
    cellc_calls, op_calls = [], []
    names = cellc_bridge.CELLC_TOOL_NAMES if cellc_bridge.available() else frozenset()
    for tc in tool_calls or []:
        name = (tc.get("function") or {}).get("name")
        (cellc_calls if name in names else op_calls).append(tc)
    return cellc_calls, op_calls


def _summarize_cellc_step(name, result):
    if result.get("tool_error"):
        return f"⚠ {name}: {result.get('stderr', 'error')[:80]}"
    if name == "cellc_check":
        if result.get("ok"):
            return "cellc_check → ✓ passed"
        lines = ", ".join(f"L{d.get('line')}" for d in (result.get("diagnostics") or [])[:3])
        return f"cellc_check → ✗ {result.get('error_count', 0)} error(s) ({lines})"
    if name == "cellc_explain":
        return f"cellc_explain → {result.get('ecode', '')}"
    return f"{name} → ok"
```
4. Merge cellc schemas into the tools list where `body_with_tools` is built:
```python
tools = OPERATIONS.tool_schemas() if OPERATIONS.enabled else []
if cellc_bridge.available():
    tools = tools + cellc_bridge.tool_schemas()
body_with_tools = {"model": model, "messages": messages, "stream": True}
if tools:
    body_with_tools["tools"] = tools
```
5. Restructure `relay` to thread `messages` and `iterations`, and add the cellc branch in the `tool_calls` handler (replace the current single `yield __tool_calls__`):
```python
async def relay(source, messages, iterations):
    nonlocal attempted_fallback, in_thinking
    async for kind, value in source:
        if kind == "thinking":
            ...   # unchanged
        elif kind == "chunk":
            ...   # unchanged
        elif kind == "tool_calls":
            cellc_calls, op_calls = _partition_cellc_calls(value)
            if op_calls:
                yield "\n" + json.dumps({"__tool_calls__": op_calls})
            if cellc_calls and iterations < MAX_CELLC_ITERS:
                if in_thinking:
                    in_thinking = False
                    yield THINK_CLOSE
                tool_msgs = []
                for c in cellc_calls:
                    name = c["function"]["name"]
                    args = c["function"].get("arguments") or {}
                    result = await asyncio.to_thread(cellc_bridge.dispatch, name, args)
                    yield "\n" + json.dumps({"__cellc_step__": {"tool": name, "summary": _summarize_cellc_step(name, result)}})
                    tool_msgs.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
                new_messages = messages + [{"role": "assistant", "content": "", "tool_calls": cellc_calls}, *tool_msgs]
                async for wire in relay(_stream_one({**body_with_tools, "messages": new_messages}), new_messages, iterations + 1):
                    yield wire
                return
            if cellc_calls and iterations >= MAX_CELLC_ITERS:
                yield "\n" + json.dumps({"__cellc_step__": {"tool": "cellc", "summary": "(reached cellc check limit)"}})
        elif kind == "stats":
            ...   # unchanged
        elif kind == "error":
            ...   # unchanged (its fallback recursion must also pass messages, iterations)
        elif kind == "done":
            ...   # unchanged
    # NOTE: the error-fallback recursion and the initial call must pass (messages, iterations);
    # use the same `messages` built at handler top and iterations=0 for the first relay.
```
(Use the Ollama tool-result key verified in Task 1 Step 1 — `tool_name` unless verification said otherwise.)
6. Update the initial call and the tools→no-tools fallback recursion to pass `messages` and `iterations` (start `relay(_stream_one(body_with_tools), messages, 0)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_chat_cellc_loop.py -v`
Expected: PASS (4 passed). Then run the full offline suite to confirm no regressions: `.venv/bin/python -m pytest -q` — all prior tests still pass.

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add backend/app.py tests/test_chat_cellc_loop.py
git commit -m "feat: cellc agentic tool-result loop in /api/chat"
```

---

### Task 3: Frontend — render interleaved `__cellc_step__` steps

**Files:**
- Modify: `frontend/app.js`

**Interfaces:**
- Consumes: the `{"__cellc_step__": {"tool", "summary"}}` newline sentinels in the chat stream.
- Produces: inline trace rendering during streaming.

**Note:** read how the existing stream reader handles `__stats__`/`__tool_calls__`/THINK sentinels first (search `__stats__`, `lastIndexOf`, `parseStatsSentinel`). Unlike those, `__cellc_step__` appears MULTIPLE times interleaved — extract EVERY occurrence, not the last.

- [ ] **Step 1: Add a parser + render hook**

Add a helper near the existing `parseStatsSentinel`/`parseToolCallSentinel`:
```javascript
// Extract ALL `{"__cellc_step__": {...}}` sentinel lines (they interleave with
// text across loop iterations, unlike the single __stats__/__tool_calls__).
// Returns {steps: [{tool, summary}], cleanedText}.
function parseCellcSteps(text) {
  const steps = [];
  const lines = text.split("\n");
  const kept = [];
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith('{"__cellc_step__"')) {
      try { steps.push(JSON.parse(t).__cellc_step__); continue; } catch {}
    }
    kept.push(line);
  }
  return { steps, cleanedText: kept.join("\n") };
}

function renderCellcSteps(steps) {
  if (!steps.length) return "";
  return steps.map((s) => `\n🔧 ${s.summary}`).join("");
}
```
Wire it into the streaming render path where the assistant message text is built: run `parseCellcSteps` on the accumulated text BEFORE the existing stats/tool-call/think parsing, render `steps` as a small inline trace block prepended/append­ed to the message body, and use `cleanedText` for the normal markdown render. Keep it minimal; reuse the existing message-append/update path (do not invent a new render path).

- [ ] **Step 2: Syntax check**

Run: `cd ~/local-chatbot && node --check frontend/app.js`
Expected: `SYNTAX OK` (no output / exit 0).

- [ ] **Step 3: Manual smoke (acceptance — no JS harness)**

With the backend running and `CELLC_BIN` set, in a chat with a tool-calling model, ask: "write a CellScript fungible token and check it compiles." Acceptance:
1. The assistant message shows one or more `🔧 cellc_check → …` trace lines in order.
2. After a failing check, a subsequent trace line shows the model re-checked (or `✓ passed`).
3. Plain chats and file-op requests are unchanged (no `🔧` lines, confirm card still works for ops).
Record these three observations in the commit message.

- [ ] **Step 4: Commit (stage only this file)**

```bash
cd ~/local-chatbot
git add frontend/app.js
git commit -m "feat: render inline cellc check steps in chat stream"
```

---

### Task 4: Gated integration test + no-regression guard

**Files:**
- Create: `tests/test_chat_cellc_integration.py`

**Interfaces:**
- Consumes: the real `cellc` binary (via the bridge) + a scripted fake-Ollama generator.
- Produces: `needs_cellc` proof that a real `cellc_check` round-trip drives the loop end to end.

- [ ] **Step 1: Write the test**

`tests/test_chat_cellc_integration.py`:
```python
import os
import shutil
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module

pytestmark = pytest.mark.needs_cellc
_HAS_CELLC = bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))
client = TestClient(app_module.app)

_BAD = "module probe::broken\n\nresource Token has store { amount: u64\n"


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_real_cellc_check_drives_one_loop(monkeypatch):
    # Turn 1: model calls cellc_check on bad source. Turn 2: model emits text.
    seq = iter([
        [("tool_calls", [{"function": {"name": "cellc_check", "arguments": {"source": _BAD}}}]), ("done", None)],
        [("chunk", "I see the error"), ("done", None)],
    ])

    def fake_stream_one(body):
        evts = next(seq)
        async def gen(_b):
            for e in evts:
                yield e
        return gen(body)

    monkeypatch.setattr(app_module, "_stream_one", fake_stream_one, raising=False)
    resp = client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "check this"}]})
    body = resp.text
    assert "__cellc_step__" in body
    assert "error" in body.lower()        # the real cellc reported the syntax error
    assert "I see the error" in body      # second turn streamed after the real tool result
```

- [ ] **Step 2: Run it**

No binary: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_chat_cellc_integration.py -v` → SKIPPED.
With binary:
```bash
export CELLC_BIN=/home/phill/CellScript/target/release/cellc
export CELLSCRIPT_REPO=/home/phill/CellScript
.venv/bin/python -m pytest tests/test_chat_cellc_integration.py -v
```
Expected: 1 PASSED (the real cellc binary produced the error summary the loop fed back).

- [ ] **Step 3: Full suite + commit (stage only this file)**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest -q` (all pass; integration skips without env).
```bash
git add tests/test_chat_cellc_integration.py
git commit -m "test: gated end-to-end cellc loop integration"
```

---

## Self-Review

**Spec coverage:**
- Bridge `tool_schemas()` + `dispatch()` (4 tools, arg validation, never-raise) → Task 1. ✓
- Ollama tool-message format verification → Task 1 Step 1 (explicit, before building). ✓
- Loop in `relay` (partition, execute, append role:tool, re-invoke, cap 5, op passthrough) → Task 2. ✓
- cellc schemas merged only when available; tools→no-tools fallback preserved → Task 2 Step 3. ✓
- Transparent `__cellc_step__` sentinel + interleaved frontend rendering → Task 2 (emit) + Task 3 (render). ✓
- Error handling (tool_error fed back, available()=false no-op, cap note) → Tasks 1–2. ✓
- Testing: dispatch unit, loop unit (execute/cap/partition/passthrough), gated integration, no-regression → Tasks 1,2,4. ✓
- Git safety (branch, stage-only-own-files) → Global Constraints + every commit step. ✓

**Placeholder scan:** No TBD/TODO. The only deferred fact (exact Ollama tool-result key) is resolved in Task 1 Step 1 before any code depends on it — not a placeholder.

**Type consistency:** `cellc_bridge` is the app.py import alias (from Phase 1). `dispatch(name, arguments)`, `tool_schemas()`, `CELLC_TOOL_NAMES` used identically across bridge, app.py loop, and tests. `_partition_cellc_calls`/`_summarize_cellc_step`/`MAX_CELLC_ITERS` defined in Task 2 and referenced by its tests. Tool-result message uses `tool_name` (subject to Task 1 verification — if Ollama wants `name`, the implementer updates Task 2 Step 3 and the integration test asserts behavior, not the key).

**Known risks flagged in-plan:** (1) Ollama tool-result key verified in Task 1 before use; (2) `_stream_one` may need lifting to module scope for patchability — called out in Task 2 with "keep behavior identical"; (3) Task 3 frontend verified manually (no JS harness) with explicit acceptance.
