# cellc ↔ wyltek-gpt Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three additive enhancements to the Phase 2 cellc loop: (A) two more read-only loop tools (`cellc_metadata`, `cellc_list_examples`); (B) inject the CellScript language reference into the system prompt on CellScript chats; (C) a confirmed `cellc_save` op that re-checks and writes a `.cell` to the workspace.

**Architecture:** A/B extend the Phase 2 bridge + loop + prompt assembly. C rides the EXISTING operations/confirm-card infrastructure: a synthetic `cellc_save` Operation (registered when cellc is available) surfaces via `OPERATIONS.tool_schemas()`, routes to the confirm card (NOT the auto-loop, since it writes), and executes in `/api/operations/run`. Core line: read = auto-loop, write = confirm card.

**Tech Stack:** Python/FastAPI, the Phase-2 cellc bridge, the existing `OperationRegistry`/confirm-card flow, pytest + TestClient.

## Global Constraints

- Repo `~/local-chatbot` on branch `feat/cellc-phase3`. `main` carries user WIP (`.gitignore`, `data/capabilities.json`, `frontend/index.html`, `frontend/style.css`). NEVER `git add -A`; stage ONLY each task's files. `backend/app.py`, `backend/bridges/cellc.py`, `backend/operations.py`, `frontend/app.js` are clean of that WIP.
- Everything gates on `cellc_bridge.available()`: unavailable → new tools not advertised, prompt block omitted, `cellc_save` not registered. App boots normally.
- `cellc_save` is NOT in `CELLC_TOOL_NAMES` — it must route to `op_calls` (confirm card), never the auto-loop. It is the ONLY write in the cellc integration; it is gated by the confirm card by construction.
- `dispatch()` never raises (preserve for new arms). Validation: source ≤ `MAX_SOURCE` (200_000), `target_profile ∈ {ckb}`, name `^[A-Za-z0-9_-]+$`.
- Reuse existing infrastructure: `OperationRegistry.add()`, `TextValidator(max_len, desc)`, `Operation.to_tool_schema()`, `_resolve_in_workspace`/`_assert_inside`, the `/api/operations/run` kind-dispatch. Do not invent a new confirm mechanism.
- Verified Phase-2 facts: Ollama tool-result key is `tool_name`; cellc tool schema shape is `{"type":"function","function":{name,description,parameters}}`.

---

## File Structure

- `backend/bridges/cellc.py` — A: grow `CELLC_TOOL_NAMES` to 6, add 2 schemas + a `list_examples` dispatch arm (clean the metadata arm).
- `backend/app.py` — A: 2 step summaries. B: one-line hint in `_full_system_prompt`, `_is_cellc_intent` + reference prepend in `/api/chat`. C: register synthetic `cellc_save` op when available; `/api/operations/run` `cellc_save` arm; `_run_cellc_save_op`.
- `backend/operations.py` — C: a helper to build the synthetic `cellc_save` Operation (mirror `make_converter_operation`), and allow `kind="cellc_save"` through the registry build (so `_build`/`add` accept it).
- `tests/test_cellc_dispatch.py` — A: extend.
- `tests/test_cellc_prompt.py` — B: new.
- `tests/test_cellc_save.py` — C: new.
- `tests/test_cellc_save_integration.py` — C: gated real-binary.

---

### Task 1: Component A — two more loop tools

**Files:**
- Modify: `backend/bridges/cellc.py`
- Modify: `backend/app.py` (only `_summarize_cellc_step`)
- Modify: `tests/test_cellc_dispatch.py`

**Interfaces:**
- Produces: `CELLC_TOOL_NAMES` (6 names); `tool_schemas()` returns 6; `dispatch` handles `cellc_metadata` + `cellc_list_examples`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cellc_dispatch.py`:
```python
def test_tool_schemas_has_six_tools():
    names = {s["function"]["name"] for s in cellc.tool_schemas()}
    assert names == {
        "cellc_check", "cellc_explain", "cellc_get_example",
        "cellc_language_reference", "cellc_metadata", "cellc_list_examples",
    }


def test_cellc_tool_names_has_six():
    assert len(cellc.CELLC_TOOL_NAMES) == 6
    assert "cellc_metadata" in cellc.CELLC_TOOL_NAMES
    assert "cellc_list_examples" in cellc.CELLC_TOOL_NAMES


def test_dispatch_metadata_forwards():
    from unittest import mock
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_metadata.return_value = {"resources_count": 2}
        out = cellc.dispatch("cellc_metadata", {"source": "module x"})
    srv.cellc_metadata.assert_called_once_with(source="module x", target_profile="ckb", full=False)
    assert out["resources_count"] == 2


def test_dispatch_list_examples():
    from unittest import mock
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_list_examples.return_value = [{"name": "token"}]
        out = cellc.dispatch("cellc_list_examples", {})
    assert out == {"examples": [{"name": "token"}]}
```

- [ ] **Step 2: Run to verify fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_dispatch.py -k "six or metadata_forwards or list_examples" -v`
Expected: FAIL (schemas return 4; `CELLC_TOOL_NAMES` has 4; no list_examples arm).

- [ ] **Step 3: Implement**

In `backend/bridges/cellc.py`:
- Change `CELLC_TOOL_NAMES` to include all six:
```python
CELLC_TOOL_NAMES = frozenset({
    "cellc_check", "cellc_explain", "cellc_get_example",
    "cellc_language_reference", "cellc_metadata", "cellc_list_examples",
})
```
- In `tool_schemas()`, add two entries to the returned list (same `fn(...)` helper):
```python
        fn("cellc_metadata",
           "Compiler metadata for a contract: resources, actions, effects, obligations (summary).",
           {"source": {"type": "string", "description": "full .cell source"},
            "target_profile": {"type": "string", "enum": ["ckb"], "description": "target profile (default ckb)"}},
           ["source"]),
        fn("cellc_list_examples",
           "List bundled example .cell contracts (names + one-line summaries).",
           {}, []),
```
- In `dispatch()`, the existing `name == "cellc_check" or name == "cellc_metadata"` arm already routes metadata (keep it). Add before the final fallthrough:
```python
    if name == "cellc_list_examples":
        return {"examples": list_examples()}
```

In `backend/app.py` `_summarize_cellc_step`, add branches before the generic fallback:
```python
    if name == "cellc_metadata":
        res = result.get("resources_count", 0)
        act = result.get("actions_count", 0)
        return f"cellc_metadata → {res} resources, {act} actions"
    if name == "cellc_list_examples":
        n = len(result.get("examples", []))
        return f"cellc_list_examples → {n} examples"
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_dispatch.py -v && .venv/bin/python -m pytest -q`
Expected: dispatch tests pass; full suite green.

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add backend/bridges/cellc.py backend/app.py tests/test_cellc_dispatch.py
git commit -m "feat: cellc_metadata + cellc_list_examples as loop tools"
```

---

### Task 2: Component B — language reference in the system prompt

**Files:**
- Modify: `backend/app.py`
- Create: `tests/test_cellc_prompt.py`

**Interfaces:**
- Produces: `_is_cellc_intent(text: str) -> bool`; a one-line cellc hint in `_full_system_prompt()` when available; the full reference prepended to the system message in `/api/chat` on cellc-intent.

- [ ] **Step 1: Write the failing tests**

`tests/test_cellc_prompt.py`:
```python
from unittest import mock

from backend import app as app_module


def test_is_cellc_intent_true():
    assert app_module._is_cellc_intent("write a CellScript token contract")
    assert app_module._is_cellc_intent("check this .cell file")
    assert app_module._is_cellc_intent("make a CKB contract")


def test_is_cellc_intent_false():
    assert not app_module._is_cellc_intent("what's the weather today")
    assert not app_module._is_cellc_intent("write a python function")


def test_full_system_prompt_has_hint_when_available(monkeypatch):
    monkeypatch.setattr(app_module.cellc_bridge, "available", lambda: True)
    assert "cellc_check" in app_module._full_system_prompt()


def test_full_system_prompt_no_hint_when_unavailable(monkeypatch):
    monkeypatch.setattr(app_module.cellc_bridge, "available", lambda: False)
    # base prompt should not advertise cellc tooling
    assert "cellc_check" not in app_module._full_system_prompt()
```

- [ ] **Step 2: Run to verify fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_prompt.py -v`
Expected: FAIL (`_is_cellc_intent` undefined; hint absent).

- [ ] **Step 3: Implement**

In `backend/app.py`:
- Add the intent matcher near the cellc helpers:
```python
_CELLC_INTENT_RE = re.compile(
    r"cellscript|\.cell\b|cell contract|ckb contract|nervos contract|\bcellc\b|resource\s+\w+\s+has",
    re.IGNORECASE,
)


def _is_cellc_intent(text: str) -> bool:
    return bool(_CELLC_INTENT_RE.search(text or ""))


_CELLC_PROMPT_HINT = (
    "\n\nCellScript (.cell) tooling is available: call cellc_check to verify "
    "Cell contracts, cellc_language_reference for syntax, and cellc_save (with "
    "confirmation) to store a checked contract."
)
```
- In `_full_system_prompt()`, append the hint when available:
```python
def _full_system_prompt() -> str:
    base = (
        CONFIG["assistant"]["system_prompt"]
        + host_context_block()
        + _operations_prompt_block()
    )
    if cellc_bridge.available():
        base = base + _CELLC_PROMPT_HINT
    return base
```
- In `/api/chat`, after building `messages` (the `[{"role":"system",...}, *user_and_assistant]` list), prepend the full reference to the system content when the latest user message is cellc-intent:
```python
    if cellc_bridge.available():
        last_user = next((m.get("content", "") for m in reversed(incoming) if m.get("role") == "user"), "")
        if _is_cellc_intent(last_user):
            ref = cellc_bridge.language_reference()
            messages[0]["content"] = messages[0]["content"] + "\n\n# CellScript language reference\n" + ref
```
(Place this right after `messages = [...]` is constructed and before the body is built. `cellc_bridge.language_reference()` returns the reference string.)

- [ ] **Step 4: Run to verify pass + full suite**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_prompt.py -v && .venv/bin/python -m pytest -q`
Expected: pass; full suite green.

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add backend/app.py tests/test_cellc_prompt.py
git commit -m "feat: inject CellScript language reference on cellc-intent chats"
```

---

### Task 3: Component C (part 1) — synthetic `cellc_save` op + registry support

**Files:**
- Modify: `backend/operations.py`
- Modify: `backend/app.py` (registration only)
- Create: `tests/test_cellc_save.py` (registration + routing tests in this task; handler tests in Task 4)

**Interfaces:**
- Produces: `make_cellc_save_operation() -> Operation` in `operations.py`; the registry accepts `kind="cellc_save"`; `cellc_save` registered in app startup when available.

- [ ] **Step 1: Write the failing tests**

`tests/test_cellc_save.py`:
```python
from unittest import mock

from backend import app as app_module
from backend.operations import make_cellc_save_operation


def test_make_cellc_save_operation_schema():
    op = make_cellc_save_operation()
    assert op.id == "cellc_save"
    assert op.kind == "cellc_save"
    schema = op.to_tool_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "name" in props and "source" in props
    assert set(schema["function"]["parameters"]["required"]) == {"name", "source"}


def test_cellc_save_not_in_auto_loop_names():
    # cellc_save must route to the confirm card, never the auto-loop
    assert "cellc_save" not in app_module.cellc_bridge.CELLC_TOOL_NAMES


def test_cellc_save_partitions_as_op_call():
    calls = [{"function": {"name": "cellc_save", "arguments": {"name": "t", "source": "x"}}}]
    cellc_calls, op_calls = app_module._partition_cellc_calls(calls)
    assert cellc_calls == []
    assert len(op_calls) == 1 and op_calls[0]["function"]["name"] == "cellc_save"
```

- [ ] **Step 2: Run to verify fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_save.py -v`
Expected: FAIL (`make_cellc_save_operation` undefined).

- [ ] **Step 3: Implement**

In `backend/operations.py`:
- Read how `make_converter_operation` builds a synthetic `Operation` (its `Param`/validator construction) and mirror it. Add:
```python
def make_cellc_save_operation() -> Operation:
    """Synthetic op: write a model-supplied .cell source to the workspace.
    kind 'cellc_save' is handled specially in /api/operations/run (re-check + write)."""
    return Operation(
        id="cellc_save",
        kind="cellc_save",
        description=(
            "Save a checked CellScript contract to the workspace as <name>.cell. "
            "Only call after cellc_check passes; the backend re-checks and refuses "
            "to save a failing contract."
        ),
        capabilities_required=("text",),
        params=(
            Param(
                name="name",
                validator=TextValidator(max_len=64, description="File stem (letters, digits, _ or -)."),
                required=True,
            ),
            Param(
                name="source",
                validator=TextValidator(max_len=200_000, description="Full .cell source."),
                required=True,
            ),
        ),
        output_ext="cell",
    )
```
(Field names verified against `make_converter_operation`: `Operation(id, kind, description, capabilities_required, params=tuple, source_param=None, output_ext)`; `Param(name=, validator=, required=)`; `TextValidator(max_len=, description=)`. `cellc_save` declares NO `source_param` — its `source` is text content, not a workspace filename. `Param`/`TextValidator`/`Operation` are already defined/exported in this module.)
- Ensure the registry's `_build`/validation accepts `kind="cellc_save"` for a programmatically-added op. Since the op is added via `OPERATIONS.add()` (not parsed from config), confirm `add()` does not reject unknown kinds (it should just append). If `_build` is only used for config entries, no change is needed — verify.

In `backend/app.py`, after `OPERATIONS` is constructed and the converter op is added, register cellc_save when available:
```python
from backend.operations import make_cellc_save_operation  # with the other imports
...
if cellc_bridge.available():
    OPERATIONS.add(make_cellc_save_operation())
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_save.py -v && .venv/bin/python -m pytest -q`
Expected: pass; full suite green.

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add backend/operations.py backend/app.py tests/test_cellc_save.py
git commit -m "feat: synthetic cellc_save op (registered when cellc available)"
```

---

### Task 4: Component C (part 2) — the `cellc_save` execution handler

**Files:**
- Modify: `backend/app.py`
- Modify: `tests/test_cellc_save.py` (add handler tests)
- Create: `tests/test_cellc_save_integration.py`

**Interfaces:**
- Consumes: `cellc_bridge.check`, `_resolve_in_workspace`/`_assert_inside`.
- Produces: `_run_cellc_save_op(op, session_dir, validated) -> Path`; `/api/operations/run` `cellc_save` arm.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cellc_save.py`:
```python
import re as _re
from pathlib import Path
from unittest import mock

import pytest

from backend import app as app_module


def test_run_cellc_save_writes_on_passing_check(tmp_path):
    with mock.patch.object(app_module.cellc_bridge, "check", return_value={"ok": True, "diagnostics": []}):
        out = app_module._run_cellc_save_op_sync(
            tmp_path, {"name": "token", "source": "module probe::token\n"}
        )
    assert out.name == "token.cell"
    assert out.read_text() == "module probe::token\n"


def test_run_cellc_save_refuses_failing_check(tmp_path):
    with mock.patch.object(app_module.cellc_bridge, "check",
                           return_value={"ok": False, "error_count": 1, "diagnostics": [{"line": 3, "message": "boom"}]}):
        with pytest.raises(app_module.HTTPException) as ei:
            app_module._run_cellc_save_op_sync(tmp_path, {"name": "bad", "source": "module x"})
    assert ei.value.status_code == 400
    assert not (tmp_path / "bad.cell").exists()


def test_run_cellc_save_rejects_bad_name(tmp_path):
    with mock.patch.object(app_module.cellc_bridge, "check", return_value={"ok": True, "diagnostics": []}):
        with pytest.raises(app_module.HTTPException):
            app_module._run_cellc_save_op_sync(tmp_path, {"name": "../evil", "source": "module x"})
```
(Note: the test targets a SYNC helper `_run_cellc_save_op_sync(session_dir, validated)` that does validation + check + write and returns a Path; the async `_run_cellc_save_op` wraps it via `asyncio.to_thread`. This keeps the write/validation logic unit-testable without an event loop. `HTTPException` is re-exported from `app_module` — it is imported there from fastapi.)

`tests/test_cellc_save_integration.py`:
```python
import os
import shutil
from pathlib import Path

import pytest

from backend import app as app_module

pytestmark = pytest.mark.needs_cellc
_HAS_CELLC = bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))

_GOOD = """\
module probe::token
resource Token has store, create, consume, replace { amount: u64, symbol: [u8; 8] }
"""
_BAD = "module probe::broken\n\nresource Token has store { amount: u64\n"


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_save_writes_good_refuses_bad(tmp_path):
    out = app_module._run_cellc_save_op_sync(tmp_path, {"name": "good", "source": _GOOD})
    assert out.read_text() == _GOOD
    with pytest.raises(app_module.HTTPException):
        app_module._run_cellc_save_op_sync(tmp_path, {"name": "bad", "source": _BAD})
    assert not (tmp_path / "bad.cell").exists()
```

- [ ] **Step 2: Run to verify fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_save.py -k run_cellc_save -v`
Expected: FAIL (`_run_cellc_save_op_sync` undefined).

- [ ] **Step 3: Implement**

In `backend/app.py`:
```python
_CELLC_SAVE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _run_cellc_save_op_sync(session_dir: Path, validated: dict) -> Path:
    name = validated["name"]
    if not _CELLC_SAVE_NAME_RE.match(name):
        raise HTTPException(400, "invalid save name (letters, digits, _ or - only)")
    source = validated["source"]
    result = cellc_bridge.check(source)
    if not result.get("ok"):
        diags = result.get("diagnostics") or []
        first = diags[0].get("message") if diags else "check failed"
        raise HTTPException(400, f"refusing to save: cellc_check failed ({result.get('error_count', '?')} error(s)): {first}")
    target = (session_dir / f"{name}.cell").resolve()
    target.relative_to(session_dir.resolve())  # path-traversal guard
    target.write_text(source, encoding="utf-8")
    return target


async def _run_cellc_save_op(op, session_dir: Path, validated: dict) -> Path:
    return await asyncio.to_thread(_run_cellc_save_op_sync, session_dir, validated)
```
In `/api/operations/run`, add the kind arm alongside local/bridge/converter:
```python
    elif op.kind == "cellc_save":
        out = await _run_cellc_save_op(op, session_dir, validated)
```

- [ ] **Step 4: Run to verify pass + full suite + gated integration**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_save.py -v && .venv/bin/python -m pytest -q`
Expected: pass; full suite green; integration SKIPS without the binary.
With the binary:
```bash
export CELLC_BIN=/home/phill/CellScript/target/release/cellc CELLSCRIPT_REPO=/home/phill/CellScript
.venv/bin/python -m pytest tests/test_cellc_save_integration.py -v
```
Expected: 1 PASSED (real binary: good writes, bad refused).

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add backend/app.py tests/test_cellc_save.py tests/test_cellc_save_integration.py
git commit -m "feat: cellc_save handler — re-check then write .cell to workspace"
```

---

### Task 5: Manual end-to-end smoke (qwen3.6:27b) + confirm-card verification

**Files:** none (verification only; record results in the commit/report).

- [ ] **Step 1: Backend up with the binary**

```bash
cd ~/local-chatbot
export CELLC_BIN=/home/phill/CellScript/target/release/cellc CELLSCRIPT_REPO=/home/phill/CellScript
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8015 >/tmp/p3smoke.log 2>&1 &
until curl -sf localhost:8015/api/cellc/status >/dev/null; do sleep 1; done
```
(If qwen3.6:27b OOMs on ROCm, ensure ComfyUI/other GPU holders are stopped — the 16GB Q4 needs the 24GB card mostly free.)

- [ ] **Step 2: Programmatic loop+save smoke**

```bash
curl -s --max-time 540 -X POST localhost:8015/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6:27b","messages":[{"role":"user","content":"Write a minimal CellScript fungible token, verify it with cellc_check, then save it as token.cell."}]}' \
  | tee /tmp/p3chat.out | tail -c 1200
grep -c "__cellc_step__" /tmp/p3chat.out   # check steps fired
grep -c "__tool_calls__" /tmp/p3chat.out   # cellc_save surfaced as an op (confirm card) — expect >=1
```
Acceptance: at least one `__cellc_step__` (the in-loop check) AND a `__tool_calls__` containing `cellc_save` (the save proposal routed to the confirm card, NOT auto-run). The language-reference injection should also be visible in the model writing valid syntax first try.

- [ ] **Step 3: Browser confirm + write (manual)**

In the UI, run the same prompt; when the `cellc_save` confirm card appears, click Y; verify `token.cell` lands in the workspace (downloadable). Record: (1) loop check steps shown, (2) save surfaced as a confirm card (not auto-run), (3) on Y the file is written and on N nothing is written.

- [ ] **Step 4: Stop the backend**

```bash
fuser -k 8015/tcp 2>/dev/null || true
```

---

## Self-Review

**Spec coverage:**
- A (metadata + list_examples loop tools) → Task 1. ✓
- B (one-line hint + cellc-intent reference injection) → Task 2. ✓
- C tool exposure + routing (synthetic op, not in CELLC_TOOL_NAMES, routes to op_calls) → Task 3. ✓
- C execution (re-check + workspace write, refuse on failing check, name guard) → Task 4. ✓
- read=auto / write=confirm split → Task 3 (routing test) + Task 4 (handler) + Task 5 (confirm-card smoke). ✓
- available() gating across A/B/C → Task 1 (schemas), Task 2 (hint/injection), Task 3 (registration). ✓
- Testing (dispatch, prompt, save unit + gated integration, no-regression, manual smoke) → Tasks 1–5. ✓
- Git safety → Global Constraints + every commit step names exact files. ✓

**Placeholder scan:** No TBD/TODO. Two "read the real field names first" notes (Task 3 `Operation`/`Param` construction; Task 3 registry `add()` acceptance) are explicit verification instructions against existing code, not placeholders.

**Type consistency:** `cellc_bridge` is the app.py alias. `CELLC_TOOL_NAMES` (6) used in Task 1 + Task 3 routing test. `make_cellc_save_operation()` defined in Task 3, consumed in Task 3 registration + tests. `_run_cellc_save_op_sync(session_dir, validated)` defined in Task 4, targeted by Task 4 + integration tests; async `_run_cellc_save_op` wraps it. `_is_cellc_intent`/`_CELLC_PROMPT_HINT` defined + used in Task 2. `HTTPException` re-exported from app_module (imported from fastapi) — tests reference `app_module.HTTPException`.

**Known risks flagged in-plan:** (1) exact `Operation`/`Param` construction must be read from operations.py (Task 3) — the converter op is the template; (2) registry `add()` must accept a programmatic `kind="cellc_save"` (Task 3 verify); (3) Task 5 is manual (no harness) with explicit acceptance; (4) 27B VRAM/ComfyUI note carried into Task 5.
