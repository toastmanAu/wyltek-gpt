# cellc ↔ wyltek-gpt Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dedicated read-only HTTP endpoints + a minimal frontend affordance to wyltek-gpt that compile/inspect CellScript via the already-built `cellc_mcp` tooling, so a user can write a `.cell` contract and get structured diagnostics back.

**Architecture:** A new in-process bridge module `backend/bridges/cellc.py` imports `cellc_mcp.server` and exposes thin JSON-able wrappers. Dedicated FastAPI endpoints under `/api/cellc/*` (modeled on the `translate` precedent — NOT the file-producing OPERATIONS registry) validate input and call the bridge. A small `frontend/app.js` affordance (`/check` slash + a button on `.cell` fences), gated on `/api/cellc/status`, posts source and renders diagnostics.

**Tech Stack:** Python 3.10+, FastAPI, pytest + FastAPI TestClient (introduced here — wyltek-gpt has no test setup yet), the `cellc_mcp` package installed editable into wyltek-gpt's `.venv`.

## Global Constraints

- Repo `~/local-chatbot` on branch `feat/cellc-phase1`. It has pre-existing uncommitted WIP (`backend/app.py`, `frontend/{app.js,index.html,style.css}`, `data/capabilities.json`). NEVER `git add -A`; stage ONLY the exact files each task touches. Never commit the user's WIP.
- All cellc endpoints are READ-ONLY: no workspace files, no Y/N/E confirm card, no OPERATIONS-registry involvement, no model `tool_calls` (that is Phase 2).
- Do NOT modify the open_palette bridge or the OPERATIONS registry.
- The bridge must never raise to the endpoint layer for a cellc problem; `cellc_mcp.server.*` already returns error dicts — preserve that.
- `cellc_mcp.server` tool function signatures (verified): `cellc_check(source=None, path=None, target_profile=None, full=False)`, `cellc_explain(code)`, `cellc_metadata(source=None, path=None, target_profile=None, full=False)`, `cellc_constraints(...)`, `cellc_language_reference()`, `cellc_list_examples()`, `cellc_get_example(name)`.
- Binary located via `CELLC_BIN` env (set to `~/CellScript/target/release/cellc` on driveThree); examples/reference via `CELLSCRIPT_REPO=~/CellScript`.
- Validation limits: `_CELLC_MAX_SOURCE = 200_000` chars; `target_profile ∈ {"ckb"}`; `code` matches `^[A-Za-z0-9_]+$`; example `name` matches `^[A-Za-z0-9_-]+$`.
- The FastAPI app object in `backend/app.py` is `app`; import `HTTPException` is already in scope there.

---

## File Structure

- `requirements-dev.txt` — new; `pytest`.
- `tests/__init__.py`, `tests/conftest.py` — new test scaffolding.
- `backend/bridges/cellc.py` — new in-process bridge (imports `cellc_mcp.server`).
- `tests/test_cellc_bridge.py` — bridge unit tests (cellc_mcp mocked).
- `backend/app.py` — ADD the `/api/cellc/*` endpoints (append a new section; do not touch existing routes).
- `tests/test_cellc_endpoints.py` — endpoint tests (bridge mocked, TestClient).
- `tests/test_cellc_integration.py` — `needs_cellc` real-binary tests.
- `frontend/app.js` — ADD the status-gated `/check` affordance (small, self-contained block).
- `README` (or `docs/cellc-setup.md`) — env/venv wiring note.

---

### Task 1: Test scaffolding + the cellc bridge module

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `backend/bridges/cellc.py`
- Create: `tests/test_cellc_bridge.py`

**Interfaces:**
- Consumes: `cellc_mcp.server` (installed into the venv), `cellc_mcp.runner.find_cellc`.
- Produces (in `backend/bridges/cellc.py`):
  - `available() -> bool`
  - `status() -> dict`  (`{"available": bool, "binary": str | None}`)
  - `check(source: str, target_profile: str = "ckb", full: bool = False) -> dict`
  - `explain(code: str) -> dict`
  - `metadata(source: str, target_profile: str = "ckb", full: bool = False) -> dict`
  - `language_reference() -> str`
  - `list_examples() -> list[dict]`
  - `get_example(name: str) -> dict`

- [ ] **Step 1: Install deps + write the failing test**

First install cellc_mcp and pytest into wyltek-gpt's venv:
```bash
~/local-chatbot/.venv/bin/pip install -e ~/cellc-mcp
~/local-chatbot/.venv/bin/pip install pytest
```

`requirements-dev.txt`:
```
pytest>=8.0
```

`tests/__init__.py`: (empty)

`tests/conftest.py`:
```python
import os
import shutil

import pytest


@pytest.fixture
def cellc_available():
    return bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))
```

`tests/test_cellc_bridge.py`:
```python
from unittest import mock

from backend.bridges import cellc


def test_check_forwards_to_cellc_mcp():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_check.return_value = {"ok": True, "diagnostics": []}
        out = cellc.check("module x", target_profile="ckb")
    srv.cellc_check.assert_called_once_with(source="module x", target_profile="ckb", full=False)
    assert out == {"ok": True, "diagnostics": []}


def test_explain_forwards_code():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_explain.return_value = {"ecode": "E0014"}
        out = cellc.explain("E0014")
    srv.cellc_explain.assert_called_once_with(code="E0014")
    assert out["ecode"] == "E0014"


def test_get_example_forwards_name():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_get_example.return_value = {"name": "token", "source": "module x"}
        out = cellc.get_example("token")
    srv.cellc_get_example.assert_called_once_with(name="token")
    assert out["name"] == "token"


def test_available_false_when_binary_missing(monkeypatch):
    # find_cellc raising → not available
    def _raise():
        raise cellc._CellcNotFound("no binary")
    monkeypatch.setattr(cellc, "_find_cellc", _raise)
    assert cellc.available() is False
    assert cellc.status() == {"available": False, "binary": None}


def test_available_true_when_binary_present(monkeypatch):
    monkeypatch.setattr(cellc, "_find_cellc", lambda: "/path/to/cellc")
    assert cellc.available() is True
    assert cellc.status() == {"available": True, "binary": "/path/to/cellc"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_bridge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.bridges.cellc'`.

- [ ] **Step 3: Write minimal implementation**

`backend/bridges/cellc.py`:
```python
"""In-process bridge to the cellc CellScript compiler tooling.

Imports the cellc-mcp server's tool functions directly (both are local
Python on driveThree) and exposes thin JSON-able wrappers. All heavy
lifting — package synthesis, output tiering, module-scan isolation, the
never-raises contract — lives in ``cellc_mcp`` and must not be duplicated
here. If ``cellc_mcp`` is not installed, this module still imports and
``available()`` returns False so wyltek-gpt boots without the feature.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from cellc_mcp import server as _server
    from cellc_mcp.runner import CellcNotFound as _CellcNotFound
    from cellc_mcp.runner import find_cellc as _find_cellc
    _IMPORT_OK = True
except Exception as exc:  # pragma: no cover - exercised only without cellc_mcp
    log.warning("cellc_mcp not importable — cellc bridge disabled: %s", exc)
    _server = None  # type: ignore[assignment]
    _CellcNotFound = Exception  # type: ignore[assignment,misc]
    _find_cellc = None  # type: ignore[assignment]
    _IMPORT_OK = False


def _binary() -> str | None:
    if not _IMPORT_OK or _find_cellc is None:
        return None
    try:
        return _find_cellc()
    except _CellcNotFound:
        return None


def available() -> bool:
    return _binary() is not None


def status() -> dict:
    binary = _binary()
    return {"available": binary is not None, "binary": binary}


def check(source: str, target_profile: str = "ckb", full: bool = False) -> dict:
    return _server.cellc_check(source=source, target_profile=target_profile, full=full)


def explain(code: str) -> dict:
    return _server.cellc_explain(code=code)


def metadata(source: str, target_profile: str = "ckb", full: bool = False) -> dict:
    return _server.cellc_metadata(source=source, target_profile=target_profile, full=full)


def language_reference() -> str:
    return _server.cellc_language_reference()


def list_examples() -> list[dict]:
    return _server.cellc_list_examples()


def get_example(name: str) -> dict:
    return _server.cellc_get_example(name=name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_bridge.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add requirements-dev.txt tests/__init__.py tests/conftest.py backend/bridges/cellc.py tests/test_cellc_bridge.py
git commit -m "feat: cellc in-process bridge + test scaffolding"
```

---

### Task 2: Dedicated `/api/cellc/*` endpoints

**Files:**
- Modify: `backend/app.py` (append a new "cellc endpoints" section near the other `/api/operations/*` routes; do not edit existing routes)
- Create: `tests/test_cellc_endpoints.py`

**Interfaces:**
- Consumes: `backend.bridges.cellc` (Task 1).
- Produces routes: `GET /api/cellc/status`, `POST /api/cellc/check`, `POST /api/cellc/explain`, `POST /api/cellc/metadata`, `GET /api/cellc/reference`, `GET /api/cellc/examples`, `GET /api/cellc/example/{name}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cellc_endpoints.py`:
```python
from unittest import mock

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def test_status_endpoint():
    with mock.patch.object(app_module.cellc_bridge, "status",
                           return_value={"available": True, "binary": "/x/cellc"}):
        r = client.get("/api/cellc/status")
    assert r.status_code == 200
    assert r.json()["available"] is True


def test_check_happy_path():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True), \
         mock.patch.object(app_module.cellc_bridge, "check",
                           return_value={"ok": False, "diagnostics": [], "error_count": 1}) as chk:
        r = client.post("/api/cellc/check", json={"source": "module x", "target_profile": "ckb"})
    assert r.status_code == 200
    assert r.json()["error_count"] == 1
    chk.assert_called_once_with(source="module x", target_profile="ckb", full=False)


def test_check_empty_source_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/check", json={"source": ""})
    assert r.status_code == 400


def test_check_bad_profile_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/check", json={"source": "module x", "target_profile": "evm"})
    assert r.status_code == 400


def test_check_oversized_source_400():
    big = "x" * 200_001
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/check", json={"source": big})
    assert r.status_code == 400


def test_check_503_when_unavailable():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=False):
        r = client.post("/api/cellc/check", json={"source": "module x"})
    assert r.status_code == 503


def test_explain_bad_code_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/explain", json={"code": "E14; rm -rf"})
    assert r.status_code == 400


def test_example_bad_name_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.get("/api/cellc/example/..%2f..%2fetc%2fpasswd")
    assert r.status_code in (400, 404)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_endpoints.py -v`
Expected: FAIL (`AttributeError: module 'backend.app' has no attribute 'cellc_bridge'` / 404s).

- [ ] **Step 3: Write minimal implementation**

In `backend/app.py`, add the import near the other bridge import (`from backend.bridges import open_palette`):
```python
from backend.bridges import cellc as cellc_bridge
```

Append this section (after the `/api/operations/*` routes):
```python
# ─── cellc (CellScript) endpoints — read-only, no workspace files ─────
# Dedicated like /api/operations/translate (cellc returns DATA, not a
# file, so it does not fit the file-producing OPERATIONS registry).

# `re` and `asyncio` are already imported at the top of backend/app.py
# (lines 3 and 8) — use them directly; do NOT add duplicate imports.
_CELLC_MAX_SOURCE = 200_000
_CELLC_PROFILES = {"ckb"}
_CELLC_CODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
_CELLC_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _cellc_require_available() -> None:
    if not cellc_bridge.available():
        raise HTTPException(
            503, "cellc not available — build it and set CELLC_BIN "
                 "(cd ~/CellScript && cargo build --release -p cellscript --bin cellc)"
        )


def _cellc_validate_source(payload: dict) -> str:
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise HTTPException(400, "missing or empty 'source'")
    if len(source) > _CELLC_MAX_SOURCE:
        raise HTTPException(413, f"source too long ({len(source)} > {_CELLC_MAX_SOURCE})")
    return source


def _cellc_validate_profile(payload: dict) -> str:
    profile = payload.get("target_profile", "ckb")
    if profile not in _CELLC_PROFILES:
        raise HTTPException(400, f"invalid target_profile {profile!r}; allowed: {sorted(_CELLC_PROFILES)}")
    return profile


@app.get("/api/cellc/status")
async def cellc_status():
    return cellc_bridge.status()


@app.post("/api/cellc/check")
async def cellc_check_endpoint(payload: dict):
    _cellc_require_available()
    source = _cellc_validate_source(payload)
    profile = _cellc_validate_profile(payload)
    full = bool(payload.get("full"))
    return await asyncio.to_thread(cellc_bridge.check, source=source, target_profile=profile, full=full)


@app.post("/api/cellc/metadata")
async def cellc_metadata_endpoint(payload: dict):
    _cellc_require_available()
    source = _cellc_validate_source(payload)
    profile = _cellc_validate_profile(payload)
    full = bool(payload.get("full"))
    return await asyncio.to_thread(cellc_bridge.metadata, source=source, target_profile=profile, full=full)


@app.post("/api/cellc/explain")
async def cellc_explain_endpoint(payload: dict):
    _cellc_require_available()
    code = payload.get("code")
    if not isinstance(code, str) or not _CELLC_CODE_RE.match(code):
        raise HTTPException(400, "invalid 'code' (expected an error code/name like E0014)")
    return await asyncio.to_thread(cellc_bridge.explain, code=code)


@app.get("/api/cellc/reference")
async def cellc_reference_endpoint():
    _cellc_require_available()
    return {"reference": cellc_bridge.language_reference()}


@app.get("/api/cellc/examples")
async def cellc_examples_endpoint():
    _cellc_require_available()
    return {"examples": cellc_bridge.list_examples()}


@app.get("/api/cellc/example/{name}")
async def cellc_example_endpoint(name: str):
    _cellc_require_available()
    if not _CELLC_NAME_RE.match(name):
        raise HTTPException(400, "invalid example name")
    result = cellc_bridge.get_example(name)
    if result.get("tool_error"):
        raise HTTPException(404, result.get("stderr", "example not found"))
    return result
```
(Verified: `backend/app.py` already imports both `asyncio` (line 3) and `re` (line 8) at module top — use them directly, add no duplicate imports.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_endpoints.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add backend/app.py tests/test_cellc_endpoints.py
git commit -m "feat: /api/cellc/* read-only endpoints"
```

---

### Task 3: Integration tests + venv/env wiring

**Files:**
- Create: `tests/test_cellc_integration.py`
- Create: `docs/cellc-setup.md`

**Interfaces:**
- Consumes: the real `cellc` binary + `cellc_mcp` (via the endpoints from Task 2).
- Produces: documented setup; `needs_cellc` smoke coverage over the HTTP layer.

- [ ] **Step 1: Write the failing/skipping tests**

`tests/test_cellc_integration.py`:
```python
import os
import shutil

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module

pytestmark = pytest.mark.needs_cellc
_HAS_CELLC = bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))
client = TestClient(app_module.app)

_GOOD = """\
module probe::token
resource Token has store, create, consume, replace { amount: u64, symbol: [u8; 8] }
action transfer_token(token: Token, to: Address) -> next_token: Token {
    verification
        consume token
        create next_token = Token { amount: token.amount, symbol: token.symbol } with_lock(to)
}
"""
_BAD = "module probe::broken\n\nresource Token has store { amount: u64\n"


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_status_reports_available():
    assert client.get("/api/cellc/status").json()["available"] is True


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_check_good_then_bad():
    good = client.post("/api/cellc/check", json={"source": _GOOD}).json()
    assert good["ok"] is True
    bad = client.post("/api/cellc/check", json={"source": _BAD}).json()
    assert bad["ok"] is False and bad["error_count"] >= 1 and bad["diagnostics"]


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_explain_e0014():
    assert client.post("/api/cellc/explain", json={"code": "E0014"}).json()["ecode"] == "E0014"
```

Register the marker so pytest doesn't warn. In `tests/conftest.py` add:
```python
def pytest_configure(config):
    config.addinivalue_line("markers", "needs_cellc: requires a built cellc binary")
```

- [ ] **Step 2: Run to verify status**

Run (no binary): `cd ~/local-chatbot && .venv/bin/python -m pytest tests/test_cellc_integration.py -v`
Expected: 3 SKIPPED.

Run (real binary):
```bash
cd ~/local-chatbot
export CELLC_BIN=/home/phill/CellScript/target/release/cellc
export CELLSCRIPT_REPO=/home/phill/CellScript
.venv/bin/python -m pytest tests/test_cellc_integration.py -v
```
Expected: 3 PASSED. If `_GOOD` does not type-check against the real binary, replace it with the body of `~/CellScript/examples/token/src/main.cell` and re-run until `ok is True`.

- [ ] **Step 3: Write the setup doc**

`docs/cellc-setup.md`:
```markdown
# cellc integration setup (wyltek-gpt, driveThree)

The /api/cellc/* endpoints call the cellc CellScript compiler via the
cellc-mcp package. To enable them:

1. Build cellc once (needs a sibling ckb-sdk-rust @ v5.1.0 checkout):
       cd ~/CellScript && cargo build --release -p cellscript --bin cellc

2. Install the bridge package into wyltek-gpt's venv:
       ~/local-chatbot/.venv/bin/pip install -e ~/cellc-mcp

3. Start uvicorn with these env vars set:
       CELLC_BIN=/home/phill/CellScript/target/release/cellc
       CELLSCRIPT_REPO=/home/phill/CellScript
       .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000

Without these, /api/cellc/status returns {"available": false} and the
write/check endpoints return 503 — wyltek-gpt still boots normally.

## Tests
    .venv/bin/python -m pytest -v                 # unit (offline)
    CELLC_BIN=... CELLSCRIPT_REPO=... .venv/bin/python -m pytest -m needs_cellc -v
```

- [ ] **Step 4: Run the full offline suite**

Run: `cd ~/local-chatbot && .venv/bin/python -m pytest -v`
Expected: all unit tests PASS; integration SKIPPED (or PASS with the binary + env).

- [ ] **Step 5: Commit (stage only these files)**

```bash
cd ~/local-chatbot
git add tests/test_cellc_integration.py tests/conftest.py docs/cellc-setup.md
git commit -m "test: cellc endpoint integration (gated) + setup doc"
```

---

### Task 4: Frontend affordance (status-gated `/check`)

**Files:**
- Modify: `frontend/app.js` (append one small self-contained block; do not refactor existing code)

**Interfaces:**
- Consumes: `GET /api/cellc/status`, `POST /api/cellc/check`.
- Produces: a `/check` slash command + a "check" action on `.cell`/`cellscript` code fences, shown only when cellc is available.

**Note:** wyltek-gpt has no JS test harness; this task is verified by a manual smoke check (acceptance below). Keep the addition minimal and self-contained so it can't regress existing rendering.

- [ ] **Step 1: Add the affordance**

Append to `frontend/app.js` (a self-contained IIFE so it doesn't collide with existing globals):
```javascript
// ─── cellc (CellScript) check affordance — Phase 1, human-driven ──────
(function cellcAffordance() {
  let cellcEnabled = false;

  async function refreshCellcStatus() {
    try {
      const r = await fetch("/api/cellc/status");
      cellcEnabled = r.ok && (await r.json()).available === true;
    } catch { cellcEnabled = false; }
    document.body.classList.toggle("cellc-enabled", cellcEnabled);
  }

  function renderCheckResult(data) {
    if (data.tool_error) return `cellc error: ${data.stderr || "unknown"}`;
    if (data.ok) return "✓ cellc check passed";
    const lines = (data.diagnostics || []).map(
      (d) => `  L${d.line}:C${d.column} [${d.code || "—"}] ${d.message}`
    );
    let out = `✗ ${data.error_count} error(s)\n` + lines.join("\n");
    if (data.truncated) out += `\n  …(+${data.truncated} more)`;
    return out;
  }

  async function checkSource(source) {
    if (!cellcEnabled) return "cellc is not available on this server.";
    try {
      const r = await fetch("/api/cellc/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source }),
      });
      if (r.status === 503) return "cellc is not available on this server.";
      if (!r.ok) return `cellc check failed: HTTP ${r.status}`;
      return renderCheckResult(await r.json());
    } catch (e) { return `cellc check failed: ${e}`; }
  }

  // Expose for the slash-command handler + fence buttons to call.
  window.cellc = { checkSource, refreshCellcStatus, get enabled() { return cellcEnabled; } };
  refreshCellcStatus();
})();
```

Wire the `/check` slash command into the existing message-send path: where the
composer handles input, add — before normal send — a branch that, if the
trimmed input starts with `/check`, extracts the most recent assistant
`.cell`/`cellscript` code fence (or the text after `/check `), calls
`await window.cellc.checkSource(src)`, and renders the returned string as a
system/info bubble using the existing message-append helper. (Use the same
append function the op-confirm path uses; do not invent a new render path.)

- [ ] **Step 2: Manual smoke test (acceptance)**

With the backend running and `CELLC_BIN` set:
1. `curl -s localhost:8000/api/cellc/status` → `{"available": true, ...}`.
2. In the UI, send `/check` after the model emits a `.cell` block (or `/check module probe::x\nresource T has store { amount: u64 }`) → an info bubble shows `✓ cellc check passed` or a diagnostics list.
3. Stop the backend env (`unset CELLC_BIN`, restart) → `/api/cellc/status` is `{"available": false}` and the `/check` bubble says "cellc is not available".

Record the three results in the commit message.

- [ ] **Step 3: Commit (stage only this file)**

```bash
cd ~/local-chatbot
git add frontend/app.js
git commit -m "feat: frontend /check affordance for cellc (status-gated)"
```

---

## Self-Review

**Spec coverage:**
- Component 1 bridge module → Task 1. ✓
- Component 2 endpoints (status/check/metadata/explain/reference/examples/example) → Task 2. ✓
- Validation rules (source/profile/code/name/size, 503-when-unavailable) → Task 2. ✓
- Component 3 frontend affordance (status-gated `/check`) → Task 4. ✓
- Component 4 wiring/env (venv install, CELLC_BIN/CELLSCRIPT_REPO) → Task 1 Step 1 + Task 3 doc. ✓
- Testing (bridge unit, endpoint unit, gated integration, introduce pytest) → Tasks 1–3. ✓
- Non-goals (no OPERATIONS registry, no confirm card, no file output, no agent loop, open_palette untouched) → enforced by construction; endpoints are dedicated + read-only. ✓
- Git safety (branch, stage-only-own-files) → Global Constraints + every commit step names exact files. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete. The one judgement call (where to wire `/check` into the composer) names the exact helper to reuse and the exact trigger — not a placeholder, a constrained instruction for an unfamiliar file.

**Type consistency:** `cellc_bridge` is the import alias used in both `backend/app.py` and the endpoint tests. Bridge functions `check(source, target_profile, full)` / `explain(code)` / `metadata(...)` / `get_example(name)` match their endpoint callers and their unit-test mocks. `status()` shape `{"available", "binary"}` consistent across bridge, endpoint, and tests. `_server` is the patched attribute name in `test_cellc_bridge.py` and the import alias in `backend/bridges/cellc.py`.

**Known risk flagged in-plan:** Task 2 notes the possible pre-existing `re`/`asyncio` imports in `app.py` (verify before adding duplicates). Task 4 is manually verified (no JS harness) with explicit acceptance steps.
