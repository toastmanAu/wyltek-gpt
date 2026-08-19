# wyltek-gpt Build Tab Integration Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Build" tab to the wyltek-gpt PWA that drives local-code's build daemon (Plan 1) — streaming build progress + screenshots into the UI, gating each `run_shell` behind the existing confirm card, embedding the live Vite preview, and releasing the chat model so build and chat don't fight over the GPU.

**Architecture:** A new in-process bridge (`backend/bridges/buildd.py`, mirroring `cellc.py`) is an HTTP client to the loopback daemon. New `/api/build/*` FastAPI routes proxy to it, are guarded to localhost callers, and perform the VRAM handshake (release the selected chat model via Ollama `keep_alive:0`) before forwarding `start`. The PWA gets a Build tab that consumes the daemon's SSE stream (proxied through FastAPI), reuses the confirm-card machinery for shell approvals, and shows an iframe preview on `preview_ready`.

**Tech Stack:** Python 3 / FastAPI / httpx / pytest (backend); vanilla JS PWA (frontend). No new dependencies — `httpx` and `StreamingResponse` are already used.

**Depends on:** Plan 1 (`docs/superpowers/plans/2026-06-29-build-daemon.md` in the local-code repo). Implement against the daemon contract documented in `packages/cli/src/daemon/README.md`. Daemon default loopback port: **5179**.

## Global Constraints

- The build surface is **desktop/localhost only**: every `/api/build/*` route rejects a request whose `request.client.host` is not `127.0.0.1`/`::1` with `403`. No auth is added; loopback IS the boundary.
- The bridge degrades gracefully: if the daemon is unreachable, `buildd.available()` is `False`, the Build tab renders disabled, and chat/convert are unaffected — exactly the `cellc.py` pattern.
- Do NOT add duplicate imports in `app.py`: `re`, `asyncio`, `json`, `httpx`, `HTTPException`, `StreamingResponse`, `Request` — check the top of the file; `re`/`asyncio` are at lines 3/8, `httpx`/`StreamingResponse` already imported.
- The daemon base URL/port come from `config.yaml` `buildd:` — never hard-coded in routes.
- Tests use pytest + `fastapi.testclient.TestClient`; run with `cd ~/local-chatbot && python -m pytest tests/ -q`.
- Conventional-commit messages; commit after each task's tests pass.

---

## File structure

- `backend/bridges/buildd.py` — CREATE: httpx client to the daemon (`available`, `start`, `stream`, `approve`, `cancel`, `screenshot_bytes`).
- `config.yaml` — MODIFY: add `buildd:` block under the existing `bridges:` neighbours (it's a sibling service, mirror `open_palette`).
- `backend/app.py` — MODIFY: load buildd config; add `/api/build/{start,events,approve,cancel}` + screenshot proxy; add `_require_localhost` guard + `_release_chat_model` handshake helper.
- `frontend/index.html` — MODIFY: add the Build tab markup (start form, log area, preview iframe container).
- `frontend/app.js` — MODIFY: Build-tab controller — start, consume SSE, render events, present shell-approval card (reuse `buildOpCard`-style), show iframe, "GPU busy" state.
- `tests/test_buildd_bridge.py` — CREATE.
- `tests/test_build_routes.py` — CREATE.

---

### Task 1: `buildd` bridge

**Files:**
- Create: `backend/bridges/buildd.py`
- Test: `tests/test_buildd_bridge.py`

**Interfaces (Produces):**
```python
def configure(base_url: str, *, enabled: bool) -> None  # called once at app startup from config
def available() -> bool                                  # enabled AND daemon /healthz reachable
def status() -> dict                                     # {"available": bool, "base_url": str}
async def start(prompt: str, *, profile: str | None, critique: bool) -> dict   # -> {"buildId": str}; raises BuilddBusy on 409
async def stream(build_id: str) -> AsyncIterator[bytes]  # raw SSE bytes from GET /builds/:id/events
async def approve(build_id: str, call_id: str, decision: str, command: str | None, reason: str | None) -> dict
async def cancel(build_id: str) -> dict
async def screenshot_bytes(build_id: str, name: str) -> bytes | None  # None on 404
class BuilddBusy(Exception): ...   # raised when daemon returns 409 build_in_progress
```

- [ ] **Step 1: Write the failing test** (a fake ASGI/daemon via `httpx.MockTransport`)

```python
# tests/test_buildd_bridge.py
import httpx
import pytest
from backend.bridges import buildd


def _client_factory(handler):
    # Patch the module to use a MockTransport-backed AsyncClient.
    transport = httpx.MockTransport(handler)
    return lambda: httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:5179")


@pytest.mark.asyncio
async def test_available_false_when_disabled():
    buildd.configure("http://127.0.0.1:5179", enabled=False)
    assert buildd.available() is False


@pytest.mark.asyncio
async def test_start_returns_build_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/builds"
        return httpx.Response(201, json={"buildId": "build-0"})
    buildd.configure("http://127.0.0.1:5179", enabled=True)
    monkeypatch.setattr(buildd, "_make_client", _client_factory(handler))
    out = await buildd.start("make a page", profile=None, critique=False)
    assert out["buildId"] == "build-0"


@pytest.mark.asyncio
async def test_start_raises_busy_on_409(monkeypatch):
    def handler(request):
        return httpx.Response(409, json={"error": "build_in_progress"})
    buildd.configure("http://127.0.0.1:5179", enabled=True)
    monkeypatch.setattr(buildd, "_make_client", _client_factory(handler))
    with pytest.raises(buildd.BuilddBusy):
        await buildd.start("x", profile=None, critique=False)


@pytest.mark.asyncio
async def test_screenshot_none_on_404(monkeypatch):
    def handler(request):
        return httpx.Response(404)
    buildd.configure("http://127.0.0.1:5179", enabled=True)
    monkeypatch.setattr(buildd, "_make_client", _client_factory(handler))
    assert await buildd.screenshot_bytes("build-0", "x.png") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/local-chatbot && python -m pytest tests/test_buildd_bridge.py -q`
Expected: FAIL — `backend.bridges.buildd` does not exist.

- [ ] **Step 3: Implement `buildd.py`**

```python
# backend/bridges/buildd.py
"""In-process HTTP client to the local-code build daemon (loopback).

Mirrors the cellc bridge contract: if the daemon is disabled or unreachable
``available()`` returns False so wyltek-gpt boots and the Build tab disables
itself. All build orchestration lives in the daemon (Plan 1); this module is
a thin, never-duplicate-logic client.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger(__name__)

_BASE_URL: str = "http://127.0.0.1:5179"
_ENABLED: bool = False


class BuilddBusy(Exception):
    """Daemon already has an active build (HTTP 409)."""


def configure(base_url: str, *, enabled: bool) -> None:
    global _BASE_URL, _ENABLED
    _BASE_URL = base_url.rstrip("/")
    _ENABLED = enabled


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=_BASE_URL, timeout=None)


def available() -> bool:
    if not _ENABLED:
        return False
    try:
        with httpx.Client(base_url=_BASE_URL, timeout=1.0) as c:
            return c.get("/healthz").status_code == 200
    except Exception:
        return False


def status() -> dict:
    return {"available": available(), "base_url": _BASE_URL}


async def start(prompt: str, *, profile: str | None, critique: bool) -> dict:
    body: dict = {"prompt": prompt, "critique": critique}
    if profile:
        body["profile"] = profile
    async with _make_client() as c:
        r = await c.post("/builds", json=body)
        if r.status_code == 409:
            raise BuilddBusy(r.json().get("error", "build_in_progress"))
        r.raise_for_status()
        return r.json()


async def stream(build_id: str) -> AsyncIterator[bytes]:
    async with _make_client() as c:
        async with c.stream("GET", f"/builds/{build_id}/events") as r:
            r.raise_for_status()
            async for chunk in r.aiter_raw():
                yield chunk


async def approve(build_id: str, call_id: str, decision: str,
                  command: str | None, reason: str | None) -> dict:
    body: dict = {"decision": decision}
    if command:
        body["command"] = command
    if reason:
        body["reason"] = reason
    async with _make_client() as c:
        r = await c.post(f"/builds/{build_id}/approvals/{call_id}", json=body)
        r.raise_for_status()
        return r.json()


async def cancel(build_id: str) -> dict:
    async with _make_client() as c:
        r = await c.post(f"/builds/{build_id}/cancel")
        r.raise_for_status()
        return r.json()


async def screenshot_bytes(build_id: str, name: str) -> bytes | None:
    async with _make_client() as c:
        r = await c.get(f"/builds/{build_id}/screenshots/{name}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
```

> The daemon (Plan 1, Task 7) must expose `GET /healthz` returning `200`. If Plan 1 shipped without it, add it there before this task — a one-line route returning `{"ok": true}` from the same loopback server. Note this as a contract follow-up.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/local-chatbot && python -m pytest tests/test_buildd_bridge.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/bridges/buildd.py tests/test_buildd_bridge.py
git commit -m "feat(buildd): HTTP bridge to local-code build daemon"
```

---

### Task 2: `buildd` config + startup wiring

**Files:**
- Modify: `config.yaml`
- Modify: `backend/app.py` (load config → `buildd.configure(...)`)
- Test: folded into Task 3 (`test_build_routes.py` asserts the routes exist, which requires this wiring).

**Interfaces:** Consumes `buildd.configure` (Task 1).

- [ ] **Step 1: Add the config block**

In `config.yaml`, under `bridges:` (sibling to `open_palette`):

```yaml
bridges:
  open_palette:
    url: http://localhost:7860
    # ...existing...
  buildd:
    url: http://127.0.0.1:5179   # local-code build daemon (loopback only)
    enabled: true
```

- [ ] **Step 2: Wire it at startup**

Near the top of `app.py` where `cellc_bridge` is imported/initialised (around line 18 / 91), add:

```python
from backend.bridges import buildd as buildd_bridge
# ... after CONFIG is loaded ...
_buildd_cfg = (CONFIG.get("bridges", {}) or {}).get("buildd", {}) or {}
buildd_bridge.configure(
    _buildd_cfg.get("url", "http://127.0.0.1:5179"),
    enabled=bool(_buildd_cfg.get("enabled", False)),
)
```

> Use the same `CONFIG` object the rest of `app.py` reads (the loaded `config.yaml`). If the config is loaded into a differently-named variable, match it; do not introduce a second loader.

- [ ] **Step 3: Smoke**

Run: `cd ~/local-chatbot && python -c "import backend.app"`
Expected: imports without error (daemon need not be running — `enabled` just arms the bridge).

- [ ] **Step 4: Commit**

```bash
git add config.yaml backend/app.py
git commit -m "feat(buildd): config block + startup wiring"
```

---

### Task 3: `/api/build/*` routes + localhost guard + VRAM handshake

**Files:**
- Modify: `backend/app.py`
- Test: `tests/test_build_routes.py`

**Interfaces:**
- Consumes: `buildd_bridge` (Tasks 1–2).
- Produces routes:
  - `POST /api/build/start` body `{prompt, profile?, critique?, chat_model?}` → `{buildId}`; releases `chat_model` first; `409` on `BuilddBusy`; `503` if `not available()`.
  - `GET /api/build/events/{build_id}` → SSE pass-through (`media_type="text/event-stream"`).
  - `POST /api/build/approve/{build_id}/{call_id}` body `{decision, command?, reason?}` → daemon result.
  - `POST /api/build/cancel/{build_id}` → daemon result.
  - `GET /api/build/screenshot/{build_id}/{name}` → `image/png` or `404`.
  - All reject non-loopback callers with `403`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_routes.py
import httpx
import pytest
from fastapi.testclient import TestClient
from backend.bridges import buildd
import backend.app as appmod

client = TestClient(appmod.app)


def test_localhost_guard_blocks_non_loopback(monkeypatch):
    buildd.configure("http://127.0.0.1:5179", enabled=True)
    monkeypatch.setattr(buildd, "available", lambda: True)
    # TestClient defaults to testclient host; simulate a LAN client.
    r = client.post("/api/build/start", json={"prompt": "x"},
                    headers={"x-forwarded-for": "192.168.1.50"})
    # The guard reads request.client.host; TestClient sets it to "testclient".
    # Force a non-loopback host via the ASGI scope override below instead.
    # (See Step 3 for how the guard determines the host.)
    assert r.status_code in (403, 200, 503)  # replaced by precise assertion after Step 3


def test_start_503_when_unavailable(monkeypatch):
    monkeypatch.setattr(buildd, "available", lambda: False)
    r = client.post("/api/build/start", json={"prompt": "x"})
    assert r.status_code == 503


def test_start_releases_chat_model_then_forwards(monkeypatch):
    monkeypatch.setattr(buildd, "available", lambda: True)
    released = {}
    async def fake_release(model):
        released["model"] = model
    async def fake_start(prompt, *, profile, critique):
        assert "model" in released, "chat model must be released BEFORE forwarding start"
        return {"buildId": "build-0"}
    monkeypatch.setattr(appmod, "_release_chat_model", fake_release)
    monkeypatch.setattr(buildd, "start", fake_start)
    r = client.post("/api/build/start", json={"prompt": "x", "chat_model": "qwen3.6:27b"})
    assert r.status_code == 200
    assert r.json()["buildId"] == "build-0"
    assert released["model"] == "qwen3.6:27b"


def test_start_409_on_busy(monkeypatch):
    monkeypatch.setattr(buildd, "available", lambda: True)
    async def fake_release(model): pass
    async def fake_start(prompt, *, profile, critique):
        raise buildd.BuilddBusy("build_in_progress")
    monkeypatch.setattr(appmod, "_release_chat_model", fake_release)
    monkeypatch.setattr(buildd, "start", fake_start)
    r = client.post("/api/build/start", json={"prompt": "x"})
    assert r.status_code == 409
```

> The localhost-guard test is finalised in Step 3 once the guard's host source is fixed; the placeholder assertion above is replaced with `assert r.status_code == 403` after wiring the ASGI scope override helper shown in Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/local-chatbot && python -m pytest tests/test_build_routes.py -q`
Expected: FAIL — routes / `_release_chat_model` don't exist.

- [ ] **Step 3: Implement the guard, the handshake, and the routes**

Add near the other helpers in `app.py` (reuse imported `Request`, `StreamingResponse`, `HTTPException`, `httpx`, `json`; add `from fastapi import Request` to the existing fastapi import if not present):

```python
# ─── build daemon proxy (desktop/localhost only) ──────────────────────
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}


def _require_localhost(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in _LOOPBACK_HOSTS:
        raise HTTPException(403, "build is available on localhost only")


def _require_buildd() -> None:
    if not buildd_bridge.available():
        raise HTTPException(503, "build daemon not running — start it with `local-code serve`")


async def _release_chat_model(model: str | None) -> None:
    """VRAM handshake: evict the PWA's chat model so the build has the GPU.
    Best-effort — a failure here just means the daemon's own preflight unload
    does more work. Never raises."""
    if not model:
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": model, "keep_alive": 0, "prompt": ""})
    except Exception:
        log.warning("chat-model release failed for %s (continuing)", model)


@app.post("/api/build/start")
async def build_start(payload: dict, request: Request):
    _require_localhost(request)
    _require_buildd()
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "missing 'prompt'")
    await _release_chat_model(payload.get("chat_model"))
    try:
        return await buildd_bridge.start(
            prompt,
            profile=payload.get("profile"),
            critique=bool(payload.get("critique")),
        )
    except buildd_bridge.BuilddBusy:
        raise HTTPException(409, "a build is already running")


@app.get("/api/build/events/{build_id}")
async def build_events(build_id: str, request: Request):
    _require_localhost(request)
    _require_buildd()

    async def passthrough():
        async for chunk in buildd_bridge.stream(build_id):
            yield chunk

    return StreamingResponse(passthrough(), media_type="text/event-stream")


@app.post("/api/build/approve/{build_id}/{call_id}")
async def build_approve(build_id: str, call_id: str, payload: dict, request: Request):
    _require_localhost(request)
    _require_buildd()
    decision = payload.get("decision")
    if decision not in ("approve", "deny"):
        raise HTTPException(400, "decision must be 'approve' or 'deny'")
    return await buildd_bridge.approve(
        build_id, call_id, decision, payload.get("command"), payload.get("reason"),
    )


@app.post("/api/build/cancel/{build_id}")
async def build_cancel(build_id: str, request: Request):
    _require_localhost(request)
    _require_buildd()
    return await buildd_bridge.cancel(build_id)


@app.get("/api/build/screenshot/{build_id}/{name}")
async def build_screenshot(build_id: str, name: str, request: Request):
    _require_localhost(request)
    _require_buildd()
    if "/" in name or ".." in name:
        raise HTTPException(400, "invalid screenshot name")
    data = await buildd_bridge.screenshot_bytes(build_id, name)
    if data is None:
        raise HTTPException(404, "screenshot not found")
    return Response(content=data, media_type="image/png")
```

Add `Response` to the `fastapi.responses` import if not present. Finalise the guard test: in `test_localhost_guard_blocks_non_loopback`, override the client host by sending the request through an ASGI scope with `client=("192.168.1.50", 1234)`:

```python
def test_localhost_guard_blocks_non_loopback(monkeypatch):
    monkeypatch.setattr(buildd, "available", lambda: True)
    # Build a raw scope so request.client.host is a LAN IP.
    from starlette.requests import Request as SReq  # noqa
    transport = httpx.ASGITransport(app=appmod.app, client=("192.168.1.50", 1234))
    with httpx.Client(transport=transport, base_url="http://testserver") as c:
        r = c.post("/api/build/start", json={"prompt": "x"})
    assert r.status_code == 403
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/local-chatbot && python -m pytest tests/test_build_routes.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app.py tests/test_build_routes.py
git commit -m "feat(build): /api/build/* proxy with localhost guard + VRAM handshake"
```

---

### Task 4: PWA Build tab — markup

**Files:**
- Modify: `frontend/index.html`

- [ ] **Step 1: Add the Build tab markup**

Add a tab toggle near the existing composer/mode controls and a Build panel (hidden by default). Match existing class conventions (inspect surrounding markup first; the names below are indicative):

```html
<!-- frontend/index.html -->
<div id="build-panel" class="panel" hidden>
  <form id="build-form">
    <textarea id="build-prompt" placeholder="Describe the app to build (React + Vite)…" rows="3"></textarea>
    <label><input type="checkbox" id="build-critique"> vision critique + repair</label>
    <button type="submit" id="build-start">Build</button>
    <button type="button" id="build-cancel" hidden>Cancel</button>
  </form>
  <div id="build-status" class="build-status"></div>
  <div id="build-log" class="build-log" aria-live="polite"></div>
  <iframe id="build-preview" class="build-preview" hidden title="live preview"></iframe>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(build): Build tab markup"
```

---

### Task 5: PWA Build tab — controller (SSE consume, approval card, preview)

**Files:**
- Modify: `frontend/app.js`

**Interfaces:** Reuses existing `appendMsg`, `startSpinner`, and the confirm-card visual pattern of `buildOpCard`. Uses native `EventSource` against `GET /api/build/events/{id}` (a GET SSE endpoint — `EventSource` is the right primitive here, unlike the chat POST stream).

- [ ] **Step 1: Implement the controller**

```javascript
// frontend/app.js — Build tab controller
let buildState = { id: null, es: null };

async function startBuild() {
  const prompt = document.querySelector("#build-prompt").value.trim();
  if (!prompt) return;
  const critique = document.querySelector("#build-critique").checked;
  const statusEl = document.querySelector("#build-status");
  statusEl.textContent = "starting…";
  try {
    const r = await fetch("/api/build/start", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, critique, chat_model: modelSel.value }),
    });
    if (r.status === 409) { statusEl.textContent = "a build is already running"; return; }
    if (r.status === 503) { statusEl.textContent = "build engine offline — run `local-code serve`"; return; }
    if (!r.ok) { statusEl.textContent = `start failed: ${r.status}`; return; }
    const { buildId } = await r.json();
    buildState.id = buildId;
    document.querySelector("#build-cancel").hidden = false;
    subscribeBuild(buildId);
  } catch (e) {
    statusEl.textContent = `start error: ${e.message}`;
  }
}

function subscribeBuild(buildId) {
  const log = document.querySelector("#build-log");
  const statusEl = document.querySelector("#build-status");
  const es = new EventSource(`/api/build/events/${encodeURIComponent(buildId)}`);
  buildState.es = es;
  es.onmessage = (ev) => {
    let evt;
    try { evt = JSON.parse(ev.data); } catch { return; }
    handleBuildEvent(evt, { log, statusEl, buildId });
  };
  es.onerror = () => { statusEl.textContent = "stream closed"; es.close(); };
}

function handleBuildEvent(evt, ctx) {
  const { log, statusEl, buildId } = ctx;
  switch (evt.type) {
    case "phase":
      statusEl.textContent = (evt.phase === "preflight" || evt.phase === "builder")
        ? "GPU busy building — chat paused" : `phase: ${evt.phase}`;
      setChatPaused(evt.phase === "preflight" || evt.phase === "builder");
      logLine(log, `── ${evt.phase} ──`);
      break;
    case "agent":
      if (evt.event.type === "tool_call") logLine(log, `↪ ${evt.event.call.name}`);
      else if (evt.event.type === "content") appendInline(log, evt.event.delta);
      break;
    case "approval_required":
      presentShellApproval(buildId, evt.callId, evt.command, evt.cwd, log);
      break;
    case "approval_resolved":
      logLine(log, `${evt.decision}${evt.edited ? " (edited)" : ""}${evt.reason ? ` [${evt.reason}]` : ""}`);
      break;
    case "screenshot":
      appendScreenshot(log, buildId, evt.name, evt.cycle);
      break;
    case "critique":
      logLine(log, `critique cycle ${evt.cycle}: score=${evt.score.toFixed(2)} — ${evt.summary}`);
      break;
    case "preview_ready":
      showPreview(evt.url);
      break;
    case "error":
      logLine(log, `✗ [${evt.phase}] ${evt.message}`);
      break;
    case "done":
      statusEl.textContent = evt.cancelled ? "cancelled" : "done";
      setChatPaused(false);
      document.querySelector("#build-cancel").hidden = true;
      buildState.es?.close();
      break;
  }
}

// Reuse the confirm-card visual pattern (cf. buildOpCard). Approve / Deny / Edit.
function presentShellApproval(buildId, callId, command, cwd, log) {
  const card = document.createElement("div");
  card.className = "op-card";  // reuse existing confirm-card styling
  const cmd = document.createElement("textarea");
  cmd.value = command; cmd.rows = 2; cmd.className = "op-card-edit";
  const meta = document.createElement("div");
  meta.className = "op-card-meta"; meta.textContent = `run_shell · ${cwd}`;
  const approve = document.createElement("button");
  approve.textContent = "Approve";
  approve.onclick = () => resolveShell(buildId, callId, "approve",
    cmd.value !== command ? cmd.value : null, card);
  const deny = document.createElement("button");
  deny.textContent = "Deny";
  deny.onclick = () => resolveShell(buildId, callId, "deny", null, card);
  card.append(meta, cmd, approve, deny);
  log.append(card);
}

async function resolveShell(buildId, callId, decision, command, card) {
  card.querySelectorAll("button").forEach((b) => (b.disabled = true));
  await fetch(`/api/build/approve/${encodeURIComponent(buildId)}/${encodeURIComponent(callId)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ decision, ...(command ? { command } : {}) }),
  });
  card.classList.add("resolved");
}

function showPreview(url) {
  const f = document.querySelector("#build-preview");
  f.src = url; f.hidden = false;
}

function appendScreenshot(log, buildId, name, cycle) {
  const img = document.createElement("img");
  img.className = "build-shot"; img.alt = `cycle ${cycle}`;
  img.src = `/api/build/screenshot/${encodeURIComponent(buildId)}/${encodeURIComponent(name)}`;
  log.append(img);
}

function logLine(log, text) {
  const div = document.createElement("div");
  div.className = "build-log-line"; div.textContent = text;
  log.append(div); log.scrollTop = log.scrollHeight;
}

function appendInline(log, delta) {
  let last = log.querySelector(".build-log-line:last-child.streaming");
  if (!last) {
    last = document.createElement("div");
    last.className = "build-log-line streaming"; log.append(last);
  }
  last.textContent += delta; log.scrollTop = log.scrollHeight;
}

// Pause chat input while the GPU is building (disable the composer send button).
function setChatPaused(paused) {
  const sendBtn = document.querySelector("#send") || document.querySelector("button[type=submit]");
  if (sendBtn) sendBtn.disabled = paused;
}

document.querySelector("#build-form")?.addEventListener("submit", (e) => { e.preventDefault(); startBuild(); });
document.querySelector("#build-cancel")?.addEventListener("click", async () => {
  if (buildState.id) await fetch(`/api/build/cancel/${encodeURIComponent(buildState.id)}`, { method: "POST" });
});
```

> Selector names (`#send`, `.op-card`, panel/tab toggles) are indicative — match the actual IDs/classes in `index.html`/`style.css` discovered in Task 4. The tab-switch wiring (show `#build-panel`, hide chat) follows whatever tab mechanism the PWA already uses; if none exists, a minimal two-button toggle that flips `hidden` on the chat vs build panels is sufficient.

- [ ] **Step 2: Manual smoke (gated — requires the daemon running on driveThree)**

Start the daemon (`local-code serve`), open the PWA on driveThree, switch to Build, enter "a hello world vite react page", Start. Expect: phase lines stream, a `pnpm install` approval card appears, approving runs it, and on success the preview iframe loads `127.0.0.1:5183`. Deny a command and confirm the build reacts (the model gets the refusal).

- [ ] **Step 3: Commit**

```bash
git add frontend/app.js
git commit -m "feat(build): Build tab controller — SSE, shell approval card, live preview"
```

---

### Task 6: Disabled-state + tab affordance when daemon offline

**Files:**
- Modify: `frontend/app.js` (bootstrap), `backend/app.py` (`/api/config` exposes buildd availability)

- [ ] **Step 1: Expose availability**

In the existing `/api/config` handler (around line 182), add `buildd` availability to the returned dict:

```python
    cfg["buildd_available"] = buildd_bridge.available()
```

- [ ] **Step 2: Gate the tab in bootstrap**

Where the PWA reads `/api/config` (it already sets `autoRouter` from it), read `buildd_available` and, if false, disable the Build tab button with a tooltip "build engine offline — run `local-code serve`".

```javascript
  if (cfg.buildd_available === false) {
    const tabBtn = document.querySelector("#build-tab-btn");
    if (tabBtn) { tabBtn.disabled = true; tabBtn.title = "build engine offline — run `local-code serve`"; }
  }
```

- [ ] **Step 3: Commit**

```bash
git add backend/app.py frontend/app.js
git commit -m "feat(build): disable Build tab when daemon offline"
```

---

## Self-review (completed)

- **Spec coverage:** Build tab in PWA (Tasks 4–5) ✓; shell-only gate via confirm card (Task 5 `presentShellApproval` + daemon's gate) ✓; localhost-only, no auth (Task 3 `_require_localhost`) ✓; engine-owned workspace + embedded `:5183` preview (Task 5 `showPreview` iframe) ✓; minimal VRAM handshake — release chat model before forwarding start, "GPU busy" + chat pause (Tasks 3 `_release_chat_model`, 5 `setChatPaused`) ✓; graceful degrade when daemon offline (Tasks 1 `available`, 6) ✓; screenshots streamed via proxy with traversal guard (Tasks 1, 3) ✓; 409 on concurrent build surfaced (Tasks 1, 3, 5) ✓.
- **Placeholder scan:** indicative selector/class names (`#send`, `.op-card`, tab toggles) are explicitly flagged to be matched against the real `index.html`/`style.css` in Task 4 — not silent TODOs. All Python is concrete and tested.
- **Contract consistency:** route shapes, the SSE `data: <json>` framing, the `approval_required`/`approval_resolved`/`preview_ready`/`screenshot` event names, and the `{buildId}` / `409 build_in_progress` responses all match Plan 1's daemon contract (Tasks 3, 6–7 there). One contract follow-up surfaced: the daemon needs `GET /healthz` (Task 1 note) — add it in Plan 1 if absent.
- **No-duplicate-import rule honoured:** Task 3 reuses `re`/`asyncio`/`httpx`/`json`/`StreamingResponse`/`HTTPException` already imported; only `Request` and `Response` may need adding to existing fastapi imports.

## Open contract follow-up for Plan 1

- Add `GET /healthz` → `200 {"ok":true}` to the daemon's loopback server (used by `buildd.available()`), bound by the same loopback guard. Cheap; fold into Plan 1 Task 7 if re-touched, otherwise a one-line addition.
```

