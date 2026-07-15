# SQLite Server-Side Sessions (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give wyltek-gpt server-side chat sessions in SQLite as the source of truth, with resume, so history stops living only in the browser and there is an authoritative place for a future compaction summary.

**Architecture:** A new `backend/sessions.py` `SessionStore` (single SQLite connection guarded by a lock, WAL mode) owns a normalized `sessions` + `messages` schema with a `summary_upto_seq` watermark. `app.py` gains session CRUD endpoints and a dual-mode `/api/chat`: **session mode** (a `session_id` is present) persists each turn and streams; **legacy mode** (no `session_id`) is exactly today's stateless behaviour. The frontend gets a sessions sidebar. Compaction (Phase 2) is a separate later plan; this schema and `load_context()` already accommodate it.

**Tech Stack:** Python 3.11, stdlib `sqlite3` (no new dependency), FastAPI, pytest + `fastapi.testclient.TestClient`, vanilla JS frontend.

**Spec:** `docs/superpowers/specs/2026-07-15-sqlite-sessions-design.md`

## Global Constraints

- No new Python dependencies — use stdlib `sqlite3`. (One line, verbatim requirement.)
- Single uvicorn worker assumption holds (same as `CapabilityCache`/`DemoStore`); use one connection + a `threading.Lock`.
- SQLite opened WAL mode, `busy_timeout=3000`.
- Persistence must never break chat: every `SessionStore` call from `app.py` is wrapped; failure logs and degrades to legacy behaviour.
- Only final visible assistant text is persisted — no thinking channel, no `__demo_step__`/`__cellc_step__`/`__stats__` sentinels.
- Tests: pytest, files under `tests/test_*.py`, `TestClient(app_module.app)`, `tmp_path`/`monkeypatch` fixtures (follow `tests/test_chat_text_files.py`). Target 80%+ coverage of new code.
- Commit messages: conventional (`feat:`/`test:`/`fix:`), no attribution footer (global setting).
- DB file lives at `data/sessions.db`; add it to `.gitignore`.

---

## File Structure

- Create `backend/sessions.py` — `SessionStore` class + `SessionStoreError`. Pure persistence, no HTTP.
- Create `tests/test_sessions_store.py` — `SessionStore` unit tests (real temp-file DB via `tmp_path`).
- Create `tests/test_sessions_endpoints.py` — CRUD endpoint integration tests.
- Create `tests/test_chat_sessions.py` — `/api/chat` dual-mode tests.
- Modify `backend/app.py` — init the store; add `/api/sessions*` endpoints; wire `/api/chat` dual-mode; add auto-title helper.
- Modify `config.yaml` — add `auto_router.summary_model`.
- Modify `.gitignore` — ignore `data/sessions.db*`.
- Modify `frontend/index.html`, `frontend/style.css`, `frontend/app.js` — sessions sidebar + session-mode send.

---

## Task 1: SessionStore core — schema, create, append, load_context (no-summary path)

**Files:**
- Create: `backend/sessions.py`
- Test: `tests/test_sessions_store.py`

**Interfaces:**
- Produces:
  - `class SessionStoreError(Exception)`
  - `class SessionStore:`
    - `__init__(self, db_path: pathlib.Path)`
    - `create_session(self, model: str, title: str | None = None) -> str` (returns a new `session_id`)
    - `append_message(self, session_id: str, role: str, content: str, tokens: int | None = None) -> int` (returns the new `seq`; raises `SessionStoreError` if session unknown)
    - `load_context(self, session_id: str) -> list[dict]` (list of `{"role","content"}`, ordered by `seq`, excluding the app system prompt; with no summary returns the full transcript)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sessions_store.py
from pathlib import Path
import pytest
from backend.sessions import SessionStore, SessionStoreError


def _store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def test_create_append_and_load_context_full_transcript(tmp_path):
    s = _store(tmp_path)
    sid = s.create_session(model="qwen3-coder:30b")
    assert isinstance(sid, str) and sid
    seq1 = s.append_message(sid, "user", "hello")
    seq2 = s.append_message(sid, "assistant", "hi there", tokens=5)
    assert (seq1, seq2) == (1, 2)
    ctx = s.load_context(sid)
    assert ctx == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_append_to_unknown_session_raises(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(SessionStoreError):
        s.append_message("nope", "user", "x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sessions_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.sessions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/sessions.py
"""Server-side chat sessions in SQLite — the source of truth for history.

Single-connection + lock (single uvicorn worker, same assumption as
CapabilityCache/DemoStore). WAL mode so the rare overlapping write on the
worker doesn't error. The summary/summary_upto_seq columns exist for Phase 2
compaction; Phase 1 never writes them, so load_context returns the full
transcript.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id               TEXT PRIMARY KEY,
  title            TEXT,
  summary          TEXT,
  summary_upto_seq INTEGER NOT NULL DEFAULT 0,
  model            TEXT,
  created          REAL NOT NULL,
  updated          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  session_id TEXT NOT NULL,
  seq        INTEGER NOT NULL,
  role       TEXT NOT NULL,
  content    TEXT NOT NULL,
  tokens     INTEGER,
  created    REAL NOT NULL,
  PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
"""

# Role used for the injected compaction summary (Phase 2). Defined here so the
# store owns the convention; app.py prepends the real system prompt separately.
SUMMARY_ROLE = "system"


class SessionStoreError(Exception):
    pass


class SessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=3000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_session(self, model: str, title: str | None = None) -> str:
        sid = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, title, summary, summary_upto_seq, model, created, updated) "
                "VALUES (?, ?, NULL, 0, ?, ?, ?)",
                (sid, title, model, now, now),
            )
            self._conn.commit()
        return sid

    def append_message(self, session_id: str, role: str, content: str,
                       tokens: int | None = None) -> int:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise SessionStoreError(f"unknown session_id {session_id!r}")
            seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages WHERE session_id = ?",
                (session_id,)).fetchone()
            seq = int(seq_row["next"])
            self._conn.execute(
                "INSERT INTO messages (session_id, seq, role, content, tokens, created) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, seq, role, content, tokens, now))
            self._conn.execute(
                "UPDATE sessions SET updated = ? WHERE id = ?", (now, session_id))
            self._conn.commit()
        return seq

    def load_context(self, session_id: str) -> list[dict]:
        with self._lock:
            srow = self._conn.execute(
                "SELECT summary, summary_upto_seq FROM sessions WHERE id = ?",
                (session_id,)).fetchone()
            if srow is None:
                raise SessionStoreError(f"unknown session_id {session_id!r}")
            upto = int(srow["summary_upto_seq"])
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? AND seq > ? ORDER BY seq",
                (session_id, upto)).fetchall()
        out: list[dict] = []
        if srow["summary"]:
            out.append({"role": SUMMARY_ROLE,
                        "content": f"[Summary of earlier conversation]\n{srow['summary']}"})
        out.extend({"role": r["role"], "content": r["content"]} for r in rows)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sessions_store.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/sessions.py tests/test_sessions_store.py
git commit -m "feat: SessionStore core — schema, create, append, load_context"
```

---

## Task 2: SessionStore — list, get_full, rename, delete

**Files:**
- Modify: `backend/sessions.py`
- Test: `tests/test_sessions_store.py`

**Interfaces:**
- Produces:
  - `list_sessions(self) -> list[dict]` — `[{"id","title","updated","model"}]`, newest `updated` first
  - `get_full(self, session_id: str) -> dict | None` — `{"title","summary","messages":[{"seq","role","content"}]}` (entire transcript incl. pre-watermark rows), or `None` if unknown
  - `rename(self, session_id: str, title: str) -> None`
  - `delete(self, session_id: str) -> None` (also removes the session's messages)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sessions_store.py  (append)
def test_list_get_full_rename_delete(tmp_path):
    s = _store(tmp_path)
    a = s.create_session(model="m1", title="first")
    b = s.create_session(model="m2", title="second")
    s.append_message(b, "user", "q")          # bump b.updated so it sorts first
    listed = s.list_sessions()
    assert [x["id"] for x in listed] == [b, a]
    assert listed[0]["title"] == "second" and listed[0]["model"] == "m2"

    full = s.get_full(a)
    assert full == {"title": "first", "summary": None, "messages": []}
    assert s.get_full("nope") is None

    s.rename(a, "renamed")
    assert s.get_full(a)["title"] == "renamed"

    s.delete(b)
    assert s.get_full(b) is None
    assert [x["id"] for x in s.list_sessions()] == [a]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sessions_store.py::test_list_get_full_rename_delete -v`
Expected: FAIL with `AttributeError: 'SessionStore' object has no attribute 'list_sessions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/sessions.py  (add methods to SessionStore)
    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, updated, model FROM sessions ORDER BY updated DESC"
            ).fetchall()
        return [{"id": r["id"], "title": r["title"], "updated": r["updated"],
                 "model": r["model"]} for r in rows]

    def get_full(self, session_id: str) -> dict | None:
        with self._lock:
            srow = self._conn.execute(
                "SELECT title, summary FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if srow is None:
                return None
            rows = self._conn.execute(
                "SELECT seq, role, content FROM messages WHERE session_id = ? ORDER BY seq",
                (session_id,)).fetchall()
        return {"title": srow["title"], "summary": srow["summary"],
                "messages": [{"seq": r["seq"], "role": r["role"], "content": r["content"]}
                             for r in rows]}

    def rename(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET title = ?, updated = ? WHERE id = ?",
                               (title, time.time(), session_id))
            self._conn.commit()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sessions_store.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/sessions.py tests/test_sessions_store.py
git commit -m "feat: SessionStore list/get_full/rename/delete"
```

---

## Task 3: SessionStore — set_summary + watermark replay (Phase 2 hook)

**Files:**
- Modify: `backend/sessions.py`
- Test: `tests/test_sessions_store.py`

**Interfaces:**
- Produces: `set_summary(self, session_id: str, summary: str, upto_seq: int) -> None` — sets `summary` + `summary_upto_seq`; after this, `load_context` returns `[summary message] + messages where seq > upto_seq`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sessions_store.py  (append)
from backend.sessions import SUMMARY_ROLE


def test_set_summary_watermark_replay(tmp_path):
    s = _store(tmp_path)
    sid = s.create_session(model="m")
    for i in range(1, 11):                      # seqs 1..10
        s.append_message(sid, "user" if i % 2 else "assistant", f"msg{i}")
    s.set_summary(sid, "we discussed msgs 1-8", upto_seq=8)

    ctx = s.load_context(sid)
    assert ctx[0] == {"role": SUMMARY_ROLE,
                      "content": "[Summary of earlier conversation]\nwe discussed msgs 1-8"}
    assert [m["content"] for m in ctx[1:]] == ["msg9", "msg10"]   # only seq > 8

    # full transcript still preserves the pre-watermark rows
    assert len(s.get_full(sid)["messages"]) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sessions_store.py::test_set_summary_watermark_replay -v`
Expected: FAIL with `AttributeError: ... no attribute 'set_summary'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/sessions.py  (add method to SessionStore)
    def set_summary(self, session_id: str, summary: str, upto_seq: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET summary = ?, summary_upto_seq = ?, updated = ? WHERE id = ?",
                (summary, int(upto_seq), time.time(), session_id))
            self._conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sessions_store.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/sessions.py tests/test_sessions_store.py
git commit -m "feat: SessionStore set_summary + watermark replay (Phase 2 hook)"
```

---

## Task 4: Init the store + session CRUD endpoints

**Files:**
- Modify: `backend/app.py` (init near the `html_demo` init block ~line 73–80; endpoints after the `/api/capabilities` block)
- Modify: `.gitignore`
- Test: `tests/test_sessions_endpoints.py`

**Interfaces:**
- Consumes: `SessionStore` (Task 1–3).
- Produces (module-level in `app.py`): `SESSIONS: SessionStore | None`. Endpoints:
  - `POST /api/sessions` body `{model, title?}` -> `{"id": sid}`
  - `GET /api/sessions` -> `[{id,title,updated,model}]`
  - `GET /api/sessions/{sid}` -> `{title,summary,messages[]}` or 404
  - `PATCH /api/sessions/{sid}` body `{title}` -> `{"ok": true}` or 404
  - `DELETE /api/sessions/{sid}` -> `{"ok": true}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sessions_endpoints.py
from fastapi.testclient import TestClient
from backend import app as app_module

client = TestClient(app_module.app)


def test_session_crud_roundtrip():
    r = client.post("/api/sessions", json={"model": "qwen3-coder:30b", "title": "hi"})
    assert r.status_code == 200
    sid = r.json()["id"]

    assert any(s["id"] == sid for s in client.get("/api/sessions").json())

    got = client.get(f"/api/sessions/{sid}")
    assert got.status_code == 200
    assert got.json()["title"] == "hi" and got.json()["messages"] == []

    assert client.patch(f"/api/sessions/{sid}", json={"title": "renamed"}).status_code == 200
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "renamed"

    assert client.get("/api/sessions/does-not-exist").status_code == 404

    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sessions_endpoints.py -v`
Expected: FAIL (404s / `id` KeyError — endpoints don't exist yet).

- [ ] **Step 3a: Init the store in `backend/app.py`**

Add after the `html_demo` init block (near line 80), following that block's try/except idiom:

```python
# ─── Sessions store (server-side history) ────────────────────────────
from backend.sessions import SessionStore, SessionStoreError  # noqa: E402  (top with other imports is fine too)

SESSIONS: SessionStore | None = None
try:
    SESSIONS = SessionStore(ROOT / "data" / "sessions.db")
    log.info("sessions store: %s", ROOT / "data" / "sessions.db")
except Exception as exc:  # never block boot on the feature
    log.warning("sessions store init failed: %s", exc)
    SESSIONS = None
```

(Move the `from backend.sessions import ...` line up beside the other `backend.` imports at the top of the file to match style; the inline comment just marks intent.)

- [ ] **Step 3b: Add the CRUD endpoints** after the `/api/capabilities` block:

```python
# ─── Sessions API ────────────────────────────────────────────────────

@app.post("/api/sessions")
async def create_session(payload: dict):
    if SESSIONS is None:
        raise HTTPException(503, "sessions store unavailable")
    model = payload.get("model") or ""
    sid = SESSIONS.create_session(model=model, title=payload.get("title"))
    return {"id": sid}


@app.get("/api/sessions")
async def list_sessions():
    if SESSIONS is None:
        return []
    return SESSIONS.list_sessions()


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    if SESSIONS is None:
        raise HTTPException(503, "sessions store unavailable")
    full = SESSIONS.get_full(sid)
    if full is None:
        raise HTTPException(404, "session not found")
    return full


@app.patch("/api/sessions/{sid}")
async def rename_session(sid: str, payload: dict):
    if SESSIONS is None or SESSIONS.get_full(sid) is None:
        raise HTTPException(404, "session not found")
    SESSIONS.rename(sid, payload.get("title") or "")
    return {"ok": True}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    if SESSIONS is not None:
        SESSIONS.delete(sid)
    return {"ok": True}
```

- [ ] **Step 3c: Ignore the DB file** — add to `.gitignore`:

```
data/sessions.db
data/sessions.db-wal
data/sessions.db-shm
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sessions_endpoints.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app.py tests/test_sessions_endpoints.py .gitignore
git commit -m "feat: session CRUD endpoints + store init"
```

---

## Task 5: `/api/chat` dual-mode — session persistence + streaming

**Files:**
- Modify: `backend/app.py` (`chat()` ~line 413–432 and the `stream()`/`relay()` region ~496–610)
- Test: `tests/test_chat_sessions.py`

**Interfaces:**
- Consumes: `SESSIONS` (Task 4), the existing `relay()`/`_stream_one()` streaming.
- Produces: `chat()` accepts optional `payload["session_id"]`. In session mode it persists the user turn before streaming and the assistant turn after the stream completes; legacy mode is unchanged. Assistant text is accumulated from `chunk`-kind events only.

**Design notes for the implementer:**
- `relay()` yields already-serialized wire strings. To capture only the *visible assistant text*, accumulate at the point where a `chunk` is emitted — i.e. add an accumulator the `stream()` closure can see, and append `value` in the `elif kind == "chunk":` branch of `relay()` (before it is yielded). Do **not** accumulate `thinking`, tool-call JSON, or `__stats__`.
- Capture `eval_count` from the `stats` branch (`value.get("eval_count")`) into the same closure so it can be stored with the assistant message.
- Persist the assistant message in the `stream()` generator *after* the `async for` loop finishes (all wire yielded), guarded by try/except so a store failure never breaks the response.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_sessions.py
import json
from fastapi.testclient import TestClient
from backend import app as app_module

client = TestClient(app_module.app)


class _FakeStreamOllama:
    """Monkeypatch target: yield a thinking bit, a text chunk, then stats+done."""
    @staticmethod
    async def _gen(*_a, **_k):
        yield ("thinking", "hmm")
        yield ("chunk", "Hello ")
        yield ("chunk", "world")
        yield ("stats", {"prompt_eval_count": 12, "eval_count": 7})
        yield ("done", None)


def _patch_stream(monkeypatch):
    monkeypatch.setattr(app_module, "_stream_one",
                        lambda body: _FakeStreamOllama._gen())


def test_session_mode_persists_user_and_assistant(monkeypatch):
    _patch_stream(monkeypatch)
    sid = client.post("/api/sessions", json={"model": "m"}).json()["id"]

    r = client.post("/api/chat", json={
        "model": "m", "chat_session_id": sid,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.text
    assert "Hello world" in body                      # streamed as before

    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    roles = [(m["role"], m["content"]) for m in msgs]
    assert roles == [("user", "hi"), ("assistant", "Hello world")]  # NOT "hmm"
    assert msgs[1]["role"] == "assistant"


def test_legacy_mode_persists_nothing(monkeypatch):
    _patch_stream(monkeypatch)
    before = len(client.get("/api/sessions").json())
    r = client.post("/api/chat", json={
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}]})   # no chat_session_id
    assert r.status_code == 200
    assert "Hello world" in r.text
    assert len(client.get("/api/sessions").json()) == before  # no new session/rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat_sessions.py -v`
Expected: FAIL — `test_session_mode_persists_user_and_assistant` fails because the assistant turn isn't persisted (only `user` present, or messages empty).

- [ ] **Step 3a: Persist the user turn + build context in `chat()`**

In `chat()`, after the existing `user_and_assistant`/`messages` assembly (~line 431–432), add session-mode context loading. **Use a distinct payload key `chat_session_id`** for the sessions store — the existing `session_id` key is the *workspace* directory (file uploads) and must not be overloaded:

```python
    sess_id = payload.get("chat_session_id")   # sessions store key; NOT the workspace session_id
    if SESSIONS is not None and sess_id:
        last_user = next((m.get("content", "") for m in reversed(incoming)
                          if m.get("role") == "user"), "")
        try:
            SESSIONS.append_message(sess_id, "user", last_user)
            history = SESSIONS.load_context(sess_id)     # [summary?] + prior turns incl. this user msg
            messages = [{"role": "system", "content": _full_system_prompt()}, *history]
        except SessionStoreError:
            sess_id = None   # unknown session -> fall back to legacy assembly below
```

(Keep the existing `messages = [...]` legacy assembly as the else/default path. The injection calls — html_demo/cellc/text_files/images — run on `messages` unchanged, after this block.)

- [ ] **Step 3b: Accumulate assistant text + eval_count**

At the top of the `stream()` closure (~line 496), add:

```python
        assistant_parts: list[str] = []
        final_eval = {"tokens": None}
```

In `relay()`, in the `elif kind == "chunk":` branch, capture before yielding:

```python
                elif kind == "chunk":
                    if in_thinking:
                        in_thinking = False
                        yield THINK_CLOSE
                    assistant_parts.append(value)   # <-- add
                    yield value
```

In the `elif kind == "stats":` branch, capture the count:

```python
                elif kind == "stats":
                    ...
                    final_eval["tokens"] = value.get("eval_count")   # <-- add
                    yield "\n" + json.dumps({"__stats__": value})
```

- [ ] **Step 3c: Persist the assistant turn after streaming**

At the end of `stream()`, after the `async for wire in relay(...)` loop (~line 608), add:

```python
        if SESSIONS is not None and sess_id and assistant_parts:
            try:
                SESSIONS.append_message(sess_id, "assistant", "".join(assistant_parts),
                                        tokens=final_eval["tokens"])
            except SessionStoreError as exc:
                log.warning("session %s: assistant persist failed: %s", sess_id, exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_chat_sessions.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full backend suite (no regressions)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (existing chat/text-file/html_demo tests unaffected — legacy mode unchanged).

- [ ] **Step 6: Commit**

```bash
git add backend/app.py tests/test_chat_sessions.py
git commit -m "feat: dual-mode /api/chat — persist session turns, legacy unchanged"
```

---

## Task 6: Config-gated auto-title

**Files:**
- Modify: `config.yaml` (`auto_router` block ~line 56–57)
- Modify: `backend/app.py` (auto-title helper + call in the session-mode assistant-persist block from Task 5)
- Test: `tests/test_sessions_endpoints.py` (append)

**Interfaces:**
- Consumes: `SESSIONS`, `CONFIG["auto_router"]`.
- Produces: `_auto_title(sess_id, first_user, first_assistant) -> None` — sets a title on a session that still has none, using `auto_router.summary_model` (fallback `captioner_model`, else truncated `first_user`). Never raises.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sessions_endpoints.py  (append)
def test_auto_title_fallback_uses_truncated_first_message(monkeypatch):
    # No summary_model / captioner configured -> fallback path (no model call).
    monkeypatch.setitem(app_module.CONFIG, "auto_router", {})
    sid = app_module.SESSIONS.create_session(model="m")   # title None
    app_module._auto_title(sid, "Explain how WebGPU pipelines work in detail", "…")
    title = app_module.SESSIONS.get_full(sid)["title"]
    assert title and len(title) <= 60
    assert title.startswith("Explain how WebGPU")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sessions_endpoints.py::test_auto_title_fallback_uses_truncated_first_message -v`
Expected: FAIL with `AttributeError: ... no attribute '_auto_title'`.

- [ ] **Step 3a: Add the config knob** — `config.yaml`:

```yaml
auto_router:
  captioner_model: qwen3-omni-captioner:30b-a3b-q4
  summary_model: ""        # titles (Phase 1) + compaction (Phase 2); empty disables the model call
```

- [ ] **Step 3b: Add the helper** in `backend/app.py` (near other `auto_router` usage ~line 247):

```python
def _auto_title(sess_id: str, first_user: str, first_assistant: str) -> None:
    """Give a still-untitled session a short title. Uses auto_router.summary_model
    (fallback captioner_model, else a truncated first message). Never raises."""
    if SESSIONS is None:
        return
    try:
        full = SESSIONS.get_full(sess_id)
        if full is None or full.get("title"):
            return   # already titled or gone
        auto = CONFIG.get("auto_router") or {}
        model = auto.get("summary_model") or auto.get("captioner_model") or ""
        title = ""
        if model:
            title = _titling_call(model, first_user, first_assistant)  # see 3c
        if not title:
            title = (first_user or "New chat").strip().splitlines()[0][:60]
        SESSIONS.rename(sess_id, title[:60])
    except Exception as exc:   # titling must never break chat
        log.warning("auto-title failed for %s: %s", sess_id, exc)
```

- [ ] **Step 3c: Add the (blocking, best-effort) model call** — a small non-streaming Ollama call, mirroring the captioner pattern. Keep it synchronous but wrapped; it runs after the response stream is already closed:

```python
def _titling_call(model: str, first_user: str, first_assistant: str) -> str:
    import httpx
    prompt = ("Give a 3-6 word title (no quotes, no punctuation at the end) for this chat:\n\n"
              f"User: {first_user[:500]}\nAssistant: {first_assistant[:500]}\nTitle:")
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate",
                       json={"model": model, "prompt": prompt, "stream": False,
                             "options": {"num_predict": 24, "temperature": 0.3}},
                       timeout=30)
        r.raise_for_status()
        return (r.json().get("response") or "").strip().strip('"')[:60]
    except Exception:
        return ""
```

- [ ] **Step 3d: Call it** in the Task-5 assistant-persist block (only on the first assistant turn):

```python
        if SESSIONS is not None and sess_id and assistant_parts:
            try:
                SESSIONS.append_message(sess_id, "assistant", "".join(assistant_parts),
                                        tokens=final_eval["tokens"])
                full = SESSIONS.get_full(sess_id)
                if full and not full.get("title") and len(full["messages"]) <= 2:
                    _auto_title(sess_id, incoming and next(
                        (m.get("content", "") for m in incoming if m.get("role") == "user"), ""),
                        "".join(assistant_parts))
            except SessionStoreError as exc:
                log.warning("session %s: assistant persist failed: %s", sess_id, exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sessions_endpoints.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.yaml backend/app.py tests/test_sessions_endpoints.py
git commit -m "feat: config-gated auto-title for sessions"
```

---

## Task 7: Frontend sessions sidebar + session-mode send

**Files:**
- Modify: `frontend/index.html` (add a sidebar container)
- Modify: `frontend/style.css` (sidebar styles)
- Modify: `frontend/app.js` (session state, sidebar render, session-mode chat send)

**Interfaces:**
- Consumes: `/api/sessions*` (Task 4) and dual-mode `/api/chat` (Task 5).
- Produces: a `currentSessionId` client state; chat send posts `session_id` instead of the full `messages` array.

**Note on testing:** the frontend has no JS unit harness; verification is a Playwright smoke check (matching how the dropbox UI was verified). Steps are still checkboxed.

- [ ] **Step 1: Add sidebar markup** — in `frontend/index.html`, add a sidebar element beside the chat log (place adjacent to the existing chat container):

```html
<aside id="session-sidebar">
  <button id="new-session">+ New chat</button>
  <ul id="session-list"></ul>
</aside>
```

- [ ] **Step 2: Style it** — in `frontend/style.css`, add minimal styles (follow existing panel styling variables):

```css
#session-sidebar { width: 240px; border-right: 1px solid var(--border, #222); overflow-y: auto; }
#session-list { list-style: none; margin: 0; padding: 0; }
#session-list li { padding: 8px 12px; cursor: pointer; display: flex; justify-content: space-between; }
#session-list li.active { background: rgba(255,255,255,.06); }
#session-list li .del { opacity: .4; }
#session-list li .del:hover { opacity: 1; }
#new-session { width: 100%; padding: 10px; cursor: pointer; }
```

- [ ] **Step 3: Add session state + rendering** — in `frontend/app.js`, near the other top-level state (e.g. `SESSION`, `history`):

```javascript
let currentSessionId = null;

async function loadSessions() {
  const list = await (await fetch("/api/sessions")).json();
  const ul = document.getElementById("session-list");
  ul.innerHTML = "";
  for (const s of list) {
    const li = document.createElement("li");
    li.textContent = s.title || "Untitled";
    if (s.id === currentSessionId) li.classList.add("active");
    li.onclick = () => openSession(s.id);
    const del = document.createElement("span");
    del.className = "del"; del.textContent = "✕";
    del.onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/sessions/${s.id}`, { method: "DELETE" });
      if (s.id === currentSessionId) { currentSessionId = null; clearChatLog(); }
      loadSessions();
    };
    li.appendChild(del);
    ul.appendChild(li);
  }
}

async function openSession(id) {
  const full = await (await fetch(`/api/sessions/${id}`)).json();
  currentSessionId = id;
  clearChatLog();
  for (const m of full.messages) renderMessage(m.role, m.content);  // reuse existing render
  loadSessions();
}

async function ensureSession(model) {
  if (currentSessionId) return currentSessionId;
  const r = await fetch("/api/sessions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  currentSessionId = (await r.json()).id;
  loadSessions();
  return currentSessionId;
}

document.getElementById("new-session").onclick = () => {
  currentSessionId = null; clearChatLog(); loadSessions();
};
```

(`clearChatLog()` and `renderMessage(role, content)` are thin wrappers around the existing chat-log DOM code — if equivalents already exist under different names, reuse them instead of adding new ones.)

- [ ] **Step 4: Switch chat send to session mode** — at the existing `/api/chat` POST (~line 1252), create/reuse the session and send `session_id` + only the new user message:

```javascript
  const sid = await ensureSession(model);
  const body = {
    model,
    chat_session_id: sid,                              // sessions store key (workspace session_id unchanged)
    messages: [{ role: "user", content: userText }],   // server loads prior turns
    // keep session_id (workspace) + image_files/text_files fields as today
  };
```

- [ ] **Step 5: Initialise on load** — call `loadSessions()` in the existing startup/init path.

- [ ] **Step 6: Playwright smoke verification**

Start the app (`./.venv/bin/uvicorn backend.app:app --port 8910`), then via Playwright MCP:
1. New chat -> send a message -> assistant reply streams.
2. Reload the page -> the session appears in the sidebar with an auto-title -> open it -> the transcript renders (resume works).
3. Send a second turn -> only the new turn is posted (check the network request body has `session_id` + one message), reply still has prior context.
4. Delete the session -> it disappears; chat log clears.

Expected: all four pass with no console errors (favicon 404 excepted).

- [ ] **Step 7: Commit**

```bash
git add frontend/index.html frontend/style.css frontend/app.js
git commit -m "feat: sessions sidebar + session-mode chat send"
```

---

## Self-Review notes

- **Spec coverage:** SessionStore (T1–3) ✓; CRUD endpoints (T4) ✓; dual-mode chat with legacy fallback (T5) ✓; auto-title (T6) ✓; frontend sidebar/resume (T7) ✓; error-degradation (wrapped store calls in T4–6) ✓; watermark/Phase-2 hooks (T3, `load_context`) ✓. Dynamic num_ctx is explicitly out of scope (separate task) and only referenced by Phase 2.
- **Type consistency:** `SessionStore` method names/signatures defined in T1–3 are used verbatim in T4–6 (`create_session`, `append_message`, `load_context`, `list_sessions`, `get_full`, `rename`, `delete`, `set_summary`).
- **Payload-key separation (resolved in-plan):** the sessions store uses a **distinct** payload key `chat_session_id`; the pre-existing `session_id` key remains the *workspace* directory for file uploads and is untouched. Task 5 (`sess_id = payload.get("chat_session_id")`), its tests, and the Task 7 frontend body all use `chat_session_id` consistently, so the workspace `"default"` id can never be mistaken for a real chat session.
