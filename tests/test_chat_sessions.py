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


def test_session_mode_store_failure_degrades(monkeypatch):
    _patch_stream(monkeypatch)
    sid = client.post("/api/sessions", json={"model": "m"}).json()["id"]
    import sqlite3 as _sq
    def _boom(*a, **k):
        raise _sq.OperationalError("database is locked")
    monkeypatch.setattr(app_module.SESSIONS, "append_message", _boom)
    r = client.post("/api/chat", json={
        "model": "m", "chat_session_id": sid,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "Hello world" in r.text          # stream still completes despite store failure
