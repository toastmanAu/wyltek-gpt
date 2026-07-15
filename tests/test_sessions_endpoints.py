# tests/test_sessions_endpoints.py
import asyncio

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


def test_auto_title_fallback_uses_truncated_first_message(monkeypatch):
    # No summary_model / captioner configured -> fallback path (no model call).
    monkeypatch.setitem(app_module.CONFIG, "auto_router", {})
    sid = app_module.SESSIONS.create_session(model="m")   # title None
    asyncio.run(app_module._auto_title(sid, "Explain how WebGPU pipelines work in detail", "…"))
    title = app_module.SESSIONS.get_full(sid)["title"]
    assert title and len(title) <= 60
    assert title.startswith("Explain how WebGPU")
