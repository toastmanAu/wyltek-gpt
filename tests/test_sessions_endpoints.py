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
