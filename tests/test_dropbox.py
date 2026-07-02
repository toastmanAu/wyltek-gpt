"""Drop-folder list + import tests.

The drop folder is a host directory the user fills outside the browser. These
tests pin: (1) flat listing that excludes subdirs/dotfiles and sorts newest
first; (2) copy-into-workspace import with explicit skipped reasons and no
path-traversal escape.
"""

import os

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _point_dropbox(tmp_path, monkeypatch):
    drop = tmp_path / "drop"
    drop.mkdir()
    monkeypatch.setattr(app_module, "DROPBOX", drop.resolve())
    return drop


# ── GET /api/dropbox ─────────────────────────────────────────────────


def test_list_returns_files_newest_first(tmp_path, monkeypatch):
    drop = _point_dropbox(tmp_path, monkeypatch)
    old = drop / "old.txt"
    old.write_text("a")
    new = drop / "new.txt"
    new.write_text("bb")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    r = client.get("/api/dropbox")
    assert r.status_code == 200
    files = r.json()["files"]
    assert [f["name"] for f in files] == ["new.txt", "old.txt"]
    assert files[1]["size"] == 1


def test_list_excludes_subdirs_and_dotfiles(tmp_path, monkeypatch):
    drop = _point_dropbox(tmp_path, monkeypatch)
    (drop / "keep.txt").write_text("x")
    (drop / ".hidden").write_text("x")
    (drop / "sub").mkdir()

    names = [f["name"] for f in client.get("/api/dropbox").json()["files"]]
    assert names == ["keep.txt"]


def test_list_missing_folder_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DROPBOX", (tmp_path / "nope").resolve())
    r = client.get("/api/dropbox")
    assert r.status_code == 200
    assert r.json() == {"files": []}


# ── POST /api/dropbox/import ─────────────────────────────────────────


def _point_both(tmp_path, monkeypatch):
    drop = tmp_path / "drop"
    drop.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(app_module, "DROPBOX", drop.resolve())
    monkeypatch.setattr(app_module, "WORKSPACE", ws.resolve())
    return drop, ws


def test_import_copies_multiple_into_session_workspace(tmp_path, monkeypatch):
    drop, ws = _point_both(tmp_path, monkeypatch)
    (drop / "a.txt").write_text("AAA")
    (drop / "b.txt").write_text("BB")

    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["a.txt", "b.txt"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert {f["name"] for f in body["imported"]} == {"a.txt", "b.txt"}
    assert body["skipped"] == []
    assert (ws / "s1" / "a.txt").read_text() == "AAA"
    assert (ws / "s1" / "b.txt").read_text() == "BB"


def test_import_traversal_name_is_skipped_not_written(tmp_path, monkeypatch):
    drop, ws = _point_both(tmp_path, monkeypatch)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP")

    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["../secret.txt"]},
    )
    assert r.status_code == 200
    # basename "secret.txt" does not exist in the drop folder -> skipped
    assert r.json()["imported"] == []
    assert r.json()["skipped"][0]["name"] == "../secret.txt"
    assert not (ws / "s1" / "secret.txt").exists()


def test_import_nonexistent_name_is_skipped_not_500(tmp_path, monkeypatch):
    _point_both(tmp_path, monkeypatch)
    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["ghost.txt"]},
    )
    assert r.status_code == 200
    assert r.json()["imported"] == []
    assert r.json()["skipped"] == [{"name": "ghost.txt", "reason": "not found"}]


def test_import_collision_overwrites(tmp_path, monkeypatch):
    drop, ws = _point_both(tmp_path, monkeypatch)
    (drop / "dup.txt").write_text("NEW")
    sess = ws / "s1"
    sess.mkdir()
    (sess / "dup.txt").write_text("OLD")

    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["dup.txt"]},
    )
    assert r.status_code == 200
    assert (sess / "dup.txt").read_text() == "NEW"
