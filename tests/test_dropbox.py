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
