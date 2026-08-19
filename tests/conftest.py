import os
import shutil

import pytest

from backend import app as app_module
from backend.sessions import SessionStore


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_cellc: requires a built cellc binary")


@pytest.fixture
def cellc_available():
    return bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))


@pytest.fixture(autouse=True)
def _isolate_sessions_store(tmp_path, monkeypatch):
    """Point the app's SESSIONS at a throwaway per-test DB so TestClient-based
    integration tests never touch the real data/sessions.db."""
    monkeypatch.setattr(app_module, "SESSIONS", SessionStore(tmp_path / "test_sessions.db"))
    yield
