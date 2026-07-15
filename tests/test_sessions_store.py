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
