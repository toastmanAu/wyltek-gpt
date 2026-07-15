from pathlib import Path
import pytest
from backend.sessions import SessionStore, SessionStoreError, SUMMARY_ROLE


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


def test_load_context_unknown_session_raises(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(SessionStoreError):
        s.load_context("nope")


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
