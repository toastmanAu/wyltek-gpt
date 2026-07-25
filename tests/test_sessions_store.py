import sqlite3
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


def test_get_full_returns_per_message_model(tmp_path):
    s = _store(tmp_path)
    sid = s.create_session(model="qwen3-coder:30b")
    s.append_message(sid, "user", "hello")
    s.append_message(sid, "assistant", "hi", tokens=3, model="qwen3-coder:30b")
    s.append_message(sid, "assistant", "yo", model="gemma4:26b")  # mid-chat model switch

    msgs = s.get_full(sid)["messages"]
    assert [(m["role"], m["model"]) for m in msgs] == [
        ("user", None),                    # user turns carry no model
        ("assistant", "qwen3-coder:30b"),
        ("assistant", "gemma4:26b"),       # reflects the switch, not the session's first model
    ]


def test_load_context_ignores_model(tmp_path):
    # model is a display concern; it must never leak into the LLM context.
    s = _store(tmp_path)
    sid = s.create_session(model="m")
    s.append_message(sid, "assistant", "hi", model="m")
    assert s.load_context(sid) == [{"role": "assistant", "content": "hi"}]


def test_migrates_legacy_db_without_model_column(tmp_path):
    # An old DB whose messages table predates the model column must gain it on
    # open, with existing rows reading back as model=None.
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY, title TEXT, summary TEXT,
          summary_upto_seq INTEGER NOT NULL DEFAULT 0, model TEXT,
          created REAL NOT NULL, updated REAL NOT NULL);
        CREATE TABLE messages (
          session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL,
          content TEXT NOT NULL, tokens INTEGER, created REAL NOT NULL,
          PRIMARY KEY (session_id, seq));
        INSERT INTO sessions VALUES ('s1','t',NULL,0,'m',1.0,1.0);
        INSERT INTO messages VALUES ('s1',1,'assistant','old reply',7,1.0);
        """
    )
    conn.commit()
    conn.close()

    s = SessionStore(db)
    msgs = s.get_full("s1")["messages"]
    assert msgs[0]["model"] is None
    # and new writes on the migrated DB carry the model through
    s.append_message("s1", "assistant", "new reply", model="m2")
    assert s.get_full("s1")["messages"][1]["model"] == "m2"


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
