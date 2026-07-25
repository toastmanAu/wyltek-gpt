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
  model      TEXT,
  created    REAL NOT NULL,
  PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
"""

# Role used for the injected compaction summary (Phase 2). Defined here so the
# store owns the convention; app.py prepends the real system prompt separately.
# Phase 2 trap: load_context prepends this as history[0], and app.py also prepends
# its own system prompt — so once a summary is set, two back-to-back system messages
# reach the model. Change this role or dedupe in app.py when compaction lands.
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
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive column migrations for DBs created before a column existed.
        CREATE TABLE IF NOT EXISTS never alters an existing table, so a store
        opened on a pre-model-column DB needs the column added explicitly."""
        cols = {row["name"] for row in
                self._conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "model" not in cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN model TEXT")

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
                       tokens: int | None = None, model: str | None = None) -> int:
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
                "INSERT INTO messages (session_id, seq, role, content, tokens, model, created) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, seq, role, content, tokens, model, now))
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
                "SELECT seq, role, content, model FROM messages WHERE session_id = ? ORDER BY seq",
                (session_id,)).fetchall()
        return {"title": srow["title"], "summary": srow["summary"],
                "messages": [{"seq": r["seq"], "role": r["role"], "content": r["content"],
                              "model": r["model"]}
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

    def set_summary(self, session_id: str, summary: str, upto_seq: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET summary = ?, summary_upto_seq = ?, updated = ? WHERE id = ?",
                (summary, int(upto_seq), time.time(), session_id))
            self._conn.commit()
