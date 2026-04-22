"""SQLite-backed persistence for Meeko sessions and turns.

The on-disk transcript is the source of truth. Every user/assistant
turn (including tool-use and tool-result rounds) is written
immediately on completion so a crash or power loss cannot drop data,
and so auto compaction can mutate the in-memory message array
without affecting the record.

Content is stored as JSON TEXT mirroring the Anthropic message shape:
  - plain string for user speech
  - list[block] for assistant turns and tool-result rounds
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    profile_name  TEXT NOT NULL,
    title         TEXT,
    tags          TEXT,
    summary       TEXT,
    created_at    TEXT NOT NULL,
    last_active   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    timestamp    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);
"""


def default_db_path() -> Path:
    override = os.environ.get("MEEKO_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".meeko" / "meeko.db"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @classmethod
    def open(cls, db_path: Path) -> SessionStore:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            conn.executescript(_SCHEMA)
        return cls(conn)

    def _create_session_sync(self, profile_name: str) -> str:
        session_id = str(uuid.uuid4())
        now = _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, profile_name, created_at, last_active) "
                "VALUES (?, ?, ?, ?)",
                (session_id, profile_name, now, now),
            )
        return session_id

    async def create_session(self, profile_name: str) -> str:
        return await asyncio.to_thread(self._create_session_sync, profile_name)

    def _persist_turn_sync(
        self, session_id: str, role: str, content: str | list[dict[str, Any]]
    ) -> None:
        now = _now()
        encoded = json.dumps(content)
        with self._conn:
            self._conn.execute(
                "INSERT INTO turns (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, encoded, now),
            )
            self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (now, session_id),
            )

    async def persist_turn(
        self, session_id: str, role: str, content: str | list[dict[str, Any]]
    ) -> None:
        await asyncio.to_thread(self._persist_turn_sync, session_id, role, content)

    def _get_latest_session_sync(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, profile_name, last_active FROM sessions "
            "ORDER BY last_active DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "profile_name": row[1], "last_active": row[2]}

    async def get_latest_session(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_latest_session_sync)

    def _get_session_sync(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, profile_name, last_active FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "profile_name": row[1], "last_active": row[2]}

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_session_sync, session_id)

    def _list_sessions_sync(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT s.id, s.profile_name, s.last_active, COUNT(t.id) "
            "FROM sessions s LEFT JOIN turns t ON t.session_id = s.id "
            "GROUP BY s.id "
            "ORDER BY s.last_active DESC"
        ).fetchall()
        return [
            {
                "id": r[0],
                "profile_name": r[1],
                "last_active": r[2],
                "turn_count": r[3],
            }
            for r in rows
        ]

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_sessions_sync)

    def _load_turns_sync(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [{"role": r[0], "content": json.loads(r[1])} for r in rows]

    async def load_turns(self, session_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._load_turns_sync, session_id)

    def _touch_session_sync(self, session_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (_now(), session_id),
            )

    async def touch_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._touch_session_sync, session_id)

    def close(self) -> None:
        self._conn.close()
