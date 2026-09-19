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
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _fts_literal_query(text: str) -> str:
    """Turn user speech into an FTS5 query that matches it literally.

    Unquoted, FTS5 reads its query language: a bare word may only hold
    letters, digits and underscores, so ordinary transcripts ("todo-app",
    "Supabase's", "e.g.", "c++") are syntax errors, and AND/OR/NOT are
    operators. Quoting each word makes it a string the tokenizer splits
    exactly as it split the indexed text; the words stay implicitly
    ANDed. A word of pure punctuation quotes to an empty phrase, which
    FTS5 ignores.
    """
    return " ".join('"' + word.replace('"', '""') + '"' for word in text.split())


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    profile_name  TEXT NOT NULL,
    title         TEXT,
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

-- FTS index populated once per session at end-of-session summarization.
-- Standalone (not content=...) so we don't have to manage rowid mapping
-- from the UUID-keyed sessions table. BM25 weights at query time give
-- title/summary precedence over transcript (see search_sessions).
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    session_id UNINDEXED,
    title,
    summary,
    transcript
);
"""


# Display placeholder for a session with no title. Titles are written by
# end-of-session summarization, so a session that's still active, or whose
# summary failed, has NULL. One constant so the CLI and the voice tools
# can't drift apart again (they had "(no title)" and "(untitled)").
UNTITLED = "(untitled)"


def default_db_path() -> Path:
    """Default DB location, following the XDG Base Directory spec.

    `$XDG_DATA_HOME/meeko/meeko.db`, falling back to
    `~/.local/share/meeko/meeko.db` when `$XDG_DATA_HOME` is unset or, per
    the spec, set to a relative (non-absolute) path.
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else None
    if base is None or not base.is_absolute():
        base = Path.home() / ".local" / "share"
    return base / "meeko" / "meeko.db"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        # Single-threaded executor serializes all SQLite access on one thread,
        # preventing concurrent-access crashes when DB calls overlap with
        # close() or with each other.
        self._executor = ThreadPoolExecutor(max_workers=1)

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

    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def create_session(self, profile_name: str) -> str:
        return await self._run(self._create_session_sync, profile_name)

    def _persist_turn_sync(
        self, session_id: str, role: str, content: str | Sequence[Any]
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
        self, session_id: str, role: str, content: str | Sequence[Any]
    ) -> None:
        await self._run(self._persist_turn_sync, session_id, role, content)

    def _get_latest_session_sync(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, profile_name, last_active FROM sessions "
            "ORDER BY last_active DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "profile_name": row[1], "last_active": row[2]}

    async def get_latest_session(self) -> dict[str, Any] | None:
        return await self._run(self._get_latest_session_sync)

    def _get_session_sync(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, profile_name, last_active, title FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "profile_name": row[1],
            "last_active": row[2],
            "title": row[3],
        }

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await self._run(self._get_session_sync, session_id)

    def _list_sessions_sync(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT s.id, s.profile_name, s.last_active, COUNT(t.id), s.title "
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
                "title": r[4],
            }
            for r in rows
        ]

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await self._run(self._list_sessions_sync)

    def _load_turns_sync(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [{"role": r[0], "content": json.loads(r[1])} for r in rows]

    async def load_turns(self, session_id: str) -> list[dict[str, Any]]:
        return await self._run(self._load_turns_sync, session_id)

    def _touch_session_sync(self, session_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (_now(), session_id),
            )

    async def touch_session(self, session_id: str) -> None:
        await self._run(self._touch_session_sync, session_id)

    def _set_session_profile_sync(self, session_id: str, profile_name: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET profile_name = ? WHERE id = ?",
                (profile_name, session_id),
            )

    async def set_session_profile(self, session_id: str, profile_name: str) -> None:
        """Record a mid-session profile switch on the session row.

        The row is created under whatever profile was active for the
        session's first utterance. Without this, resume and load_session
        would restore that starting profile rather than the one the
        session was left in. Leaves `last_active` alone: a switch isn't
        conversation activity, and the turn that made it already bumped it.
        """
        await self._run(self._set_session_profile_sync, session_id, profile_name)

    def _update_session_metadata_sync(
        self,
        session_id: str,
        title: str,
        summary: str,
        transcript: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET title = ?, summary = ? WHERE id = ?",
                (title, summary, session_id),
            )
            # Upsert into the standalone FTS table: drop any prior row for
            # this session (re-summarization is rare but safe) and reinsert.
            self._conn.execute(
                "DELETE FROM sessions_fts WHERE session_id = ?",
                (session_id,),
            )
            self._conn.execute(
                "INSERT INTO sessions_fts (session_id, title, summary, transcript) "
                "VALUES (?, ?, ?, ?)",
                (session_id, title, summary, transcript),
            )

    def _list_untitled_sessions_with_turns_sync(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT s.id FROM sessions s "
            "WHERE s.title IS NULL "
            "AND EXISTS (SELECT 1 FROM turns t WHERE t.session_id = s.id) "
            "ORDER BY s.last_active"
        ).fetchall()
        return [r[0] for r in rows]

    async def list_untitled_sessions_with_turns(self) -> list[str]:
        return await self._run(self._list_untitled_sessions_with_turns_sync)

    async def update_session_metadata(
        self,
        session_id: str,
        title: str,
        summary: str,
        transcript: str,
    ) -> None:
        """Persist the end-of-session summary.

        Writes ``title`` and ``summary`` to the sessions row and upserts
        the corresponding FTS row. ``transcript`` is stored only in the
        FTS table (fallback recall when the summary misses a keyword);
        the on-disk turn log remains the source of truth."""
        await self._run(
            self._update_session_metadata_sync,
            session_id,
            title,
            summary,
            transcript,
        )

    def _search_sessions_sync(
        self,
        query: str | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        match = _fts_literal_query(query) if query else ""
        date_clauses: list[str] = []
        date_params: list[Any] = []
        if since is not None:
            date_clauses.append("s.last_active >= ?")
            date_params.append(since)
        if until is not None:
            date_clauses.append("s.last_active < ?")
            date_params.append(until)

        if match:
            sql = (
                "SELECT f.session_id, f.title, s.last_active "
                "FROM sessions_fts f JOIN sessions s ON s.id = f.session_id "
                "WHERE sessions_fts MATCH ?"
            )
            params: list[Any] = [match]
            for c in date_clauses:
                sql += f" AND {c}"
            params.extend(date_params)
            sql += " ORDER BY bm25(sessions_fts, 0.0, 10.0, 5.0, 1.0) LIMIT ?"
            params.append(limit)
        elif date_clauses:
            # Date-only path hits the base sessions table so unfinalized
            # sessions (which haven't been written to sessions_fts yet)
            # are still counted — important for "today"/"yesterday" asks.
            sql = (
                "SELECT s.id, s.title, s.last_active FROM sessions s WHERE "
                + " AND ".join(date_clauses)
                + " ORDER BY s.last_active DESC LIMIT ?"
            )
            params = [*date_params, limit]
        else:
            return []

        rows = self._conn.execute(sql, params).fetchall()
        return [{"session_id": r[0], "title": r[1], "last_active": r[2]} for r in rows]

    async def search_sessions(
        self,
        query: str | None = None,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search sessions by keyword and/or `last_active` range.

        ``query`` runs an FTS5 match over titles/summaries/transcripts of
        finalized sessions (each word quoted so user speech is treated
        as a literal multi-token match). ``since`` (inclusive) and
        ``until`` (exclusive) are ISO-8601 timestamps compared lexically
        against ``last_active``; either bound may be omitted.

        Returns up to ``limit`` rows. With a query, results are ordered
        by BM25 relevance (title matches rank highest); date-only
        queries are ordered by ``last_active`` descending."""
        return await self._run(self._search_sessions_sync, query, since, until, limit)

    async def close(self) -> None:
        await self._run(self._conn.close)
        self._executor.shutdown(wait=False, cancel_futures=True)
