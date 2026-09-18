"""Session-management tools exposed to Sonnet.

Provides four tools:

- ``end_session``: user signals they want to end the conversation ("end conversation",
  "end session", "that's enough for today", etc.). After SPEAKING the post-turn
  hook finalizes the session and returns to IDLE, re-arming the wake word.
- ``new_session``: user wants to start a fresh thread without stopping
  Meeko ("let's start fresh", "different topic"). After SPEAKING the
  post-turn hook finalizes the current session, allocates a new one, and
  stays in LISTENING.
- ``list_sessions``: search prior sessions by keyword. The handler runs
  an FTS query and returns a human-readable list Sonnet can read back.
- ``load_session``: resume a prior session by id. After SPEAKING the
  post-turn hook swaps the in-memory message array and rebinds to the
  target session's SQLite row, then stays in LISTENING.

The post-turn hook is ``_apply_post_turn_session_change``, which
``TurnWorker`` calls once each turn's speech has finished.

The end/new flags are mutually exclusive. Load is independent — the
post-turn hook always fires the summary for the abandoned session when
load is requested, so Sonnet doesn't need to chain end_session first.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, time
from typing import TYPE_CHECKING

from meeko.sessions import UNTITLED
from meeko.tools.dispatch import ToolDefinition

if TYPE_CHECKING:
    from meeko.sessions import SessionStore

logger = logging.getLogger("meeko")


class SessionManager:
    """Shared state between the session-tool handlers and the post-turn
    hook, ``_apply_post_turn_session_change``.

    Handlers only set flags — they do NOT tear down state directly.
    ``TurnWorker`` drains Sonnet's acknowledgement via TTS first and
    then, after SPEAKING ends, the hook performs the actual session
    transition.

    The end/new request flags are mutually exclusive: the two intents
    contradict each other, so a later request overrides any earlier
    one within the same turn rather than letting both be true at once.
    That keeps the hook's branch selection unambiguous.

    Load takes precedence over end if both are set in the same turn —
    the hook's load branch already summarizes the abandoned
    session, so an explicit end_session call is redundant but harmless."""

    def __init__(self) -> None:
        self._should_end = False
        self._should_start_new = False
        self._load_target: str | None = None

    def request_end(self) -> None:
        self._should_end = True
        self._should_start_new = False

    def should_end(self) -> bool:
        return self._should_end

    def request_new(self) -> None:
        self._should_start_new = True
        self._should_end = False
        self._load_target = None

    def should_start_new(self) -> bool:
        return self._should_start_new

    def request_load(self, session_id: str) -> None:
        self._load_target = session_id
        self._should_start_new = False

    def should_load(self) -> bool:
        return self._load_target is not None

    def get_load_target(self) -> str | None:
        return self._load_target

    def clear(self) -> None:
        self._should_end = False
        self._should_start_new = False
        self._load_target = None


def _local_date_to_utc_iso(local_date: str) -> str:
    """Convert a local-date ``YYYY-MM-DD`` to a UTC ISO timestamp string
    suitable for lexicographic comparison against the ``last_active``
    column (which is also written via ``datetime.now(UTC).isoformat()``)."""
    parsed = datetime.strptime(local_date, "%Y-%m-%d").date()
    local_tz = datetime.now().astimezone().tzinfo
    local_midnight = datetime.combine(parsed, time.min, tzinfo=local_tz)
    return local_midnight.astimezone(UTC).isoformat()


def _format_local_date(utc_iso: str | None) -> str:
    """Render a UTC ISO timestamp as a local-date YYYY-MM-DD.

    `last_active` is stored in UTC; results are displayed alongside the
    user's local "today", so we convert to the same local timezone to
    avoid an off-by-one date when sessions end late in the evening."""
    if not utc_iso:
        return "?"
    try:
        return datetime.fromisoformat(utc_iso).astimezone().strftime("%Y-%m-%d")
    except ValueError:
        return "?"


def _describe_criteria(query: str | None, since: str | None, until: str | None) -> str:
    parts: list[str] = []
    if query:
        parts.append(f"matching '{query}'")
    if since and until:
        parts.append(f"from {since} to {until}")
    elif since:
        parts.append(f"since {since}")
    elif until:
        parts.append(f"before {until}")
    return " ".join(parts) if parts else ""


def get_tool_definitions() -> list[ToolDefinition]:
    return [
        {
            "name": "end_session",
            "description": (
                "End the current conversation and return Meeko to idle. "
                "Call this when the user clearly indicates they want to "
                "end the conversation — e.g. 'end conversation', "
                "'end session', 'that's enough for today', 'let's pick "
                "this up later'. Do not respond verbally; call this "
                "tool immediately and silently so Meeko turns off "
                "without speaking. A bare "
                "'stop' (or 'wait', 'hold on', 'never mind') is an "
                "interrupt, not an end-session signal — do not call this "
                "tool in that case. Also do not call when the user is "
                "ambiguous or immediately walks back the signal ('that's "
                "enough about X, let's talk about Y')."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "new_session",
            "description": (
                "Finalize the current conversation and start a fresh one, "
                "without stopping Meeko. Call this when the user clearly "
                "wants to switch to a new thread — e.g. 'let's start "
                "fresh', 'different topic', 'new conversation'. Before "
                "calling this tool, confirm briefly that the user wants "
                "to discard the current thread (e.g. 'Got it, starting "
                "fresh.'). Do not call this tool when the user is just "
                "changing subjects within the same conversation "
                "('that's enough about X, let's talk about Y') — only "
                "when they explicitly want to wipe the slate."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_sessions",
            "description": (
                "Search prior conversations by keyword and/or date range. "
                "Call this when the user asks about a previous conversation "
                "— e.g. 'what were we working on?', 'find the todo app "
                "session', 'how many conversations did we have yesterday?'. "
                "Pass key terms as `query` and/or a date range as `since` "
                "(inclusive) and `until` (exclusive), formatted YYYY-MM-DD "
                "in the user's local timezone. The system prompt tells you "
                "today's local date — use it to convert relative phrases: "
                "'yesterday' is since=<yesterday>, until=<today>; 'today' "
                "is since=<today>, until=<tomorrow>; 'last week' is a "
                "7-day range ending today. At least one of `query`, "
                "`since`, `until` must be provided. Read the top result "
                "titles back to the user so they can confirm which "
                "session to load."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Keywords to search for in session titles,"
                            " summaries, and transcripts. Optional."
                        ),
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Inclusive start date, YYYY-MM-DD in the "
                            "user's local timezone. Optional."
                        ),
                    },
                    "until": {
                        "type": "string",
                        "description": (
                            "Exclusive end date, YYYY-MM-DD in the "
                            "user's local timezone. Optional."
                        ),
                    },
                },
            },
        },
        {
            "name": "load_session",
            "description": (
                "Resume a prior conversation by loading its history. Call "
                "this after the user confirms which session to load (from "
                "list_sessions results). Pass the exact session `id` from "
                "the list. After SPEAKING completes, Meeko will swap its "
                "memory to the loaded session and continue from there. "
                "Meeko automatically finalizes the current conversation "
                "in the background before swapping, so you don't need to "
                "call end_session first."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The session id to load (from list_sessions).",
                    }
                },
                "required": ["id"],
            },
        },
    ]


def _handle_end(manager: SessionManager) -> str:
    manager.request_end()
    logger.info("end_session tool called; will transition after SPEAKING")
    return "Session ended. Meeko will return to idle after this turn."


def _handle_new(manager: SessionManager) -> str:
    manager.request_new()
    logger.info("new_session tool called; will rotate session after SPEAKING")
    return "Starting a fresh session. Meeko will swap the context after this turn."


async def _handle_list(args: dict, store: SessionStore | None) -> str:
    query = str(args.get("query", "")).strip() or None
    since_arg = str(args.get("since", "")).strip() or None
    until_arg = str(args.get("until", "")).strip() or None
    if not (query or since_arg or until_arg):
        return "Please provide a search query or date range."
    if store is None:
        return "Session search is not available."
    try:
        since_iso = _local_date_to_utc_iso(since_arg) if since_arg else None
        until_iso = _local_date_to_utc_iso(until_arg) if until_arg else None
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD for `since` and `until`."
    results = await store.search_sessions(query, since=since_iso, until=until_iso)
    criteria = _describe_criteria(query, since_arg, until_arg)
    if not results:
        return f"No sessions found {criteria}."
    lines = [f"Found {len(results)} session(s) {criteria}:"]
    for i, r in enumerate(results, 1):
        title = r["title"] or UNTITLED
        lines.append(
            f'{i}. "{title}" — {_format_local_date(r["last_active"])} '
            f"(id: {r['session_id']})"
        )
    return "\n".join(lines)


async def _handle_load(
    args: dict,
    manager: SessionManager,
    store: SessionStore | None,
    current_session_id: str | None,
) -> str:
    session_id = str(args.get("id", "")).strip()
    if not session_id:
        return "Please provide a session id."
    if store is None:
        return "Session loading is not available."
    row = await store.get_session(session_id)
    if row is None:
        return (
            f"No session found with id '{session_id}'. "
            "Call list_sessions to find the correct id."
        )
    if session_id == current_session_id:
        return "That's the current session — already loaded."
    manager.request_load(session_id)
    title = row.get("title") or UNTITLED
    logger.info(
        "load_session tool called for %s; will swap history after SPEAKING",
        session_id[:8],
    )
    return f"Loading '{title}'. Continuing from there after this turn."


async def handle(
    fn_name: str,
    args: dict,
    *,
    manager: SessionManager,
    store: SessionStore | None = None,
    current_session_id: str | None = None,
) -> str:
    if fn_name == "end_session":
        return _handle_end(manager)
    if fn_name == "new_session":
        return _handle_new(manager)
    if fn_name == "list_sessions":
        return await _handle_list(args, store)
    if fn_name == "load_session":
        return await _handle_load(args, manager, store, current_session_id)
    return f"Unknown session function: {fn_name}"
