"""Session-management tools exposed to Sonnet.

Provides four tools:

- ``end_session``: user signals they want to stop ("stop", "goodnight",
  "that's enough for today", etc.). After SPEAKING the orchestrator
  finalizes the session and returns to IDLE, re-arming the wake word.
- ``new_session``: user wants to start a fresh thread without stopping
  Meeko ("let's start fresh", "different topic"). After SPEAKING the
  orchestrator finalizes the current session, allocates a new one, and
  stays in LISTENING.
- ``list_sessions``: search prior sessions by keyword. The handler runs
  an FTS query and returns a human-readable list Sonnet can read back.
- ``load_session``: resume a prior session by id. After SPEAKING the
  orchestrator swaps the in-memory message array and rebinds to the
  target session's SQLite row, then stays in LISTENING.

The end/new flags are mutually exclusive. The load target is independent
of end (so Sonnet can chain end_session + load_session in one turn to
finalize the current session and resume a prior one in a single step).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from meeko.tools.dispatch import ToolDefinition

if TYPE_CHECKING:
    from meeko.sessions import SessionStore

logger = logging.getLogger("meeko")


class SessionManager:
    """Shared state between the session-tool handlers and the
    orchestrator's post-SPEAKING hook.

    Handlers only set flags — they do NOT tear down state directly.
    The orchestrator drains Sonnet's acknowledgement via TTS first and
    then, after SPEAKING ends, performs the actual session transition.

    The end/new request flags are mutually exclusive: the two intents
    contradict each other, so a later request overrides any earlier
    one within the same turn rather than letting both be true at once.
    That keeps the orchestrator branch selection unambiguous.

    The load target is independent of the end flag: Sonnet can chain
    end_session + load_session in one turn ("wrap up this one and pick
    up the todo app conversation"). The orchestrator fires the summary
    for the ending session before swapping history."""

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


def get_tool_definitions() -> list[ToolDefinition]:
    return [
        {
            "name": "end_session",
            "description": (
                "End the current conversation and return Meeko to idle. "
                "Call this when the user clearly indicates they want to "
                "stop — e.g. 'stop', 'goodnight', 'that's enough for "
                "today', 'let's pick this up later'. Before calling this "
                "tool, respond with a brief verbal acknowledgement (e.g. "
                "'Goodnight!'). Do not call this tool when the user is "
                "ambiguous or immediately walks back the signal "
                "('stop interrupting me', 'that's enough about X, let's "
                "talk about Y')."
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
                "Search prior conversations by keyword and return a short "
                "list of matches. Call this when the user asks about a "
                "previous conversation — e.g. 'what were we working on?', "
                "'find the todo app session', 'go back to our discussion "
                "about Supabase'. Pass the user's key terms as `query`. "
                "Read the top result titles back to the user so they can "
                "confirm which session to load."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Keywords to search for in session titles,"
                            " summaries, and transcripts."
                        ),
                    }
                },
                "required": ["query"],
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
                "If the user wants to resume mid-conversation (not at "
                "end_session time), call end_session first to finalize "
                "the current thread, then call load_session in the same "
                "turn."
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


async def handle(
    fn_name: str,
    args: dict,
    *,
    manager: SessionManager,
    store: SessionStore | None = None,
) -> str:
    if fn_name == "end_session":
        manager.request_end()
        logger.info("end_session tool called; will transition after SPEAKING")
        return "Session ended. Meeko will return to idle after this turn."

    if fn_name == "new_session":
        manager.request_new()
        logger.info("new_session tool called; will rotate session after SPEAKING")
        return "Starting a fresh session. Meeko will swap the context after this turn."

    if fn_name == "list_sessions":
        query = str(args.get("query", "")).strip()
        if not query:
            return "Please provide a search query."
        if store is None:
            return "Session search is not available."
        results = await store.search_sessions(query)
        if not results:
            return f"No sessions found matching '{query}'."
        lines = [f"Found {len(results)} session(s) matching '{query}':"]
        for i, r in enumerate(results, 1):
            title = r["title"] or "(no title)"
            date = r["last_active"][:10] if r["last_active"] else "?"
            lines.append(f'{i}. "{title}" — {date} (id: {r["session_id"]})')
        return "\n".join(lines)

    if fn_name == "load_session":
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
        manager.request_load(session_id)
        logger.info(
            "load_session tool called for %s; will swap history after SPEAKING",
            session_id[:8],
        )
        return f"Loading session '{session_id}'. Continuing from there after this turn."

    return f"Unknown session function: {fn_name}"
