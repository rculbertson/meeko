"""End-of-session summarization.

Runs after ``end_session`` / ``new_session`` finalize a SQLite session.
Loads the full transcript from the on-disk turn log, asks Sonnet for a
short title + 2-4 sentence summary, and writes both back to the
``sessions`` row plus the FTS index.

Summarization is fire-and-forget from the orchestrator's perspective —
failures log and skip; the session just isn't indexed for voice resume.
Never raises into the caller, so a flaky Anthropic call cannot block
the wake cycle.

The ``transcript`` stored in the FTS index comes from extracting
``text`` blocks from the JSON-encoded turn content. Tool-use and
tool-result blocks are dropped (they'd leak internal tool IDs into the
search index and aren't useful for topic matching).

Model choice: Sonnet (not Haiku). Sessions can grow to 50k-150k tokens
and the summary drives voice-resume recall — quality matters more than
the per-call cost at personal-use volumes. See ARCHITECTURE.md §6.3.
"""

import asyncio
import json
import logging
from typing import Any

import anthropic

from meeko.sessions import SessionStore

logger = logging.getLogger("meeko")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

# Chunking for very long transcripts is deferred (ARCHITECTURE.md §6.3
# notes ~150k tokens as the threshold). Today we send the whole thing;
# if it exceeds Sonnet's context window the API will error and we'll log
# and skip, which is acceptable for v1 personal-use volumes.

_PROMPT = """Summarize this brainstorming conversation for future reference.

Return a bare JSON object — no prose before or after, no markdown code fences:
{{
    "title": "short descriptive title, 3-8 words",
    "summary": "2-4 sentences covering: main topic, key ideas explored, \
decisions made, open questions, anything to return to"
}}

Transcript:
{transcript}"""


def _text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    """Extract concatenated text from an assistant content-block list,
    skipping tool_use / tool_result blocks."""
    parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    return " ".join(p for p in parts if p)


def _build_transcript(turns: list[dict[str, Any]]) -> str:
    """Build a plain-text transcript from the SQLite turn rows.

    User rows store the raw string; assistant rows store a list of
    content blocks. Tool-result user rows (list-of-dicts where every
    dict is a ``tool_result``) are dropped — they're execution noise.
    """
    lines: list[str] = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if isinstance(content, str):
            lines.append(f"{role.upper()}: {content}")
            continue
        if isinstance(content, list):
            # Assistant turn with blocks, or a user "turn" carrying
            # tool_result blocks (which we skip).
            if all(b.get("type") == "tool_result" for b in content):
                continue
            text = _text_from_blocks(content)
            if text:
                lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines)


def _first_text(content: list[Any]) -> str | None:
    """Return the first text block's string from a Messages API response."""
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            return getattr(block, "text", None)
    return None


def _extract_json_object(raw: str) -> str | None:
    """Pull the outermost JSON object out of a model response.

    Sonnet, despite being asked for JSON only, sometimes wraps the
    output in a ```json fenced block or prefixes with prose ("Sure,
    here's the summary:"). Slicing between the first ``{`` and last
    ``}`` handles both cases without regex gymnastics. Returns None
    if no braces found."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return raw[start : end + 1]


async def summarize_session(
    store: SessionStore,
    session_id: str,
    anthropic_client: anthropic.AsyncAnthropic,
) -> None:
    """Generate and persist the end-of-session summary for ``session_id``.

    Returns None on success or on any failure — errors are logged, never
    raised."""
    try:
        turns = await store.load_turns(session_id)
    except Exception:
        logger.exception("summarize_session: failed to load turns for %s", session_id)
        return

    if not turns:
        logger.info("summarize_session: no turns for %s, skipping", session_id)
        return

    transcript = _build_transcript(turns)
    if not transcript.strip():
        logger.info(
            "summarize_session: empty transcript for %s (all tool noise), skipping",
            session_id,
        )
        return

    try:
        response = await anthropic_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": _PROMPT.format(transcript=transcript)}
            ],
        )
    except Exception:
        logger.exception("summarize_session: Anthropic call failed for %s", session_id)
        return

    raw = _first_text(response.content)
    if raw is None:
        logger.warning(
            "summarize_session: no text block in response for %s (stop=%s)",
            session_id,
            getattr(response, "stop_reason", None),
        )
        return

    candidate = _extract_json_object(raw) or raw
    try:
        parsed = json.loads(candidate)
        title = parsed["title"]
        summary = parsed["summary"]
    except json.JSONDecodeError, KeyError, TypeError:
        logger.warning(
            "summarize_session: invalid JSON for %s; raw=%r", session_id, raw[:500]
        )
        return

    try:
        await store.update_session_metadata(
            session_id=session_id,
            title=title,
            summary=summary,
            transcript=transcript,
        )
    except Exception:
        logger.exception(
            "summarize_session: failed to persist metadata for %s", session_id
        )
        return

    logger.info("summarize_session: wrote summary for %s (title=%r)", session_id, title)


# Startup backfill can find many untitled sessions at once (e.g. after a
# run of crashes). Cap concurrent summary calls so they don't hammer the
# Anthropic rate limit.
_BACKFILL_CONCURRENCY = 3


class SummaryScheduler:
    """Owns the lifecycle of background summarization tasks.

    Summaries are fire-and-forget from the orchestrator's point of view,
    but not from shutdown's: they hold the SQLite store and a dedicated
    Anthropic client, so both must outlive every in-flight task.
    ``aclose()`` cancels the tasks first and closes the client after.

    The client is injected rather than built here so the caller decides
    its lifetime. It is separate from the live conversation client,
    which sidesteps any concern about concurrent use of one client from
    the turn loop and from background work.
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        client: anthropic.AsyncAnthropic,
    ) -> None:
        self._store = store
        self._client = client
        self._tasks: set[asyncio.Task] = set()

    def fire(self, session_id: str | None) -> None:
        """Summarize a just-finalized session in the background.

        No-op on None: a session with no turns never got a row (creation
        is lazy), so there is nothing to summarize — e.g. end_session
        called before the user said anything.
        """
        if session_id is None:
            return
        self._track(summarize_session(self._store, session_id, self._client))

    async def backfill(self, *, active_session_id: str | None) -> None:
        """Schedule summaries for sessions a prior run never finished.

        Any session with turns but no title was left behind by a run that
        was killed before its summary completed; without this,
        `list_sessions` keeps reporting it as "(no title)" and voice
        recall can't find it. Skips ``active_session_id`` — a resumed
        session is still being added to, so summarizing it now would
        index a transcript that's about to grow.
        """
        untitled = await self._store.list_untitled_sessions_with_turns()
        if untitled:
            if len(untitled) > 10:
                logger.warning(
                    "Backfilling %d untitled session(s); "
                    "startup may be slower than usual",
                    len(untitled),
                )
            else:
                logger.info("Backfilling %d untitled session(s)", len(untitled))
        sem = asyncio.Semaphore(_BACKFILL_CONCURRENCY)

        async def _rate_limited(sid: str) -> None:
            async with sem:
                await summarize_session(self._store, sid, self._client)

        for sid in untitled:
            if sid == active_session_id:
                continue
            self._track(_rate_limited(sid))

    async def aclose(self) -> None:
        """Cancel in-flight summaries, then close the client.

        Call before closing the store: a summary still running would
        write into a closed SQLite connection and crash, and a half-done
        summary isn't worth delaying shutdown for.

        Never raises. It runs mid-teardown, and an exception here would
        skip everything the caller still has to close after it.
        """
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            await self._client.close()
        except Exception:
            logger.exception("Failed to close summary client")

    def _track(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
