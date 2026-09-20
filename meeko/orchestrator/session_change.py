"""The post-turn hook: apply the session change a turn asked for.

Session tools (`end_session`, `new_session`, `load_session`) only set
flags on `SessionManager` while Sonnet is still talking. Once the turn's
speech has finished, `TurnWorker` calls `apply_post_turn_session_change`,
which performs the actual transition — finalize and summarize, rotate,
or swap in a prior transcript — and returns the state to enter next.
"""

import logging
from collections.abc import Callable

from meeko.claude_client import ClaudeClient
from meeko.orchestrator.state import State
from meeko.sessions import SessionStore
from meeko.tools.profile import ProfileManager
from meeko.tools.session import SessionManager
from meeko.wake_word import WakeWordDetector

logger = logging.getLogger("meeko")


async def apply_post_turn_session_change(
    *,
    session_manager: SessionManager,
    profile_manager: ProfileManager,
    claude: ClaudeClient,
    store: SessionStore,
    wake_detector: WakeWordDetector | None,
    session_id: str | None,
    fire_summary: Callable[[str | None], None],
) -> State:
    """Apply pending session-management actions queued during the last turn.

    Returns the next State to enter. `claude.session_id` tracks the live
    bound row internally (cleared by `reset_session`, rebound by
    `rebind_session`), so the caller never needs to thread the id back
    out. `should_load` can coexist with `should_end` (chain: finalize
    current then load a prior one in one turn). When the current session
    has no turns yet (session_id is None — lazy creation hasn't fired),
    there's nothing to summarize; `fire_summary` no-ops on None.
    """
    if session_manager.should_load():
        target_id = session_manager.get_load_target()
        assert target_id is not None  # guaranteed by should_load()
        fire_summary(session_id)
        if session_id is not None:
            logger.info(
                "load_session: fired summary for abandoned %s",
                session_id[:8],
            )
        # Rebind the runtime profile (system prompt + voice) to match
        # the loaded session's persisted profile_name before swapping
        # history, so subsequent turns persist back into a row whose
        # profile_name matches what is actually driving the model. A
        # missing row would mean Sonnet handed us a bogus id (or the
        # session got deleted between list_sessions and load_session) —
        # abort rather than binding Claude to a nonexistent session.
        target_row = await store.get_session(target_id)
        if target_row is None:
            logger.error(
                "load_session: target session %s not found; aborting load",
                target_id[:8],
            )
            session_manager.clear()
            return State.LISTENING
        profile_manager.rebind_profile(str(target_row["profile_name"]))
        turns = await store.load_turns(target_id)
        claude.load_history(turns)
        claude.rebind_session(target_id)
        await store.touch_session(target_id)
        session_manager.clear()
        logger.info(
            "load_session: swapped history to %s (%d turns), continuing in LISTENING",
            target_id[:8],
            len(turns),
        )
        return State.LISTENING

    if session_manager.should_end():
        finalized_sid = session_id
        claude.reset_session()
        session_manager.clear()
        fire_summary(finalized_sid)
        if wake_detector is not None:
            wake_detector.reset()
            logger.info(
                "Session ended; returning to IDLE "
                "(say the wake word to start a new conversation)"
            )
            return State.IDLE
        logger.info("Session ended; wake word disabled, returning to LISTENING")
        return State.LISTENING

    if session_manager.should_start_new():
        active = profile_manager.active_profile
        finalized_sid = session_id
        claude.reset_session()
        session_manager.clear()
        fire_summary(finalized_sid)
        logger.info(
            "new_session: rotated (profile=%s, row deferred), continuing in LISTENING",
            active.name,
        )
        return State.LISTENING

    return State.LISTENING
