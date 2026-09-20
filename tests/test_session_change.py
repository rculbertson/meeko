"""Tests for meeko.orchestrator.session_change: the post-turn hook."""

from unittest.mock import MagicMock

from meeko.config import Profile
from meeko.orchestrator.session_change import apply_post_turn_session_change
from meeko.orchestrator.state import State
from meeko.sessions import SessionStore
from meeko.tools.profile import ProfileManager
from meeko.tools.session import SessionManager


async def test_apply_post_turn_session_change_rebinds_profile_on_load(tmp_path):
    """Direct test of the should_load branch in apply_post_turn_session_change.

    When the loaded session was created under a different profile than
    the one currently active, the runtime profile (system prompt + voice
    via ProfileManager) must be rebound before history is swapped, so
    subsequent turns persist into a row whose profile_name still matches
    what is driving the model. Regression test for #50.
    """

    profiles = {
        "query": Profile(
            name="query",
            prompt="QUERY-PROMPT",
            voice="thalia",
        ),
        "conversation": Profile(
            name="conversation",
            prompt="CONVERSATION-PROMPT",
            voice="orion",
        ),
    }

    db_path = tmp_path / "meeko.db"
    store = SessionStore.open(db_path)
    try:
        # Target session was created under the *conversation* profile —
        # different from the "query" profile we'll have active.
        target_id = await store.create_session("conversation")
        await store.persist_turn(target_id, "user", "let's keep chatting")
        await store.persist_turn(
            target_id, "assistant", [{"type": "text", "text": "Sure!"}]
        )

        claude = MagicMock()
        claude.session_id = None
        speaker = MagicMock()

        profile_manager = ProfileManager(
            profiles, claude_client=claude, active_name="query"
        )
        profile_manager.set_speaker(speaker)

        session_manager = SessionManager()
        session_manager.request_load(target_id)

        summary_calls: list[str | None] = []

        def fire_summary(sid: str | None) -> None:
            summary_calls.append(sid)

        new_state = await apply_post_turn_session_change(
            session_manager=session_manager,
            profile_manager=profile_manager,
            claude=claude,
            store=store,
            wake_detector=None,
            session_id=None,
            fire_summary=fire_summary,
        )

        assert new_state == State.LISTENING
        # Profile was rebound to the loaded session's profile.
        assert profile_manager.active_profile.name == "conversation"
        claude.set_system_prompt.assert_called_once_with("CONVERSATION-PROMPT")
        speaker.set_profile.assert_called_once_with(profiles["conversation"])
        # History was loaded and the session id rebound to the target.
        claude.load_history.assert_called_once()
        claude.rebind_session.assert_called_once_with(target_id)
    finally:
        await store.close()


async def test_apply_post_turn_session_change_aborts_load_on_missing_row(
    tmp_path, caplog
):
    """If Sonnet hands us a bogus session id (or the row was deleted),
    abort the load rather than binding Claude to a nonexistent session."""

    profiles = {
        "query": Profile(name="query", prompt="QUERY-PROMPT", voice=None),
    }

    db_path = tmp_path / "meeko.db"
    store = SessionStore.open(db_path)
    try:
        claude = MagicMock()
        claude.session_id = None
        speaker = MagicMock()
        profile_manager = ProfileManager(
            profiles, claude_client=claude, active_name="query"
        )
        profile_manager.set_speaker(speaker)

        session_manager = SessionManager()
        bogus_id = "00000000-0000-0000-0000-000000000000"
        session_manager.request_load(bogus_id)

        with caplog.at_level("ERROR", logger="meeko"):
            new_state = await apply_post_turn_session_change(
                session_manager=session_manager,
                profile_manager=profile_manager,
                claude=claude,
                store=store,
                wake_detector=None,
                session_id=None,
                fire_summary=lambda _sid: None,
            )

        assert new_state == State.LISTENING
        claude.load_history.assert_not_called()
        claude.rebind_session.assert_not_called()
        # Pending load flag cleared so we don't loop on the bogus id.
        assert not session_manager.should_load()
        assert any("not found" in r.getMessage() for r in caplog.records)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Returned state, flag clearing, reset and summary for each branch. These
# are the hook's whole contract with TurnWorker; the run() drive-throughs in
# test_main.py only check what reached Claude and TTS, so a wrong state or
# a flag left set would otherwise go unnoticed.
# ---------------------------------------------------------------------------


async def _apply(
    session_manager, *, wake_detector=None, store=None, session_id="sess-1"
):
    """Run the hook with a mock Claude client and a recording fire_summary.
    Returns (new_state, claude, summarized session ids)."""
    profiles = {"query": Profile(name="query", prompt="P", voice=None)}
    claude = MagicMock()
    profile_manager = ProfileManager(
        profiles, claude_client=claude, active_name="query"
    )
    profile_manager.set_speaker(MagicMock())
    summarized: list[str | None] = []
    new_state = await apply_post_turn_session_change(
        session_manager=session_manager,
        profile_manager=profile_manager,
        claude=claude,
        store=store if store is not None else MagicMock(),
        wake_detector=wake_detector,
        session_id=session_id,
        fire_summary=summarized.append,
    )
    return new_state, claude, summarized


async def test_no_pending_change_stays_listening():
    """An ordinary turn: keep listening, and touch nothing else."""
    detector = MagicMock()
    new_state, claude, summarized = await _apply(
        SessionManager(), wake_detector=detector
    )

    assert new_state == State.LISTENING
    claude.reset_session.assert_not_called()
    detector.reset.assert_not_called()
    assert summarized == []


async def test_end_session_with_wake_word_returns_to_idle():
    session_manager = SessionManager()
    session_manager.request_end()
    detector = MagicMock()

    new_state, claude, summarized = await _apply(
        session_manager, wake_detector=detector
    )

    assert new_state == State.IDLE
    detector.reset.assert_called_once()
    claude.reset_session.assert_called_once()
    assert summarized == ["sess-1"]
    # Cleared, so the next turn doesn't end the session again.
    assert not session_manager.should_end()


async def test_end_session_without_wake_word_stays_listening():
    """With no wake word there is no way out of IDLE, so ending the session
    must leave Meeko listening."""
    session_manager = SessionManager()
    session_manager.request_end()

    new_state, claude, summarized = await _apply(session_manager, wake_detector=None)

    assert new_state == State.LISTENING
    claude.reset_session.assert_called_once()
    assert summarized == ["sess-1"]
    assert not session_manager.should_end()


async def test_new_session_rotates_and_keeps_listening():
    """new_session finalizes the current session but, unlike end_session,
    doesn't go back to sleep."""
    session_manager = SessionManager()
    session_manager.request_new()
    detector = MagicMock()

    new_state, claude, summarized = await _apply(
        session_manager, wake_detector=detector
    )

    assert new_state == State.LISTENING
    detector.reset.assert_not_called()
    claude.reset_session.assert_called_once()
    assert summarized == ["sess-1"]
    assert not session_manager.should_start_new()


async def test_load_session_summarizes_abandoned_and_bumps_target(tmp_path):
    """Loading a session summarizes the one being left, marks the target as
    most recently active (so it tops --list-sessions), and clears the
    request."""
    store = SessionStore.open(tmp_path / "meeko.db")
    try:
        target_id = await store.create_session("query")
        await store.persist_turn(target_id, "user", "earlier")
        stale = "2020-01-01T00:00:00+00:00"
        store._conn.execute(
            "UPDATE sessions SET last_active = ? WHERE id = ?", (stale, target_id)
        )
        session_manager = SessionManager()
        session_manager.request_load(target_id)

        new_state, claude, summarized = await _apply(session_manager, store=store)

        assert new_state == State.LISTENING
        claude.rebind_session.assert_called_once_with(target_id)
        assert summarized == ["sess-1"]
        assert (await store.get_session(target_id))["last_active"] > stale
        assert not session_manager.should_load()
    finally:
        await store.close()
