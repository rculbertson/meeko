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
            wake_word="meeko",
            prompt="QUERY-PROMPT",
            voice="thalia",
        ),
        "conversation": Profile(
            name="conversation",
            wake_word="meeko",
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
        "query": Profile(
            name="query", wake_word="meeko", prompt="QUERY-PROMPT", voice=None
        ),
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
