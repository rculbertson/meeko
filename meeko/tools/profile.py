"""Profile switching tools for the voice agent.

Provides switch_profile and list_profiles functionality.
"""

import logging

from deepgram.agent.v1.types import (
    AgentV1SettingsAgentThinkOneItemFunctionsItem,
    AgentV1UpdatePrompt,
)

from meeko.profiles import Profile

logger = logging.getLogger("meeko")


class ProfileManager:
    """Manages profile state and switching."""

    def __init__(self, profiles: dict[str, Profile]):
        self._profiles = profiles
        self._active: str = "default"

    @property
    def active_profile(self) -> Profile:
        return self._profiles[self._active]

    async def switch_profile(self, profile_name: str, connection) -> str:
        if profile_name not in self._profiles:
            available = ", ".join(sorted(self._profiles))
            return f"Unknown profile '{profile_name}'. Available profiles: {available}"

        if profile_name == self._active:
            return f"Already using the {profile_name} profile."

        profile = self._profiles[profile_name]
        await connection.send_update_prompt(
            AgentV1UpdatePrompt(type="UpdatePrompt", prompt=profile.prompt)
        )
        self._active = profile_name
        logger.info("Switched to profile: %s", profile_name)
        return f"Switched to {profile_name} mode."

    def list_profiles(self) -> str:
        lines = []
        for name in sorted(self._profiles):
            marker = " (active)" if name == self._active else ""
            lines.append(f"{name}{marker}")
        return "Available profiles: " + ", ".join(lines)


def get_tool_definitions(
    profiles: dict[str, Profile],
) -> list[AgentV1SettingsAgentThinkOneItemFunctionsItem]:
    profile_names = sorted(profiles)
    names_list = ", ".join(profile_names)
    return [
        AgentV1SettingsAgentThinkOneItemFunctionsItem(
            name="switch_profile",
            description=(
                "Switch the assistant to a different profile/persona. "
                f"Available profiles: {names_list}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "profile_name": {
                        "type": "string",
                        "description": (
                            "The name of the profile to switch to. "
                            f"Must be one of: {names_list}"
                        ),
                        "enum": profile_names,
                    },
                },
                "required": ["profile_name"],
            },
        ),
        AgentV1SettingsAgentThinkOneItemFunctionsItem(
            name="list_profiles",
            description=(
                "List all available profiles and which one is currently active."
            ),
            parameters={"type": "object", "properties": {}},
        ),
    ]


async def handle(
    fn_name: str,
    args: dict,
    connection,
    *,
    manager: ProfileManager,
) -> str:
    """Handle a profile-related function call."""
    if fn_name == "switch_profile":
        return await manager.switch_profile(
            profile_name=args.get("profile_name", ""),
            connection=connection,
        )
    elif fn_name == "list_profiles":
        return manager.list_profiles()
    return f"Unknown profile function: {fn_name}"
