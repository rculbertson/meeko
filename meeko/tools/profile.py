"""Profile-switching tools for the voice assistant.

Provides switch_profile and list_profiles. Switching updates the system
prompt held by the ClaudeClient; the change takes effect on the next
Claude call.
"""

import logging
from typing import TYPE_CHECKING

from meeko.profiles import Profile
from meeko.tools.dispatch import ToolDefinition

if TYPE_CHECKING:
    from meeko.claude_client import ClaudeClient

logger = logging.getLogger("meeko")


class ProfileManager:
    """Manages profile state and switching.

    ``claude_client`` is expected to expose ``set_system_prompt(str)``.
    We take it as a protocol-ish dependency to avoid an import cycle.
    """

    def __init__(
        self, profiles: dict[str, Profile], claude_client: "ClaudeClient | None" = None
    ):
        self._profiles = profiles
        self._active: str = "default"
        self._claude = claude_client

    def set_claude_client(self, claude_client: "ClaudeClient") -> None:
        self._claude = claude_client

    @property
    def active_profile(self) -> Profile:
        return self._profiles[self._active]

    async def switch_profile(self, profile_name: str) -> str:
        if profile_name not in self._profiles:
            available = ", ".join(sorted(self._profiles))
            return f"Unknown profile '{profile_name}'. Available profiles: {available}"

        if profile_name == self._active:
            return f"Already using the {profile_name} profile."

        profile = self._profiles[profile_name]
        if self._claude is not None:
            self._claude.set_system_prompt(profile.prompt)
        self._active = profile_name
        logger.info("Switched to profile: %s", profile_name)
        return f"Switched to {profile_name} mode."

    def list_profiles(self) -> str:
        lines = []
        for name in sorted(self._profiles):
            marker = " (active)" if name == self._active else ""
            lines.append(f"{name}{marker}")
        return "Available profiles: " + ", ".join(lines)


def get_tool_definitions(profiles: dict[str, Profile]) -> list[ToolDefinition]:
    profile_names = sorted(profiles)
    names_list = ", ".join(profile_names)
    return [
        {
            "name": "switch_profile",
            "description": (
                "Switch the assistant to a different profile/persona. "
                f"Available profiles: {names_list}"
            ),
            "input_schema": {
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
        },
        {
            "name": "list_profiles",
            "description": (
                "List all available profiles and which one is currently active."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
    ]


async def handle(
    fn_name: str,
    args: dict,
    *,
    manager: ProfileManager,
) -> str:
    """Handle a profile-related function call."""
    if fn_name == "switch_profile":
        return await manager.switch_profile(
            profile_name=args.get("profile_name", ""),
        )
    elif fn_name == "list_profiles":
        return manager.list_profiles()
    return f"Unknown profile function: {fn_name}"
