"""First-class adapter for native and OpenAI-compatible Together clients."""

from __future__ import annotations

from typing import Any

from solwyn.providers.openai_compatible import COMPAT_PROFILES, OpenAICompatibleAdapter

_TOGETHER_CLIENT_NAMES = frozenset({"Together", "AsyncTogether"})
_TOGETHER_PROFILE = next(profile for profile in COMPAT_PROFILES if profile.name == "together")


class TogetherAdapter(OpenAICompatibleAdapter):
    """Together adapter with duck-typed native client detection."""

    def __init__(self) -> None:
        super().__init__(_TOGETHER_PROFILE)

    def detect_client(self, client: Any) -> bool:
        """Claim native Together clients or Together-hosted OpenAI clients."""
        client_type = type(client)
        module = getattr(client_type, "__module__", "")
        if module == "together" or module.startswith("together."):
            return getattr(client_type, "__name__", "") in _TOGETHER_CLIENT_NAMES
        return super().detect_client(client)
