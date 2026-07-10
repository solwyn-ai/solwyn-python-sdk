"""First-class adapter for native and OpenAI-compatible Together clients."""

from __future__ import annotations

from typing import Any

from solwyn.providers.openai_compatible import COMPAT_PROFILES, OpenAICompatibleAdapter

_TOGETHER_CLIENT_NAMES = frozenset({"Together", "AsyncTogether"})
# Live probe 2026-07-10 (Together SDK 2.22.1): sync and async streams carried
# usage on the terminal chunk. GLM-5.2 reported cached prompt tokens at
# prompt_tokens_details.cached_tokens (5,440 of 5,471), while Llama 3.3 70B
# reported usage.cached_tokens (4,608 of 4,701); both were prompt-token subsets.
_TOGETHER_PROFILE = next(profile for profile in COMPAT_PROFILES if profile.name == "together")


class TogetherAdapter(OpenAICompatibleAdapter):
    """Together adapter with duck-typed native client detection."""

    unmetered_spend_surfaces = frozenset(
        {
            "completions",
            "embeddings",
            "images",
            "videos",
            "audio",
            "rerank",
            "code_interpreter",
            "batches",
            "fine_tuning",
            "evals",
        }
    )

    def __init__(self) -> None:
        super().__init__(_TOGETHER_PROFILE)

    def detect_client(self, client: Any) -> bool:
        """Claim native Together clients or Together-hosted OpenAI clients."""
        client_type = type(client)
        module = getattr(client_type, "__module__", "")
        if module == "together" or module.startswith("together."):
            return getattr(client_type, "__name__", "") in _TOGETHER_CLIENT_NAMES
        return super().detect_client(client)
