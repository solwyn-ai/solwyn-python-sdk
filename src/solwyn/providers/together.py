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

    # Together-specific untracked spend surfaces that the central per-dialect map
    # (_base._UNSHIPPED_SPEND_SURFACES) does NOT already cover. The warn-once
    # posture unions this with that map (Together speaks the "openai" dialect):
    #   - translations: omitted here — the central map's one remaining openai
    #     warn-once surface, so declaring it again would be redundant.
    #   - embeddings / images / audio.transcriptions / audio.speech / videos:
    #     omitted — they are intercepted, so they must NOT advertise themselves as
    #     untracked (metered surfaces; a videos.create on this compat client fails
    #     loud with UnsupportedSurfaceError, Sora being OpenAI-only).
    #   - batches / fine_tuning: omitted — per the posture taxonomy these
    #     are truly-unrelated resources that pass through SILENTLY, not warn-once.
    # What remains are Together's genuinely-billable, not-yet-tracked extras.
    unmetered_spend_surfaces = frozenset(
        {
            "completions",
            "rerank",
            "code_interpreter",
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
