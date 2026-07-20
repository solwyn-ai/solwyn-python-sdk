"""Token counting with tiktoken and heuristic fallback.

Uses tiktoken (cl100k_base) for OpenAI models when available,
heuristic estimation for Anthropic models, and character-based
fallback when tiktoken is not installed.

- tiktoken is optional (import guarded)
- optional ``anthropic_client`` parameter for exact counting
- provider is passed explicitly by the caller (no auto-detection)
"""

from __future__ import annotations

import logging
from typing import Any

from solwyn._privacy import estimate_content_length, estimate_tokens_from_length

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)


class TokenizerManager:
    """Manage token counting for OpenAI, Anthropic, and Google models.

    Usage::

        tm = TokenizerManager()
        est = tm.estimate_tokens("Hello world", model="gpt-5.5", provider="openai")

        # With an Anthropic client for exact counting
        tm = TokenizerManager(anthropic_client=my_client)
        exact = tm.count_tokens(messages, model="claude-sonnet-5", provider="anthropic")
    """

    def __init__(self, anthropic_client: Any | None = None) -> None:
        """Initialise the tokenizer manager.

        Args:
            anthropic_client: Optional Anthropic SDK client instance.
                When provided, ``count_tokens`` can use the Anthropic
                ``messages.count_tokens()`` API for exact counts.
        """
        self._tiktoken_encoders: dict[str, Any] = {}
        self._anthropic_client = anthropic_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str, model: str, provider: str) -> int:
        """Estimate token count for *text* given a *model* name and *provider*.

        Caller must pass provider explicitly; the tokenizer does not
        re-detect it (that would require importing the provider
        registry, creating a circular dependency).

        Args:
            text: The text to tokenize.
            model: Model name (e.g. ``"gpt-5.5"``, ``"claude-sonnet-5"``).
            provider: Provider name (e.g. ``"openai"``, ``"anthropic"``,
                ``"google"``).  Any other value falls back to ``len(text) // 4``.

        Returns:
            Estimated token count.
        """
        if provider == "openai":
            return self._estimate_openai_tokens(text, model)
        elif provider == "anthropic":
            return self._estimate_anthropic_tokens(text, model)
        elif provider == "google":
            return self._estimate_google_tokens(text)
        else:
            return len(text) // 4

    def count_tokens(
        self,
        messages: list[Any],
        model: str,
        provider: str,
        system: str | None = None,
    ) -> int | None:
        """Return an exact token count, or ``None`` if unavailable.

        For Anthropic models with an ``anthropic_client`` configured,
        this calls ``client.messages.count_tokens()``.  On failure it
        falls back to the heuristic.

        Caller must pass provider explicitly; the tokenizer does not
        re-detect it (that would require importing the provider
        registry, creating a circular dependency).

        Args:
            messages: Message list in the provider's format.
            model: Model name.
            provider: Provider name (e.g. ``"anthropic"``).
            system: Optional system prompt (Anthropic only).

        Returns:
            Exact token count, heuristic estimate, or ``None``.
        """
        if provider == "anthropic" and self._anthropic_client is not None:
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                }
                if system is not None:
                    kwargs["system"] = system

                response = self._anthropic_client.messages.count_tokens(**kwargs)
                return int(response.input_tokens)
            except Exception as exc:
                # Do NOT log the exception value — it may include echoed
                # request content from the Anthropic SDK. Log only the
                # exception type and the model name.
                logger.warning(
                    "Failed to get exact Anthropic tokens: exc_type=%s model=%s",
                    type(exc).__name__,
                    model,
                )
                # Fall through to length-only heuristic. Reuse
                # estimate_content_length so content-block lists are
                # walked correctly (len(list) != character count).
                _kwargs: dict[str, Any] = {"messages": messages}
                if isinstance(system, str):
                    _kwargs["system"] = system
                total_chars = estimate_content_length(_kwargs)
                return estimate_tokens_from_length(total_chars, provider="anthropic")

        # For OpenAI or when no Anthropic client, return None (no exact
        # server-side counting API).
        return None

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    def _estimate_openai_tokens(self, text: str, model: str) -> int:
        """Estimate OpenAI tokens using tiktoken (if available)."""
        if tiktoken is None:
            # Fallback: rough character-based estimate
            return max(1, len(text) // 4)

        try:
            if model not in self._tiktoken_encoders:
                # cl100k_base covers GPT-4, GPT-3.5, and newer models
                self._tiktoken_encoders[model] = tiktoken.get_encoding("cl100k_base")

            encoder = self._tiktoken_encoders[model]
            return len(encoder.encode(text))
        except Exception as exc:
            # Log only the exception's class name — type-name-only convention
            # (fix [D]); never the exception object (str() may embed input text).
            logger.warning("Failed to use tiktoken for %s: %s", model, type(exc).__name__)
            return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Google
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_google_tokens(text: str) -> int:
        """Estimate Google/Gemini tokens via character heuristic (char / 4)."""
        return max(1, len(text) // 4)

    @staticmethod
    def _estimate_anthropic_tokens(text: str, model: str) -> int:
        """Estimate Anthropic tokens via heuristic character ratios."""
        char_count = len(text)

        if "haiku" in model:
            tokens = char_count / 4.5
        elif "sonnet" in model:
            tokens = char_count / 4.2
        elif "opus" in model:
            tokens = char_count / 3.8
        else:
            tokens = char_count / 4

        return max(1, int(tokens))
