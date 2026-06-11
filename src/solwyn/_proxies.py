"""Provider-specific proxy classes for LLM API interception.

These thin delegation wrappers let ``Solwyn.chat.completions.create()``
(and the Anthropic/Google equivalents) route through ``_intercepted_call``
while passing everything else through to the underlying client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from solwyn._types import ProviderName

if TYPE_CHECKING:
    from solwyn.client import AsyncSolwyn, Solwyn


def _bedrock_internal_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Rename boto3's ``modelId`` to the pipeline's uniform ``model`` key.

    The whole interception pipeline (estimation, budget check, candidate walk,
    translation) keys on ``kwargs["model"]``; dispatch renames it back to
    ``modelId`` for the actual boto3 call. Raises TypeError (the same class
    boto3 raises for a missing required kwarg) when ``modelId`` is absent.
    """
    if "modelId" not in kwargs:
        raise TypeError("converse() requires the 'modelId' keyword argument")
    renamed = dict(kwargs)
    renamed["model"] = renamed.pop("modelId")
    return renamed


# ---------------------------------------------------------------------------
# Sync proxies
# ---------------------------------------------------------------------------


class _SyncChatCompletionsProxy:
    """Proxy for client.chat.completions that intercepts create()."""

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn

    def create(self, **kwargs: Any) -> Any:
        """Intercept chat.completions.create() with budget/circuit/reporting."""
        return self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to OpenAI's chat.completions."""
        return getattr(self._solwyn._client.chat.completions, name)


class _SyncChatProxy:
    """Proxy for client.chat that provides .completions.create()."""

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self.completions = _SyncChatCompletionsProxy(solwyn)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-completions attributes (OpenAI only).

        This proxy is only constructed for OpenAI clients. Any attribute
        that is not ``completions`` (set in __init__) falls through here.
        """
        if self._solwyn._detected_provider == ProviderName.OPENAI:
            return getattr(self._solwyn._client.chat, name)
        raise AttributeError(
            f"'chat.{name}' is not supported. "
            f"The Solwyn chat proxy is OpenAI-specific; Anthropic uses "
            f"'messages' and Google uses 'models'."
        )


class _SyncMessagesProxy:
    """Proxy for client.messages that intercepts create().

    Enables ``client.messages.create()`` (Anthropic's documented API)
    to go through _intercepted_call instead of __getattr__ pass-through.
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn

    def create(self, **kwargs: Any) -> Any:
        return self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solwyn._client.messages, name)


class _SyncModelsProxy:
    """Proxy for client.models that intercepts generate_content() and generate_content_stream().

    Enables ``client.models.generate_content()`` (Google's documented API)
    to go through _intercepted_call. The generate_content_stream() method
    passes _force_stream=True so _intercepted_call dispatches to the correct
    underlying SDK method.
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn

    def generate_content(self, **kwargs: Any) -> Any:
        return self._solwyn._intercepted_call(**kwargs)

    def generate_content_stream(self, **kwargs: Any) -> Any:
        return self._solwyn._intercepted_call(_force_stream=True, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solwyn._client.models, name)


# ---------------------------------------------------------------------------
# Async proxies
# ---------------------------------------------------------------------------


class _AsyncChatCompletionsProxy:
    """Async proxy for client.chat.completions that intercepts create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn

    async def create(self, **kwargs: Any) -> Any:
        """Intercept chat.completions.create() with budget/circuit/reporting."""
        return await self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to OpenAI's chat.completions."""
        return getattr(self._solwyn._client.chat.completions, name)


class _AsyncChatProxy:
    """Async proxy for client.chat that provides .completions.create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self.completions = _AsyncChatCompletionsProxy(solwyn)

    def __getattr__(self, name: str) -> Any:
        if self._solwyn._detected_provider == ProviderName.OPENAI:
            return getattr(self._solwyn._client.chat, name)
        raise AttributeError(
            f"'chat.{name}' is not supported. "
            f"The Solwyn chat proxy is OpenAI-specific; Anthropic uses "
            f"'messages' and Google uses 'models'."
        )


class _AsyncMessagesProxy:
    """Async proxy for client.messages that intercepts create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn

    async def create(self, **kwargs: Any) -> Any:
        return await self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solwyn._client.messages, name)


class _AsyncModelsProxy:
    """Async proxy for client.models.

    Intercepts generate_content() and generate_content_stream().
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn

    async def generate_content(self, **kwargs: Any) -> Any:
        return await self._solwyn._intercepted_call(**kwargs)

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        return await self._solwyn._intercepted_call(_force_stream=True, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solwyn._client.models, name)
