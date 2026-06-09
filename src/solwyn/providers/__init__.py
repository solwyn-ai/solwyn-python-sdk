"""Provider adapter registry.

Registry functions look up the correct adapter by provider name, model string,
or SDK client instance. Adapters are loaded lazily on first access to avoid
importing provider SDKs that may not be installed.
"""

from __future__ import annotations

from typing import Any

from solwyn.providers._protocol import ProviderAdapter

__all__ = [
    "ProviderAdapter",
    "get_adapter_by_name",
    "get_adapter_for_client",
    "get_adapter_for_model",
]

# Lazy-loaded on first call to any registry function.
_ADAPTERS: list[ProviderAdapter] | None = None
_ADAPTER_BY_NAME: dict[str, ProviderAdapter] | None = None


def _ensure_loaded() -> None:
    global _ADAPTERS, _ADAPTER_BY_NAME
    if _ADAPTERS is None:
        from solwyn.providers.anthropic import AnthropicAdapter
        from solwyn.providers.bedrock import BedrockAdapter
        from solwyn.providers.google import GoogleAdapter
        from solwyn.providers.openai import OpenAIAdapter
        from solwyn.providers.openai_compatible import build_compat_adapters

        # ORDER IS LOAD-BEARING for detect_client: OpenAI-compatible adapters
        # claim openai-module clients whose base_url targets another vendor
        # (named profiles first, the generic catch-all last among them), so
        # the plain OpenAIAdapter must come AFTER them — it matches ANY
        # openai-module client and would otherwise shadow every compat
        # provider as "openai". Bedrock detection is exact (botocore client
        # shape), so its position is unconstrained.
        _ADAPTERS = [
            *build_compat_adapters(),
            OpenAIAdapter(),
            AnthropicAdapter(),
            GoogleAdapter(),
            BedrockAdapter(),
        ]
        _ADAPTER_BY_NAME = {a.name: a for a in _ADAPTERS}


def get_adapter_by_name(name: str) -> ProviderAdapter:
    """Return the adapter registered under *name* or raise ValueError."""
    _ensure_loaded()
    if _ADAPTER_BY_NAME is None:
        raise RuntimeError("Provider registry failed to initialise")  # should never happen
    adapter = _ADAPTER_BY_NAME.get(name)
    if adapter is None:
        raise ValueError(f"Unknown provider '{name}'. Known: {sorted(_ADAPTER_BY_NAME)}")
    return adapter


def get_adapter_for_model(model: str) -> ProviderAdapter:
    """Return the first adapter whose detect_model() returns True, or raise."""
    _ensure_loaded()
    if _ADAPTERS is None:
        raise RuntimeError("Provider registry failed to initialise")  # should never happen
    for adapter in _ADAPTERS:
        if adapter.detect_model(model):
            return adapter
    raise ValueError(f"No provider adapter found for model '{model}'")


def get_adapter_for_client(client: Any) -> ProviderAdapter:
    """Return the first adapter whose detect_client() returns True, or raise."""
    _ensure_loaded()
    if _ADAPTERS is None:
        raise RuntimeError("Provider registry failed to initialise")  # should never happen
    for adapter in _ADAPTERS:
        if adapter.detect_client(client):
            return adapter
    raise ValueError(f"No provider adapter found for client type '{type(client).__name__}'")
