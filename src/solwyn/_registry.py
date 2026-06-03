"""Provider runtime registry — a sans-I/O holder of configured clients.

Each ``ProviderRuntime`` binds a caller-supplied SDK client to the adapter that
handles it (detected by *type* only, never via a network call) and the
``ProviderEntry`` that describes its model/default params. ``build_runtimes``
turns the primary client plus an ordered list of fallback specs into the
``[primary, *fallbacks]`` chain the router walks.

No business logic, no network, no credentials: ``ProviderEntry`` deliberately
carries no api_key/base_url (Decision D). Provider credentials live only on the
caller's client objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solwyn._types import ProviderEntry, ProviderName
from solwyn.exceptions import ConfigurationError
from solwyn.providers import get_adapter_for_client
from solwyn.providers._protocol import ProviderAdapter


@dataclass(frozen=True)
class ProviderRuntime:
    """A configured provider link: its entry, SDK client, and detected adapter."""

    entry: ProviderEntry
    sdk_client: Any
    adapter: ProviderAdapter


def _runtime_for(client: Any, model: str, default_params: dict[str, Any]) -> ProviderRuntime:
    """Detect *client*'s adapter and build a ProviderRuntime for it (no I/O)."""
    adapter = get_adapter_for_client(client)
    entry = ProviderEntry(
        provider=ProviderName(adapter.name),
        model=model,
        default_params=default_params,
    )
    return ProviderRuntime(entry=entry, sdk_client=client, adapter=adapter)


def _parse_fallback_spec(spec: object) -> tuple[Any, str, dict[str, Any]]:
    """Validate one fallback spec tuple, returning (client, model, default_params).

    Accepts ``(client, model)`` or ``(client, model, default_params)``. Raises
    ConfigurationError with a clear message for any malformed shape. Never
    includes prompt content (specs carry none).
    """
    if not isinstance(spec, tuple) or len(spec) not in (2, 3):
        raise ConfigurationError(
            "Fallback spec must be a (client, model) or "
            "(client, model, default_params) tuple, got "
            f"{type(spec).__name__}",
            field="fallback_specs",
        )

    client = spec[0]
    model = spec[1]
    default_params: dict[str, Any] = spec[2] if len(spec) == 3 else {}

    if not isinstance(model, str):
        raise ConfigurationError(
            f"Fallback model must be a str, got {type(model).__name__}",
            field="fallback_specs",
        )
    if not isinstance(default_params, dict):
        raise ConfigurationError(
            f"Fallback default_params must be a dict, got {type(default_params).__name__}",
            field="fallback_specs",
        )

    return client, model, default_params


def build_runtimes(
    primary_client: Any,
    primary_model: str | None,
    fallback_specs: list[Any],
) -> list[ProviderRuntime]:
    """Build the ordered ``[primary, *fallbacks]`` provider runtime chain.

    Args:
        primary_client: The primary SDK client object. Its adapter is detected
            by type only (no I/O).
        primary_model: Model for the primary entry, or ``None``. The per-call
            model wins for the primary, so ``None`` becomes an ``""`` placeholder.
        fallback_specs: Ordered list of ``(client, model)`` or
            ``(client, model, default_params)`` tuples.

    Returns:
        Runtimes in order ``[primary, *fallbacks]``. The primary entry's
        ``default_params`` is always ``{}`` (global defaults live on config).

    Raises:
        ConfigurationError: If a fallback spec tuple is malformed.
    """
    runtimes = [_runtime_for(primary_client, primary_model or "", {})]

    for spec in fallback_specs:
        client, model, default_params = _parse_fallback_spec(spec)
        runtimes.append(_runtime_for(client, model, default_params))

    return runtimes
