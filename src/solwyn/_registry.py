"""Provider runtime registry — a sans-I/O holder of configured clients.

Each ``ProviderRuntime`` binds a caller-supplied SDK client to the adapter that
handles it (detected by *type* and ``base_url`` only, never via a network call)
and the ``ProviderEntry`` that describes its model/default params.
``build_runtimes`` turns the primary client plus an ordered list of fallback
specs into the ``[primary, *fallbacks]`` chain the router walks.

Provider identity may be OVERRIDDEN explicitly (``provider="groq"`` on the
client constructor, or a 4th element in a fallback spec tuple) for endpoints
auto-detection cannot name — e.g. a vLLM server on a non-default port. An
override may only relabel within the same API dialect; a mismatch is a
loud ConfigurationError.

No business logic, no network, no credentials: ``ProviderEntry`` deliberately
carries no api_key/base_url. Provider credentials live only on the
caller's client objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solwyn._types import ProviderEntry, ProviderName
from solwyn.exceptions import ConfigurationError
from solwyn.providers import get_adapter_by_name, get_adapter_for_client
from solwyn.providers._protocol import ProviderAdapter


@dataclass(frozen=True)
class ProviderRuntime:
    """A configured provider link: its entry, SDK client, and detected adapter."""

    entry: ProviderEntry
    sdk_client: Any
    adapter: ProviderAdapter


def _resolve_adapter(client: Any, provider_override: str | None) -> ProviderAdapter:
    """Detect *client*'s adapter, honoring an explicit provider override.

    Without an override this is pure detection. With one, the named adapter is
    used — but only after verifying it speaks the SAME dialect the client was
    detected as (an override relabels attribution; it cannot make an Anthropic
    client speak the OpenAI wire shape). Fail loud on any mismatch.
    """
    if provider_override is None:
        return get_adapter_for_client(client)

    try:
        adapter = get_adapter_by_name(provider_override)
    except ValueError as exc:
        raise ConfigurationError(str(exc), field="provider") from exc

    try:
        detected = get_adapter_for_client(client)
    except ValueError as exc:
        raise ConfigurationError(
            f"provider override '{provider_override}' was given, but the client "
            f"type '{type(client).__name__}' is not a recognized provider SDK "
            "client (openai / anthropic / google-genai)",
            field="provider",
        ) from exc
    if detected.dialect != adapter.dialect:
        raise ConfigurationError(
            f"provider override '{provider_override}' speaks the "
            f"'{adapter.dialect}' dialect but the client speaks "
            f"'{detected.dialect}' — the override can only relabel a client "
            "within its own dialect",
            field="provider",
        )
    return adapter


def _runtime_for(
    client: Any,
    model: str,
    default_params: dict[str, Any],
    provider_override: str | None = None,
) -> ProviderRuntime:
    """Detect *client*'s adapter and build a ProviderRuntime for it (no I/O)."""
    adapter = _resolve_adapter(client, provider_override)
    entry = ProviderEntry(
        provider=ProviderName(adapter.name),
        model=model,
        default_params=default_params,
    )
    return ProviderRuntime(entry=entry, sdk_client=client, adapter=adapter)


def _parse_fallback_spec(spec: object) -> tuple[Any, str, dict[str, Any], str | None]:
    """Validate one fallback spec tuple.

    Returns ``(client, model, default_params, provider_override)``. Accepts
    ``(client, model)``, ``(client, model, default_params)``, or
    ``(client, model, default_params, provider)``. Raises ConfigurationError
    with a clear message for any malformed shape. Never includes prompt
    content (specs carry none).
    """
    if not isinstance(spec, tuple) or len(spec) not in (2, 3, 4):
        raise ConfigurationError(
            "Fallback spec must be a (client, model), "
            "(client, model, default_params), or "
            "(client, model, default_params, provider) tuple, got "
            f"{type(spec).__name__}",
            field="fallback_specs",
        )

    client = spec[0]
    model = spec[1]
    default_params: dict[str, Any] = spec[2] if len(spec) >= 3 else {}
    provider_override: str | None = spec[3] if len(spec) == 4 else None

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
    if provider_override is not None and not isinstance(provider_override, str):
        raise ConfigurationError(
            f"Fallback provider must be a str, got {type(provider_override).__name__}",
            field="fallback_specs",
        )

    return client, model, default_params, provider_override


def build_runtimes(
    primary_client: Any,
    primary_model: str | None,
    fallback_specs: list[Any],
    *,
    primary_provider: str | None = None,
) -> list[ProviderRuntime]:
    """Build the ordered ``[primary, *fallbacks]`` provider runtime chain.

    Args:
        primary_client: The primary SDK client object. Its adapter is detected
            by type/base_url only (no I/O).
        primary_model: Model for the primary entry, or ``None``. The per-call
            model wins for the primary, so ``None`` becomes an ``""`` placeholder.
        fallback_specs: Ordered list of ``(client, model)``,
            ``(client, model, default_params)``, or
            ``(client, model, default_params, provider)`` tuples.
        primary_provider: Optional explicit provider name for the primary
            client (e.g. ``"vllm"`` for a server auto-detection cannot name).

    Returns:
        Runtimes in order ``[primary, *fallbacks]``. The primary entry's
        ``default_params`` is always ``{}`` (global defaults live on config).

    Raises:
        ConfigurationError: If a fallback spec tuple is malformed, or a
            provider override is unknown / dialect-mismatched.
    """
    runtimes = [_runtime_for(primary_client, primary_model or "", {}, primary_provider)]

    for spec in fallback_specs:
        client, model, default_params, provider_override = _parse_fallback_spec(spec)
        runtimes.append(_runtime_for(client, model, default_params, provider_override))

    return runtimes
