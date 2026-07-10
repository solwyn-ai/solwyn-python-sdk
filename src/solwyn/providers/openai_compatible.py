"""OpenAI-compatible provider adapters — token extraction only, no pricing.

One adapter CLASS, many provider IDENTITIES: every provider here speaks the
OpenAI Chat Completions dialect (``dialect == "openai"``) but is a distinct
provider for attribution, budgets, pricing, and circuit breaking. The
differences between them are pure data, captured in a ``CompatProfile``:

- which base_url hosts / local ports identify the provider,
- which model-name prefixes are unambiguous,
- whether ``stream_options={"include_usage": True}`` is safe to send.

THE COST PATH IS THE COMPATIBILITY SURFACE THAT MATTERS. A compat endpoint can
return a perfect completion and still omit usage — especially when streaming.
The accumulator therefore has three tiers:

1. standard ``usage`` block (the last chunk whose usage parses to non-zero
   counts wins; zeroed placeholders never latch),
2. Groq's legacy ``x_groq.usage`` final-chunk field,
3. explicit length-based estimation (input = pre-call estimate, output =
   accumulated delta character lengths via the privacy module), marked
   ``TokenDetails(is_estimated=True)`` and logged at WARNING — degradation is
   loud and flagged on the wire, never silently zero.

Privacy: this module never logs, stores, or concatenates prompt/response
content. Length measurement is delegated to ``solwyn._privacy`` (the
content-privileged module) which returns irreversible integer counts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from solwyn._privacy import (
    estimate_response_content_length,
    estimate_stream_chunk_content_length,
    estimate_tokens_from_length,
)
from solwyn._token_details import TokenDetails
from solwyn.exceptions import UnsupportedSurfaceError
from solwyn.providers.openai import _extract_openai_usage, _extract_service_tier

logger = logging.getLogger(__name__)


def _is_openai_host(host: str) -> bool:
    """True for OpenAI's own hosts — the generic catch-all must never claim
    them (an openai-module client pointed here belongs to the OpenAIAdapter).

    Covers the regional data-residency endpoints (eu.api.openai.com,
    us.api.openai.com) via the suffix match.
    """
    return host == "api.openai.com" or host.endswith(".api.openai.com")


# Hosts treated as "local" for port-based detection of local inference servers.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})


@dataclass(frozen=True)
class CompatProfile:
    """Static description of one OpenAI-compatible provider.

    ``name`` must be a valid ``ProviderName`` value — it is the attribution
    identity sent on the wire. Detection fields are matched against the
    client's ``base_url``; ``client_class_prefixes`` additionally matches the
    client's class name (Azure's dedicated client classes).
    """

    name: str
    hosts: tuple[str, ...] = ()
    host_suffixes: tuple[str, ...] = ()
    local_ports: tuple[int, ...] = ()
    model_prefixes: tuple[str, ...] = ()
    client_class_prefixes: tuple[str, ...] = ()
    # Whether stream_options={"include_usage": True} is safe for this provider.
    # False for providers with strict request validation that reject unknown
    # params (their streams must carry usage another way, or fall back to
    # estimation). When False, prepare_streaming never injects it, and on a
    # CROSS-PROVIDER failover hop it also strips a caller-supplied
    # stream_options (meant for the original target) so the hop does not 4xx.
    supports_include_usage: bool = True
    # The generic fallback profile: claims any openai-module client whose
    # base_url is a real http(s) URL pointing at neither OpenAI nor a host
    # matched by an earlier named profile (registry order enforces "earlier").
    catch_all: bool = field(default=False, kw_only=True)

    def matches_url(self, scheme: str, host: str, port: int | None) -> bool:
        """Return True if a parsed base_url belongs to this provider."""
        if self.catch_all:
            return scheme in ("http", "https") and not _is_openai_host(host)
        if host in self.hosts:
            return True
        if any(host.endswith(suffix) for suffix in self.host_suffixes):
            return True
        return bool(self.local_ports) and host in _LOCAL_HOSTS and port in self.local_ports

    def matched_by_local_port(self, host: str) -> bool:
        """True when this profile identifies providers by local-port heuristic."""
        return bool(self.local_ports) and host in _LOCAL_HOSTS


# Detection order matters and is preserved by the registry: named profiles
# first (most-specific hosts), the catch-all last, and the plain OpenAIAdapter
# after all of these.
#
# supports_include_usage follows each provider's verified behavior (July 2026):
# the ecosystem-wide failure mode is strict request validation rejecting
# stream_options with a 4xx, so the flag is True ONLY where documented-safe or
# successfully live-probed.
# Providers that auto-include usage in the final stream chunk don't need it;
# anything that ends up usage-less falls back to flagged estimation.
COMPAT_PROFILES: tuple[CompatProfile, ...] = (
    # xAI ERRORS on stream_options ("Argument not supported") and instead
    # always emits usage automatically in a final usage-only chunk.
    CompatProfile(
        name="xai",
        hosts=("api.x.ai",),
        model_prefixes=("grok-",),
        supports_include_usage=False,
    ),
    # DeepSeek: include_usage documented and required for streaming usage.
    CompatProfile(name="deepseek", hosts=("api.deepseek.com",), model_prefixes=("deepseek-",)),
    # Mistral La Plateforme validates requests strictly — stream_options has
    # returned 422 "Extra inputs are not permitted". Never send it; rely on
    # final-chunk usage or the estimation fallback.
    CompatProfile(
        name="mistral",
        hosts=("api.mistral.ai",),
        model_prefixes=(
            "mistral-",
            "ministral-",
            "codestral-",
            "magistral-",
            "pixtral-",
            "devstral-",
            "open-mistral-",
            "open-mixtral-",
        ),
        supports_include_usage=False,
    ),
    # Qwen DashScope compatible-mode: include_usage documented and required.
    CompatProfile(
        name="qwen",
        hosts=(
            "dashscope.aliyuncs.com",
            "dashscope-intl.aliyuncs.com",
            "dashscope-us.aliyuncs.com",
        ),
        model_prefixes=("qwen", "qwq-", "qvq-"),
    ),
    # Z.ai: live probe verified include_usage support on 2026-07-09.
    CompatProfile(
        name="zai",
        hosts=("api.z.ai",),
        model_prefixes=("glm-",),
        supports_include_usage=True,
    ),
    # Groq: include_usage documented; legacy x_groq.usage handled by the
    # accumulator for older response formats.
    CompatProfile(name="groq", hosts=("api.groq.com",)),
    # Together: live probe 2026-07-10 (SDK 2.22.1) verified terminal-chunk usage
    # for sync and async streams. It also verified nested (GLM-5.2) and flat
    # (Llama 3.3 70B) cached-token shapes, both as prompt-token subsets.
    # stream_options remains undocumented — don't send (estimation nets any gap).
    CompatProfile(
        name="together",
        hosts=("api.together.xyz", "api.together.ai"),
        supports_include_usage=False,
    ),
    # Fireworks: usage always auto-included in the final chunk; stream_options
    # undocumented — don't send.
    CompatProfile(
        name="fireworks",
        hosts=("api.fireworks.ai",),
        model_prefixes=("accounts/fireworks/",),
        supports_include_usage=False,
    ),
    # Perplexity: stream_options is not a documented param and validation is
    # strict — never send. Usage rides on streamed chunks.
    CompatProfile(
        name="perplexity",
        hosts=("api.perplexity.ai",),
        model_prefixes=("sonar",),
        supports_include_usage=False,
    ),
    # Azure OpenAI: include_usage supported from API version 2024-06-01.
    # (Caveat handled in prepare_streaming: the "on your data" data_sources
    # pipeline rejects stream_options with a 422.)
    CompatProfile(
        name="azure_openai",
        host_suffixes=(".openai.azure.com", ".cognitiveservices.azure.com"),
        client_class_prefixes=("AzureOpenAI", "AsyncAzureOpenAI"),
    ),
    # OpenRouter: usage is always-on in the final chunk and stream_options is
    # deprecated with no effect — don't send.
    CompatProfile(
        name="openrouter",
        hosts=("openrouter.ai",),
        supports_include_usage=False,
    ),
    # Local inference servers: detected by their conventional default ports on
    # a local host. A non-default port still works via the generic catch-all,
    # or precisely via the ``provider=`` override. All three document
    # include_usage support on current versions; OLD versions accept-but-ignore
    # it (Ollama pre-0.3.x, LM Studio pre-0.3.18) and the estimation fallback
    # nets the gap.
    CompatProfile(name="ollama", local_ports=(11434,)),
    CompatProfile(name="vllm", local_ports=(8000,)),
    CompatProfile(name="lmstudio", local_ports=(1234,)),
    # Generic catch-all: per ecosystem best practice, inject stream_options
    # ONLY where known-safe — an unknown endpoint may 4xx on it. Auto-included
    # final-chunk usage (common) or flagged estimation covers the rest.
    CompatProfile(name="openai_compatible", catch_all=True, supports_include_usage=False),
)


def _client_base_url_parts(client: Any) -> tuple[str, str, int | None] | None:
    """Parse (scheme, host, port) from a client's base_url, or None.

    Total by construction: detection runs on arbitrary caller objects (and
    spec'd mocks in tests), so any unreadable/unparseable base_url returns
    None rather than raising.
    """
    try:
        base_url = getattr(client, "base_url", None)
        if base_url is None:
            return None
        parsed = urlsplit(str(base_url))
        host = parsed.hostname
        if not host:
            return None
        return parsed.scheme.lower(), host.lower(), parsed.port
    except Exception:
        return None


def _uses_azure_data_sources(kwargs: dict[str, Any]) -> bool:
    """True when the request uses Azure's "on your data" RAG pipeline.

    ``data_sources`` may be passed top-level or via the OpenAI SDK's
    ``extra_body`` escape hatch. Key-presence check only — values are never
    read.
    """
    if kwargs.get("data_sources") is not None:
        return True
    extra_body = kwargs.get("extra_body")
    return isinstance(extra_body, dict) and extra_body.get("data_sources") is not None


def _usage_int(usage: Any, key: str) -> int:
    """Read a non-negative int usage field from an attr-object OR a plain dict.

    The OpenAI SDK preserves unknown response fields (e.g. Groq's ``x_groq``)
    as raw dicts rather than model objects, so both shapes occur in the wild.
    Negative values are garbage (TokenDetails fields are ``ge=0``) and degrade
    to 0 — missing — so the estimation fallback takes over.
    """
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


class OpenAICompatibleAdapter:
    """Extraction adapter for one OpenAI-compatible provider profile."""

    def __init__(self, profile: CompatProfile) -> None:
        self._profile = profile
        # Log-once latches. Adapter instances are registry singletons; a rare
        # double-log under concurrency is acceptable.
        self._warned_missing_usage = False
        self._noted_port_heuristic = False

    @property
    def name(self) -> str:
        return self._profile.name

    @property
    def dialect(self) -> Literal["openai"]:
        return "openai"

    @property
    def profile(self) -> CompatProfile:
        return self._profile

    def detect_client(self, client: Any) -> bool:
        """Match an openai-SDK client whose base_url/class targets this provider."""
        if "openai" not in getattr(type(client), "__module__", ""):
            return False
        class_name = getattr(type(client), "__name__", "")
        if any(class_name.startswith(prefix) for prefix in self._profile.client_class_prefixes):
            return True
        parts = _client_base_url_parts(client)
        if parts is None:
            return False
        matched = self._profile.matches_url(*parts)
        if (
            matched
            and self._profile.matched_by_local_port(parts[1])
            and not self._noted_port_heuristic
        ):
            # The identity was GUESSED from a conventional local port — say so
            # once (provider name only; never the URL, which may carry creds).
            self._noted_port_heuristic = True
            logger.info(
                "Detected a local OpenAI-compatible server on its conventional "
                "port as provider '%s'; pass provider=... to Solwyn to override.",
                self._profile.name,
            )
        return matched

    def detect_model(self, model: str) -> bool:
        """Match only this provider's unambiguous model-name prefixes.

        Profiles for providers that serve shared open-weight catalogs
        (Groq/Together/OpenRouter/local servers) carry no prefixes — a bare
        ``llama-3-8b`` stays ambiguous and is never silently claimed.
        """
        return any(model.startswith(prefix) for prefix in self._profile.model_prefixes)

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract token usage from a Chat Completions-shaped response.

        Never raises (protocol contract): value-level garbage from an
        arbitrary compat endpoint degrades to zeros, which the estimation
        fallback then treats as a usage gap.
        """
        try:
            return _extract_openai_usage(response)
        except Exception:
            return TokenDetails()

    def estimate_missing_usage(
        self, response: Any, *, estimated_input_tokens: int
    ) -> TokenDetails | None:
        """Length-based estimate when a response carries no USABLE usage.

        Usable means: a usage block that parses to non-zero counts, OR an
        all-zero block on a genuinely empty response (provider truth). A
        missing block — or one in a foreign shape / zeroed-out while the
        response visibly carries content — falls through to estimation, so a
        misbehaving endpoint can never settle real output as silent zero
        spend. The estimate is explicitly marked ``is_estimated=True`` and
        logged once at WARNING.
        """
        output_chars = estimate_response_content_length(response)
        try:
            if getattr(response, "usage", None) is not None:
                extracted = _extract_openai_usage(response)
                if extracted.input_tokens or extracted.output_tokens:
                    return None  # provider-reported usage — never overridden
                if not output_chars:
                    return None  # zero usage on an empty response — provider truth
        except Exception:
            pass  # unreadable/garbage usage values — fall through to estimation
        self._warn_missing_usage()
        return TokenDetails(
            input_tokens=estimated_input_tokens,
            output_tokens=(
                estimate_tokens_from_length(output_chars, provider="openai") if output_chars else 0
            ),
            is_estimated=True,
        )

    def extract_service_tier(self, response: Any) -> str | None:
        """Bounded service_tier passthrough (some compat providers expose one)."""
        return _extract_service_tier(response)

    def prepare_streaming(
        self, kwargs: dict[str, Any], *, cross_provider: bool = False
    ) -> dict[str, Any]:
        """Opt in to streaming usage where the provider supports it.

        For providers with strict request validation
        (``supports_include_usage=False``) Solwyn never INJECTS stream_options.
        On a CROSS-PROVIDER failover hop it additionally STRIPS a caller-
        supplied stream_options (it was meant for the original target and
        would 4xx here); on the caller's own configured target the explicit
        option is passed through untouched — a drop-in wrapper does not mutate
        parameters the caller deliberately chose for their own endpoint.
        """
        kwargs = dict(kwargs)
        if not self._profile.supports_include_usage:
            if cross_provider:
                kwargs.pop("stream_options", None)
            return kwargs
        if self._profile.name == "azure_openai" and _uses_azure_data_sources(kwargs):
            # Azure "on your data" rejects stream_options with a 422 — never
            # INJECT include_usage here (the estimation fallback nets the
            # missing usage). Azure-only: another profile whose extra_body
            # happens to carry a data_sources key keeps normal injection.
            # Mirrors the supports_include_usage=False branch above: a
            # caller-supplied stream_options is stripped only on a
            # cross-provider hop; on the caller's own Azure target it passes
            # through untouched (drop-in contract).
            if cross_provider:
                kwargs.pop("stream_options", None)
            return kwargs
        stream_options = dict(kwargs.get("stream_options") or {})
        stream_options["include_usage"] = True
        kwargs["stream_options"] = stream_options
        return kwargs

    def create_stream_accumulator(
        self, *, estimated_input_tokens: int = 0
    ) -> CompatStreamAccumulator:
        return CompatStreamAccumulator(adapter=self, estimated_input_tokens=estimated_input_tokens)

    def extract_region(self, client: Any) -> str | None:
        """Compat providers carry no per-region pricing contract."""
        return None

    def prepare_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Chat Completions hop (openai dialect): streaming rides ``stream=True``.

        timeout/max_retries are ignored — the dispatcher already applied them
        via the openai SDK's ``with_options``.
        """
        kwargs = dict(kwargs)
        if is_streaming:
            kwargs["stream"] = True
        return client.chat.completions.create, kwargs

    def prepare_media_call(
        self,
        surface: str,
        client: Any,
        kwargs: dict[str, Any],
        *,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Per-surface dispatch seam for non-chat media surfaces.

        FOUNDATION: no media surface is wired yet, so every surface raises
        ``UnsupportedSurfaceError``. Later batches add branches here (P1.7 wires
        ``client.embeddings.create`` for this adapter — one branch covers all 14
        compat profiles since they share the openai dialect), each returning
        ``(method, shaped_kwargs)`` exactly as ``prepare_call`` does for chat.
        timeout/max_retries are ignored — the dispatcher already applied them
        via the openai SDK's ``with_options``.
        """
        raise UnsupportedSurfaceError(surface=surface, provider=self.name)

    def unwrap_stream_source(self, response: Any) -> Any:
        """The streaming call returns the iterable itself."""
        return response

    def wrap_stream_result(self, wrapper: Any, served_response: Any) -> Any:
        """OpenAI-dialect callers iterate the stream object directly."""
        return wrapper

    def _warn_missing_usage(self) -> None:
        """One structural WARNING per adapter when estimation kicks in."""
        if self._warned_missing_usage:
            return
        self._warned_missing_usage = True
        logger.warning(
            "Provider '%s' returned no usage data; Solwyn is reporting "
            "length-based ESTIMATED token counts (marked is_estimated=true). "
            "Budgets still enforce, but per-call costs are approximate.",
            self._profile.name,
        )


class CompatStreamAccumulator:
    """Accumulates usage from OpenAI-dialect streaming chunks, with fallbacks.

    Tier 1: the standard ``usage`` block — the LAST chunk whose usage PARSES
    to non-zero counts wins (some providers attach usage to every chunk, most
    only to the final one; zeroed placeholder blocks never latch).
    Tier 2: Groq's legacy ``x_groq.usage`` final-chunk field.
    Tier 3: explicit estimation — input from the pre-call length estimate,
    output from accumulated delta character lengths (length-only, measured by
    the privacy module). Marked ``is_estimated=True`` and logged at WARNING.

    Privacy: usage and service_tier are EXTRACTED at observe time — the chunk
    object itself (whose deltas carry response content on providers that ride
    usage on content chunks) is never retained.
    """

    def __init__(self, *, adapter: OpenAICompatibleAdapter, estimated_input_tokens: int) -> None:
        self._adapter = adapter
        self._estimated_input = estimated_input_tokens
        self._usage_details: TokenDetails | None = None
        self._x_groq_details: TokenDetails | None = None
        self._service_tier: str | None = None
        self._content_chars = 0

    def observe(self, chunk: Any) -> None:
        # Best-effort extraction: value-level garbage from an arbitrary
        # endpoint must never raise out of observe — the stream wrapper would
        # settle a deliverable stream as a provider FAILURE (breaker hit
        # against a content-healthy provider). A failed tier degrades to
        # estimation; content-length accumulation below always runs.
        try:
            if getattr(chunk, "usage", None) is not None:
                extracted = _extract_openai_usage(chunk)
                if extracted.input_tokens or extracted.output_tokens:
                    self._usage_details = extracted
            tier = _extract_service_tier(chunk)
            if tier is not None:
                self._service_tier = tier
            x_groq = getattr(chunk, "x_groq", None)
            if x_groq is not None:
                x_usage = (
                    x_groq.get("usage")
                    if isinstance(x_groq, dict)
                    else getattr(x_groq, "usage", None)
                )
                if x_usage is not None:
                    x_details = TokenDetails(
                        input_tokens=_usage_int(x_usage, "prompt_tokens"),
                        output_tokens=_usage_int(x_usage, "completion_tokens"),
                    )
                    if x_details.input_tokens or x_details.output_tokens:
                        self._x_groq_details = x_details
        except Exception:
            pass  # unreadable/garbage usage values — estimation tier covers it
        self._content_chars += estimate_stream_chunk_content_length(chunk)

    def finalize(self) -> TokenDetails:
        if self._usage_details is not None:
            return self._usage_details
        if self._x_groq_details is not None:
            return self._x_groq_details
        self._adapter._warn_missing_usage()
        return TokenDetails(
            input_tokens=self._estimated_input,
            output_tokens=(
                estimate_tokens_from_length(self._content_chars, provider="openai")
                if self._content_chars
                else 0
            ),
            is_estimated=True,
        )

    def get_service_tier(self) -> str | None:
        """The last bounded service_tier observed on any chunk, when present."""
        return self._service_tier


def build_compat_adapters() -> list[OpenAICompatibleAdapter]:
    """One adapter instance per profile, in detection-priority order."""
    from solwyn.providers.together import TogetherAdapter

    return [
        TogetherAdapter() if profile.name == "together" else OpenAICompatibleAdapter(profile)
        for profile in COMPAT_PROFILES
    ]
