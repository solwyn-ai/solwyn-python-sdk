"""Bedrock provider adapter — token extraction only, no pricing.

Bedrock Converse usage normalization (the AWS-documented formula):

    input_tokens (normalized) = inputTokens + cacheReadInputTokens + cacheWriteInputTokens

The Converse user guide is explicit that ``inputTokens`` covers only the
NON-cached input tokens — the same additive convention as direct Anthropic.
``inputTokens`` / ``outputTokens`` / ``totalTokens`` are contractually required
on every Converse response; the cache fields are optional (absent for model
families without prompt caching).

Cache-write TTL split: ``usage.cacheDetails`` (``[{inputTokens, ttl}]``)
itemizes cache writes by TTL — Claude 4.5-generation models on Bedrock support
a 1-hour tier alongside the default 5-minute TTL. When ``cacheDetails`` is
absent, the aggregate ``cacheWriteInputTokens`` is attributed to the 5m bucket
(Bedrock's default TTL), mirroring the Anthropic adapter's aggregate fallback.

Everything here is DICT access: botocore deserializes Converse responses and
ConverseStream events to plain dicts (no attribute objects), and this module
never imports boto3/botocore — detection is duck-typed on the generated client
shape (``client.meta.service_model.service_name == "bedrock-runtime"``).

Streaming usage arrives in the terminal ``metadata`` event of the
ConverseStream event stream. A stream that errors or is abandoned before that
event settles at zeros with an explicit warning — never silently wrong counts.

reasoning_tokens stays 0 — Converse usage does not break out reasoning tokens
(reasoningContent output is folded into outputTokens; documented blind spot).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal

from solwyn._constants import SERVICE_TIER_MAX_LENGTH
from solwyn._token_details import TokenDetails

logger = logging.getLogger(__name__)

# Model-vendor namespaces observed on Bedrock. detect_model anchors on
# ``[geo.]vendor.`` so any current or future geographic inference-profile
# prefix (us. / eu. / apac. / jp. / au. / global. / us-gov. / ...) matches
# without hardcoding AWS's open-ended prefix set.
_BEDROCK_VENDORS = (
    "ai21",
    "amazon",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "luma",
    "meta",
    "minimax",
    "mistral",
    "moonshot",
    "nvidia",
    "openai",
    "qwen",
    "stability",
    "twelvelabs",
    "writer",
)

_BEDROCK_MODEL_RE = re.compile(
    # Bedrock ARNs (foundation models, inference profiles, prompts). The
    # partition segment follows the documented ``arn:aws(-[^:]+)?:bedrock:``.
    r"^arn:aws(?:-[a-z-]+)?:bedrock:"
    # ``[geo.]vendor.model`` ids, e.g. ``us.anthropic.claude-...`` or
    # ``amazon.nova-pro-v1:0``. The geo prefix is open-ended by design.
    r"|^(?:[a-z]{2,6}(?:-[a-z]{2,6})?\.)?(?:" + "|".join(_BEDROCK_VENDORS) + r")\.\S+"
)


def _count(usage: Mapping[str, Any], key: str) -> int:
    """Read a non-negative int token counter off a usage mapping, else 0.

    Never raises: None, absent, or garbage-typed values all degrade to 0 so
    extraction can never break a user's completed LLM call.
    """
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if value >= 0 else 0


def _split_cache_write(cache_write: int, details: Any) -> tuple[int, int]:
    """Split aggregate cache-write tokens into (5m, 1h) TTL buckets.

    ``cacheDetails`` itemizes writes by TTL when present. The 1h share is
    whatever the details attribute to ``"1h"`` (clamped to the aggregate); the
    remainder stays in the 5m bucket so the total write count is never lost
    even when the breakdown is partial or absent.
    """
    cache_1h = 0
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, Mapping) and detail.get("ttl") == "1h":
                cache_1h += _count(detail, "inputTokens")
    cache_1h = min(cache_1h, cache_write)
    return cache_write - cache_1h, cache_1h


def _extract_bedrock_usage(usage: Any) -> TokenDetails:
    """Extract token usage from a Converse ``usage`` mapping.

    Module-level so both BedrockAdapter and BedrockStreamAccumulator can call
    it. Returns TokenDetails() with all zeros when usage is missing or not a
    mapping. Never raises.
    """
    if not isinstance(usage, Mapping):
        return TokenDetails()

    base_input = _count(usage, "inputTokens")
    output = _count(usage, "outputTokens")
    cache_read = _count(usage, "cacheReadInputTokens")
    cache_write = _count(usage, "cacheWriteInputTokens")
    cache_5m, cache_1h = _split_cache_write(cache_write, usage.get("cacheDetails"))

    return TokenDetails(
        input_tokens=base_input + cache_read + cache_write,
        output_tokens=output,
        cached_input_tokens=cache_read,
        cache_creation_5m_tokens=cache_5m,
        cache_creation_1h_tokens=cache_1h,
    )


def _extract_bedrock_service_tier(carrier: Any) -> str | None:
    """Return the pricing tier from a Converse response / stream metadata event.

    ``serviceTier.type`` (flex/priority-style processing tiers) wins when
    echoed; otherwise ``performanceConfig.latency`` ("optimized" inference
    prices differently, and the RESPONSE echo is the billing ground truth —
    requests over the optimized quota are served and billed as standard).
    """
    if not isinstance(carrier, Mapping):
        return None
    service_tier = carrier.get("serviceTier")
    tier = service_tier.get("type") if isinstance(service_tier, Mapping) else None
    if not isinstance(tier, str):
        performance = carrier.get("performanceConfig")
        tier = performance.get("latency") if isinstance(performance, Mapping) else None
    if not isinstance(tier, str):
        return None
    if len(tier) > SERVICE_TIER_MAX_LENGTH:
        logger.warning(
            "Bedrock service tier exceeds %d characters; truncating",
            SERVICE_TIER_MAX_LENGTH,
        )
        return tier[:SERVICE_TIER_MAX_LENGTH]
    return tier


class BedrockAdapter:
    """Extracts normalized TokenDetails from Bedrock Converse responses."""

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def dialect(self) -> Literal["bedrock"]:
        return "bedrock"

    def detect_client(self, client: Any) -> bool:
        """Return True for a boto3/aioboto3 ``bedrock-runtime`` client.

        Duck-typed — boto3 is never imported. botocore generates client
        classes dynamically in ``botocore.client`` (``aiobotocore.client``
        async), and the wrapped service is identified by
        ``client.meta.service_model.service_name``. The ``bedrock`` control
        plane client deliberately does NOT match: it serves no inference.
        """
        module = getattr(type(client), "__module__", "")
        if "botocore" not in module:
            return False
        service_model = getattr(getattr(client, "meta", None), "service_model", None)
        return getattr(service_model, "service_name", None) == "bedrock-runtime"

    def detect_model(self, model: str) -> bool:
        """Return True for Bedrock model identities.

        Matches ``vendor.model`` ids, geo-prefixed cross-region inference
        profiles (``us.`` / ``eu.`` / ``jp.`` / ``global.`` / ...), and
        Bedrock ARNs (inference profiles, application inference profiles).
        Direct-provider ids (``claude-*``, ``gpt-*``, ``gemini-*``) do not
        match — they belong to the native adapters.
        """
        return _BEDROCK_MODEL_RE.match(model) is not None

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract token usage from a Converse response dict.

        Returns TokenDetails() with all zeros when usage is unavailable.
        Never raises — returns zeros for any unexpected response shape.
        """
        if not isinstance(response, Mapping):
            return TokenDetails()
        return _extract_bedrock_usage(response.get("usage"))

    def estimate_missing_usage(
        self, response: Any, *, estimated_input_tokens: int
    ) -> TokenDetails | None:
        """Converse responses contractually carry usage — no estimated fallback."""
        return None

    def extract_service_tier(self, response: Any) -> str | None:
        """Return the Bedrock pricing tier echoed on the response, or None."""
        return _extract_bedrock_service_tier(response)

    def extract_region(self, client: Any) -> str | None:
        """Return the client's AWS region (Bedrock pricing is per-region)."""
        region = getattr(getattr(client, "meta", None), "region_name", None)
        if isinstance(region, str) and region:
            return region
        return None

    def prepare_streaming(
        self, kwargs: dict[str, Any], *, cross_provider: bool = False
    ) -> dict[str, Any]:
        """ConverseStream always emits the terminal metadata event — no changes."""
        return dict(kwargs)

    def create_stream_accumulator(
        self, *, estimated_input_tokens: int = 0
    ) -> BedrockStreamAccumulator:
        """Converse streams always carry the terminal usage event — the input
        estimate is not needed."""
        return BedrockStreamAccumulator()

    def prepare_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Converse hop: rename the model key back; streaming picks the method.

        The uniform internal ``model`` key renames back to boto3's ``modelId``,
        and Converse takes no ``stream`` kwarg — streaming intent selects the
        dedicated method. timeout/max_retries are unusable: boto3 has no
        ``with_options`` / per-call timeout override, so the chain Deadline
        plus the caller's botocore Config govern this hop.
        """
        kwargs = dict(kwargs)
        kwargs["modelId"] = kwargs.pop("model")
        kwargs.pop("stream", None)
        if is_streaming:
            return client.converse_stream, kwargs
        return client.converse, kwargs

    def unwrap_stream_source(self, response: Any) -> Any:
        """ConverseStream nests the iterable: wrap the INNER event stream.

        ``response["stream"]`` is the EventStream whose close() releases the
        connection — that is what the stream wrapper must own, not the dict.
        """
        return response["stream"]

    def wrap_stream_result(self, wrapper: Any, served_response: Any) -> Any:
        """Bedrock-dialect callers iterate ``result["stream"]``.

        Return the boto3 contract shape with the wrapped stream swapped in.
        On a cross-provider hop the served response is a foreign stream
        object, not a Converse dict — there is nothing to preserve.
        """
        if isinstance(served_response, Mapping):
            return {**served_response, "stream": wrapper}
        return {"stream": wrapper}


class BedrockStreamAccumulator:
    """Accumulates usage from ConverseStream events.

    Events are parsed dicts keyed by event type (``{"messageStart": ...}``,
    ``{"contentBlockDelta": ...}``, ``{"metadata": ...}``). Usage and the
    pricing tier arrive in the terminal ``metadata`` event (documented as the
    last event of the stream). Only the usage/tier values are retained — never
    content-bearing events.
    """

    def __init__(self) -> None:
        self._usage: Mapping[str, Any] | None = None
        self._service_tier: str | None = None
        self._saw_event = False

    def observe(self, chunk: Any) -> None:
        if not isinstance(chunk, Mapping):
            return
        # Any well-formed event marks stream activity. Setting this for metadata
        # events too (not only the non-metadata branch) lets finalize() warn when
        # the terminal metadata event arrives WITHOUT usable usage, instead of
        # silently settling at zero — preserving the "never silently wrong" count.
        self._saw_event = True
        metadata = chunk.get("metadata")
        if isinstance(metadata, Mapping):
            usage = metadata.get("usage")
            if isinstance(usage, Mapping):
                self._usage = usage
            tier = _extract_bedrock_service_tier(metadata)
            if tier is not None:
                self._service_tier = tier

    def finalize(self) -> TokenDetails:
        if self._usage is None:
            if self._saw_event:
                logger.warning(
                    "Bedrock stream settled at zero tokens: no usage in the "
                    "terminal metadata event (stream abandoned, or the metadata "
                    "event carried no usage)"
                )
            return TokenDetails()
        return _extract_bedrock_usage(self._usage)

    def get_service_tier(self) -> str | None:
        """Return the tier observed in the stream's metadata event, or None."""
        return self._service_tier
