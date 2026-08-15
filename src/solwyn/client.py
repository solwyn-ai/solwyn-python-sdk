"""Solwyn (sync) and AsyncSolwyn (async) client wrappers.

Drop-in wrappers for openai.OpenAI / anthropic.Anthropic that add
budget enforcement, circuit breaking, and metadata reporting.

Usage::

    from openai import OpenAI
    from solwyn import Solwyn

    client = Solwyn(
        OpenAI(),
        api_key="sk_proj_...",
    )
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping
from typing import Any, Literal, NamedTuple, cast

import httpx
from pydantic import ValidationError

from solwyn._base import (
    MediaSurfaceSpec,
    _AttemptContext,
    _client_shape,
    _effective_output_bound,
    _normalized_openai_output_cap_layer,
    _responses_output_bound,
    _SolwynBase,
)
from solwyn._privacy import (
    estimate_content_length,
    estimate_responses_content_length,
    estimate_tokens_from_length,
    merge_google_generate_content_kwargs,
    normalize_google_translation_source_kwargs,
    normalize_legacy_google_generate_content_args,
    prepare_legacy_google_metering_kwargs,
    prepare_legacy_google_translation_source_kwargs,
)
from solwyn._proxies import (
    _AsyncAudioProxy,
    _AsyncChatProxy,
    _AsyncEmbeddingsProxy,
    _AsyncImagesProxy,
    _AsyncMessagesProxy,
    _AsyncModelsProxy,
    _AsyncResponsesProxy,
    _AsyncVideosProxy,
    _bedrock_internal_kwargs,
    _reject_responses_background,
    _SyncAudioProxy,
    _SyncChatProxy,
    _SyncEmbeddingsProxy,
    _SyncImagesProxy,
    _SyncMessagesProxy,
    _SyncModelsProxy,
    _SyncResponsesProxy,
    _SyncVideosProxy,
)
from solwyn._registry import ProviderRuntime, build_runtimes
from solwyn._routing import RoutingRequest, SelectionPolicy
from solwyn._run import _capture_run_context, _RunContextSnapshot
from solwyn._surfaces import SurfaceSource
from solwyn._token_details import TokenDetails
from solwyn._types import CallStatus, FailoverReason, ProviderName
from solwyn.budget import (
    DEFAULT_COST_PER_TOKEN,
    AsyncBudgetEnforcer,
    BudgetCheckResult,
    BudgetEnforcer,
)
from solwyn.config import SolwynConfig
from solwyn.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    ProviderUnavailableError,
    RunStoppedError,
    UnsupportedSurfaceError,
    UntranslatableModelError,
    UntranslatableRequestError,
)
from solwyn.providers import _translation
from solwyn.providers._errors import Disposition, classify_exception, retry_after_seconds
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter
from solwyn.stream import AsyncStreamWrapper, SyncStreamWrapper

logger = logging.getLogger(__name__)


def _budget_denial_error(
    *,
    budget: BudgetCheckResult,
    agent_run_id: str | None,
    estimated_cost: float,
) -> BudgetExceededError:
    """Build the public typed error for one denied pre-flight."""
    budget_period = getattr(budget, "denied_by_period", None)
    if budget_period is None:
        budget_period = "unknown"
    if budget_period == "run_stopped" and agent_run_id is not None:
        return RunStoppedError(
            agent_run_id=agent_run_id,
            project_id=budget.project_id,
            budget_limit=budget.budget_limit,
            current_usage=budget.current_usage,
            estimated_cost=estimated_cost,
            mode=budget.mode.value,
        )
    return BudgetExceededError(
        project_id=budget.project_id,
        budget_limit=budget.budget_limit,
        current_usage=budget.current_usage,
        estimated_cost=estimated_cost,
        budget_period=budget_period,
        mode=budget.mode.value,
    )


# Floor for a per-hop dispatch timeout: even when the chain deadline is nearly
# spent we give a hop at least this long rather than passing it ~0s.
_MIN_HOP_TIMEOUT = 1.0
_CHAT_ONLY_DEFAULT_KEYS = frozenset({"max_tokens", "max_completion_tokens", "stream_options"})
_SOURCE_COMPATIBLE_DEFAULT_KEYS = {
    ProviderName.OPENAI.value: frozenset(
        {
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "stop",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        }
    ),
    ProviderName.ANTHROPIC.value: frozenset(
        {
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "stop_sequences",
            "stream",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        }
    ),
    ProviderName.GOOGLE.value: frozenset({"config", "max_output_tokens", "stream"}),
    ProviderName.BEDROCK.value: frozenset({"system", "inferenceConfig", "toolConfig", "stream"}),
}

# Per-call transport params the openai SDK forwards verbatim onto the HTTP
# request. They are ENDPOINT-SCOPED: extra_headers routinely carries gateway /
# observability credentials (e.g. Helicone-Auth, x-portkey-api-key) authored
# for the caller's own target, and extra_query/extra_body carry vendor
# extensions. Params authored for the original target must never reach a
# DIFFERENT vendor on a failover hop — the cross-dialect path excludes them by
# reconstructing from the canonical subset; the same-dialect passthrough
# strips them explicitly.
_ENDPOINT_SCOPED_KEYS = frozenset({"extra_headers", "extra_query", "extra_body"})

# Why invoke_model fails loud instead of passing through untracked: it is a
# PRIMARY completion surface (unlike, say, embeddings), so silent pass-through
# would be a budget bypass; and its usage lives inside a consume-once response
# body that also carries response CONTENT, so extracting it would require
# buffering customer content — off the table by architecture.
_INVOKE_MODEL_GUIDANCE = (
    "Solwyn intercepts Bedrock through the Converse API only — use converse() / "
    "converse_stream(), which work across all Bedrock chat models. invoke_model "
    "responses carry usage inside a consume-once body, so Solwyn cannot budget-track "
    "them without buffering response content. To make untracked calls anyway, use "
    "the unwrapped boto3 client directly."
)

# Why start_async_invoke fails loud, joining invoke_model: it is a PRIMARY spend
# surface (Bedrock's asynchronous, video-scale invocation path — e.g. Nova Reel),
# so silent pass-through would be a budget bypass at the highest per-call cost of
# any surface. Unlike invoke_model the usage is not even in-band — the call returns
# only an invocationArn and the output/usage land later in an S3 object the SDK
# never sees — so there is nothing to extract without out-of-band polling. Failing
# loud (rather than passing through untracked) closes that silent hole.
_START_ASYNC_INVOKE_GUIDANCE = (
    "Solwyn does not budget-track Bedrock's asynchronous invocation surface "
    "(start_async_invoke) — the call returns only an invocationArn and its usage "
    "lands out-of-band in S3, so Solwyn can neither enforce a budget nor emit a "
    "cost event for it. Rather than pass this video-scale spend through untracked, "
    "Solwyn refuses it. To make untracked async-invoke calls anyway, use the "
    "unwrapped boto3 client directly."
)


def _effective_responses_kwargs(
    *,
    global_defaults: Mapping[str, object],
    primary_defaults: Mapping[str, object],
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Build the one Responses request view used by preflight and dispatch.

    Only defaults are filtered: caller kwargs remain authoritative and pass
    through unchanged.  Responses v1 is primary-only, so the primary entry's
    defaults are the complete per-runtime layer.
    """
    defaults = {
        key: value
        for key, value in {**global_defaults, **primary_defaults}.items()
        if key != "solwyn_tags" and key not in _CHAT_ONLY_DEFAULT_KEYS
    }
    return defaults | dict(kwargs)


_RESPONSES_PARSE_STREAM_GUIDANCE = (
    "OpenAI responses.parse is non-streaming and cannot meter a streaming response. "
    "Use responses.create(stream=True) for metered streaming calls."
)


def _responses_is_streaming(
    kwargs: Mapping[str, object],
    *,
    leaf: str,
    force_stream: bool = False,
) -> bool:
    """Resolve Responses streaming mode and refuse it for the parse leaf.

    OpenAI merges ``extra_body`` into the request body after named parameters,
    so an explicit structural ``stream`` entry there has final precedence for
    parse. Create deliberately retains its established top-level semantics.
    """
    if leaf != "parse":
        return bool(kwargs.get("stream", False)) or force_stream

    extra_body = kwargs.get("extra_body")
    stream = (
        extra_body["stream"]
        if isinstance(extra_body, Mapping) and "stream" in extra_body
        else kwargs.get("stream", False)
    )
    if bool(stream) or force_stream:
        raise ConfigurationError(_RESPONSES_PARSE_STREAM_GUIDANCE, field="stream")
    return False


def _source_compatible_defaults(dialect: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return target defaults that are also legal in the source DIALECT.

    Keyed by dialect, not provider name: every OpenAI-compatible provider
    shares the ``openai`` key set.
    """
    allowed = _SOURCE_COMPATIBLE_DEFAULT_KEYS.get(dialect, frozenset())
    return {key: value for key, value in params.items() if key in allowed}


def _budget_timeout(deadline: Deadline, check_timeout: float) -> float:
    """Timeout for the budget pre-flight, clamped by the chain deadline."""
    return max(0.001, min(check_timeout, deadline.remaining()))


def _settlement_keys(budget: Any) -> tuple[str | None, str | None, int | None]:
    """The wire settlement keys plus the process-local lease claim capability.

    Exactly one is ever set: a lease-funded admission carries no reservation.
    Lease fields are read defensively — pre-lease budget doubles in tests (and
    any caller-supplied result object) may not carry them.
    """
    token = getattr(budget, "lease_claim_token", None)
    return (
        getattr(budget, "reservation_id", None),
        getattr(budget, "lease_id", None),
        token if isinstance(token, int) and not isinstance(token, bool) else None,
    )


def _lease_claim_token(budget: Any) -> int | None:
    """Return the exact local reservation capability, never a mock sentinel."""
    token = getattr(budget, "lease_claim_token", None)
    return token if isinstance(token, int) and not isinstance(token, bool) else None


def _safe_extract_region(runtime: ProviderRuntime) -> str | None:
    """Fail-soft region read (R5): bookkeeping must never raise into the caller."""
    try:
        return runtime.adapter.extract_region(runtime.sdk_client)
    except Exception as exc:
        logger.warning("settlement.extract_region_failed_fail_soft: %s", type(exc).__name__)
        return None


def _safe_extract_service_tier(runtime: ProviderRuntime, response: Any) -> str | None:
    """Fail-soft tier read from the RAW served response (R5)."""
    try:
        return runtime.adapter.extract_service_tier(response)
    except Exception as exc:
        logger.warning("settlement.extract_service_tier_failed_fail_soft: %s", type(exc).__name__)
        return None


class _FailSoftUsage(NamedTuple):
    """Settlement usage plus whether ANY of it was actually measured.

    ``unmeasured`` is the synthetic bottom of the ladder: no adapter read
    succeeded, so ``token_details`` is a pre-flight input estimate with no
    output term. It is deliberately NOT a field on ``TokenDetails`` — that is
    a wire model, and this distinction is local bookkeeping about the read,
    not a new fact for the API to parse.
    """

    token_details: TokenDetails
    unmeasured: bool


def _extract_usage_fail_soft(
    runtime: ProviderRuntime, response: Any, *, estimated_input_tokens: int
) -> _FailSoftUsage:
    """Usage for settlement, degrading to estimates instead of raising (R5).

    Ladder: provider-reported usage -> adapter estimate -> synthetic
    length-based estimate. An adapter raise on an unexpected response shape
    must never destroy a paid, successful response — the worst case is
    estimated spend telemetry (is_estimated=True), which the API already
    prices distinctly.

    The synthetic tier is flagged ``unmeasured``: it carries no output term at
    all, so a lease-funded call must settle its local reservation at the
    reserved bound rather than true it up to this under-measure.
    """
    token_details: TokenDetails | None = None
    try:
        token_details = runtime.adapter.extract_usage(response)
    except Exception as exc:
        logger.warning("settlement.extract_usage_failed_fail_soft: %s", type(exc).__name__)
    # Explicit-degradation fallback (pre-existing semantics): a non-None
    # estimate REPLACES the extracted details.
    estimated: TokenDetails | None = None
    try:
        estimated = runtime.adapter.estimate_missing_usage(
            response, estimated_input_tokens=estimated_input_tokens
        )
    except Exception as exc:
        logger.warning("settlement.estimate_usage_failed_fail_soft: %s", type(exc).__name__)
    if estimated is not None:
        token_details = estimated
    if token_details is None:
        return _FailSoftUsage(
            TokenDetails(input_tokens=estimated_input_tokens, output_tokens=0, is_estimated=True),
            unmeasured=True,
        )
    return _FailSoftUsage(token_details, unmeasured=False)


def _hop_connect_slice(deadline: Deadline, remaining_candidates: int) -> float:
    """CONNECT/POOL slice for one hop, never exceeding the remaining failover
    window. Pre-send hangs are failover-safe, so they must stay inside the
    window; the hop's READ is deliberately NOT derived from the deadline
    (see _hop_httpx_timeout)."""
    remaining = deadline.remaining()
    if remaining <= 0:
        return 0.001
    slice_timeout = remaining / max(1, remaining_candidates)
    return min(remaining, max(_MIN_HOP_TIMEOUT, slice_timeout))


def _hop_httpx_timeout(connect_slice: float, read_timeout: float) -> httpx.Timeout:
    """Granular per-hop bound for with_options clients (PJ-8/R7).

    connect/pool take the deadline-derived slice: a pre-send hang is provably
    failover-safe, so it must fail inside the failover window. read/write take
    the decoupled hop read bound: a read timeout is POST_SEND_AMBIGUOUS and
    re-raises under default idempotency - cutting it at the failover window
    buys no failover, it only converts legitimate slow generations into
    ambiguous spend.
    """
    return httpx.Timeout(
        connect=connect_slice,
        read=read_timeout,
        write=read_timeout,
        pool=connect_slice,
    )


class Deadline:
    """A monotonic FAILOVER-WINDOW deadline.

    Stamped once at ``_intercepted_call`` entry from ``failover_total_timeout``;
    it bounds the budget pre-flight, each hop's CONNECT/POOL slice, Retry-After
    sleeps, and between-hop advancement. It deliberately does NOT bound an
    in-flight hop's READ (PJ-8/R7): a read timeout is POST_SEND_AMBIGUOUS and
    re-raises under default idempotency, so cutting a legitimate slow
    generation at the failover window would only convert it into ambiguous
    spend. Reads are bounded per hop by ``failover_hop_read_timeout`` instead;
    because expiry gates advancement BETWEEN hops, at most ONE hop can consume
    the full read bound in a single call.
    """

    def __init__(self, total: float) -> None:
        self._total = total
        self._start = time.monotonic()

    def remaining(self) -> float:
        """Seconds left on the chain deadline (never negative)."""
        return max(0.0, self._total - (time.monotonic() - self._start))

    def expired(self) -> bool:
        """True once the chain deadline has elapsed."""
        return self.remaining() <= 0.0

    def replace_total(self, total: float) -> None:
        """Replace the total duration while preserving the original start."""
        self._total = total


class _MaterializedStream:
    """A first-chunk-materialized SYNC stream that owns the ORIGINAL generator.

    ``__iter__`` replays the buffered first chunk, then drains the ORIGINAL
    provider stream — so the wrapper observes + yields every chunk exactly once
    (no double-emit). Critically, ``close()`` forwards to the ORIGINAL provider
    stream's ``close()`` (fix [C]): the wrapper's getattr/close forwarding reaches
    THIS object, which in turn releases the real provider connection. Without this
    seam the wrapper would close an ``itertools.chain`` (a no-op) and leak the
    connection when a caller abandons the stream.

    ``_original`` is the SOURCE stream object (whose ``close()`` releases the
    connection); ``_iter`` is the iterator drained for chunks. Forwarding close()
    to the SOURCE — not the iterator — is what reaches a provider stream whose
    ``__iter__`` returns a distinct generator object.
    """

    def __init__(
        self, first: Any, original: Any, iterator: Iterator[Any], *, empty: bool = False
    ) -> None:
        self._first = first
        self._original = original
        self._iter = iterator
        self._empty = empty

    def __iter__(self) -> Iterator[Any]:
        if self._empty:
            return
        yield self._first
        yield from self._iter

    def close(self) -> None:
        close = getattr(self._original, "close", None)
        if close is not None:
            close()


def _materialize_stream(stream: Any) -> _MaterializedStream:
    """Force the first chunk of a streaming response (first-byte rule).

    GOOGLE ONLY (fix [B]): pulls ``next()`` on the raw lazy generator BEFORE the
    wrapper is returned. For a Google ``generate_content_stream`` LAZY generator
    this ``next()`` is what actually establishes the connection — so an
    establishment error PROPAGATES out of here, landing in the candidate-walk
    except where ``classify_exception`` can route it to failover, exactly like
    OpenAI/Anthropic's eager ``raise_for_status``.

    No double-emit: the buffered first chunk is replayed by ``_MaterializedStream``
    so the wrapper observes + yields it exactly once. An empty stream
    (``StopIteration`` on the first pull) yields nothing. The returned object
    forwards ``close()`` to the ORIGINAL generator (fix [C]).
    """
    it = iter(stream)
    try:
        first = next(it)
    except StopIteration:
        return _MaterializedStream(None, stream, it, empty=True)
    return _MaterializedStream(first, stream, it)


class _MaterializedAsyncStream:
    """Async mirror of ``_MaterializedStream`` (fix [B]/[C]).

    ``__aiter__`` replays the buffered first chunk then drains the ORIGINAL async
    iterator; ``aclose()`` forwards to the ORIGINAL SOURCE stream's ``aclose()``
    (or ``close()``) so abandoning the stream releases the provider connection.
    """

    def __init__(
        self, first: Any, original: Any, iterator: AsyncIterator[Any], *, empty: bool = False
    ) -> None:
        self._first = first
        self._original = original
        self._iter = iterator
        self._empty = empty

    async def __aiter__(self) -> AsyncIterator[Any]:
        if self._empty:
            return
        yield self._first
        async for chunk in self._iter:
            yield chunk

    async def aclose(self) -> None:
        aclose = getattr(self._original, "aclose", None)
        if aclose is not None:
            await aclose()
            return
        close = getattr(self._original, "close", None)
        if close is not None:
            close()


async def _materialize_stream_async(stream: Any) -> _MaterializedAsyncStream:
    """Async first-chunk materialization. Mirror of ``_materialize_stream``.

    ``await anext()`` forces a Google async lazy generator to establish, so an
    establishment error PROPAGATES out of THIS coroutine — landing in the
    candidate-walk except (failover-eligible). Because this is an ``async def``
    (not an async generator), awaiting it runs the eager ``anext`` immediately and
    returns the wrapper; the establishment error therefore surfaces at the
    ``await`` site inside the dispatch try. No double-emit: the first chunk is
    replayed ahead of the remaining iterator and yielded exactly once. The
    returned object forwards ``aclose()`` to the ORIGINAL SOURCE stream (fix [C]).
    """
    it = stream.__aiter__()
    try:
        first = await it.__anext__()
    except StopAsyncIteration:
        return _MaterializedAsyncStream(None, stream, it, empty=True)
    return _MaterializedAsyncStream(first, stream, it)


def _make_chunk_translator(*, served: str, requested: str) -> Callable[[Any], list[Any]]:
    """Closure that reshapes ONE raw served chunk into caller-dialect chunks.

    Routes the raw chunk straight to ``_translation.translate_stream_chunk`` (the
    content-privileged seam). This function — and the wrapper that calls it — never
    logs, stores, or stringifies chunk content; the chunk passes through opaquely.
    """

    def translate(chunk: Any) -> list[Any]:
        return _translation.translate_stream_chunk(served=served, requested=requested, chunk=chunk)

    return translate


def _normalize_fallback(fallback: object) -> list[Any]:
    """Normalize the ``fallback=`` constructor arg into a list of specs.

    ``None`` -> ``[]``. Each item is a ``(client, model)`` or
    ``(client, model, default_params)`` tuple validated downstream by
    ``build_runtimes``.
    """
    if fallback is None:
        return []
    return list(cast("list[Any]", fallback))


def _success_failover_reason(
    *, is_provider_fallback: bool, is_model_fallback: bool, primary_errored: bool
) -> FailoverReason | None:
    """Pick the success-path failover reason for a served candidate.

    Distinguishes WHY the router advanced past the primary on a cross-provider
    success:
      * ``PRIMARY_ERROR`` — the primary runtime was ATTEMPTED in this walk and
        ERRORED (a transport failure -> reactive failover).
      * ``CIRCUIT_OPEN`` — the primary was SKIPPED because its breaker was
        already OPEN (proactive reroute; the primary was never attempted here).
      * ``MODEL_FALLBACK`` — a same-provider model swap.
    ``None`` for a primary success.
    """
    if is_provider_fallback:
        return FailoverReason.PRIMARY_ERROR if primary_errored else FailoverReason.CIRCUIT_OPEN
    if is_model_fallback:
        return FailoverReason.MODEL_FALLBACK
    return None


def _build_hop_kwargs(
    *,
    primary: ProviderRuntime,
    rt: ProviderRuntime,
    is_primary: bool,
    is_provider_fallback: bool,
    is_streaming: bool,
    global_defaults: dict[str, Any],
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Build the native kwargs for one candidate hop.

    Fill-absent precedence: per-call kwargs > per-entry default_params > global.
    For a PRIMARY or SAME-PROVIDER model-swap hop the merged kwargs pass straight
    through to the native SDK (no translation). For a CROSS-PROVIDER hop the
    merged kwargs are run through the translation contract:

        canonical  = to_canonical(primary, merged_kwargs)
        call_kwargs = from_canonical(target, canonical, model=...)

    then the target entry default_params are re-applied as fill-absent so the
    target's own required/default fields (e.g. Anthropic ``max_tokens``) fill
    when the caller omitted them on the source dialect.

    The Untranslatable* errors raised here carry STRUCTURAL labels only; this
    function never logs, never stringifies content, and passes dicts straight
    through to ``_translation`` (a content-privileged module).
    """
    if (
        is_primary
        and not is_provider_fallback
        and not global_defaults
        and not rt.entry.default_params
        and "solwyn_tags" not in kwargs
        and (
            rt.adapter.dialect != ProviderName.OPENAI.value
            or ("max_tokens" not in kwargs and "max_completion_tokens" not in kwargs)
        )
    ):
        # PJ-9/P7 fast path: PRIMARY native passthrough with nothing to merge,
        # filter, or normalize — one defensive shallow copy (downstream
        # prepare_call may mutate) instead of the five dict builds below. The
        # output-cap guard keeps _normalized_openai_output_cap_layer
        # authoritative whenever an OpenAI-dialect hop carries a cap key.
        return dict(kwargs)

    provider_global_defaults = {
        key: value for key, value in global_defaults.items() if key != "solwyn_tags"
    }
    provider_entry_defaults = {
        key: value for key, value in rt.entry.default_params.items() if key != "solwyn_tags"
    }
    provider_kwargs = {key: value for key, value in kwargs.items() if key != "solwyn_tags"}

    same_google_dialect = (
        primary.adapter.dialect == ProviderName.GOOGLE.value
        and rt.adapter.dialect == ProviderName.GOOGLE.value
    )
    legacy_google_primary = (
        primary.adapter.dialect == ProviderName.GOOGLE.value
        and _client_shape(primary.sdk_client, primary.adapter.dialect) == "google_generativeai"
    )
    target_shape = (
        _client_shape(rt.sdk_client, rt.adapter.dialect)
        if rt.adapter.dialect == ProviderName.GOOGLE.value
        else ""
    )
    if same_google_dialect and not is_primary and legacy_google_primary:
        # Legacy _prepare_request is the no-I/O source of truth for merging the
        # model's constructor defaults below Solwyn/global/entry/call layers.
        # The privacy seam converts its effective request to modern Google shape;
        # the served-shape merge then adapts it to this fallback client.
        legacy_explicit = merge_google_generate_content_kwargs(
            provider_global_defaults,
            provider_entry_defaults,
            provider_kwargs,
            target_shape="google_generativeai",
        )
        merged_kwargs = merge_google_generate_content_kwargs(
            {},
            prepare_legacy_google_translation_source_kwargs(
                primary.sdk_client,
                legacy_explicit,
            ),
            target_shape=target_shape,
        )
    elif same_google_dialect:
        # Normalize EACH provenance layer through the content-privileged merge
        # seam so aliases across Google SDK generations cannot let a lower-
        # priority default beat a per-call value.
        merged_kwargs = merge_google_generate_content_kwargs(
            provider_global_defaults,
            provider_entry_defaults,
            provider_kwargs,
            target_shape=target_shape,
        )
    else:
        merged_kwargs = {
            **provider_global_defaults,
            **provider_entry_defaults,
            **provider_kwargs,
        }
    if not is_provider_fallback:
        # PRIMARY hop is native passthrough; same-provider hop only swaps model.
        # Same-provider streaming (incl. model swap) keeps working unchanged.
        target_model = cast(str, merged_kwargs["model"]) if is_primary else rt.entry.model
        if rt.adapter.dialect == ProviderName.OPENAI.value:
            # Normalize aliases within EACH provenance layer before merging;
            # otherwise a lower-priority modern key can survive beside a
            # higher-priority legacy key and silently win.
            target_name = rt.adapter.name

            def _native_target_layer(layer: dict[str, object]) -> dict[str, object]:
                return _normalized_openai_output_cap_layer(
                    target_name,
                    target_model,
                    layer,
                )

            merged_kwargs = {
                **_native_target_layer(provider_global_defaults),
                **_native_target_layer(provider_entry_defaults),
                **_native_target_layer(provider_kwargs),
            }
        if is_primary:
            return merged_kwargs
        return {**merged_kwargs, "model": rt.entry.model}

    # CROSS-PROVIDER hop. Defensive structural guard (fix [G]): the target
    # entry MUST carry a concrete model for this provider. An empty/falsy model
    # would otherwise be sent to a healthy provider and 400, burning a chain hop.
    # Raise UntranslatableModelError up front (structural, content-free) so the
    # caller aborts the chain like every other Untranslatable* error — BEFORE any
    # network call, with no breaker mutation and no advance.
    if not rt.entry.model:
        raise UntranslatableModelError(model=rt.entry.model, provider=rt.adapter.name)

    source_dialect = primary.adapter.dialect
    target_dialect = rt.adapter.dialect
    if source_dialect == target_dialect:
        # SAME-DIALECT cross-provider hop (e.g. Groq -> OpenRouter): both ends
        # speak the same wire shape, so this is native passthrough with a model
        # swap — no canonical-subset restriction. Tools, structured output, and
        # streaming all carry over; the served adapter's prepare_streaming
        # applies its own stream_options policy, and the completion-token key
        # is rewritten in whichever direction the TARGET requires.
        # Endpoint-scoped transport params authored for the ORIGINAL target
        # must not reach a different vendor: strip them from the passthrough,
        # then re-apply the TARGET entry's own default_params for those keys
        # (authored for this endpoint — same survival as the cross-dialect
        # path's trailing default_params merge).
        #
        # The legacy-key rewrite is applied PER SOURCE LAYER, before the
        # precedence merge. Rewriting once on the merged dict is
        # provenance-blind: an entry-default max_completion_tokens would
        # collapse onto max_tokens AFTER the merge and silently beat a
        # per-call max_tokens (a money-relevant output cap). With every layer
        # normalized to the target's key first, the standard merge order
        # below enforces per-call > entry default > global regardless of
        # which key each side used. Do NOT rewrite the merged result again.
        target_name = rt.adapter.name

        if target_dialect == ProviderName.GOOGLE.value:
            passthrough = {
                key: value
                for key, value in merged_kwargs.items()
                if key not in _ENDPOINT_SCOPED_KEYS
            }
            return {**passthrough, "model": rt.entry.model}

        def _fallback_target_layer(layer: dict[str, object]) -> dict[str, object]:
            return _normalized_openai_output_cap_layer(
                target_name,
                rt.entry.model,
                layer,
            )

        normalized: dict[str, object] = {
            **_fallback_target_layer(provider_global_defaults),
            **_fallback_target_layer(provider_entry_defaults),
            **_fallback_target_layer(provider_kwargs),
        }
        passthrough = {
            key: value for key, value in normalized.items() if key not in _ENDPOINT_SCOPED_KEYS
        }
        passthrough.update(
            {
                key: value
                for key, value in provider_entry_defaults.items()
                if key in _ENDPOINT_SCOPED_KEYS
            }
        )
        return {**passthrough, "model": rt.entry.model}

    # CROSS-DIALECT hop: translate via the canonical subset (may RAISE an
    # Untranslatable* error BEFORE any network call; the caller aborts the
    # chain). Translation starts from SOURCE-dialect values only: the target
    # entry's default_params may contain target-native keys such as Anthropic
    # top_k.
    source_defaults = _source_compatible_defaults(source_dialect, provider_entry_defaults)
    if source_dialect == ProviderName.OPENAI.value:
        source_name = primary.adapter.name
        source_model = cast(str, kwargs["model"])

        def _source_layer(layer: dict[str, object]) -> dict[str, object]:
            return _normalized_openai_output_cap_layer(
                source_name,
                source_model,
                layer,
            )

        source_kwargs: dict[str, object] = {
            **_source_layer(provider_global_defaults),
            **_source_layer(source_defaults),
            **_source_layer(provider_kwargs),
        }
    elif source_dialect == ProviderName.GOOGLE.value:
        # The canonical translator's Google schema is the modern ``config=``
        # shape. Normalize each source layer before its precedence merge so a
        # legacy primary can fail over cross-dialect without leaking old-SDK
        # aliases into the documented translation subset.
        google_source_kwargs = merge_google_generate_content_kwargs(
            provider_global_defaults,
            source_defaults,
            provider_kwargs,
            target_shape=("google_generativeai" if legacy_google_primary else "google_genai"),
        )
        if legacy_google_primary:
            google_source_kwargs = prepare_legacy_google_translation_source_kwargs(
                primary.sdk_client,
                google_source_kwargs,
            )
        source_kwargs = normalize_google_translation_source_kwargs(google_source_kwargs)
    else:
        source_kwargs = {
            **provider_global_defaults,
            **source_defaults,
            **provider_kwargs,
        }
    canonical = _translation.to_canonical(source_dialect, source_kwargs)

    # CROSS-DIALECT STREAMING. A PLAIN-TEXT cross-dialect
    # streamed response is normalized per-chunk by the wrapper, so it proceeds. A
    # TOOL-using streamed response cannot be normalized cross-dialect (tool-call
    # deltas are out of the v1 streaming subset) — so FAIL LOUD here, BEFORE
    # dispatch, aborting the chain cleanly (no foreign stream returned). Checking
    # the canonical keeps this structural and content-free.
    if is_streaming and canonical.tools is not None:
        _translation.fail_cross_provider_tool_stream(source=source_dialect, target=target_dialect)

    call_kwargs = _translation.from_canonical(target_dialect, canonical, model=rt.entry.model)
    if target_dialect == ProviderName.GOOGLE.value:
        # Target-native defaults remain fill-absent, while translated caller
        # values win after both layers are expressed in the served SDK shape.
        return merge_google_generate_content_kwargs(
            provider_entry_defaults,
            call_kwargs,
            target_shape=target_shape,
        )
    # Re-apply target entry defaults as fill-absent (e.g. Anthropic max_tokens).
    return {**provider_entry_defaults, **call_kwargs}


def _legacy_google_candidate_output_bound(
    *,
    primary: ProviderRuntime,
    runtimes: list[ProviderRuntime],
    global_defaults: dict[str, Any],
    kwargs: dict[str, object],
    is_streaming: bool,
    default_bound: int,
) -> int:
    """Return the largest effective cap among legacy Google candidates."""
    if primary.adapter.dialect != ProviderName.GOOGLE.value:
        return 0
    bounds: list[int] = []
    for runtime in runtimes:
        if (
            runtime.adapter.dialect != ProviderName.GOOGLE.value
            or _client_shape(runtime.sdk_client, runtime.adapter.dialect) != "google_generativeai"
        ):
            continue
        layers: tuple[dict[str, Any], ...]
        if runtime is primary:
            layers = (global_defaults, runtime.entry.default_params, kwargs)
            projected = prepare_legacy_google_metering_kwargs(runtime.sdk_client, *layers)
        else:
            try:
                hop_kwargs = _build_hop_kwargs(
                    primary=primary,
                    rt=runtime,
                    is_primary=False,
                    is_provider_fallback=runtime.entry.provider != primary.entry.provider,
                    is_streaming=is_streaming,
                    global_defaults=global_defaults,
                    kwargs=kwargs,
                )
                projected = prepare_legacy_google_metering_kwargs(
                    runtime.sdk_client,
                    hop_kwargs,
                )
            except UntranslatableRequestError:
                # This candidate cannot be dispatched for the current request,
                # so it cannot contribute spend. Preserve the ordinary runtime
                # behavior: a healthy earlier candidate may still serve, while
                # the structural error remains visible if the walk reaches it.
                continue
        bounds.append(
            _effective_output_bound(
                primary=runtime,
                runtimes=[runtime],
                global_defaults={},
                kwargs=projected,
                default_bound=default_bound,
            )
        )
    return max(bounds, default=0)


def _media_prepare(
    adapter: Any,
    surface: str,
    client: Any,
    kwargs: dict[str, object],
    *,
    timeout: float,
    max_retries: int,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Resolve the media SDK method + shaped kwargs, or raise UnsupportedSurfaceError.

    The per-surface dispatch seam lives on adapters that serve media surfaces
    (``MediaSurfaceAdapter.prepare_media_call``); it is intentionally absent
    from the base ``ProviderAdapter`` protocol, so it is discovered duck-typed
    here. An adapter that never grew the seam — or whose seam has no branch for
    this surface — is a surface the SDK cannot serve, so fail loud with the
    structural (content-free) ``UnsupportedSurfaceError``. Shared by the sync
    and async media dispatchers (only the invoke/await differs).
    """
    prepare = getattr(adapter, "prepare_media_call", None)
    if prepare is None:
        raise UnsupportedSurfaceError(surface=surface, provider=adapter.name)
    return cast(
        "tuple[Callable[..., Any], dict[str, Any]]",
        prepare(surface, client, kwargs, timeout=timeout, max_retries=max_retries),
    )


class Solwyn(_SolwynBase):
    """Synchronous Solwyn client wrapper.

    Wraps an OpenAI or Anthropic client with budget enforcement,
    circuit breaking, and metadata reporting.

    Usage::

        from openai import OpenAI
        from solwyn import Solwyn

        client = Solwyn(
            OpenAI(),
            api_key="sk_proj_...",
        )
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "Hello"}],
        )
        client.close()
    """

    def __init__(
        self,
        client: object,
        *,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        fallback: object = None,
        default_params: dict[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        on_unmetered: Literal["warn", "raise", "allow"] | None = None,
        acknowledge_untracked: Collection[str] | None = None,
        selection_policy: SelectionPolicy | None = None,
        **config_kwargs: object,
    ) -> None:
        # self._client is typed Any because each provider SDK has a different
        # public surface (chat/messages/models). A unified Protocol would not
        # match all three. Type safety stops at the _sync_dispatch boundary.
        self._client: Any = client

        if "project_id" in config_kwargs:
            raise TypeError("unexpected keyword argument 'project_id'")

        # Build the [primary, *fallbacks] runtime chain. All chain clients are
        # constructed up front so the first failover is pure dispatch.
        # ``provider`` optionally overrides auto-detection for the primary
        # (e.g. provider="vllm" for a server on a non-default port).
        fallback_specs = _normalize_fallback(fallback)
        runtimes = build_runtimes(client, model, fallback_specs, primary_provider=provider)

        # The primary runtime's adapter is the detected (or overridden)
        # provider identity for usage extraction and proxy selection.
        self._adapter = runtimes[0].adapter
        self._dialect = runtimes[0].adapter.dialect

        # Build config — SolwynConfig._load_from_env fills missing
        # values from SOLWYN_API_KEY env var.
        # cfg_kwargs stays dict[str, Any]: mypy can't verify Pydantic's **kwargs
        # validation against SolwynConfig's typed fields, so tightening here
        # adds noise without type-safety gain. SolwynConfig validates at runtime.
        cfg_kwargs: dict[str, Any] = {
            "providers": [rt.entry for rt in runtimes],
            "default_params": default_params or {},
            "tags": tags,
            **config_kwargs,
        }
        if api_key is not None:
            cfg_kwargs["api_key"] = api_key
        if on_unmetered is not None:
            cfg_kwargs["on_unmetered"] = on_unmetered
        if acknowledge_untracked is not None:
            cfg_kwargs["acknowledge_untracked"] = acknowledge_untracked
        try:
            config = SolwynConfig(**cfg_kwargs)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else None
            raise ConfigurationError(
                first["msg"] if first else str(exc),
                field=str(first["loc"][-1]) if first else None,
            ) from exc
        super().__init__(config, runtimes, selection_policy=selection_policy, mode="sync")

        # Budget enforcer
        self._budget = BudgetEnforcer(
            api_url=config.api_url,
            api_key=config.api_key,
            budget_mode=config.budget_mode,
            fail_open=config.fail_open,
            cache_ttl=config.budget_check_cache_ttl,
            control_plane_breaker=self._control_plane_breaker,
            # PJ-2: the SDK instance id IS the lease holder identity.
            holder_id=self._sdk_instance_id,
            lease_enabled=config.lease_enabled,
            lease_output_bound_default=config.lease_output_bound_default,
        )

        # Metadata reporter
        self._reporter = MetadataReporter(
            config.api_url,
            config.api_key,
            batch_size=config.reporter_batch_size,
            flush_interval=config.reporter_flush_interval,
            max_queue_size=config.reporter_max_queue_size,
            max_in_flight=config.reporter_max_in_flight,
            breaker_snapshots=self._get_breaker_snapshots,
            sdk_instance_id=self._sdk_instance_id,
            breaker_reporting_enabled=config.breaker_reporting_enabled,
            report_untracked_surfaces=config.report_untracked_surfaces,
            breaker_report_heartbeat=config.breaker_report_heartbeat,
            control_plane_breaker=self._control_plane_breaker,
            max_send_attempts=config.reporter_max_send_attempts,
            retry_backoff_base=config.reporter_retry_backoff_base,
            retry_backoff_cap=config.reporter_retry_backoff_cap,
            shutdown_deadline=config.reporter_shutdown_deadline,
        )
        self._untracked_observation_notifier = (
            self._reporter.observe_untracked_surface if config.report_untracked_surfaces else None
        )

    @functools.cached_property
    def chat(self) -> _SyncChatProxy:
        """Return a proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _SyncChatProxy(self)

    @functools.cached_property
    def responses(self) -> Any:
        """Expose metered Responses create and parse for native OpenAI clients.

        Both native leaves use the primary-only Responses pipeline; other
        Responses leaves stay guarded raw operations. OpenAI-compatible
        providers, including Azure, retain their existing raw namespace and
        unmetered posture. Cached because provider identity is construction-time
        state.
        """
        if (
            self._adapter.name == "openai"
            and self._inspect_static_attribute(self._client, "responses") is not None
        ):
            return _SyncResponsesProxy(self)
        return self._resolve_public_attribute(
            self._client,
            name="responses",
            path="responses",
            source=SurfaceSource.RAW,
        )

    @functools.cached_property
    def embeddings(self) -> _SyncEmbeddingsProxy:
        """Return a proxy that routes embeddings.create() through the media lifecycle.

        Unconditional (like ``chat``): the embeddings surface is the openai
        dialect, shared by native OpenAI and every OpenAI-compatible provider.
        On a non-openai client ``.create()`` fails loud with
        ``UnsupportedSurfaceError`` (that adapter serves no embeddings seam).
        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _SyncEmbeddingsProxy(self)

    @functools.cached_property
    def images(self) -> _SyncImagesProxy:
        """Return a proxy that routes images.generate()/edit() through the media lifecycle.

        Unconditional (like ``chat`` / ``embeddings``): the images surface is the
        openai dialect, shared by native OpenAI (token-billed gpt-image) and every
        OpenAI-compatible provider (per-image). On a non-openai client
        ``.generate()`` / ``.edit()`` fails loud with ``UnsupportedSurfaceError``
        (that adapter serves no images seam). Cached: provider is fixed at
        construction so this is safe to create once.
        """
        return _SyncImagesProxy(self)

    @functools.cached_property
    def audio(self) -> _SyncAudioProxy:
        """Return a proxy that routes audio transcriptions/speech through the lifecycle.

        Unconditional (like ``chat`` / ``embeddings`` / ``images``): the audio
        transcriptions AND speech (TTS) surfaces are the openai dialect, shared by
        native OpenAI and every OpenAI-compatible provider (incl. Groq whisper). On
        a non-openai client ``.transcriptions.create()`` / ``.speech.create()``
        fail loud with ``UnsupportedSurfaceError`` (that adapter serves no audio
        seam). The proxy's ``translations`` sub-surface uses the shared untracked
        posture. Cached: provider is fixed at construction so this is safe to
        create once.
        """
        return _SyncAudioProxy(self)

    @functools.cached_property
    def videos(self) -> _SyncVideosProxy:
        """Return a proxy that routes videos.create() (Sora) through the media lifecycle.

        Unconditional (like ``chat`` / ``embeddings`` / ``images`` / ``audio``):
        the video surface is the openai dialect. Sora is OpenAI-only, so on a
        non-openai client (including OpenAI-compatible profiles) ``.create()`` fails
        loud with ``UnsupportedSurfaceError`` (that adapter serves no video seam).
        The returned async video job is passed back untouched — callers poll it
        themselves. Cached: provider is fixed at construction so this is safe to
        create once.
        """
        return _SyncVideosProxy(self)

    @functools.cached_property
    def messages(self) -> Any:
        """Anthropic-compatible: client.messages.create() goes through interception.

        Cached: the dialect is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._dialect == "anthropic":
            return _SyncMessagesProxy(self)
        return self._resolve_public_attribute(
            self._client,
            name="messages",
            path="messages",
            source=SurfaceSource.RAW,
        )

    @functools.cached_property
    def models(self) -> Any:
        """google-genai: client.models generation goes through interception.

        The legacy ``google.generativeai`` shape has no ``models`` namespace;
        its root ``generate_content`` method is intercepted separately. Cached:
        the client shape is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._surface_context.client_shape == "google_genai":
            return _SyncModelsProxy(self)
        return self._resolve_public_attribute(
            self._client,
            name="models",
            path="models",
            source=SurfaceSource.RAW,
        )

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept legacy ``google.generativeai`` root generation calls."""
        if self._surface_context.client_shape != "google_generativeai":
            method = self._resolve_public_attribute(
                self._client,
                name="generate_content",
                path="generate_content",
                source=SurfaceSource.RAW,
            )
            return method(*args, **kwargs)

        self._enforce_explicit_surface(
            "generate_content",
            source=SurfaceSource.WRAPPER,
        )
        call_kwargs = normalize_legacy_google_generate_content_args(args, kwargs)
        if "model" not in call_kwargs:
            requested_model = self._runtimes[0].entry.model
            if not requested_model:
                structural_model = getattr(self._client, "model_name", None)
                requested_model = (
                    structural_model.removeprefix("models/")
                    if isinstance(structural_model, str)
                    else ""
                )
                if not requested_model:
                    raise ConfigurationError(
                        "legacy Google client has no usable model_name",
                        field="model",
                    )
            call_kwargs["model"] = requested_model
        return self._intercepted_call(**call_kwargs)

    def converse(self, **kwargs: Any) -> Any:
        """Bedrock-compatible: client.converse(modelId=...) goes through interception."""
        self._enforce_explicit_surface("converse", source=SurfaceSource.WRAPPER)
        if self._dialect == "bedrock":
            return self._intercepted_call(**_bedrock_internal_kwargs(kwargs))
        return self._client.converse(**kwargs)

    def converse_stream(self, **kwargs: Any) -> Any:
        """Bedrock-compatible streaming: returns the boto3 dict with a wrapped stream."""
        self._enforce_explicit_surface("converse_stream", source=SurfaceSource.WRAPPER)
        if self._dialect == "bedrock":
            return self._intercepted_call(_force_stream=True, **_bedrock_internal_kwargs(kwargs))
        return self._client.converse_stream(**kwargs)

    def invoke_model(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        self._enforce_explicit_surface(
            "invoke_model",
            source=SurfaceSource.WRAPPER,
            blocked_reason=_INVOKE_MODEL_GUIDANCE,
        )
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return self._client.invoke_model(**kwargs)

    def invoke_model_with_response_stream(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        self._enforce_explicit_surface(
            "invoke_model_with_response_stream",
            source=SurfaceSource.WRAPPER,
            blocked_reason=_INVOKE_MODEL_GUIDANCE,
        )
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return self._client.invoke_model_with_response_stream(**kwargs)

    def start_async_invoke(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's async invocation surface (untracked video-scale spend)."""
        self._enforce_explicit_surface(
            "start_async_invoke",
            source=SurfaceSource.WRAPPER,
            blocked_reason=_START_ASYNC_INVOKE_GUIDANCE,
        )
        if self._dialect == "bedrock":
            raise ConfigurationError(_START_ASYNC_INVOKE_GUIDANCE, field="start_async_invoke")
        return self._client.start_async_invoke(**kwargs)

    def _sync_dispatch(
        self,
        runtime: ProviderRuntime,
        kwargs: dict[str, object],
        *,
        is_streaming: bool,
        timeout: float,
        read_timeout: float,
        max_retries: int,
        surface: str = "chat",
        responses_leaf: str = "create",
    ) -> Any:
        """Dispatch one hop to the runtime's SDK client. Pure I/O - no metrics.

        ``timeout`` is the CONNECT/POOL slice of the failover window;
        ``read_timeout`` is the decoupled per-hop READ/WRITE bound (PJ-8/R7).
        with_options clients receive the granular httpx.Timeout (kills the
        SDK's own retry stacking via max_retries too); SDKs without
        ``with_options`` apply their own bound inside ``prepare_call``, which
        receives the READ bound (google can only set one whole-request
        timeout; a slow legitimate generation must survive it).
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(
                timeout=_hop_httpx_timeout(timeout, read_timeout),
                max_retries=max_retries,
            )
        if surface == "responses":
            prepare = getattr(runtime.adapter, "prepare_responses_call", None)
            if prepare is None:
                raise UnsupportedSurfaceError(
                    surface=f"responses.{responses_leaf}", provider=runtime.adapter.name
                )
            method, call_kwargs = prepare(
                client,
                cast("dict[str, Any]", kwargs),
                is_streaming=is_streaming,
                leaf=responses_leaf,
            )
            return method(**call_kwargs)
        method, call_kwargs = runtime.adapter.prepare_call(
            client,
            cast("dict[str, Any]", kwargs),
            is_streaming=is_streaming,
            timeout=read_timeout,
            max_retries=max_retries,
        )
        return method(**call_kwargs)

    def _media_dispatch(
        self,
        runtime: ProviderRuntime,
        surface: str,
        kwargs: dict[str, object],
        *,
        timeout: float,
        read_timeout: float,
        max_retries: int,
    ) -> Any:
        """Dispatch one media hop to the runtime's SDK client. Pure I/O — no metrics.

        The media analogue of ``_sync_dispatch``: ``timeout`` is the
        CONNECT/POOL slice of the failover window, ``read_timeout`` is the
        decoupled per-hop READ/WRITE bound (PJ-8/R7), applied via
        ``with_options`` where available and handed to the adapter's
        ``prepare_media_call`` seam otherwise. No streaming and no candidate
        walk — a media call is served by the primary runtime alone.
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(
                timeout=_hop_httpx_timeout(timeout, read_timeout),
                max_retries=max_retries,
            )
        method, call_kwargs = _media_prepare(
            runtime.adapter, surface, client, kwargs, timeout=read_timeout, max_retries=max_retries
        )
        return method(**call_kwargs)

    def _media_call(self, spec: MediaSurfaceSpec, **kwargs: object) -> Any:
        """Lean lifecycle for a non-chat media surface (embeddings/images/audio/video).

        estimate -> budget check -> provider call -> extract/measure -> confirm +
        report. Deliberately NOT the chat pipeline: embedding vectors (and image
        / audio / video outputs) are not interchangeable across providers, so
        there is no candidate walk, no cross-dialect translation, and no model
        fallback (``is_model_fallback`` is always False). The call is served by
        the PRIMARY runtime alone.

        Billable quantity is per-surface via the spec hooks: response usage where
        it exists (``extract_usage``), request-derived where it does not
        (``measure_request``). An unobservable quantity stays None — never a
        zero-filled default — so a real $0 price is never settled.
        """
        agent_run = _capture_run_context(
            kwargs.pop("solwyn_tags", None),
            default_tags=self._config.tags,
        )
        requested_model = cast(str, kwargs["model"])
        call_id = str(uuid.uuid4())
        runtime = self._runtimes[0]
        provider = runtime.adapter.name
        deadline = Deadline(self._config.failover_total_timeout)

        # 1. Estimate quantities (length/config-only; never materializes content).
        #    Token estimate from request length; media estimate (non-token
        #    quantities like image count) from the spec's request-derived hook.
        char_count = estimate_content_length(kwargs)
        est_in = estimate_tokens_from_length(char_count, provider=provider) if char_count else 0
        estimated_media = spec.estimate_media(kwargs) if spec.estimate_media is not None else None

        # 2. Budget check against the primary (no failover chain to hint). The
        #    surface's estimated_media rides the check so the server prices a
        #    precise per-unit pre-flight cost (precise pre-flight denial is the
        #    product story for images).
        #    Posture on 422-for-unbilled-model: if the server deliberately does not
        #    price this model/modality it answers /budgets/check with a 422. The
        #    enforcer treats that like any non-2xx — it raises inside check_budget,
        #    is caught there, and (absent a prior hard-deny) resolves via the
        #    standard fail-open path: allowed=True with a warning. So an
        #    unbilled-model media call proceeds untracked rather than being denied;
        #    fail-open is intentional here — Solwyn never blocks a call just because
        #    it cannot yet price it.
        budget = self._budget.check_budget(
            estimated_input_tokens=est_in,
            model=requested_model,
            provider=provider,
            timeout=_budget_timeout(deadline, self._config.budget_check_timeout),
            modality=spec.modality,
            estimated_media=estimated_media,
            agent_run_id=agent_run[0],
            tags=agent_run[2],
            call_id=call_id,
        )
        # PJ-8/R12: ONE immutable tuning snapshot per call - the walk below
        # must never re-read self._config (the directive writer mutates it
        # under a lock; unlocked re-reads can tear).
        tuning = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        deadline.replace_total(tuning.failover_total_timeout)
        self._reporter.observe_project_id(budget.project_id)
        if budget.price_hints is not None:
            self.update_price_hints(budget.price_hints)

        provider_region = runtime.adapter.extract_region(runtime.sdk_client)
        if not budget.allowed:
            try:
                self._reporter.report(
                    self._build_metadata_event(
                        model=requested_model,
                        provider=provider,
                        input_tokens=est_in,
                        output_tokens=0,
                        token_details=None,
                        latency_ms=0.0,
                        status=CallStatus.BUDGET_DENIED,
                        is_model_fallback=False,
                        call_id=call_id,
                        agent_run=agent_run,
                        provider_region=provider_region,
                        modality=spec.modality,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to report budget_denied metadata event: %s",
                    type(exc).__name__,
                )
            raise _budget_denial_error(
                budget=budget,
                agent_run_id=agent_run[0],
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
            )

        # 3. Deadline gate, mirroring the chat walk (PJ-8/R7 follow-up). Since
        #    connect and read are decoupled, an expired window no longer bounds
        #    the call on its own: _hop_connect_slice returns its 0.001s floor,
        #    which a WARM POOLED connection satisfies, and the hop would then
        #    read for the full (default 600s) bound. Whether an expired media
        #    call escapes would come down to pool state. Gate it explicitly
        #    instead — chat rejects this exact state, and no provider I/O may
        #    start outside the window.
        if deadline.expired():
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            raise ProviderUnavailableError(
                "failover deadline expired",
                attempted=[provider],
            )

        # 4. Provider call — PRIMARY only. A dispatch error is reported (parity
        #    with the chat error path) then re-raised unchanged (drop-in contract).
        start = time.monotonic()
        try:
            response = self._media_dispatch(
                runtime,
                spec.surface,
                kwargs,
                timeout=_hop_connect_slice(deadline, 1),
                read_timeout=tuning.failover_hop_read_timeout,
                max_retries=0,
            )
        except Exception as exc:
            # Nothing will settle this call: hand any lease reservation back
            # rather than stranding it until the 900s sweep.
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            self._reporter.report(
                self._build_error_event(
                    model=requested_model,
                    provider=provider,
                    latency_ms=(time.monotonic() - start) * 1000,
                    is_model_fallback=False,
                    failover_error_class=type(exc).__name__,
                    call_id=call_id,
                    agent_run=agent_run,
                    provider_region=provider_region,
                )
            )
            raise
        latency_ms = (time.monotonic() - start) * 1000

        # 5. Billable quantities: TOKEN basis (response usage first, request-
        #    derived fallback) AND, when the surface has a media channel, the
        #    non-token MediaUsage basis. BOTH ride the confirm when observable —
        #    the server's pricing card unit picks (e.g. native gpt-image sends
        #    token usage with image buckets AND request-derived image quantities).
        #
        # Fail-soft bookkeeping (R5): the provider has answered — a surface-spec
        # or adapter raise must not destroy the paid media response. Usage
        # degrades to the request-derived measure, then to None; a None/None
        # pair simply skips the confirm exactly as before.
        token_details: TokenDetails | None = None
        try:
            token_details = spec.extract_usage(response)
        except Exception as exc:
            logger.warning(
                "settlement.media_extract_usage_failed_fail_soft: %s",
                type(exc).__name__,
            )
        if token_details is None:
            try:
                token_details = spec.measure_request(kwargs)
            except Exception as exc:
                logger.warning(
                    "settlement.media_measure_request_failed_fail_soft: %s",
                    type(exc).__name__,
                )
        media_usage = None
        if spec.measure_media is not None:
            try:
                media_usage = spec.measure_media(kwargs, response)
            except Exception as exc:
                logger.warning(
                    "settlement.media_measure_media_failed_fail_soft: %s",
                    type(exc).__name__,
                )

        # 6. Settle OFF the hot path: build the confirm sans-I/O and enqueue it
        #    with the metadata event as one ordered settlement (same path as
        #    chat + streaming). Confirm fires when EITHER basis is observable;
        #    skipped only when both are None (never settle a real $0 price).
        #    When only media is observed, a zeroed TokenDetails carries the
        #    confirm's required token field.
        service_tier = _safe_extract_service_tier(runtime, response)
        confirm = None
        reservation_id, lease_id, lease_claim_token = _settlement_keys(budget)
        if (reservation_id or lease_id) and (token_details is not None or media_usage is not None):
            confirm = self._budget.build_confirm_request(
                reservation_id=reservation_id,
                lease_id=lease_id,
                lease_claim_token=lease_claim_token,
                model=requested_model,
                token_details=token_details if token_details is not None else TokenDetails(),
                provider=provider,
                is_provider_fallback=False,
                call_id=call_id,
                provider_region=provider_region,
                service_tier=service_tier,
                modality=spec.modality,
                media_usage=media_usage,
            )
        event = self._build_metadata_event(
            model=requested_model,
            provider=provider,
            input_tokens=token_details.input_tokens if token_details is not None else 0,
            output_tokens=token_details.output_tokens if token_details is not None else 0,
            token_details=token_details,
            latency_ms=latency_ms,
            status=CallStatus.SUCCESS,
            is_model_fallback=False,
            attempt_index=0,
            call_id=call_id,
            service_tier=service_tier,
            agent_run=agent_run,
            provider_region=provider_region,
            modality=spec.modality,
            media_usage=media_usage,
        )
        if confirm is not None:
            self._reporter.report_settlement(confirm, event)
        else:
            self._reporter.report(event)
        return response

    def _intercepted_call(
        self,
        *,
        _force_stream: bool = False,
        _surface: str = "chat",
        _responses_leaf: str = "create",
        **kwargs: object,
    ) -> Any:
        """Core interception logic: the classified candidate walk."""
        agent_run = _capture_run_context(
            kwargs.pop("solwyn_tags", None),
            default_tags=self._config.tags,
        )
        requested_model = cast(str, kwargs["model"])
        # One reconciliation join key per intercepted call: threaded into
        # every served-provider metadata event AND its confirm so the Cloud API
        # can join them (and dedup cache-hit / abandoned-stream spend).
        call_id = str(uuid.uuid4())
        primary = self._runtimes[0]
        if (
            _surface == "responses"
            and getattr(primary.adapter, "prepare_responses_call", None) is None
        ):
            raise UnsupportedSurfaceError(
                surface=f"responses.{_responses_leaf}", provider=primary.adapter.name
            )
        # Deadline starts here — it encompasses the budget pre-flight.
        deadline = Deadline(self._config.failover_total_timeout)

        request_semantics = kwargs
        responses_idempotent_override: bool | None = None
        if _surface == "responses":
            # Keep Solwyn's private per-call routing hint out of the provider
            # request while preserving the contract that unrecognized DEFAULTS
            # are not silently stripped.
            responses_idempotent_override = cast(
                "bool | None", kwargs.pop("solwyn_idempotent", None)
            )
            request_semantics = _effective_responses_kwargs(
                global_defaults=self._config.default_params,
                primary_defaults=primary.entry.default_params,
                kwargs=kwargs,
            )
            _reject_responses_background(request_semantics)
        is_streaming = (
            _responses_is_streaming(
                request_semantics,
                leaf=_responses_leaf,
                force_stream=_force_stream,
            )
            if _surface == "responses"
            else bool(request_semantics.get("stream", False)) or _force_stream
        )

        # Legacy GenerativeModel constructor defaults are applied by its
        # no-I/O _prepare_request seam, not exposed in the raw call kwargs.
        # Build an ephemeral effective view solely for metering; dispatch below
        # still receives the original kwargs so provider behavior stays native.
        metering_kwargs = request_semantics
        if (
            _surface != "responses"
            and _client_shape(primary.sdk_client, primary.adapter.dialect) == "google_generativeai"
        ):
            metering_kwargs = prepare_legacy_google_metering_kwargs(
                primary.sdk_client,
                self._config.default_params,
                primary.entry.default_params,
                kwargs,
            )

        # 1. Estimate input tokens (length-only; never materializes joined string).
        char_count = (
            estimate_responses_content_length(cast("dict[str, Any]", request_semantics))
            if _surface == "responses"
            else estimate_content_length(metering_kwargs)
        )
        est_in = (
            estimate_tokens_from_length(char_count, provider=primary.adapter.name)
            if char_count
            else 0
        )

        # 2. Check budget against the PRIMARY (we don't yet know who serves).
        if _surface == "responses":
            fallback_providers: list[str] = []
            fallback_models: list[str] = []
            estimated_output_bound = _responses_output_bound(
                request_semantics,
                self._config.lease_output_bound_default,
            )
        else:
            fallback_providers = [r.entry.provider.value for r in self._runtimes[1:]]
            fallback_models = [r.entry.model for r in self._runtimes[1:]]
            estimated_output_bound = max(
                _legacy_google_candidate_output_bound(
                    primary=primary,
                    runtimes=self._runtimes,
                    global_defaults=self._config.default_params,
                    kwargs=kwargs,
                    is_streaming=is_streaming,
                    default_bound=self._config.lease_output_bound_default,
                ),
                _effective_output_bound(
                    primary=primary,
                    runtimes=self._runtimes,
                    global_defaults=self._config.default_params,
                    kwargs=kwargs,
                    default_bound=self._config.lease_output_bound_default,
                ),
            )
        if _surface == "responses":
            budget = self._budget.check_budget(
                estimated_input_tokens=est_in,
                model=requested_model,
                provider=primary.adapter.name,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=_budget_timeout(deadline, self._config.budget_check_timeout),
                modality="text",
                agent_run_id=agent_run[0],
                tags=agent_run[2],
                call_id=call_id,
                estimated_output_bound=estimated_output_bound,
            )
        else:
            budget = self._budget.check_budget(
                estimated_input_tokens=est_in,
                model=requested_model,
                provider=primary.adapter.name,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=_budget_timeout(deadline, self._config.budget_check_timeout),
                agent_run_id=agent_run[0],
                tags=agent_run[2],
                call_id=call_id,
                estimated_output_bound=estimated_output_bound,
            )
        # PJ-8/R12: ONE immutable tuning snapshot per call - the walk below
        # must never re-read self._config (the directive writer mutates it
        # under a lock; unlocked re-reads can tear).
        tuning = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        deadline.replace_total(tuning.failover_total_timeout)
        self._reporter.observe_project_id(budget.project_id)
        # Refresh the CostPolicy signal from the server. Price hints are advisory
        # and slow-moving, so they PERSIST across hint-less responses — a budget
        # cache hit (price_hints None) leaves the last-known hints in place; we
        # only overwrite when the server actually returns hints. The SDK never
        # computes price — it only forwards this relative signal.
        if budget.price_hints is not None:
            self.update_price_hints(budget.price_hints)

        if not budget.allowed:
            # Report estimated tokens so the API keeps an accurate running total
            # even for calls that were blocked by hard-deny. The PRIMARY's
            # endpoint region rides along so denied-Bedrock spend stays
            # analyzable per region (None-skipped for other providers).
            try:
                event = self._build_metadata_event(
                    model=requested_model,
                    provider=primary.adapter.name,
                    input_tokens=est_in,
                    output_tokens=0,
                    token_details=None,
                    latency_ms=0.0,
                    status=CallStatus.BUDGET_DENIED,
                    is_model_fallback=False,
                    call_id=call_id,
                    agent_run=agent_run,
                    provider_region=primary.adapter.extract_region(primary.sdk_client),
                )
                self._reporter.report(event)
            except Exception as exc:
                logger.warning(
                    "Failed to report budget_denied metadata event: %s",
                    type(exc).__name__,
                )

            raise _budget_denial_error(
                budget=budget,
                agent_run_id=agent_run[0],
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
            )

        # 3. Resolve per-call idempotency override (strip before dispatch).
        idempotent_override = (
            responses_idempotent_override
            if _surface == "responses"
            else cast("bool | None", kwargs.pop("solwyn_idempotent", None))
        )
        if idempotent_override is True:
            effective_idempotency = "always"
        elif idempotent_override is False:
            effective_idempotency = "safe"
        else:
            effective_idempotency = tuning.failover_idempotency
        allow_cross_provider = effective_idempotency != "never"
        allow_ambiguous_failover = effective_idempotency == "always"

        # 4. Router returns ordered, health-filtered candidates (non-mutating reads).
        candidates = self._select_candidates(
            RoutingRequest(
                requested_provider=primary.entry.provider,
                estimated_input_tokens=est_in,
            )
        )
        if not allow_cross_provider:
            candidates = [c for c in candidates if c.entry.provider == primary.entry.provider]
        if _surface == "responses":
            # v1 is native-primary only. Compat Responses support is not uniform,
            # and no cross-dialect translation subset exists for this request shape.
            candidates = [c for c in candidates if c is primary]
        if not candidates:
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            raise ProviderUnavailableError("all providers unavailable", attempted=[])
        if deadline.remaining() <= 0.0:
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            raise ProviderUnavailableError(
                "failover deadline expired",
                attempted=[r.adapter.name for r in candidates],
            )

        # 5. Walk the candidates.
        failed_providers: set[str] = set()
        last_exc: Exception | None = None
        # Fix [A]: did the PRIMARY runtime get attempted-and-error in this
        # walk? Drives the cross-provider success reason (PRIMARY_ERROR if the
        # primary was attempted and raised; CIRCUIT_OPEN if it was skipped OPEN).
        primary_errored = False
        for idx, rt in enumerate(candidates):
            if deadline.expired():
                break
            provider = rt.adapter.name
            cb = self._get_circuit_breaker(provider)
            admission = cb.admit()  # probe CONSUMED only here, for the attempted candidate
            if not admission.allowed:
                continue

            is_primary = rt is primary
            is_provider_fallback = rt.entry.provider != primary.entry.provider
            is_model_fallback = (not is_provider_fallback) and not is_primary
            # Fix [B]: attempt_index is the served runtime's position in the
            # CONFIGURED chain (0=primary, 1=first fallback, ...),
            # NOT the candidate-walk index. When the primary breaker is
            # OPEN-not-eligible it is dropped from the health-filtered candidate
            # list, so the walk index would mislabel the first fallback as 0 and
            # corrupt the dashboard chain-depth funnel. The per-hop timeout slice
            # still uses the candidate-walk ``idx`` (remaining candidates, not
            # chain depth).
            chain_index = next(i for i, r in enumerate(self._runtimes) if r is rt)

            # Build native kwargs for this hop. A cross-provider hop runs the
            # translation contract and may RAISE an Untranslatable* error here,
            # BEFORE any network call — that aborts the WHOLE chain:
            # do NOT classify it as transport, advance, or record a breaker failure.
            # This is a no-health-signal abort: if admit() already consumed
            # this breaker's single HALF_OPEN probe slot, free it before the error
            # propagates so the provider is not stranded HALF_OPEN with no probe.
            try:
                if _surface == "responses":
                    call_kwargs = dict(request_semantics)
                    served_model = requested_model
                else:
                    call_kwargs = _build_hop_kwargs(
                        primary=primary,
                        rt=rt,
                        is_primary=is_primary,
                        is_provider_fallback=is_provider_fallback,
                        is_streaming=is_streaming,
                        global_defaults=self._config.default_params,
                        kwargs=kwargs,
                    )
                    served_model = requested_model if is_primary else rt.entry.model
                    if is_streaming:
                        call_kwargs = rt.adapter.prepare_streaming(
                            call_kwargs, cross_provider=is_provider_fallback
                        )
            except Exception:
                cb.release_probe(admission)
                self._budget.release_reservation(
                    call_id,
                    lease_claim_token=_lease_claim_token(budget),
                )
                raise

            # Same-provider retry budget for THIS chain entry (config seam,
            # default 0). Consumed inside the inner attempt loop below.
            same_retries_left = tuning.same_provider_retries
            advanced = False
            while True:
                ctx = _AttemptContext(
                    model=served_model,
                    start_time=time.monotonic(),
                    is_provider_fallback=is_provider_fallback,
                    attempt_index=chain_index,
                )
                try:
                    response = self._sync_dispatch(
                        rt,
                        call_kwargs,
                        is_streaming=is_streaming,
                        # Per-hop bounds (PJ-8/R7): connect/pool get a
                        # shrinking slice of the remaining FAILOVER window so a
                        # pre-send hang cannot eat the whole budget; read/write
                        # get the decoupled hop read bound (a read timeout is
                        # POST_SEND_AMBIGUOUS - never failover - so the chain
                        # deadline must not cut legitimate slow generations).
                        timeout=_hop_connect_slice(deadline, len(candidates) - idx),
                        read_timeout=tuning.failover_hop_read_timeout,
                        max_retries=0,
                        surface=_surface,
                        responses_leaf=_responses_leaf,
                    )
                    if (
                        is_streaming
                        and _client_shape(rt.sdk_client, rt.adapter.dialect) == "google_genai"
                    ):
                        # Modern google-genai returns a lazy generator, so its
                        # first pull must establish INSIDE this candidate try.
                        # Legacy google.generativeai already consumes the first
                        # provider response while building GenerateContentResponse;
                        # iterating that wrapper here lookaheads another response
                        # and could misclassify a later stream error as an
                        # establishment failure eligible for failover. Key this to
                        # the SERVED runtime shape so fallback hops behave correctly.
                        # The buffered modern first chunk is replayed by the wrapper.
                        response = _materialize_stream(response)
                except Exception as exc:
                    disp = classify_exception(exc)
                    # Fix [A]: the PRIMARY was attempted and raised in this walk
                    # -> a later cross-provider success is a REACTIVE failover
                    # (PRIMARY_ERROR), not a proactive breaker-open reroute.
                    if is_primary:
                        primary_errored = True
                    # Same-provider retry: a 429 the provider asked us to retry (a
                    # usable Retry-After that fits the remaining deadline, leaving a
                    # min hop for the re-attempt) sleeps then re-attempts the SAME
                    # provider before burning a cross-provider hop. We HOLD this
                    # admission across the sleep — an unresolved 429 is neither a
                    # success nor a failure, so NO breaker verdict is recorded and the
                    # HALF_OPEN probe slot stays ours (never stranded). The terminal
                    # outcome (success below, or the exhausted/unretryable failure
                    # here) is the single verdict that frees the slot.
                    if disp is Disposition.FAILOVER and same_retries_left > 0:
                        retry_delay = retry_after_seconds(exc)
                        if (
                            retry_delay is not None
                            and retry_delay + _MIN_HOP_TIMEOUT <= deadline.remaining()
                        ):
                            same_retries_left -= 1
                            time.sleep(retry_delay)
                            if not deadline.expired():
                                continue  # re-attempt the SAME candidate
                    # Breaker accounting: FAILOVER and POST_SEND_AMBIGUOUS are
                    # provider-health signals and DO count; FAIL_FAST (4xx/refusal) is
                    # a request-shaped error, not a health signal, so it must NOT open
                    # the breaker. Same-provider double-count guard: at most one
                    # failure per provider per logical attempt.
                    if disp is not Disposition.FAIL_FAST and provider not in failed_providers:
                        cb.record_failure()
                        failed_providers.add(provider)
                    else:
                        # No NEW health verdict for this hop: FAIL_FAST is request-shaped,
                        # or this provider was already counted this walk (double-count
                        # guard). If the hop consumed a HALF_OPEN probe slot, free it
                        # (no state change) so the breaker is not stranded HALF_OPEN.
                        cb.release_probe(admission)
                    # A correctly-not-failed-over post-send-ambiguous abort
                    # emits an ERROR event with possibly_succeeded=True so the Cloud API
                    # can reconcile the (possibly-landed, never-confirmed) reservation.
                    possibly_succeeded = (
                        disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover
                    )
                    self._reporter.report(
                        self._build_error_event(
                            model=served_model,
                            provider=provider,
                            latency_ms=ctx.elapsed_ms(),
                            is_model_fallback=is_model_fallback,
                            is_provider_fallback=is_provider_fallback,
                            requested_provider=(
                                primary.entry.provider if is_provider_fallback else None
                            ),
                            requested_model=requested_model if is_provider_fallback else None,
                            failover_error_class=type(exc).__name__,
                            attempt_index=chain_index,
                            call_id=call_id,
                            possibly_succeeded=True if possibly_succeeded else None,
                            agent_run=agent_run,
                            provider_region=_safe_extract_region(rt),
                        )
                    )
                    if disp is Disposition.FAIL_FAST:
                        self._budget.release_reservation(
                            call_id,
                            lease_claim_token=_lease_claim_token(budget),
                        )
                        raise  # 4xx/404/refusal — do NOT advance the chain
                    if disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover:
                        # The call MAY have landed, but no confirm will ever
                        # settle it here: the server reconciles the possibly-
                        # succeeded attempt from the error event.
                        self._budget.release_reservation(
                            call_id,
                            lease_claim_token=_lease_claim_token(budget),
                        )
                        raise  # re-raise ORIGINAL exception (drop-in contract)
                    last_exc = exc
                    advanced = True
                # Reached on a successful hop OR on a terminal failure that advances
                # the chain; a same-provider retry `continue`s above and never lands
                # here. `advanced` distinguishes the two so success falls through to
                # settlement below.
                break
            if advanced:
                continue  # pre-send safe -> advance to the next candidate

            # 6. SUCCESS — settle against the SERVED runtime.
            #
            # Fix [A]: for the STREAMING branch we do NOT credit the breaker here.
            # The single success is settled ONLY when the stream completes, by the
            # wrapper's on_complete (which records success + latency + confirm +
            # metadata exactly once via the _settled guard). Crediting both here
            # AND in on_complete would double-credit a HALF_OPEN breaker, closing
            # it after a single streaming probe (defeating anti-flap recovery), and
            # a stream that establishes then errors mid-flight would record a
            # spurious success before its on_error failure. So record_success()
            # runs ONLY on the non-streaming path, AFTER the streaming early return.
            if is_streaming:
                return self._wrap_stream(
                    rt,
                    response,
                    ctx,
                    budget,
                    primary,
                    requested_model=requested_model,
                    is_model_fallback=is_model_fallback,
                    primary_errored=primary_errored,
                    call_id=call_id,
                    agent_run=agent_run,
                    estimated_input_tokens=est_in,
                    estimate_empty_usage=_surface == "responses",
                )
            cb.record_success()

            # LatencyPolicy signal: record this served hop's success latency
            # (non-streaming). Streaming records in on_complete when the stream
            # settles. Pure signal store — no I/O, no routing change here.
            self.record_latency(provider, ctx.elapsed_ms())

            # Fail-soft bookkeeping (R5): a paid, successful response is never
            # destroyed by extraction — usage degrades to estimates
            # (is_estimated=True), region/tier degrade to None.
            token_details, usage_unmeasured = _extract_usage_fail_soft(
                rt, response, estimated_input_tokens=est_in
            )
            # Per-region pricing attribution: the SERVED runtime's endpoint region.
            provider_region = _safe_extract_region(rt)
            # The tier echoed on the RAW served response is the billing ground
            # truth. Extracted ONCE: confirm and metadata for one call_id must
            # carry the same tier.
            service_tier = _safe_extract_service_tier(rt, response)
            result = response
            if is_provider_fallback and rt.adapter.dialect != primary.adapter.dialect:
                # Cross-DIALECT hop: reshape the served response back to the
                # caller's native dialect BEFORE confirm/report success. (A
                # same-dialect cross-provider hop needs no reshape.) If the
                # served shape is unexpected, do not mark Solwyn billing settled.
                result = _translation.normalize_response(
                    served=rt.adapter.dialect,
                    requested=primary.adapter.dialect,
                    response=response,
                )
            # Settle OFF the hot path: build the confirm sans-I/O and enqueue
            # it with the metadata event as one ordered settlement — the same
            # path streaming on_complete uses. The caller gets the provider
            # response without waiting on a Solwyn round-trip.
            confirm = None
            reservation_id, lease_id, lease_claim_token = _settlement_keys(budget)
            if reservation_id or lease_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=reservation_id,
                    lease_id=lease_id,
                    lease_claim_token=lease_claim_token,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
                    # Nothing about this call's usage was measurable: settle the
                    # local lease reservation at its bound, never below it.
                    usage_unmeasured=usage_unmeasured,
                )
            event = self._build_metadata_event(
                model=served_model,
                provider=provider,
                input_tokens=token_details.input_tokens,
                output_tokens=token_details.output_tokens,
                token_details=token_details,
                latency_ms=ctx.elapsed_ms(),
                status=CallStatus.SUCCESS,
                is_model_fallback=is_model_fallback,
                is_provider_fallback=is_provider_fallback,
                requested_provider=primary.entry.provider if is_provider_fallback else None,
                requested_model=requested_model if is_provider_fallback else None,
                failover_reason=_success_failover_reason(
                    is_provider_fallback=is_provider_fallback,
                    is_model_fallback=is_model_fallback,
                    primary_errored=primary_errored,
                ),
                attempt_index=chain_index,
                call_id=call_id,
                service_tier=service_tier,
                agent_run=agent_run,
                provider_region=provider_region,
            )
            if confirm is not None:
                self._reporter.report_settlement(confirm, event)
            else:
                self._reporter.report(event)
            return result

        # Every candidate failed (or none was attempted): no settlement will
        # follow, so the lease reservation goes back now.
        self._budget.release_reservation(
            call_id,
            lease_claim_token=_lease_claim_token(budget),
        )
        if last_exc is not None:
            raise last_exc
        raise ProviderUnavailableError(
            "all providers unavailable",
            attempted=[r.adapter.name for r in candidates],
        )

    def _wrap_stream(
        self,
        runtime: ProviderRuntime,
        response: Any,
        ctx: _AttemptContext,
        budget: Any,
        primary: ProviderRuntime,
        *,
        requested_model: str,
        is_model_fallback: bool,
        primary_errored: bool,
        call_id: str,
        agent_run: _RunContextSnapshot,
        estimated_input_tokens: int = 0,
        estimate_empty_usage: bool = False,
    ) -> Any:
        """Wrap a streaming response, settling against the SERVED runtime.

        Return shape is dialect-dependent: Bedrock callers get back the boto3
        contract shape (a dict whose ``"stream"`` value is the wrapped event
        stream); every other dialect gets the wrapper directly. The wrapped
        ITERABLE is always the served stream itself — for a served-Bedrock hop
        that is the inner ``response["stream"]`` event stream, not the dict.
        """
        provider = runtime.adapter.name
        served_model = ctx.model
        is_provider_fallback = ctx.is_provider_fallback
        # The pre-call input estimate feeds the compat accumulators'
        # missing-usage fallback; always-reporting providers ignore it.
        accumulator = runtime.adapter.create_stream_accumulator(
            estimated_input_tokens=estimated_input_tokens
        )
        # Accumulator construction is not fail-soft wrapped: it is a pure
        # constructor with no response parsing or extraction to degrade.
        # Per-region pricing attribution for the SERVED runtime (None for
        # providers without regional pricing). Captured once, closed over.
        provider_region = _safe_extract_region(runtime)

        def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            usage_unmeasured = False
            if (
                estimate_empty_usage
                and token_details.input_tokens == 0
                and token_details.output_tokens == 0
            ):
                token_details = TokenDetails(
                    input_tokens=estimated_input_tokens,
                    is_estimated=True,
                )
                usage_unmeasured = True
            self._get_circuit_breaker(provider).record_success()
            # LatencyPolicy signal: record the SERVED provider's latency as the
            # stream settles (mirrors the non-streaming path). Pure signal store.
            self.record_latency(provider, ctx.elapsed_ms())
            # Extracted ONCE — confirm and metadata for one call_id must carry
            # the same tier or the enforcement counter and durable cost diverge.
            service_tier = accumulator.get_service_tier()
            confirm = None
            reservation_id, lease_id, lease_claim_token = _settlement_keys(budget)
            if reservation_id or lease_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=reservation_id,
                    lease_id=lease_id,
                    lease_claim_token=lease_claim_token,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
                    usage_unmeasured=usage_unmeasured,
                )
            event = self._build_metadata_event(
                model=served_model,
                provider=provider,
                input_tokens=token_details.input_tokens,
                output_tokens=token_details.output_tokens,
                token_details=token_details,
                latency_ms=ctx.elapsed_ms(),
                status=CallStatus.SUCCESS,
                is_model_fallback=is_model_fallback,
                is_provider_fallback=is_provider_fallback,
                requested_provider=primary.entry.provider if is_provider_fallback else None,
                requested_model=requested_model if is_provider_fallback else None,
                failover_reason=_success_failover_reason(
                    is_provider_fallback=is_provider_fallback,
                    is_model_fallback=is_model_fallback,
                    primary_errored=primary_errored,
                ),
                attempt_index=ctx.attempt_index,
                call_id=call_id,
                service_tier=service_tier,
                agent_run=agent_run,
                provider_region=provider_region,
            )
            if confirm is not None:
                self._reporter.report_settlement(confirm, event)
            else:
                self._reporter.report(event)

        def on_error(_exc: Exception) -> None:
            self._get_circuit_breaker(provider).record_failure()
            # A stream that dies mid-flight never reaches on_complete, so its
            # lease reservation is handed back here (the _settled guard makes
            # on_complete / on_error mutually exclusive).
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            self._reporter.report(
                self._build_error_event(
                    model=served_model,
                    provider=provider,
                    latency_ms=ctx.elapsed_ms(),
                    is_model_fallback=is_model_fallback,
                    is_provider_fallback=is_provider_fallback,
                    requested_provider=primary.entry.provider if is_provider_fallback else None,
                    requested_model=requested_model if is_provider_fallback else None,
                    attempt_index=ctx.attempt_index,
                    call_id=call_id,
                    possibly_succeeded=True,
                    agent_run=agent_run,
                    provider_region=provider_region,
                )
            )

        # Cross-DIALECT hop: reshape each served chunk back to the caller's
        # native streaming dialect. Same-dialect (primary / same-provider
        # model swap / compat-to-compat failover) passes None -> zero
        # translation. The accumulator still observes the RAW served chunk
        # inside the wrapper, so usage settles against the served provider.
        chunk_translator = (
            _make_chunk_translator(
                served=runtime.adapter.dialect, requested=primary.adapter.dialect
            )
            if is_provider_fallback and runtime.adapter.dialect != primary.adapter.dialect
            else None
        )
        # The SERVED adapter owns its stream nesting (Bedrock wraps the INNER
        # event stream); the PRIMARY adapter owns the caller-dialect result
        # shape (Bedrock callers get the boto3 contract dict back).
        source = runtime.adapter.unwrap_stream_source(response)
        wrapper = SyncStreamWrapper(
            source, accumulator, on_complete, on_error, chunk_translator=chunk_translator
        )
        return primary.adapter.wrap_stream_result(wrapper, response)

    def close(self) -> None:
        """Shut down the reporter and close HTTP clients."""
        self._reporter.close()
        self._budget.close()

    def __enter__(self) -> Solwyn:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        """Resolve public pass-throughs through the contextual capability guard."""
        return self._resolve_public_attribute(
            self._client,
            name=name,
            path=name,
            source=SurfaceSource.RAW,
        )


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class AsyncSolwyn(_SolwynBase):
    """Asynchronous Solwyn client wrapper.

    Same API and behaviour as Solwyn, but async.

    Usage::

        from openai import AsyncOpenAI
        from solwyn import AsyncSolwyn

        async with AsyncSolwyn(
            AsyncOpenAI(),
            api_key="sk_proj_...",
        ) as client:
            response = await client.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """

    def __init__(
        self,
        client: object,
        *,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        fallback: object = None,
        default_params: dict[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        on_unmetered: Literal["warn", "raise", "allow"] | None = None,
        acknowledge_untracked: Collection[str] | None = None,
        selection_policy: SelectionPolicy | None = None,
        **config_kwargs: object,
    ) -> None:
        # See sync Solwyn.__init__ for why _client is typed Any.
        self._client: Any = client

        if "project_id" in config_kwargs:
            raise TypeError("unexpected keyword argument 'project_id'")

        # Build the [primary, *fallbacks] runtime chain (constructed up front).
        # ``provider`` optionally overrides auto-detection for the primary.
        fallback_specs = _normalize_fallback(fallback)
        runtimes = build_runtimes(client, model, fallback_specs, primary_provider=provider)

        # The primary runtime's adapter is the detected (or overridden)
        # provider identity for usage extraction and proxy selection.
        self._adapter = runtimes[0].adapter
        self._dialect = runtimes[0].adapter.dialect

        # cfg_kwargs stays dict[str, Any]: mypy can't verify Pydantic's **kwargs
        # validation against SolwynConfig's typed fields, so tightening here
        # adds noise without type-safety gain. SolwynConfig validates at runtime.
        cfg_kwargs: dict[str, Any] = {
            "providers": [rt.entry for rt in runtimes],
            "default_params": default_params or {},
            "tags": tags,
            **config_kwargs,
        }
        if api_key is not None:
            cfg_kwargs["api_key"] = api_key
        if on_unmetered is not None:
            cfg_kwargs["on_unmetered"] = on_unmetered
        if acknowledge_untracked is not None:
            cfg_kwargs["acknowledge_untracked"] = acknowledge_untracked
        try:
            config = SolwynConfig(**cfg_kwargs)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else None
            raise ConfigurationError(
                first["msg"] if first else str(exc),
                field=str(first["loc"][-1]) if first else None,
            ) from exc
        super().__init__(config, runtimes, selection_policy=selection_policy, mode="async")

        self._budget = AsyncBudgetEnforcer(
            api_url=config.api_url,
            api_key=config.api_key,
            budget_mode=config.budget_mode,
            fail_open=config.fail_open,
            cache_ttl=config.budget_check_cache_ttl,
            control_plane_breaker=self._control_plane_breaker,
            # PJ-2: the SDK instance id IS the lease holder identity.
            holder_id=self._sdk_instance_id,
            lease_enabled=config.lease_enabled,
            lease_output_bound_default=config.lease_output_bound_default,
        )

        self._reporter = AsyncMetadataReporter(
            config.api_url,
            config.api_key,
            batch_size=config.reporter_batch_size,
            flush_interval=config.reporter_flush_interval,
            max_queue_size=config.reporter_max_queue_size,
            max_in_flight=config.reporter_max_in_flight,
            breaker_snapshots=self._get_breaker_snapshots,
            sdk_instance_id=self._sdk_instance_id,
            breaker_reporting_enabled=config.breaker_reporting_enabled,
            report_untracked_surfaces=config.report_untracked_surfaces,
            breaker_report_heartbeat=config.breaker_report_heartbeat,
            control_plane_breaker=self._control_plane_breaker,
            max_send_attempts=config.reporter_max_send_attempts,
            retry_backoff_base=config.reporter_retry_backoff_base,
            retry_backoff_cap=config.reporter_retry_backoff_cap,
            shutdown_deadline=config.reporter_shutdown_deadline,
        )
        self._untracked_observation_notifier = (
            self._reporter.observe_untracked_surface if config.report_untracked_surfaces else None
        )

    @functools.cached_property
    def chat(self) -> _AsyncChatProxy:
        """Return an async proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _AsyncChatProxy(self)

    @functools.cached_property
    def responses(self) -> Any:
        """Expose metered Responses create and parse for native OpenAI clients.

        The async proxy mirrors the sync contract: native foreground create and
        non-streaming parse are intercepted through the primary Responses
        pipeline; all other leaves and every OpenAI-compatible provider remain
        guarded raw operations. Cached because provider identity is
        construction-time state.
        """
        if (
            self._adapter.name == "openai"
            and self._inspect_static_attribute(self._client, "responses") is not None
        ):
            return _AsyncResponsesProxy(self)
        return self._resolve_public_attribute(
            self._client,
            name="responses",
            path="responses",
            source=SurfaceSource.RAW,
        )

    @functools.cached_property
    def embeddings(self) -> _AsyncEmbeddingsProxy:
        """Return an async proxy that routes embeddings.create() through the media lifecycle.

        Unconditional (like ``chat``): the embeddings surface is the openai
        dialect, shared by native OpenAI and every OpenAI-compatible provider.
        On a non-openai client ``.create()`` fails loud with
        ``UnsupportedSurfaceError``. Cached: provider is fixed at construction.
        """
        return _AsyncEmbeddingsProxy(self)

    @functools.cached_property
    def images(self) -> _AsyncImagesProxy:
        """Return an async proxy that routes images.generate()/edit() through the lifecycle.

        Unconditional (like ``chat`` / ``embeddings``): the images surface is the
        openai dialect, shared by native OpenAI (token-billed gpt-image) and every
        OpenAI-compatible provider (per-image). On a non-openai client
        ``.generate()`` / ``.edit()`` fails loud with ``UnsupportedSurfaceError``.
        Cached: provider is fixed at construction.
        """
        return _AsyncImagesProxy(self)

    @functools.cached_property
    def audio(self) -> _AsyncAudioProxy:
        """Return an async proxy that routes audio transcriptions/speech through the lifecycle.

        Unconditional (like ``chat`` / ``embeddings`` / ``images``): the audio
        transcriptions AND speech (TTS) surfaces are the openai dialect, shared by
        native OpenAI and every OpenAI-compatible provider (incl. Groq whisper). On
        a non-openai client ``.transcriptions.create()`` / ``.speech.create()``
        fail loud with ``UnsupportedSurfaceError``. The proxy's ``translations``
        sub-surface uses the shared untracked posture. Cached: provider is fixed at
        construction.
        """
        return _AsyncAudioProxy(self)

    @functools.cached_property
    def videos(self) -> _AsyncVideosProxy:
        """Return an async proxy that routes videos.create() (Sora) through the lifecycle.

        Unconditional (like ``chat`` / ``embeddings`` / ``images`` / ``audio``):
        the video surface is the openai dialect. Sora is OpenAI-only, so on a
        non-openai client (including OpenAI-compatible profiles) ``.create()`` fails
        loud with ``UnsupportedSurfaceError``. The returned async video job is
        passed back untouched — callers poll it themselves. Cached: provider is
        fixed at construction.
        """
        return _AsyncVideosProxy(self)

    @functools.cached_property
    def messages(self) -> Any:
        """Anthropic-compatible: client.messages.create() goes through interception.

        Cached: the dialect is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._dialect == "anthropic":
            return _AsyncMessagesProxy(self)
        return self._resolve_public_attribute(
            self._client,
            name="messages",
            path="messages",
            source=SurfaceSource.RAW,
        )

    @functools.cached_property
    def models(self) -> Any:
        """google-genai: client.models generation goes through interception.

        Cached: the client shape is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._surface_context.client_shape == "google_genai":
            return _AsyncModelsProxy(self)
        return self._resolve_public_attribute(
            self._client,
            name="models",
            path="models",
            source=SurfaceSource.RAW,
        )

    async def converse(self, **kwargs: Any) -> Any:
        """Bedrock-compatible: client.converse(modelId=...) goes through interception."""
        self._enforce_explicit_surface("converse", source=SurfaceSource.WRAPPER)
        if self._dialect == "bedrock":
            return await self._intercepted_call(**_bedrock_internal_kwargs(kwargs))
        return await self._client.converse(**kwargs)

    async def converse_stream(self, **kwargs: Any) -> Any:
        """Bedrock-compatible streaming: returns the boto3 dict with a wrapped stream."""
        self._enforce_explicit_surface("converse_stream", source=SurfaceSource.WRAPPER)
        if self._dialect == "bedrock":
            return await self._intercepted_call(
                _force_stream=True, **_bedrock_internal_kwargs(kwargs)
            )
        return await self._client.converse_stream(**kwargs)

    async def invoke_model(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        self._enforce_explicit_surface(
            "invoke_model",
            source=SurfaceSource.WRAPPER,
            blocked_reason=_INVOKE_MODEL_GUIDANCE,
        )
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return await self._client.invoke_model(**kwargs)

    async def invoke_model_with_response_stream(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        self._enforce_explicit_surface(
            "invoke_model_with_response_stream",
            source=SurfaceSource.WRAPPER,
            blocked_reason=_INVOKE_MODEL_GUIDANCE,
        )
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return await self._client.invoke_model_with_response_stream(**kwargs)

    async def start_async_invoke(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's async invocation surface (untracked video-scale spend)."""
        self._enforce_explicit_surface(
            "start_async_invoke",
            source=SurfaceSource.WRAPPER,
            blocked_reason=_START_ASYNC_INVOKE_GUIDANCE,
        )
        if self._dialect == "bedrock":
            raise ConfigurationError(_START_ASYNC_INVOKE_GUIDANCE, field="start_async_invoke")
        return await self._client.start_async_invoke(**kwargs)

    async def _async_dispatch(
        self,
        runtime: ProviderRuntime,
        kwargs: dict[str, object],
        *,
        is_streaming: bool,
        timeout: float,
        read_timeout: float,
        max_retries: int,
        surface: str = "chat",
        responses_leaf: str = "create",
    ) -> Any:
        """Dispatch one hop to the runtime's async SDK client. Pure I/O.

        Mirrors ``_sync_dispatch``: ``timeout`` is the CONNECT/POOL slice of
        the failover window, ``read_timeout`` is the decoupled per-hop
        READ/WRITE bound (PJ-8/R7); every provider-specific quirk lives on
        the adapter's ``prepare_call``, and the served hop's stream-method
        selection is driven by the dispatch-level ``is_streaming`` boolean
        (fix [A]). The adapter returns the same attribute path on an async
        client — only the await differs here.
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(
                timeout=_hop_httpx_timeout(timeout, read_timeout),
                max_retries=max_retries,
            )
        if surface == "responses":
            prepare = getattr(runtime.adapter, "prepare_responses_call", None)
            if prepare is None:
                raise UnsupportedSurfaceError(
                    surface=f"responses.{responses_leaf}", provider=runtime.adapter.name
                )
            method, call_kwargs = prepare(
                client,
                cast("dict[str, Any]", kwargs),
                is_streaming=is_streaming,
                leaf=responses_leaf,
            )
            return await method(**call_kwargs)
        method, call_kwargs = runtime.adapter.prepare_call(
            client,
            cast("dict[str, Any]", kwargs),
            is_streaming=is_streaming,
            timeout=read_timeout,
            max_retries=max_retries,
        )
        return await method(**call_kwargs)

    async def _media_dispatch(
        self,
        runtime: ProviderRuntime,
        surface: str,
        kwargs: dict[str, object],
        *,
        timeout: float,
        read_timeout: float,
        max_retries: int,
    ) -> Any:
        """Dispatch one media hop to the runtime's async SDK client. Pure I/O.

        Mirrors the sync ``_media_dispatch``: ``timeout`` is the CONNECT/POOL
        slice of the failover window, ``read_timeout`` is the decoupled
        per-hop READ/WRITE bound (PJ-8/R7), applied via ``with_options`` where
        available; the adapter's ``prepare_media_call`` seam returns the same
        attribute path on the async client — only the await differs here. No
        streaming and no candidate walk (primary-only).
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(
                timeout=_hop_httpx_timeout(timeout, read_timeout),
                max_retries=max_retries,
            )
        method, call_kwargs = _media_prepare(
            runtime.adapter, surface, client, kwargs, timeout=read_timeout, max_retries=max_retries
        )
        return await method(**call_kwargs)

    async def _media_call(self, spec: MediaSurfaceSpec, **kwargs: object) -> Any:
        """Async lean lifecycle for a non-chat media surface.

        Mirror of the sync ``_media_call``: estimate -> budget check -> provider
        call -> extract/measure -> confirm + report. No candidate walk, no
        translation, ``is_model_fallback`` always False; served by the PRIMARY
        runtime alone. Billable quantity comes from the spec hooks (response
        usage first, request-derived fallback) and stays None when unobservable.
        """
        agent_run = _capture_run_context(
            kwargs.pop("solwyn_tags", None),
            default_tags=self._config.tags,
        )
        requested_model = cast(str, kwargs["model"])
        call_id = str(uuid.uuid4())
        runtime = self._runtimes[0]
        provider = runtime.adapter.name
        deadline = Deadline(self._config.failover_total_timeout)

        char_count = estimate_content_length(kwargs)
        est_in = estimate_tokens_from_length(char_count, provider=provider) if char_count else 0
        estimated_media = spec.estimate_media(kwargs) if spec.estimate_media is not None else None

        budget = await self._budget.check_budget(
            estimated_input_tokens=est_in,
            model=requested_model,
            provider=provider,
            timeout=_budget_timeout(deadline, self._config.budget_check_timeout),
            modality=spec.modality,
            estimated_media=estimated_media,
            agent_run_id=agent_run[0],
            tags=agent_run[2],
            call_id=call_id,
        )
        # PJ-8/R12: ONE immutable tuning snapshot per call - the walk below
        # must never re-read self._config (the directive writer mutates it
        # under a lock; unlocked re-reads can tear).
        tuning = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        deadline.replace_total(tuning.failover_total_timeout)
        self._reporter.observe_project_id(budget.project_id)
        if budget.price_hints is not None:
            self.update_price_hints(budget.price_hints)

        provider_region = runtime.adapter.extract_region(runtime.sdk_client)
        if not budget.allowed:
            try:
                self._reporter.report(
                    self._build_metadata_event(
                        model=requested_model,
                        provider=provider,
                        input_tokens=est_in,
                        output_tokens=0,
                        token_details=None,
                        latency_ms=0.0,
                        status=CallStatus.BUDGET_DENIED,
                        is_model_fallback=False,
                        call_id=call_id,
                        agent_run=agent_run,
                        provider_region=provider_region,
                        modality=spec.modality,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to report budget_denied metadata event: %s",
                    type(exc).__name__,
                )
            raise _budget_denial_error(
                budget=budget,
                agent_run_id=agent_run[0],
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
            )

        # Deadline gate, mirroring the sync media path and the chat walk: with
        # connect and read decoupled (PJ-8/R7), an expired window would
        # otherwise let a warm pooled connection clear the 0.001s connect floor
        # and read for the full hop bound. See the sync mirror.
        if deadline.expired():
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            raise ProviderUnavailableError(
                "failover deadline expired",
                attempted=[provider],
            )

        start = time.monotonic()
        try:
            response = await self._media_dispatch(
                runtime,
                spec.surface,
                kwargs,
                timeout=_hop_connect_slice(deadline, 1),
                read_timeout=tuning.failover_hop_read_timeout,
                max_retries=0,
            )
        except Exception as exc:
            # Nothing will settle this call: hand any lease reservation back
            # rather than stranding it until the 900s sweep.
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            self._reporter.report(
                self._build_error_event(
                    model=requested_model,
                    provider=provider,
                    latency_ms=(time.monotonic() - start) * 1000,
                    is_model_fallback=False,
                    failover_error_class=type(exc).__name__,
                    call_id=call_id,
                    agent_run=agent_run,
                    provider_region=provider_region,
                )
            )
            raise
        latency_ms = (time.monotonic() - start) * 1000

        # Fail-soft bookkeeping (R5): the provider has answered — a surface-spec
        # or adapter raise must not destroy the paid media response. Usage
        # degrades to the request-derived measure, then to None; a None/None
        # pair simply skips the confirm exactly as before. Mirrors the sync
        # ``_media_call``.
        token_details: TokenDetails | None = None
        try:
            token_details = spec.extract_usage(response)
        except Exception as exc:
            logger.warning(
                "settlement.media_extract_usage_failed_fail_soft: %s",
                type(exc).__name__,
            )
        if token_details is None:
            try:
                token_details = spec.measure_request(kwargs)
            except Exception as exc:
                logger.warning(
                    "settlement.media_measure_request_failed_fail_soft: %s",
                    type(exc).__name__,
                )
        media_usage = None
        if spec.measure_media is not None:
            try:
                media_usage = spec.measure_media(kwargs, response)
            except Exception as exc:
                logger.warning(
                    "settlement.media_measure_media_failed_fail_soft: %s",
                    type(exc).__name__,
                )

        # Settle OFF the hot path: build the confirm sans-I/O and enqueue it
        # with the metadata event as one ordered settlement (same path as chat
        # + streaming). Confirm fires when EITHER basis is observable; skipped
        # only when both are None. See the sync mirror.
        service_tier = _safe_extract_service_tier(runtime, response)
        confirm = None
        reservation_id, lease_id, lease_claim_token = _settlement_keys(budget)
        if (reservation_id or lease_id) and (token_details is not None or media_usage is not None):
            confirm = self._budget.build_confirm_request(
                reservation_id=reservation_id,
                lease_id=lease_id,
                lease_claim_token=lease_claim_token,
                model=requested_model,
                token_details=token_details if token_details is not None else TokenDetails(),
                provider=provider,
                is_provider_fallback=False,
                call_id=call_id,
                provider_region=provider_region,
                service_tier=service_tier,
                modality=spec.modality,
                media_usage=media_usage,
            )
        event = self._build_metadata_event(
            model=requested_model,
            provider=provider,
            input_tokens=token_details.input_tokens if token_details is not None else 0,
            output_tokens=token_details.output_tokens if token_details is not None else 0,
            token_details=token_details,
            latency_ms=latency_ms,
            status=CallStatus.SUCCESS,
            is_model_fallback=False,
            attempt_index=0,
            call_id=call_id,
            service_tier=service_tier,
            agent_run=agent_run,
            provider_region=provider_region,
            modality=spec.modality,
            media_usage=media_usage,
        )
        if confirm is not None:
            self._reporter.report_settlement(confirm, event)
        else:
            self._reporter.report(event)
        return response

    async def _intercepted_call(
        self,
        *,
        _force_stream: bool = False,
        _surface: str = "chat",
        _responses_leaf: str = "create",
        **kwargs: object,
    ) -> Any:
        """Async core interception logic: the classified candidate walk."""
        agent_run = _capture_run_context(
            kwargs.pop("solwyn_tags", None),
            default_tags=self._config.tags,
        )
        requested_model = cast(str, kwargs["model"])
        # One reconciliation join key per intercepted call: see the sync
        # _intercepted_call for the join/dedup contract.
        call_id = str(uuid.uuid4())
        primary = self._runtimes[0]
        if (
            _surface == "responses"
            and getattr(primary.adapter, "prepare_responses_call", None) is None
        ):
            raise UnsupportedSurfaceError(
                surface=f"responses.{_responses_leaf}", provider=primary.adapter.name
            )
        deadline = Deadline(self._config.failover_total_timeout)

        request_semantics = kwargs
        responses_idempotent_override: bool | None = None
        if _surface == "responses":
            # Mirrors the sync path: caller routing metadata stays private,
            # while non-filtered defaults retain their pass-through semantics.
            responses_idempotent_override = cast(
                "bool | None", kwargs.pop("solwyn_idempotent", None)
            )
            request_semantics = _effective_responses_kwargs(
                global_defaults=self._config.default_params,
                primary_defaults=primary.entry.default_params,
                kwargs=kwargs,
            )
            _reject_responses_background(request_semantics)
        is_streaming = (
            _responses_is_streaming(
                request_semantics,
                leaf=_responses_leaf,
                force_stream=_force_stream,
            )
            if _surface == "responses"
            else bool(request_semantics.get("stream", False)) or _force_stream
        )

        char_count = (
            estimate_responses_content_length(cast("dict[str, Any]", request_semantics))
            if _surface == "responses"
            else estimate_content_length(request_semantics)
        )
        est_in = (
            estimate_tokens_from_length(char_count, provider=primary.adapter.name)
            if char_count
            else 0
        )

        if _surface == "responses":
            fallback_providers: list[str] = []
            fallback_models: list[str] = []
            estimated_output_bound = _responses_output_bound(
                request_semantics,
                self._config.lease_output_bound_default,
            )
        else:
            fallback_providers = [r.entry.provider.value for r in self._runtimes[1:]]
            fallback_models = [r.entry.model for r in self._runtimes[1:]]
            estimated_output_bound = _effective_output_bound(
                primary=primary,
                runtimes=self._runtimes,
                global_defaults=self._config.default_params,
                kwargs=kwargs,
                default_bound=self._config.lease_output_bound_default,
            )
        if _surface == "responses":
            budget = await self._budget.check_budget(
                estimated_input_tokens=est_in,
                model=requested_model,
                provider=primary.adapter.name,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=_budget_timeout(deadline, self._config.budget_check_timeout),
                modality="text",
                agent_run_id=agent_run[0],
                tags=agent_run[2],
                call_id=call_id,
                estimated_output_bound=estimated_output_bound,
            )
        else:
            budget = await self._budget.check_budget(
                estimated_input_tokens=est_in,
                model=requested_model,
                provider=primary.adapter.name,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=_budget_timeout(deadline, self._config.budget_check_timeout),
                agent_run_id=agent_run[0],
                tags=agent_run[2],
                call_id=call_id,
                estimated_output_bound=estimated_output_bound,
            )
        # PJ-8/R12: ONE immutable tuning snapshot per call - the walk below
        # must never re-read self._config (the directive writer mutates it
        # under a lock; unlocked re-reads can tear).
        tuning = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        deadline.replace_total(tuning.failover_total_timeout)
        self._reporter.observe_project_id(budget.project_id)
        # Refresh the CostPolicy signal from the server. Hints PERSIST across
        # hint-less responses (cache hits) until the server sends new ones — see
        # the sync _intercepted_call for the rationale.
        if budget.price_hints is not None:
            self.update_price_hints(budget.price_hints)

        if not budget.allowed:
            # See the sync _intercepted_call: region rides the denied event.
            try:
                event = self._build_metadata_event(
                    model=requested_model,
                    provider=primary.adapter.name,
                    input_tokens=est_in,
                    output_tokens=0,
                    token_details=None,
                    latency_ms=0.0,
                    status=CallStatus.BUDGET_DENIED,
                    is_model_fallback=False,
                    call_id=call_id,
                    agent_run=agent_run,
                    provider_region=primary.adapter.extract_region(primary.sdk_client),
                )
                self._reporter.report(event)
            except Exception as exc:
                logger.warning(
                    "Failed to report budget_denied metadata event: %s",
                    type(exc).__name__,
                )

            raise _budget_denial_error(
                budget=budget,
                agent_run_id=agent_run[0],
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
            )

        idempotent_override = (
            responses_idempotent_override
            if _surface == "responses"
            else cast("bool | None", kwargs.pop("solwyn_idempotent", None))
        )
        if idempotent_override is True:
            effective_idempotency = "always"
        elif idempotent_override is False:
            effective_idempotency = "safe"
        else:
            effective_idempotency = tuning.failover_idempotency
        allow_cross_provider = effective_idempotency != "never"
        allow_ambiguous_failover = effective_idempotency == "always"

        candidates = self._select_candidates(
            RoutingRequest(
                requested_provider=primary.entry.provider,
                estimated_input_tokens=est_in,
            )
        )
        if not allow_cross_provider:
            candidates = [c for c in candidates if c.entry.provider == primary.entry.provider]
        if _surface == "responses":
            # v1 is native-primary only; see the synchronous candidate walk.
            candidates = [c for c in candidates if c is primary]
        if not candidates:
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            raise ProviderUnavailableError("all providers unavailable", attempted=[])
        if deadline.remaining() <= 0.0:
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            raise ProviderUnavailableError(
                "failover deadline expired",
                attempted=[r.adapter.name for r in candidates],
            )

        failed_providers: set[str] = set()
        last_exc: Exception | None = None
        # Fix [A]: did the PRIMARY runtime get attempted-and-error in this
        # walk? Drives the cross-provider success reason (PRIMARY_ERROR if the
        # primary was attempted and raised; CIRCUIT_OPEN if it was skipped OPEN).
        primary_errored = False
        for idx, rt in enumerate(candidates):
            if deadline.expired():
                break
            provider = rt.adapter.name
            cb = self._get_circuit_breaker(provider)
            admission = cb.admit()
            if not admission.allowed:
                continue

            is_primary = rt is primary
            is_provider_fallback = rt.entry.provider != primary.entry.provider
            is_model_fallback = (not is_provider_fallback) and not is_primary
            # Fix [B]: attempt_index is the served runtime's position in the
            # CONFIGURED chain (0=primary, 1=first fallback, ...),
            # NOT the candidate-walk index. When the primary breaker is
            # OPEN-not-eligible it is dropped from the health-filtered candidate
            # list, so the walk index would mislabel the first fallback as 0 and
            # corrupt the dashboard chain-depth funnel. The per-hop timeout slice
            # still uses the candidate-walk ``idx`` (remaining candidates, not
            # chain depth).
            chain_index = next(i for i, r in enumerate(self._runtimes) if r is rt)

            # Build native kwargs for this hop. A cross-provider hop runs the
            # translation contract and may RAISE an Untranslatable* error here,
            # BEFORE any network call — that aborts the WHOLE chain:
            # do NOT classify it as transport, advance, or record a breaker failure.
            # This is a no-health-signal abort: if admit() already consumed
            # this breaker's single HALF_OPEN probe slot, free it before the error
            # propagates so the provider is not stranded HALF_OPEN with no probe.
            try:
                if _surface == "responses":
                    call_kwargs = dict(request_semantics)
                    served_model = requested_model
                else:
                    call_kwargs = _build_hop_kwargs(
                        primary=primary,
                        rt=rt,
                        is_primary=is_primary,
                        is_provider_fallback=is_provider_fallback,
                        is_streaming=is_streaming,
                        global_defaults=self._config.default_params,
                        kwargs=kwargs,
                    )
                    served_model = requested_model if is_primary else rt.entry.model
                    if is_streaming:
                        call_kwargs = rt.adapter.prepare_streaming(
                            call_kwargs, cross_provider=is_provider_fallback
                        )
            except Exception:
                cb.release_probe(admission)
                self._budget.release_reservation(
                    call_id,
                    lease_claim_token=_lease_claim_token(budget),
                )
                raise

            # Same-provider retry budget for THIS chain entry (mirrors the sync
            # walk); consumed inside the inner attempt loop below.
            same_retries_left = tuning.same_provider_retries
            advanced = False
            while True:
                ctx = _AttemptContext(
                    model=served_model,
                    start_time=time.monotonic(),
                    is_provider_fallback=is_provider_fallback,
                    attempt_index=chain_index,
                )
                try:
                    response = await self._async_dispatch(
                        rt,
                        call_kwargs,
                        is_streaming=is_streaming,
                        # Per-hop bounds (PJ-8/R7): connect/pool get a
                        # shrinking slice of the remaining FAILOVER window so a
                        # pre-send hang cannot eat the whole budget; read/write
                        # get the decoupled hop read bound (a read timeout is
                        # POST_SEND_AMBIGUOUS - never failover - so the chain
                        # deadline must not cut legitimate slow generations).
                        timeout=_hop_connect_slice(deadline, len(candidates) - idx),
                        read_timeout=tuning.failover_hop_read_timeout,
                        max_retries=0,
                        surface=_surface,
                        responses_leaf=_responses_leaf,
                    )
                    if (
                        is_streaming
                        and _client_shape(rt.sdk_client, rt.adapter.dialect) == "google_genai"
                    ):
                        # Async google-genai is lazy and needs first-chunk
                        # establishment inside this try. The exact SERVED runtime
                        # shape keeps fallback hops correct; legacy
                        # google.generativeai has no async client and its sync
                        # response wrapper establishes eagerly during dispatch.
                        response = await _materialize_stream_async(response)
                except Exception as exc:
                    disp = classify_exception(exc)
                    # Fix [A]: the PRIMARY was attempted and raised in this walk
                    # -> a later cross-provider success is a REACTIVE failover
                    # (PRIMARY_ERROR), not a proactive breaker-open reroute.
                    if is_primary:
                        primary_errored = True
                    # Same-provider retry: a 429 the provider asked us to retry (a
                    # usable Retry-After that fits the remaining deadline, leaving a
                    # min hop for the re-attempt) sleeps then re-attempts the SAME
                    # provider before burning a cross-provider hop. We HOLD this
                    # admission across the sleep — an unresolved 429 is neither a
                    # success nor a failure, so NO breaker verdict is recorded and the
                    # HALF_OPEN probe slot stays ours (never stranded). The terminal
                    # outcome (success below, or the exhausted/unretryable failure
                    # here) is the single verdict that frees the slot.
                    if disp is Disposition.FAILOVER and same_retries_left > 0:
                        retry_delay = retry_after_seconds(exc)
                        if (
                            retry_delay is not None
                            and retry_delay + _MIN_HOP_TIMEOUT <= deadline.remaining()
                        ):
                            same_retries_left -= 1
                            await asyncio.sleep(retry_delay)
                            if not deadline.expired():
                                continue  # re-attempt the SAME candidate
                    # Breaker accounting: count FAILOVER + POST_SEND_AMBIGUOUS
                    # (real health signals); skip FAIL_FAST (request-shaped, not a
                    # health signal). Same-provider double-count guard.
                    if disp is not Disposition.FAIL_FAST and provider not in failed_providers:
                        cb.record_failure()
                        failed_providers.add(provider)
                    else:
                        # No NEW health verdict for this hop: FAIL_FAST is request-shaped,
                        # or this provider was already counted this walk (double-count
                        # guard). If the hop consumed a HALF_OPEN probe slot, free it
                        # (no state change) so the breaker is not stranded HALF_OPEN.
                        cb.release_probe(admission)
                    # A correctly-not-failed-over post-send-ambiguous abort
                    # emits an ERROR event with possibly_succeeded=True so the Cloud API
                    # can reconcile the (possibly-landed, never-confirmed) reservation.
                    possibly_succeeded = (
                        disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover
                    )
                    self._reporter.report(
                        self._build_error_event(
                            model=served_model,
                            provider=provider,
                            latency_ms=ctx.elapsed_ms(),
                            is_model_fallback=is_model_fallback,
                            is_provider_fallback=is_provider_fallback,
                            requested_provider=(
                                primary.entry.provider if is_provider_fallback else None
                            ),
                            requested_model=requested_model if is_provider_fallback else None,
                            failover_error_class=type(exc).__name__,
                            attempt_index=chain_index,
                            call_id=call_id,
                            possibly_succeeded=True if possibly_succeeded else None,
                            agent_run=agent_run,
                            provider_region=_safe_extract_region(rt),
                        )
                    )
                    if disp is Disposition.FAIL_FAST:
                        self._budget.release_reservation(
                            call_id,
                            lease_claim_token=_lease_claim_token(budget),
                        )
                        raise
                    if disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover:
                        self._budget.release_reservation(
                            call_id,
                            lease_claim_token=_lease_claim_token(budget),
                        )
                        raise
                    last_exc = exc
                    advanced = True
                # Reached on a successful hop OR on a terminal failure that advances
                # the chain; a same-provider retry `continue`s above and never lands
                # here. `advanced` distinguishes the two so success falls through to
                # settlement below.
                break
            if advanced:
                continue

            # SUCCESS — settle against the SERVED runtime.
            #
            # Fix [A]: for the STREAMING branch we do NOT credit the breaker here.
            # The single success is settled ONLY when the stream completes, by the
            # wrapper's on_complete (success + latency + confirm + metadata once,
            # via the _settled guard). Crediting both here AND in on_complete would
            # double-credit a HALF_OPEN breaker (closing it after one streaming
            # probe), and a stream that establishes then errors mid-flight would
            # log a spurious success before its on_error failure. So record_success()
            # runs ONLY on the non-streaming path, AFTER the streaming early return.
            if is_streaming:
                return self._wrap_stream_async(
                    rt,
                    response,
                    ctx,
                    budget,
                    primary,
                    requested_model=requested_model,
                    is_model_fallback=is_model_fallback,
                    primary_errored=primary_errored,
                    call_id=call_id,
                    agent_run=agent_run,
                    estimated_input_tokens=est_in,
                    estimate_empty_usage=_surface == "responses",
                )
            cb.record_success()

            # LatencyPolicy signal: record this served hop's success latency
            # (non-streaming). Streaming records in on_complete when the stream
            # settles. Pure signal store — no I/O, no routing change here.
            self.record_latency(provider, ctx.elapsed_ms())

            # Fail-soft bookkeeping (R5): a paid, successful response is never
            # destroyed by extraction — usage degrades to estimates
            # (is_estimated=True), region/tier degrade to None.
            token_details, usage_unmeasured = _extract_usage_fail_soft(
                rt, response, estimated_input_tokens=est_in
            )
            # Per-region pricing attribution: the SERVED runtime's endpoint region.
            provider_region = _safe_extract_region(rt)
            # Extracted ONCE from the RAW served response — confirm and
            # metadata must carry the same tier (see the sync path).
            service_tier = _safe_extract_service_tier(rt, response)
            result = response
            if is_provider_fallback and rt.adapter.dialect != primary.adapter.dialect:
                # Cross-DIALECT hop: reshape the served response back to the
                # caller's native dialect BEFORE confirm/report success. (A
                # same-dialect cross-provider hop needs no reshape.) If the
                # served shape is unexpected, do not mark Solwyn billing settled.
                result = _translation.normalize_response(
                    served=rt.adapter.dialect,
                    requested=primary.adapter.dialect,
                    response=response,
                )
            # Settle OFF the hot path: build the confirm sans-I/O and enqueue
            # it with the metadata event as one ordered settlement — the same
            # path streaming on_complete uses. The caller gets the provider
            # response without waiting on a Solwyn round-trip.
            confirm = None
            reservation_id, lease_id, lease_claim_token = _settlement_keys(budget)
            if reservation_id or lease_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=reservation_id,
                    lease_id=lease_id,
                    lease_claim_token=lease_claim_token,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
                    # Nothing about this call's usage was measurable: settle the
                    # local lease reservation at its bound, never below it.
                    usage_unmeasured=usage_unmeasured,
                )
            event = self._build_metadata_event(
                model=served_model,
                provider=provider,
                input_tokens=token_details.input_tokens,
                output_tokens=token_details.output_tokens,
                token_details=token_details,
                latency_ms=ctx.elapsed_ms(),
                status=CallStatus.SUCCESS,
                is_model_fallback=is_model_fallback,
                is_provider_fallback=is_provider_fallback,
                requested_provider=primary.entry.provider if is_provider_fallback else None,
                requested_model=requested_model if is_provider_fallback else None,
                failover_reason=_success_failover_reason(
                    is_provider_fallback=is_provider_fallback,
                    is_model_fallback=is_model_fallback,
                    primary_errored=primary_errored,
                ),
                attempt_index=chain_index,
                call_id=call_id,
                service_tier=service_tier,
                agent_run=agent_run,
                provider_region=provider_region,
            )
            if confirm is not None:
                self._reporter.report_settlement(confirm, event)
            else:
                self._reporter.report(event)
            return result

        # Every candidate failed (or none was attempted): no settlement will
        # follow, so the lease reservation goes back now.
        self._budget.release_reservation(
            call_id,
            lease_claim_token=_lease_claim_token(budget),
        )
        if last_exc is not None:
            raise last_exc
        raise ProviderUnavailableError(
            "all providers unavailable",
            attempted=[r.adapter.name for r in candidates],
        )

    def _wrap_stream_async(
        self,
        runtime: ProviderRuntime,
        response: Any,
        ctx: _AttemptContext,
        budget: Any,
        primary: ProviderRuntime,
        *,
        requested_model: str,
        is_model_fallback: bool,
        primary_errored: bool,
        call_id: str,
        agent_run: _RunContextSnapshot,
        estimated_input_tokens: int = 0,
        estimate_empty_usage: bool = False,
    ) -> Any:
        """Wrap an async streaming response, settling against the SERVED runtime.

        Return shape mirrors the sync ``_wrap_stream``: Bedrock-dialect callers
        get the boto3 contract dict with the wrapped (async) event stream under
        ``"stream"``; every other dialect gets the wrapper directly.
        """
        provider = runtime.adapter.name
        served_model = ctx.model
        is_provider_fallback = ctx.is_provider_fallback
        # The pre-call input estimate feeds the compat accumulators'
        # missing-usage fallback; always-reporting providers ignore it.
        accumulator = runtime.adapter.create_stream_accumulator(
            estimated_input_tokens=estimated_input_tokens
        )
        # Accumulator construction is not fail-soft wrapped: it is a pure
        # constructor with no response parsing or extraction to degrade.
        # Per-region pricing attribution for the SERVED runtime (None for
        # providers without regional pricing). Captured once, closed over.
        provider_region = _safe_extract_region(runtime)

        async def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            usage_unmeasured = False
            if (
                estimate_empty_usage
                and token_details.input_tokens == 0
                and token_details.output_tokens == 0
            ):
                token_details = TokenDetails(
                    input_tokens=estimated_input_tokens,
                    is_estimated=True,
                )
                usage_unmeasured = True
            self._get_circuit_breaker(provider).record_success()
            # LatencyPolicy signal: record the SERVED provider's latency as the
            # stream settles (mirrors the non-streaming path). Pure signal store.
            self.record_latency(provider, ctx.elapsed_ms())
            # Extracted ONCE — confirm and metadata must carry the same tier
            # (see the sync on_complete).
            service_tier = accumulator.get_service_tier()
            confirm = None
            reservation_id, lease_id, lease_claim_token = _settlement_keys(budget)
            if reservation_id or lease_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=reservation_id,
                    lease_id=lease_id,
                    lease_claim_token=lease_claim_token,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
                    usage_unmeasured=usage_unmeasured,
                )
            event = self._build_metadata_event(
                model=served_model,
                provider=provider,
                input_tokens=token_details.input_tokens,
                output_tokens=token_details.output_tokens,
                token_details=token_details,
                latency_ms=ctx.elapsed_ms(),
                status=CallStatus.SUCCESS,
                is_model_fallback=is_model_fallback,
                is_provider_fallback=is_provider_fallback,
                requested_provider=primary.entry.provider if is_provider_fallback else None,
                requested_model=requested_model if is_provider_fallback else None,
                failover_reason=_success_failover_reason(
                    is_provider_fallback=is_provider_fallback,
                    is_model_fallback=is_model_fallback,
                    primary_errored=primary_errored,
                ),
                attempt_index=ctx.attempt_index,
                call_id=call_id,
                service_tier=service_tier,
                agent_run=agent_run,
                provider_region=provider_region,
            )
            if confirm is not None:
                self._reporter.report_settlement(confirm, event)
            else:
                self._reporter.report(event)

        async def on_error(_exc: Exception) -> None:
            self._get_circuit_breaker(provider).record_failure()
            # A stream that dies mid-flight never reaches on_complete, so its
            # lease reservation is handed back here (the _settled guard makes
            # on_complete / on_error mutually exclusive).
            self._budget.release_reservation(
                call_id,
                lease_claim_token=_lease_claim_token(budget),
            )
            self._reporter.report(
                self._build_error_event(
                    model=served_model,
                    provider=provider,
                    latency_ms=ctx.elapsed_ms(),
                    is_model_fallback=is_model_fallback,
                    is_provider_fallback=is_provider_fallback,
                    requested_provider=primary.entry.provider if is_provider_fallback else None,
                    requested_model=requested_model if is_provider_fallback else None,
                    attempt_index=ctx.attempt_index,
                    call_id=call_id,
                    possibly_succeeded=True,
                    agent_run=agent_run,
                    provider_region=provider_region,
                )
            )

        # Cross-DIALECT hop: reshape each served chunk back to the caller's
        # native streaming dialect. Same-dialect passes None (zero
        # translation). See the sync ``_wrap_stream`` for the accounting note.
        chunk_translator = (
            _make_chunk_translator(
                served=runtime.adapter.dialect, requested=primary.adapter.dialect
            )
            if is_provider_fallback and runtime.adapter.dialect != primary.adapter.dialect
            else None
        )
        # Stream nesting and caller-dialect result shape are adapter-owned —
        # see the sync ``_wrap_stream``.
        source = runtime.adapter.unwrap_stream_source(response)
        wrapper = AsyncStreamWrapper(
            source, accumulator, on_complete, on_error, chunk_translator=chunk_translator
        )
        return primary.adapter.wrap_stream_result(wrapper, response)

    async def close(self) -> None:
        """Shut down the reporter and close HTTP clients."""
        await self._reporter.close()
        await self._budget.close()

    async def __aenter__(self) -> AsyncSolwyn:
        self._reporter.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def __getattr__(self, name: str) -> Any:
        """Resolve public pass-throughs through the contextual capability guard."""
        return self._resolve_public_attribute(
            self._client,
            name=name,
            path=name,
            source=SurfaceSource.RAW,
        )
