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
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, cast

from pydantic import ValidationError

from solwyn._base import (
    MediaSurfaceSpec,
    _AttemptContext,
    _SolwynBase,
    _warn_unmetered_spend_surface_once,
    _with_legacy_max_tokens_key,
    _with_openai_completion_token_key,
)
from solwyn._privacy import estimate_content_length, estimate_tokens_from_length
from solwyn._proxies import (
    _AsyncAudioProxy,
    _AsyncChatProxy,
    _AsyncEmbeddingsProxy,
    _AsyncImagesProxy,
    _AsyncMessagesProxy,
    _AsyncModelsProxy,
    _AsyncVideosProxy,
    _bedrock_internal_kwargs,
    _SyncAudioProxy,
    _SyncChatProxy,
    _SyncEmbeddingsProxy,
    _SyncImagesProxy,
    _SyncMessagesProxy,
    _SyncModelsProxy,
    _SyncVideosProxy,
)
from solwyn._registry import ProviderRuntime, build_runtimes
from solwyn._routing import RoutingRequest, SelectionPolicy
from solwyn._run import _capture_run_context, _RunContextSnapshot
from solwyn._token_details import TokenDetails
from solwyn._types import CallStatus, FailoverReason, ProviderName
from solwyn.budget import (
    DEFAULT_COST_PER_TOKEN,
    AsyncBudgetEnforcer,
    BudgetEnforcer,
)
from solwyn.config import SolwynConfig
from solwyn.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    ProviderUnavailableError,
    UnsupportedSurfaceError,
    UntranslatableModelError,
)
from solwyn.providers import _translation
from solwyn.providers._errors import Disposition, classify_exception, retry_after_seconds
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter
from solwyn.stream import AsyncStreamWrapper, SyncStreamWrapper

logger = logging.getLogger(__name__)


# Floor for a per-hop dispatch timeout: even when the chain deadline is nearly
# spent we give a hop at least this long rather than passing it ~0s.
_MIN_HOP_TIMEOUT = 1.0
_BUDGET_CHECK_TIMEOUT = 5.0
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


def _source_compatible_defaults(dialect: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return target defaults that are also legal in the source DIALECT.

    Keyed by dialect, not provider name: every OpenAI-compatible provider
    shares the ``openai`` key set.
    """
    allowed = _SOURCE_COMPATIBLE_DEFAULT_KEYS.get(dialect, frozenset())
    return {key: value for key, value in params.items() if key in allowed}


def _budget_timeout(deadline: Deadline) -> float:
    """Timeout for the budget pre-flight, clamped by the chain deadline."""
    return max(0.001, min(_BUDGET_CHECK_TIMEOUT, deadline.remaining()))


def _hop_timeout(deadline: Deadline, remaining_candidates: int) -> float:
    """Timeout for one provider hop, never exceeding the remaining deadline."""
    remaining = deadline.remaining()
    if remaining <= 0:
        return 0.001
    slice_timeout = remaining / max(1, remaining_candidates)
    return min(remaining, max(_MIN_HOP_TIMEOUT, slice_timeout))


class Deadline:
    """A monotonic chain deadline.

    Stamped once at ``_intercepted_call`` entry from ``failover_total_timeout``;
    it encompasses the budget pre-flight and every per-hop dispatch. Per-hop
    timeouts are derived from ``remaining()`` so a hung-but-connected provider
    cannot stack the 600s SDK read default across the chain.
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
    provider_global_defaults = {
        key: value for key, value in global_defaults.items() if key != "solwyn_tags"
    }
    provider_entry_defaults = {
        key: value for key, value in rt.entry.default_params.items() if key != "solwyn_tags"
    }
    provider_kwargs = {key: value for key, value in kwargs.items() if key != "solwyn_tags"}

    merged_defaults = {**provider_global_defaults, **provider_entry_defaults}
    merged_kwargs: dict[str, object] = {**merged_defaults, **provider_kwargs}
    if not is_provider_fallback:
        # PRIMARY hop is native passthrough; same-provider hop only swaps model.
        # Same-provider streaming (incl. model swap) keeps working unchanged.
        if is_primary:
            target_model = cast(str, merged_kwargs["model"])
            return _with_openai_completion_token_key(rt.adapter.name, target_model, merged_kwargs)
        return _with_openai_completion_token_key(
            rt.adapter.name,
            rt.entry.model,
            {**merged_kwargs, "model": rt.entry.model},
        )

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
        normalized: dict[str, object] = {
            **_with_legacy_max_tokens_key(target_name, provider_global_defaults),
            **_with_legacy_max_tokens_key(target_name, provider_entry_defaults),
            **_with_legacy_max_tokens_key(target_name, provider_kwargs),
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
        return _with_openai_completion_token_key(
            target_name,
            rt.entry.model,
            {**passthrough, "model": rt.entry.model},
        )

    # CROSS-DIALECT hop: translate via the canonical subset (may RAISE an
    # Untranslatable* error BEFORE any network call; the caller aborts the
    # chain). Translation starts from SOURCE-dialect values only: the target
    # entry's default_params may contain target-native keys such as Anthropic
    # top_k.
    source_defaults = _source_compatible_defaults(source_dialect, provider_entry_defaults)
    source_kwargs: dict[str, object] = {
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
    # Re-apply target entry defaults as fill-absent (e.g. Anthropic max_tokens).
    return {**provider_entry_defaults, **call_kwargs}


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
            model="gpt-4o",
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
            **config_kwargs,
        }
        if api_key is not None:
            cfg_kwargs["api_key"] = api_key
        try:
            config = SolwynConfig(**cfg_kwargs)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else None
            raise ConfigurationError(
                first["msg"] if first else str(exc),
                field=str(first["loc"][-1]) if first else None,
            ) from exc
        super().__init__(config, runtimes, selection_policy=selection_policy)

        # Budget enforcer
        self._budget = BudgetEnforcer(
            api_url=config.api_url,
            api_key=config.api_key,
            budget_mode=config.budget_mode,
            fail_open=config.fail_open,
            cache_ttl=config.budget_check_cache_ttl,
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
        )

    @functools.cached_property
    def chat(self) -> _SyncChatProxy:
        """Return a proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _SyncChatProxy(self)

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
        seam). The proxy's ``translations`` sub-surface warns-once then passes
        through untracked. Cached: provider is fixed at construction so this is safe
        to create once.
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
        return self._client.messages

    @functools.cached_property
    def models(self) -> Any:
        """Google-compatible: client.models.generate_content() goes through interception.

        Cached: the dialect is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._dialect == "google":
            return _SyncModelsProxy(self)
        return self._client.models

    def converse(self, **kwargs: Any) -> Any:
        """Bedrock-compatible: client.converse(modelId=...) goes through interception."""
        if self._dialect == "bedrock":
            return self._intercepted_call(**_bedrock_internal_kwargs(kwargs))
        return self._client.converse(**kwargs)

    def converse_stream(self, **kwargs: Any) -> Any:
        """Bedrock-compatible streaming: returns the boto3 dict with a wrapped stream."""
        if self._dialect == "bedrock":
            return self._intercepted_call(_force_stream=True, **_bedrock_internal_kwargs(kwargs))
        return self._client.converse_stream(**kwargs)

    def invoke_model(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return self._client.invoke_model(**kwargs)

    def invoke_model_with_response_stream(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return self._client.invoke_model_with_response_stream(**kwargs)

    def start_async_invoke(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's async invocation surface (untracked video-scale spend)."""
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
        max_retries: int,
    ) -> Any:
        """Dispatch one hop to the runtime's SDK client. Pure I/O — no metrics.

        Applies the mandatory per-hop bound via ``with_options`` (kills the
        600s SDK read default and any internal retry stacking); SDKs without
        ``with_options`` apply their own bound inside ``prepare_call``.

        The served hop's stream-method selection is driven by the dispatch-level
        ``is_streaming`` boolean — NOT the original ``_force_stream``/canonical
        flag (fix [A]). A caller who streamed via OpenAI/Anthropic ``stream=True``
        and fails over to Google/Bedrock must still hit the dedicated stream
        method, and vice versa. Every provider-specific quirk (stream kwarg vs
        dedicated method, model-key rename, HTTP bound) lives on the adapter's
        ``prepare_call`` — adding a provider never touches this method.
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        method, call_kwargs = runtime.adapter.prepare_call(
            client,
            cast("dict[str, Any]", kwargs),
            is_streaming=is_streaming,
            timeout=timeout,
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
        max_retries: int,
    ) -> Any:
        """Dispatch one media hop to the runtime's SDK client. Pure I/O — no metrics.

        The media analogue of ``_sync_dispatch``: applies the per-hop bound via
        ``with_options`` where available, then hands off to the adapter's
        ``prepare_media_call`` seam. No streaming and no candidate walk — a media
        call is served by the primary runtime alone.
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        method, call_kwargs = _media_prepare(
            runtime.adapter, surface, client, kwargs, timeout=timeout, max_retries=max_retries
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
        agent_run = _capture_run_context(kwargs.pop("solwyn_tags", None))
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
            timeout=_budget_timeout(deadline),
            modality=spec.modality,
            estimated_media=estimated_media,
            agent_run_id=agent_run[0],
        )
        effective_total = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        if effective_total is not None:
            deadline.replace_total(effective_total)
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
            raise BudgetExceededError(
                project_id=budget.project_id,
                budget_limit=budget.budget_limit,
                current_usage=budget.current_usage,
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
                budget_period="unknown",
                mode=budget.mode.value,
            )

        # 3. Provider call — PRIMARY only. A dispatch error is reported (parity
        #    with the chat error path) then re-raised unchanged (drop-in contract).
        start = time.monotonic()
        try:
            response = self._media_dispatch(
                runtime,
                spec.surface,
                kwargs,
                timeout=_hop_timeout(deadline, 1),
                max_retries=0,
            )
        except Exception as exc:
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

        # 4. Billable quantities: TOKEN basis (response usage first, request-
        #    derived fallback) AND, when the surface has a media channel, the
        #    non-token MediaUsage basis. BOTH ride the confirm when observable —
        #    the server's pricing card unit picks (e.g. native gpt-image sends
        #    token usage with image buckets AND request-derived image quantities).
        token_details = spec.extract_usage(response)
        if token_details is None:
            token_details = spec.measure_request(kwargs)
        media_usage = (
            spec.measure_media(kwargs, response) if spec.measure_media is not None else None
        )

        # 5. Confirm + report the served call (is_model_fallback=False, primary-only).
        #    Confirm fires when EITHER basis is observable; skipped only when both
        #    are None (never settle a real $0 price). When only media is observed,
        #    a zeroed TokenDetails carries the confirm's required token field.
        service_tier = runtime.adapter.extract_service_tier(response)
        if budget.reservation_id and (token_details is not None or media_usage is not None):
            self._budget.confirm_cost(
                budget.reservation_id,
                requested_model,
                token_details if token_details is not None else TokenDetails(),
                provider=provider,
                is_provider_fallback=False,
                call_id=call_id,
                provider_region=provider_region,
                service_tier=service_tier,
                modality=spec.modality,
                media_usage=media_usage,
            )
        self._reporter.report(
            self._build_metadata_event(
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
        )
        return response

    def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: object) -> Any:
        """Core interception logic: the classified candidate walk."""
        agent_run = _capture_run_context(kwargs.pop("solwyn_tags", None))
        requested_model = cast(str, kwargs["model"])
        is_streaming = bool(kwargs.get("stream", False)) or _force_stream
        # One reconciliation join key per intercepted call: threaded into
        # every served-provider metadata event AND its confirm so the Cloud API
        # can join them (and dedup cache-hit / abandoned-stream spend).
        call_id = str(uuid.uuid4())
        primary = self._runtimes[0]
        # Deadline starts here — it encompasses the budget pre-flight.
        deadline = Deadline(self._config.failover_total_timeout)

        # 1. Estimate input tokens (length-only; never materializes joined string).
        char_count = estimate_content_length(kwargs)
        est_in = (
            estimate_tokens_from_length(char_count, provider=primary.adapter.name)
            if char_count
            else 0
        )

        # 2. Check budget against the PRIMARY (we don't yet know who serves).
        budget = self._budget.check_budget(
            estimated_input_tokens=est_in,
            model=requested_model,
            provider=primary.adapter.name,
            fallback_providers=[r.entry.provider.value for r in self._runtimes[1:]],
            fallback_models=[r.entry.model for r in self._runtimes[1:]],
            timeout=_budget_timeout(deadline),
            agent_run_id=agent_run[0],
        )
        effective_total = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        if effective_total is not None:
            deadline.replace_total(effective_total)
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

            raise BudgetExceededError(
                project_id=budget.project_id,
                budget_limit=budget.budget_limit,
                current_usage=budget.current_usage,
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
                budget_period="unknown",
                mode=budget.mode.value,
            )

        # 3. Resolve per-call idempotency override (strip before dispatch).
        idempotent_override = cast(bool | None, kwargs.pop("solwyn_idempotent", None))
        if idempotent_override is True:
            effective_idempotency = "always"
        elif idempotent_override is False:
            effective_idempotency = "safe"
        else:
            effective_idempotency = self._config.failover_idempotency
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
        if not candidates:
            raise ProviderUnavailableError("all providers unavailable", attempted=[])
        if deadline.remaining() <= 0.0:
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
                raise

            # Same-provider retry budget for THIS chain entry (config seam,
            # default 0). Consumed inside the inner attempt loop below.
            same_retries_left = self._config.same_provider_retries
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
                        # Shrinking per-hop slice: divide what's left of the
                        # chain deadline across the candidates not yet attempted so a
                        # single hung hop cannot consume the whole budget.
                        timeout=_hop_timeout(deadline, len(candidates) - idx),
                        max_retries=0,
                    )
                    if is_streaming and rt.adapter.dialect == "google":
                        # First-chunk materialization — GOOGLE DIALECT ONLY (fix
                        # [B]). A Google lazy generator does no network I/O until the
                        # first pull, so we force it INSIDE this try; an establishment
                        # error then falls into the candidate-walk except ->
                        # classify_exception -> failover, exactly like OpenAI/Anthropic's
                        # eager raise_for_status. OpenAI/Anthropic .create(stream=True)
                        # ALREADY established eagerly at dispatch (its establishment
                        # errors raised above and are failover-eligible), so pre-pulling
                        # their first chunk is unnecessary AND would misclassify a
                        # post-connect first-chunk read error as pre-send (double-spend
                        # risk) — so we DON'T materialize them. No double-emit: the
                        # buffered first chunk is replayed via the wrapper.
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
                            provider_region=rt.adapter.extract_region(rt.sdk_client),
                        )
                    )
                    if disp is Disposition.FAIL_FAST:
                        raise  # 4xx/404/refusal — do NOT advance the chain
                    if disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover:
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
                )
            cb.record_success()

            # LatencyPolicy signal: record this served hop's success latency
            # (non-streaming). Streaming records in on_complete when the stream
            # settles. Pure signal store — no I/O, no routing change here.
            self.record_latency(provider, ctx.elapsed_ms())

            token_details = rt.adapter.extract_usage(response)
            # Explicit-degradation fallback: a compat provider that omitted the
            # usage block yields a length-based estimate (is_estimated=True)
            # instead of silently recording zero spend. None when usage was
            # present or the adapter always reports usage.
            estimated_details = rt.adapter.estimate_missing_usage(
                response, estimated_input_tokens=est_in
            )
            if estimated_details is not None:
                token_details = estimated_details
            # Per-region pricing attribution: the SERVED runtime's endpoint
            # region (None for providers without regional pricing).
            provider_region = rt.adapter.extract_region(rt.sdk_client)
            # The tier echoed on the RAW served response is the billing ground
            # truth. Extracted ONCE: confirm and metadata for one call_id must
            # carry the same tier or the enforcement counter and the durable
            # tier-repriced cost diverge.
            service_tier = rt.adapter.extract_service_tier(response)
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
            if budget.reservation_id:
                self._budget.confirm_cost(
                    budget.reservation_id,
                    served_model,
                    token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
                )
            self._reporter.report(
                self._build_metadata_event(
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
            )
            return result

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
        # Per-region pricing attribution for the SERVED runtime (None for
        # providers without regional pricing). Captured once, closed over.
        provider_region = runtime.adapter.extract_region(runtime.sdk_client)

        def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            self._get_circuit_breaker(provider).record_success()
            # LatencyPolicy signal: record the SERVED provider's latency as the
            # stream settles (mirrors the non-streaming path). Pure signal store.
            self.record_latency(provider, ctx.elapsed_ms())
            # Extracted ONCE — confirm and metadata for one call_id must carry
            # the same tier or the enforcement counter and durable cost diverge.
            service_tier = accumulator.get_service_tier()
            confirm = None
            if budget.reservation_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=budget.reservation_id,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
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
        """Pass through non-intercepted attributes to the underlying client."""
        attribute = getattr(self._client, name)
        _warn_unmetered_spend_surface_once(
            adapter=self._adapter, dialect=self._dialect, surface=name
        )
        return attribute


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
                model="gpt-4o",
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
            **config_kwargs,
        }
        if api_key is not None:
            cfg_kwargs["api_key"] = api_key
        try:
            config = SolwynConfig(**cfg_kwargs)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else None
            raise ConfigurationError(
                first["msg"] if first else str(exc),
                field=str(first["loc"][-1]) if first else None,
            ) from exc
        super().__init__(config, runtimes, selection_policy=selection_policy)

        self._budget = AsyncBudgetEnforcer(
            api_url=config.api_url,
            api_key=config.api_key,
            budget_mode=config.budget_mode,
            fail_open=config.fail_open,
            cache_ttl=config.budget_check_cache_ttl,
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
        )

    @functools.cached_property
    def chat(self) -> _AsyncChatProxy:
        """Return an async proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _AsyncChatProxy(self)

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
        sub-surface warns-once then passes through untracked. Cached: provider is
        fixed at construction.
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
        return self._client.messages

    @functools.cached_property
    def models(self) -> Any:
        """Google-compatible: client.models.generate_content() goes through interception.

        Cached: the dialect is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._dialect == "google":
            return _AsyncModelsProxy(self)
        return self._client.models

    async def converse(self, **kwargs: Any) -> Any:
        """Bedrock-compatible: client.converse(modelId=...) goes through interception."""
        if self._dialect == "bedrock":
            return await self._intercepted_call(**_bedrock_internal_kwargs(kwargs))
        return await self._client.converse(**kwargs)

    async def converse_stream(self, **kwargs: Any) -> Any:
        """Bedrock-compatible streaming: returns the boto3 dict with a wrapped stream."""
        if self._dialect == "bedrock":
            return await self._intercepted_call(
                _force_stream=True, **_bedrock_internal_kwargs(kwargs)
            )
        return await self._client.converse_stream(**kwargs)

    async def invoke_model(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return await self._client.invoke_model(**kwargs)

    async def invoke_model_with_response_stream(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's legacy per-model surface (budget bypass risk)."""
        if self._dialect == "bedrock":
            raise ConfigurationError(_INVOKE_MODEL_GUIDANCE, field="invoke_model")
        return await self._client.invoke_model_with_response_stream(**kwargs)

    async def start_async_invoke(self, **kwargs: Any) -> Any:
        """Fail loud on Bedrock's async invocation surface (untracked video-scale spend)."""
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
        max_retries: int,
    ) -> Any:
        """Dispatch one hop to the runtime's async SDK client. Pure I/O.

        Mirrors ``_sync_dispatch``: the per-hop bound applies via
        ``with_options`` where available, every provider-specific quirk lives
        on the adapter's ``prepare_call``, and the served hop's stream-method
        selection is driven by the dispatch-level ``is_streaming`` boolean
        (fix [A]). The adapter returns the same attribute path on an async
        client — only the await differs here.
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        method, call_kwargs = runtime.adapter.prepare_call(
            client,
            cast("dict[str, Any]", kwargs),
            is_streaming=is_streaming,
            timeout=timeout,
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
        max_retries: int,
    ) -> Any:
        """Dispatch one media hop to the runtime's async SDK client. Pure I/O.

        Mirrors the sync ``_media_dispatch``: the per-hop bound applies via
        ``with_options`` where available, then the adapter's ``prepare_media_call``
        seam returns the same attribute path on the async client — only the await
        differs here. No streaming and no candidate walk (primary-only).
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        method, call_kwargs = _media_prepare(
            runtime.adapter, surface, client, kwargs, timeout=timeout, max_retries=max_retries
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
        agent_run = _capture_run_context(kwargs.pop("solwyn_tags", None))
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
            timeout=_budget_timeout(deadline),
            modality=spec.modality,
            estimated_media=estimated_media,
            agent_run_id=agent_run[0],
        )
        effective_total = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        if effective_total is not None:
            deadline.replace_total(effective_total)
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
            raise BudgetExceededError(
                project_id=budget.project_id,
                budget_limit=budget.budget_limit,
                current_usage=budget.current_usage,
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
                budget_period="unknown",
                mode=budget.mode.value,
            )

        start = time.monotonic()
        try:
            response = await self._media_dispatch(
                runtime,
                spec.surface,
                kwargs,
                timeout=_hop_timeout(deadline, 1),
                max_retries=0,
            )
        except Exception as exc:
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

        token_details = spec.extract_usage(response)
        if token_details is None:
            token_details = spec.measure_request(kwargs)
        media_usage = (
            spec.measure_media(kwargs, response) if spec.measure_media is not None else None
        )

        # Confirm fires when EITHER basis is observable (both ride the confirm
        # when both are); skipped only when both are None. See the sync mirror.
        service_tier = runtime.adapter.extract_service_tier(response)
        if budget.reservation_id and (token_details is not None or media_usage is not None):
            await self._budget.confirm_cost(
                budget.reservation_id,
                requested_model,
                token_details if token_details is not None else TokenDetails(),
                provider=provider,
                is_provider_fallback=False,
                call_id=call_id,
                provider_region=provider_region,
                service_tier=service_tier,
                modality=spec.modality,
                media_usage=media_usage,
            )
        self._reporter.report(
            self._build_metadata_event(
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
        )
        return response

    async def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: object) -> Any:
        """Async core interception logic: the classified candidate walk."""
        agent_run = _capture_run_context(kwargs.pop("solwyn_tags", None))
        requested_model = cast(str, kwargs["model"])
        is_streaming = bool(kwargs.get("stream", False)) or _force_stream
        # One reconciliation join key per intercepted call: see the sync
        # _intercepted_call for the join/dedup contract.
        call_id = str(uuid.uuid4())
        primary = self._runtimes[0]
        deadline = Deadline(self._config.failover_total_timeout)

        char_count = estimate_content_length(kwargs)
        est_in = (
            estimate_tokens_from_length(char_count, provider=primary.adapter.name)
            if char_count
            else 0
        )

        budget = await self._budget.check_budget(
            estimated_input_tokens=est_in,
            model=requested_model,
            provider=primary.adapter.name,
            fallback_providers=[r.entry.provider.value for r in self._runtimes[1:]],
            fallback_models=[r.entry.model for r in self._runtimes[1:]],
            timeout=_budget_timeout(deadline),
            agent_run_id=agent_run[0],
        )
        effective_total = self._apply_failover_tuning_directive(
            getattr(budget, "failover_tuning_allowed", None)
        )
        if effective_total is not None:
            deadline.replace_total(effective_total)
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

            raise BudgetExceededError(
                project_id=budget.project_id,
                budget_limit=budget.budget_limit,
                current_usage=budget.current_usage,
                estimated_cost=est_in * DEFAULT_COST_PER_TOKEN,
                budget_period="unknown",
                mode=budget.mode.value,
            )

        idempotent_override = cast(bool | None, kwargs.pop("solwyn_idempotent", None))
        if idempotent_override is True:
            effective_idempotency = "always"
        elif idempotent_override is False:
            effective_idempotency = "safe"
        else:
            effective_idempotency = self._config.failover_idempotency
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
        if not candidates:
            raise ProviderUnavailableError("all providers unavailable", attempted=[])
        if deadline.remaining() <= 0.0:
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
                raise

            # Same-provider retry budget for THIS chain entry (mirrors the sync
            # walk); consumed inside the inner attempt loop below.
            same_retries_left = self._config.same_provider_retries
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
                        # Shrinking per-hop slice: divide what's left of the
                        # chain deadline across the candidates not yet attempted so a
                        # single hung hop cannot consume the whole budget.
                        timeout=_hop_timeout(deadline, len(candidates) - idx),
                        max_retries=0,
                    )
                    if is_streaming and rt.adapter.dialect == "google":
                        # First-chunk materialization — GOOGLE DIALECT ONLY (fix
                        # [B]). Awaiting this runs the eager anext, so a Google
                        # lazy-generator establishment error falls into THIS except ->
                        # classify_exception -> failover. OpenAI/Anthropic established
                        # eagerly at dispatch, so we DON'T materialize them (a
                        # post-connect first-chunk error must NOT be misread as pre-send
                        # -> double-spend risk). No double-emit: the buffered first
                        # chunk is replayed via the wrapper.
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
                            provider_region=rt.adapter.extract_region(rt.sdk_client),
                        )
                    )
                    if disp is Disposition.FAIL_FAST:
                        raise
                    if disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover:
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
                )
            cb.record_success()

            # LatencyPolicy signal: record this served hop's success latency
            # (non-streaming). Streaming records in on_complete when the stream
            # settles. Pure signal store — no I/O, no routing change here.
            self.record_latency(provider, ctx.elapsed_ms())

            token_details = rt.adapter.extract_usage(response)
            # Explicit-degradation fallback: a compat provider that omitted the
            # usage block yields a length-based estimate (is_estimated=True)
            # instead of silently recording zero spend. None when usage was
            # present or the adapter always reports usage.
            estimated_details = rt.adapter.estimate_missing_usage(
                response, estimated_input_tokens=est_in
            )
            if estimated_details is not None:
                token_details = estimated_details
            # Per-region pricing attribution: the SERVED runtime's endpoint
            # region (None for providers without regional pricing).
            provider_region = rt.adapter.extract_region(rt.sdk_client)
            # Extracted ONCE from the RAW served response — confirm and
            # metadata must carry the same tier (see the sync path).
            service_tier = rt.adapter.extract_service_tier(response)
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
            if budget.reservation_id:
                await self._budget.confirm_cost(
                    budget.reservation_id,
                    served_model,
                    token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
                )
            self._reporter.report(
                self._build_metadata_event(
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
            )
            return result

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
        # Per-region pricing attribution for the SERVED runtime (None for
        # providers without regional pricing). Captured once, closed over.
        provider_region = runtime.adapter.extract_region(runtime.sdk_client)

        async def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            self._get_circuit_breaker(provider).record_success()
            # LatencyPolicy signal: record the SERVED provider's latency as the
            # stream settles (mirrors the non-streaming path). Pure signal store.
            self.record_latency(provider, ctx.elapsed_ms())
            # Extracted ONCE — confirm and metadata must carry the same tier
            # (see the sync on_complete).
            service_tier = accumulator.get_service_tier()
            confirm = None
            if budget.reservation_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=budget.reservation_id,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
                    provider_region=provider_region,
                    service_tier=service_tier,
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
        """Pass through non-intercepted attributes to the underlying client."""
        attribute = getattr(self._client, name)
        _warn_unmetered_spend_surface_once(
            adapter=self._adapter, dialect=self._dialect, surface=name
        )
        return attribute
