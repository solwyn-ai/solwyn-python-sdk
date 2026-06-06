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

from solwyn._base import _AttemptContext, _SolwynBase
from solwyn._privacy import estimate_content_length, estimate_tokens_from_length
from solwyn._proxies import (
    _AsyncChatProxy,
    _AsyncMessagesProxy,
    _AsyncModelsProxy,
    _SyncChatProxy,
    _SyncMessagesProxy,
    _SyncModelsProxy,
)
from solwyn._registry import ProviderRuntime, build_runtimes
from solwyn._routing import RoutingRequest, SelectionPolicy
from solwyn._run import current_run
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
    UntranslatableModelError,
)
from solwyn.providers import _translation, get_adapter_for_client
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
}


def _source_compatible_defaults(provider: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return target defaults that are also legal in the source dialect."""
    allowed = _SOURCE_COMPATIBLE_DEFAULT_KEYS.get(provider, frozenset())
    return {key: value for key, value in params.items() if key in allowed}


def _mapping_from_config(value: object) -> dict[str, Any]:
    """Return a shallow dict view of a provider config object."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dict(dumped)
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return dict(attrs)
    return {}


def _with_google_http_bound(
    kwargs: dict[str, object], *, timeout: float, max_retries: int
) -> dict[str, object]:
    """Inject google-genai per-request HTTP options without importing the extra.

    google-genai accepts ``config={"http_options": ...}`` for generate_content
    calls, and its timeout is milliseconds. We preserve caller config/http_options
    but override the timeout and retry attempts because the chain deadline is a
    mandatory Solwyn bound.
    """
    bounded = dict(kwargs)
    config = _mapping_from_config(bounded.get("config"))
    http_options = _mapping_from_config(config.get("http_options"))
    retry_options = _mapping_from_config(http_options.get("retry_options"))

    timeout_ms = max(1, int(timeout * 1000))
    retry_options["attempts"] = max(1, max_retries + 1)
    http_options["timeout"] = timeout_ms
    http_options["retry_options"] = retry_options
    config["http_options"] = http_options
    bounded["config"] = config
    return bounded


def _openai_uses_max_completion_tokens(model: str) -> bool:
    """Return whether an OpenAI model rejects the legacy max_tokens key."""
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


def _with_openai_completion_token_key(
    provider: str,
    model: str,
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Rewrite OpenAI max_tokens for models that require max_completion_tokens."""
    if provider != ProviderName.OPENAI.value or not _openai_uses_max_completion_tokens(model):
        return kwargs
    if "max_tokens" not in kwargs:
        return kwargs
    rewritten = dict(kwargs)
    if "max_completion_tokens" not in rewritten:
        rewritten["max_completion_tokens"] = rewritten["max_tokens"]
    del rewritten["max_tokens"]
    return rewritten


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
    merged_defaults = {**global_defaults, **rt.entry.default_params}
    merged_kwargs: dict[str, object] = {**merged_defaults, **kwargs}
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

    # Translate via the canonical subset (may RAISE an Untranslatable* error
    # BEFORE any network call; the caller aborts the chain). Cross-provider
    # translation starts from SOURCE-dialect values only: the target entry's
    # default_params may contain target-native keys such as Anthropic top_k.
    source_defaults = _source_compatible_defaults(primary.adapter.name, rt.entry.default_params)
    source_kwargs: dict[str, object] = {**global_defaults, **source_defaults, **kwargs}
    canonical = _translation.to_canonical(primary.adapter.name, source_kwargs)

    # CROSS-PROVIDER STREAMING. A PLAIN-TEXT cross-provider
    # streamed response is normalized per-chunk by the wrapper, so it proceeds. A
    # TOOL-using streamed response cannot be normalized cross-provider (tool-call
    # deltas are out of the v1 streaming subset) — so FAIL LOUD here, BEFORE
    # dispatch, aborting the chain cleanly (no foreign stream returned). Checking
    # the canonical keeps this structural and content-free.
    if is_streaming and canonical.tools is not None:
        _translation.fail_cross_provider_tool_stream(
            source=primary.adapter.name, target=rt.adapter.name
        )

    call_kwargs = _translation.from_canonical(rt.adapter.name, canonical, model=rt.entry.model)
    # Re-apply target entry defaults as fill-absent (e.g. Anthropic max_tokens).
    return {**rt.entry.default_params, **call_kwargs}


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
        fallback: object = None,
        default_params: dict[str, Any] | None = None,
        selection_policy: SelectionPolicy | None = None,
        **config_kwargs: object,
    ) -> None:
        # Detect provider and store adapter for usage extraction
        self._adapter = get_adapter_for_client(client)
        self._detected_provider = ProviderName(self._adapter.name)
        # self._client is typed Any because each provider SDK has a different
        # public surface (chat/messages/models). A unified Protocol would not
        # match all three. Type safety stops at the _sync_dispatch boundary.
        self._client: Any = client

        if "project_id" in config_kwargs:
            raise TypeError("unexpected keyword argument 'project_id'")

        # Build the [primary, *fallbacks] runtime chain. All chain clients are
        # constructed up front so the first failover is pure dispatch.
        fallback_specs = _normalize_fallback(fallback)
        runtimes = build_runtimes(client, model, fallback_specs)

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
        )

    @functools.cached_property
    def chat(self) -> _SyncChatProxy:
        """Return a proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _SyncChatProxy(self)

    @functools.cached_property
    def messages(self) -> Any:
        """Anthropic-compatible: client.messages.create() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.ANTHROPIC:
            return _SyncMessagesProxy(self)
        return self._client.messages

    @functools.cached_property
    def models(self) -> Any:
        """Google-compatible: client.models.generate_content() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.GOOGLE:
            return _SyncModelsProxy(self)
        return self._client.models

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
        600s SDK read default and any internal retry stacking).

        The served hop's stream-method selection is driven by the dispatch-level
        ``is_streaming`` boolean — NOT the original ``_force_stream``/canonical
        flag (fix [A]). A caller who streamed via OpenAI/Anthropic ``stream=True``
        and fails over to Google must still hit ``generate_content_stream``; a
        Google streamer that fails over to OpenAI/Anthropic must serve with
        ``stream=True``. OpenAI/Anthropic stream via ``stream=True`` in kwargs;
        Google streams via its dedicated method (and the ``stream`` kwarg, which
        its SDK does not accept, is stripped).
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        name = runtime.adapter.name
        if name == ProviderName.OPENAI.value:
            if is_streaming:
                kwargs["stream"] = True
            return client.chat.completions.create(**kwargs)
        if name == ProviderName.ANTHROPIC.value:
            if is_streaming:
                kwargs["stream"] = True
            return client.messages.create(**kwargs)
        # Google: its generate_content[_stream] methods take no ``stream`` kwarg.
        kwargs = _with_google_http_bound(kwargs, timeout=timeout, max_retries=max_retries)
        kwargs.pop("stream", None)
        if is_streaming:
            return client.models.generate_content_stream(**kwargs)
        return client.models.generate_content(**kwargs)

    def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: object) -> Any:
        """Core interception logic: the classified candidate walk."""
        requested_model = cast(str, kwargs["model"])
        is_streaming = bool(kwargs.get("stream", False)) or _force_stream
        agent_run = current_run()
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
        )
        # Refresh the CostPolicy signal from the server. Price hints are advisory
        # and slow-moving, so they PERSIST across hint-less responses — a budget
        # cache hit (price_hints None) leaves the last-known hints in place; we
        # only overwrite when the server actually returns hints. The SDK never
        # computes price — it only forwards this relative signal.
        if budget.price_hints is not None:
            self.update_price_hints(budget.price_hints)

        if not budget.allowed:
            # Report estimated tokens so the API keeps an accurate running total
            # even for calls that were blocked by hard-deny.
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
                )
                self._reporter.report(event)
            except Exception:
                logger.warning("Failed to report budget_denied metadata event", exc_info=True)

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
                    call_kwargs = rt.adapter.prepare_streaming(call_kwargs)
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
                    if is_streaming and provider == ProviderName.GOOGLE.value:
                        # First-chunk materialization — GOOGLE ONLY (fix
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
                    if (
                        disp is Disposition.FAILOVER
                        and same_retries_left > 0
                        and provider not in failed_providers
                    ):
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
                )
            cb.record_success()

            # LatencyPolicy signal: record this served hop's success latency
            # (non-streaming). Streaming records in on_complete when the stream
            # settles. Pure signal store — no I/O, no routing change here.
            self.record_latency(provider, ctx.elapsed_ms())

            token_details = rt.adapter.extract_usage(response)
            result = response
            if is_provider_fallback:
                # Cross-provider hop: reshape the served response back to the
                # caller's native dialect BEFORE confirm/report success. If the
                # served shape is unexpected, do not mark Solwyn billing settled.
                result = _translation.normalize_response(
                    served=rt.adapter.name,
                    requested=primary.adapter.name,
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
                    service_tier=rt.adapter.extract_service_tier(response),
                    agent_run=agent_run,
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
        agent_run: tuple[str | None, str | None],
    ) -> SyncStreamWrapper:
        """Wrap a streaming response, settling against the SERVED runtime."""
        provider = runtime.adapter.name
        served_model = ctx.model
        is_provider_fallback = ctx.is_provider_fallback
        accumulator = runtime.adapter.create_stream_accumulator()

        def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            self._get_circuit_breaker(provider).record_success()
            # LatencyPolicy signal: record the SERVED provider's latency as the
            # stream settles (mirrors the non-streaming path). Pure signal store.
            self.record_latency(provider, ctx.elapsed_ms())
            confirm = None
            if budget.reservation_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=budget.reservation_id,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
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
                service_tier=accumulator.get_service_tier(),
                agent_run=agent_run,
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
                )
            )

        # Cross-provider hop: reshape each served chunk back to the caller's
        # native streaming dialect. Same-dialect (primary / same-provider
        # model swap) passes None -> zero translation. The accumulator still
        # observes the RAW served chunk inside the wrapper, so usage settles
        # against the served provider.
        chunk_translator = (
            _make_chunk_translator(served=provider, requested=primary.adapter.name)
            if is_provider_fallback
            else None
        )
        return SyncStreamWrapper(
            response, accumulator, on_complete, on_error, chunk_translator=chunk_translator
        )

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
        return getattr(self._client, name)


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
        fallback: object = None,
        default_params: dict[str, Any] | None = None,
        selection_policy: SelectionPolicy | None = None,
        **config_kwargs: object,
    ) -> None:
        # Detect provider and store adapter for usage extraction
        self._adapter = get_adapter_for_client(client)
        self._detected_provider = ProviderName(self._adapter.name)
        # See sync Solwyn.__init__ for why _client is typed Any.
        self._client: Any = client

        if "project_id" in config_kwargs:
            raise TypeError("unexpected keyword argument 'project_id'")

        # Build the [primary, *fallbacks] runtime chain (constructed up front).
        fallback_specs = _normalize_fallback(fallback)
        runtimes = build_runtimes(client, model, fallback_specs)

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
        )

    @functools.cached_property
    def chat(self) -> _AsyncChatProxy:
        """Return an async proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _AsyncChatProxy(self)

    @functools.cached_property
    def messages(self) -> Any:
        """Anthropic-compatible: client.messages.create() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.ANTHROPIC:
            return _AsyncMessagesProxy(self)
        return self._client.messages

    @functools.cached_property
    def models(self) -> Any:
        """Google-compatible: client.models.generate_content() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.GOOGLE:
            return _AsyncModelsProxy(self)
        return self._client.models

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

        Applies the mandatory per-hop bound via ``with_options``. Mirrors
        ``_sync_dispatch``: the served hop's stream-method selection is driven by
        the dispatch-level ``is_streaming`` boolean (fix [A]).
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        name = runtime.adapter.name
        if name == ProviderName.OPENAI.value:
            if is_streaming:
                kwargs["stream"] = True
            return await client.chat.completions.create(**kwargs)
        if name == ProviderName.ANTHROPIC.value:
            if is_streaming:
                kwargs["stream"] = True
            return await client.messages.create(**kwargs)
        # Google: its generate_content[_stream] methods take no ``stream`` kwarg.
        kwargs = _with_google_http_bound(kwargs, timeout=timeout, max_retries=max_retries)
        kwargs.pop("stream", None)
        if is_streaming:
            return await client.models.generate_content_stream(**kwargs)
        return await client.models.generate_content(**kwargs)

    async def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: object) -> Any:
        """Async core interception logic: the classified candidate walk."""
        requested_model = cast(str, kwargs["model"])
        is_streaming = bool(kwargs.get("stream", False)) or _force_stream
        agent_run = current_run()
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
        )
        # Refresh the CostPolicy signal from the server. Hints PERSIST across
        # hint-less responses (cache hits) until the server sends new ones — see
        # the sync _intercepted_call for the rationale.
        if budget.price_hints is not None:
            self.update_price_hints(budget.price_hints)

        if not budget.allowed:
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
                )
                self._reporter.report(event)
            except Exception:
                logger.warning("Failed to report budget_denied metadata event", exc_info=True)

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
                    call_kwargs = rt.adapter.prepare_streaming(call_kwargs)
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
                    if is_streaming and provider == ProviderName.GOOGLE.value:
                        # First-chunk materialization — GOOGLE ONLY (fix
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
                    if (
                        disp is Disposition.FAILOVER
                        and same_retries_left > 0
                        and provider not in failed_providers
                    ):
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
                )
            cb.record_success()

            # LatencyPolicy signal: record this served hop's success latency
            # (non-streaming). Streaming records in on_complete when the stream
            # settles. Pure signal store — no I/O, no routing change here.
            self.record_latency(provider, ctx.elapsed_ms())

            token_details = rt.adapter.extract_usage(response)
            result = response
            if is_provider_fallback:
                # Cross-provider hop: reshape the served response back to the
                # caller's native dialect BEFORE confirm/report success. If the
                # served shape is unexpected, do not mark Solwyn billing settled.
                result = _translation.normalize_response(
                    served=rt.adapter.name,
                    requested=primary.adapter.name,
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
                    service_tier=rt.adapter.extract_service_tier(response),
                    agent_run=agent_run,
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
        agent_run: tuple[str | None, str | None],
    ) -> AsyncStreamWrapper:
        """Wrap an async streaming response, settling against the SERVED runtime."""
        provider = runtime.adapter.name
        served_model = ctx.model
        is_provider_fallback = ctx.is_provider_fallback
        accumulator = runtime.adapter.create_stream_accumulator()

        async def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            self._get_circuit_breaker(provider).record_success()
            # LatencyPolicy signal: record the SERVED provider's latency as the
            # stream settles (mirrors the non-streaming path). Pure signal store.
            self.record_latency(provider, ctx.elapsed_ms())
            confirm = None
            if budget.reservation_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=budget.reservation_id,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                    call_id=call_id,
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
                service_tier=accumulator.get_service_tier(),
                agent_run=agent_run,
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
                )
            )

        # Cross-provider hop: reshape each served chunk back to the caller's
        # native streaming dialect. Same-dialect passes None (zero
        # translation). See the sync ``_wrap_stream`` for the accounting note.
        chunk_translator = (
            _make_chunk_translator(served=provider, requested=primary.adapter.name)
            if is_provider_fallback
            else None
        )
        return AsyncStreamWrapper(
            response, accumulator, on_complete, on_error, chunk_translator=chunk_translator
        )

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
        return getattr(self._client, name)
