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

import functools
import logging
import time
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
from solwyn._routing import RoutingRequest
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
)
from solwyn.providers import _translation, get_adapter_for_client
from solwyn.providers._errors import Disposition, classify_exception
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter
from solwyn.stream import AsyncStreamWrapper, SyncStreamWrapper

logger = logging.getLogger(__name__)


# Floor for a per-hop dispatch timeout: even when the chain deadline is nearly
# spent we give a hop at least this long rather than passing it ~0s (§6.3).
_MIN_HOP_TIMEOUT = 1.0


class Deadline:
    """A monotonic chain deadline (§6.3).

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
    *, is_provider_fallback: bool, is_model_fallback: bool
) -> FailoverReason | None:
    """Pick the success-path failover reason for a served candidate (§4.6)."""
    if is_provider_fallback:
        return FailoverReason.CIRCUIT_OPEN
    if is_model_fallback:
        return FailoverReason.MODEL_FALLBACK
    return None


def _detect_provider(client: object) -> ProviderName:
    """Auto-detect the LLM provider from the client instance.

    Delegates to the provider adapter registry for consistent detection.
    """
    try:
        adapter = get_adapter_for_client(client)
        return ProviderName(adapter.name)
    except ValueError as err:
        raise ValueError(
            f"Cannot auto-detect provider for client type {type(client).__name__}. "
            f"Supported: openai.OpenAI, anthropic.Anthropic, "
            f"google.generativeai.GenerativeModel"
        ) from err


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
    """Build the native kwargs for one candidate hop (§4.6, §5).

    Fill-absent precedence: per-call kwargs > per-entry default_params > global.
    For a PRIMARY or SAME-PROVIDER model-swap hop the merged kwargs pass straight
    through to the native SDK (no translation). For a CROSS-PROVIDER hop the
    merged kwargs are run through the §5 translation contract:

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
            return merged_kwargs
        return {**merged_kwargs, "model": rt.entry.model}

    # CROSS-PROVIDER hop. Streaming failover requires stream normalization, which
    # is P3 — not P2. Without it a cross-provider streaming hop would serve a
    # foreign-dialect stream the caller cannot consume on its native access path,
    # so FAIL LOUD here, BEFORE dispatch, aborting the chain cleanly (no foreign
    # stream returned). P3 lifts this restriction (adds stream normalization).
    if is_streaming:
        _translation.fail_cross_provider_streaming(
            source=primary.adapter.name, target=rt.adapter.name
        )

    # Translate via the canonical subset (may RAISE an Untranslatable* error
    # BEFORE any network call; the caller aborts the chain).
    canonical = _translation.to_canonical(primary.adapter.name, merged_kwargs)
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
        super().__init__(config, runtimes)

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
        _force_stream: bool,
        timeout: float,
        max_retries: int,
    ) -> Any:
        """Dispatch one hop to the runtime's SDK client. Pure I/O — no metrics.

        Applies the mandatory per-hop bound via ``with_options`` (kills the
        600s SDK read default and any internal retry stacking; §6.3).
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        name = runtime.adapter.name
        if name == ProviderName.OPENAI.value:
            return client.chat.completions.create(**kwargs)
        if name == ProviderName.ANTHROPIC.value:
            return client.messages.create(**kwargs)
        if _force_stream:
            if name != ProviderName.GOOGLE.value:
                raise RuntimeError(f"_force_stream is Google-only but provider is {name}")
            return client.models.generate_content_stream(**kwargs)
        return client.models.generate_content(**kwargs)

    def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: object) -> Any:
        """Core interception logic: the classified candidate walk (§4.6)."""
        requested_model = cast(str, kwargs["model"])
        is_streaming = bool(kwargs.get("stream", False)) or _force_stream
        agent_run = current_run()
        primary = self._runtimes[0]
        # Deadline starts here — it encompasses the budget pre-flight (§6.3).
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
        )

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

        # 5. Walk the candidates.
        failed_providers: set[str] = set()
        last_exc: Exception | None = None
        for idx, rt in enumerate(candidates):
            if deadline.expired():
                break
            provider = rt.adapter.name
            cb = self._get_circuit_breaker(provider)
            if not cb.can_proceed():  # probe CONSUMED only here, for the attempted candidate
                continue

            is_primary = rt is primary
            is_provider_fallback = rt.entry.provider != primary.entry.provider
            is_model_fallback = (not is_provider_fallback) and not is_primary

            # Build native kwargs for this hop (§5). A cross-provider hop runs the
            # §5 translation contract and may RAISE an Untranslatable* error here,
            # BEFORE any network call — that aborts the WHOLE chain (§4.6, §6.8):
            # do NOT classify it as transport, advance, or record a breaker failure.
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

            ctx = _AttemptContext(
                model=served_model,
                start_time=time.monotonic(),
                is_provider_fallback=is_provider_fallback,
                attempt_index=idx,
            )
            try:
                response = self._sync_dispatch(
                    rt,
                    call_kwargs,
                    _force_stream=_force_stream,
                    # Shrinking per-hop slice (§6.3): divide what's left of the
                    # chain deadline across the candidates not yet attempted so a
                    # single hung hop cannot consume the whole budget.
                    timeout=max(_MIN_HOP_TIMEOUT, deadline.remaining() / (len(candidates) - idx)),
                    max_retries=0,
                )
            except Exception as exc:
                disp = classify_exception(exc)
                # Breaker accounting (§6.1): FAILOVER and POST_SEND_AMBIGUOUS are
                # provider-health signals and DO count; FAIL_FAST (4xx/refusal) is
                # a request-shaped error, not a health signal, so it must NOT open
                # the breaker. Same-provider double-count guard (§4.6): at most one
                # failure per provider per logical attempt.
                if disp is not Disposition.FAIL_FAST and provider not in failed_providers:
                    cb.record_failure()
                    failed_providers.add(provider)
                self._reporter.report(
                    self._build_error_event(
                        model=served_model,
                        provider=provider,
                        latency_ms=ctx.elapsed_ms(),
                        is_model_fallback=is_model_fallback,
                        is_provider_fallback=is_provider_fallback,
                        requested_provider=primary.entry.provider if is_provider_fallback else None,
                        requested_model=requested_model if is_provider_fallback else None,
                        failover_error_class=type(exc).__name__,
                        attempt_index=idx,
                        agent_run=agent_run,
                    )
                )
                if disp is Disposition.FAIL_FAST:
                    raise  # 4xx/404/refusal — do NOT advance the chain
                if disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover:
                    raise  # re-raise ORIGINAL exception (drop-in contract, §6.5)
                last_exc = exc
                continue  # pre-send safe -> advance to the next candidate

            # 6. SUCCESS — settle against the SERVED runtime.
            cb.record_success()
            if is_streaming:
                return self._wrap_stream(
                    rt,
                    response,
                    ctx,
                    budget,
                    primary,
                    requested_model=requested_model,
                    is_model_fallback=is_model_fallback,
                    agent_run=agent_run,
                )

            token_details = self._adapter_extract_usage(rt, response)
            if budget.reservation_id:
                self._budget.confirm_cost(
                    budget.reservation_id,
                    served_model,
                    token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
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
                    ),
                    attempt_index=idx,
                    service_tier=rt.adapter.extract_service_tier(response),
                    agent_run=agent_run,
                )
            )
            if is_provider_fallback:
                # Cross-provider hop: reshape the served response back to the
                # caller's native dialect (§5). Primary / same-provider hops
                # return the raw response unchanged.
                return _translation.normalize_response(
                    served=rt.adapter.name,
                    requested=primary.adapter.name,
                    response=response,
                )
            return response

        if last_exc is not None:
            raise last_exc
        raise ProviderUnavailableError(
            "all providers unavailable",
            attempted=[r.adapter.name for r in candidates],
        )

    @staticmethod
    def _adapter_extract_usage(runtime: ProviderRuntime, response: Any) -> TokenDetails:
        """Extract token usage via the served runtime's adapter."""
        return runtime.adapter.extract_usage(response)

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
        agent_run: tuple[str | None, str | None],
    ) -> SyncStreamWrapper:
        """Wrap a streaming response, settling against the SERVED runtime."""
        provider = runtime.adapter.name
        served_model = ctx.model
        is_provider_fallback = ctx.is_provider_fallback
        accumulator = runtime.adapter.create_stream_accumulator()

        def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            self._get_circuit_breaker(provider).record_success()
            if budget.reservation_id:
                confirm = self._budget.build_confirm_request(
                    reservation_id=budget.reservation_id,
                    model=served_model,
                    token_details=token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
                )
                self._reporter.report_confirm(confirm)
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
                    ),
                    attempt_index=ctx.attempt_index,
                    service_tier=accumulator.get_service_tier(),
                    agent_run=agent_run,
                )
            )

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
                    agent_run=agent_run,
                )
            )

        return SyncStreamWrapper(response, accumulator, on_complete, on_error)

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
        super().__init__(config, runtimes)

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
        _force_stream: bool,
        timeout: float,
        max_retries: int,
    ) -> Any:
        """Dispatch one hop to the runtime's async SDK client. Pure I/O.

        Applies the mandatory per-hop bound via ``with_options`` (§6.3).
        """
        client = runtime.sdk_client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        name = runtime.adapter.name
        if name == ProviderName.OPENAI.value:
            return await client.chat.completions.create(**kwargs)
        if name == ProviderName.ANTHROPIC.value:
            return await client.messages.create(**kwargs)
        if _force_stream:
            if name != ProviderName.GOOGLE.value:
                raise RuntimeError(f"_force_stream is Google-only but provider is {name}")
            return await client.models.generate_content_stream(**kwargs)
        return await client.models.generate_content(**kwargs)

    async def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: object) -> Any:
        """Async core interception logic: the classified candidate walk (§4.6)."""
        requested_model = cast(str, kwargs["model"])
        is_streaming = bool(kwargs.get("stream", False)) or _force_stream
        agent_run = current_run()
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
        )

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

        failed_providers: set[str] = set()
        last_exc: Exception | None = None
        for idx, rt in enumerate(candidates):
            if deadline.expired():
                break
            provider = rt.adapter.name
            cb = self._get_circuit_breaker(provider)
            if not cb.can_proceed():
                continue

            is_primary = rt is primary
            is_provider_fallback = rt.entry.provider != primary.entry.provider
            is_model_fallback = (not is_provider_fallback) and not is_primary

            # Build native kwargs for this hop (§5). A cross-provider hop runs the
            # §5 translation contract and may RAISE an Untranslatable* error here,
            # BEFORE any network call — that aborts the WHOLE chain (§4.6, §6.8):
            # do NOT classify it as transport, advance, or record a breaker failure.
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

            ctx = _AttemptContext(
                model=served_model,
                start_time=time.monotonic(),
                is_provider_fallback=is_provider_fallback,
                attempt_index=idx,
            )
            try:
                response = await self._async_dispatch(
                    rt,
                    call_kwargs,
                    _force_stream=_force_stream,
                    # Shrinking per-hop slice (§6.3): divide what's left of the
                    # chain deadline across the candidates not yet attempted so a
                    # single hung hop cannot consume the whole budget.
                    timeout=max(_MIN_HOP_TIMEOUT, deadline.remaining() / (len(candidates) - idx)),
                    max_retries=0,
                )
            except Exception as exc:
                disp = classify_exception(exc)
                # Breaker accounting (§6.1): count FAILOVER + POST_SEND_AMBIGUOUS
                # (real health signals); skip FAIL_FAST (request-shaped, not a
                # health signal). Same-provider double-count guard (§4.6).
                if disp is not Disposition.FAIL_FAST and provider not in failed_providers:
                    cb.record_failure()
                    failed_providers.add(provider)
                self._reporter.report(
                    self._build_error_event(
                        model=served_model,
                        provider=provider,
                        latency_ms=ctx.elapsed_ms(),
                        is_model_fallback=is_model_fallback,
                        is_provider_fallback=is_provider_fallback,
                        requested_provider=primary.entry.provider if is_provider_fallback else None,
                        requested_model=requested_model if is_provider_fallback else None,
                        failover_error_class=type(exc).__name__,
                        attempt_index=idx,
                        agent_run=agent_run,
                    )
                )
                if disp is Disposition.FAIL_FAST:
                    raise
                if disp is Disposition.POST_SEND_AMBIGUOUS and not allow_ambiguous_failover:
                    raise
                last_exc = exc
                continue

            cb.record_success()
            if is_streaming:
                return self._wrap_stream_async(
                    rt,
                    response,
                    ctx,
                    budget,
                    primary,
                    requested_model=requested_model,
                    is_model_fallback=is_model_fallback,
                    agent_run=agent_run,
                )

            token_details = rt.adapter.extract_usage(response)
            if budget.reservation_id:
                await self._budget.confirm_cost(
                    budget.reservation_id,
                    served_model,
                    token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
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
                    ),
                    attempt_index=idx,
                    service_tier=rt.adapter.extract_service_tier(response),
                    agent_run=agent_run,
                )
            )
            if is_provider_fallback:
                # Cross-provider hop: reshape the served response back to the
                # caller's native dialect (§5). Primary / same-provider hops
                # return the raw response unchanged.
                return _translation.normalize_response(
                    served=rt.adapter.name,
                    requested=primary.adapter.name,
                    response=response,
                )
            return response

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
        agent_run: tuple[str | None, str | None],
    ) -> AsyncStreamWrapper:
        """Wrap an async streaming response, settling against the SERVED runtime."""
        provider = runtime.adapter.name
        served_model = ctx.model
        is_provider_fallback = ctx.is_provider_fallback
        accumulator = runtime.adapter.create_stream_accumulator()

        async def on_complete(token_details: TokenDetails, _elapsed_ms: float) -> None:
            self._get_circuit_breaker(provider).record_success()
            if budget.reservation_id:
                await self._budget.confirm_cost(
                    budget.reservation_id,
                    served_model,
                    token_details,
                    provider=provider,
                    is_provider_fallback=is_provider_fallback,
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
                    ),
                    attempt_index=ctx.attempt_index,
                    service_tier=accumulator.get_service_tier(),
                    agent_run=agent_run,
                )
            )

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
                    agent_run=agent_run,
                )
            )

        return AsyncStreamWrapper(response, accumulator, on_complete, on_error)

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
