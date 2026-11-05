"""Regressions for the refusal holes found in the TypeScript SDK's review (WS-14).

One test (or one small family) per class of bug, each checked against the
Python implementation:

1. A 401 whose body stalls must refuse before the read timeout surfaces.
2. Every physical provider dispatch (failover hops, same-provider retries,
   streaming, media) checks the binding, whatever funds it.
3. Unmetered spend surfaces honour stored refusals, including references taken
   before the refusal and acknowledged tokens.
4. Retained authority (cached allows, held leases) is tied to the binding and
   the pinned project; late grants and renewals are fenced.
5. A lease-funded rerun after a discarded result takes per-call admission.
6. A directly built binding validates its project selector.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import pytest
from conftest import (
    VALID_API_KEY,
    VALID_PROJECT_ID,
    auth_key_body,
    patch_wrapper_local,
    validated_binding,
)

from solwyn._api_key_errors import Refusal
from solwyn._constants import AUTH_KEY_PATH
from solwyn._project import ProjectBinding
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.exceptions import ApiKeyRefusedError, ConfigurationError
from solwyn.testing import FakeControlPlane

pytestmark = pytest.mark.project_validation

openai = pytest.importorskip("openai")

_URL = "https://api.test.solwyn.ai"
_CHECK_PATH = "/api/v1/budgets/check"
_LEASE_PATH = "/api/v1/budgets/lease"
_MESSAGES = [{"role": "user", "content": "hi"}]
_GROQ_URL = "https://api.groq.com/openai/v1"
_CHAT_BODY = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-5.5",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
}


def _check_body() -> dict[str, Any]:
    return {
        "allowed": True,
        "remaining_budget": 50.0,
        "reservation_id": "res-1",
        "mode": "alert_only",
        "budget_limit": 100.0,
        "current_usage": 50.0,
        "project_id": VALID_PROJECT_ID,
    }


class _StallingBody(httpx.SyncByteStream):
    """A response body that never arrives: reading it times out."""

    def __init__(self, request: httpx.Request) -> None:
        self._request = request

    def __iter__(self) -> Iterator[bytes]:
        raise httpx.ReadTimeout("scripted stalled body", request=self._request)


class _AsyncStallingBody(httpx.AsyncByteStream):
    def __init__(self, request: httpx.Request) -> None:
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.ReadTimeout("scripted stalled body", request=self._request)
        yield b""  # pragma: no cover - makes this an async generator


# ── 1. stalled 401 bodies ─────────────────────────────────────────────────


@pytest.mark.unit
def test_1_stalled_401_body_on_a_check_blocks_instead_of_failing_open() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(401, stream=_StallingBody(request))

    binding = validated_binding()
    enforcer = BudgetEnforcer(
        _URL,
        binding,
        fail_open=True,
        cache_ttl=0,
        lease_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        for _ in range(2):
            with pytest.raises(ApiKeyRefusedError):
                enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()

    refusal = binding.refusal
    assert refusal is not None
    assert (refusal.code, refusal.posture, refusal.effect) == (
        "api_key_invalid",
        "unknown",
        "block",
    )
    # The check, then the immediate revalidation its 401 scheduled (also a
    # stalled 401, so still blocked); the second call never left the gate.
    assert seen == [_CHECK_PATH, AUTH_KEY_PATH]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_1_async_stalled_401_body_on_a_check_blocks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=_AsyncStallingBody(request))

    binding = validated_binding()
    enforcer = AsyncBudgetEnforcer(
        _URL,
        binding,
        fail_open=True,
        cache_ttl=0,
        lease_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ApiKeyRefusedError):
            await enforcer.check_budget(
                estimated_input_tokens=1, model="gpt-5.5", provider="openai"
            )
    finally:
        await enforcer.close()

    assert binding.state == "refused"


@pytest.mark.unit
def test_1_stalled_401_body_on_startup_validation_blocks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_KEY_PATH:
            return httpx.Response(401, stream=_StallingBody(request))
        return httpx.Response(200, json=_check_body())

    binding = ProjectBinding(VALID_API_KEY, "fake-project")
    enforcer = BudgetEnforcer(
        _URL, binding, fail_open=True, lease_enabled=False, transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ApiKeyRefusedError):
            enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()

    assert binding.state == "refused"


@pytest.mark.unit
def test_1_stalled_401_body_tightens_an_allow_unmetered_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=_StallingBody(request))

    binding = validated_binding()
    clock = [10_000.0]
    binding._clock = lambda: clock[0]
    binding.refuse(Refusal(code="api_key_expired", posture="alert_only"), binding.generation)
    assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"
    clock[0] += 61  # the gate's next revalidation is due
    enforcer = BudgetEnforcer(
        _URL, binding, fail_open=True, lease_enabled=False, transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ApiKeyRefusedError):
            enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()

    assert binding.refusal is not None and binding.refusal.effect == "block"


@pytest.mark.unit
def test_1_alert_only_client_blocks_on_a_401_status_then_relaxes_promptly() -> None:
    plane = FakeControlPlane(mode="alert_only")
    stub = _StubOpenAI()
    client = plane.wrap(stub, lease_enabled=False)
    effects: list[str] = []
    try:
        client.validate()
        client._solwyn_binding.add_listener(
            lambda: effects.append(
                client._solwyn_binding.refusal.effect if client._solwyn_binding.refusal else "none"
            )
        )
        plane.expire_key()
        auth_before = len(plane.auth_key_requests)
        client.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
    finally:
        client.close()

    # The 401 status blocked at once (the brief window); the immediate
    # revalidation, sent by the same call, relaxed it and the call ran.
    assert effects[:2] == ["block", "allow_unmetered"]
    assert len(plane.auth_key_requests) == auth_before + 1
    assert stub.chat.completions.calls == 1


# ── 2. every physical dispatch checks the binding ─────────────────────────


class _StubCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs: object) -> object:
        from types import SimpleNamespace

        self.calls += 1
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))


class _StubOpenAI:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.chat = SimpleNamespace(completions=_StubCompletions())

    def with_options(self, **_kwargs: object) -> _StubOpenAI:
        return self


_StubOpenAI.__module__ = "openai._client"
_StubOpenAI.__name__ = "OpenAI"


class _Provider:
    def __init__(self, status: int = 200, *, on_request: Callable[[], None] | None = None) -> None:
        self.status = status
        self.on_request = on_request
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.on_request is not None:
            self.on_request()
        if self.status != 200:
            return httpx.Response(
                self.status,
                json={"error": {"message": "x", "type": "x", "code": None}},
                headers={"retry-after": "0"},
            )
        return httpx.Response(200, json=_CHAT_BODY)

    def client(self, base_url: str | None = None) -> Any:
        return openai.OpenAI(
            api_key="sk-test",
            base_url=base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            max_retries=0,
        )


def _refuse_now(holder: list[Any]) -> Callable[[], None]:
    def refuse() -> None:
        binding: ProjectBinding = holder[0]._solwyn_binding
        binding.refuse(Refusal(code="api_key_invalid", posture="hard_deny"), binding.generation, from_validation=True)

    return refuse


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_2_alert_only_failover_hop_never_dispatches_after_a_mid_call_refusal(
    stream: bool,
) -> None:
    plane = FakeControlPlane(mode="alert_only")
    holder: list[Any] = []
    primary = _Provider(429, on_request=_refuse_now(holder))
    fallback = _Provider()
    wrapped = plane.wrap(primary.client(), fallback=[(fallback.client(_GROQ_URL), "llama-3.3-70b")])
    holder.append(wrapped)
    with wrapped, pytest.raises(ApiKeyRefusedError):
        wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES, stream=stream)

    assert primary.requests == 1
    assert fallback.requests == 0


@pytest.mark.unit
def test_2_same_provider_retry_never_dispatches_after_a_mid_call_refusal() -> None:
    plane = FakeControlPlane(mode="alert_only")
    holder: list[Any] = []
    primary = _Provider(429, on_request=_refuse_now(holder))
    wrapped = plane.wrap(primary.client(), same_provider_retries=2)
    holder.append(wrapped)
    with wrapped, pytest.raises(ApiKeyRefusedError):
        wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)

    assert primary.requests == 1


@pytest.mark.unit
def test_2_lease_funded_hop_releases_its_claim_when_blocked() -> None:
    import solwyn

    plane = FakeControlPlane(mode="hard_deny")
    holder: list[Any] = []
    primary = _Provider(429, on_request=_refuse_now(holder))
    fallback = _Provider()
    wrapped = plane.wrap(
        primary.client(),
        fallback=[(fallback.client(_GROQ_URL), "llama-3.3-70b")],
        lease_enabled=True,
    )
    holder.append(wrapped)
    with wrapped, solwyn.run("blocked-hop"), pytest.raises(ApiKeyRefusedError):
        wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
    ledger = wrapped._solwyn_budget._lease

    assert fallback.requests == 0
    assert ledger._call_index == {}


@pytest.mark.unit
def test_2_media_dispatch_checks_the_binding_after_admission() -> None:
    plane = FakeControlPlane(mode="alert_only")
    embeddings = _Provider()
    wrapped = plane.wrap(embeddings.client(), lease_enabled=False)
    original = wrapped._apply_failover_tuning_directive

    def admitted_then_refused(*args: Any, **kwargs: Any) -> Any:
        # Runs after the gate and the exit check admitted the call, before
        # any provider I/O: a refusal recorded now (by another call or a
        # background send) must still stop the dispatch.
        _refuse_now([wrapped])()
        return original(*args, **kwargs)

    with (
        wrapped,
        patch_wrapper_local(wrapped, "_apply_failover_tuning_directive", admitted_then_refused),
        pytest.raises(ApiKeyRefusedError),
    ):
        wrapped.embeddings.create(model="text-embedding-3-small", input="hi")

    assert embeddings.requests == 0


# ── 3. unmetered spend surfaces ───────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("acknowledged", [False, True], ids=["allow", "acknowledged"])
def test_3_unmetered_spend_honours_a_refusal_even_through_an_old_reference(
    acknowledged: bool,
) -> None:
    provider = _Provider()
    plane = FakeControlPlane()
    options: dict[str, Any] = {"on_unmetered": "allow"}
    if acknowledged:
        options = {"acknowledge_untracked": {"batches.create"}}
    wrapped = plane.wrap(provider.client(), **options)
    create = wrapped.batches.create  # taken before the refusal
    binding = wrapped._solwyn_binding
    binding.refuse(Refusal(code="api_key_invalid", posture="hard_deny"), binding.generation, from_validation=True)
    with wrapped:
        with pytest.raises(ApiKeyRefusedError):
            create(input_file_id="f", endpoint="/v1/chat/completions", completion_window="24h")
        with pytest.raises(ApiKeyRefusedError):
            wrapped.batches.create(
                input_file_id="f", endpoint="/v1/chat/completions", completion_window="24h"
            )

    assert provider.requests == 0


@pytest.mark.unit
def test_3_unmetered_spend_runs_under_allow_unmetered() -> None:
    provider = _Provider()
    plane = FakeControlPlane()
    wrapped = plane.wrap(provider.client(), on_unmetered="allow")
    binding = wrapped._solwyn_binding
    binding.apply_auth_key(200, auth_key_body(project_id=plane.project_id), binding.generation)
    binding.refuse(Refusal(code="api_key_expired", posture="alert_only"), binding.generation)
    # The stub answers a chat body; only the dispatch matters here.
    with wrapped, contextlib.suppress(Exception):
        wrapped.batches.create(
            input_file_id="f", endpoint="/v1/chat/completions", completion_window="24h"
        )

    assert provider.requests == 1


# ── 4. retained authority follows the binding and the pinned project ──────


@pytest.mark.unit
def test_4_pinning_a_different_project_drops_held_leases() -> None:
    import solwyn

    plane = FakeControlPlane(mode="hard_deny")
    client = plane.wrap(_StubOpenAI(), lease_enabled=True)
    clock = [10_000.0]
    client._solwyn_binding._clock = lambda: clock[0]
    ledger = client._solwyn_budget._lease
    with client, solwyn.run("pin-change"):
        run_id = solwyn.current_run()[0]
        with plane.outage(path=AUTH_KEY_PATH):
            client.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)  # lease
        old_lease = ledger.lease_id_for(run_id)
        assert old_lease is not None
        # The project is renamed and a new project takes its name: the
        # selector now resolves to a different id.
        plane.project_id = "proj_" + "1" * 24
        clock[0] += 31
        client.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        new_lease = ledger.lease_id_for(run_id)

    assert client._solwyn_binding.pinned_id == "proj_" + "1" * 24
    # The old project's lease was dropped and the call was granted afresh.
    assert new_lease is not None and new_lease != old_lease
    assert len(plane.lease_grants) == 2


@pytest.mark.unit
def test_4_pinning_a_different_project_drops_cached_allows() -> None:
    plane = FakeControlPlane(mode="alert_only")
    client = plane.wrap(_StubOpenAI(), lease_enabled=False, budget_check_cache_ttl=60)
    clock = [10_000.0]
    client._solwyn_binding._clock = lambda: clock[0]
    with client:
        with plane.outage(path=AUTH_KEY_PATH):
            client.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        assert client._solwyn_budget._allow_cache
        plane.project_id = "proj_" + "1" * 24
        clock[0] += 31
        checks_before = len(plane.checks)
        client.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)

    # The old project's cached allow was not replayed: a fresh check ran.
    assert len(plane.checks) == checks_before + 1


@pytest.mark.unit
def test_4_a_grant_answered_after_the_binding_changed_is_surrendered_not_installed() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    binding = validated_binding(plane.api_key, project_id=plane.project_id)

    class _Transport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == _LEASE_PATH:
                # A re-pin lands while the grant is on the wire.
                binding.apply_auth_key(
                    200, auth_key_body(project_id=plane.project_id), binding.generation
                )
            return plane.transport.handle_request(request)

    enforcer = BudgetEnforcer(plane.api_url, binding, cache_ttl=0, transport=_Transport())
    try:
        result = enforcer.check_budget(
            estimated_input_tokens=1, model="gpt-5.5", provider="openai", agent_run_id="r1"
        )
        installed = enforcer._lease.active_run_ids()
    finally:
        enforcer.close()

    assert result.allowed is True
    assert installed == []
    assert len(plane.lease_surrenders) == 1


@pytest.mark.unit
def test_4_an_allow_decided_under_an_older_generation_is_not_cached() -> None:
    binding = validated_binding()
    checks = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal checks
        checks += 1
        if checks == 1:
            binding.apply_auth_key(200, auth_key_body(), binding.generation)
        return httpx.Response(200, json=_check_body())

    enforcer = BudgetEnforcer(
        _URL, binding, cache_ttl=60, lease_enabled=False, transport=httpx.MockTransport(handler)
    )
    try:
        enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
        cache_after_first = dict(enforcer._allow_cache)
    finally:
        enforcer.close()

    # The first attempt's allow (older generation) was discarded and re-run;
    # only the re-run's allow may be cached.
    assert checks == 2
    assert len(cache_after_first) == 1


# ── 5. lease-funded rerun ─────────────────────────────────────────────────


@pytest.mark.unit
def test_5_lease_funded_rerun_takes_per_call_admission() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    binding = validated_binding(plane.api_key, project_id=plane.project_id)
    enforcer = BudgetEnforcer(plane.api_url, binding, cache_ttl=0, transport=plane.transport)
    original = enforcer._install_grant
    bumped: list[bool] = []

    def install_then_revalidate(*args: Any, **kwargs: Any) -> Any:
        outcome = original(*args, **kwargs)
        if not bumped:
            bumped.append(True)
            # A concurrent revalidation lands after the grant installed: the
            # lease-funded result is discarded and the check re-runs once.
            binding.apply_auth_key(
                200, auth_key_body(project_id=plane.project_id), binding.generation
            )
        return outcome

    enforcer._install_grant = install_then_revalidate  # type: ignore[method-assign]
    try:
        result = enforcer.check_budget(
            estimated_input_tokens=1,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="rerun",
            call_id="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        )
    finally:
        enforcer.close()

    assert result.allowed is True
    assert result.lease_id is None and result.reservation_id is not None
    assert [check.model for check in plane.checks] == ["gpt-5.5"]


# ── 6. a directly built binding validates its selector ────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("selector", ["", "bad/name", "x" * 101])
def test_6_project_binding_rejects_a_bad_selector(selector: str) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        ProjectBinding(VALID_API_KEY, selector)

    assert excinfo.value.field == "project"


@pytest.mark.unit
def test_6_project_binding_trims_like_the_config() -> None:
    assert ProjectBinding(VALID_API_KEY, "  checkout-bot\t").selector == "checkout-bot"


# ── 7. a provisional refusal's delayed body is fenced too ─────────────────


def _status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError("refused", request=response.request, response=response)


def _expired_body(posture: str) -> bytes:
    return json.dumps(
        {"detail": {"code": "api_key_expired", "project_budget_mode": posture}}
    ).encode()


@pytest.mark.unit
@pytest.mark.parametrize(
    "finalizing_body",
    [
        json.dumps({"detail": {"code": "api_key_invalid"}}).encode(),  # identical
        _expired_body("hard_deny"),  # different, still blocking
    ],
    ids=["identical", "different"],
)
def test_7_a_delayed_refinement_never_weakens_a_refusal_finalized_meanwhile(
    finalizing_body: bytes,
) -> None:
    binding = validated_binding()
    request_a = httpx.Request("POST", f"{_URL}{_CHECK_PATH}")
    _headers, dispatched_a = binding.request_headers()
    status_a = httpx.Response(401, request=request_a)
    binding.observe_status(status_a)  # A's status: provisional block
    assert binding.refusal is not None and binding.refusal.effect == "block"

    # Another response finalizes a blocking refusal before A's body arrives.
    _headers, dispatched_b = binding.request_headers()
    response_b = httpx.Response(
        401, request=httpx.Request("POST", f"{_URL}{_CHECK_PATH}"), content=finalizing_body
    )
    assert binding.note_refusal(_status_error(response_b), dispatched_b)

    # A's delayed body says alert_only: it must not relax B's final answer.
    response_a = httpx.Response(401, request=request_a, content=_expired_body("alert_only"))
    assert binding.note_refusal(_status_error(response_a), dispatched_a)

    assert binding.refusal is not None
    assert binding.refusal.effect == "block"


@pytest.mark.unit
def test_7_a_401_body_never_relaxes_its_own_status_block() -> None:
    binding = validated_binding()
    request = httpx.Request("POST", f"{_URL}{_CHECK_PATH}")
    _headers, dispatched_at = binding.request_headers()
    binding.observe_status(httpx.Response(401, request=request))
    response = httpx.Response(401, request=request, content=_expired_body("alert_only"))

    assert binding.note_refusal(_status_error(response), dispatched_at)

    assert binding.refusal is not None
    assert binding.refusal.effect == "block"


# ── 8. special native paths re-check the stored refusal ───────────────────


@pytest.mark.unit
def test_8_decisions_create_rechecks_after_the_exit_check() -> None:
    plane = FakeControlPlane(mode="alert_only")
    provider = _Provider()
    wrapped = plane.wrap(provider.client(), lease_enabled=False)
    original = wrapped._apply_failover_tuning_directive

    def admitted_then_refused(*args: Any, **kwargs: Any) -> Any:
        _refuse_now([wrapped])()
        return original(*args, **kwargs)

    with (
        wrapped,
        patch_wrapper_local(wrapped, "_apply_failover_tuning_directive", admitted_then_refused),
        pytest.raises(ApiKeyRefusedError),
    ):
        wrapped.decisions.create(model="gpt-5.5", input="hi")

    assert provider.requests == 0


@pytest.mark.unit
def test_8_untracked_tts_passthrough_rechecks_the_stored_refusal() -> None:
    plane = FakeControlPlane()
    provider = _Provider()
    wrapped = plane.wrap(provider.client(), on_unmetered="allow")
    binding = wrapped._solwyn_binding
    binding.refuse(Refusal(code="api_key_invalid", posture="hard_deny"), binding.generation, from_validation=True)
    with wrapped, pytest.raises(ApiKeyRefusedError):
        wrapped.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input="hi")

    assert provider.requests == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "method",
    [
        "converse",
        "converse_stream",
        "invoke_model",
        "invoke_model_with_response_stream",
        "start_async_invoke",
    ],
)
def test_8_non_bedrock_native_passthroughs_recheck_the_stored_refusal(method: str) -> None:
    provider = _Provider()
    raw = provider.client()
    calls: list[str] = []

    def native(**_kwargs: object) -> object:
        calls.append(method)
        return {}

    object.__setattr__(raw, method, native)
    plane = FakeControlPlane()
    wrapped = plane.wrap(raw, on_unmetered="allow")
    binding = wrapped._solwyn_binding
    binding.refuse(Refusal(code="api_key_invalid", posture="hard_deny"), binding.generation, from_validation=True)
    with wrapped, pytest.raises(ApiKeyRefusedError):
        getattr(wrapped, method)(modelId="m")

    assert calls == []
