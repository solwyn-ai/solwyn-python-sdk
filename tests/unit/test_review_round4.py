"""Round-4 classes from the TypeScript SDK review, checked against Python.

1. An independent refusal of the SAME strength still advances the ordering,
   so a ``/auth/key`` answer dispatched before it cannot relax or clear it;
   only a request's own status and body coalesce.
2. Handles from derived clients (``with_options``, ``copy``, raw-response
   namespaces) and nested namespaces stay fenced.
3. Values produced later by a fenced value (an awaited page, an entered live
   session or realtime connection, a page's continuation) are fenced too.
4. A ``200 read_only`` revalidation resets the 60 s cooldown.
5. A re-check that cannot reach Solwyn while refused retries after 5 s.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import auth_key_body, validated_binding

from solwyn._api_key_errors import Refusal
from solwyn._base import _defer_key_refusal
from solwyn._constants import AUTH_KEY_PATH
from solwyn._project import REFUSED_OUTAGE_RETRY_S, REFUSED_RETRY_S
from solwyn.exceptions import ApiKeyRefusedError
from solwyn.testing import FakeControlPlane

pytestmark = [pytest.mark.unit, pytest.mark.project_validation]

_URL = "https://api.test.solwyn.ai"
_HARD_401 = json.dumps(
    {"detail": {"code": "api_key_expired", "project_budget_mode": "hard_deny"}}
).encode()
_ALERT_401 = json.dumps(
    {"detail": {"code": "api_key_expired", "project_budget_mode": "alert_only"}}
).encode()


def _status_error(request: httpx.Request, body: bytes) -> httpx.HTTPStatusError:
    response = httpx.Response(401, request=request, content=body)
    return httpx.HTTPStatusError("refused", request=request, response=response)


# ── 1. same-strength independent refusals advance the ordering ────────────


def _blocked_binding() -> Any:
    binding = validated_binding()
    request = httpx.Request("POST", f"{_URL}/api/v1/budgets/check")
    _headers, dispatched_at = binding.request_headers()
    binding.observe_status(httpx.Response(401, request=request))
    binding.note_refusal(_status_error(request, _HARD_401), dispatched_at)
    return binding


@pytest.mark.parametrize("answer", ["200", "alert_only_401"])
def test_1_an_identical_independent_refusal_fences_an_earlier_auth_key_answer(
    answer: str,
) -> None:
    binding = _blocked_binding()
    _headers, revalidation_dispatch = binding.request_headers()  # /auth/key goes out

    # Meanwhile an independent request's late answer repeats the same refusal.
    other = httpx.Request("POST", f"{_URL}/api/v1/budgets/confirm")
    _headers, other_dispatch = binding.request_headers()
    binding.note_refusal(_status_error(other, _HARD_401), other_dispatch)

    if answer == "200":
        outcome, _ = binding.apply_auth_key(200, auth_key_body(), revalidation_dispatch)
        assert outcome == "fenced"
    else:
        binding.apply_auth_key(401, _ALERT_401, revalidation_dispatch)
    assert binding.refusal is not None and binding.refusal.effect == "block"


def test_1_an_independent_401_status_on_a_stored_block_advances_the_ordering() -> None:
    binding = _blocked_binding()
    _headers, revalidation_dispatch = binding.request_headers()
    gen = binding.refusal_gen

    late = httpx.Request("POST", f"{_URL}/api/v1/metadata/ingest")
    binding.observe_status(httpx.Response(401, request=late))

    assert binding.refusal_gen > gen
    outcome, _ = binding.apply_auth_key(200, auth_key_body(), revalidation_dispatch)
    assert outcome == "fenced"
    assert binding.state == "refused"


def test_1_a_requests_own_status_and_body_coalesce() -> None:
    binding = validated_binding()
    request = httpx.Request("POST", f"{_URL}/api/v1/budgets/check")
    _headers, dispatched_at = binding.request_headers()
    binding.observe_status(httpx.Response(401, request=request))
    gen = binding.refusal_gen

    binding.note_refusal(_status_error(request, _HARD_401), dispatched_at)

    assert binding.refusal_gen == gen  # its own body adds details only
    assert binding.refusal is not None and binding.refusal.code == "api_key_expired"


# ── 2. derived clients and nested namespaces stay fenced ──────────────────


openai = pytest.importorskip("openai")


class _Provider:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.method == "GET" and request.url.path.endswith("/batches"):
            # A few pages only: a regression fails on the request count
            # instead of paginating forever.
            pages = sum(1 for path in self.paths if path.endswith("/batches"))
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "b1", "object": "batch"}],
                    "has_more": pages < 3,
                    "first_id": "b1",
                    "last_id": "b1",
                },
            )
        return httpx.Response(200, json={"id": "x", "object": "batch"})

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)


def _wrapped(provider: _Provider) -> Any:
    raw = openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(provider.handler)),
    )
    return FakeControlPlane().wrap(raw, on_unmetered="allow")


def _refuse(wrapped: Any) -> None:
    binding = wrapped._solwyn_binding
    binding.refuse(Refusal("api_key_invalid", "hard_deny"), binding.generation, from_validation=True)


_BATCH = {"input_file_id": "f", "endpoint": "/v1/chat/completions", "completion_window": "24h"}


@pytest.mark.parametrize(
    "derive",
    [
        lambda w: w.with_options(timeout=5),
        lambda w: w.copy(),
        lambda w: w.with_options(timeout=5).with_options(max_retries=1),
    ],
    ids=["with_options", "copy", "nested_with_options"],
)
@pytest.mark.parametrize(
    "use",
    [
        lambda d: d.batches.create(**_BATCH),
        lambda d: d.batches.with_raw_response.create(**_BATCH),
        lambda d: d.chat.completions.create(model="gpt-5.5", messages=[]),
    ],
    ids=["batches", "raw_response", "chat_on_derived_client"],
)
def test_2_a_derived_client_made_before_a_refusal_stays_fenced(derive: Any, use: Any) -> None:
    provider = _Provider()
    wrapped = _wrapped(provider)
    with wrapped:
        derived = derive(wrapped)  # before the refusal
        assert isinstance(derived, openai.OpenAI)  # still type-transparent
        _refuse(wrapped)
        with pytest.raises(ApiKeyRefusedError):
            use(derived)

    assert provider.paths == []


# ── 3. values produced later are fenced too ───────────────────────────────


@pytest.mark.parametrize(
    "continue_",
    [lambda page: page.get_next_page(), lambda page: list(page)],
    ids=["get_next_page", "auto_pagination"],
)
def test_3_a_page_made_before_a_refusal_never_fetches_another(continue_: Any) -> None:
    provider = _Provider()
    wrapped = _wrapped(provider)
    with wrapped:
        page = wrapped.batches.list()
        provider.paths.clear()
        _refuse(wrapped)
        with pytest.raises(ApiKeyRefusedError):
            continue_(page)

    assert provider.paths == []


@pytest.mark.asyncio
async def test_3_an_awaited_async_page_is_fenced_for_its_continuation() -> None:
    provider = _Provider()
    raw = openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.async_handler)),
    )
    wrapped = FakeControlPlane().wrap_async(raw, on_unmetered="allow")
    try:
        page = await wrapped.batches.list()
        provider.paths.clear()
        _refuse(wrapped)
        with pytest.raises(ApiKeyRefusedError):
            await page.get_next_page()
    finally:
        await wrapped.close()

    assert provider.paths == []


class _Owner:
    """Stands in for the Solwyn client: refused once ``refused`` is set."""

    def __init__(self) -> None:
        self.refused = False

    def _enforce_key_refusal(self, path: str) -> None:
        if self.refused:
            raise ApiKeyRefusedError("refused", code="api_key_invalid", field="api_key")

    async def _enforce_key_refusal_async(self, path: str) -> None:
        self._enforce_key_refusal(path)


class AsyncSession:
    """A Google Live style session: it sends on each method call."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.response = _RealtimeResponseResource(self.sent)

    async def send_client_content(self, **_kwargs: object) -> None:
        self.sent.append("send_client_content")


class _RealtimeResponseResource:
    def __init__(self, sent: list[str]) -> None:
        self._sent = sent

    def create(self) -> None:
        self._sent.append("response.create")


AsyncSession.__module__ = "google.genai.live"
_RealtimeResponseResource.__module__ = "openai.resources.realtime"


class LiveConnectManager:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *_args: object) -> None:
        return None


LiveConnectManager.__module__ = "google.genai.live"


@pytest.mark.asyncio
async def test_3_a_session_entered_before_a_refusal_never_sends_after_it() -> None:
    owner = _Owner()
    session = AsyncSession()
    manager = _defer_key_refusal(LiveConnectManager(session), owner, "live.connect")  # type: ignore[arg-type]

    async with manager as live:
        assert isinstance(live, AsyncSession)
        owner.refused = True
        with pytest.raises(ApiKeyRefusedError):
            await live.send_client_content(turns=[])
        with pytest.raises(ApiKeyRefusedError):
            live.response.create()  # a nested namespace of the session

    assert session.sent == []


@pytest.mark.asyncio
async def test_3_a_value_produced_by_a_fenced_coroutine_is_fenced() -> None:
    owner = _Owner()
    session = AsyncSession()

    async def connect() -> AsyncSession:
        return session

    live = await _defer_key_refusal(connect(), owner, "live.connect")  # type: ignore[arg-type]
    owner.refused = True
    with pytest.raises(ApiKeyRefusedError):
        await live.send_client_content(turns=[])

    assert session.sent == []


# ── 4 and the outage retry: revalidation cadence while refused ────────────


class _Completions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))


class _OpenAIStub:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())

    def with_options(self, **_kwargs: object) -> _OpenAIStub:
        return self


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


class _Clock:
    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now


def _blocked_client(plane: FakeControlPlane) -> tuple[Any, _Clock]:
    client = plane.wrap(_OpenAIStub(), lease_enabled=False)
    clock = _Clock()
    client._solwyn_binding._clock = clock
    client.validate()
    plane.revoke_key()
    with pytest.raises(ApiKeyRefusedError):
        client.chat.completions.create(model="gpt-5.5", messages=[])
    return client, clock


def test_4_a_200_read_only_revalidation_resets_the_sixty_second_cooldown() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    client, clock = _blocked_client(plane)
    try:
        plane.restore_key()
        plane.set_key_scope("read_only")
        clock.now += REFUSED_RETRY_S
        with pytest.raises(ApiKeyRefusedError) as excinfo:
            client.chat.completions.create(model="gpt-5.5", messages=[])
        assert excinfo.value.code == "read_only_key"
        auth_after = len(plane.auth_key_requests)

        for _ in range(5):
            clock.now += 10
            with pytest.raises(ApiKeyRefusedError):
                client.chat.completions.create(model="gpt-5.5", messages=[])
    finally:
        client.close()

    assert len(plane.auth_key_requests) == auth_after  # no /auth/key per call


@pytest.mark.parametrize("failure", ["transport_error", "timeout", "503"])
def test_outage_while_refused_retries_after_five_seconds_not_sixty(failure: str) -> None:
    from solwyn.budget import BudgetEnforcer

    clock = _Clock()
    binding = validated_binding()
    binding._clock = clock
    binding.refuse(Refusal("api_key_invalid", "hard_deny"), binding.generation)
    attempts: list[float] = []
    solwyn_up = [False]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == AUTH_KEY_PATH  # nothing else while refused
        attempts.append(clock.now)
        if solwyn_up[0]:
            return httpx.Response(200, content=auth_key_body(), request=request)
        if failure == "transport_error":
            raise httpx.ConnectError("down", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(503, content=b"unavailable", request=request)

    enforcer = BudgetEnforcer(
        _URL, binding, lease_enabled=False, transport=httpx.MockTransport(handler)
    )

    def check() -> None:
        enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")

    try:
        start = clock.now
        for _ in range(12):  # 12 s of calls, one per second, while Solwyn is down
            with pytest.raises(ApiKeyRefusedError):
                check()
            clock.now += 1
        solwyn_up[0] = True
        clock.now += REFUSED_OUTAGE_RETRY_S
        check()  # the next re-check, 5 s later, recovers
    finally:
        enforcer.close()

    assert REFUSED_OUTAGE_RETRY_S == 5.0
    # Single-flight, at most one /auth/key per 5 s while down, never per call.
    down = [at - start for at in attempts[:-1]]
    assert down == [0, 5, 10]
    assert binding.state == "valid"


def test_a_refusal_answer_still_backs_off_sixty_seconds() -> None:
    from solwyn.budget import BudgetEnforcer

    clock = _Clock()
    binding = validated_binding()
    binding._clock = clock
    binding.refuse(Refusal("api_key_invalid", "hard_deny"), binding.generation)
    attempts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(clock.now)
        return httpx.Response(401, content=_HARD_401, request=request)

    enforcer = BudgetEnforcer(
        _URL, binding, lease_enabled=False, transport=httpx.MockTransport(handler)
    )
    try:
        start = clock.now
        for _ in range(70):
            with pytest.raises(ApiKeyRefusedError):
                enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
            clock.now += 1
    finally:
        enforcer.close()

    assert [at - start for at in attempts] == [0, 60]


# ── round 5: a second 401 status during an in-flight re-check ─────────────


@pytest.mark.parametrize("stale", ["200", "alert_only_401", "hard_401", "outage"])
def test_5_a_second_401_during_a_recheck_fences_it_and_queues_a_fresh_one(stale: str) -> None:
    from solwyn.budget import BudgetEnforcer

    clock = _Clock()
    binding = validated_binding()
    binding._clock = clock
    first = httpx.Request("POST", f"{_URL}/api/v1/budgets/check")
    binding.observe_status(httpx.Response(401, request=first))  # blocked, re-check due now
    seen: list[str] = []
    effects: list[str] = []
    binding.add_listener(
        lambda: effects.append(binding.refusal.effect if binding.refusal else "cleared")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == AUTH_KEY_PATH
        seen.append(stale if not seen else "fresh")
        if len(seen) == 1:
            # While this re-check is in flight, another request's 401 STATUS
            # arrives (its body has not): the block must hold.
            second = httpx.Request("POST", f"{_URL}/api/v1/metadata/ingest")
            binding.observe_status(httpx.Response(401, request=second))
            if stale == "200":
                return httpx.Response(200, content=auth_key_body(), request=request)
            if stale == "outage":
                raise httpx.ConnectError("down", request=request)
            body = _ALERT_401 if stale == "alert_only_401" else _HARD_401
            return httpx.Response(401, content=body, request=request)
        # The fresh re-check, dispatched after the second 401, decides.
        return httpx.Response(401, content=_ALERT_401, request=request)

    enforcer = BudgetEnforcer(
        _URL, binding, lease_enabled=False, transport=httpx.MockTransport(handler)
    )
    try:
        result = enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()

    # The stale answer never cleared or relaxed the block ...
    assert "cleared" not in effects
    assert "allow_unmetered" not in effects[:-1]
    # ... and the queued fresh re-check ran right after it, in the same call.
    assert seen == [stale, "fresh"]
    assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"
    assert result.allowed is True and result.reservation_id is None
