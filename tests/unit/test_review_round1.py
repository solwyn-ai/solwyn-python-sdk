"""Review round 1 of PR #109: the reviewer's probes as regression tests.

Each test reproduced a blocking finding against 60a9b18 and runs with the
gate's real automatic validation (``project_validation``), like the reviewer's
probes, which ran outside the unit conftest's validation-disabling fixture.

1. provisional install racing a newer final refusal
2. deferred unmetered calls (coroutines, streaming managers) after a refusal
3. a new Responses stream entered after a refusal
4. an unread 401 body cancelled mid-read
5. exit advisory sends that met a refusal
6. ``validate()`` on an outage after an earlier success
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading

import httpx
import pytest

from solwyn._api_key_errors import Refusal
from solwyn._project import ProjectBinding
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.exceptions import ApiKeyRefusedError
from solwyn.testing import FakeControlPlane

openai = pytest.importorskip("openai")

pytestmark = [pytest.mark.unit, pytest.mark.project_validation]

KEY = "solwyn_sk_" + "0" * 64
PROJECT_ID = "proj_" + "f" * 24
AUTH = {
    "account_id": "00000000-0000-4000-8000-0000000000ac",
    "key_id": "00000000-0000-4000-8000-00000000000e",
    "key_prefix": "solwyn_sk_0000000000000000",
    "key_name": "test-key",
    "scope": "full",
    "created_via": "cli",
    "expires_at": "2027-11-04T10:12:00Z",
    "rotation_grace_until": None,
    "project_id": PROJECT_ID,
    "project_name": "review",
    "budget_mode": "hard_deny",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledged", [False, True])
async def test_async_unmetered_coroutine_observes_refusal_before_dispatch(acknowledged):
    requests = []

    def provider(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"id": "batch_1", "object": "batch"})

    raw = openai.AsyncOpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
    )
    plane = FakeControlPlane(mode="hard_deny")
    options = (
        {"acknowledge_untracked": {"batches.create"}} if acknowledged else {"on_unmetered": "allow"}
    )
    wrapped = plane.wrap_async(raw, **options)
    try:
        await wrapped.validate()
        pending = wrapped.batches.create(
            input_file_id="file_test",
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        assert requests == []
        plane.expire_key()
        with pytest.raises(ApiKeyRefusedError):
            await wrapped.validate()
        assert wrapped._solwyn_binding.refusal.effect == "block"
        with contextlib.suppress(ApiKeyRefusedError):
            await pending
        assert requests == [], f"provider dispatched after persistent refusal: {requests}"
    finally:
        await wrapped.close()


def test_sync_unmetered_streaming_context_observes_refusal_before_dispatch():
    requests = []

    def provider(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"id": "batch_1", "object": "batch"})

    raw = openai.OpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    plane = FakeControlPlane(mode="hard_deny")
    wrapped = plane.wrap(raw, on_unmetered="allow")
    try:
        wrapped.validate()
        pending = wrapped.batches.with_streaming_response.create(
            input_file_id="file_test", endpoint="/v1/chat/completions", completion_window="24h"
        )
        assert requests == []
        plane.expire_key()
        with pytest.raises(ApiKeyRefusedError):
            wrapped.validate()
        try:
            with pending:
                pass
        except ApiKeyRefusedError:
            pass
        assert requests == [], f"sync deferred unmetered call dispatched after refusal: {requests}"
    finally:
        wrapped.close()


@pytest.mark.asyncio
async def test_async_unmetered_streaming_context_observes_refusal_before_dispatch():
    requests = []

    def provider(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"id": "batch_1", "object": "batch"})

    raw = openai.AsyncOpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
    )
    plane = FakeControlPlane(mode="hard_deny")
    wrapped = plane.wrap_async(raw, on_unmetered="allow")
    try:
        await wrapped.validate()
        pending = wrapped.batches.with_streaming_response.create(
            input_file_id="file_test", endpoint="/v1/chat/completions", completion_window="24h"
        )
        assert requests == []
        plane.expire_key()
        with pytest.raises(ApiKeyRefusedError):
            await wrapped.validate()
        try:
            async with pending:
                pass
        except ApiKeyRefusedError:
            pass
        assert requests == [], f"async deferred unmetered call dispatched after refusal: {requests}"
    finally:
        await wrapped.close()


class StalledAsyncBody(httpx.AsyncByteStream):
    def __init__(self, started):
        self.started = started

    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b""


@pytest.mark.asyncio
async def test_cancelled_401_body_tightens_existing_unmetered_refusal():
    binding = ProjectBinding(KEY, "review")
    assert binding.apply_auth_key(200, json.dumps(AUTH).encode(), 0)[0] == "valid"
    binding.refuse(Refusal("api_key_expired", "alert_only"), binding.generation)
    assert binding.refusal.effect == "allow_unmetered"
    started = asyncio.Event()

    async def control(request):
        return httpx.Response(401, stream=StalledAsyncBody(started))

    enforcer = AsyncBudgetEnforcer(
        "https://api.test.solwyn.ai",
        binding,
        lease_enabled=False,
        transport=httpx.MockTransport(control),
    )
    try:
        task = asyncio.create_task(enforcer.validate())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert binding.refusal.effect == "block", "unread 401 was forgotten after cancellation"
    finally:
        await enforcer.close()


@pytest.mark.asyncio
async def test_cancelled_401_body_allows_next_wrapped_provider_call():
    from solwyn import AsyncSolwyn

    started = asyncio.Event()
    provider_requests = []

    class Control:
        def __init__(self):
            self.auth_calls = 0

        def handle_request(self, request):
            return httpx.Response(204)

        async def handle_async_request(self, request):
            if request.url.path == "/api/v1/auth/key":
                self.auth_calls += 1
                if self.auth_calls == 1:
                    return httpx.Response(200, json=AUTH)
                if self.auth_calls == 2:
                    return httpx.Response(
                        401,
                        json={
                            "detail": {
                                "code": "api_key_expired",
                                "project_budget_mode": "alert_only",
                            }
                        },
                    )
                return httpx.Response(401, stream=StalledAsyncBody(started))
            if request.url.path == "/api/v1/metadata/ingest":
                return httpx.Response(202, json={"accepted": 1, "rejected": []})
            return httpx.Response(204)

    def provider(request):
        provider_requests.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    raw = openai.AsyncOpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
    )
    wrapped = AsyncSolwyn(
        raw,
        api_key=KEY,
        project="review",
        api_url="https://plane.test",
        control_plane_transport=Control(),
        lease_enabled=False,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    try:
        await wrapped.validate()
        with pytest.raises(ApiKeyRefusedError):
            await wrapped.validate()
        assert wrapped._solwyn_binding.refusal.effect == "allow_unmetered"
        task = asyncio.create_task(wrapped.validate())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with contextlib.suppress(ApiKeyRefusedError):
            await wrapped.chat.completions.create(
                model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
            )
        assert provider_requests == [], (
            f"unmetered provider dispatch after unread 401 was cancelled: {provider_requests}"
        )
    finally:
        await wrapped.close()


def test_new_responses_stream_rechecks_refusal_at_context_entry():
    requests = []

    def provider(request):
        requests.append(str(request.url))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n"
        )

    raw = openai.OpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    plane = FakeControlPlane(mode="hard_deny")
    wrapped = plane.wrap(raw, lease_enabled=False)
    try:
        manager = wrapped.responses.stream(model="gpt-5.5", input="hi")
        assert requests == []
        plane.expire_key()
        with pytest.raises(ApiKeyRefusedError):
            wrapped.validate()
        assert wrapped._solwyn_binding.refusal.effect == "block"
        try:
            with manager:
                pass
        except ApiKeyRefusedError:
            pass
        assert requests == [], f"new response dispatched at context entry after refusal: {requests}"
    finally:
        wrapped.close()


class _CountingLock:
    def __init__(self):
        self._inner = threading.Lock()
        self.acquisitions = 0

    def __enter__(self):
        self._inner.acquire()
        self.acquisitions += 1
        return self

    def __exit__(self, *args):
        self._inner.release()


def test_status_hook_checks_and_installs_in_one_critical_section():
    binding = ProjectBinding(KEY, "review")
    binding.apply_auth_key(200, json.dumps(AUTH).encode(), 0)
    lock = _CountingLock()
    binding._lock = lock
    request = httpx.Request("POST", "https://api.test.solwyn.ai/api/v1/budgets/check")

    binding.observe_status(httpx.Response(401, request=request))
    acquisitions = lock.acquisitions

    assert acquisitions == 1, "the no-refusal check and the install must not be split"
    assert binding.refusal is not None and binding.refusal.effect == "block"


def _expired(posture):
    return json.dumps(
        {"detail": {"code": "api_key_expired", "project_budget_mode": posture}}
    ).encode()


@pytest.mark.parametrize("order", ["old_status_first", "newer_final_first"])
def test_an_older_response_never_weakens_a_newer_final_refusal(order):
    binding = ProjectBinding(KEY, "review")
    binding.apply_auth_key(200, json.dumps(AUTH).encode(), 0)
    old_request = httpx.Request("POST", "https://api.test.solwyn.ai/api/v1/budgets/check")
    _headers, old_dispatch = binding.request_headers()

    def newer_final():
        _headers, newer_dispatch = binding.request_headers()
        response = httpx.Response(
            401,
            request=httpx.Request("POST", "https://api.test.solwyn.ai/x"),
            content=_expired("hard_deny"),
        )
        binding.note_refusal(
            httpx.HTTPStatusError("x", request=response.request, response=response),
            newer_dispatch,
        )

    if order == "newer_final_first":
        newer_final()
        binding.observe_status(httpx.Response(401, request=old_request))
    else:
        binding.observe_status(httpx.Response(401, request=old_request))
        newer_final()
    old_body = httpx.Response(401, request=old_request, content=_expired("alert_only"))
    binding.note_refusal(
        httpx.HTTPStatusError("x", request=old_request, response=old_body), old_dispatch
    )

    assert binding.refusal.effect == "block"
    assert binding.refusal.posture == "hard_deny"


@pytest.mark.parametrize("failure", ["timeout", "5xx"])
def test_explicit_revalidation_returns_none_on_outage_after_prior_success(failure):
    plane = FakeControlPlane()
    requests = []

    def provider(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={})

    raw = openai.OpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    wrapped = plane.wrap(raw)
    try:
        assert wrapped.validate() is not None
        if failure == "timeout":
            with plane.outage(path="/api/v1/auth/key"):
                result = wrapped.validate()
        else:
            with plane.raw_response(503, b"unavailable", path="/api/v1/auth/key"):
                result = wrapped.validate()
        assert result is None, f"validate() returned stale cached auth after {failure}: {result}"
    finally:
        wrapped.close()


@pytest.mark.parametrize(
    "prior,status,body",
    [
        ("valid", 403, b'{"detail":{"code":"read_only_key"}}'),
        # A refused binding sends nothing (see the test below), so the hard
        # 401 is observed from a valid one.
        (
            "valid",
            401,
            b'{"detail":{"code":"api_key_expired","project_budget_mode":"hard_deny"}}',
        ),
    ],
)
def test_exit_advisory_refusal_reaches_the_shared_binding(prior, status, body):
    from collections import deque
    from datetime import UTC, datetime

    from solwyn._lifecycle import _drain_queues_blocking
    from solwyn._project import sync_status_hook
    from solwyn._surfaces import SurfaceContext
    from solwyn.reporter import _UntrackedReportState

    binding = ProjectBinding(KEY, "review")
    binding.apply_auth_key(200, json.dumps(AUTH).encode(), 0)
    if prior == "unmetered":
        binding.refuse(Refusal("api_key_expired", "alert_only"), binding.generation)
    advisory = _UntrackedReportState("review-advisory-instance")
    advisory.observe(
        context=SurfaceContext(
            provider="openai", dialect="openai", client_shape="openai_sdk", mode="async"
        ),
        surface="batches.create",
        rule_kind="unmetered_spend",
        capability_scope="operation",
        posture="allow",
        seen_at=datetime.now(UTC),
    )
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(status, content=body)

    def client():
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            event_hooks={"response": [sync_status_hook(binding)]},
        )

    _drain_queues_blocking(
        deque(),
        deque(),
        deque(),
        "https://plane.test",
        binding,
        None,
        2.0,
        lambda *_args: None,
        client,
        untracked_state=advisory,
    )
    assert paths == ["/api/v1/untracked-surfaces"]
    assert binding.refusal is not None and binding.refusal.effect == "block", (
        "exit advisory response skipped refusal classification"
    )


def test_gc_advisory_hard_401_cannot_leave_a_live_enforcer_unmetered():
    import gc
    import weakref
    from datetime import UTC, datetime

    from solwyn._surfaces import SurfaceContext
    from solwyn.reporter import AsyncMetadataReporter

    binding = ProjectBinding(KEY, "review")
    binding.apply_auth_key(200, json.dumps(AUTH).encode(), 0)
    paths = []

    def control(request):
        paths.append(request.url.path)
        return httpx.Response(
            401, json={"detail": {"code": "api_key_expired", "project_budget_mode": "hard_deny"}}
        )

    transport = httpx.MockTransport(control)
    reporter = AsyncMetadataReporter(
        "https://plane.test", binding, transport=transport, sdk_instance_id="review-gc-advisory"
    )
    # Retaining the HTTP client permits explicit cleanup after reporter GC;
    # the factory in the real weakref finalizer creates its own sync client.
    async_http = reporter._http
    reporter.observe_untracked_surface(
        context=SurfaceContext(
            provider="openai", dialect="openai", client_shape="openai_sdk", mode="async"
        ),
        surface="batches.create",
        rule_kind="unmetered_spend",
        capability_scope="operation",
        posture="allow",
        seen_at=datetime.now(UTC),
    )
    reference = weakref.ref(reporter)
    enforcer = BudgetEnforcer(
        "https://plane.test", binding, transport=transport, lease_enabled=False
    )
    try:
        del reporter
        gc.collect()
        assert reference() is None
        assert paths == ["/api/v1/untracked-surfaces"]
        try:
            result = enforcer.check_budget(
                estimated_input_tokens=1, model="gpt-5.5", provider="openai"
            )
        except ApiKeyRefusedError:
            return
        pytest.fail(
            f"live enforcer still admits unmetered after GC flush observed hard 401: {result}"
        )
    finally:
        asyncio.run(async_http.aclose())
        enforcer.close()


class _Interrupted(BaseException):
    """Stands in for KeyboardInterrupt / SystemExit during a body read."""


class _InterruptedBody(httpx.SyncByteStream):
    def __iter__(self):
        raise _Interrupted()


def test_interrupted_sync_401_body_tightens_existing_unmetered_refusal():
    binding = ProjectBinding(KEY, "review")
    binding.apply_auth_key(200, json.dumps(AUTH).encode(), 0)
    binding.refuse(Refusal("api_key_expired", "alert_only"), binding.generation)
    assert binding.refusal.effect == "allow_unmetered"
    enforcer = BudgetEnforcer(
        "https://api.test.solwyn.ai",
        binding,
        lease_enabled=False,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, stream=_InterruptedBody())
        ),
    )
    try:
        with pytest.raises(_Interrupted):
            enforcer.validate()
    finally:
        enforcer.close()

    assert binding.refusal.effect == "block", "an interrupted 401 body was forgotten"


def test_exit_advisory_sends_nothing_while_the_binding_is_refused():
    """A refused binding (any effect) sends Solwyn nothing but /auth/key."""
    from collections import deque
    from datetime import UTC, datetime

    from solwyn._lifecycle import _drain_queues_blocking
    from solwyn._project import sync_status_hook
    from solwyn._surfaces import SurfaceContext
    from solwyn.reporter import _UntrackedReportState

    binding = ProjectBinding(KEY, "review")
    binding.apply_auth_key(200, json.dumps(AUTH).encode(), 0)
    binding.refuse(Refusal("api_key_expired", "alert_only"), binding.generation)
    advisory = _UntrackedReportState("review-advisory-instance")
    advisory.observe(
        context=SurfaceContext(
            provider="openai", dialect="openai", client_shape="openai_sdk", mode="async"
        ),
        surface="batches.create",
        rule_kind="unmetered_spend",
        capability_scope="operation",
        posture="allow",
        seen_at=datetime.now(UTC),
    )
    paths = []

    def client():
        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(202, json={"accepted": 1})

        return httpx.Client(
            transport=httpx.MockTransport(handler),
            event_hooks={"response": [sync_status_hook(binding)]},
        )

    _drain_queues_blocking(
        deque(),
        deque(),
        deque(),
        "https://plane.test",
        binding,
        None,
        2.0,
        lambda *_args: None,
        client,
        untracked_state=advisory,
    )

    assert paths == []
    assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"
