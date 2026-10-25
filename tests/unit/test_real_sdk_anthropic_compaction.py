"""Real-SDK check: Anthropic beta Messages and server-side compaction metering.

Drives the genuine ``anthropic`` client (beta pydantic types, SSE decoding,
beta stream events) through its own in-memory ``httpx2.MockTransport`` and the
hermetic ``FakeControlPlane``. Compaction usage arrives only in
``usage.iterations`` and must be folded into the settled token counts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from solwyn.exceptions import ConfigurationError, UntrackedSpendSurfaceError
from solwyn.testing import FakeControlPlane

anthropic = pytest.importorskip("anthropic", minversion="1.14.0")
httpx2 = pytest.importorskip("httpx2")
openai = pytest.importorskip("openai")

pytestmark = pytest.mark.unit

_PROMPT = "private prompt text that must never leave the process"
_COMPLETION = "private completion text"
_THRESHOLD_BETA = "compact-2026-01-12"
_ON_DEMAND_BETA = "compact-2026-09-04"


def _iteration(kind: str, input_tokens: int, output_tokens: int, **cache: int) -> dict[str, Any]:
    return {
        "type": kind,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache.get("cache_read", 0),
        "cache_creation_input_tokens": cache.get("cache_creation", 0),
    }


# (top-level usage, iterations, expected settled token_details subset)
_THRESHOLD = (
    {"input_tokens": 500, "output_tokens": 1_000, "cache_read_input_tokens": 20_000,
     "cache_creation_input_tokens": 2_500},
    [
        _iteration("compaction", 2_000, 3_500, cache_read=178_000),
        _iteration("message", 500, 1_000, cache_read=20_000, cache_creation=2_500),
    ],
    {
        "input_tokens": 203_000,
        "output_tokens": 4_500,
        "cached_input_tokens": 198_000,
        "cache_creation_5m_tokens": 2_500,
        "compaction_input_tokens": 180_000,
        "compaction_cached_input_tokens": 178_000,
        "compaction_output_tokens": 3_500,
    },
)  # fmt: skip
_ON_DEMAND = (
    {"input_tokens": 0, "output_tokens": 0},
    [_iteration("compaction", 144, 276)],
    {
        "input_tokens": 144,
        "output_tokens": 276,
        "compaction_input_tokens": 144,
        "compaction_output_tokens": 276,
    },
)


def _message(usage: dict[str, Any], model: str = "claude-sonnet-5") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": _COMPLETION}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def _sse(usage: dict[str, Any], iterations: list[dict[str, Any]]) -> str:
    start_usage = {key: value for key, value in usage.items() if key != "output_tokens"} | {
        "output_tokens": 1
    }
    start = {**_message(start_usage), "content": [], "stop_reason": None}
    partial = iterations[:1] if len(iterations) > 1 else []
    events = [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": _COMPLETION},
        },
        {"type": "content_block_stop", "index": 0},
        # A cumulative interim delta, then the final one: iterations replace.
        {
            "type": "message_delta",
            "delta": {"stop_reason": None, "stop_sequence": None},
            "usage": {"output_tokens": 1, "iterations": partial},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {**usage, "iterations": iterations},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _handler(
    seen: list[Any], usage: dict[str, Any], iterations: list[dict[str, Any]]
) -> Callable[[Any], Any]:
    def handle(request: Any) -> Any:
        seen.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx2.Response(
                200,
                request=request,
                text=_sse(usage, iterations),
                headers={"content-type": "text/event-stream"},
            )
        payload = _message(
            {**usage, "iterations": iterations}, model=body.get("model", "claude-sonnet-5")
        )
        return httpx2.Response(200, request=request, json=payload)

    return handle


def _anthropic_client(handler: Callable[[Any], Any], *, is_async: bool = False) -> Any:
    if is_async:
        return anthropic.AsyncAnthropic(
            api_key="sk-ant-test",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
    return anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


async def _call(wrapper: Any, is_async: bool, streaming: bool, **kwargs: Any) -> Any:
    response = wrapper.beta.messages.create(stream=streaming, **kwargs)
    if is_async:
        response = await response
    if streaming:
        if is_async:
            return [event.type async for event in response]
        return [event.type for event in response]
    return response


async def _close(wrapper: Any, is_async: bool) -> None:
    if is_async:
        await wrapper.close()
    else:
        wrapper.close()


def _settled(plane: FakeControlPlane) -> dict[str, Any]:
    assert len(plane.ingested) == 1
    event = plane.ingested[0]
    assert event.token_details is not None
    return event.token_details.model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["message", "stream"])
@pytest.mark.parametrize(
    ("mode", "case", "request_fields"),
    [
        (
            "threshold",
            _THRESHOLD,
            {"context_management": {"edits": [{"type": "compact_20260112"}]}},
        ),
        ("on-demand", _ON_DEMAND, {"compaction": {"type": "summarize"}}),
    ],
    ids=["threshold", "on-demand"],
)
async def test_beta_messages_create_settles_compaction_iterations(
    is_async: bool,
    streaming: bool,
    mode: str,
    case: tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]],
    request_fields: dict[str, Any],
) -> None:
    usage, iterations, expected = case
    seen: list[Any] = []
    provider = _anthropic_client(_handler(seen, usage, iterations), is_async=is_async)
    plane = FakeControlPlane()
    wrapper = (plane.wrap_async if is_async else plane.wrap)(
        provider, lease_enabled=False, on_unmetered="raise"
    )
    beta = _THRESHOLD_BETA if mode == "threshold" else _ON_DEMAND_BETA
    try:
        result = await _call(
            wrapper,
            is_async,
            streaming,
            model="claude-sonnet-5",
            max_tokens=64,
            messages=[{"role": "user", "content": _PROMPT}],
            betas=[beta],
            **request_fields,
        )
    finally:
        await _close(wrapper, is_async)

    if streaming:
        assert result[0] == "message_start"
        assert result[-1] == "message_stop"
    else:
        assert type(result).__name__ == "BetaMessage"
        assert result.content[0].text == _COMPLETION

    # The beta request reached the provider untouched.
    assert len(seen) == 1
    request = seen[0]
    # anthropic 2.3 stopped sending ``?beta=true``; the header marks the beta call.
    assert request.url.path == "/v1/messages"
    assert request.headers["anthropic-beta"] == beta
    body = json.loads(request.content)
    for key, value in request_fields.items():
        assert body[key] == value
    assert body["messages"] == [{"role": "user", "content": _PROMPT}]

    settled = _settled(plane)
    for key, value in expected.items():
        assert settled[key] == value, key
    # Zero compaction buckets stay off the wire.
    absent = {
        "compaction_input_tokens",
        "compaction_cached_input_tokens",
        "compaction_cache_creation_5m_tokens",
        "compaction_cache_creation_1h_tokens",
        "compaction_output_tokens",
    } - set(expected)
    assert not absent & set(settled)

    event = plane.ingested[0]
    assert event.input_tokens == expected["input_tokens"]
    assert event.output_tokens == expected["output_tokens"]
    wire = event.model_dump_json()
    assert _PROMPT not in wire
    assert _COMPLETION not in wire
    assert "context_management" not in wire
    assert "summarize" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["message", "stream"])
async def test_beta_messages_without_iterations_settle_like_messages_create(
    is_async: bool, streaming: bool
) -> None:
    usage = {"input_tokens": 25, "output_tokens": 11, "cache_read_input_tokens": 5}
    results: dict[str, dict[str, Any]] = {}
    for surface in ("messages", "beta"):
        seen: list[Any] = []

        def handle(request: Any, seen: list[Any] = seen) -> Any:
            seen.append(request)
            body = json.loads(request.content)
            if body.get("stream"):
                text = _sse(usage, []).replace(',"iterations": []', "")
                return httpx2.Response(
                    200, request=request, text=text, headers={"content-type": "text/event-stream"}
                )
            return httpx2.Response(200, request=request, json=_message(usage))

        plane = FakeControlPlane()
        wrapper = (plane.wrap_async if is_async else plane.wrap)(
            _anthropic_client(handle, is_async=is_async), lease_enabled=False
        )
        try:
            namespace = wrapper.beta.messages if surface == "beta" else wrapper.messages
            response = namespace.create(
                model="claude-sonnet-5",
                max_tokens=16,
                messages=[{"role": "user", "content": "hi"}],
                stream=streaming,
            )
            if is_async:
                response = await response
            if streaming and is_async:
                assert [event async for event in response]
            elif streaming:
                assert list(response)
        finally:
            await _close(wrapper, is_async)
        results[surface] = _settled(plane)
    assert results["beta"] == results["messages"]
    assert results["beta"]["input_tokens"] == 30
    assert not any(key.startswith("compaction_") for key in results["beta"])


def test_non_beta_messages_create_counts_compaction_iterations_from_untyped_extra() -> None:
    # Compaction requested through extra_body on the GA surface: the GA Usage
    # model carries ``iterations`` as an untyped extra list of plain dicts.
    usage, iterations, expected = _THRESHOLD
    seen: list[Any] = []
    plane = FakeControlPlane()
    wrapper = plane.wrap(_anthropic_client(_handler(seen, usage, iterations)), lease_enabled=False)
    with wrapper:
        response = wrapper.messages.create(
            model="claude-sonnet-5",
            max_tokens=64,
            messages=[{"role": "user", "content": _PROMPT}],
            extra_headers={"anthropic-beta": _THRESHOLD_BETA},
            extra_body={"context_management": {"edits": [{"type": "compact_20260112"}]}},
        )
    assert type(response).__name__ == "Message"
    assert isinstance(response.usage.iterations[0], dict)  # untyped extra
    settled = _settled(plane)
    for key, value in expected.items():
        assert settled[key] == value, key


@pytest.mark.parametrize(
    ("leaf", "kwargs"),
    [
        ("stream", {"model": "claude-sonnet-5", "max_tokens": 8, "messages": []}),
        ("count_tokens", {"model": "claude-sonnet-5", "messages": []}),
        ("parse", {"model": "claude-sonnet-5", "max_tokens": 8, "messages": []}),
    ],
)
def test_other_beta_messages_leaves_keep_their_unmetered_posture(
    leaf: str, kwargs: dict[str, Any]
) -> None:
    plane = FakeControlPlane()
    provider = _anthropic_client(lambda request: pytest.fail("must not dispatch"))
    with plane.wrap(provider, lease_enabled=False, on_unmetered="raise") as wrapper:
        with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
            getattr(wrapper.beta.messages, leaf)
        assert exc_info.value.surface == f"beta.messages.{leaf}"
        with pytest.raises(UntrackedSpendSurfaceError):
            _ = wrapper.beta.messages.batches.create
        with pytest.raises(UntrackedSpendSurfaceError):
            _ = wrapper.beta.messages.with_raw_response
    assert plane.ingested == []


def test_beta_messages_create_acknowledgment_is_rejected_as_metered() -> None:
    plane = FakeControlPlane()
    provider = _anthropic_client(lambda request: pytest.fail("must not dispatch"))
    with pytest.raises(ConfigurationError):
        plane.wrap(
            provider,
            lease_enabled=False,
            acknowledge_untracked=frozenset({"beta.messages.create"}),
        )


# ---------------------------------------------------------------------------
# Failover: beta requests never cross providers
# ---------------------------------------------------------------------------


def _rate_limited(seen: list[Any]) -> Callable[[Any], Any]:
    def handle(request: Any) -> Any:
        seen.append(request)
        return httpx2.Response(
            429,
            request=request,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
        )

    return handle


def test_beta_call_never_fails_over_to_another_provider() -> None:
    anthropic_seen: list[Any] = []
    openai_seen: list[Any] = []
    primary = _anthropic_client(_rate_limited(anthropic_seen))
    fallback = openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: openai_seen.append(request) or pytest.fail("crossed providers")
            )
        ),
    )
    plane = FakeControlPlane()
    wrapper = plane.wrap(primary, lease_enabled=False, fallback=[(fallback, "gpt-5.5")])
    with wrapper, pytest.raises(anthropic.RateLimitError) as exc_info:
        wrapper.beta.messages.create(
            model="claude-sonnet-5",
            max_tokens=8,
            messages=[{"role": "user", "content": _PROMPT}],
            betas=[_ON_DEMAND_BETA],
            compaction={"type": "summarize"},
        )
    assert len(anthropic_seen) == 1
    assert openai_seen == []
    assert _PROMPT not in str(exc_info.value)


def test_beta_call_fails_over_to_same_provider_model_on_the_beta_surface() -> None:
    primary_seen: list[Any] = []
    fallback_seen: list[Any] = []
    usage, iterations, expected = _ON_DEMAND
    primary = _anthropic_client(_rate_limited(primary_seen))
    fallback = _anthropic_client(_handler(fallback_seen, usage, iterations))
    plane = FakeControlPlane()
    wrapper = plane.wrap(primary, lease_enabled=False, fallback=[(fallback, "claude-haiku-5")])
    with wrapper:
        response = wrapper.beta.messages.create(
            model="claude-sonnet-5",
            max_tokens=8,
            messages=[{"role": "user", "content": _PROMPT}],
            betas=[_ON_DEMAND_BETA],
            compaction={"type": "summarize"},
        )
    assert type(response).__name__ == "BetaMessage"
    assert len(primary_seen) == 1
    assert len(fallback_seen) == 1
    request = fallback_seen[0]
    assert request.url.path == "/v1/messages"
    assert request.headers["anthropic-beta"] == _ON_DEMAND_BETA
    body = json.loads(request.content)
    assert body["model"] == "claude-haiku-5"
    assert body["compaction"] == {"type": "summarize"}
    served = [event for event in plane.ingested if event.token_details is not None]
    assert served[-1].model == "claude-haiku-5"
    assert served[-1].token_details is not None
    assert served[-1].token_details.compaction_output_tokens == expected["compaction_output_tokens"]


# ---------------------------------------------------------------------------
# Platform clients share the adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda handler: anthropic.AnthropicVertex(
            region="us-east5",
            project_id="project",
            access_token="token",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        ),
        lambda handler: anthropic.AnthropicFoundry(
            api_key="key",
            resource="resource",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        ),
    ],
    ids=["vertex", "foundry"],
)
def test_platform_clients_meter_beta_messages_create(factory: Callable[[Any], Any]) -> None:
    usage, iterations, expected = _THRESHOLD
    seen: list[Any] = []
    plane = FakeControlPlane()
    with plane.wrap(factory(_handler(seen, usage, iterations)), lease_enabled=False) as wrapper:
        response = wrapper.beta.messages.create(
            model="claude-sonnet-5",
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
            betas=[_THRESHOLD_BETA],
            context_management={"edits": [{"type": "compact_20260112"}]},
        )
    assert type(response).__name__ == "BetaMessage"
    assert len(seen) == 1
    assert _settled(plane)["compaction_input_tokens"] == expected["compaction_input_tokens"]


_AWS_PROFILE = "solwyn-test"
_AWS_PROFILE_KEY_ID = "AKIDSOLWYNPROFILE"


@pytest.fixture
def aws_profile(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """A named AWS profile in isolated config files, with no ambient credentials."""
    pytest.importorskip("botocore")
    credentials = tmp_path / "credentials"
    credentials.write_text(
        f"[{_AWS_PROFILE}]\naws_access_key_id = {_AWS_PROFILE_KEY_ID}\n"
        "aws_secret_access_key = secret\n"
    )
    config = tmp_path / "config"
    config.write_text(f"[profile {_AWS_PROFILE}]\nregion = us-east-1\n")
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    return _AWS_PROFILE


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("beta", [True, False], ids=["beta-messages", "messages"])
async def test_bedrock_hop_keeps_the_callers_aws_profile_and_transport(
    aws_profile: str, is_async: bool, beta: bool
) -> None:
    # anthropic's Bedrock copy() drops aws_profile and http_client; the real
    # with_options() runs here, so the hop must still sign with the named
    # profile and reach the caller's transport.
    usage, iterations, expected = _THRESHOLD
    seen: list[Any] = []
    handler = _handler(seen, usage, iterations)
    if is_async:
        provider: Any = anthropic.AsyncAnthropicBedrock(
            aws_profile=aws_profile,
            aws_region="us-east-1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
    else:
        provider = anthropic.AnthropicBedrock(
            aws_profile=aws_profile,
            aws_region="us-east-1",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )
    plane = FakeControlPlane()
    wrap = plane.wrap_async if is_async else plane.wrap
    wrapper: Any = wrap(provider, lease_enabled=False)
    kwargs: dict[str, Any] = {
        "model": "claude-sonnet-5",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if beta:
        response = wrapper.beta.messages.create(betas=[_THRESHOLD_BETA], **kwargs)
    else:
        response = wrapper.messages.create(**kwargs)
    if is_async:
        response = await response
    await _close(wrapper, is_async)
    assert type(response).__name__ == ("BetaMessage" if beta else "Message")
    assert len(seen) == 1
    assert f"Credential={_AWS_PROFILE_KEY_ID}/" in seen[0].headers["authorization"]
    if beta:
        assert json.loads(seen[0].content)["anthropic_beta"] == [_THRESHOLD_BETA]
        assert _settled(plane)["compaction_output_tokens"] == expected["compaction_output_tokens"]
    else:
        assert len(plane.ingested) == 1


def test_bedrock_mantle_hop_keeps_the_callers_transport() -> None:
    usage, iterations, expected = _THRESHOLD
    seen: list[Any] = []
    transport = httpx2.MockTransport(_handler(seen, usage, iterations))
    provider = anthropic.AnthropicBedrockMantle(
        aws_region="us-east-1",
        api_key="key",
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
    )
    plane = FakeControlPlane()
    with plane.wrap(provider, lease_enabled=False) as wrapper:
        wrapper.beta.messages.create(
            model="claude-sonnet-5",
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
            betas=[_THRESHOLD_BETA],
        )
    assert len(seen) == 1
    assert _settled(plane)["compaction_input_tokens"] == expected["compaction_input_tokens"]
