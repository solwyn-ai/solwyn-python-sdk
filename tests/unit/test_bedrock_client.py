"""Bedrock client integration: converse/converse_stream interception.

Covers the Bedrock-specific dispatch seams:
- ``solwyn.converse(modelId=...)`` routes through ``_intercepted_call`` with
  the uniform internal ``model`` key (renamed back to ``modelId`` at dispatch).
- ``converse_stream`` returns the boto3 contract shape — a dict whose
  ``"stream"`` value is the (wrapped) event stream — and settles usage from
  the terminal metadata event.
- ``invoke_model`` fails LOUDLY instead of silently bypassing budget tracking.
- Cross-provider failover both directions (bedrock -> anthropic,
  openai -> bedrock) translates requests and reshapes responses.
- ``provider_region`` rides success metadata + confirms for per-region pricing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock as AsyncMockFn
from unittest.mock import MagicMock, patch

import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY

from solwyn._surfaces import SurfaceSource
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ConfigurationError

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _converse_response(*, input_tokens: int = 12, output_tokens: int = 7) -> dict[str, Any]:
    return {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
        "metrics": {"latencyMs": 50},
    }


def _converse_stream_events() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
                "metrics": {"latencyMs": 80},
            }
        },
    ]


def _mock_bedrock_client(region: str = "us-east-1") -> MagicMock:
    """Mock that duck-types as a boto3 bedrock-runtime client."""
    client = MagicMock()
    client.__class__.__module__ = "botocore.client"
    client.__class__.__name__ = "BedrockRuntime"
    client.meta.service_model.service_name = "bedrock-runtime"
    client.meta.region_name = region
    # Real boto3 clients have no with_options; the MagicMock auto-attr must at
    # least be transparent so the configured converse mock is the one invoked.
    client.with_options.return_value = client
    client.converse.return_value = _converse_response()
    client.converse_stream.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "stream": iter(_converse_stream_events()),
    }
    return client


def _mock_anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.with_options.return_value = client
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello")],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0),
    )
    return client


def _mock_openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.with_options.return_value = client
    return client


class _Status(Exception):
    """Duck-typed transport error with a numeric status (429 -> FAILOVER)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status={status_code}")
        self.status_code = status_code


class _ClientError(Exception):
    """Mirrors botocore.exceptions.ClientError: status lives in a response DICT."""

    def __init__(self, error_response: dict[str, Any], operation_name: str = "Converse") -> None:
        super().__init__("An error occurred")
        self.response = error_response
        self.operation_name = operation_name


def _botocore_client_error(code: str, status: int) -> Exception:
    """Build a fake Bedrock service exception, botocore-shaped.

    botocore generates one ClientError subclass per modeled error shape with
    the class NAME equal to the error code, and buries the HTTP status in
    ``response["ResponseMetadata"]["HTTPStatusCode"]`` — no ``status_code``
    attribute anywhere. Driving these through the client exercises
    ``_numeric_status``'s dict path end-to-end.
    """
    response: dict[str, Any] = {
        "Error": {"Code": code, "Message": "boom"},
        "ResponseMetadata": {"HTTPStatusCode": status, "HTTPHeaders": {}},
    }
    cls = type(code, (_ClientError,), {})
    return cls(response)


def _make_solwyn(client: Any, **overrides: Any) -> Solwyn:
    defaults: dict[str, Any] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    return Solwyn(client, **defaults)


def _mock_budget(solwyn: Any, response: dict[str, Any] | None = None) -> Any:
    resp = MagicMock()
    resp.json.return_value = response or ALLOW_BUDGET_RESPONSE
    resp.raise_for_status = MagicMock()
    return patch.object(solwyn._budget._http, "post", return_value=resp)


# ---------------------------------------------------------------------------
# Non-streaming interception
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBedrockConverseInterception:
    def test_converse_is_intercepted_and_returns_raw_response(self) -> None:
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Settlement rides report_settlement; forward its SUCCESS event into the
        # same list so the metadata assertions observe it.
        solwyn._reporter.report_settlement = lambda _c, e: reported.append(e)

        with _mock_budget(solwyn):
            result = solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
                inferenceConfig={"maxTokens": 256},
            )

        client.converse.assert_called_once()
        # Dispatch renames the internal model key BACK to boto3's modelId.
        called_kwargs = client.converse.call_args.kwargs
        assert called_kwargs["modelId"] == BEDROCK_MODEL
        assert "model" not in called_kwargs
        assert "stream" not in called_kwargs
        # Drop-in contract: the raw Converse dict comes back untouched.
        assert result is client.converse.return_value

        assert len(reported) == 1
        event = reported[0]
        assert event.provider == "bedrock"
        assert event.model == BEDROCK_MODEL
        assert event.input_tokens == 12
        assert event.output_tokens == 7
        assert event.provider_region == "us-east-1"
        solwyn.close()

    def test_converse_confirm_carries_provider_region(self) -> None:
        client = _mock_bedrock_client(region="eu-west-1")
        solwyn = _make_solwyn(client)
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        confirm, _event = settlements[0]
        assert confirm.provider_region == "eu-west-1"
        solwyn.close()

    def test_converse_confirm_and_event_carry_same_service_tier(self) -> None:
        # The tier echoed on the response settles the enforcement counter at
        # the tier-repriced rate. Confirm and metadata for one call_id MUST
        # carry the SAME tier or the counter and durable cost diverge.
        client = _mock_bedrock_client()
        response = _converse_response()
        response["serviceTier"] = {"type": "priority"}
        client.converse.return_value = response
        solwyn = _make_solwyn(client)
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        # Confirm + metadata event settle together; both carry the same tier.
        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert confirm.service_tier == "priority"
        assert event.service_tier == "priority"
        solwyn.close()

    def test_converse_requires_model_id(self) -> None:
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        with pytest.raises(TypeError, match="modelId"):
            solwyn.converse(messages=[{"role": "user", "content": [{"text": "Hi"}]}])
        client.converse.assert_not_called()
        solwyn.close()

    def test_all_explicit_bedrock_methods_resolve_policy_before_dispatch(self) -> None:
        solwyn = _make_solwyn(_mock_bedrock_client())

        with (
            patch.object(
                solwyn,
                "_enforce_explicit_surface",
                wraps=solwyn._enforce_explicit_surface,
            ) as enforce,
            patch.object(solwyn, "_intercepted_call", return_value=object()),
        ):
            solwyn.converse(modelId=BEDROCK_MODEL)
            enforce.assert_called_once_with("converse", source=SurfaceSource.WRAPPER)
            enforce.reset_mock()

            solwyn.converse_stream(modelId=BEDROCK_MODEL)
            enforce.assert_called_once_with("converse_stream", source=SurfaceSource.WRAPPER)
            enforce.reset_mock()

            blocked = (
                ("invoke_model", {"modelId": BEDROCK_MODEL, "body": b"{}"}),
                (
                    "invoke_model_with_response_stream",
                    {"modelId": BEDROCK_MODEL, "body": b"{}"},
                ),
                ("start_async_invoke", {"modelId": BEDROCK_MODEL, "modelInput": {}}),
            )
            for path, kwargs in blocked:
                with pytest.raises(ConfigurationError):
                    getattr(solwyn, path)(**kwargs)
                assert enforce.call_args.args[0] == path
                assert enforce.call_args.kwargs["source"] is SurfaceSource.WRAPPER
                enforce.reset_mock()

        solwyn.close()

    def test_invoke_model_raises_loudly(self) -> None:
        # Silent pass-through would be a budget bypass on Bedrock's primary
        # legacy completion surface — fail loud and point at Converse.
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        with pytest.raises(ConfigurationError, match="converse"):
            solwyn.invoke_model(modelId=BEDROCK_MODEL, body=b"{}")
        client.invoke_model.assert_not_called()
        solwyn.close()

    def test_invoke_model_with_response_stream_raises_loudly(self) -> None:
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        with pytest.raises(ConfigurationError, match="converse"):
            solwyn.invoke_model_with_response_stream(modelId=BEDROCK_MODEL, body=b"{}")
        solwyn.close()

    def test_start_async_invoke_raises_loudly(self) -> None:
        # start_async_invoke joins the invoke_model guard: it is Bedrock's
        # async, video-scale spend surface whose usage lands out-of-band in S3,
        # so silent pass-through would be a budget bypass at the highest cost.
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        with pytest.raises(ConfigurationError, match="start_async_invoke"):
            solwyn.start_async_invoke(modelId=BEDROCK_MODEL, modelInput={})
        client.start_async_invoke.assert_not_called()
        solwyn.close()


# ---------------------------------------------------------------------------
# Streaming interception
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBedrockConverseStream:
    def test_converse_stream_preserves_boto3_response_shape(self) -> None:
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        settlements: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            result = solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        client.converse_stream.assert_called_once()
        called_kwargs = client.converse_stream.call_args.kwargs
        assert called_kwargs["modelId"] == BEDROCK_MODEL
        assert "stream" not in called_kwargs

        # boto3 contract: a dict whose "stream" yields the events unchanged.
        assert result["ResponseMetadata"] == {"HTTPStatusCode": 200}
        events = list(result["stream"])
        assert events == _converse_stream_events()

        # Exhaustion settles usage from the terminal metadata event.
        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert event.provider == "bedrock"
        assert event.input_tokens == 9
        assert event.output_tokens == 4
        assert event.provider_region == "us-east-1"
        assert confirm.provider_region == "us-east-1"
        assert confirm.token_details.input_tokens == 9
        solwyn.close()

    def test_converse_stream_confirm_and_event_carry_same_service_tier(self) -> None:
        # Streaming tier arrives on the terminal metadata event; the stream
        # settlement's confirm and metadata event must carry the same value.
        events = _converse_stream_events()
        events[-1]["metadata"]["serviceTier"] = {"type": "flex"}
        client = _mock_bedrock_client()
        client.converse_stream.return_value = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "stream": iter(events),
        }
        solwyn = _make_solwyn(client)
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            result = solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )
            list(result["stream"])

        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert confirm.service_tier == "flex"
        assert event.service_tier == "flex"
        solwyn.close()

    def test_converse_stream_early_abandon_close_settles_exactly_once(self) -> None:
        # Finding #6: abandoning result["stream"] early strands the budget
        # reservation unless the caller closes the wrapper — the same
        # one-level-deep close obligation raw boto3's EventStream imposes.
        # close() must settle EXACTLY once (the _settled guard) and stay
        # idempotent on repeat calls. Abandoned before the terminal metadata
        # event, the settlement carries the zeros the accumulator observed
        # (the explicit zero-settle warning is pinned in
        # test_providers/test_bedrock.py).
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            result = solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )
            consumed = []
            for event in result["stream"]:
                consumed.append(event)
                break  # abandon after the FIRST event
            result["stream"].close()

        assert consumed == [{"messageStart": {"role": "assistant"}}]
        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert confirm.reservation_id == "res_123"
        assert event.provider == "bedrock"
        # No terminal metadata event was consumed — settles at zero usage.
        assert event.input_tokens == 0
        assert event.output_tokens == 0
        assert confirm.token_details.input_tokens == 0
        assert confirm.token_details.output_tokens == 0

        # close() is idempotent: a second call must NOT double-settle.
        result["stream"].close()
        assert len(settlements) == 1
        solwyn.close()

    def test_converse_stream_early_abandon_context_manager_settles_exactly_once(self) -> None:
        # The documented `with result["stream"]:` form — __exit__ closes and
        # settles exactly once after an early break.
        client = _mock_bedrock_client()
        solwyn = _make_solwyn(client)
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            result = solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )
            with result["stream"]:
                for _event in result["stream"]:
                    break  # abandon after the FIRST event

        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert confirm.reservation_id == "res_123"
        assert event.input_tokens == 0
        assert event.output_tokens == 0
        solwyn.close()


# ---------------------------------------------------------------------------
# Cross-provider failover (both directions)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBedrockFailover:
    def test_bedrock_primary_fails_over_to_anthropic(self) -> None:
        bedrock = _mock_bedrock_client()
        # A real botocore ThrottlingException (status in the response dict,
        # not .status_code) must classify FAILOVER through the live dispatch.
        bedrock.converse.side_effect = _botocore_client_error("ThrottlingException", 429)
        anthropic = _mock_anthropic_client()
        solwyn = _make_solwyn(bedrock, fallback=[(anthropic, "claude-sonnet-5")])
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        solwyn._reporter.report_settlement = lambda _c, e: reported.append(e)

        with _mock_budget(solwyn):
            result = solwyn.converse(
                modelId=BEDROCK_MODEL,
                system=[{"text": "be brief"}],
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
                inferenceConfig={"maxTokens": 256, "temperature": 0.2},
            )

        # The served Anthropic hop got a TRANSLATED native request.
        anthropic_kwargs = anthropic.messages.create.call_args.kwargs
        assert anthropic_kwargs["model"] == "claude-sonnet-5"
        assert anthropic_kwargs["max_tokens"] == 256
        assert anthropic_kwargs["temperature"] == 0.2
        assert anthropic_kwargs["system"] == "be brief"

        # The response is reshaped back INTO the Bedrock dialect (a dict).
        assert result["output"]["message"]["content"][0]["text"] == "Hello"
        assert result["stopReason"] == "end_turn"

        success = [e for e in reported if e.status == "success"]
        assert len(success) == 1
        assert success[0].provider == "anthropic"
        assert success[0].is_provider_fallback is True
        assert success[0].requested_provider == "bedrock"
        assert success[0].requested_model == BEDROCK_MODEL
        # Anthropic has no regional pricing — region must NOT leak across hops.
        assert success[0].provider_region is None
        solwyn.close()

    def test_openai_primary_fails_over_to_bedrock(self) -> None:
        openai = _mock_openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        bedrock = _mock_bedrock_client(region="eu-central-1")
        solwyn = _make_solwyn(openai, fallback=[(bedrock, BEDROCK_MODEL)])
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        solwyn._reporter.report_settlement = lambda _c, e: reported.append(e)

        with _mock_budget(solwyn):
            result = solwyn.chat.completions.create(
                model="gpt-5.5",
                max_tokens=64,
                messages=[{"role": "user", "content": "Hello"}],
            )

        # The served Bedrock hop got a TRANSLATED Converse request.
        converse_kwargs = bedrock.converse.call_args.kwargs
        assert converse_kwargs["modelId"] == BEDROCK_MODEL
        assert "model" not in converse_kwargs
        assert converse_kwargs["inferenceConfig"]["maxTokens"] == 64
        assert converse_kwargs["messages"] == [{"role": "user", "content": [{"text": "Hello"}]}]

        # The Converse dict response is reshaped back to the OpenAI dialect.
        assert result.choices[0].message.content == "ok"

        success = [e for e in reported if e.status == "success"]
        assert len(success) == 1
        assert success[0].provider == "bedrock"
        assert success[0].model == BEDROCK_MODEL
        assert success[0].is_provider_fallback is True
        assert success[0].provider_region == "eu-central-1"
        solwyn.close()

    def test_openai_stream_fails_over_to_bedrock_stream(self) -> None:
        # An OpenAI-dialect caller streaming via stream=True must, when served
        # by Bedrock, get back a DIRECTLY-ITERABLE wrapper yielding
        # OpenAI-shaped chunks — not the boto3 {"stream": ...} dict.
        openai = _mock_openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        bedrock = _mock_bedrock_client()
        solwyn = _make_solwyn(openai, fallback=[(bedrock, BEDROCK_MODEL)])
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            stream = solwyn.chat.completions.create(
                model="gpt-5.5",
                max_tokens=64,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            texts = [
                chunk.choices[0].delta.content
                for chunk in stream
                if chunk.choices[0].delta.content is not None
            ]

        bedrock.converse_stream.assert_called_once()
        assert "".join(texts) == "Hello"
        # Usage still settles against the SERVED provider's raw events.
        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert event.provider == "bedrock"
        assert event.input_tokens == 9
        assert event.provider_region == "us-east-1"

        solwyn.close()

    def test_bedrock_stream_fails_over_to_anthropic_stream(self) -> None:
        # A Bedrock-dialect caller using converse_stream must, when served by
        # Anthropic, still get the boto3 contract shape: a dict whose "stream"
        # yields Bedrock-shaped event dicts.
        bedrock = _mock_bedrock_client()
        bedrock.converse_stream.side_effect = _botocore_client_error("ThrottlingException", 429)
        anthropic = _mock_anthropic_client()
        anthropic.messages.create.return_value = iter(
            [
                SimpleNamespace(
                    type="message_start",
                    message=SimpleNamespace(
                        usage=SimpleNamespace(input_tokens=11, cache_read_input_tokens=0)
                    ),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Hi"),
                ),
                SimpleNamespace(
                    type="message_delta",
                    delta=SimpleNamespace(stop_reason="end_turn"),
                    usage=SimpleNamespace(output_tokens=3),
                ),
            ]
        )
        solwyn = _make_solwyn(bedrock, fallback=[(anthropic, "claude-sonnet-5")])
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            result = solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
                inferenceConfig={"maxTokens": 256},
            )
            events = list(result["stream"])

        # Served Anthropic chunks arrive reshaped as Bedrock event dicts.
        assert {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}} in events
        assert {"messageStop": {"stopReason": "end_turn"}} in events
        # Usage settles from the RAW Anthropic events (served provider).
        assert len(settlements) == 1
        _, event = settlements[0]
        assert event.provider == "anthropic"
        assert event.input_tokens == 11
        assert event.output_tokens == 3
        assert event.provider_region is None

        solwyn.close()

    def test_model_timeout_does_not_fail_over_and_reconciles(self) -> None:
        # ModelTimeoutException is HTTP 408 — by status it looks request-shaped,
        # but the model RAN. End-to-end through _intercepted_call it must
        # re-raise the ORIGINAL exception (never failover, even with a fallback
        # configured) and emit possibly_succeeded=True so the Cloud API
        # reconciles the possibly-landed charge.
        bedrock = _mock_bedrock_client()
        timeout_exc = _botocore_client_error("ModelTimeoutException", 408)
        bedrock.converse.side_effect = timeout_exc
        anthropic = _mock_anthropic_client()
        solwyn = _make_solwyn(bedrock, fallback=[(anthropic, "claude-sonnet-5")])
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)

        with _mock_budget(solwyn), pytest.raises(_ClientError) as excinfo:
            solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        assert excinfo.value is timeout_exc  # the ORIGINAL exception, re-raised
        anthropic.messages.create.assert_not_called()  # no failover
        errors = [e for e in reported if e.status == "error"]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].failover_error_class == "ModelTimeoutException"
        solwyn.close()


# ---------------------------------------------------------------------------
# Error-path provider_region attribution
# ---------------------------------------------------------------------------


def _failing_stream_events() -> Any:
    yield {"messageStart": {"role": "assistant"}}
    raise _Status(500)


@pytest.mark.unit
class TestBedrockErrorRegionAttribution:
    """Error events carry provider_region — the reconciliation-critical path.

    A POST_SEND_AMBIGUOUS abort (possibly_succeeded=True) is exactly where the
    Cloud API needs the region to reconcile a possibly-landed Bedrock charge
    against its per-(model, region) price.
    """

    def test_dispatch_error_event_carries_provider_region(self) -> None:
        client = _mock_bedrock_client(region="ap-southeast-2")
        client.converse.side_effect = _Status(500)  # POST_SEND_AMBIGUOUS
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)

        with _mock_budget(solwyn), pytest.raises(_Status):
            solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        errors = [e for e in reported if e.status == "error"]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].provider_region == "ap-southeast-2"
        solwyn.close()

    def test_budget_denied_event_carries_provider_region(self) -> None:
        # Denied-Bedrock spend must stay analyzable per region: the primary's
        # endpoint region rides the BUDGET_DENIED event too.
        from solwyn._types import BudgetMode
        from solwyn.exceptions import BudgetExceededError

        client = _mock_bedrock_client(region="eu-central-1")
        solwyn = _make_solwyn(client, budget_mode=BudgetMode.HARD_DENY)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        deny = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "denied_by_period": "daily",
            "mode": "hard_deny",
        }

        with _mock_budget(solwyn, response=deny), pytest.raises(BudgetExceededError):
            solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        client.converse.assert_not_called()
        denied = [e for e in reported if e.status == "budget_denied"]
        assert len(denied) == 1
        assert denied[0].provider_region == "eu-central-1"
        solwyn.close()

    def test_mid_stream_error_event_carries_provider_region(self) -> None:
        client = _mock_bedrock_client(region="eu-west-1")
        client.converse_stream.return_value = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "stream": _failing_stream_events(),
        }
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)

        with _mock_budget(solwyn):
            result = solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )
            with pytest.raises(_Status):
                list(result["stream"])

        errors = [e for e in reported if e.status == "error"]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].provider_region == "eu-west-1"
        solwyn.close()


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class _AsyncEventStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            for event in self._events:
                yield event

        return _gen()


def _make_async_solwyn(client: Any, **overrides: Any) -> AsyncSolwyn:
    defaults: dict[str, Any] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    return AsyncSolwyn(client, **defaults)


@pytest.mark.unit
class TestAsyncBedrockConverse:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_converse_is_intercepted(self) -> None:
        client = _mock_bedrock_client()
        client.converse = AsyncMockFn(return_value=_converse_response())
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        solwyn._reporter.report_settlement = lambda _c, e: reported.append(e)

        with _mock_budget(solwyn):
            result = await solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        client.converse.assert_called_once()
        assert client.converse.call_args.kwargs["modelId"] == BEDROCK_MODEL
        assert result["output"]["message"]["content"][0]["text"] == "ok"
        assert len(reported) == 1
        assert reported[0].provider == "bedrock"
        assert reported[0].provider_region == "us-east-1"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_async_explicit_bedrock_methods_resolve_policy_before_dispatch(
        self,
    ) -> None:
        solwyn = _make_async_solwyn(_mock_bedrock_client())

        with (
            patch.object(
                solwyn,
                "_enforce_explicit_surface",
                wraps=solwyn._enforce_explicit_surface,
            ) as enforce,
            patch.object(solwyn, "_intercepted_call", new=AsyncMockFn(return_value=object())),
        ):
            await solwyn.converse(modelId=BEDROCK_MODEL)
            enforce.assert_called_once_with("converse", source=SurfaceSource.WRAPPER)
            enforce.reset_mock()

            await solwyn.converse_stream(modelId=BEDROCK_MODEL)
            enforce.assert_called_once_with("converse_stream", source=SurfaceSource.WRAPPER)
            enforce.reset_mock()

            blocked = (
                ("invoke_model", {"modelId": BEDROCK_MODEL, "body": b"{}"}),
                (
                    "invoke_model_with_response_stream",
                    {"modelId": BEDROCK_MODEL, "body": b"{}"},
                ),
                ("start_async_invoke", {"modelId": BEDROCK_MODEL, "modelInput": {}}),
            )
            for path, kwargs in blocked:
                with pytest.raises(ConfigurationError):
                    await getattr(solwyn, path)(**kwargs)
                assert enforce.call_args.args[0] == path
                assert enforce.call_args.kwargs["source"] is SurfaceSource.WRAPPER
                enforce.reset_mock()

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_converse_stream_wraps_inner_event_stream(self) -> None:
        client = _mock_bedrock_client()
        client.converse_stream = AsyncMockFn(
            return_value={
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "stream": _AsyncEventStream(_converse_stream_events()),
            }
        )
        solwyn = _make_async_solwyn(client)
        reported: list = []
        settlements: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            result = await solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        events = [event async for event in result["stream"]]
        assert events == _converse_stream_events()
        assert len(settlements) == 1
        _, event = settlements[0]
        assert event.input_tokens == 9
        assert event.output_tokens == 4

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_converse_stream_early_abandon_close_settles_exactly_once(self) -> None:
        # Async mirror of finding #6: abandoning result["stream"] early must
        # settle the reservation via `await result["stream"].close()` — exactly
        # once (the _settled guard), idempotent on repeat calls, at the zero
        # usage observed before the terminal metadata event.
        client = _mock_bedrock_client()
        client.converse_stream = AsyncMockFn(
            return_value={
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "stream": _AsyncEventStream(_converse_stream_events()),
            }
        )
        solwyn = _make_async_solwyn(client)
        settlements: list = []
        solwyn._reporter.report = lambda e: None
        solwyn._reporter.report_settlement = lambda c, e: settlements.append((c, e))

        with _mock_budget(solwyn):
            result = await solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )
            consumed = []
            async for event in result["stream"]:
                consumed.append(event)
                break  # abandon after the FIRST event
            await result["stream"].close()

        assert consumed == [{"messageStart": {"role": "assistant"}}]
        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert confirm.reservation_id == "res_123"
        assert event.provider == "bedrock"
        # No terminal metadata event was consumed — settles at zero usage.
        assert event.input_tokens == 0
        assert event.output_tokens == 0
        assert confirm.token_details.input_tokens == 0

        # close() is idempotent: a second call must NOT double-settle.
        await result["stream"].close()
        assert len(settlements) == 1

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_invoke_model_raises_loudly(self) -> None:
        # Mirror of the sync fail-loud test: a regression that let async
        # invoke_model pass through silently would be a budget bypass.
        client = _mock_bedrock_client()
        solwyn = _make_async_solwyn(client)

        with pytest.raises(ConfigurationError, match="converse"):
            await solwyn.invoke_model(modelId=BEDROCK_MODEL, body=b"{}")

        client.invoke_model.assert_not_called()
        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_invoke_model_with_response_stream_raises_loudly(self) -> None:
        client = _mock_bedrock_client()
        solwyn = _make_async_solwyn(client)

        with pytest.raises(ConfigurationError, match="converse"):
            await solwyn.invoke_model_with_response_stream(modelId=BEDROCK_MODEL, body=b"{}")

        client.invoke_model_with_response_stream.assert_not_called()
        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_start_async_invoke_raises_loudly(self) -> None:
        # Mirror of the sync fail-loud test: async start_async_invoke passing
        # through silently would be a video-scale budget bypass.
        client = _mock_bedrock_client()
        solwyn = _make_async_solwyn(client)

        with pytest.raises(ConfigurationError, match="start_async_invoke"):
            await solwyn.start_async_invoke(modelId=BEDROCK_MODEL, modelInput={})

        client.start_async_invoke.assert_not_called()
        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_dispatch_error_event_carries_provider_region(self) -> None:
        client = _mock_bedrock_client(region="ap-southeast-2")
        client.converse = AsyncMockFn(side_effect=_Status(500))  # POST_SEND_AMBIGUOUS
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)

        with _mock_budget(solwyn), pytest.raises(_Status):
            await solwyn.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )

        errors = [e for e in reported if e.status == "error"]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].provider_region == "ap-southeast-2"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_mid_stream_error_event_carries_provider_region(self) -> None:
        class _FailingAsyncEventStream:
            def __aiter__(self) -> Any:
                async def _gen() -> Any:
                    yield {"messageStart": {"role": "assistant"}}
                    raise _Status(500)

                return _gen()

        client = _mock_bedrock_client(region="eu-west-1")
        client.converse_stream = AsyncMockFn(
            return_value={
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "stream": _FailingAsyncEventStream(),
            }
        )
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)

        with _mock_budget(solwyn):
            result = await solwyn.converse_stream(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            )
            with pytest.raises(_Status):
                _ = [event async for event in result["stream"]]

        errors = [e for e in reported if e.status == "error"]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].provider_region == "eu-west-1"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ---------------------------------------------------------------------------
# Unbounded botocore read timeout (PJ-8/R8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnboundedReadTimeoutWarning:
    """PJ-8/R8: Solwyn cannot bound boto3 hops per-call, so an explicitly
    unbounded botocore read (Config(read_timeout=None)) must warn at build."""

    def test_read_timeout_none_warns_at_build(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _mock_bedrock_client()
        client.meta.config = SimpleNamespace(read_timeout=None)
        caplog.set_level("WARNING", logger="solwyn._base")
        solwyn = _make_solwyn(client, model=BEDROCK_MODEL)
        messages = [r.getMessage() for r in caplog.records if r.name == "solwyn._base"]
        assert any("read_timeout=None" in m for m in messages)
        assert any(BEDROCK_MODEL in m for m in messages)
        solwyn.close()

    def test_finite_read_timeout_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _mock_bedrock_client()
        client.meta.config = SimpleNamespace(read_timeout=60)
        caplog.set_level("WARNING", logger="solwyn._base")
        solwyn = _make_solwyn(client, model=BEDROCK_MODEL)
        assert not any(
            "read_timeout" in r.getMessage() for r in caplog.records if r.name == "solwyn._base"
        )
        solwyn.close()

    def test_missing_config_attribute_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # A client without the meta.config path (exotic wrapper, test double
        # without the attribute) must read as bounded - never a false alarm.
        client = _mock_bedrock_client()
        client.meta.config = None
        caplog.set_level("WARNING", logger="solwyn._base")
        solwyn = _make_solwyn(client, model=BEDROCK_MODEL)
        assert not any(
            "read_timeout" in r.getMessage() for r in caplog.records if r.name == "solwyn._base"
        )
        solwyn.close()

    def test_unbounded_bedrock_fallback_also_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        # The check covers EVERY runtime in the chain, not just the primary.
        primary = _mock_openai_client()
        bedrock = _mock_bedrock_client()
        bedrock.meta.config = SimpleNamespace(read_timeout=None)
        caplog.set_level("WARNING", logger="solwyn._base")
        solwyn = _make_solwyn(
            primary,
            model="gpt-5.5",
            fallback=[(bedrock, BEDROCK_MODEL)],
        )
        assert any(
            "read_timeout=None" in r.getMessage()
            for r in caplog.records
            if r.name == "solwyn._base"
        )
        solwyn.close()
