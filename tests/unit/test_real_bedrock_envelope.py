"""Bedrock usage extraction against botocore's REAL Converse service model.

Every other Bedrock test (``test_bedrock_client.py``,
``test_providers/test_bedrock.py``) drives extraction through synthetic
``MagicMock`` clients fed hand-written envelope dicts — convenient, but blind
to drift between the SDK's assumptions and AWS's actual Converse response
shape. This module instead uses a GENUINE ``boto3`` ``bedrock-runtime`` client
driven by ``botocore.stub.Stubber``:

- ``Stubber.add_response("converse", envelope, expected_params)`` validates the
  canned envelope against botocore's generated Converse OUTPUT shape. Any usage
  field the SDK reads that AWS's local service model does not define fails at
  ``add_response`` time — surfacing SDK-vs-AWS drift instead of hiding it behind
  a hand-rolled dict. (Confirmed: passing ``usage.bogusField`` raises
  ``ParamValidationError``.)
- The dict the stubbed client returns is the model-validated envelope, so key
  casing / nesting / types are AWS-truthful.

No AWS credentials, no network — fake creds construct the client fully offline.
boto3/botocore are dev-only deps; the module skips cleanly when absent.

Findings against botocore 1.43.51: NONE. The Converse output model defines
every usage field the adapter reads — ``cacheReadInputTokens``,
``cacheWriteInputTokens``, ``cacheDetails`` (``[{inputTokens, ttl}]`` with
``ttl`` enum ``{"5m", "1h"}``), plus top-level ``serviceTier`` (``type`` enum
``{priority, default, flex, reserved}``) and ``performanceConfig``
(``latency`` enum ``{standard, optimized}``). So the full, cache-bearing
envelope below is entirely model-validated — no hand-dict split was needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY

from solwyn.client import Solwyn
from solwyn.providers import get_adapter_for_client
from solwyn.providers.bedrock import BedrockAdapter

# Constructed via the return value (not an ``import`` statement) so the module
# skips — rather than errors — when boto3/botocore are absent, mirroring
# tests/unit/test_real_sdk_detection.py's convention.
boto3 = pytest.importorskip("boto3")
Stubber = pytest.importorskip("botocore.stub").Stubber

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _bedrock_client(region: str = "us-east-1") -> Any:
    """A genuine bedrock-runtime client — fake creds, no network, no AWS."""
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


def _messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"text": "Hello"}]}]


def _full_envelope() -> dict[str, Any]:
    """A cache-bearing Converse envelope — EVERY field model-validated.

    ``cacheWriteInputTokens=20`` with ``cacheDetails=[{inputTokens:15, ttl:1h}]``
    exercises the TTL split: 15 -> 1h bucket, remainder 5 -> 5m bucket.
    """
    return {
        "ResponseMetadata": {"HTTPStatusCode": 200, "RequestId": "req-abc123"},
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 40,
            "totalTokens": 140,
            "cacheReadInputTokens": 30,
            "cacheWriteInputTokens": 20,
            "cacheDetails": [{"inputTokens": 15, "ttl": "1h"}],
        },
        "metrics": {"latencyMs": 123},
        "serviceTier": {"type": "priority"},
    }


def _mock_budget(solwyn: Any, response: dict[str, Any] | None = None) -> Any:
    """Patch the budget enforcer's HTTP seam with an ALLOW response.

    Mirrors ``tests/unit/test_bedrock_client.py::_mock_budget`` exactly.
    """
    resp = MagicMock()
    resp.json.return_value = response or ALLOW_BUDGET_RESPONSE
    resp.raise_for_status = MagicMock()
    return patch.object(solwyn._solwyn_budget._http, "post", return_value=resp)


# ---------------------------------------------------------------------------
# 1. Detection sanity through real botocore plumbing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_real_bedrock_runtime_client_resolves_bedrock_adapter() -> None:
    # Duck-typed detection must claim a GENUINE botocore client, not just the
    # synthetic module/service_name doubles the other suites use.
    client = _bedrock_client()
    assert get_adapter_for_client(client).name == "bedrock"


# ---------------------------------------------------------------------------
# 2. Adapter extraction from a model-validated Stubber envelope
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adapter_extracts_usage_from_model_validated_envelope() -> None:
    # The envelope round-trips through botocore's Converse output model
    # (add_response would reject any field AWS doesn't define), so these
    # assertions pin the AWS-additive formula against AWS-truthful plumbing.
    client = _bedrock_client()
    stubber = Stubber(client)
    params = {"modelId": BEDROCK_MODEL, "messages": _messages()}
    stubber.add_response("converse", _full_envelope(), params)

    with stubber:
        response = client.converse(**params)
    stubber.assert_no_pending_responses()

    details = BedrockAdapter().extract_usage(response)
    # input = inputTokens + cacheRead + cacheWrite (AWS-additive convention).
    assert details.input_tokens == 100 + 30 + 20
    assert details.output_tokens == 40
    assert details.cached_input_tokens == 30
    # cacheWrite 20 split by cacheDetails: 15 -> 1h, remainder 5 -> 5m.
    assert details.cache_creation_5m_tokens == 5
    assert details.cache_creation_1h_tokens == 15

    # serviceTier is a modeled top-level output field: type wins as the tier.
    assert BedrockAdapter().extract_service_tier(response) == "priority"


@pytest.mark.unit
def test_adapter_service_tier_falls_back_to_performance_config() -> None:
    # performanceConfig.latency is the modeled fallback when serviceTier is
    # absent — the RESPONSE echo is Bedrock's billing ground truth.
    client = _bedrock_client()
    stubber = Stubber(client)
    envelope = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        # ``metrics`` is a REQUIRED member of the Converse output shape — omit it
        # and Stubber.add_response rejects the envelope (proof the validation is
        # real, not cosmetic).
        "metrics": {"latencyMs": 42},
        "performanceConfig": {"latency": "optimized"},
    }
    params = {"modelId": BEDROCK_MODEL, "messages": _messages()}
    stubber.add_response("converse", envelope, params)

    with stubber:
        response = client.converse(**params)
    stubber.assert_no_pending_responses()

    assert BedrockAdapter().extract_service_tier(response) == "optimized"


# ---------------------------------------------------------------------------
# 3. Wrapped end-to-end converse through the real stubbed client
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wrapped_converse_reports_normalized_usage_and_region() -> None:
    # Wrap the REAL stubbed client and drive one converse through the full
    # interception pipeline. The proxy renames modelId->model internally and
    # BACK at dispatch, so the stubber's expected_params see modelId.
    client = _bedrock_client()
    stubber = Stubber(client)
    params = {"modelId": BEDROCK_MODEL, "messages": _messages()}
    stubber.add_response("converse", _full_envelope(), params)

    solwyn = Solwyn(client, api_key=VALID_API_KEY)
    settlements: list = []
    solwyn._solwyn_reporter.report = lambda e: None
    solwyn._solwyn_reporter.report_settlement = lambda c, e: settlements.append((c, e))

    with _mock_budget(solwyn), stubber:
        result = solwyn.converse(modelId=BEDROCK_MODEL, messages=_messages())
    stubber.assert_no_pending_responses()

    # Drop-in passthrough: the raw boto3 Converse contract shape survives.
    assert "output" in result
    assert "usage" in result
    assert result["output"]["message"]["content"][0]["text"] == "ok"

    # Confirm + metadata event settle together off the hot path.
    assert len(settlements) == 1
    confirm, event = settlements[0]

    # The reported metadata event carries normalized token details + region.
    assert event.provider == "bedrock"
    assert event.model == BEDROCK_MODEL
    assert event.input_tokens == 100 + 30 + 20
    assert event.output_tokens == 40
    assert event.provider_region == "us-east-1"
    assert event.service_tier == "priority"
    assert event.token_details.cached_input_tokens == 30
    assert event.token_details.cache_creation_5m_tokens == 5
    assert event.token_details.cache_creation_1h_tokens == 15

    # The confirm seam carries the SAME normalized details + region.
    assert confirm.provider_region == "us-east-1"
    assert confirm.service_tier == "priority"
    assert confirm.token_details.input_tokens == 100 + 30 + 20
    assert confirm.token_details.cache_creation_1h_tokens == 15

    solwyn.close()


# ---------------------------------------------------------------------------
# 4. converse_stream — skipped: no honest EventStream through the Stubber
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skip(
    reason=(
        "botocore.stub.Stubber has no first-class EventStream support: it returns"
        " the canned dict as-is and never constructs a genuine botocore"
        " EventStream (which requires a raw streaming HTTP body + the eventstream"
        " parser). Faking result['stream'] with a plain iterator would test our"
        " own fake, not AWS plumbing — see test_bedrock_client.py for the"
        " synthetic-stream coverage. Skipped honestly per the task brief."
    )
)
def test_wrapped_converse_stream_through_stubber() -> None:  # pragma: no cover
    raise AssertionError("unreachable — skipped")
