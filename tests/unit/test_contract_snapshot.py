"""Contract snapshot — pin the SDK<->Cloud-API wire contract.

This is the wire-contract backstop. It freezes the exact serialized
field SETS of the three wire models so ANY future drift — an added, removed, or
renamed field — fails CI loudly instead of silently breaking the Cloud-API
contract (which deploys API-first with no inference window). It also pins
a representative ``model_dump(mode="json")`` key set for each and proves the
None-skipping serializer keeps non-failover events byte-compatible with the
pre-failover wire (``possibly_succeeded`` omitted, ``call_id`` always present).

Beyond the static snapshots it carries the behavioral proofs that exercise
the live dispatch path (sync + async, streaming + non-streaming):

  * call_id is the per-call reconciliation join key: the served-provider
    success MetadataEvent and its /budgets/confirm carry the SAME call_id;
    two separate calls get DIFFERENT call_ids.
  * possibly_succeeded: a POST_SEND_AMBIGUOUS primary failure under the
    default "safe" idempotency re-raises the ORIGINAL exception AND emits an
    error event with possibly_succeeded=True; a plain FAILOVER (429) and a plain
    success emit events with possibly_succeeded None/absent.
  * is_model_fallback NARROWED: same-provider model swap -> True +
    is_provider_fallback False; cross-provider failover -> False +
    is_provider_fallback True; primary success -> both False.
  * failover_error_class FIREWALL: the field is ONLY ever populated
    from ``type(exc).__name__`` — a SENTINEL embedded in ``str(exc)`` never
    reaches the field NOR any byte of the event's model_dump.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import NoneType, SimpleNamespace
from typing import Any, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, call_uuid
from pydantic import ValidationError

from solwyn import _constants as wire_constants
from solwyn._token_details import TokenDetails
from solwyn._types import (
    SERVICE_TIER_MAX_LENGTH,
    BreakerStateReport,
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    CallStatus,
    FailoverDirective,
    LeaseGrantRequest,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    MediaUsage,
    MetadataEvent,
    ProviderName,
    UntrackedSurfaceReport,
)
from solwyn.client import AsyncSolwyn, Solwyn

# The name of THIS contract-snapshot test module, surfaced in the report.
CONTRACT_SNAPSHOT_TEST = "tests/unit/test_contract_snapshot.py"


# ── EXACT expected field sets (the pinned contract) ──────────────────────────
# Hand-written, NOT derived from the model, so a drift in
# either direction (model gains/loses a field) trips the assertion.

EXPECTED_CHECK_FIELDS = {
    "estimated_input_tokens",
    "model",
    "provider",
    "modality",
    "estimated_media",
    "fallback_providers",
    "fallback_models",
    "agent_run_id",
    "tags",
    "failover_directive_version",
}

# Optional check fields the None-skipping serializer drops when unset. Runtime
# request construction intentionally sets the directive version; only direct or
# otherwise unopted model construction omits it.
_NONE_SKIPPED_CHECK_FIELDS = {
    "estimated_media",
    "agent_run_id",
    "tags",
    "failover_directive_version",
}

EXPECTED_CHECK_RESPONSE_FIELDS = {
    "allowed",
    "remaining_budget",
    "reservation_id",
    "mode",
    "budget_limit",
    "current_usage",
    "denied_by_period",
    "project_id",
    "price_hints",
    "failover_directive",
}

EXPECTED_CONFIRM_FIELDS = {
    "reservation_id",
    "lease_id",
    "model",
    "provider",
    "modality",
    "token_details",
    "media_usage",
    "is_provider_fallback",
    "call_id",
    "provider_region",
    "service_tier",
}

# The optional confirm fields the None-skipping serializer drops when unset —
# bearer-key providers' confirm wire bytes stay byte-identical to pre-Bedrock;
# media_usage (window 2) is dropped on every chat/token confirm the same way.
# lease_id is dropped on every reservation-settled confirm (the settlement key
# is exactly-one-of, enforced by a model validator).
_NONE_SKIPPED_CONFIRM_FIELDS = {"provider_region", "service_tier", "media_usage", "lease_id"}

# Provider identifiers ARE wire values (events/confirms/checks carry them).
# Adding one is an API-first deploy: the Cloud API must accept it BEFORE any
# SDK release that can emit it.
EXPECTED_PROVIDER_NAMES = {
    "openai",
    "anthropic",
    "google",
    "bedrock",
    # OpenAI-compatible providers (Chat Completions dialect, distinct names)
    "xai",
    "deepseek",
    "mistral",
    "qwen",
    "zai",
    "groq",
    "together",
    "fireworks",
    "perplexity",
    "azure_openai",
    "openrouter",
    "ollama",
    "vllm",
    "lmstudio",
    "openai_compatible",
}

# TokenDetails is an EMBEDDED wire object (BudgetConfirmRequest.token_details,
# MetadataEvent.token_details) — its field set is contract too. is_estimated is
# omit-when-False on the wire: only estimation-fallback payloads carry it, so
# provider-reported payloads stay byte-identical and the Cloud API must accept
# the key BEFORE any SDK release that can emit it (API-first deploy).
EXPECTED_TOKEN_DETAILS_FIELDS = {
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_5m_tokens",
    "cache_creation_1h_tokens",
    "reasoning_tokens",
    "audio_input_tokens",
    "audio_output_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
    "tool_use_input_tokens",
    "image_input_tokens",
    "image_output_tokens",
    "is_estimated",
}

# MediaUsage is a NEW embedded wire object (window 2): non-token billable
# quantities + variant selectors for non-text modalities, carried on
# BudgetConfirmRequest.media_usage / MetadataEvent.media_usage /
# BudgetCheckRequest.estimated_media. Its field set is contract too — pin it
# like TokenDetails so a drift in either direction fails CI loudly.
EXPECTED_MEDIA_USAGE_FIELDS = {
    "image_count",
    "generation_count",
    "video_seconds",
    "audio_seconds",
    "input_characters",
    "resolution",
    "quality",
    "is_estimated",
}

# BreakerStateReport is a standalone API-first SDK snapshot. It is deliberately
# separate from the existing event/check/confirm pending-set choreography.
EXPECTED_BREAKER_STATE_REPORT_FIELDS = {
    "provider",
    "state",
    "failure_count",
    "success_count",
    "reported_at",
    "sdk_instance_id",
}

EXPECTED_UNTRACKED_SURFACE_REPORT_FIELDS = {
    "provider",
    "client_shape",
    "mode",
    "surface",
    "rule_kind",
    "capability_scope",
    "posture",
    "occurrences",
    "first_seen_at",
    "last_seen_at",
    "sdk_instance_id",
    "report_id",
}

EXPECTED_METADATA_FIELDS = {
    "model",
    "provider",
    "modality",
    "input_tokens",
    "output_tokens",
    "token_details",
    "media_usage",
    "latency_ms",
    "status",
    "is_model_fallback",
    "service_tier",
    "sdk_instance_id",
    "timestamp",
    "agent_run_id",
    "parent_agent_run_id",
    "agent_run_name",
    "is_provider_fallback",
    "requested_provider",
    "requested_model",
    "failover_reason",
    "failover_error_class",
    "attempt_index",
    "call_id",
    "possibly_succeeded",
    "provider_region",
    "tags",
}


# --------------------------------------------------------------------------- #
# 1. Static field-set + dump snapshots                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestWireModelFieldSets:
    """Freeze the exact ``model_fields`` set of each wire model."""

    def test_budget_check_request_field_set(self) -> None:
        assert set(BudgetCheckRequest.model_fields) == EXPECTED_CHECK_FIELDS

    def test_budget_check_response_field_set(self) -> None:
        assert set(BudgetCheckResponse.model_fields) == EXPECTED_CHECK_RESPONSE_FIELDS

    def test_every_nullable_budget_check_response_field_is_optional(self) -> None:
        nullable_fields = {
            name: field
            for name, field in BudgetCheckResponse.model_fields.items()
            if NoneType in get_args(field.annotation)
        }

        assert nullable_fields
        assert all(field.default is None for field in nullable_fields.values())
        assert all(not field.is_required() for field in nullable_fields.values())

    def test_directive_v1_allow_response_parses_without_denied_by_period(self) -> None:
        # The Cloud API serializes directive-v1 check responses — the only wire
        # the SDK requests; _build_check_request always opts in — with
        # exclude_none (core budgets router), so an ALLOW response carries NO
        # denied_by_period / price_hints keys at all. denied_by_period is
        # therefore INTENTIONALLY defaulted, not required-nullable: restoring
        # Field(...) would reject every live allow response. Server-side drift
        # for this field is covered live by
        # tests/integration/test_live_contract.py.
        payload = {
            "allowed": True,
            "remaining_budget": 99.99,
            "reservation_id": "res_123",
            "mode": "alert_only",
            "budget_limit": 100.0,
            "current_usage": 0.01,
            "project_id": "proj_abc",
            "failover_directive": {"version": "1", "failover_tuning_allowed": False},
        }

        parsed = BudgetCheckResponse.model_validate(payload)

        assert parsed.denied_by_period is None
        assert parsed.price_hints is None
        assert parsed.failover_directive is not None
        assert parsed.failover_directive.failover_tuning_allowed is False

    @pytest.mark.parametrize("denied_by_period", ["agent_run", "run_stopped"])
    def test_directive_v1_deny_response_parses_with_run_scoped_period(
        self, denied_by_period: str
    ) -> None:
        # "agent_run" and "run_stopped" are the wire literals that run-scoped
        # sticky denial keys on.
        # (budget.py _cache_response). This pins the SDK-side parse of the
        # exact deny shape the server emits (exclude_none drops the
        # reservation_id: deny responses reserve nothing).
        payload = {
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "budget_limit": 100.0,
            "current_usage": 2.5,
            "denied_by_period": denied_by_period,
            "project_id": "proj_abc",
            "failover_directive": {"version": "1", "failover_tuning_allowed": True},
        }

        parsed = BudgetCheckResponse.model_validate(payload)

        assert parsed.denied_by_period == denied_by_period
        assert parsed.reservation_id is None

    def test_budget_confirm_request_field_set(self) -> None:
        assert set(BudgetConfirmRequest.model_fields) == EXPECTED_CONFIRM_FIELDS

    def test_metadata_event_field_set(self) -> None:
        assert set(MetadataEvent.model_fields) == EXPECTED_METADATA_FIELDS

    def test_provider_name_values(self) -> None:
        assert {p.value for p in ProviderName} == EXPECTED_PROVIDER_NAMES

    def test_token_details_field_set(self) -> None:
        assert set(TokenDetails.model_fields) == EXPECTED_TOKEN_DETAILS_FIELDS

    def test_media_usage_field_set(self) -> None:
        assert set(MediaUsage.model_fields) == EXPECTED_MEDIA_USAGE_FIELDS

    def test_breaker_state_report_field_set(self) -> None:
        assert set(BreakerStateReport.model_fields) == EXPECTED_BREAKER_STATE_REPORT_FIELDS

    def test_untracked_surface_report_field_set(self) -> None:
        assert set(UntrackedSurfaceReport.model_fields) == EXPECTED_UNTRACKED_SURFACE_REPORT_FIELDS


def _confirm(**overrides: Any) -> BudgetConfirmRequest:
    base: dict[str, Any] = {
        "reservation_id": "res_123",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "token_details": TokenDetails(input_tokens=10, output_tokens=5),
        "call_id": call_uuid("call-fixed"),
    }
    base.update(overrides)
    return BudgetConfirmRequest(**base)


def _metadata_event(**overrides: object) -> MetadataEvent:
    base: dict[str, object] = {
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_ms": 12.0,
        "status": CallStatus.SUCCESS,
        "is_model_fallback": False,
        "sdk_instance_id": "sdk_abc",
        "timestamp": datetime(2026, 6, 2, tzinfo=UTC),
        "call_id": call_uuid("call-test-123"),
    }
    base.update(overrides)
    return MetadataEvent(**base)  # type: ignore[arg-type]


@pytest.mark.unit
class TestWireModelDumpSnapshots:
    """Pin a representative serialized dump (mode='json') of each wire model."""

    def test_budget_check_request_dump_keys(self) -> None:
        req = BudgetCheckRequest(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            fallback_providers=[ProviderName.ANTHROPIC],
            fallback_models=["claude-x"],
        )
        # A text/chat check: the None-skipping serializer drops the unset optional
        # estimated_media, so the wire stays byte-identical to the pre-window-2
        # check (the deployed Cloud-API model rejects unknown keys).
        assert (
            set(req.model_dump(mode="json")) == EXPECTED_CHECK_FIELDS - _NONE_SKIPPED_CHECK_FIELDS
        )
        assert req.model_dump(mode="json") == {
            "estimated_input_tokens": 10,
            "model": "gpt-5.5",
            "provider": "openai",
            "modality": "text",
            "fallback_providers": ["anthropic"],
            "fallback_models": ["claude-x"],
        }

    def test_breaker_state_report_dump_keys(self) -> None:
        report = BreakerStateReport(
            provider="openai",
            state="open",
            failure_count=3,
            success_count=0,
            reported_at="2026-07-14T12:00:00Z",
            sdk_instance_id="sdk-instance-1",
        )

        assert report.model_dump(mode="json") == {
            "provider": "openai",
            "state": "open",
            "failure_count": 3,
            "success_count": 0,
            "reported_at": "2026-07-14T12:00:00Z",
            "sdk_instance_id": "sdk-instance-1",
        }

    def test_untracked_surface_report_dump_keys(self) -> None:
        report = UntrackedSurfaceReport(
            provider="openai",
            client_shape="openai_sdk",
            mode="sync",
            surface="responses.create",
            rule_kind="unmetered_spend",
            capability_scope="operation",
            posture="warn",
            occurrences=3,
            first_seen_at="2026-08-13T12:00:00Z",
            last_seen_at="2026-08-13T12:01:00Z",
            sdk_instance_id="sdk-instance-1",
            report_id="3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
        )

        assert report.model_dump(mode="json") == {
            "provider": "openai",
            "client_shape": "openai_sdk",
            "mode": "sync",
            "surface": "responses.create",
            "rule_kind": "unmetered_spend",
            "capability_scope": "operation",
            "posture": "warn",
            "occurrences": 3,
            "first_seen_at": "2026-08-13T12:00:00Z",
            "last_seen_at": "2026-08-13T12:01:00Z",
            "sdk_instance_id": "sdk-instance-1",
            "report_id": "3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
        }

    def test_budget_check_request_scoped_dump_carries_agent_run_id(self) -> None:
        req = BudgetCheckRequest(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            agent_run_id="run_abc",
        )

        dumped = req.model_dump(mode="json")

        assert set(dumped) == EXPECTED_CHECK_FIELDS - {
            "estimated_media",
            "tags",
            "failover_directive_version",
        }
        assert dumped["agent_run_id"] == "run_abc"

    def test_budget_check_request_tagged_dump_carries_bounded_tags(self) -> None:
        req = BudgetCheckRequest(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            tags={"team": "research"},
        )

        dumped = req.model_dump(mode="json")

        assert dumped["tags"] == {"team": "research"}
        tags_schema = BudgetCheckRequest.model_json_schema()["properties"]["tags"]["anyOf"][0]
        assert tags_schema == {
            "type": "object",
            "maxProperties": 10,
            "propertyNames": {"minLength": 1, "maxLength": 64},
            "additionalProperties": {"type": "string", "maxLength": 256},
        }

        with pytest.raises(ValidationError):
            BudgetCheckRequest(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                tags={f"key-{index}": "value" for index in range(11)},
            )

    def test_budget_check_request_agent_run_id_length_boundary(self) -> None:
        accepted = BudgetCheckRequest(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            agent_run_id="x" * 256,
        )

        assert accepted.agent_run_id == "x" * 256
        with pytest.raises(ValidationError):
            BudgetCheckRequest(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                agent_run_id="x" * 257,
            )

    def test_budget_check_request_media_dump_carries_estimated_media(self) -> None:
        # A non-text check: estimated_media rides the wire so the server prices a
        # precise per-unit pre-flight cost.
        req = BudgetCheckRequest(
            estimated_input_tokens=0,
            model="gpt-image-2",
            provider=ProviderName.OPENAI,
            modality="image",
            estimated_media=MediaUsage(image_count=2, resolution="1024x1024", quality="low"),
        )
        dumped = req.model_dump(mode="json")
        assert set(dumped) == EXPECTED_CHECK_FIELDS - {
            "agent_run_id",
            "tags",
            "failover_directive_version",
        }
        assert dumped["estimated_media"]["image_count"] == 2
        assert dumped["modality"] == "image"

    def test_budget_check_response_dump_keys(self) -> None:
        response = BudgetCheckResponse(
            allowed=True,
            remaining_budget=80.0,
            reservation_id="res_123",
            mode="alert_only",
            budget_limit=100.0,
            current_usage=20.0,
            denied_by_period=None,
            project_id="proj_abc",
            price_hints={ProviderName.OPENAI: 1.0, ProviderName.ANTHROPIC: 2.0},
        )
        assert set(response.model_dump(mode="json")) == EXPECTED_CHECK_RESPONSE_FIELDS

    def test_budget_check_response_coerces_raw_string_price_hint_keys(self) -> None:
        response = BudgetCheckResponse(
            allowed=True,
            remaining_budget=80.0,
            reservation_id="res_123",
            mode="alert_only",
            budget_limit=100.0,
            current_usage=20.0,
            denied_by_period=None,
            project_id="proj_abc",
            price_hints={"openai": 1.0, "anthropic": 2.0},
        )

        assert response.price_hints == {
            ProviderName.OPENAI: 1.0,
            ProviderName.ANTHROPIC: 2.0,
        }
        assert response.model_dump(mode="json")["price_hints"] == {
            "openai": 1.0,
            "anthropic": 2.0,
        }

    def test_failover_directive_v1_dump_and_rejects_other_versions(self) -> None:
        directive = FailoverDirective(version="1", failover_tuning_allowed=False)

        assert directive.model_dump(mode="json") == {
            "version": "1",
            "failover_tuning_allowed": False,
        }
        with pytest.raises(ValidationError):
            FailoverDirective.model_validate({"version": "2", "failover_tuning_allowed": True})

    def test_budget_confirm_request_dump_keys(self) -> None:
        req = BudgetConfirmRequest(
            reservation_id="res_123",
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            token_details=TokenDetails(input_tokens=10, output_tokens=5),
            call_id=call_uuid("call-fixed"),
        )
        # The None-skipping serializer drops the unset optional fields for the
        # bearer-key providers, so their confirm wire bytes are unchanged —
        # the deployed Cloud-API model rejects unknown keys.
        assert (
            set(req.model_dump(mode="json"))
            == EXPECTED_CONFIRM_FIELDS - _NONE_SKIPPED_CONFIRM_FIELDS
        )

    def test_confirm_media_usage_omitted_when_none_present_when_set(self) -> None:
        # media_usage rides the None-skipping serializer: absent for every
        # chat/token confirm (wire bytes unchanged), present for a non-text
        # served call so the server settles the enforcement counter per-unit.
        assert "media_usage" not in _confirm().model_dump(mode="json")
        dumped = _confirm(
            token_details=TokenDetails(),
            media_usage=MediaUsage(image_count=3, resolution="1024x1024"),
        ).model_dump(mode="json")
        assert dumped["media_usage"]["image_count"] == 3
        assert dumped["media_usage"]["resolution"] == "1024x1024"

    def test_metadata_media_usage_omitted_when_none_present_when_set(self) -> None:
        # Mirror on the metadata-event wire (the other media_usage embedding).
        assert "media_usage" not in _metadata_event().model_dump(mode="json")
        dumped = _metadata_event(media_usage=MediaUsage(image_count=1, quality="high")).model_dump(
            mode="json"
        )
        assert dumped["media_usage"]["image_count"] == 1
        assert dumped["media_usage"]["quality"] == "high"

    def test_metadata_tags_omitted_when_none_present_when_set(self) -> None:
        assert "tags" not in _metadata_event().model_dump(mode="json")
        dumped = _metadata_event(tags={"team": "research"}).model_dump(mode="json")
        assert dumped["tags"] == {"team": "research"}

    def test_token_details_default_dump_omits_is_estimated(self) -> None:
        # Provider-reported (non-estimated) counts: is_estimated stays OFF the
        # wire entirely — never "is_estimated": false — so every existing
        # provider's payload is byte-identical to the pre-is_estimated wire.
        dumped = TokenDetails(input_tokens=10, output_tokens=5).model_dump(mode="json")
        assert "is_estimated" not in dumped
        assert set(dumped) == EXPECTED_TOKEN_DETAILS_FIELDS - {"is_estimated"}

    def test_token_details_estimated_dump_includes_is_estimated(self) -> None:
        # The estimation fallback fired: the degradation marker MUST serialize.
        dumped = TokenDetails(input_tokens=10, output_tokens=5, is_estimated=True).model_dump(
            mode="json"
        )
        assert dumped["is_estimated"] is True
        assert set(dumped) == EXPECTED_TOKEN_DETAILS_FIELDS

    def test_confirm_with_default_token_details_carries_no_is_estimated_key(self) -> None:
        # The deployment-ordering guarantee, enforced on the EXACT dump mode the
        # reporter posts (model_dump(mode="json")): a confirm carrying
        # provider-reported counts contains no is_estimated key anywhere, so a
        # deployed Cloud API that forbids unknown TokenDetails keys never 422s.
        nested = _confirm().model_dump(mode="json")["token_details"]
        assert "is_estimated" not in nested
        # And the estimation-fallback confirm DOES carry it.
        estimated = _confirm(
            token_details=TokenDetails(input_tokens=10, output_tokens=5, is_estimated=True)
        ).model_dump(mode="json")["token_details"]
        assert estimated["is_estimated"] is True

    def test_metadata_event_with_default_token_details_carries_no_is_estimated_key(self) -> None:
        # Same guarantee on the metadata-event wire (the other embedding).
        ev = _metadata_event(token_details=TokenDetails(input_tokens=10, output_tokens=5))
        assert "is_estimated" not in ev.model_dump(mode="json")["token_details"]

    def test_confirm_provider_region_omitted_when_none_present_when_set(self) -> None:
        # Mirror of test_metadata_provider_region_omitted_when_none_present_when_set:
        # absent for bearer-key providers, present for Bedrock where pricing is
        # keyed by (provider, model, region).
        assert "provider_region" not in _confirm().model_dump(mode="json")
        dumped = _confirm(provider_region="us-east-1").model_dump(mode="json")
        assert dumped["provider_region"] == "us-east-1"

    def test_confirm_service_tier_omitted_when_none_present_when_set(self) -> None:
        # service_tier settles the enforcement counter at the tier-repriced
        # rate (flex 0.5x / priority 1.75x / optimized 1.25x). The pinned wire
        # shape is key-ABSENT when None (never "service_tier": null) — the
        # Cloud API settles key-absent confirms at Standard rates.
        assert "service_tier" not in _confirm().model_dump(mode="json")
        dumped = _confirm(service_tier="priority").model_dump(mode="json")
        assert dumped["service_tier"] == "priority"

    def test_confirm_service_tier_accepts_every_pinned_literal(self) -> None:
        # The seven values pinned lock-step with the Cloud API's ServiceTier
        # literal — its confirm model is extra="forbid"-strict on values too.
        for tier in ("auto", "default", "flex", "scale", "priority", "standard", "optimized"):
            assert _confirm(service_tier=tier).service_tier == tier

    def test_confirm_service_tier_rejects_unknown_value(self) -> None:
        # A novel tier echo must never reach the wire: the Cloud API would 422
        # the confirm and strand the reservation. The model is the backstop;
        # build_confirm_request narrows unknown echoes to None before this.
        with pytest.raises(ValidationError):
            _confirm(service_tier="hyperspeed")

    def test_non_failover_metadata_dump_omits_possibly_succeeded_keeps_call_id(self) -> None:
        # A plain non-failover SUCCESS event: the None-skipping serializer drops
        # every None-valued optional (possibly_succeeded among them) so the wire
        # stays byte-compatible with the pre-failover event, BUT call_id (the
        # always-present reconciliation join key) is NEVER skipped.
        ev = _metadata_event(call_id=call_uuid("call-fixed-123"))
        dumped = ev.model_dump(mode="json")

        assert "possibly_succeeded" not in dumped  # None -> skipped
        assert "failover_reason" not in dumped  # None -> skipped
        assert "requested_provider" not in dumped  # None -> skipped
        assert dumped["call_id"] == call_uuid("call-fixed-123")  # always on the wire

    def test_failover_metadata_dump_includes_reconciliation_fields(self) -> None:
        # A reconciliation abort event: possibly_succeeded=True is on the wire,
        # and so is call_id. The key set is the always-present fields plus the
        # populated failover fields.
        ev = _metadata_event(
            status=CallStatus.ERROR,
            is_provider_fallback=False,
            failover_error_class="APITimeoutError",
            possibly_succeeded=True,
            call_id=call_uuid("call-abort-9"),
        )
        dumped = ev.model_dump(mode="json")

        assert dumped["possibly_succeeded"] is True
        assert dumped["failover_error_class"] == "APITimeoutError"
        assert dumped["call_id"] == call_uuid("call-abort-9")


# --------------------------------------------------------------------------- #
# 1b. Load-bearing field CONSTRAINTS (not just names)                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestWireModelFieldConstraints:
    """Pin the constraints the Cloud-API contract depends on, not just the names.

    A field can keep its name while silently losing a required/length constraint;
    the name-set snapshots above would not catch that. These assertions freeze the
    load-bearing constraints.
    """

    def test_confirm_provider_is_required(self) -> None:
        # the served provider is REQUIRED on confirm so the Cloud API can
        # price against the provider that actually served the call.
        assert BudgetConfirmRequest.model_fields["provider"].is_required() is True

    def test_breaker_state_report_constraints_match_cloud(self) -> None:
        BreakerStateReport(
            provider="openai",
            state="closed",
            failure_count=0,
            success_count=1,
            reported_at="2026-07-14T12:00:00Z",
            sdk_instance_id="x" * 100,
        )

        invalid_payload = {
            "provider": "openai",
            "state": "degraded",
            "failure_count": -1,
            "success_count": 0,
            "reported_at": "2026-07-14T12:00:00Z",
            "sdk_instance_id": "x" * 101,
            "message": "must never ride breaker telemetry",
        }
        with pytest.raises(ValidationError) as exc_info:
            BreakerStateReport(**invalid_payload)

        errors = {error["type"] for error in exc_info.value.errors()}
        assert {"enum", "greater_than_equal", "string_too_long", "extra_forbidden"} <= errors

    def test_confirm_call_id_is_required(self) -> None:
        assert BudgetConfirmRequest.model_fields["call_id"].is_required() is True

    def test_confirm_call_id_must_be_a_canonical_uuid(self) -> None:
        # call_id is DURABLE SPEND IDENTITY: the API's cost-event ledger dedups
        # on it, so the wire pins the canonical lowercase RFC 4122 text form
        # that str(uuid.uuid4()) emits (core's shared CALL_ID_PATTERN) and 422s
        # anything else. The SDK has always SENT that shape; pinning it here
        # makes a drifted id fail at the seam that built it instead of arriving
        # as a rejected settlement whose spend then goes unconfirmed.
        assert wire_constants.CALL_ID_PATTERN == (
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )

        canonical = "3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b"
        assert _confirm(call_id=canonical).call_id == canonical

        for rejected in (
            "",
            "call-fixed",
            canonical.upper(),
            f"{{{canonical}}}",
            f"urn:uuid:{canonical}",
            f" {canonical}",
            "x" * 36,
        ):
            with pytest.raises(ValidationError):
                _confirm(call_id=rejected)

    def test_confirm_without_provider_raises_validation_error(self) -> None:
        # Constructing a confirm with no provider must hard-fail, not default.
        with pytest.raises(ValidationError):
            BudgetConfirmRequest(  # type: ignore[call-arg]
                reservation_id="res_123",
                model="gpt-5.5",
                token_details=TokenDetails(input_tokens=10, output_tokens=5),
            )

    def test_confirm_without_call_id_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            BudgetConfirmRequest(  # type: ignore[call-arg]
                reservation_id="res_123",
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                token_details=TokenDetails(input_tokens=10, output_tokens=5),
            )

    def test_metadata_call_id_is_required(self) -> None:
        assert MetadataEvent.model_fields["call_id"].is_required() is True

    def test_metadata_call_id_must_be_a_canonical_uuid(self) -> None:
        # The OTHER half of the same identity contract: the metadata event and
        # its confirm carry the SAME call_id (the reconciliation join key), and
        # the API pins the canonical UUID form on both. A shape only one side
        # accepts would let an event through that its confirm could never join.
        canonical = "3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b"
        assert _metadata_event(call_id=canonical).call_id == canonical

        for rejected in (
            "",
            "call-test-123",
            canonical.upper(),
            f"{{{canonical}}}",
            f"urn:uuid:{canonical}",
            f" {canonical}",
            "x" * 36,
        ):
            with pytest.raises(ValidationError):
                _metadata_event(call_id=rejected)

    def test_metadata_tags_bounds_are_pinned(self) -> None:
        assert wire_constants.TAGS_MAX_KEYS == 10
        assert wire_constants.TAG_KEY_MAX_LENGTH == 64
        assert wire_constants.TAG_VALUE_MAX_LENGTH == 256

    def test_metadata_parent_agent_run_id_matches_agent_run_id_bound(self) -> None:
        schema = MetadataEvent.model_json_schema()["properties"]

        assert (
            schema["parent_agent_run_id"]["anyOf"][0]["maxLength"]
            == schema["agent_run_id"]["anyOf"][0]["maxLength"]
        )

    def test_metadata_tags_schema_exposes_all_wire_bounds(self) -> None:
        tags_schema = MetadataEvent.model_json_schema()["properties"]["tags"]["anyOf"][0]

        assert tags_schema["type"] == "object"
        assert tags_schema["maxProperties"] == 10
        assert tags_schema["propertyNames"] == {"minLength": 1, "maxLength": 64}
        assert tags_schema["additionalProperties"] == {"type": "string", "maxLength": 256}

    def test_metadata_without_call_id_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            MetadataEvent(
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                input_tokens=10,
                output_tokens=5,
                latency_ms=12.0,
                status=CallStatus.SUCCESS,
                is_model_fallback=False,
                sdk_instance_id="sdk_abc",
                timestamp=datetime(2026, 6, 2, tzinfo=UTC),
            )

    def test_metadata_requested_model_max_length_pinned(self) -> None:
        # requested_model is bounded so a malformed/oversized value can't bloat
        # the wire payload. Pin the exact bound (2048 — the AWS Converse modelId
        # contract; Bedrock inference-profile ARNs exceed the old 100).
        field = MetadataEvent.model_fields["requested_model"]
        max_lengths = [m.max_length for m in field.metadata if hasattr(m, "max_length")]
        assert 2048 in max_lengths

    def test_metadata_service_tier_max_length_lock_step_pinned(self) -> None:
        # SERVICE_TIER_MAX_LENGTH is vendored lock-step with core
        # shared/constants.py (32). Since the API rejects per-event, an
        # over-length tier in the ENVELOPE is the remaining whole-batch 422
        # path — the adapter clamps (tier[:32], pinned in test_providers/) and
        # this bound are the only guards. Drift here silently re-opens
        # total batch loss.
        assert SERVICE_TIER_MAX_LENGTH == 32
        field = MetadataEvent.model_fields["service_tier"]
        max_lengths = [m.max_length for m in field.metadata if hasattr(m, "max_length")]
        assert SERVICE_TIER_MAX_LENGTH in max_lengths

    def test_metadata_failover_error_class_max_length_pinned(self) -> None:
        # failover_error_class carries only type(exc).__name__; the 64 bound caps
        # it defensively (a class name can't legitimately exceed it).
        field = MetadataEvent.model_fields["failover_error_class"]
        max_lengths = [m.max_length for m in field.metadata if hasattr(m, "max_length")]
        assert 64 in max_lengths

    def test_budget_check_fallback_models_element_max_length_pinned(self) -> None:
        with pytest.raises(ValidationError):
            BudgetCheckRequest(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                fallback_providers=[ProviderName.ANTHROPIC],
                fallback_models=["x" * 2049],
            )

    def test_lease_request_fallback_chain_max_items_pinned(self) -> None:
        # Core's lease schemas cap the declared chain at 8. Without the SDK
        # mirror a 9-model chain validates here and 422s server-side, so the
        # bound is pinned on BOTH lease requests that carry a chain.
        for schema in (
            LeaseGrantRequest.model_json_schema(),
            LeaseRenewRequest.model_json_schema(),
        ):
            assert schema["properties"]["fallback_models"]["maxItems"] == 8

    def test_lease_id_max_length_pinned(self) -> None:
        # The lease id echoed back to the API is bounded lock-step with core.
        assert wire_constants.LEASE_ID_MAX_LENGTH == 64
        for model in (LeaseRenewRequest, LeaseSurrenderRequest):
            field = model.model_fields["lease_id"]
            max_lengths = [m.max_length for m in field.metadata if hasattr(m, "max_length")]
            assert 64 in max_lengths

    def test_metadata_provider_region_omitted_when_none_present_when_set(self) -> None:
        # provider_region rides the None-skipping serializer: absent for the
        # bearer-key providers (wire bytes unchanged), present for Bedrock where
        # pricing is keyed by (provider, model, region).
        assert "provider_region" not in _metadata_event().model_dump(mode="json")
        dumped = _metadata_event(provider_region="us-east-1").model_dump(mode="json")
        assert dumped["provider_region"] == "us-east-1"


# --------------------------------------------------------------------------- #
# Behavioral helpers (live dispatch path)                                     #
# --------------------------------------------------------------------------- #
class _Status(Exception):
    """Duck-typed transport error carrying an HTTP ``status_code``.

    429 classifies FAILOVER; 500 classifies POST_SEND_AMBIGUOUS.
    """

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    return client


def _anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"
    client.with_options.return_value = client
    return client


def _openai_response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def _anthropic_message_start(input_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=input_tokens, cache_read_input_tokens=0)
        ),
    )


def _anthropic_text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _anthropic_message_delta(output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="end_turn"),
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


def _allow_budget(reservation_id: str | None = "res_123") -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=reservation_id,
        project_id=None,
        budget_limit=100.0,
        current_usage=0.0,
        mode=SimpleNamespace(value="alert_only"),
        price_hints=None,
    )


def _make_solwyn(client: object, **overrides: object) -> Solwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    solwyn._solwyn_reporter.report = MagicMock()
    # Non-streaming settlement now rides report_settlement(confirm, event); keep
    # _reported_events observable by forwarding the settled event to report().
    solwyn._solwyn_reporter.report_settlement = lambda _req, event: solwyn._solwyn_reporter.report(
        event
    )
    return solwyn


def _make_async_solwyn(client: object, **overrides: object) -> AsyncSolwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._solwyn_reporter.report = MagicMock()
    solwyn._solwyn_reporter.report_settlement = lambda _req, event: solwyn._solwyn_reporter.report(
        event
    )
    return solwyn


def _force_primary_open(solwyn: Solwyn | AsyncSolwyn, provider: str = "openai") -> None:
    """Trip the primary breaker OPEN so the chain skips straight to the fallback."""
    cb = solwyn._get_circuit_breaker(provider)
    for _ in range(cb.failure_threshold):
        cb.record_failure()


def _close(solwyn: Solwyn) -> None:
    solwyn._solwyn_reporter._http.close()
    solwyn._solwyn_budget._http.close()


async def _aclose(solwyn: AsyncSolwyn) -> None:
    await solwyn._solwyn_reporter._http.aclose()
    await solwyn._solwyn_budget._http.aclose()


def _reported_events(solwyn: Solwyn | AsyncSolwyn) -> list[MetadataEvent]:
    return [c.args[0] for c in solwyn._solwyn_reporter.report.call_args_list]


_PLAIN_REQUEST = {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}
_STREAM_REQUEST = {**_PLAIN_REQUEST, "stream": True}


# --------------------------------------------------------------------------- #
# 1c. Advisory chain-hint wiring into /budgets/check                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestChainHintWiring:
    """The client threads the configured failover chain into check_budget."""

    def test_check_budget_carries_fallback_chain_from_runtimes(self) -> None:
        # The advisory chain hint: check_budget must receive
        # fallback_providers + fallback_models drawn from runtimes[1:] (the
        # fallback entries — primary excluded), aligned element-for-element.
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        gemini = _openai_client()  # second OpenAI-shaped fallback (distinct model)
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[
                (anthropic, "claude-sonnet-5", {"max_tokens": 256}),
                (gemini, "gpt-5.4-mini"),
            ],
        )

        check_spy = MagicMock(return_value=_allow_budget())
        with patch.object(solwyn._solwyn_budget, "check_budget", check_spy):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        check_spy.assert_called_once()
        kwargs = check_spy.call_args.kwargs
        # Primary is provider=gpt-5.5/openai; the hint excludes it and lists ONLY
        # runtimes[1:] in order, aligned element-for-element.
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-5.5"
        runtimes = solwyn._solwyn_runtimes
        assert kwargs["fallback_providers"] == [r.entry.provider.value for r in runtimes[1:]]
        assert kwargs["fallback_models"] == [r.entry.model for r in runtimes[1:]]
        # Concretely: the two configured fallbacks, in order.
        assert kwargs["fallback_providers"] == ["anthropic", "openai"]
        assert kwargs["fallback_models"] == ["claude-sonnet-5", "gpt-5.4-mini"]
        assert len(kwargs["fallback_providers"]) == len(kwargs["fallback_models"])

        _close(solwyn)


# --------------------------------------------------------------------------- #
# 2. call_id consistency across metadata + confirm                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestCallIdConsistencyCrossProvider:
    """The success event and its confirm share one call_id on the failover path."""

    def test_non_streaming_cross_provider_failover_shares_call_id(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        settlements: list[tuple[Any, Any]] = []

        def report_settlement(req: Any, event: Any) -> None:
            settlements.append((req, event))
            solwyn._solwyn_reporter.report(event)

        solwyn._solwyn_reporter.report_settlement = report_settlement
        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        assert success[0].is_provider_fallback is True
        assert len(settlements) == 1
        confirm, _event = settlements[0]
        # Same join key on the served-provider success event AND its confirm.
        assert confirm.call_id == success[0].call_id
        assert confirm.provider == ProviderName.ANTHROPIC

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_non_streaming_cross_provider_confirm_uses_served_provider(
        self,
    ) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status(429))
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())
        solwyn = _make_async_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        settlements: list[tuple[Any, Any]] = []

        def report_settlement(req: Any, event: Any) -> None:
            settlements.append((req, event))
            solwyn._solwyn_reporter.report(event)

        solwyn._solwyn_reporter.report_settlement = report_settlement
        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        assert len(settlements) == 1
        confirm, _event = settlements[0]
        assert confirm.call_id == success[0].call_id
        assert confirm.provider == ProviderName.ANTHROPIC

        await _aclose(solwyn)

    def test_two_separate_calls_get_different_call_ids(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-5.5")

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        ids = [e.call_id for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(ids) == 2
        assert ids[0] != ids[1]

        _close(solwyn)

    def test_streaming_on_complete_settlement_confirm_and_event_share_call_id(self) -> None:
        # Sync streaming settles via reporter.report_settlement (fire-and-forget)
        # -- NOT confirm_cost. The on_complete success event's call_id must equal
        # the confirm request's call_id (the join key).
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter(
            [
                _anthropic_message_start(input_tokens=31),
                _anthropic_text_chunk("hel"),
                _anthropic_text_chunk("lo"),
                _anthropic_message_delta(output_tokens=9),
            ]
        )
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        settlements: list[tuple[Any, Any]] = []

        def report_settlement(req: Any, event: Any) -> None:
            settlements.append((req, event))
            solwyn._solwyn_reporter.report(event)

        solwyn._solwyn_reporter.report_settlement = report_settlement

        with patch.object(
            solwyn._solwyn_budget,
            "check_budget",
            return_value=_allow_budget(reservation_id="resv_s"),
        ):
            stream = solwyn.chat.completions.create(**_STREAM_REQUEST)
            list(stream)  # drain to fire on_complete

        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        assert len(settlements) == 1
        confirm, event = settlements[0]
        # The fire-and-forget confirm carries the SAME join key as the event.
        assert confirm.call_id == event.call_id == success[0].call_id
        assert confirm.provider == ProviderName.ANTHROPIC

        _close(solwyn)


@pytest.mark.unit
class TestNormalizeBeforeSettlement:
    """Cross-provider success must normalize before confirm + SUCCESS metadata."""

    def test_sync_normalize_failure_does_not_confirm_or_report_success(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        settle_spy = MagicMock()
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            patch.object(solwyn._solwyn_reporter, "report_settlement", settle_spy),
            patch("solwyn.client._translation.normalize_response", side_effect=RuntimeError),
            pytest.raises(RuntimeError),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        settle_spy.assert_not_called()
        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert success == []

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_normalize_failure_does_not_confirm_or_report_success(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status(429))
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())
        solwyn = _make_async_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        settle_spy = MagicMock()
        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
            patch.object(solwyn._solwyn_reporter, "report_settlement", settle_spy),
            patch("solwyn.client._translation.normalize_response", side_effect=RuntimeError),
            pytest.raises(RuntimeError),
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        settle_spy.assert_not_called()
        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert success == []

        await _aclose(solwyn)


# --------------------------------------------------------------------------- #
# 3. possibly_succeeded matrix                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestPossiblySucceededMatrix:
    """possibly_succeeded=True ONLY on a correctly-not-failed-over PSA abort."""

    @pytest.mark.parametrize("status_code", [500, 503])
    def test_post_send_ambiguous_reraises_and_flags_possibly_succeeded(
        self, status_code: int
    ) -> None:
        # A 5xx is POST_SEND_AMBIGUOUS. Under the default "safe" idempotency (no
        # fallback configured) the ORIGINAL exception re-raises AND the error
        # event flags possibly_succeeded=True for Cloud-API reconciliation.
        openai = _openai_client()
        original = _Status(status_code, "upstream blew up")
        openai.chat.completions.create.side_effect = original
        solwyn = _make_solwyn(openai, model="gpt-5.5")

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)
        assert exc_info.value is original  # ORIGINAL re-raised (drop-in contract)

        errors = [e for e in _reported_events(solwyn) if e.status is CallStatus.ERROR]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].call_id  # join key still present

        _close(solwyn)

    def test_plain_failover_429_leaves_possibly_succeeded_absent(self) -> None:
        # A 429 on the primary is a clean FAILOVER: the next candidate serves and
        # the primary error event must NOT flag possibly_succeeded (None/absent).
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        events = _reported_events(solwyn)
        errors = [e for e in events if e.status is CallStatus.ERROR]
        success = [e for e in events if e.status is CallStatus.SUCCESS]
        assert len(errors) == 1  # primary 429 error event
        assert len(success) == 1  # anthropic served
        # The FAILOVER error event leaves possibly_succeeded None (skipped on wire).
        assert errors[0].possibly_succeeded is None
        assert "possibly_succeeded" not in errors[0].model_dump()
        # And the success event likewise.
        assert success[0].possibly_succeeded is None

        _close(solwyn)

    def test_plain_success_leaves_possibly_succeeded_absent(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-5.5")

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        assert success[0].possibly_succeeded is None
        assert "possibly_succeeded" not in success[0].model_dump()

        _close(solwyn)


# --------------------------------------------------------------------------- #
# 4. is_model_fallback NARROWED                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestIsModelFallbackNarrowed:
    """is_model_fallback is same-provider model swap ONLY; never cross-provider."""

    def test_same_provider_model_swap_is_model_fallback_only(self) -> None:
        # Primary gpt-5.5 429s (FAILOVER); the SAME OpenAI client serves the
        # gpt-5.4-mini swap. The shared OpenAI breaker records one failure but
        # stays closed, so the same-provider second candidate proceeds.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = [_Status(429), _openai_response()]
        solwyn = _make_solwyn(openai, model="gpt-5.5", fallback=[(openai, "gpt-5.4-mini")])

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        assert success[0].is_model_fallback is True
        assert success[0].is_provider_fallback is False
        assert success[0].model == "gpt-5.4-mini"

        _close(solwyn)

    def test_cross_provider_failover_is_provider_fallback_only(self) -> None:
        openai = _openai_client()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        _force_primary_open(solwyn, "openai")

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        assert success[0].is_model_fallback is False
        assert success[0].is_provider_fallback is True

        _close(solwyn)

    def test_primary_success_both_flags_false(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-5.5")

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in _reported_events(solwyn) if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        assert success[0].is_model_fallback is False
        assert success[0].is_provider_fallback is False

        _close(solwyn)


# --------------------------------------------------------------------------- #
# 5. failover_error_class FIREWALL                                             #
# --------------------------------------------------------------------------- #
class _LeakyError(Exception):
    """A POST_SEND_AMBIGUOUS error whose str() embeds a content SENTINEL.

    Carries status_code=500 so it classifies POST_SEND_AMBIGUOUS — the path that
    populates failover_error_class on the error event. The SENTINEL in the
    message must NEVER reach failover_error_class (only type(exc).__name__) NOR
    any byte of the emitted event's model_dump.
    """

    status_code = 500


@pytest.mark.unit
class TestFailoverErrorClassFirewall:
    """failover_error_class is populated ONLY from type(exc).__name__."""

    def test_sentinel_in_str_exc_never_reaches_event(self) -> None:
        sentinel = "SECRET_PROMPT_LEAK_a1b2c3"
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _LeakyError(f"boom {sentinel} in body")
        solwyn = _make_solwyn(openai, model="gpt-5.5")

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_LeakyError),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        errors = [e for e in _reported_events(solwyn) if e.status is CallStatus.ERROR]
        assert len(errors) == 1
        ev = errors[0]
        # The field equals the class name ONLY — never str(exc).
        assert ev.failover_error_class == "_LeakyError"
        # And the SENTINEL is absent from EVERY byte of the wire payload.
        blob = json.dumps(ev.model_dump(mode="json"))
        assert sentinel not in blob

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_sentinel_in_str_exc_never_reaches_event(self) -> None:
        sentinel = "SECRET_PROMPT_LEAK_a1b2c3"
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(
            side_effect=_LeakyError(f"boom {sentinel} in body")
        )
        solwyn = _make_async_solwyn(openai, model="gpt-5.5")

        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
            pytest.raises(_LeakyError),
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        errors = [e for e in _reported_events(solwyn) if e.status is CallStatus.ERROR]
        assert len(errors) == 1
        ev = errors[0]
        assert ev.failover_error_class == "_LeakyError"
        blob = json.dumps(ev.model_dump(mode="json"))
        assert sentinel not in blob

        await _aclose(solwyn)
