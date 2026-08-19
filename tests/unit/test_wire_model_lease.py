"""Wire-model tests for the PJ-2 budget-lease contract.

Covers the three lease request models (None-skipping serializer), the shared
grant/renew response model (conditionally-omitted fields tolerate omission),
and the ``BudgetConfirmRequest`` exactly-one-of ``reservation_id``/``lease_id``
validator.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import call_uuid
from pydantic import ValidationError

from solwyn._constants import AGENT_RUN_ID_MAX_LENGTH
from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetConfirmRequest,
    BudgetMode,
    LeaseGrantRequest,
    LeaseGrantResponse,
    LeasePosture,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    ProviderName,
    RunControlDirective,
)

# A grant/renew response with every optional field present.
_FULL_RESPONSE: dict[str, Any] = {
    "eligible": True,
    "allowed": True,
    "lease_id": "lse_abc",
    "generation": 1,
    "granted_tokens": 32_000,
    "refresh_interval_s": 15.0,
    "lease_length_s": 120.0,
    "headroom_share_tokens": 500_000,
    "posture": {"mode": "alert_only", "on_unreachable": "fail_open"},
    "final_grant": False,
    "project_id": "proj_123",
    "mode": "alert_only",
    "budget_limit": 100.0,
    "current_usage": 12.5,
    "remaining_budget": 87.5,
}

# The minimum an eligible+allowed response must always carry (display snapshot).
_DISPLAY_ONLY: dict[str, Any] = {
    "eligible": False,
    "allowed": True,
    "project_id": "proj_123",
    "mode": "hard_deny",
    "budget_limit": 100.0,
    "current_usage": 12.5,
    "remaining_budget": 87.5,
}


@pytest.mark.unit
class TestLeaseGrantRequest:
    """Grant request shape + None-skipping serialization."""

    def test_minimal_payload_defaults(self) -> None:
        # Arrange / Act
        request = LeaseGrantRequest(
            agent_run_id="run_1",
            holder_id="sdk-instance-1",
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
        )

        # Assert
        assert request.fallback_providers == []
        assert request.fallback_models == []
        assert request.fail_open is True
        assert request.estimated_input_tokens == 0

    def test_serializes_without_none_values(self) -> None:
        # Arrange
        request = LeaseGrantRequest(
            agent_run_id="run_1",
            holder_id="sdk-instance-1",
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            estimated_input_tokens=1_000,
        )

        # Act
        payload = request.model_dump(mode="json")

        # Assert — no key may carry a null (the Cloud-API model forbids extras
        # and every field here is non-nullable).
        assert None not in payload.values()
        assert payload["agent_run_id"] == "run_1"
        assert payload["provider"] == "openai"
        assert payload["estimated_input_tokens"] == 1_000
        assert "run_directive_version" not in payload

    def test_run_directive_opt_in_round_trips(self) -> None:
        request = LeaseGrantRequest(
            agent_run_id="run_1",
            holder_id="sdk-instance-1",
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            run_directive_version="1",
        )

        assert request.model_dump(mode="json")["run_directive_version"] == "1"

        with pytest.raises(ValidationError):
            LeaseGrantRequest(
                agent_run_id="run_1",
                holder_id="sdk-instance-1",
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                run_directive_version="2",  # type: ignore[arg-type]
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            LeaseGrantRequest(
                agent_run_id="run_1",
                holder_id="sdk-instance-1",
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                unexpected="oops",  # type: ignore[call-arg]
            )

    def test_rejects_negative_estimated_input_tokens(self) -> None:
        with pytest.raises(ValidationError):
            LeaseGrantRequest(
                agent_run_id="run_1",
                holder_id="sdk-instance-1",
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                estimated_input_tokens=-1,
            )

    def test_rejects_holder_id_over_64_chars(self) -> None:
        with pytest.raises(ValidationError):
            LeaseGrantRequest(
                agent_run_id="run_1",
                holder_id="x" * 65,
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
            )

    def test_rejects_a_fallback_chain_over_eight_models(self) -> None:
        # Contract parity with core (and with BudgetCheckRequest): a 9-model
        # chain must fail HERE, not with a 422 after the round-trip.
        with pytest.raises(ValidationError):
            LeaseGrantRequest(
                agent_run_id="run_1",
                holder_id="sdk-instance-1",
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                fallback_providers=[ProviderName.ANTHROPIC] * 9,
                fallback_models=[f"claude-{index}" for index in range(9)],
            )

    def test_accepts_a_full_eight_model_fallback_chain(self) -> None:
        request = LeaseGrantRequest(
            agent_run_id="run_1",
            holder_id="sdk-instance-1",
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            fallback_providers=[ProviderName.ANTHROPIC] * 8,
            fallback_models=[f"claude-{index}" for index in range(8)],
        )
        assert len(request.fallback_models) == 8

    def test_rejects_misaligned_fallback_chain(self) -> None:
        # The chain is aligned element-for-element, same as BudgetCheckRequest.
        with pytest.raises(ValidationError):
            LeaseGrantRequest(
                agent_run_id="run_1",
                holder_id="sdk-instance-1",
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                fallback_providers=[ProviderName.ANTHROPIC],
                fallback_models=[],
            )


@pytest.mark.unit
class TestLeaseRenewRequest:
    """Renew request shape + None-skipping serialization."""

    def test_optional_redeclaration_omitted_when_none(self) -> None:
        # Arrange
        request = LeaseRenewRequest(
            lease_id="lse_abc",
            holder_id="sdk-instance-1",
            generation=3,
            spent_tokens=1_234,
            reserved_tokens=42,
            uncounted_calls=2,
            uncounted_tokens=900,
        )

        # Act
        payload = request.model_dump(mode="json")

        # Assert — the None-skipping serializer drops the unset re-declaration.
        assert "model" not in payload
        assert "provider" not in payload
        assert payload["generation"] == 3
        assert payload["spent_tokens"] == 1_234
        assert payload["uncounted_calls"] == 2
        assert "run_directive_version" not in payload

    def test_run_directive_opt_in_round_trips(self) -> None:
        request = LeaseRenewRequest(
            lease_id="lse_abc",
            holder_id="sdk-instance-1",
            generation=3,
            run_directive_version="1",
        )

        assert request.model_dump(mode="json")["run_directive_version"] == "1"

    def test_redeclaration_present_when_set(self) -> None:
        request = LeaseRenewRequest(
            lease_id="lse_abc",
            holder_id="sdk-instance-1",
            generation=3,
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            fallback_providers=[ProviderName.ANTHROPIC],
            fallback_models=["claude-4"],
        )

        payload = request.model_dump(mode="json")

        assert payload["model"] == "gpt-5.5"
        assert payload["provider"] == "openai"
        assert payload["fallback_models"] == ["claude-4"]

    def test_defaults_are_zero(self) -> None:
        request = LeaseRenewRequest(lease_id="lse_abc", holder_id="sdk-instance-1", generation=1)

        assert request.spent_tokens == 0
        assert request.reserved_tokens == 0
        assert request.uncounted_calls == 0
        assert request.uncounted_tokens == 0

    def test_rejects_a_fallback_chain_over_eight_models(self) -> None:
        with pytest.raises(ValidationError):
            LeaseRenewRequest(
                lease_id="lse_abc",
                holder_id="sdk-instance-1",
                generation=1,
                fallback_providers=[ProviderName.ANTHROPIC] * 9,
                fallback_models=[f"claude-{index}" for index in range(9)],
            )

    def test_rejects_lease_id_over_64_chars(self) -> None:
        with pytest.raises(ValidationError):
            LeaseRenewRequest(
                lease_id="x" * 65,
                holder_id="sdk-instance-1",
                generation=1,
            )

    def test_rejects_negative_counters(self) -> None:
        with pytest.raises(ValidationError):
            LeaseRenewRequest(
                lease_id="lse_abc",
                holder_id="sdk-instance-1",
                generation=1,
                spent_tokens=-1,
            )


@pytest.mark.unit
class TestLeaseSurrenderRequest:
    """Surrender request carries the final true-up report."""

    def test_shape(self) -> None:
        request = LeaseSurrenderRequest(
            lease_id="lse_abc", holder_id="sdk-instance-1", generation=4, spent_tokens=77
        )

        payload = request.model_dump(mode="json")

        assert payload == {
            "lease_id": "lse_abc",
            "holder_id": "sdk-instance-1",
            "generation": 4,
            "spent_tokens": 77,
        }

    def test_rejects_lease_id_over_64_chars(self) -> None:
        with pytest.raises(ValidationError):
            LeaseSurrenderRequest(
                lease_id="x" * 65,
                holder_id="sdk-instance-1",
                generation=4,
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            LeaseSurrenderRequest(
                lease_id="lse_abc",
                holder_id="sdk-instance-1",
                generation=4,
                released_tokens=1,  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestLeaseGrantResponse:
    """The shared grant/renew response tolerates every exclude-none omission."""

    def test_full_response_round_trips(self) -> None:
        response = LeaseGrantResponse.model_validate(_FULL_RESPONSE)

        assert response.lease_id == "lse_abc"
        assert response.generation == 1
        assert response.granted_tokens == 32_000
        assert response.posture is not None
        assert response.posture.mode is BudgetMode.ALERT_ONLY
        assert response.posture.on_unreachable == "fail_open"
        assert response.final_grant is False
        assert response.ineligible_reason is None
        assert response.denied_by_period is None
        assert response.run_control is None

    def test_response_with_and_without_run_control_is_compatible(self) -> None:
        legacy = LeaseGrantResponse.model_validate(_FULL_RESPONSE)
        directed = LeaseGrantResponse.model_validate(
            {
                **_FULL_RESPONSE,
                "run_control": {
                    "version": "1",
                    "action": "terminate",
                    "agent_run_id": "run_1",
                    "reason": "manual_kill",
                },
            }
        )

        assert legacy.run_control is None
        assert directed.run_control == RunControlDirective(
            version="1",
            action="terminate",
            agent_run_id="run_1",
            reason="manual_kill",
        )


@pytest.mark.unit
class TestLeaseGrantResponseOmissions:
    def test_ineligible_response_omits_every_lease_field(self) -> None:
        # Arrange — an ineligible run's response carries only the display snapshot
        # plus the reason (core serializes directive-v1 style exclude_none).
        payload = dict(_DISPLAY_ONLY, ineligible_reason="unit_priced_model")

        # Act
        response = LeaseGrantResponse.model_validate(payload)

        # Assert
        assert response.eligible is False
        assert response.ineligible_reason == "unit_priced_model"
        assert response.lease_id is None
        assert response.generation is None
        assert response.granted_tokens is None
        assert response.refresh_interval_s is None
        assert response.lease_length_s is None
        assert response.headroom_share_tokens is None
        assert response.posture is None
        assert response.final_grant is None

    def test_deny_response_carries_denied_by_period(self) -> None:
        payload = dict(_DISPLAY_ONLY, eligible=True, allowed=False, denied_by_period="agent_run")

        response = LeaseGrantResponse.model_validate(payload)

        assert response.allowed is False
        assert response.denied_by_period == "agent_run"

    def test_display_snapshot_fields_are_required(self) -> None:
        # The always-present snapshot must never silently default.
        for missing in ("project_id", "mode", "budget_limit", "current_usage", "remaining_budget"):
            payload = dict(_FULL_RESPONSE)
            del payload[missing]
            with pytest.raises(ValidationError):
                LeaseGrantResponse.model_validate(payload)

    def test_eligible_and_allowed_are_required(self) -> None:
        for missing in ("eligible", "allowed"):
            payload = dict(_FULL_RESPONSE)
            del payload[missing]
            with pytest.raises(ValidationError):
                LeaseGrantResponse.model_validate(payload)

    def test_rejects_lease_id_over_64_chars(self) -> None:
        # The RESPONSE bound mirrors the renew/surrender REQUEST bound: a
        # drifted server lease id must be rejected HERE, where the enforcer
        # already degrades to the legacy path, instead of detonating later
        # when the renewal or surrender request is built on a customer call.
        with pytest.raises(ValidationError):
            LeaseGrantResponse.model_validate(dict(_FULL_RESPONSE, lease_id="x" * 65))

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            LeaseGrantResponse.model_validate(dict(_FULL_RESPONSE, surprise=1))

    def test_posture_rejects_unknown_on_unreachable(self) -> None:
        with pytest.raises(ValidationError):
            LeasePosture.model_validate({"mode": "alert_only", "on_unreachable": "explode"})

    def test_posture_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            LeasePosture.model_validate(
                {"mode": "alert_only", "on_unreachable": "fail_open", "extra": 1}
            )


@pytest.mark.unit
class TestRunControlDirective:
    @pytest.mark.parametrize(
        "override",
        [
            {"version": "2"},
            {"action": "pause"},
            {"agent_run_id": "x" * (AGENT_RUN_ID_MAX_LENGTH + 1)},
            {"reason": "x" * 65},
            {"surprise": True},
        ],
    )
    def test_rejects_invalid_contract_shapes(self, override: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "version": "1",
            "action": "terminate",
            "agent_run_id": "run_1",
            "reason": "manual_kill",
        }
        payload.update(override)

        with pytest.raises(ValidationError):
            RunControlDirective.model_validate(payload)


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


@pytest.mark.unit
class TestBudgetConfirmRequestSettlementKey:
    """A confirm settles EITHER a reservation OR a lease — never both/neither."""

    def test_reservation_only_is_valid(self) -> None:
        request = _confirm()

        payload = request.model_dump(mode="json")

        assert payload["reservation_id"] == "res_123"
        assert "lease_id" not in payload

    def test_lease_only_is_valid(self) -> None:
        request = _confirm(reservation_id=None, lease_id="lse_abc")

        payload = request.model_dump(mode="json")

        assert payload["lease_id"] == "lse_abc"
        assert "reservation_id" not in payload

    def test_both_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _confirm(lease_id="lse_abc")

    def test_neither_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _confirm(reservation_id=None)
