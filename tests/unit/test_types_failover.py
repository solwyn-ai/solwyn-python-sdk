"""Failover-related additions to wire-format models in ``_types.py``.

Covers ProviderEntry, FailoverReason, the new MetadataEvent fields, and the
BudgetCheckRequest chain-hint alignment validator. All additive (P1 FOUNDATION).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from solwyn._types import (
    BudgetCheckRequest,
    CallStatus,
    FailoverReason,
    MetadataEvent,
    ProviderEntry,
    ProviderName,
)


@pytest.mark.unit
class TestProviderEntry:
    """ProviderEntry models a configured chain link — never carries creds."""

    def test_minimal_construction(self) -> None:
        entry = ProviderEntry(provider=ProviderName.OPENAI, model="gpt-4o")

        assert entry.provider is ProviderName.OPENAI
        assert entry.model == "gpt-4o"
        assert entry.default_params == {}

    def test_default_params_independent_per_instance(self) -> None:
        # default_factory must not share a single dict across instances.
        a = ProviderEntry(provider=ProviderName.OPENAI, model="gpt-4o")
        b = ProviderEntry(provider=ProviderName.ANTHROPIC, model="claude-x")

        a.default_params["temperature"] = 0.1

        assert b.default_params == {}

    def test_rejects_api_key_via_extra_forbid(self) -> None:
        # Decision D: ProviderEntry must NEVER accept a provider credential.
        with pytest.raises(ValidationError):
            ProviderEntry(
                provider=ProviderName.OPENAI,
                model="gpt-4o",
                api_key="sk-secret",  # type: ignore[call-arg]
            )

    def test_rejects_base_url_via_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            ProviderEntry(
                provider=ProviderName.OPENAI,
                model="gpt-4o",
                base_url="https://example.test",  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestFailoverReason:
    """FailoverReason enumerates why the router advanced past the primary."""

    def test_values(self) -> None:
        assert FailoverReason.CIRCUIT_OPEN == "circuit_open"
        assert FailoverReason.PRIMARY_ERROR == "primary_error"
        assert FailoverReason.MODEL_FALLBACK == "model_fallback"

    def test_member_set(self) -> None:
        assert {r.value for r in FailoverReason} == {
            "circuit_open",
            "primary_error",
            "model_fallback",
        }


def _metadata_event(**overrides: object) -> MetadataEvent:
    base: dict[str, object] = {
        "model": "gpt-4o",
        "provider": ProviderName.OPENAI,
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_ms": 12.0,
        "status": CallStatus.SUCCESS,
        "is_model_fallback": False,
        "sdk_instance_id": "sdk_abc",
        "timestamp": datetime(2026, 6, 2, tzinfo=UTC),
    }
    base.update(overrides)
    return MetadataEvent(**base)  # type: ignore[arg-type]


@pytest.mark.unit
class TestMetadataEventFailoverFields:
    """New provider-fallover telemetry fields are additive + None-skipped."""

    def test_new_fields_default(self) -> None:
        ev = _metadata_event()

        assert ev.is_provider_fallback is False
        assert ev.requested_provider is None
        assert ev.requested_model is None
        assert ev.failover_reason is None
        assert ev.failover_error_class is None
        assert ev.attempt_index == 0

    def test_none_valued_fields_skipped_in_dump(self) -> None:
        ev = _metadata_event()
        dumped = ev.model_dump()

        # None-skipping serializer drops the optional Nones entirely.
        assert "requested_provider" not in dumped
        assert "requested_model" not in dumped
        assert "failover_reason" not in dumped
        assert "failover_error_class" not in dumped
        # Non-None defaults remain on the wire.
        assert dumped["is_provider_fallback"] is False
        assert dumped["attempt_index"] == 0

    def test_populated_failover_fields_serialize(self) -> None:
        ev = _metadata_event(
            is_provider_fallback=True,
            requested_provider=ProviderName.OPENAI,
            requested_model="gpt-4o",
            failover_reason=FailoverReason.PRIMARY_ERROR,
            failover_error_class="APITimeoutError",
            attempt_index=1,
        )
        dumped = ev.model_dump()

        assert dumped["is_provider_fallback"] is True
        assert dumped["requested_provider"] == "openai"
        assert dumped["requested_model"] == "gpt-4o"
        assert dumped["failover_reason"] == "primary_error"
        assert dumped["failover_error_class"] == "APITimeoutError"
        assert dumped["attempt_index"] == 1

    def test_attempt_index_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            _metadata_event(attempt_index=-1)

    def test_error_class_length_bounded(self) -> None:
        with pytest.raises(ValidationError):
            _metadata_event(failover_error_class="x" * 65)

    def test_requested_model_length_bounded(self) -> None:
        with pytest.raises(ValidationError):
            _metadata_event(requested_model="m" * 101)


@pytest.mark.unit
class TestBudgetCheckRequestChainHints:
    """fallback_providers / fallback_models must align element-for-element."""

    def test_defaults_empty(self) -> None:
        req = BudgetCheckRequest(
            estimated_input_tokens=10,
            model="gpt-4o",
            provider=ProviderName.OPENAI,
        )

        assert req.fallback_providers == []
        assert req.fallback_models == []

    def test_aligned_hints_accepted(self) -> None:
        req = BudgetCheckRequest(
            estimated_input_tokens=10,
            model="gpt-4o",
            provider=ProviderName.OPENAI,
            fallback_providers=[ProviderName.ANTHROPIC],
            fallback_models=["claude-x"],
        )

        assert req.fallback_providers == [ProviderName.ANTHROPIC]
        assert req.fallback_models == ["claude-x"]

    def test_misaligned_lengths_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BudgetCheckRequest(
                estimated_input_tokens=10,
                model="gpt-4o",
                provider=ProviderName.OPENAI,
                fallback_providers=[ProviderName.ANTHROPIC],
                fallback_models=[],
            )

    def test_fallback_models_capped_at_eight(self) -> None:
        with pytest.raises(ValidationError):
            BudgetCheckRequest(
                estimated_input_tokens=10,
                model="gpt-4o",
                provider=ProviderName.OPENAI,
                fallback_providers=[ProviderName.ANTHROPIC] * 9,
                fallback_models=["claude-x"] * 9,
            )
