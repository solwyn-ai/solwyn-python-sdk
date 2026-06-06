"""Tests for agent_run_id / agent_run_name fields on MetadataEvent.

The API-side wire model (solwyn-shared) accepts both fields as optional.
The SDK's vendored MetadataEvent must mirror that shape so the cross-repo
contract-parity test in solwyn-ai/core stays green.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from solwyn._types import MetadataEvent, ProviderName


def _make_event(**overrides: object) -> MetadataEvent:
    """Create a MetadataEvent with sensible test defaults."""
    defaults: dict[str, object] = {
        "model": "gpt-4o",
        "provider": ProviderName.OPENAI,
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 200.0,
        "status": "success",
        "is_model_fallback": False,
        "sdk_instance_id": "test-instance-001",
        "timestamp": datetime.now(UTC),
        "call_id": "call_agent_run_event",
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestMetadataEventAgentRunFields:
    """The two new optional fields must be accepted and round-trip cleanly."""

    def test_defaults_to_none(self) -> None:
        # Arrange / Act
        event = _make_event()

        # Assert — backward-compat: events emitted outside a run scope
        # produce no agent_run_* in the payload (API auto-fallback engages).
        assert event.agent_run_id is None
        assert event.agent_run_name is None

    def test_accepts_explicit_values(self) -> None:
        # Arrange / Act
        event = _make_event(
            agent_run_id="run_K7qZ3xR1pNvL9wMs",
            agent_run_name="nightly-batch",
        )

        # Assert
        assert event.agent_run_id == "run_K7qZ3xR1pNvL9wMs"
        assert event.agent_run_name == "nightly-batch"

    def test_round_trip_through_json(self) -> None:
        # Arrange
        original = _make_event(
            agent_run_id="run_K7qZ3xR1pNvL9wMs",
            agent_run_name="nightly-batch",
        )

        # Act
        raw = original.model_dump_json()
        restored = MetadataEvent.model_validate_json(raw)

        # Assert
        assert restored.agent_run_id == "run_K7qZ3xR1pNvL9wMs"
        assert restored.agent_run_name == "nightly-batch"

    def test_default_serialization_omits_no_scope_agent_run_fields(self) -> None:
        # MetadataEvent owns the no-null wire invariant so future reporters do
        # not have to remember exclude_none=True at every call site.
        event = _make_event()

        payload = event.model_dump(mode="json")
        raw = event.model_dump_json()

        assert "agent_run_id" not in payload
        assert "agent_run_name" not in payload
        assert "agent_run_id" not in raw
        assert "agent_run_name" not in raw

    def test_no_scope_json_wire_shape_omits_agent_run_fields(self) -> None:
        # No-scope events rely on absent agent_run_* keys so the API's
        # server-side auto-run fallback can synthesize the denominator.
        event = _make_event()

        raw = event.model_dump_json(exclude_none=True)
        restored = MetadataEvent.model_validate_json(raw)

        assert "agent_run_id" not in raw
        assert "agent_run_name" not in raw
        assert restored.agent_run_id is None
        assert restored.agent_run_name is None

    def test_extra_field_still_forbidden(self) -> None:
        # Asserts extra="forbid" is preserved — typos in field names must
        # raise loudly rather than be silently dropped.
        with pytest.raises(ValidationError):
            _make_event(unexpected_field="oops")

    def test_agent_run_id_max_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(agent_run_id="x" * 256)

    def test_agent_run_name_max_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(agent_run_name="x" * 256)

    def test_exclude_none_drops_unset_fields(self) -> None:
        # The reporter sends payloads with exclude_none=True. Outside a run
        # scope, the API must receive a payload with NO agent_run_* keys so
        # its server-side auto-fallback can engage.
        event = _make_event()
        payload = event.model_dump(mode="json", exclude_none=True)
        assert "agent_run_id" not in payload
        assert "agent_run_name" not in payload

    def test_exclude_none_keeps_set_fields(self) -> None:
        event = _make_event(agent_run_id="run_abc", agent_run_name="batch")
        payload = event.model_dump(mode="json", exclude_none=True)
        assert payload["agent_run_id"] == "run_abc"
        assert payload["agent_run_name"] == "batch"
