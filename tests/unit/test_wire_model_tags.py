"""Tests for bounded explicit customer tags on ``MetadataEvent``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from solwyn._types import MetadataEvent, ProviderName


def _make_event(**overrides: object) -> MetadataEvent:
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
        "call_id": "call_tags_event",
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestMetadataEventTags:
    def test_defaults_to_none_and_omits_key_from_wire(self) -> None:
        event = _make_event()

        assert event.tags is None
        assert "tags" not in event.model_dump(mode="json")
        assert '"tags":' not in event.model_dump_json()

    def test_plain_mapping_round_trips_at_inclusive_bounds(self) -> None:
        tags = {"team": "research", "empty-value": "", "k" * 64: "v" * 256}
        event = _make_event(tags=tags)

        dumped = event.model_dump(mode="json")
        restored = MetadataEvent.model_validate_json(event.model_dump_json())

        assert dumped["tags"] == tags
        assert restored.tags == tags

    @pytest.mark.parametrize(
        "tags",
        [
            {f"key-{index}": "value" for index in range(11)},
            {"": "value"},
            {"k" * 65: "value"},
            {"key": "v" * 257},
            {1: "value"},
            {"key": 1},
        ],
        ids=[
            "more-than-10-keys",
            "empty-key",
            "key-over-64",
            "value-over-256",
            "non-string-key",
            "non-string-value",
        ],
    )
    def test_rejects_invalid_mapping(self, tags: dict[object, object]) -> None:
        with pytest.raises(ValidationError):
            _make_event(tags=tags)
