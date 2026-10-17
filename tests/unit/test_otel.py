"""Optional trace capture and the content-free correlation wire contract."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from conftest import VALID_API_KEY, call_uuid
from pydantic import ValidationError

from solwyn._types import MetadataEvent, ProviderEntry, ProviderName
from solwyn.config import SolwynConfig

TRACE_ID = "00000000000000000000000000000abc"
SPAN_ID = "00000000000000de"


def _event(**overrides: object) -> MetadataEvent:
    values: dict[str, object] = {
        "model": "gpt-5.5",
        "provider": "openai",
        "input_tokens": 3,
        "output_tokens": 2,
        "latency_ms": 1.0,
        "status": "success",
        "is_model_fallback": False,
        "call_id": call_uuid("trace-wire"),
        "sdk_instance_id": "trace-test",
        "timestamp": datetime(2026, 10, 16, tzinfo=UTC),
    }
    values.update(overrides)
    return MetadataEvent(**values)


@pytest.mark.unit
def test_cold_import_does_not_load_opentelemetry() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import solwyn; "
            "assert not any(n.startswith('opentelemetry') for n in sys.modules)",
        ],
        check=True,
    )


@pytest.mark.unit
def test_capture_without_optional_package() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class BlockOTel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'opentelemetry' or fullname.startswith('opentelemetry.'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOTel())
from solwyn._otel import capture_trace
assert capture_trace() is None
""",
        ],
        check=True,
    )


@pytest.mark.unit
def test_trace_wire_payloads_are_exact_and_none_skipped() -> None:
    plain = _event().model_dump(mode="json")
    assert set(plain) == {
        "model",
        "provider",
        "modality",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "status",
        "is_model_fallback",
        "is_provider_fallback",
        "attempt_index",
        "call_id",
        "sdk_instance_id",
        "timestamp",
    }
    traced = _event(trace_id=TRACE_ID, span_id=SPAN_ID, trace_sampled=False)
    assert traced.model_dump(mode="json") == {
        **plain,
        "trace_id": TRACE_ID,
        "span_id": SPAN_ID,
        "trace_sampled": False,
    }
    assert _event(trace_id=None, span_id=None, trace_sampled=None).model_dump_json() == (
        _event().model_dump_json()
    )
    assert json.loads(traced.model_dump_json()) == traced.model_dump(mode="json")


@pytest.mark.unit
@pytest.mark.parametrize(
    "fields",
    [
        {"trace_id": "0" * 32},
        {"trace_id": TRACE_ID.upper()},
        {"trace_id": "f" * 31},
        {"trace_id": "f" * 33},
        {"trace_id": "g" * 32},
        {"trace_id": TRACE_ID + "\n"},
        {"trace_id": TRACE_ID, "span_id": "0" * 16},
        {"trace_id": TRACE_ID, "span_id": SPAN_ID.upper()},
        {"trace_id": TRACE_ID, "span_id": "f" * 15},
        {"trace_id": TRACE_ID, "span_id": "f" * 17},
        {"span_id": SPAN_ID},
        {"trace_sampled": False},
        {"trace_sampled": True},
    ],
)
def test_trace_validator_rejects_malformed_or_orphaned_fields(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _event(**fields)


@pytest.mark.unit
def test_trace_id_can_stand_alone() -> None:
    assert _event(trace_id=TRACE_ID).trace_id == TRACE_ID


@pytest.mark.unit
def test_constants_pin_trace_and_span_wire_widths() -> None:
    from solwyn import _constants

    assert _constants.TRACE_ID_HEX_LENGTH == 32
    assert _constants.SPAN_ID_HEX_LENGTH == 16
    assert _constants.TRACE_ID_PATTERN == r"^[0-9a-f]{32}$"
    assert _constants.SPAN_ID_PATTERN == r"^[0-9a-f]{16}$"


@pytest.mark.unit
@pytest.mark.parametrize("error", [False, True])
def test_builders_only_use_explicit_snapshots(error: bool) -> None:
    from solwyn._base import _SolwynBase
    from solwyn._otel import _TraceSnapshot

    base = _SolwynBase.__new__(_SolwynBase)
    base._solwyn_sdk_instance_id = "builder-trace"
    kwargs = dict(
        model="gpt-5.5",
        provider="openai",
        latency_ms=1,
        is_model_fallback=False,
        call_id=call_uuid("builder-trace"),
    )
    if error:
        builder = base._build_error_event
    else:
        builder = base._build_metadata_event
        kwargs.update(input_tokens=1, output_tokens=2, token_details=None, status="success")
    snapshot = _TraceSnapshot(TRACE_ID, SPAN_ID, False)
    with patch("solwyn._otel.capture_trace", return_value=snapshot) as capture:
        plain = builder(**kwargs)
        traced = builder(**kwargs, trace=snapshot)
        capture.assert_not_called()
    assert plain.trace_id is plain.span_id is plain.trace_sampled is None
    assert (traced.trace_id, traced.span_id, traced.trace_sampled) == snapshot


@pytest.mark.unit
@pytest.mark.parametrize("value, expected", [("false", False), ("0", False), ("yes", True)])
def test_trace_config_environment_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("SOLWYN_TRACE_CORRELATION", value)
    assert (
        SolwynConfig(
            api_key=VALID_API_KEY,
            providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
        ).trace_correlation
        is expected
    )
    assert SolwynConfig(
        api_key=VALID_API_KEY,
        providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
        trace_correlation=not expected,
    ).trace_correlation is (not expected)


@pytest.mark.unit
def test_trace_correlation_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOLWYN_TRACE_CORRELATION", raising=False)
    assert (
        SolwynConfig(
            api_key=VALID_API_KEY,
            providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
        ).trace_correlation
        is True
    )
