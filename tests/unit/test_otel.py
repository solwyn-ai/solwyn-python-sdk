"""Optional trace capture and the content-free correlation wire contract."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from conftest import VALID_API_KEY, call_uuid
from pydantic import ValidationError

from solwyn._types import MetadataEvent, ProviderEntry, ProviderName
from solwyn.config import SolwynConfig

TRACE_ID = "00000000000000000000000000000abc"
SPAN_ID = "00000000000000de"

_INVALID_TRACE_FIELDS = [
    pytest.param({"trace_id": "0" * 32}, id="zero-trace"),
    pytest.param({"trace_id": TRACE_ID.upper()}, id="uppercase-trace"),
    pytest.param({"trace_id": "f" * 31}, id="short-trace"),
    pytest.param({"trace_id": "f" * 33}, id="long-trace"),
    pytest.param({"trace_id": "g" * 32}, id="nonhex-trace"),
    pytest.param({"trace_id": TRACE_ID + "\n"}, id="newline-trace"),
    pytest.param({"trace_id": "-" + "0" * 30 + "1"}, id="negative-trace"),
    pytest.param({"trace_id": None}, id="missing-trace"),
    pytest.param({"trace_id": 1}, id="integer-trace"),
    pytest.param({"span_id": "0" * 16}, id="zero-span"),
    pytest.param({"span_id": SPAN_ID.upper()}, id="uppercase-span"),
    pytest.param({"span_id": "f" * 15}, id="short-span"),
    pytest.param({"span_id": "f" * 17}, id="long-span"),
    pytest.param({"span_id": "g" * 16}, id="nonhex-span"),
    pytest.param({"span_id": SPAN_ID + "\n"}, id="newline-span"),
    pytest.param({"span_id": "-" + "0" * 14 + "1"}, id="negative-span"),
    pytest.param({"span_id": None}, id="missing-span"),
    pytest.param({"span_id": 1}, id="integer-span"),
    pytest.param({"trace_sampled": None}, id="missing-sampled"),
    pytest.param({"trace_sampled": 0}, id="zero-sampled"),
    pytest.param({"trace_sampled": 1}, id="one-sampled"),
    pytest.param({"trace_sampled": "true"}, id="string-sampled"),
    pytest.param({"trace_sampled": object()}, id="object-sampled"),
]


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
@pytest.mark.parametrize("spans_enabled", ["false", "true"])
def test_cold_import_does_not_load_opentelemetry(monkeypatch, spans_enabled) -> None:
    monkeypatch.setenv("SOLWYN_OTEL_SPANS", spans_enabled)
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
@pytest.mark.parametrize("resolution", ["missing", "broken", "available"])
def test_optional_api_resolution_is_cached_across_threads_and_idle_calls(resolution: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import importlib.abc
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

resolution = sys.argv[1]
imports = 0
lookups = 0

class OTelLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.trace = SimpleNamespace(
            get_current_span=lambda: SimpleNamespace(
                get_span_context=lambda: SimpleNamespace(is_valid=False)
            )
        )

class FindOTel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        global lookups
        if fullname == 'opentelemetry' or fullname.startswith('opentelemetry.'):
            lookups += 1
            if resolution == 'missing':
                raise ModuleNotFoundError(fullname)
            if resolution == 'broken':
                raise RuntimeError('optional API cannot initialize')
            return importlib.util.spec_from_loader(fullname, OTelLoader())

sys.meta_path.insert(0, FindOTel())
from solwyn._otel import capture_trace
original_import = builtins.__import__
def count_import(name, *args, **kwargs):
    global imports
    if name == 'opentelemetry':
        imports += 1
    return original_import(name, *args, **kwargs)
builtins.__import__ = count_import

barrier = Barrier(8)
def capture(_):
    barrier.wait(timeout=5)
    return capture_trace()
with ThreadPoolExecutor(max_workers=8) as executor:
    assert list(executor.map(capture, range(8))) == [None] * 8
assert imports == lookups == 1, (imports, lookups)

# A deterministic idle-overhead check: repeated calls never re-enter imports.
for _ in range(1000):
    assert capture_trace() is None
assert imports == lookups == 1, (imports, lookups)
""",
            resolution,
        ],
        check=True,
        timeout=15,
    )


@pytest.mark.unit
@pytest.mark.parametrize("fields", _INVALID_TRACE_FIELDS)
def test_capture_discards_malformed_context_values(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, object]
) -> None:
    from solwyn import _otel

    values = dict(trace_id=TRACE_ID, span_id=SPAN_ID, trace_sampled=True)
    values.update(fields)
    context = SimpleNamespace(
        is_valid=True,
        trace_id=values["trace_id"],
        span_id=values["span_id"],
        trace_flags=SimpleNamespace(sampled=values["trace_sampled"]),
    )
    api = SimpleNamespace(
        get_current_span=lambda: SimpleNamespace(get_span_context=lambda: context),
        format_trace_id=lambda value: value,
        format_span_id=lambda value: value,
    )
    monkeypatch.setattr(_otel, "_trace_api", api)
    monkeypatch.setattr(_otel, "_trace_api_resolved", True)
    assert _otel.capture_trace() is None


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
@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("fields", _INVALID_TRACE_FIELDS)
def test_builders_discard_malformed_snapshots(error: bool, fields: dict[str, object]) -> None:
    from solwyn._base import _SolwynBase
    from solwyn._otel import _TraceSnapshot

    base = _SolwynBase.__new__(_SolwynBase)
    base._solwyn_sdk_instance_id = "builder-trace"
    values = dict(trace_id=TRACE_ID, span_id=SPAN_ID, trace_sampled=True)
    values.update(fields)
    kwargs = dict(
        model="gpt-5.5",
        provider="openai",
        latency_ms=1,
        is_model_fallback=False,
        call_id=call_uuid("builder-trace"),
        trace=_TraceSnapshot(**values),
    )
    if error:
        event = base._build_error_event(**kwargs)
    else:
        event = base._build_metadata_event(
            **kwargs, input_tokens=1, output_tokens=2, token_details=None, status="success"
        )
    assert {"trace_id", "span_id", "trace_sampled"}.isdisjoint(event.model_dump())


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


@pytest.mark.unit
def test_optional_import_can_reenter_trace_capture_without_deadlock() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace
from solwyn import _otel

class Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        # Simulate an entry-point plugin calling a wrapped client during import.
        assert _otel.capture_trace() is None
        module.trace = SimpleNamespace(get_current_span=lambda: SimpleNamespace(
            get_span_context=lambda: SimpleNamespace(is_valid=False)
        ))
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'opentelemetry':
            return importlib.util.spec_from_loader(fullname, Loader())
sys.meta_path.insert(0, Finder())
assert _otel.capture_trace() is None
assert _otel._trace_api_resolved
assert not _otel._trace_api_loading
""",
        ],
        check=True,
        timeout=10,
    )


@pytest.mark.unit
@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="requires fork")
def test_optional_import_lock_is_reset_in_forked_child() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import signal
from threading import Event, Thread
from solwyn import _otel

held, release = Event(), Event()
def hold():
    with _otel._trace_api_lock:
        _otel._trace_api_loading = True
        held.set()
        release.wait(timeout=10)
thread = Thread(target=hold)
thread.start()
assert held.wait(timeout=5)
pid = os.fork()
if pid == 0:
    signal.alarm(3)
    assert _otel.capture_trace() is None
    assert _otel._trace_api_resolved
    os._exit(0)
release.set()
thread.join(timeout=5)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
""",
        ],
        check=True,
        timeout=15,
    )
