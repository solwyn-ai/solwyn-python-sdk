"""``failover_error_class`` always reaches the wire in the API's exact shape.

The API validates the field against ``^[A-Za-z][A-Za-z0-9_.]*$`` (max 64) and
its ingest route validates the batch as ONE list, so a single event naming a
class like grpc's ``_InactiveRpcError`` used to 422 the whole batch — a terminal
rejection the reporter drops, other calls' spend events included. A class name
longer than 64 characters failed earlier still: the SDK's own model raised
inside the candidate walk's error handler, masking the provider's exception and
skipping the reservation's terminal step.

``_base._wire_error_class`` normalizes the NAME at the one place an event is
built. The wire model pins the API's pattern so drift fails here, and the walk
treats the receipt as best-effort so no receipt failure can mask the provider.

Seams are service boundaries only: the provider client is a stub, the control
plane is ``FakeControlPlane``, and the non-name construction failure is injected
at the provider-adapter seam. The real enforcer, lease ledger, breaker, reporter
queue, and wire models run.
"""

from __future__ import annotations

import random
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from conftest import VALID_API_KEY
from pydantic import ValidationError

import solwyn
from solwyn import _constants as wire_constants
from solwyn._base import _SolwynBase, _wire_error_class
from solwyn._registry import build_runtimes
from solwyn._types import CallStatus, MetadataEvent, ProviderEntry, ProviderName
from solwyn.config import SolwynConfig
from solwyn.testing import FakeControlPlane

# A LITERAL copy of core's ``IngestMetadataEvent.failover_error_class`` pin
# (``solwyn_shared/models.py``) — deliberately not imported from the SDK, so the
# checks below hold the SDK to the server's shape rather than to itself.
_SERVER_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")
_SERVER_MAX_LENGTH = 64

_LONG_NAME = ("GeneratedProviderTransportError" * 3)[:70]


def _accepted_by_server(value: str | None) -> bool:
    if value is None:
        return True  # None-skipped on the wire: the field is simply absent
    return len(value) <= _SERVER_MAX_LENGTH and _SERVER_PATTERN.fullmatch(value) is not None


# ---------------------------------------------------------------------------
# The sanitizer
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("KeyboardInterrupt", "KeyboardInterrupt", id="public-unchanged"),
        pytest.param("APITimeoutError", "APITimeoutError", id="provider-unchanged"),
        pytest.param(
            "botocore.exceptions.ClientError", "botocore.exceptions.ClientError", id="dots"
        ),
        pytest.param("_InactiveRpcError", "InactiveRpcError", id="grpc-private"),
        pytest.param("_MultiThreadedRendezvous", "MultiThreadedRendezvous", id="grpc-rendezvous"),
        pytest.param("__Mangled", "Mangled", id="dunder-prefix"),
        pytest.param("_9Lives", "Lives", id="digits-before-first-letter"),
        pytest.param("9", None, id="digit-only"),
        pytest.param("__", None, id="all-underscore"),
        pytest.param("", None, id="empty"),
        pytest.param("...", None, id="dots-only"),
        pytest.param(None, None, id="none-passes-through"),
        pytest.param(_LONG_NAME, _LONG_NAME[:64], id="70-chars-cut-to-64"),
        pytest.param("_" + "E" * 70, "E" * 64, id="strip-then-cut"),
        pytest.param("Erreur_Réseau", "Erreur_R_seau", id="non-ascii-letter-replaced"),
        pytest.param("Ошибка", None, id="no-ascii-letter"),
        pytest.param("ÉchecRéseau", "checR_seau", id="non-ascii-leading-stripped"),
        pytest.param("Bad Name-With:Odd/Chars", "Bad_Name_With_Odd_Chars", id="punctuation"),
        pytest.param("Generic[int]", "Generic_int_", id="brackets"),
        pytest.param("Trailing\n", "Trailing_", id="trailing-newline"),
        pytest.param("<locals>.Inner", "locals_.Inner", id="local-qualname"),
    ],
)
def test_wire_error_class_table(raw: str | None, expected: str | None) -> None:
    # Act
    result = _wire_error_class(raw)

    # Assert
    assert result == expected
    assert _accepted_by_server(result)
    # Deterministic and idempotent: a normalized name is a fixed point.
    assert _wire_error_class(raw) == result
    assert _wire_error_class(result) == result


@pytest.mark.unit
def test_wire_error_class_output_is_always_none_or_server_valid() -> None:
    # Arrange: a seeded corpus over an alphabet that includes every class of
    # character the pattern treats differently, plus pathological lengths.
    alphabet = "aZ09_.-:/ \t\n\r\x00\x7f[]<>'\"\\éÖßЖ中\u200b\U0001f600"
    rng = random.Random(0xE44C1A55)
    corpus: list[Any] = [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 200))) for _ in range(5000)
    ]
    corpus += ["_" * 100_000 + "Tail", "x" * 100_000, "_" * 100_000, "\n", " ", "é"]
    # Totality covers a mistyped caller too: a non-string is dropped, not raised.
    corpus += [123, b"BytesError", object(), ["ListError"]]

    for raw in corpus:
        # Act
        result = _wire_error_class(raw)

        # Assert
        assert _accepted_by_server(result), repr(raw)[:80]
        if result is not None:
            # What survives is accepted by the SDK's own wire model as well.
            assert _wire_event(failover_error_class=result).failover_error_class == result


@pytest.mark.unit
def test_wire_error_class_bounds_mirror_the_server() -> None:
    assert wire_constants.FAILOVER_ERROR_CLASS_MAX_LENGTH == _SERVER_MAX_LENGTH
    assert wire_constants.FAILOVER_ERROR_CLASS_PATTERN == r"^[A-Za-z][A-Za-z0-9_.]*$"
    assert f"^{_SERVER_PATTERN.pattern}$" == r"^[A-Za-z][A-Za-z0-9_.]*$"


# ---------------------------------------------------------------------------
# The wire model pins the server's shape; the builder is the choke point
# ---------------------------------------------------------------------------


def _make_base() -> _SolwynBase:
    runtimes = build_runtimes(_openai_stub(_SyncCompletions)(None), "gpt-5.5", [])
    config = SolwynConfig(
        api_key=VALID_API_KEY,
        providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
    )
    return _SolwynBase(config, runtimes)


def _wire_event(**overrides: Any) -> MetadataEvent:
    fields: dict[str, Any] = {
        "model": "gpt-5.5",
        "provider": "openai",
        "input_tokens": 0,
        "output_tokens": 0,
        "token_details": None,
        "latency_ms": 1.0,
        "status": "error",
        "is_model_fallback": False,
        "call_id": "3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
        "sdk_instance_id": "sdk-instance",
        "timestamp": "2026-09-20T00:00:00Z",
    }
    fields.update(overrides)
    return MetadataEvent.model_validate(fields)


@pytest.mark.unit
@pytest.mark.parametrize(
    "rejected",
    ["_InactiveRpcError", "9Lives", "", "Bad-Name", "Trailing\n", "Réseau", "E" * 65],
)
def test_wire_model_rejects_what_the_server_rejects(rejected: str) -> None:
    # The model is the drift tripwire, exactly as strict as the API.
    with pytest.raises(ValidationError):
        _wire_event(failover_error_class=rejected)


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["_InactiveRpcError", _LONG_NAME, "__", "Bad Name", None])
def test_event_builders_normalize_before_construction(raw: str | None) -> None:
    # Arrange
    base = _make_base()
    common: dict[str, Any] = {
        "model": "gpt-5.5",
        "provider": "openai",
        "latency_ms": 1.0,
        "is_model_fallback": False,
        "call_id": "3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
        "failover_error_class": raw,
    }

    # Act: neither builder may raise, whatever the class was called.
    error_event = base._build_error_event(**common)
    metadata_event = base._build_metadata_event(
        **common, input_tokens=0, output_tokens=0, token_details=None, status=CallStatus.ERROR
    )

    # Assert
    for event in (error_event, metadata_event):
        assert event.failover_error_class == _wire_error_class(raw)
        assert _accepted_by_server(event.failover_error_class)


# ---------------------------------------------------------------------------
# End to end: provider stubs (openai-shaped; detection is duck-typed)
# ---------------------------------------------------------------------------


class _InactiveRpcError(Exception):
    """Named exactly like grpc's private error, reachable under legacy google."""


_LongNamedError = type(_LONG_NAME, (Exception,), {})


def _response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _SyncCompletions:
    def __init__(self, error: Exception | None) -> None:
        self.calls = 0
        self.error = error

    def create(self, **_kwargs: object) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _response()


class _AsyncCompletions(_SyncCompletions):
    async def create(self, **_kwargs: object) -> Any:  # type: ignore[override]
        return _SyncCompletions.create(self)


def _openai_stub(completions: type[_SyncCompletions]) -> Any:
    is_async = completions is _AsyncCompletions

    class _Stub:
        def __init__(self, error: Exception | None) -> None:
            self.chat = SimpleNamespace(completions=completions(error))
            # Media surfaces dispatch through the same stubbed seam.
            self.embeddings = completions(error)

        def with_options(self, **_kwargs: object) -> Any:
            return self

    _Stub.__module__ = "openai._client"
    _Stub.__name__ = "AsyncOpenAI" if is_async else "OpenAI"
    return _Stub


_REQUEST: dict[str, Any] = {"model": "gpt-5.5", "messages": [], "max_completion_tokens": 20}


def _plane() -> FakeControlPlane:
    return FakeControlPlane(granted_tokens=20, headroom_share_tokens=0, final_grant=True)


def _wrap(plane: FakeControlPlane, mode: str, error: Exception) -> tuple[Any, Any, Any]:
    stub = _openai_stub(_AsyncCompletions if mode == "async" else _SyncCompletions)
    primary, fallback = stub(error), stub(None)
    wrap = plane.wrap_async if mode == "async" else plane.wrap
    return wrap(primary, fallback=[(fallback, "gpt-5.5-mini")]), primary, fallback


async def _call(wrapped: Any, mode: str) -> None:
    result = wrapped.chat.completions.create(**_REQUEST)
    if mode == "async":
        await result


async def _close(wrapped: Any, mode: str) -> None:
    if mode == "async":
        await wrapped.close()
    else:
        wrapped.close()


def _assert_reservation_ended(wrapped: Any, run_id: str, *, bound_retained: bool) -> str:
    """The claim is gone NOW — nothing is left for the 900s sweep to refund."""
    ledger = wrapped._solwyn_budget._lease
    state = ledger.state_for(run_id)
    assert state is not None
    assert state.reservations == {}
    assert state.reserved_tokens == 0
    # An ambiguous abort keeps its spent bound; a fail-fast refusal is refunded.
    assert state.granted_remaining_tokens == (0 if bound_retained else 20)
    assert state.spent_tokens_since_report == (20 if bound_retained else 0)
    call_id = next(iter(ledger._call_claims))
    assert call_id not in ledger._call_index
    return str(call_id)


_DISPOSITIONS = [
    pytest.param(503, True, id="post-send-ambiguous"),
    pytest.param(400, False, id="fail-fast"),
]
_ERROR_TYPES = [
    pytest.param(_InactiveRpcError, "InactiveRpcError", id="underscore-prefixed"),
    pytest.param(_LongNamedError, _LONG_NAME[:64], id="70-char-name"),
]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(("status_code", "ambiguous"), _DISPOSITIONS)
@pytest.mark.parametrize(("error_type", "reported"), _ERROR_TYPES)
async def test_unrepresentable_class_name_never_masks_the_provider_error(
    mode: str,
    status_code: int,
    ambiguous: bool,
    error_type: type[Exception],
    reported: str,
) -> None:
    # Arrange
    plane = _plane()
    error = error_type("synthetic dispatch failure")
    error.status_code = status_code  # type: ignore[attr-defined]
    wrapped, primary, fallback = _wrap(plane, mode, error)

    try:
        with solwyn.run("error-class-wire") as run_id:
            # Act
            with pytest.raises(error_type) as raised:
                await _call(wrapped, mode)

            # Assert: the ORIGINAL exception, and the reservation's terminal step.
            assert raised.value is error
            assert primary.chat.completions.calls == 1
            assert fallback.chat.completions.calls == 0
            call_id = _assert_reservation_ended(wrapped, run_id, bound_retained=ambiguous)
    finally:
        await _close(wrapped, mode)

    events = [event for event in plane.ingested if event.call_id == call_id]
    assert len(events) == 1, plane.ingested
    assert events[0].status == "error"
    assert events[0].possibly_succeeded is (True if ambiguous else None)
    assert events[0].failover_error_class == reported
    assert type(error).__name__ != reported
    assert _accepted_by_server(events[0].failover_error_class)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_private_class_error_cannot_cost_another_call_its_spend_event(mode: str) -> None:
    # Arrange: one call fails with a private class, the next one is paid. The
    # API rejects a batch as a whole, so every event in it must be acceptable.
    plane = _plane()
    error = _InactiveRpcError("synthetic dispatch failure")
    error.status_code = 400  # type: ignore[attr-defined]
    wrapped, primary, _fallback = _wrap(plane, mode, error)

    try:
        with solwyn.run("error-class-batch"):
            # Act
            with pytest.raises(_InactiveRpcError):
                await _call(wrapped, mode)
            primary.chat.completions.error = None
            await _call(wrapped, mode)
    finally:
        await _close(wrapped, mode)

    # Assert
    assert sorted(event.status for event in plane.ingested) == ["error", "success"]
    assert all(_accepted_by_server(event.failover_error_class) for event in plane.ingested)
    success = next(event for event in plane.ingested if event.status == "success")
    assert (success.input_tokens, success.output_tokens) == (10, 5)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(("status_code", "ambiguous"), _DISPOSITIONS)
async def test_receipt_failure_for_another_reason_never_masks_the_provider_error(
    mode: str, status_code: int, ambiguous: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange: the error receipt cannot be built for a reason that has nothing to
    # do with the class name — the served adapter reports a region the wire
    # model's 32-char bound rejects.
    plane = _plane()
    error = RuntimeError("synthetic dispatch failure")
    error.status_code = status_code  # type: ignore[attr-defined]
    wrapped, primary, fallback = _wrap(plane, mode, error)
    adapter = wrapped._solwyn_runtimes[0].adapter

    try:
        with solwyn.run("error-receipt-unbuildable") as run_id:
            # Act
            with (
                caplog.at_level("WARNING", logger="solwyn.client"),
                patch.object(adapter, "extract_region", return_value="r" * 33),
                pytest.raises(RuntimeError) as raised,
            ):
                await _call(wrapped, mode)

            # Assert: still the ORIGINAL exception and the same terminal step.
            assert raised.value is error
            assert primary.chat.completions.calls == 1
            assert fallback.chat.completions.calls == 0
            call_id = _assert_reservation_ended(wrapped, run_id, bound_retained=ambiguous)
    finally:
        await _close(wrapped, mode)

    # The receipt is what was lost, and the loss names a class — never content.
    assert [event for event in plane.ingested if event.call_id == call_id] == []
    warnings = [r.getMessage() for r in caplog.records if "error_receipt_failed" in r.message]
    assert warnings == ["call.error_receipt_failed: ValidationError"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("unbuildable_receipt", [False, True])
async def test_media_dispatch_error_keeps_its_identity(
    mode: str, unbuildable_receipt: bool
) -> None:
    # Arrange: the primary-only media path has its own error handler. The class
    # is private AND, in the second case, the receipt cannot be built at all.
    plane = _plane()
    error = _InactiveRpcError("synthetic dispatch failure")
    wrapped, primary, _fallback = _wrap(plane, mode, error)
    adapter = wrapped._solwyn_runtimes[0].adapter
    region = "r" * 33 if unbuildable_receipt else None

    try:
        # Act
        with (
            patch.object(adapter, "extract_region", return_value=region),
            pytest.raises(_InactiveRpcError) as raised,
        ):
            result = wrapped.embeddings.create(model="text-embedding-3-small", input="x")
            if mode == "async":
                await result

        # Assert
        assert raised.value is error
        assert primary.embeddings.calls == 1
    finally:
        await _close(wrapped, mode)

    classes = [event.failover_error_class for event in plane.ingested]
    assert classes == ([] if unbuildable_receipt else ["InactiveRpcError"])
