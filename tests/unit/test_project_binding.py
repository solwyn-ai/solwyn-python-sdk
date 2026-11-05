"""ProjectBinding: state transitions, the effect rule and the generation fence.

Spec section 6.2. These drive the sans-I/O binding directly; the end-to-end
fixtures F1 to F18 live in ``test_api_key_errors.py``.
"""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import VALID_API_KEY, VALID_PROJECT, VALID_PROJECT_ID, auth_key_body

from solwyn._api_key_errors import Refusal, classify
from solwyn._constants import PROJECT_HEADER
from solwyn._project import REFUSED_RETRY_S, UNVALIDATED_RETRY_S, ProjectBinding

pytestmark = pytest.mark.project_validation


class _Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


def _binding() -> tuple[ProjectBinding, _Clock]:
    binding = ProjectBinding(VALID_API_KEY, VALID_PROJECT)
    clock = _Clock()
    binding._clock = clock
    return binding, clock


def _validate(binding: ProjectBinding, **body: str) -> str:
    _headers, dispatched_at = binding.request_headers()
    outcome, _auth = binding.apply_auth_key(200, auth_key_body(**body), dispatched_at)
    return outcome


def _expired(posture: str | None) -> Refusal:
    return Refusal(code="api_key_expired", posture=posture or "unknown", expired_on="2026-11-01")  # type: ignore[arg-type]


@pytest.mark.unit
def test_initial_state_sends_the_selector_and_is_due_for_validation() -> None:
    binding, _clock = _binding()

    headers, dispatched_at = binding.request_headers()

    assert binding.state == "unvalidated"
    assert binding.ever_valid is False
    assert binding.pinned_id is None
    assert dispatched_at == 0
    assert headers == {
        "Authorization": f"Bearer {VALID_API_KEY}",
        PROJECT_HEADER: VALID_PROJECT,
        "Content-Type": "application/json",
    }
    assert binding.revalidation_due() is True


@pytest.mark.unit
def test_full_scope_200_pins_the_id_and_every_later_header_uses_it() -> None:
    binding, _clock = _binding()

    assert _validate(binding) == "valid"
    headers, dispatched_at = binding.request_headers(content_type=False)

    assert binding.state == "valid"
    assert binding.ever_valid is True
    assert binding.pinned_id == VALID_PROJECT_ID
    assert binding.last_auth is not None and binding.last_auth.project_id == VALID_PROJECT_ID
    assert dispatched_at == 1
    assert headers == {"Authorization": f"Bearer {VALID_API_KEY}", PROJECT_HEADER: VALID_PROJECT_ID}
    assert binding.revalidation_due() is False


@pytest.mark.unit
def test_read_only_200_is_a_blocking_refusal_and_never_sets_ever_valid() -> None:
    binding, _clock = _binding()

    assert _validate(binding, scope="read_only") == "refused"

    refusal = binding.refusal
    assert refusal is not None
    assert (refusal.code, refusal.effect, refusal.field) == ("read_only_key", "block", "api_key")
    assert binding.ever_valid is False
    assert binding.pinned_id is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "content"),
    [
        (500, b"oops"),
        (503, b""),
        (429, b'{"detail": "slow down"}'),
        (200, b"<html>"),
        (200, b'{"project_id": "proj_x"}'),
        (418, b"{}"),
        (403, json.dumps({"detail": {"code": "permission_denied"}}).encode()),
    ],
)
def test_unavailable_answers_keep_state_and_schedule_a_retry(status: int, content: bytes) -> None:
    binding, clock = _binding()
    _headers, dispatched_at = binding.request_headers()

    outcome, _ = binding.apply_auth_key(status, content, dispatched_at)

    assert outcome == "unavailable"
    assert binding.state == "unvalidated"
    assert binding.generation == 0
    assert binding.revalidation_due() is False
    clock.now += UNVALIDATED_RETRY_S
    assert binding.revalidation_due() is True


@pytest.mark.unit
def test_refused_state_retries_every_sixty_seconds() -> None:
    binding, clock = _binding()
    binding.refuse(Refusal(code="project_not_found"), binding.generation)

    assert binding.revalidation_due() is False
    clock.now += REFUSED_RETRY_S - 1
    assert binding.revalidation_due() is False
    clock.now += 1
    assert binding.revalidation_due() is True
    binding.validation_unavailable()
    assert binding.revalidation_due() is False


@pytest.mark.unit
def test_a_data_plane_401_schedules_an_immediate_revalidation() -> None:
    binding, clock = _binding()
    _validate(binding)

    binding.refuse(Refusal(code="api_key_expired", posture="alert_only"), binding.generation)

    assert binding.revalidation_due() is True


@pytest.mark.unit
def test_a_validation_answer_keeps_the_sixty_second_cadence() -> None:
    binding, clock = _binding()
    _headers, dispatched_at = binding.request_headers()

    binding.apply_auth_key(401, b'{"detail":{"code":"api_key_invalid"}}', dispatched_at)

    assert binding.revalidation_due() is False
    clock.now += REFUSED_RETRY_S
    assert binding.revalidation_due() is True


@pytest.mark.unit
def test_a_401_status_blocks_and_its_own_body_never_relaxes_it() -> None:
    binding, _clock = _binding()
    _validate(binding)
    request = httpx.Request("POST", "https://api.test.solwyn.ai/api/v1/budgets/check")
    _headers, dispatched_at = binding.request_headers()

    binding.observe_status(httpx.Response(401, request=request))
    assert binding.refusal is not None and binding.refusal.effect == "block"
    body = httpx.Response(
        401,
        request=request,
        content=b'{"detail":{"code":"api_key_expired","project_budget_mode":"alert_only"}}',
    )
    error = httpx.HTTPStatusError("refused", request=request, response=body)

    assert binding.note_refusal(error, dispatched_at)
    assert binding.refusal.effect == "block"


@pytest.mark.unit
def test_only_a_revalidation_dispatched_after_the_refusal_relaxes_it() -> None:
    binding, _clock = _binding()
    _validate(binding)
    request = httpx.Request("POST", "https://api.test.solwyn.ai/api/v1/budgets/check")
    binding.observe_status(httpx.Response(401, request=request))
    _headers, revalidation_dispatch = binding.request_headers()
    body = b'{"detail":{"code":"api_key_expired","project_budget_mode":"alert_only"}}'

    outcome, _ = binding.apply_auth_key(401, body, revalidation_dispatch)

    assert outcome == "refused"
    assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"


@pytest.mark.unit
def test_a_401_to_auth_key_does_not_block_at_its_status() -> None:
    binding, _clock = _binding()
    _validate(binding)
    binding.refuse(Refusal(code="api_key_expired", posture="alert_only"), binding.generation)
    request = httpx.Request("GET", "https://api.test.solwyn.ai/api/v1/auth/key")

    binding.observe_status(httpx.Response(401, request=request))

    assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("code", "posture", "ever_valid", "effect"),
    [
        ("api_key_expired", "alert_only", True, "allow_unmetered"),
        ("api_key_invalid", "alert_only", True, "allow_unmetered"),
        ("api_key_expired", "alert_only", False, "block"),
        ("api_key_expired", "hard_deny", True, "block"),
        ("api_key_expired", "unknown", True, "block"),
        ("read_only_key", "unknown", True, "block"),
        ("project_not_found", "unknown", True, "block"),
        ("key_not_dedicated", "unknown", True, "block"),
        ("project_selector_required", "unknown", True, "block"),
    ],
)
def test_effect_rule(code: str, posture: str, ever_valid: bool, effect: str) -> None:
    binding, _clock = _binding()
    if ever_valid:
        _validate(binding)

    binding.refuse(Refusal(code=code, posture=posture), binding.generation)  # type: ignore[arg-type]

    assert binding.refusal is not None
    assert binding.refusal.effect == effect


@pytest.mark.unit
def test_tightening_always_applies_even_from_an_old_dispatch() -> None:
    binding, _clock = _binding()
    _validate(binding)
    old = binding.generation
    binding.refuse(_expired("alert_only"), old)
    assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"

    changed = binding.refuse(_expired("hard_deny"), old)

    assert changed is True
    assert binding.refusal.effect == "block"


@pytest.mark.unit
def test_weakening_needs_a_dispatch_after_the_stored_refusal() -> None:
    binding, _clock = _binding()
    _validate(binding)
    stale = binding.generation
    binding.refuse(_expired("hard_deny"), stale)

    assert binding.refuse(_expired("alert_only"), stale) is False
    assert binding.refusal is not None and binding.refusal.effect == "block"

    _headers, fresh = binding.request_headers()
    assert fresh == binding.refusal_gen
    assert binding.refuse(_expired("alert_only"), fresh) is True
    assert binding.refusal.effect == "allow_unmetered"


@pytest.mark.unit
def test_an_identical_independent_refusal_keeps_the_state_but_advances_the_ordering() -> None:
    binding, _clock = _binding()
    binding.refuse(Refusal(code="api_key_invalid"), 0)
    stored = binding.refusal
    refusal_gen = binding.refusal_gen

    assert binding.refuse(Refusal(code="api_key_invalid"), 0) is False  # no new state
    assert binding.refusal == stored
    assert binding.refusal_gen > refusal_gen  # nothing dispatched before may relax it


@pytest.mark.unit
def test_a_success_dispatched_before_a_refusal_is_fenced() -> None:
    binding, _clock = _binding()
    _validate(binding)
    _headers, dispatched_at = binding.request_headers()
    binding.refuse(Refusal(code="api_key_invalid"), dispatched_at)

    outcome, _ = binding.apply_auth_key(200, auth_key_body(), dispatched_at)

    assert outcome == "fenced"
    assert binding.state == "refused"


@pytest.mark.unit
def test_a_fresh_success_clears_the_refusal_and_notifies_listeners() -> None:
    binding, _clock = _binding()
    calls: list[str] = []
    binding.add_listener(lambda: calls.append(binding.state))
    binding.refuse(Refusal(code="api_key_invalid"), 0)

    assert _validate(binding) == "valid"

    assert binding.state == "valid"
    assert binding.refusal is None
    assert calls == ["refused", "valid"]


@pytest.mark.unit
def test_admit_and_exit_verdicts() -> None:
    binding, _clock = _binding()
    g0 = binding.admit()
    assert g0 == 0
    assert binding.exit_verdict(0) == "keep"

    _validate(binding)
    assert binding.exit_verdict(0) == "rerun"

    binding.refuse(Refusal(code="api_key_invalid"), binding.generation)
    assert binding.exit_verdict(binding.generation) == "refused"
    stored = binding.admit()
    assert not isinstance(stored, int)
    assert stored.code == "api_key_invalid"


@pytest.mark.unit
def test_configuration_error_logs_once_per_code(caplog: pytest.LogCaptureFixture) -> None:
    binding, _clock = _binding()
    with caplog.at_level("ERROR"):
        binding.refuse(Refusal(code="project_not_found"), 0)
        binding.refuse(Refusal(code="api_key_invalid"), binding.generation)
        binding.refuse(Refusal(code="project_not_found"), binding.generation)

    assert caplog.text.count("solwyn.configuration_error.project_not_found") == 1
    assert caplog.text.count("solwyn.configuration_error.api_key_invalid") == 1


@pytest.mark.unit
def test_single_flight_joins_the_leader() -> None:
    binding, _clock = _binding()

    flight, leader = binding.begin_flight()
    joined, second_leader = binding.begin_flight()
    binding.end_flight(flight)
    after, third_leader = binding.begin_flight()

    assert (leader, second_leader, third_leader) == (True, False, True)
    assert joined is flight and flight.done.is_set()
    assert after is not flight


# ── the classifier (spec section 6.4) ─────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            401,
            {"detail": {"code": "api_key_expired", "project_budget_mode": "hard_deny"}},
            ("api_key_expired", "hard_deny"),
        ),
        (
            401,
            {"detail": {"code": "api_key_expired", "project_budget_mode": "alert_only"}},
            ("api_key_expired", "alert_only"),
        ),
        (
            401,
            {"detail": {"code": "api_key_invalid", "project_budget_mode": None}},
            ("api_key_invalid", "unknown"),
        ),
        (
            401,
            {"detail": {"code": "api_key_invalid", "project_budget_mode": "HARD_DENY"}},
            ("api_key_invalid", "unknown"),
        ),
        (401, {"detail": {"code": "api_key_not_allowed"}}, ("api_key_invalid", "unknown")),
        (401, {"detail": "Unrecognized token format"}, ("api_key_invalid", "unknown")),
        (401, [], ("api_key_invalid", "unknown")),
        (
            400,
            {"detail": {"code": "project_selector_required"}},
            ("project_selector_required", "unknown"),
        ),
        (
            400,
            {"detail": {"code": "project_selector_conflict", "reason": "url_mismatch"}},
            ("project_selector_conflict", "unknown"),
        ),
        (404, {"detail": {"code": "project_not_found"}}, ("project_not_found", "unknown")),
        (
            403,
            {"detail": {"code": "key_not_dedicated", "project_id": "proj_x"}},
            ("key_not_dedicated", "unknown"),
        ),
        (403, {"detail": {"code": "read_only_key"}}, ("read_only_key", "unknown")),
        (403, {"detail": {"code": "api_key_not_allowed"}}, None),
        (404, {"detail": "not found"}, None),
        (422, {"detail": {"code": "unknown_model"}}, None),
        (409, {"detail": {"code": "project_not_found"}}, None),
        (500, {"detail": {"code": "api_key_invalid"}}, None),
    ],
)
def test_classify(status: int, body: object, expected: tuple[str, str] | None) -> None:
    refusal = classify(status, json.dumps(body).encode())

    assert (None if refusal is None else (refusal.code, refusal.posture)) == expected


@pytest.mark.unit
def test_classify_reads_the_expiry_date() -> None:
    refusal = classify(
        401,
        json.dumps(
            {"detail": {"code": "api_key_expired", "expired_at": "2027-03-04T10:00:00Z"}}
        ).encode(),
    )

    assert refusal is not None and refusal.expired_on == "2027-03-04"
