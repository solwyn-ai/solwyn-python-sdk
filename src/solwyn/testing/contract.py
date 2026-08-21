"""Behavior-bearing control-plane wire contract shared by fake and live lanes.

The functions in this module deliberately use explicit ``AssertionError``
checks instead of Python assertion statements.  Production modules are
importable under ``python -O`` and the repository forbids assertion statements
under ``src/solwyn``.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    CallStatus,
    DenySource,
    LeaseGrantRequest,
    LeaseGrantResponse,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    MetadataEvent,
    ProviderName,
)

_CHECK_PATH = "/api/v1/budgets/check"
_CONFIRM_PATH = "/api/v1/budgets/confirm"
_INGEST_PATH = "/api/v1/metadata/ingest"
_LEASE_PATH = "/api/v1/budgets/lease"
_LEASE_RENEW_PATH = "/api/v1/budgets/lease/renew"
_LEASE_SURRENDER_PATH = "/api/v1/budgets/lease/surrender"

_LEASE_BLOCK_KEYS = frozenset(
    {
        "lease_id",
        "generation",
        "granted_tokens",
        "refresh_interval_s",
        "lease_length_s",
        "headroom_share_tokens",
        "posture",
        "final_grant",
    }
)
_DISPLAY_KEYS = frozenset(
    {
        "project_id",
        "mode",
        "budget_limit",
        "current_usage",
        "remaining_budget",
    }
)
_RUN_CONTROL_KEYS = frozenset({"version", "action", "agent_run_id", "reason"})
_RUN_CONTROL_REASON_MAX_LENGTH = 64
_UNPRICED_LEASE_MODEL = "no-such-model-for-leases"
_BODY_PREVIEW_LIMIT = 200
# BudgetCheckResponse.price_hints is a strict dict[ProviderName, float]; a key
# outside this set makes the WHOLE check response unreadable to the SDK.
_KNOWN_PROVIDER_NAMES = frozenset(provider.value for provider in ProviderName)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _body_preview(response: httpx.Response) -> str:
    raw = response.content
    preview = raw[:_BODY_PREVIEW_LIMIT].decode("utf-8", errors="replace")
    suffix = "..." if len(raw) > _BODY_PREVIEW_LIMIT else ""
    return f"{preview!r}{suffix}"


def _require_status(response: httpx.Response, status: int, context: str) -> None:
    if response.status_code != status:
        raise AssertionError(
            f"{context} returned status {response.status_code}, expected {status}; "
            f"body preview: {_body_preview(response)}"
        )


def _object_json(response: httpx.Response, context: str) -> dict[str, Any]:
    try:
        body = response.json()
    except (UnicodeDecodeError, ValueError) as exc:
        raise AssertionError(
            f"{context} returned malformed JSON; body preview: {_body_preview(response)}"
        ) from exc
    _require(
        isinstance(body, dict),
        f"{context} returned non-object JSON; body preview: {_body_preview(response)}",
    )
    return cast(dict[str, Any], body)


def _validate_model(
    model: type[_ModelT],
    payload: dict[str, Any],
    context: str,
) -> _ModelT:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        raise AssertionError(f"{context} schema validation failed: {errors!r}") from exc


def _post(
    http: httpx.Client,
    api_key: str,
    path: str,
    payload: dict[str, Any],
) -> httpx.Response:
    return http.post(path, json=payload, headers=_headers(api_key))


def _check_payload(
    *,
    agent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
    run_directive_version: Literal["1"] | None = None,
) -> dict[str, Any]:
    """Build the probe check exactly as the SDK builds a production one.

    ``price_hints_version`` is as load-bearing as the directive opt-ins: the
    server states nothing about pricing without it, so an unopted probe could
    never observe — let alone validate — the hint map every SDK check consumes.
    ``run_directive_version`` stays a parameter because the pack asserts both
    the opted-in and the legacy run-control shapes.
    """
    return BudgetCheckRequest(
        estimated_input_tokens=1000,
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        agent_run_id=agent_run_id,
        tags=tags,
        failover_directive_version="1",
        run_directive_version=run_directive_version,
        price_hints_version="1",
    ).model_dump(mode="json")


def _require_numeric(value: object, label: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be a JSON number, got {value!r}",
    )


def _require_finite_number(value: object, label: str) -> None:
    _require_numeric(value, label)
    _require(
        math.isfinite(cast(float, value)),
        f"{label} must be a finite JSON number, got {value!r}",
    )


def _require_integer(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be a JSON integer, got {value!r}",
    )
    return cast(int, value)


def _require_display_shape(payload: dict[str, Any], context: str) -> None:
    missing = _DISPLAY_KEYS.difference(payload)
    _require(not missing, f"{context} omitted display keys: {sorted(missing)}")
    _require(isinstance(payload["project_id"], str), f"{context} project_id must be a string")
    _require(
        payload["mode"] in {"alert_only", "hard_deny"},
        f"{context} returned unknown mode: {payload['mode']!r}",
    )
    for key in ("budget_limit", "current_usage", "remaining_budget"):
        _require_numeric(payload[key], f"{context}.{key}")


def _require_directive_shape(payload: dict[str, Any], context: str) -> None:
    raw_directive = payload.get("failover_directive")
    _require(
        isinstance(raw_directive, dict),
        f"{context} omitted directive-v1 block: {payload!r}",
    )
    directive = cast(dict[str, Any], raw_directive)
    _require(directive.get("version") == "1", f"{context} directive version drifted")
    _require(
        isinstance(directive.get("failover_tuning_allowed"), bool),
        f"{context} directive entitlement must be a JSON boolean",
    )
    _require_price_hint_shape(payload, context)


def _require_price_hint_shape(payload: dict[str, Any], context: str) -> None:
    """Assert the served hint map is one the SDK can actually read.

    Every check opts into ``price_hints_version="1"``, so the server answers
    with a populated provider->hint map, an explicit ``{}`` (nothing priceable
    for this modality or primary), or the null statement — which the
    directive-v1 wire expresses by omitting the key. Null and absence are the
    same statement and both pass.

    A present map is strict: ``BudgetCheckResponse.price_hints`` is typed
    ``dict[ProviderName, float]``, so one unknown provider key fails validation
    for the ENTIRE response and the SDK fails open with no reservation. Name
    the offending key here rather than leaving it as schema-validation noise.
    """
    hints = payload.get("price_hints")
    if hints is None:
        return
    _require(
        isinstance(hints, dict),
        f"{context} price_hints must be an object or null",
    )
    for provider, hint in cast(dict[Any, Any], hints).items():
        _require(isinstance(provider, str), f"{context} price hint provider must be a string")
        _require(
            provider in _KNOWN_PROVIDER_NAMES,
            f"{context} price hint provider {provider!r} is not a known ProviderName; "
            "the SDK cannot parse this check response",
        )
        _require_finite_number(hint, f"{context}.price_hints[{provider!r}]")


def _post_check(
    http: httpx.Client,
    api_key: str,
    *,
    context: str,
    agent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
    run_directive_version: Literal["1"] | None = None,
) -> dict[str, Any]:
    response = _post(
        http,
        api_key,
        _CHECK_PATH,
        _check_payload(
            agent_run_id=agent_run_id,
            tags=tags,
            run_directive_version=run_directive_version,
        ),
    )
    _require_status(response, 200, context)
    return _object_json(response, context)


def _require_deny_shape(payload: dict[str, Any], period: str) -> None:
    context = f"{period} denial"
    _require(payload.get("allowed") is False, f"{context} was not denied: {payload!r}")
    _require(
        payload.get("denied_by_period") == period,
        f"{context} literal drifted: {payload!r}",
    )
    _require(payload.get("mode") == "hard_deny", f"{context} must use hard_deny mode")
    _require("reservation_id" not in payload, f"{context} leaked reservation_id")
    _require_display_shape(payload, context)
    _require_directive_shape(payload, context)
    parsed = _validate_model(BudgetCheckResponse, payload, context)
    _require(parsed.denied_by_period == period, f"{context} failed SDK response validation")


def assert_check_contract(http: httpx.Client, api_key: str) -> None:
    """Assert the check response shapes that change SDK behavior."""
    monthly = _post_check(
        http,
        api_key,
        context="monthly denial",
        tags={"contract_case": "monthly"},
    )
    _require_deny_shape(monthly, "monthly")

    stopped = _post_check(
        http,
        api_key,
        context="run_stopped denial",
        agent_run_id="contract-stopped-run",
    )
    _require_deny_shape(stopped, "run_stopped")

    tag = _post_check(
        http,
        api_key,
        context="tag denial",
        tags={"customer": "acme"},
    )
    _require_deny_shape(tag, "tag")

    agent_run = _post_check(
        http,
        api_key,
        context="agent_run denial",
        agent_run_id="contract-agent-run",
    )
    _require_deny_shape(agent_run, "agent_run")

    allowed = _post_check(http, api_key, context="allow check")
    _require(allowed.get("allowed") is True, f"allow check was denied: {allowed!r}")
    _require("denied_by_period" not in allowed, "allow check serialized denied_by_period")
    reservation_id = allowed.get("reservation_id")
    _require(
        isinstance(reservation_id, str) and bool(reservation_id),
        f"allow check omitted reservation_id: {allowed!r}",
    )
    _require_display_shape(allowed, "allow check")
    _require_directive_shape(allowed, "allow check")
    parsed = _validate_model(BudgetCheckResponse, allowed, "allow check")
    _require(parsed.allowed is True, "allow check failed SDK response validation")
    _require(parsed.denied_by_period is None, "allow parsed a denied period")


def _reservation(http: httpx.Client, api_key: str) -> str:
    payload = _post_check(http, api_key, context="confirm setup check")
    _require(payload.get("allowed") is True, "confirm setup check was denied")
    reservation_id = payload.get("reservation_id")
    _require(isinstance(reservation_id, str), "confirm setup omitted reservation_id")
    return cast(str, reservation_id)


def _confirm_payload(
    *,
    call_id: str,
    reservation_id: str | None = None,
    lease_id: str | None = None,
) -> dict[str, Any]:
    request = BudgetConfirmRequest(
        reservation_id=reservation_id,
        lease_id=lease_id,
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        call_id=call_id,
        token_details=TokenDetails(input_tokens=10, output_tokens=5),
    )
    return request.model_dump(mode="json")


def _raw_confirm_base() -> dict[str, Any]:
    return {
        "model": "gpt-5.5",
        "provider": "openai",
        "call_id": str(uuid.uuid4()),
        "token_details": {"input_tokens": 10, "output_tokens": 5},
    }


def assert_confirm_contract(http: httpx.Client, api_key: str) -> None:
    """Assert valid, invalid, and replay-safe settlement wire shapes."""
    reservation_id = _reservation(http, api_key)
    call_id = str(uuid.uuid4())
    payload = _confirm_payload(call_id=call_id, reservation_id=reservation_id)
    _require("lease_id" not in payload, "reservation confirm serialized lease_id")
    _require(payload.get("reservation_id") == reservation_id, "reservation key drifted")

    first = _post(http, api_key, _CONFIRM_PATH, payload)
    _require_status(first, 204, "valid confirm")
    _require(first.content == b"", "204 confirm returned a response body")

    duplicate_call = _post(http, api_key, _CONFIRM_PATH, payload)
    _require_status(duplicate_call, 204, "call-id replay")

    reservation_replay = _post(
        http,
        api_key,
        _CONFIRM_PATH,
        _confirm_payload(call_id=str(uuid.uuid4()), reservation_id=reservation_id),
    )
    _require_status(reservation_replay, 204, "reservation replay")

    both = _post(
        http,
        api_key,
        _CONFIRM_PATH,
        {
            **_raw_confirm_base(),
            "reservation_id": "res_contract_both",
            "lease_id": "lse_contract_both",
        },
    )
    _require_status(both, 422, "confirm with both settlement keys")

    neither = _post(http, api_key, _CONFIRM_PATH, _raw_confirm_base())
    _require_status(neither, 422, "confirm with missing settlement key")

    unknown_reservation = _post(
        http,
        api_key,
        _CONFIRM_PATH,
        _confirm_payload(call_id=str(uuid.uuid4()), reservation_id="res_contract_unknown"),
    )
    _require_status(unknown_reservation, 404, "confirm with unknown reservation")
    unknown_body = _object_json(unknown_reservation, "confirm with unknown reservation")
    _require(
        unknown_body.get("detail") == "Reservation not found or expired",
        f"confirm with unknown reservation detail drifted: {unknown_body!r}",
    )


def _grant_payload(
    run_id: str,
    holder_id: str,
    *,
    model: str = "gpt-5.5",
    fail_open: bool = True,
) -> dict[str, Any]:
    return LeaseGrantRequest(
        agent_run_id=run_id,
        holder_id=holder_id,
        model=model,
        provider=ProviderName.OPENAI,
        fallback_providers=[],
        fallback_models=[],
        fail_open=fail_open,
        estimated_input_tokens=1000,
    ).model_dump(mode="json")


def _post_grant(
    http: httpx.Client,
    api_key: str,
    run_id: str,
    holder_id: str,
    *,
    model: str = "gpt-5.5",
    fail_open: bool = True,
) -> httpx.Response:
    return _post(
        http,
        api_key,
        _LEASE_PATH,
        _grant_payload(run_id, holder_id, model=model, fail_open=fail_open),
    )


def _require_lease_error(response: httpx.Response, status: int, code: str) -> None:
    _require_status(response, status, code)
    body = _object_json(response, code)
    raw_detail = body.get("detail")
    _require(isinstance(raw_detail, dict), f"{code} omitted structured detail")
    detail = cast(dict[str, Any], raw_detail)
    _require(detail.get("code") == code, f"{code} literal drifted: {body!r}")


def _require_no_lease_block(payload: dict[str, Any], context: str) -> None:
    leaked = _LEASE_BLOCK_KEYS.intersection(payload)
    _require(not leaked, f"{context} leaked lease keys: {sorted(leaked)}")


def _require_ineligible_shape(payload: dict[str, Any], context: str) -> None:
    _require(payload.get("eligible") is False, f"{context} was unexpectedly eligible")
    _require(payload.get("allowed") is True, f"{context} was unexpectedly denied")
    _require(
        payload.get("ineligible_reason") == "zero_rate_model",
        f"{context} reason drifted: {payload!r}",
    )
    _require("denied_by_period" not in payload, f"{context} serialized denied_by_period")
    _require_no_lease_block(payload, context)
    _require_display_shape(payload, context)
    parsed = _validate_model(LeaseGrantResponse, payload, context)
    _require(parsed.lease_id is None, f"{context} parsed a lease id")


def _require_eligible_lease_shape(
    payload: dict[str, Any],
    context: str,
    *,
    expected_lease_id: str | None,
    expected_generation: int,
) -> tuple[str, int]:
    _require(payload.get("eligible") is True, f"{context} was marked ineligible")
    _require(payload.get("allowed") is True, f"{context} was denied")
    _require("ineligible_reason" not in payload, f"{context} leaked ineligible reason")
    _require("denied_by_period" not in payload, f"{context} leaked denied period")
    missing = _LEASE_BLOCK_KEYS.difference(payload)
    _require(not missing, f"{context} omitted lease keys: {sorted(missing)}")
    _require_display_shape(payload, context)

    lease_id = payload.get("lease_id")
    _require(
        isinstance(lease_id, str) and 0 < len(lease_id) <= 64,
        f"{context} lease_id drifted: {lease_id!r}",
    )
    typed_lease_id = cast(str, lease_id)
    if expected_lease_id is not None:
        _require(typed_lease_id == expected_lease_id, f"{context} changed lease id")

    generation = _require_integer(payload.get("generation"), f"{context}.generation")
    _require(
        generation == expected_generation,
        f"{context} generation was {generation!r}, expected {expected_generation}",
    )
    for key in ("granted_tokens", "headroom_share_tokens"):
        value = _require_integer(payload.get(key), f"{context}.{key}")
        _require(value > 0, f"{context}.{key} must be positive")

    raw_refresh_interval = payload.get("refresh_interval_s")
    raw_lease_length = payload.get("lease_length_s")
    _require_numeric(raw_refresh_interval, f"{context}.refresh_interval_s")
    _require_numeric(raw_lease_length, f"{context}.lease_length_s")
    refresh_interval = cast(float, raw_refresh_interval)
    lease_length = cast(float, raw_lease_length)
    _require(refresh_interval > 0, f"{context} refresh interval must be positive")
    _require(lease_length > refresh_interval, f"{context} lacks a renewal window")

    raw_posture = payload.get("posture")
    _require(isinstance(raw_posture, dict), f"{context} omitted posture")
    posture = cast(dict[str, Any], raw_posture)
    _require(
        set(posture) == {"mode", "on_unreachable"},
        f"{context} posture keys drifted: {sorted(posture)}",
    )
    _require(posture.get("mode") == payload["mode"], f"{context} posture mode drifted")
    _require(
        posture.get("on_unreachable") == "fail_open",
        f"{context} posture did not echo fail_open",
    )
    _require(payload.get("final_grant") is False, f"{context} must not be final")
    _validate_model(LeaseGrantResponse, payload, context)
    return typed_lease_id, generation


def _surrender(
    http: httpx.Client,
    api_key: str,
    *,
    lease_id: str,
    holder_id: str,
    generation: int,
) -> httpx.Response:
    payload = LeaseSurrenderRequest(
        lease_id=lease_id,
        holder_id=holder_id,
        generation=generation,
    ).model_dump(mode="json")
    return _post(http, api_key, _LEASE_SURRENDER_PATH, payload)


def assert_lease_contract(http: httpx.Client, api_key: str) -> None:
    """Assert lease grant, refusal, fencing, settlement, and release shapes."""
    holder_cap = _post_grant(
        http,
        api_key,
        "contract-holder-cap",
        "contract-holder-cap-overflow",
    )
    _require_lease_error(holder_cap, 409, "lease_holder_cap_exceeded")

    denied = _post_grant(
        http,
        api_key,
        "contract-lease-denied",
        "contract-denied-holder",
    )
    _require_status(denied, 200, "lease deny")
    denied_payload = _object_json(denied, "lease deny")
    _require(denied_payload.get("eligible") is True, "lease deny was marked ineligible")
    _require(denied_payload.get("allowed") is False, "lease deny was allowed")
    _require(denied_payload.get("mode") == "hard_deny", "lease deny mode drifted")
    _require(
        denied_payload.get("denied_by_period") == "monthly",
        "lease deny period drifted",
    )
    _require("ineligible_reason" not in denied_payload, "lease deny leaked ineligible reason")
    _require_no_lease_block(denied_payload, "lease deny")
    _require_display_shape(denied_payload, "lease deny")
    denied_parsed = _validate_model(LeaseGrantResponse, denied_payload, "lease deny")
    _require(denied_parsed.lease_id is None, "lease deny parsed a lease id")

    ineligible = _post_grant(
        http,
        api_key,
        "contract-lease-ineligible",
        "contract-ineligible-holder",
        model=_UNPRICED_LEASE_MODEL,
    )
    _require_status(ineligible, 200, "ineligible grant")
    _require_ineligible_shape(_object_json(ineligible, "ineligible grant"), "ineligible grant")

    holder_id = f"contract-holder-{uuid.uuid4().hex[:12]}"
    granted = _post_grant(
        http,
        api_key,
        f"contract-lease-{uuid.uuid4().hex[:12]}",
        holder_id,
    )
    _require_status(granted, 200, "eligible grant")
    payload = _object_json(granted, "eligible grant")
    lease_id, generation = _require_eligible_lease_shape(
        payload,
        "eligible grant",
        expected_lease_id=None,
        expected_generation=1,
    )

    unknown = LeaseRenewRequest(
        lease_id="lse_not-a-real-lease",
        holder_id="contract-unknown-holder",
        generation=1,
    ).model_dump(mode="json")
    _require_lease_error(
        _post(http, api_key, _LEASE_RENEW_PATH, unknown),
        404,
        "lease_not_found",
    )

    wrong_generation = LeaseRenewRequest(
        lease_id=lease_id,
        holder_id=holder_id,
        generation=generation + 98,
    ).model_dump(mode="json")
    _require_lease_error(
        _post(http, api_key, _LEASE_RENEW_PATH, wrong_generation),
        409,
        "lease_generation_conflict",
    )

    renewal = LeaseRenewRequest(
        lease_id=lease_id,
        holder_id=holder_id,
        generation=generation,
    ).model_dump(mode="json")
    renewed_response = _post(http, api_key, _LEASE_RENEW_PATH, renewal)
    _require_status(renewed_response, 200, "lease renewal")
    renewed = _object_json(renewed_response, "lease renewal")
    _renewed_lease_id, renewed_generation = _require_eligible_lease_shape(
        renewed,
        "lease renewal",
        expected_lease_id=lease_id,
        expected_generation=generation + 1,
    )

    confirm_payload = _confirm_payload(call_id=str(uuid.uuid4()), lease_id=lease_id)
    _require("reservation_id" not in confirm_payload, "lease confirm serialized reservation_id")
    _require(confirm_payload.get("lease_id") == lease_id, "lease confirm key drifted")
    confirmed = _post(http, api_key, _CONFIRM_PATH, confirm_payload)
    _require_status(confirmed, 204, "lease-tagged confirm")
    replay = _post(http, api_key, _CONFIRM_PATH, confirm_payload)
    _require_status(replay, 204, "lease-tagged confirm replay")

    released = _surrender(
        http,
        api_key,
        lease_id=lease_id,
        holder_id=holder_id,
        generation=renewed_generation,
    )
    _require_status(released, 200, "lease surrender")
    released_body = _object_json(released, "lease surrender")
    _require(set(released_body) == {"released_tokens"}, "surrender response keys drifted")
    released_tokens = _require_integer(
        released_body.get("released_tokens"),
        "lease surrender.released_tokens",
    )
    _require(released_tokens > 0, "first surrender did not release positive tokens")

    repeated = _surrender(
        http,
        api_key,
        lease_id=lease_id,
        holder_id=holder_id,
        generation=renewed_generation,
    )
    _require_status(repeated, 200, "repeated surrender")
    repeated_body = _object_json(repeated, "repeated surrender")
    _require(
        set(repeated_body) == {"released_tokens"},
        "repeated surrender response keys drifted",
    )
    repeated_tokens = _require_integer(
        repeated_body.get("released_tokens"),
        "repeated surrender.released_tokens",
    )
    _require(
        repeated_tokens == 0,
        "repeated surrender was not idempotent",
    )


def _require_terminate_directive(
    payload: dict[str, Any],
    context: str,
    *,
    stopped_run_id: str,
) -> None:
    raw_directive = payload.get("run_control")
    _require(
        isinstance(raw_directive, dict),
        f"{context} omitted the run-control directive: {payload!r}",
    )
    directive = cast(dict[str, Any], raw_directive)
    _require(
        set(directive) == _RUN_CONTROL_KEYS,
        f"{context} run_control keys drifted: {sorted(directive)}",
    )
    _require(directive.get("version") == "1", f"{context} run_control version drifted")
    _require(
        directive.get("action") == "terminate",
        f"{context} run_control action drifted: {directive!r}",
    )
    _require(
        directive.get("agent_run_id") == stopped_run_id,
        f"{context} run_control echoed the wrong run id: {directive!r}",
    )
    reason = directive.get("reason")
    _require(
        isinstance(reason, str) and 0 < len(reason) <= _RUN_CONTROL_REASON_MAX_LENGTH,
        f"{context} run_control reason drifted: {reason!r}",
    )
    parsed = _validate_model(BudgetCheckResponse, payload, context)
    _require(parsed.run_control is not None, f"{context} failed SDK directive validation")
    _require(
        parsed.run_control is not None and parsed.run_control.agent_run_id == stopped_run_id,
        f"{context} parsed a directive for a different run",
    )


def assert_run_control_contract(
    http: httpx.Client,
    api_key: str,
    *,
    stopped_run_id: str,
) -> None:
    """Assert the run-control directive shapes that stop a run client-side.

    The caller stops ``stopped_run_id`` before calling this — server-state
    manipulation is lane-specific (the double scripts it, the live lane POSTs
    the dashboard stop), exactly like the pack's other setup.
    """
    opted_in = _post_check(
        http,
        api_key,
        context="stopped run directive check",
        agent_run_id=stopped_run_id,
        run_directive_version="1",
    )
    _require_deny_shape(opted_in, "run_stopped")
    _require_terminate_directive(
        opted_in,
        "stopped run directive check",
        stopped_run_id=stopped_run_id,
    )

    without_opt_in = _post_check(
        http,
        api_key,
        context="stopped run legacy check",
        agent_run_id=stopped_run_id,
    )
    _require_deny_shape(without_opt_in, "run_stopped")
    _require(
        "run_control" not in without_opt_in,
        f"stopped run legacy check sent an unrequested directive: {without_opt_in!r}",
    )

    other_run_id = f"contract-unstopped-{uuid.uuid4().hex[:12]}"
    other_run = _post_check(
        http,
        api_key,
        context="unstopped run check",
        agent_run_id=other_run_id,
        run_directive_version="1",
    )
    _require(
        "run_control" not in other_run,
        f"unstopped run check received a directive: {other_run!r}",
    )
    _validate_model(BudgetCheckResponse, other_run, "unstopped run check")


def _receipt_event(
    *,
    deny_source: DenySource,
    input_tokens: int,
    receipt_aggregate_count: int | None = None,
) -> MetadataEvent:
    return MetadataEvent(
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        input_tokens=input_tokens,
        output_tokens=0,
        token_details=None,
        latency_ms=0.0,
        status=CallStatus.BUDGET_DENIED,
        is_model_fallback=False,
        sdk_instance_id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        agent_run_id=f"contract-receipt-{uuid.uuid4().hex[:12]}",
        call_id=str(uuid.uuid4()),
        deny_source=deny_source,
        deny_reason="monthly",
        denied_by_period="monthly",
        estimated_output_bound=512,
        velocity_flags=["monotonic_growth", "repeat_size"],
        receipt_aggregate_count=receipt_aggregate_count,
    )


def _post_receipt(
    http: httpx.Client,
    api_key: str,
    event: MetadataEvent,
    context: str,
) -> None:
    response = http.post(
        _INGEST_PATH,
        json=[event.model_dump(mode="json")],
        headers=_headers(api_key),
    )
    _require_status(response, 202, context)
    body = _object_json(response, context)
    _require(
        body == {"ingested": 1, "rejected": []},
        f"{context} ingest body drifted: {body!r}",
    )


def assert_receipt_ingest_contract(http: httpx.Client, api_key: str) -> None:
    """Assert that denial receipts and aggregate replays ingest cleanly.

    Both events ride the ordinary metadata-ingest path and must be accepted
    whole: a fully-populated per-call receipt and the coarse aggregate replay
    the reporter emits for receipts it could not deliver. Every call id is
    fresh so a live server's ingest dedup can never mask a rejection.
    """
    _post_receipt(
        http,
        api_key,
        _receipt_event(deny_source="server", input_tokens=1000),
        "denial receipt ingest",
    )
    _post_receipt(
        http,
        api_key,
        _receipt_event(
            deny_source="aggregate_replay",
            input_tokens=3000,
            receipt_aggregate_count=3,
        ),
        "aggregate replay ingest",
    )
