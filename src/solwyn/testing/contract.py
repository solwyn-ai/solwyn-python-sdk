"""Behavior-bearing control-plane wire contract shared by fake and live lanes.

The functions in this module deliberately use explicit ``AssertionError``
checks instead of Python assertion statements.  Production modules are
importable under ``python -O`` and the repository forbids assertion statements
under ``src/solwyn``.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import httpx

from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    LeaseGrantRequest,
    LeaseGrantResponse,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    ProviderName,
)

_CHECK_PATH = "/api/v1/budgets/check"
_CONFIRM_PATH = "/api/v1/budgets/confirm"
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
_UNPRICED_LEASE_MODEL = "no-such-model-for-leases"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _object_json(response: httpx.Response, context: str) -> dict[str, Any]:
    body = response.json()
    _require(isinstance(body, dict), f"{context} returned non-object JSON: {body!r}")
    return cast(dict[str, Any], body)


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
) -> dict[str, Any]:
    return BudgetCheckRequest(
        estimated_input_tokens=1000,
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        agent_run_id=agent_run_id,
        tags=tags,
        failover_directive_version="1",
    ).model_dump(mode="json")


def _require_numeric(value: object, label: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be a JSON number, got {value!r}",
    )


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
    if "price_hints" in payload:
        hints = payload["price_hints"]
        _require(isinstance(hints, dict), f"{context} price_hints must be an object")
        for provider, hint in hints.items():
            _require(isinstance(provider, str), f"{context} price hint provider must be a string")
            _require_numeric(hint, f"{context}.price_hints[{provider!r}]")


def _post_check(
    http: httpx.Client,
    api_key: str,
    *,
    agent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = _post(
        http,
        api_key,
        _CHECK_PATH,
        _check_payload(agent_run_id=agent_run_id, tags=tags),
    )
    _require(response.status_code == 200, f"budget check returned {response.status_code}")
    return _object_json(response, "budget check")


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
    parsed = BudgetCheckResponse.model_validate(payload)
    _require(parsed.denied_by_period == period, f"{context} failed SDK response validation")


def assert_check_contract(http: httpx.Client, api_key: str) -> None:
    """Assert the check response shapes that change SDK behavior."""
    monthly = _post_check(
        http,
        api_key,
        tags={"contract_case": "monthly"},
    )
    _require_deny_shape(monthly, "monthly")

    stopped = _post_check(
        http,
        api_key,
        agent_run_id="contract-stopped-run",
    )
    _require_deny_shape(stopped, "run_stopped")

    tag = _post_check(
        http,
        api_key,
        tags={"customer": "acme"},
    )
    _require_deny_shape(tag, "tag")

    agent_run = _post_check(
        http,
        api_key,
        agent_run_id="contract-agent-run",
    )
    _require_deny_shape(agent_run, "agent_run")

    allowed = _post_check(http, api_key)
    _require(allowed.get("allowed") is True, f"allow check was denied: {allowed!r}")
    _require("denied_by_period" not in allowed, "allow check serialized denied_by_period")
    reservation_id = allowed.get("reservation_id")
    _require(
        isinstance(reservation_id, str) and bool(reservation_id),
        f"allow check omitted reservation_id: {allowed!r}",
    )
    _require_display_shape(allowed, "allow check")
    _require_directive_shape(allowed, "allow check")
    parsed = BudgetCheckResponse.model_validate(allowed)
    _require(parsed.allowed is True, "allow check failed SDK response validation")
    _require(parsed.denied_by_period is None, "allow parsed a denied period")


def _reservation(http: httpx.Client, api_key: str) -> str:
    payload = _post_check(http, api_key)
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
    _require(first.status_code == 204, f"valid confirm returned {first.status_code}")
    _require(first.content == b"", "204 confirm returned a response body")

    duplicate_call = _post(http, api_key, _CONFIRM_PATH, payload)
    _require(duplicate_call.status_code == 204, "call-id replay was not idempotent")

    reservation_replay = _post(
        http,
        api_key,
        _CONFIRM_PATH,
        _confirm_payload(call_id=str(uuid.uuid4()), reservation_id=reservation_id),
    )
    _require(reservation_replay.status_code == 204, "reservation replay was not idempotent")

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
    _require(both.status_code == 422, f"both settlement keys returned {both.status_code}")

    neither = _post(http, api_key, _CONFIRM_PATH, _raw_confirm_base())
    _require(neither.status_code == 422, f"missing settlement key returned {neither.status_code}")


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
    _require(response.status_code == status, f"{code} returned {response.status_code}")
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
    parsed = LeaseGrantResponse.model_validate(payload)
    _require(parsed.lease_id is None, f"{context} parsed a lease id")


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
    unavailable = _post_grant(
        http,
        api_key,
        "contract-lease-unavailable",
        "contract-unavailable-holder",
        model=_UNPRICED_LEASE_MODEL,
    )
    if unavailable.status_code == 503:
        _require_lease_error(unavailable, 503, "lease_unavailable")
    else:
        _require(
            unavailable.status_code == 200,
            "lease-unavailable probe returned unexpected status",
        )
        _require_ineligible_shape(
            _object_json(unavailable, "lease-unavailable fallback"),
            "lease-unavailable fallback",
        )

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
    _require(denied.status_code == 200, f"lease deny returned {denied.status_code}")
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
    denied_parsed = LeaseGrantResponse.model_validate(denied_payload)
    _require(denied_parsed.lease_id is None, "lease deny parsed a lease id")

    ineligible = _post_grant(
        http,
        api_key,
        "contract-lease-ineligible",
        "contract-ineligible-holder",
        model=_UNPRICED_LEASE_MODEL,
    )
    _require(ineligible.status_code == 200, f"ineligible grant returned {ineligible.status_code}")
    _require_ineligible_shape(_object_json(ineligible, "ineligible grant"), "ineligible grant")

    holder_id = f"contract-holder-{uuid.uuid4().hex[:12]}"
    granted = _post_grant(
        http,
        api_key,
        f"contract-lease-{uuid.uuid4().hex[:12]}",
        holder_id,
    )
    _require(granted.status_code == 200, f"eligible grant returned {granted.status_code}")
    payload = _object_json(granted, "eligible grant")
    _require(payload.get("eligible") is True, "eligible grant was marked ineligible")
    _require(payload.get("allowed") is True, "eligible grant was denied")
    _require("ineligible_reason" not in payload, "eligible grant leaked ineligible reason")
    _require("denied_by_period" not in payload, "eligible grant leaked denied period")
    missing = _LEASE_BLOCK_KEYS.difference(payload)
    _require(not missing, f"eligible grant omitted lease keys: {sorted(missing)}")
    _require_display_shape(payload, "eligible grant")

    lease_id = payload.get("lease_id")
    generation = payload.get("generation")
    _require(
        isinstance(lease_id, str) and 0 < len(lease_id) <= 64,
        f"eligible grant lease_id drifted: {lease_id!r}",
    )
    _require(generation == 1, f"fresh lease generation was not 1: {generation!r}")
    lease_id = cast(str, lease_id)
    generation = cast(int, generation)
    for key in ("granted_tokens", "headroom_share_tokens"):
        value = payload.get(key)
        _require(isinstance(value, int) and value > 0, f"eligible grant {key} must be positive")
    raw_refresh_interval = payload.get("refresh_interval_s")
    raw_lease_length = payload.get("lease_length_s")
    _require_numeric(raw_refresh_interval, "eligible grant.refresh_interval_s")
    _require_numeric(raw_lease_length, "eligible grant.lease_length_s")
    refresh_interval = cast(float, raw_refresh_interval)
    lease_length = cast(float, raw_lease_length)
    _require(refresh_interval > 0, "eligible grant refresh interval must be positive")
    _require(lease_length > refresh_interval, "lease must contain a renewal window")
    raw_posture = payload.get("posture")
    _require(isinstance(raw_posture, dict), "eligible grant omitted posture")
    posture = cast(dict[str, Any], raw_posture)
    _require(posture.get("mode") == payload["mode"], "lease posture mode drifted")
    _require(posture.get("on_unreachable") == "fail_open", "lease posture did not echo fail_open")
    _require(payload.get("final_grant") is False, "fresh grant must not be final")
    LeaseGrantResponse.model_validate(payload)

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
    _require(renewed_response.status_code == 200, "valid lease renewal failed")
    renewed = _object_json(renewed_response, "lease renewal")
    _require(renewed.get("lease_id") == lease_id, "renewal changed lease id")
    renewed_generation = renewed.get("generation")
    _require(
        renewed_generation == generation + 1,
        "renewal did not advance generation by one",
    )
    _require(renewed.get("eligible") is True, "renewal was marked ineligible")
    _require(renewed.get("allowed") is True, "renewal was denied")
    LeaseGrantResponse.model_validate(renewed)

    confirm_payload = _confirm_payload(call_id=str(uuid.uuid4()), lease_id=lease_id)
    _require("reservation_id" not in confirm_payload, "lease confirm serialized reservation_id")
    _require(confirm_payload.get("lease_id") == lease_id, "lease confirm key drifted")
    confirmed = _post(http, api_key, _CONFIRM_PATH, confirm_payload)
    _require(confirmed.status_code == 204, "lease-tagged confirm failed")
    replay = _post(http, api_key, _CONFIRM_PATH, confirm_payload)
    _require(replay.status_code == 204, "lease-tagged confirm replay failed")

    released = _surrender(
        http,
        api_key,
        lease_id=lease_id,
        holder_id=holder_id,
        generation=generation + 1,
    )
    _require(released.status_code == 200, "lease surrender failed")
    released_body = _object_json(released, "lease surrender")
    _require(set(released_body) == {"released_tokens"}, "surrender response keys drifted")
    released_tokens = released_body.get("released_tokens")
    _require(
        isinstance(released_tokens, int) and released_tokens > 0,
        "first surrender did not release positive tokens",
    )

    repeated = _surrender(
        http,
        api_key,
        lease_id=lease_id,
        holder_id=holder_id,
        generation=generation + 1,
    )
    _require(repeated.status_code == 200, "idempotent surrender failed")
    _require(
        _object_json(repeated, "repeated surrender") == {"released_tokens": 0},
        "repeated surrender was not idempotent",
    )
