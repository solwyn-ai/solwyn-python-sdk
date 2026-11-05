"""Classify Solwyn API refusals of the key or its project (spec section 6.4).

The rule is status first, body second. Every 401 from the Solwyn API is a
refusal whatever its body: a recognised ``api_key_expired`` / ``api_key_invalid``
body carries the project's posture, anything else (invalid JSON, an empty body,
a string ``detail``, a missing or unknown ``code``) is ``api_key_invalid`` with
posture ``unknown``, which blocks. 400/403/404 classify only on a recognised
project or scope code. Nothing here reads or logs a body beyond the
``detail.code``, ``detail.project_budget_mode`` and ``detail.expired_at`` fields.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import httpx

Posture = Literal["hard_deny", "alert_only", "unknown"]

# 400/403/404 codes that refuse the key or its project. Any other code on
# those statuses keeps today's handling (it is not a key refusal).
_NON_401_CODES = frozenset(
    {
        "project_selector_required",
        "project_selector_conflict",
        "project_not_found",
        "key_not_dedicated",
        "read_only_key",
    }
)
_NON_401_STATUSES = frozenset({400, 403, 404})

_INVALID_KEY_MESSAGE = "The Solwyn API key was not recognised or has been revoked."
_READ_ONLY_MESSAGE = (
    "This Solwyn API key is read-only. The SDK reports spending, so it needs a "
    "full-scope key. Create one in the dashboard or with solwyn keys create."
)


@dataclass(frozen=True, slots=True)
class Refusal:
    """A classified refusal: the code, the project posture, and the expiry date."""

    code: str
    posture: Posture = "unknown"
    # ``YYYY-MM-DD`` from ``detail.expired_at`` on ``api_key_expired``.
    expired_on: str | None = None


def _detail_of(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    return detail if isinstance(detail, dict) else None


def _parse(content: bytes) -> object:
    try:
        return json.loads(content)
    except (ValueError, UnicodeDecodeError, TypeError):
        return None


def _posture(value: object) -> Posture:
    if value == "hard_deny":
        return "hard_deny"
    if value == "alert_only":
        return "alert_only"
    return "unknown"


def _expired_on(value: object) -> str | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        return None


def classify(status: int, content: bytes) -> Refusal | None:
    """Return the refusal a Solwyn API answer carries, or None when it is not one."""
    return _classify(status, lambda: _parse(content))


def _classify(status: int, payload: Callable[[], object]) -> Refusal | None:
    if status == 401:
        detail = _detail_of(payload())
        code = detail.get("code") if detail is not None else None
        if detail is not None and code in ("api_key_expired", "api_key_invalid"):
            return Refusal(
                code=code,
                posture=_posture(detail.get("project_budget_mode")),
                expired_on=(
                    _expired_on(detail.get("expired_at")) if code == "api_key_expired" else None
                ),
            )
        return Refusal(code="api_key_invalid")
    if status in _NON_401_STATUSES:
        detail = _detail_of(payload())
        code = detail.get("code") if detail is not None else None
        if isinstance(code, str) and code in _NON_401_CODES:
            return Refusal(code=code)
    return None


def classify_response(response: httpx.Response) -> Refusal | None:
    """Classify a received response; an unreadable body counts as no body."""

    def payload() -> object:
        try:
            return response.json()
        except Exception:
            return None

    return _classify(response.status_code, payload)


def classify_exception(exc: BaseException) -> Refusal | None:
    """Classify a ``raise_for_status`` failure; every other exception is not a refusal."""
    if isinstance(exc, httpx.HTTPStatusError):
        return classify_response(exc.response)
    return None


def refusal_message(refusal: Refusal, selector: str) -> tuple[str, str]:
    """Return the section 6.1 ``(message, field)`` for a refusal."""
    code = refusal.code
    if code == "api_key_expired":
        if refusal.expired_on is not None:
            return (
                f"The Solwyn API key expired on {refusal.expired_on}. Rotate it in the "
                "dashboard or with solwyn keys rotate.",
                "api_key",
            )
        return (
            "The Solwyn API key has expired. Rotate it in the dashboard or with "
            "solwyn keys rotate.",
            "api_key",
        )
    if code == "api_key_invalid":
        return _INVALID_KEY_MESSAGE, "api_key"
    if code == "read_only_key":
        return _READ_ONLY_MESSAGE, "api_key"
    if code == "project_not_found":
        return (
            f"Solwyn project '{selector}' was not found for this API key. Check "
            "SOLWYN_PROJECT. A renamed project's old name stops working; its proj_ id "
            "always works.",
            "project",
        )
    if code == "key_not_dedicated":
        return (
            f"Solwyn project '{selector}' only accepts API keys dedicated to it, and this "
            "key also reaches other projects. Use a key that holds only this project.",
            "project",
        )
    # project_selector_required / project_selector_conflict: the SDK always
    # sends exactly one header, so either one means the SDK misbehaved.
    return (
        f"Solwyn refused the project header ({code}). This is an SDK bug; please report it.",
        "project",
    )
