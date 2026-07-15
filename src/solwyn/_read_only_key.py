"""Recognize and report the API's narrow read-only-key configuration error."""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_log_lock = threading.Lock()
_logged_read_only_key_error = False


def _is_read_only_key_error(exc: Exception) -> bool:
    """Match only the structured 403 contract, without exposing its body."""
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code != 403:
        return False
    try:
        payload: Any = exc.response.json()
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    detail = payload.get("detail")
    return isinstance(detail, dict) and detail.get("code") == "read_only_key"


def handle_read_only_key_error(exc: Exception) -> bool:
    """Log the configuration diagnostic once per process when recognized."""
    if not _is_read_only_key_error(exc):
        return False

    global _logged_read_only_key_error
    with _log_lock:
        if not _logged_read_only_key_error:
            logger.error(
                "solwyn.configuration_error.read_only_key: "
                "the configured API key is read-only; use a full-scope project key "
                "for SDK budget enforcement and metadata reporting"
            )
            _logged_read_only_key_error = True
    return True
