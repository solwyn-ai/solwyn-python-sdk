"""Pure scalar, finish-reason, image, and json/text helpers for translation.

================================ PRIVACY-CRITICAL ==============================
Part of the content-privileged ``providers/_translation`` package. These helpers
reshape customer prompt CONTENT through an in-memory transform ONLY; they never
store it on a long-lived object. No I/O: this module imports neither the stdlib
log module nor any HTTP client, and holds no log handle of any kind. Pure: same
input -> same output; no global state. Keep the firewall tests
(``tests/unit/test_privacy_firewall.py`` / ``test_translation.py``) green.
===============================================================================
"""

from __future__ import annotations

import json
from typing import Any, Literal

from ._guardrails import _raise
from ._models import ImagePart


def _common_scalars(provider: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull temperature/top_p/stop/stream out of a flat kwargs scope."""
    out: dict[str, Any] = {}
    temperature = kwargs.get("temperature")
    if temperature is not None:
        if float(temperature) > 1.0:
            _raise(provider, "*", "temperature>1.0")
        out["temperature"] = float(temperature)
    top_p = kwargs.get("top_p")
    if top_p is not None:
        out["top_p"] = float(top_p)
    out["stop"] = _normalize_stop(provider, kwargs)
    out["stream"] = bool(kwargs.get("stream", False))
    return out


def _normalize_stop(provider: str, scope: dict[str, Any]) -> list[str] | None:
    raw = scope.get("stop")
    if raw is None and isinstance(scope.get("stop_sequences"), list):
        raw = scope.get("stop_sequences")
    if raw is None:
        return None
    stops = [raw] if isinstance(raw, str) else list(raw)
    if len(stops) > 4:
        _raise(provider, "*", "stop>4")
    return stops


def _secure_image_url(provider: str, url: str) -> ImagePart:
    """Accept only ``https://`` image URLs; an ``http://`` (insecure) or other
    non-https URL RAISES ``image.insecure_url`` (fix [H2])."""
    if url.startswith("https://"):
        return ImagePart(url=url)
    _raise(provider, "*", "image.insecure_url")


# --------------------------------------------------------------------------- #
# §5.4 Finish-reason normalization (and the inverse for response shaping).      #
# --------------------------------------------------------------------------- #
_FINISH_TO_CANONICAL: dict[str, dict[str, str]] = {
    "openai": {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "content_filter",
    },
    "anthropic": {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "pause_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_use",
        "refusal": "content_filter",
    },
    "google": {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "BLOCKLIST": "content_filter",
    },
}

_CANONICAL_TO_FINISH: dict[str, dict[str, str]] = {
    "openai": {
        "stop": "stop",
        "length": "length",
        "tool_use": "tool_calls",
        "content_filter": "content_filter",
    },
    "anthropic": {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_use": "tool_use",
        "content_filter": "refusal",
    },
    "google": {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "tool_use": "STOP",
        "content_filter": "SAFETY",
    },
}


def normalize_finish_reason(
    served: str, raw: str | None
) -> Literal["stop", "length", "tool_use", "content_filter"] | None:
    """Map a provider-native finish/stop reason onto the canonical 4-value set."""
    if raw is None:
        return None
    mapped = _FINISH_TO_CANONICAL.get(served, {}).get(raw)
    return mapped  # type: ignore[return-value]


def _denormalize_finish_reason(requested: str, canonical: str | None) -> str | None:
    if canonical is None:
        return None
    return _CANONICAL_TO_FINISH.get(requested, {}).get(canonical)


# --------------------------------------------------------------------------- #
# Small in-memory helpers (pure; never log, never store).                      #
# --------------------------------------------------------------------------- #
def _parse_data_uri(uri: str) -> tuple[str, str]:
    """Split ``data:<media_type>;base64,<data>`` -> (media_type, data) VERBATIM.

    Anthropic strictly validates ``media_type`` vs bytes (§5.4), so the media
    type is preserved exactly as written.
    """
    header, _, data = uri.partition(",")
    meta = header[len("data:") :]
    media_type = meta.split(";", 1)[0]
    return media_type, data


def _loads_args(raw: Any) -> dict[str, Any]:
    """Decode a tool-call ``arguments`` value to a JSON object.

    OpenAI encodes arguments as a JSON STRING; Anthropic/Google as an object.
    Never logs the value; a malformed payload yields an empty dict structurally.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _as_text(content: Any) -> str:
    """Coerce a tool-result payload to a plain string (in-memory only)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                chunks.append(str(block["text"]))
            elif isinstance(block, str):
                chunks.append(block)
        return "".join(chunks)
    return json.dumps(content)
