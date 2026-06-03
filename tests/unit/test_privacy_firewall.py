"""Structural firewall tests for the SDK prompt-privacy promise.

These tests enforce the design-spec §7 content-touching contract:

  1. Customer prompts/responses are never passed into a log statement —
     not via a bareword name, not via a kwargs/payload dict.
  2. Content is reshaped in EXACTLY TWO allowlisted modules
     (``_privacy.py`` + ``providers/_translation.py``) and nowhere else.
  3. The translation module is pure: no logger, no httpx, no I/O.
  4. SDK<->API wire models stay structurally content-free.
  5. Failover's new leak vectors (``add_note`` interpolation, ``exc_info=True``
     with a provider exception in scope) are structurally forbidden.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY

SDK_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "solwyn"

# The content-touching allowlist (spec §7). EXACTLY these two modules may reshape
# customer prompt content. This explicit ALLOWLIST replaces the older
# "_privacy not in name" exclusion.
CONTENT_PRIVILEGED = {"_privacy.py", "_translation.py"}

# Field names that would carry prompt/response content onto the SDK<->API wire.
FORBIDDEN_FIELDS = {
    "messages",
    "content",
    "system",
    "contents",
    "prompt",
    "text",
    "input",
    "response",
}


def _iter_source_files() -> list[Path]:
    """All SDK source files EXCEPT the content-privileged allowlist."""
    return [p for p in SDK_SRC.rglob("*.py") if p.name not in CONTENT_PRIVILEGED]


def _content_privileged_paths() -> list[Path]:
    """The concrete on-disk paths of the allowlisted content-touching modules."""
    paths = [p for p in SDK_SRC.rglob("*.py") if p.name in CONTENT_PRIVILEGED]
    # Tripwire: the allowlist names must all actually resolve to a file.
    assert {p.name for p in paths} == CONTENT_PRIVILEGED, (
        f"CONTENT_PRIVILEGED names do not all resolve to a file: "
        f"found {sorted(p.name for p in paths)}"
    )
    return paths


# --------------------------------------------------------------------------- #
# Logger leak vectors                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_no_logger_calls_receive_prompt_variables() -> None:
    """Source files must not pass a prompt/content-bearing variable into a
    logger call. Covers bareword content names AND the kwargs/payload dict
    names that the failover dispatch threads (spec §7 leak vector — the
    bareword regex alone misses ``kwargs``/``translated_kwargs``/``payload``)."""
    leak_words = [
        # bareword content names
        "text",
        "content",
        "messages",
        "system",
        "prompt",
        "contents",
        "response",
        # kwargs / payload dict names introduced by the failover dispatch
        "kwargs",
        "fallback_kwargs",
        "translated_kwargs",
        "call_kwargs",
        "payload",
        "request_body",
    ]
    patterns = [
        re.compile(
            r"logger\.(debug|info|warning|error|exception)\s*\([^)]*\b" + re.escape(word) + r"\b"
        )
        for word in leak_words
    ]
    violations: list[str] = []
    for path in _iter_source_files():
        source = path.read_text()
        for line_no, line in enumerate(source.splitlines(), start=1):
            for pat in patterns:
                if pat.search(line):
                    violations.append(f"{path.relative_to(SDK_SRC)}:{line_no}: {line.strip()}")
    assert not violations, (
        "Privacy violation candidates found — logger calls must never receive "
        "prompt/content/kwargs/payload variables:\n" + "\n".join(violations)
    )


@pytest.mark.unit
def test_extract_text_from_kwargs_is_removed() -> None:
    """The original _extract_text_from_kwargs that materialized the full
    joined prompt string must no longer exist."""
    client_py = (SDK_SRC / "client.py").read_text()
    assert "_extract_text_from_kwargs" not in client_py, (
        "client.py must not define or call _extract_text_from_kwargs — "
        "use solwyn._privacy.estimate_content_length instead."
    )


@pytest.mark.unit
def test_no_print_calls_in_sdk_source() -> None:
    """SDK source files must not contain print() calls (a trivial leak path)."""
    violations: list[str] = []
    for path in SDK_SRC.rglob("*.py"):
        source = path.read_text()
        for line_no, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("print(") or " print(" in stripped:
                violations.append(f"{path.relative_to(SDK_SRC)}:{line_no}: {stripped}")
    assert not violations, (
        "SDK source must not contain print() calls — use logging instead "
        "(but never for prompt content):\n" + "\n".join(violations)
    )


# --------------------------------------------------------------------------- #
# §7 — content-privileged allowlist enforcement                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_content_privileged_modules_have_no_logging_import() -> None:
    """Neither content-privileged module may import the logging module —
    prompt-adjacent code must never have access to a logger."""
    for path in _content_privileged_paths():
        src = path.read_text()
        assert "import logging" not in src, f"{path.name} must not import logging"


@pytest.mark.unit
def test_content_privileged_modules_carry_privacy_banner() -> None:
    """Each content-privileged module opens with a visible PRIVACY-CRITICAL
    banner in the first ~600 chars so contributors know the contract."""
    for path in _content_privileged_paths():
        head = path.read_text()[:600]
        assert "PRIVACY-CRITICAL" in head, (
            f"{path.name} must open with a PRIVACY-CRITICAL banner in the first ~600 chars."
        )


@pytest.mark.unit
def test_translation_module_does_no_io() -> None:
    """_translation.py must import no HTTP client — not even as a bare token.
    It is pure / sans-I/O and may never reach a client pointed at config.api_url."""
    src = (SDK_SRC / "providers" / "_translation.py").read_text()
    assert "import httpx" not in src, "_translation.py must not import httpx"
    assert "httpx" not in src, "_translation.py must not even name httpx"


@pytest.mark.unit
def test_translation_module_has_no_logger_token() -> None:
    """_translation.py must hold no logger and not even name one — a missing
    logger is the strongest guarantee that content cannot be logged here."""
    src = (SDK_SRC / "providers" / "_translation.py").read_text()
    assert "logger" not in src, "_translation.py must not name a logger"
    assert "logging" not in src, "_translation.py must not reference logging"


@pytest.mark.unit
def test_privacy_banner_set_equals_allowlist() -> None:
    """Allowlist drift tripwire: the set of modules whose first ~600 chars
    contain the literal 'PRIVACY-CRITICAL' banner must equal CONTENT_PRIVILEGED
    EXACTLY. Adding the banner to a third file (or dropping it from one of the
    two) trips this immediately — keeping the allowlist honest."""
    bannered = {p.name for p in SDK_SRC.rglob("*.py") if "PRIVACY-CRITICAL" in p.read_text()[:600]}
    assert bannered == CONTENT_PRIVILEGED, (
        "Exactly the two content-privileged modules may open with a "
        f"PRIVACY-CRITICAL banner. Found: {sorted(bannered)}"
    )


# --------------------------------------------------------------------------- #
# Wire models stay structurally content-free                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_wire_models_stay_content_free() -> None:
    """SDK<->API wire models must never carry a content-bearing field name.
    Failover adds only STRUCTURAL fields (provider identifiers, error class)."""
    from solwyn._types import BudgetCheckRequest, BudgetConfirmRequest, MetadataEvent

    for model in (MetadataEvent, BudgetCheckRequest, BudgetConfirmRequest):
        leaked = set(model.model_fields) & FORBIDDEN_FIELDS
        assert not leaked, f"{model.__name__} leaks content-bearing fields: {leaked}"


# --------------------------------------------------------------------------- #
# Failover leak vectors: add_note interpolation + exc_info with provider exc   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_add_note_interpolates_only_exception_type_name() -> None:
    """Provider SDK exceptions embed request content in str(exc)/.body. Any
    add_note(f"…") in SDK source may only interpolate ``type(...).__name__`` —
    never str(exc), .body, or any other expression (spec §7 leak vector)."""
    # Matches an f-string add_note and captures every {...} placeholder inside it.
    note_call = re.compile(r"add_note\(\s*f(['\"])(.*?)\1", re.DOTALL)
    placeholder = re.compile(r"\{([^{}]*)\}")
    # The only permitted interpolation: type(<anything>).__name__
    allowed = re.compile(r"^type\([^()]*\)\.__name__$")

    violations: list[str] = []
    for path in SDK_SRC.rglob("*.py"):
        source = path.read_text()
        for note_match in note_call.finditer(source):
            fstring_body = note_match.group(2)
            for ph in placeholder.finditer(fstring_body):
                expr = ph.group(1).split("!")[0].split(":")[0].strip()
                if not allowed.match(expr):
                    line_no = source.count("\n", 0, note_match.start()) + 1
                    violations.append(
                        f"{path.relative_to(SDK_SRC)}:{line_no}: add_note interpolates "
                        f"{{{expr}}} — only type(...).__name__ is permitted"
                    )
    assert not violations, (
        'add_note(f"…") may only interpolate type(...).__name__ so a provider '
        "exception's content cannot leak into __notes__:\n" + "\n".join(violations)
    )


@pytest.mark.unit
def test_no_exc_info_logging_in_failover_provider_except_blocks() -> None:
    """In client.py, the failover candidate-walk except-blocks bind a live
    provider exception (``except ... as exc:`` whose body calls
    ``classify_exception``). A ``logger.*(..., exc_info=True)`` there would
    render the provider exception (whose str()/body embeds request content)
    into the log record. Assert no such call lives in those blocks (§7)."""
    client_py = (SDK_SRC / "client.py").read_text()
    lines = client_py.splitlines()

    except_re = re.compile(r"^(\s*)except\b.*\bas\s+\w+\s*:")
    exc_info_re = re.compile(r"logger\.\w+\s*\([^)]*exc_info\s*=\s*True")

    violations: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = except_re.match(lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        # Collect the block body (lines more-indented than the `except`).
        body_start = i + 1
        j = body_start
        while j < n:
            stripped = lines[j].strip()
            if stripped == "":
                j += 1
                continue
            cur_indent = len(lines[j]) - len(lines[j].lstrip())
            if cur_indent <= indent:
                break
            j += 1
        body = lines[body_start:j]
        body_text = "\n".join(body)
        # Only the failover provider-exception except-blocks classify the exc.
        is_failover_block = "classify_exception" in body_text
        if is_failover_block:
            for offset, line in enumerate(body):
                if exc_info_re.search(line):
                    violations.append(
                        f"client.py:{body_start + offset + 1}: exc_info=True logged in a "
                        f"failover provider-exception except-block: {line.strip()}"
                    )
        i = j

    assert not violations, (
        "A provider exception's content can leak via exc_info=True. The failover "
        "except-blocks (which classify_exception) must not log with exc_info:\n"
        + "\n".join(violations)
    )


# --------------------------------------------------------------------------- #
# THE authoritative backstop — behavioral, end-to-end (spec §7)               #
# --------------------------------------------------------------------------- #
# The structural scans above prove import/log-call/add_note shapes, but cannot
# model new-field / __repr__ / translated-kwargs leaks. This test is the
# behavioral proof: even on the TRANSLATING cross-provider failover path, NOT
# ONE BYTE of prompt content reaches the Solwyn Cloud API. It must ship.


class _RecordingResponse:
    """A minimal stand-in for an httpx.Response (sync .json(), 200 OK)."""

    status_code = 200

    def __init__(self, body: dict[str, object]) -> None:
        self._body = body

    def json(self) -> dict[str, object]:
        return self._body

    def raise_for_status(self) -> None:
        return None


class _RecordingHttpClient:
    """Fake httpx client capturing every ``json=`` body POSTed to the Cloud API.

    Stands in for BOTH ``_budget._http`` and ``_reporter._http`` so the test
    sees the budget check, the confirm, AND the metadata-ingest payloads. The
    budget-check URL returns an allow-response so the call proceeds and the
    cross-provider hop (translation + Anthropic serve) actually fires.
    """

    def __init__(self, captured: list[object]) -> None:
        self._captured = captured

    def post(self, url: str, *, json: object = None, **_kw: object) -> _RecordingResponse:
        self._captured.append(json)
        return _RecordingResponse(dict(ALLOW_BUDGET_RESPONSE))

    def close(self) -> None:  # pragma: no cover - drained in teardown
        return None


@pytest.mark.unit
def test_failover_solwyn_payloads_carry_no_content() -> None:
    """End-to-end: on the translating cross-provider failover path, no prompt
    content reaches the Solwyn Cloud API across budget check, confirm, and
    metadata ingest. SENTINEL must never appear in ANY captured ``json=`` body."""
    from solwyn.client import Solwyn

    SENTINEL = "SUPER_SECRET_PROMPT_a1b2c3"

    # ── Arrange: fake provider clients (no real SDKs importable). ──────────
    def _provider_client(module: str, name: str) -> object:
        from unittest.mock import MagicMock

        client = MagicMock()
        client.__class__.__module__ = module
        client.__class__.__name__ = name
        client.with_options.return_value = client
        return client

    openai = _provider_client("openai._client", "OpenAI")
    anthropic = _provider_client("anthropic._client", "Anthropic")
    # Anthropic serves with a native-shaped response carrying the SENTINEL in its
    # OUTPUT too — proving the response text never reaches the Cloud API either.
    anthropic.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=f"echo {SENTINEL}")],
        stop_reason="end_turn",
        model="claude-3-5-sonnet",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )

    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )
    # Quiesce the reporter's background thread so the flush is deterministic.
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)

    # Capture every json= body POSTed to config.api_url on BOTH clients.
    captured: list[object] = []
    solwyn._budget._http = _RecordingHttpClient(captured)  # type: ignore[assignment]
    solwyn._reporter._http = _RecordingHttpClient(captured)  # type: ignore[assignment]

    # Force the primary circuit OPEN so the chain skips OpenAI and the
    # cross-provider Anthropic hop fires (translation + normalization run).
    openai_cb = solwyn._get_circuit_breaker("openai")
    for _ in range(openai_cb.failure_threshold):
        openai_cb.record_failure()

    request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": f"system {SENTINEL}"},
            {"role": "user", "content": f"user prompt {SENTINEL}"},
        ],
    }

    # ── Act: run the call; then drain the reporter queue through the fake. ─
    result = solwyn.chat.completions.create(**request)
    solwyn._reporter._flush_remaining()  # drives metadata-ingest through the fake

    # Sanity: the cross-provider hop actually served (so translation DID run on
    # SENTINEL-bearing content) and the response normalized back to OpenAI shape.
    anthropic.messages.create.assert_called_once()
    assert result.choices[0].message.content == f"echo {SENTINEL}"

    # ── Assert: at least the budget check + confirm + ingest were captured… ─
    assert len(captured) >= 3, f"expected budget/confirm/ingest payloads, got {len(captured)}"
    # …and NOT ONE captured payload carries any prompt or response content.
    blob = json.dumps(captured)
    assert SENTINEL not in blob, (
        "PRIVACY BREACH: prompt/response content reached the Solwyn Cloud API on "
        "the translating cross-provider failover path."
    )

    solwyn._budget._http.close()
    solwyn._reporter._http.close()
