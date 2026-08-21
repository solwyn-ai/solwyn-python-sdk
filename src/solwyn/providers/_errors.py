"""Status-first, duck-typed transport-error classification.

Maps an arbitrary transport exception onto a failover ``Disposition`` WITHOUT
importing any provider SDK. The core depends only on ``httpx`` + ``pydantic``;
clients remain bring-your-own, so we cannot assume any provider exception class
(``OverloadedError``/``ServiceUnavailableError``/``DeadlineExceededError``)
exists at runtime — they appear and disappear across versions. Classification is
therefore duck-typed: match ``httpx`` transport errors by ``isinstance``, read
numeric status from ``getattr(exc, "status_code", ...)`` (OpenAI/Anthropic) or
``getattr(exc, "code", ...)`` (Google ``APIError``), and match provider class
*names* via the MRO rather than the classes themselves. Unknown → ``FAIL_FAST``;
we never failover into the unknown.

Anthropic 1.x uses the separate ``httpx2`` distribution internally. Core does
not depend on or import it: an exception is admitted as an httpx2 transport
error only when its MRO contains the real ``httpx2.TransportError`` provenance
(matching both class name and module root). Within that gated family the same
safe split applies: ConnectTimeout/PoolTimeout/ConnectError are provably
pre-send; every other transport error is post-send-possible.

THE ORDERING TRAP (the reason the branch order below is load-bearing).
``APITimeoutError`` SUBCLASSES ``APIConnectionError`` in both the
openai and anthropic SDKs. A read timeout is POST-send: the request may have
already reached the model and run (a real charge + real side effects), so it is
ambiguous and must NOT failover. A bare ``APIConnectionError`` is PRE-send (DNS /
TLS / connection refused — provably never delivered), so it is safe to failover.
A naive ``except APIConnectionError: failover`` would catch the timeout subclass
and silently double-spend. The same trap exists in ``httpx``: ``ConnectTimeout``
subclasses ``TimeoutException`` (the generic, ambiguous case), and both
``ConnectError`` and ``RemoteProtocolError`` subclass ``TransportError``.

THE MIRROR TRAP (why ``APITimeoutError`` is not simply post-send). Both SDKs
wrap the WHOLE ``httpx.TimeoutException`` family in ``APITimeoutError`` — so a
provably pre-send ``ConnectTimeout``/``PoolTimeout`` arrives wearing the same
class name as a post-send ``ReadTimeout``. Treating the name as post-send would
strand every connect-slice timeout on a dead primary: default-safe idempotency
re-raises and the chain never advances. Since PJ-8/R7 gave connect its own
short deadline-derived slice (while read keeps a long decoupled bound), the
connect slice is the one that fires FIRST on a stalled provider, making this the
common case rather than the corner. Both wrapper branches therefore inspect the
chained httpx cause; their DEFAULTS are deliberately opposite, each falling back
to the canonical meaning of its own class name:

  * ``APITimeoutError``    with no inspectable cause -> POST_SEND_AMBIGUOUS
                            (a bare timeout is canonically a read timeout)
  * ``APIConnectionError`` with no inspectable cause -> FAILOVER
                            (a bare connection error is canonically refused)

The same duck-typing covers botocore (Bedrock): service errors are
``ClientError`` subclasses whose class NAME equals the AWS error code and whose
HTTP status lives at ``exc.response["ResponseMetadata"]["HTTPStatusCode"]`` (a
dict, not an attribute). Two Bedrock errors carry MISLEADING 4xx statuses:
``ModelTimeoutException`` (408) and ``ModelErrorException`` (424) both mean the
request REACHED the model and ran — by status alone they would be FAIL_FAST,
silently dropping the reservation reconciliation, so their names are matched
BEFORE the status check. botocore transport errors have no status at all and
classify purely by name; note ``ReadTimeoutError`` subclasses
``HTTPClientError``, NOT botocore's ``ConnectionError`` — and botocore's
``ConnectionError`` shares its NAME with the Python builtin (whose
``ConnectionResetError`` subclass is post-send-possible), so the bare name
"ConnectionError" is NEVER matched.

Consequently the checks MUST run in this exact order, narrowest-and-safest
first:

  1. ``APITimeoutError`` (by MRO name) -> classify by its chained transport cause:
       pre-send httpx/httpx2 (ConnectTimeout/PoolTimeout/ConnectError) -> FAILOVER
       read/write/any other/no cause (post-send possible) -> POST_SEND_AMBIGUOUS
  2. httpx ReadTimeout / WriteTimeout, and
     botocore ``ReadTimeoutError`` (by name)  -> POST_SEND_AMBIGUOUS
  3. httpx ConnectTimeout / PoolTimeout /
     ConnectError (provably pre-send), and
     botocore ``EndpointConnectionError`` /
     ``ConnectTimeoutError`` /
     ``ProxyConnectionError`` (by name)       -> FAILOVER
  4. httpx TimeoutException (any other)       -> POST_SEND_AMBIGUOUS
  4b. Bedrock post-send-by-name:
     ``ModelTimeoutException`` /
     ``ModelErrorException`` (misleading 4xx
     statuses — name wins over status), and
     botocore ``ConnectionClosedError``       -> POST_SEND_AMBIGUOUS
  5. numeric status (.status_code, then .code,
     then botocore ``response["ResponseMetadata"]["HTTPStatusCode"]``):
       429 / 529 -> FAILOVER
       4xx       -> FAIL_FAST
       5xx       -> POST_SEND_AMBIGUOUS
  6. ``APIConnectionError`` (by MRO name) -> classify by its chained transport cause:
       pre-send httpx/httpx2 (ConnectError/ConnectTimeout/PoolTimeout) -> FAILOVER
       any other recognized TransportError cause (post-send possible) -> POST_SEND_AMBIGUOUS
       no inspectable transport cause -> FAILOVER (canonical connect-refused outage)
  7. httpx/httpx2 TransportError (any other)  -> POST_SEND_AMBIGUOUS
  8. default                                  -> FAIL_FAST

Steps 1 and 2 MUST precede step 3 (timeout-before-connection). Step 3 MUST
precede step 4 (specific pre-send timeouts before the generic ambiguous
timeout). Step 4b MUST precede step 5 (Bedrock's post-send 408/424 names win
over their misleading statuses). Step 6 (bare connection failure) MUST come
AFTER the status check so a connection error that somehow carries a status is
classified by status first. The cause-inspection in steps 1 and 6 closes the
double-spend/stranded-chain vector from both sides: each wrapper covers BOTH
provably pre-send failures AND post-send ones (a server disconnect mid-response
surfaces as ``httpx.RemoteProtocolError``; a connect stall surfaces as
``httpx.ConnectTimeout``), so neither class name alone settles whether the
request landed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum

import httpx

# botocore transport errors classified purely by MRO name (they carry no
# status). Pre-send: the connection was never established. Post-send: bytes
# were sent or the connection dropped before a valid response — the model may
# have run. The bare name "ConnectionError" is deliberately ABSENT from both
# sets (it collides with the Python builtin; see the module docstring).
_BOTOCORE_PRE_SEND_NAMES = frozenset(
    {"EndpointConnectionError", "ConnectTimeoutError", "ProxyConnectionError"}
)
# Bedrock service errors whose NAME must win over their misleading 4xx status
# (408/424 mean the model ran), plus botocore's mid-response connection drop.
_POST_SEND_BY_NAME = frozenset(
    {"ModelTimeoutException", "ModelErrorException", "ConnectionClosedError"}
)
# httpx classes that PROVE the request never reached the model: the TCP/TLS
# connect never completed, or we never even got a connection out of the pool.
# Used both directly (step 3) and as the cause-inspection allowlist for the
# openai/anthropic wrappers (steps 1 and 6).
_PRE_SEND_HTTPX = (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ConnectError)
_PRE_SEND_TRANSPORT_NAMES = frozenset({"ConnectTimeout", "PoolTimeout", "ConnectError"})


def _httpx2_transport_names(exc: BaseException) -> frozenset[str]:
    """Return MRO names only for a proven ``httpx2.TransportError`` family.

    A bare class name is insufficient: provider and user exceptions can choose
    the same names. Requiring the ``TransportError`` base to originate from the
    ``httpx2`` module keeps this optional-dependency recognition scoped without
    importing Anthropic's private transport dependency into core.
    """
    mro = type(exc).__mro__
    if not any(
        cls.__name__ == "TransportError" and cls.__module__.partition(".")[0] == "httpx2"
        for cls in mro
    ):
        return frozenset()
    return frozenset(cls.__name__ for cls in mro)


def _transport_disposition(exc: BaseException) -> Disposition | None:
    """Classify recognized httpx/httpx2 transport errors by send certainty."""
    if isinstance(exc, _PRE_SEND_HTTPX):
        return Disposition.FAILOVER
    if isinstance(exc, httpx.TransportError):
        return Disposition.POST_SEND_AMBIGUOUS

    httpx2_names = _httpx2_transport_names(exc)
    if not httpx2_names:
        return None
    if httpx2_names & _PRE_SEND_TRANSPORT_NAMES:
        return Disposition.FAILOVER
    return Disposition.POST_SEND_AMBIGUOUS


class Disposition(StrEnum):
    """How the dispatch loop should react to a transport exception."""

    FAILOVER = "failover"
    """Provably pre-send rejection. Safe to advance to the next candidate."""

    POST_SEND_AMBIGUOUS = "post_send_ambiguous"
    """Request may have run. Re-raise the original exception; never failover."""

    FAIL_FAST = "fail_fast"
    """Same error everywhere (4xx / refusal / unknown). Stop the chain."""


def _numeric_status(exc: BaseException) -> int | None:
    """Return the first int among ``exc.status_code``, ``exc.code``, then the
    botocore ``exc.response["ResponseMetadata"]["HTTPStatusCode"]`` dict path.

    OpenAI/Anthropic carry the HTTP status on ``status_code``; Google's
    ``APIError`` carries it on ``code``; botocore's ``ClientError`` buries it
    in the parsed response dict. ``bool`` subclasses ``int``, so a truthy
    ``status_code = True`` would otherwise be misread as HTTP 1 — guard
    against it explicitly.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, Mapping):
            status = metadata.get("HTTPStatusCode")
            if isinstance(status, int) and not isinstance(status, bool):
                return status
    return None


def classify_exception(exc: BaseException) -> Disposition:
    """Classify a transport exception into a failover ``Disposition``.

    Status-first and duck-typed: never ``isinstance`` on a provider SDK class
    (they may be absent or differ across versions). See the module docstring for
    why the branch ordering below is load-bearing (the double-spend trap).
    """
    # MRO names let us match provider classes by name without importing them.
    names = {cls.__name__ for cls in type(exc).__mro__}

    # 1. APITimeoutError subclasses APIConnectionError — it MUST win first.
    #    But the openai/anthropic SDKs wrap EVERY httpx.TimeoutException in it,
    #    and ConnectTimeout/PoolTimeout subclass TimeoutException — so the class
    #    name alone does NOT prove the request landed. Inspect the chained cause
    #    exactly like step 6: a provably pre-send cause is failover-safe.
    #    The DEFAULT here is the MIRROR IMAGE of step 6's: a bare timeout with no
    #    inspectable cause is canonically a READ timeout (post-send), whereas a
    #    bare connection wrapper is canonically connect-refused (pre-send).
    if "APITimeoutError" in names:
        cause = exc.__cause__ or exc.__context__
        if cause is not None and _transport_disposition(cause) is Disposition.FAILOVER:
            return Disposition.FAILOVER
        return Disposition.POST_SEND_AMBIGUOUS

    # 2. httpx read/write timeouts are post-send: bytes were already sent.
    #    botocore's ReadTimeoutError is the same post-send case (note: it
    #    subclasses HTTPClientError, NOT botocore's ConnectionError).
    if isinstance(exc, httpx.ReadTimeout | httpx.WriteTimeout):
        return Disposition.POST_SEND_AMBIGUOUS
    if "ReadTimeoutError" in names:
        return Disposition.POST_SEND_AMBIGUOUS

    # 3. Provably pre-send transport failures — safe to failover.
    #    ConnectTimeout subclasses TimeoutException, so it MUST be checked
    #    before the generic TimeoutException branch (step 4) below. The
    #    botocore pre-send names are the boto equivalents (never the bare
    #    "ConnectionError" — that name collides with the Python builtin).
    if isinstance(exc, _PRE_SEND_HTTPX):
        return Disposition.FAILOVER
    if names & _BOTOCORE_PRE_SEND_NAMES:
        return Disposition.FAILOVER

    # 4. Any other timeout we cannot prove pre-send -> ambiguous.
    if isinstance(exc, httpx.TimeoutException):
        return Disposition.POST_SEND_AMBIGUOUS

    # 4b. Post-send by NAME, before the status check: Bedrock's
    #     ModelTimeoutException (408) and ModelErrorException (424) carry
    #     request-shaped statuses but mean the model RAN — classifying them by
    #     status would skip reservation reconciliation (silent double-count).
    #     ConnectionClosedError is botocore's mid-response connection drop.
    if names & _POST_SEND_BY_NAME:
        return Disposition.POST_SEND_AMBIGUOUS

    # 5. Numeric status classification (OpenAI/Anthropic .status_code,
    #    Google .code). Comes before the bare-connection branch so a status
    #    always wins over class-name matching.
    status = _numeric_status(exc)
    if status is not None:
        if status in (429, 529):
            return Disposition.FAILOVER
        if 400 <= status < 500:
            return Disposition.FAIL_FAST
        if status >= 500:
            return Disposition.POST_SEND_AMBIGUOUS

    # 6. Provider connection-error wrapper (no status, not itself a timeout).
    #    The openai/anthropic APIConnectionError wraps BOTH provably pre-send
    #    failures (connect refused / DNS / TLS) AND post-send transport drops
    #    (a server disconnect mid-response surfaces as httpx.RemoteProtocolError),
    #    so the class name alone does NOT prove the request never landed. Inspect
    #    the chained transport cause: only a recognized pre-send class is failover-safe;
    #    any other transport cause is post-send-possible and stays ambiguous.
    #    Real SDK exceptions always chain a cause; a bare wrapper with no cause is
    #    the canonical connect-refused outage case failover exists to handle.
    if "APIConnectionError" in names:
        cause = exc.__cause__ or exc.__context__
        if cause is not None:
            cause_disposition = _transport_disposition(cause)
            if cause_disposition is not None:
                return cause_disposition
        return Disposition.FAILOVER

    # 7. Other transport errors (e.g. RemoteProtocolError mid-flight) -> ambiguous.
    transport_disposition = _transport_disposition(exc)
    if transport_disposition is not None:
        return transport_disposition

    # 8. Default: never failover into the unknown.
    return Disposition.FAIL_FAST


def retry_after_seconds(exc: BaseException) -> float | None:
    """Return a non-negative ``Retry-After`` delay (seconds) for an HTTP 429, else None.

    Duck-typed and status-first, exactly like :func:`classify_exception` — never
    imports a provider SDK. Only HTTP 429 qualifies (the provider explicitly asked
    us to retry); every other status (including 529) returns ``None`` so the caller
    falls through to normal failover. The header is read off ``exc.response.headers``
    (the OpenAI/Anthropic/httpx shape) and honored in both RFC 7231 forms:
    delta-seconds (``"2"``) and an HTTP-date (``"Wed, 21 Oct 2026 07:28:00 GMT"``). A
    past HTTP-date clamps to ``0.0``. A missing or unparseable header returns ``None``
    — we never synthesize a backoff the provider did not request.
    """
    if _numeric_status(exc) != 429:
        return None
    raw = _retry_after_header(exc)
    if raw is None:
        return None
    return _parse_retry_after(raw)


def _retry_after_header(exc: BaseException) -> str | None:
    """Read the ``Retry-After`` header off a transport exception, duck-typed.

    Looks for a ``headers`` mapping on ``exc.response`` (the OpenAI/Anthropic
    shape), falls back to a ``headers`` attribute on the exception itself, and
    handles the botocore shape where ``exc.response`` is a parsed DICT carrying
    headers at ``["ResponseMetadata"]["HTTPHeaders"]`` (botocore lowercases the
    keys). Lookup is case-insensitive: ``httpx.Headers.get`` already is, and a
    plain dict carrier is scanned case-insensitively. Total by construction —
    this runs inside the dispatch except-handler on an arbitrary third-party
    exception, so a pathological ``headers`` object is swallowed and treated as
    "no header" rather than allowed to raise over the original provider
    exception.
    """
    for carrier in (getattr(exc, "response", None), exc):
        headers = getattr(carrier, "headers", None)
        if headers is None and isinstance(carrier, Mapping):
            metadata = carrier.get("ResponseMetadata")
            if isinstance(metadata, Mapping):
                headers = metadata.get("HTTPHeaders")
        if headers is None:
            continue
        try:
            getter = getattr(headers, "get", None)
            if getter is not None:
                # httpx.Headers.get resolves any casing; a plain dict matches lowercase.
                value = getter("retry-after")
                if value is not None:
                    return str(value)
            items = getattr(headers, "items", None)
            if items is not None:
                # Plain dict / non-httpx mapping with a non-canonical key casing.
                for key, value in items():
                    if value is not None and str(key).lower() == "retry-after":
                        return str(value)
        except Exception:
            # A headers carrier whose get()/items() raises must not escape into the
            # dispatch loop and mask the original exception; treat as no header.
            continue
    return None


def _parse_retry_after(raw: str) -> float | None:
    """Parse a ``Retry-After`` value (delta-seconds or HTTP-date) into seconds."""
    text = raw.strip()
    if text and all("0" <= ch <= "9" for ch in text):
        try:
            seconds = float(text)
        except OverflowError:
            return None
        return seconds if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())
