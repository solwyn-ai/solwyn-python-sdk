"""Send-certainty-first, duck-typed transport-error classification.

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
(matching the already-loaded module's exported class by identity). Within that
gated family the same safe split applies: ConnectTimeout/PoolTimeout/ConnectError
are provably pre-send; every other transport error is post-send-possible.

THE ORDERING TRAP (the reason the branch order below is load-bearing).
``APITimeoutError`` SUBCLASSES ``APIConnectionError`` in both the
openai and anthropic SDKs. A read timeout is POST-send: the request may have
already reached the model and run (a real charge + real side effects), so it is
ambiguous and must NOT failover. A bare ``APIConnectionError`` is PRE-send (DNS /
TLS / connection refused — provably never delivered), so it is safe to failover.
A naive ``except APIConnectionError: failover`` would catch the timeout subclass
and silently double-spend. The same trap exists in ``httpx``: ``ConnectTimeout``
subclasses ``TimeoutException`` (the generic, ambiguous case), and both
``ConnectError`` and ``RemoteProtocolError`` subclass ``TransportError`` — which
is why ``_transport_disposition`` tests the narrow pre-send tuple before the
generic transport base, for both stacks.

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
  2. recognized httpx/httpx2 transport error
     (``_transport_disposition``, ONE branch for
     BOTH stacks) -> classify by send certainty:
       ConnectTimeout / PoolTimeout /
       ConnectError (provably pre-send)        -> FAILOVER
       every other transport error (read/write
       timeout, generic TimeoutException,
       RemoteProtocolError, ReadError, ...)    -> POST_SEND_AMBIGUOUS
  3. botocore ``ReadTimeoutError`` (by name)  -> POST_SEND_AMBIGUOUS
  4. botocore ``EndpointConnectionError`` /
     ``ConnectTimeoutError`` /
     ``ProxyConnectionError`` (by name)       -> FAILOVER
  5. Bedrock post-send-by-name:
     ``ModelTimeoutException`` /
     ``ModelErrorException`` (misleading 4xx
     statuses — name wins over status), and
     botocore ``ConnectionClosedError``       -> POST_SEND_AMBIGUOUS
  6. numeric status (.status_code, then .code,
     then botocore ``response["ResponseMetadata"]["HTTPStatusCode"]``):
       429 / 529 -> FAILOVER
       4xx       -> FAIL_FAST
       5xx       -> POST_SEND_AMBIGUOUS
  7. ``APIConnectionError`` (by MRO name) -> classify by its chained transport cause:
       pre-send httpx/httpx2 (ConnectError/ConnectTimeout/PoolTimeout) -> FAILOVER
       any other recognized TransportError cause (post-send possible) -> POST_SEND_AMBIGUOUS
       no inspectable transport cause -> FAILOVER (canonical connect-refused outage)
  8. default                                  -> FAIL_FAST

Step 1 MUST precede step 7 (``APITimeoutError`` subclasses
``APIConnectionError``; the timeout must not be settled by the bare-connection
branch). Step 2 is ONE branch covering both transport stacks so an httpx2
exception can never be classified at a different position than its httpx1 twin
— the ordering trap inside the family (specific pre-send classes before the
generic ambiguous ``TimeoutException``/``TransportError``) lives in
``_transport_disposition``. Step 2 MUST precede step 6: a PROVEN transport class
settles send certainty, so it outranks any numeric status attached to the same
exception (an httpx ``ReadTimeout`` carrying a 429 is still post-send). It
cannot shadow steps 3-5 either: ``_transport_disposition`` returns ``None`` for
everything outside the two transport families, and no httpx/httpx2 MRO name
collides with the botocore name sets. Step 5 MUST precede step 6 (Bedrock's
post-send 408/424 names win over their misleading statuses). Step 7 (bare
connection failure) MUST come AFTER the status check so a connection error that
somehow carries a status is classified by status first. The cause-inspection in
steps 1 and 7 closes the double-spend/stranded-chain vector from both sides:
each wrapper covers BOTH provably pre-send failures AND post-send ones (a server
disconnect mid-response surfaces as ``httpx.RemoteProtocolError``; a connect
stall surfaces as ``httpx.ConnectTimeout``), so neither class name alone settles
whether the request landed.
"""

from __future__ import annotations

import inspect
import math
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import cast

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
# Used both directly (via ``_transport_disposition``, step 2) and as the
# cause-inspection allowlist for the openai/anthropic wrappers (steps 1 and 7).
_PRE_SEND_HTTPX = (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ConnectError)
_PRE_SEND_TRANSPORT_NAMES = frozenset({"ConnectTimeout", "PoolTimeout", "ConnectError"})

# Sentinel for static attribute lookups: distinguishes "absent" from a real
# ``None`` export without ever running the owner's attribute machinery.
_STATIC_MISSING = object()


def _static_class_name(cls: type) -> str:
    """Read ``cls.__name__`` without executing a metaclass ``__name__`` property.

    Classification runs inside the dispatch except-handler on an ARBITRARY
    third-party exception, so every attribute read here must be total: a
    metaclass that shadows ``__name__`` with a property must not raise over the
    original provider exception. ``type.__dict__["__name__"]`` is the real
    getset descriptor, bound directly to the class.
    """
    return cast(str, type.__dict__["__name__"].__get__(cls))


def _httpx2_transport_names(exc: BaseException) -> frozenset[str]:
    """Return MRO names only for a proven ``httpx2.TransportError`` family.

    A bare class or module name is insufficient: provider and user exceptions
    can choose both. The optional module must already be loaded, its exported
    ``TransportError`` class must appear by identity in the exception MRO, and
    each returned family name must resolve back to that exact exported class.
    This keeps recognition scoped without importing Anthropic's transport
    dependency into core, and prevents an external subclass name from changing
    the send-certainty of its real httpx2 ancestor.

    Every lookup is STATIC (``inspect.getattr_static``, like ``client.py``):
    ``httpx2/__init__.py`` defines a module-level ``__getattr__`` that
    lazy-imports on some names (``main`` pulls in ``click``, an optional CLI
    dependency), so a plain ``getattr`` would let an MRO class named after one
    of those hooks raise ``ModuleNotFoundError`` out of the classifier.
    """
    httpx2_mod = sys.modules.get("httpx2")
    if httpx2_mod is None:
        return frozenset()
    mro = type(exc).__mro__
    transport_error_type = inspect.getattr_static(httpx2_mod, "TransportError", _STATIC_MISSING)
    if not isinstance(transport_error_type, type) or not any(
        mro_class is transport_error_type for mro_class in mro
    ):
        return frozenset()
    family_names: set[str] = set()
    for mro_class in mro:
        name = _static_class_name(mro_class)
        if inspect.getattr_static(httpx2_mod, name, _STATIC_MISSING) is mro_class:
            family_names.add(name)
    return frozenset(family_names)


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

    Duck-typed: never ``isinstance`` on a provider SDK class (they may be absent
    or differ across versions). Certainty-first: a PROVEN httpx/httpx2 transport
    class settles whether the request could have landed before any class name or
    numeric status is consulted; everything unrecognized falls through to the
    status check. See the module docstring for why the branch ordering below is
    load-bearing (the double-spend trap).
    """
    # MRO names let us match provider classes by name without importing them.
    names = {_static_class_name(cls) for cls in type(exc).__mro__}

    # 1. APITimeoutError subclasses APIConnectionError — it MUST win first.
    #    But the openai/anthropic SDKs wrap EVERY httpx.TimeoutException in it,
    #    and ConnectTimeout/PoolTimeout subclass TimeoutException — so the class
    #    name alone does NOT prove the request landed. Inspect the chained cause
    #    exactly like step 7: a provably pre-send cause is failover-safe.
    #    The DEFAULT here is the MIRROR IMAGE of step 7's: a bare timeout with no
    #    inspectable cause is canonically a READ timeout (post-send), whereas a
    #    bare connection wrapper is canonically connect-refused (pre-send).
    if "APITimeoutError" in names:
        cause = exc.__cause__ or exc.__context__
        if cause is not None and _transport_disposition(cause) is Disposition.FAILOVER:
            return Disposition.FAILOVER
        return Disposition.POST_SEND_AMBIGUOUS

    # 2. A PROVEN httpx/httpx2 transport class settles send certainty on its
    #    own — read/write timeouts and protocol drops are post-send, while
    #    ConnectTimeout/PoolTimeout/ConnectError are provably pre-send. ONE
    #    branch serves BOTH stacks (httpx1 by isinstance, httpx2 by gated MRO
    #    identity) so anthropic's httpx2 exceptions are never classified at a
    #    later position than their httpx1 twins. It sits above the by-name and
    #    numeric-status branches because transport-class certainty outranks a
    #    status attached to the same exception; it cannot shadow them, since
    #    _transport_disposition returns None outside the two families.
    transport_disposition = _transport_disposition(exc)
    if transport_disposition is not None:
        return transport_disposition

    # 3. botocore's ReadTimeoutError is the same post-send case as an httpx
    #    read timeout (note: it subclasses HTTPClientError, NOT botocore's
    #    ConnectionError). It carries no status, so it classifies by name.
    if "ReadTimeoutError" in names:
        return Disposition.POST_SEND_AMBIGUOUS

    # 4. Provably pre-send botocore transport failures — safe to failover.
    #    These are the boto equivalents of _PRE_SEND_HTTPX (never the bare
    #    "ConnectionError" — that name collides with the Python builtin).
    if names & _BOTOCORE_PRE_SEND_NAMES:
        return Disposition.FAILOVER

    # 5. Post-send by NAME, before the status check: Bedrock's
    #    ModelTimeoutException (408) and ModelErrorException (424) carry
    #    request-shaped statuses but mean the model RAN — classifying them by
    #    status would skip reservation reconciliation (silent double-count).
    #    ConnectionClosedError is botocore's mid-response connection drop.
    if names & _POST_SEND_BY_NAME:
        return Disposition.POST_SEND_AMBIGUOUS

    # 6. Numeric status classification (OpenAI/Anthropic .status_code,
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

    # 7. Provider connection-error wrapper (no status, not itself a timeout).
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

    # 8. Default: never failover into the unknown.
    return Disposition.FAIL_FAST


def retry_after_seconds(exc: BaseException) -> float | None:
    """Return a non-negative ``Retry-After`` delay (seconds) for an HTTP 429, else None.

    Duck-typed exactly like :func:`classify_exception`, and purely status-driven
    — never imports a provider SDK. Only HTTP 429 qualifies (the provider explicitly asked
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
