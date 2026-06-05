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

Consequently the checks MUST run in this exact order, narrowest-and-safest
first:

  1. ``APITimeoutError`` (by MRO name)        -> POST_SEND_AMBIGUOUS
  2. httpx ReadTimeout / WriteTimeout         -> POST_SEND_AMBIGUOUS
  3. httpx ConnectTimeout / PoolTimeout /
     ConnectError (provably pre-send)         -> FAILOVER
  4. httpx TimeoutException (any other)       -> POST_SEND_AMBIGUOUS
  5. numeric status (.status_code, then .code):
       429 / 529 -> FAILOVER
       4xx       -> FAIL_FAST
       5xx       -> POST_SEND_AMBIGUOUS
  6. ``APIConnectionError`` (by MRO name) -> classify by its chained httpx cause:
       pre-send httpx (ConnectError/ConnectTimeout/PoolTimeout) -> FAILOVER
       any other httpx TransportError cause (post-send possible) -> POST_SEND_AMBIGUOUS
       no inspectable httpx cause -> FAILOVER (canonical connect-refused outage)
  7. httpx TransportError (any other)         -> POST_SEND_AMBIGUOUS
  8. default                                  -> FAIL_FAST

Steps 1 and 2 MUST precede step 3 (timeout-before-connection). Step 3 MUST
precede step 4 (specific pre-send timeouts before the generic ambiguous
timeout). Step 6 (bare connection failure) MUST come AFTER the status check so a
connection error that somehow carries a status is classified by status first.
The cause-inspection in step 6 closes a subtle double-spend vector: the
openai/anthropic ``APIConnectionError`` wrapper covers BOTH provably pre-send
failures AND post-send transport drops (a server disconnect mid-response
surfaces as ``httpx.RemoteProtocolError``), so the class name alone does not
prove the request never landed.
"""

from __future__ import annotations

from enum import StrEnum

import httpx


class Disposition(StrEnum):
    """How the dispatch loop should react to a transport exception."""

    FAILOVER = "failover"
    """Provably pre-send rejection. Safe to advance to the next candidate."""

    POST_SEND_AMBIGUOUS = "post_send_ambiguous"
    """Request may have run. Re-raise the original exception; never failover."""

    FAIL_FAST = "fail_fast"
    """Same error everywhere (4xx / refusal / unknown). Stop the chain."""


def _numeric_status(exc: BaseException) -> int | None:
    """Return the first int among ``exc.status_code`` then ``exc.code``.

    OpenAI/Anthropic carry the HTTP status on ``status_code``; Google's
    ``APIError`` carries it on ``code``. ``bool`` subclasses ``int``, so a
    truthy ``status_code = True`` would otherwise be misread as HTTP 1 — guard
    against it explicitly.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
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
    #    A read timeout is post-send and ambiguous; never failover.
    if "APITimeoutError" in names:
        return Disposition.POST_SEND_AMBIGUOUS

    # 2. httpx read/write timeouts are post-send: bytes were already sent.
    if isinstance(exc, httpx.ReadTimeout | httpx.WriteTimeout):
        return Disposition.POST_SEND_AMBIGUOUS

    # 3. Provably pre-send transport failures — safe to failover.
    #    ConnectTimeout subclasses TimeoutException, so it MUST be checked
    #    before the generic TimeoutException branch (step 4) below.
    if isinstance(exc, httpx.ConnectTimeout | httpx.PoolTimeout | httpx.ConnectError):
        return Disposition.FAILOVER

    # 4. Any other timeout we cannot prove pre-send -> ambiguous.
    if isinstance(exc, httpx.TimeoutException):
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
    #    the chained httpx cause: only a pre-send httpx class is failover-safe;
    #    any other transport cause is post-send-possible and stays ambiguous.
    #    Real SDK exceptions always chain a cause; a bare wrapper with no cause is
    #    the canonical connect-refused outage case failover exists to handle.
    if "APIConnectionError" in names:
        cause = exc.__cause__ or exc.__context__
        if isinstance(cause, httpx.ConnectTimeout | httpx.PoolTimeout | httpx.ConnectError):
            return Disposition.FAILOVER
        if isinstance(cause, httpx.TransportError):
            return Disposition.POST_SEND_AMBIGUOUS
        return Disposition.FAILOVER

    # 7. Other transport errors (e.g. RemoteProtocolError mid-flight) -> ambiguous.
    if isinstance(exc, httpx.TransportError):
        return Disposition.POST_SEND_AMBIGUOUS

    # 8. Default: never failover into the unknown.
    return Disposition.FAIL_FAST
