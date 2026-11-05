"""One client's binding to its Solwyn project (spec section 6.2).

A ``ProjectBinding`` is shared by the budget enforcer, the metadata reporter
and the exit drain of one client. It owns:

- the header every request sends (``Solwyn-Project``: the pinned ``proj_`` id
  once ``GET /api/v1/auth/key`` answered 200, the configured selector before);
- the validation state (``unvalidated``, ``valid``, ``refused``) and the stored
  refusal with its effect (``block`` or ``allow_unmetered``);
- one rule for every change: a refusal only ever TIGHTENS the stored state.
  It is relaxed (``block`` to ``allow_unmetered``, or cleared) only by the
  classified answer of a request dispatched after the stored refusal was
  recorded, in practice the single-flight ``/auth/key`` revalidation. Every
  request captures ``d = generation`` when it builds its headers, and every
  mutation takes ``d``. A data-plane 401 status records a block before its
  body is read (a body that stalls or is cancelled cannot fail open) and
  schedules an immediate revalidation, whose answer may relax it; that 401's
  own body never does.

The binding is sans-I/O and thread-safe. The enforcers perform the
``/auth/key`` request and call back into it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from solwyn._api_key_errors import (
    Posture,
    Refusal,
    classify,
    classify_exception,
    refusal_message,
)
from solwyn._constants import AUTH_KEY_PATH, PROJECT_HEADER
from solwyn._types import AuthKeyResponse
from solwyn._validation import validate_project_selector
from solwyn.exceptions import ApiKeyRefusedError, ConfigurationError

logger = logging.getLogger(__name__)

BindingState = Literal["unvalidated", "valid", "refused"]
Effect = Literal["block", "allow_unmetered"]
ValidationOutcome = Literal["valid", "refused", "unavailable", "fenced"]
ExitVerdict = Literal["keep", "refused", "rerun"]

# Retry cadence of the gate's automatic validation (spec section 6.2).
UNVALIDATED_RETRY_S = 30.0
# A re-check whose own answer was a refusal backs off 60 s ...
REFUSED_RETRY_S = 60.0
# ... but a re-check that could not reach Solwyn while refused retries after
# 5 s (still single-flight), so a 401-triggered block on an alert-only project
# does not outlast a short Solwyn outage by a minute.
REFUSED_OUTAGE_RETRY_S = 5.0

# Request extension the 401 status hook sets, so the caller can tell an unread
# 401 (a refusal already recorded) from an outage.
_STATUS_401_EXTENSION = "solwyn.status_401"
# The ``refusal_gen`` this request's own 401 status recorded: only the same
# request's body coalesces with it; every other answer advances the ordering.
_REFUSAL_GEN_EXTENSION = "solwyn.refusal_gen"

# Codes whose refusal may run calls unmetered once the client has validated:
# a key that stopped working on a project that promises to block nothing.
_RELAXABLE_CODES = frozenset({"api_key_expired", "api_key_invalid"})


@dataclass(frozen=True, slots=True)
class StoredRefusal:
    """The refusal a binding is holding, with the effect it decided."""

    code: str
    message: str
    field: str
    effect: Effect
    posture: Posture

    def error(self) -> ApiKeyRefusedError:
        return ApiKeyRefusedError(self.message, code=self.code, field=self.field)


class _Flight:
    """One ``/auth/key`` request other gates may wait on (single-flight)."""

    __slots__ = ("done", "future", "loop")

    def __init__(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self.done = threading.Event()
        self.loop = loop
        self.future: asyncio.Future[None] | None = (
            loop.create_future() if loop is not None else None
        )

    def finish(self) -> None:
        self.done.set()
        future = self.future
        if future is not None and not future.done():
            future.set_result(None)


def _request_of(exc: BaseException) -> httpx.Request | None:
    if not isinstance(exc, httpx.HTTPError):
        return None
    try:
        return exc.request
    except RuntimeError:
        return None


class _ValidationBodyGuard(httpx.SyncByteStream, httpx.AsyncByteStream):
    """The body of a 401 answer to ``/auth/key``: refuse if it cannot be read.

    A validation 401 does not block at its status (its classified body is the
    one answer that may relax a stored refusal), so a body that times out, is
    cancelled or fails in any other way records the blocking refusal here.
    """

    def __init__(self, inner: Any, binding: ProjectBinding) -> None:
        self._inner = inner
        self._binding = binding

    def _refuse_unread(self) -> None:
        self._binding.refuse(
            Refusal(code="api_key_invalid"), self._binding.generation, from_validation=True
        )

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self._inner
        except BaseException:
            self._refuse_unread()
            raise

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                yield chunk
        except BaseException:
            self._refuse_unread()
            raise

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


def _own_refusal_gen(exc: BaseException) -> int | None:
    request = _request_of(exc)
    if request is None:
        return None
    try:
        value = request.extensions.get(_REFUSAL_GEN_EXTENSION)
    except Exception:
        return None
    return value if isinstance(value, int) else None


def saw_401(exc: BaseException) -> bool:
    """Whether ``exc`` ended a request whose 401 status the hook already saw."""
    request = _request_of(exc)
    if request is None:
        return False
    try:
        return request.extensions.get(_STATUS_401_EXTENSION) is True
    except Exception:
        return False


def sync_status_hook(binding: ProjectBinding) -> Callable[[httpx.Response], None]:
    """``event_hooks["response"]`` entry for a sync control-plane client."""

    def hook(response: httpx.Response) -> None:
        binding.observe_status(response)

    return hook


def async_status_hook(binding: ProjectBinding) -> Callable[[httpx.Response], Awaitable[None]]:
    """``event_hooks["response"]`` entry for an async control-plane client."""

    async def hook(response: httpx.Response) -> None:
        binding.observe_status(response)

    return hook


class ProjectBinding:
    """Header, validation state and refusal fence for one client."""

    def __init__(self, api_key: str, selector: str) -> None:
        try:
            selector = validate_project_selector(selector)
        except ValueError as exc:
            raise ConfigurationError(str(exc), field="project") from exc
        self.api_key = api_key
        self.selector = selector
        self._lock = threading.Lock()
        self._clock: Callable[[], float] = time.monotonic
        self._pinned_id: str | None = None
        self._state: BindingState = "unvalidated"
        self._ever_valid = False
        self._refusal: StoredRefusal | None = None
        self._generation = 0
        self._refusal_gen = 0
        self._next_validate_at = 0.0
        self._flight: _Flight | None = None
        self._last_auth: AuthKeyResponse | None = None
        self._listeners: list[Callable[[], None]] = []
        self._logged_codes: set[str] = set()

    def _reset_after_fork_in_child(self) -> None:
        """A forked child keeps the state but never an inherited lock or flight."""
        self._lock = threading.Lock()
        self._flight = None

    # ── reads ──────────────────────────────────────────────────────────

    @property
    def state(self) -> BindingState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def pinned_id(self) -> str | None:
        with self._lock:
            return self._pinned_id

    @property
    def refused(self) -> bool:
        """True in any refused state (``block`` or ``allow_unmetered``).

        While refused the SDK sends Solwyn nothing but the single-flight
        ``/auth/key`` revalidation: data-plane items are dropped locally.
        """
        with self._lock:
            return self._state == "refused"

    @property
    def ever_valid(self) -> bool:
        with self._lock:
            return self._ever_valid

    @property
    def refusal(self) -> StoredRefusal | None:
        with self._lock:
            return self._refusal

    @property
    def last_auth(self) -> AuthKeyResponse | None:
        """The body of the last applied ``/auth/key`` success."""
        with self._lock:
            return self._last_auth

    @property
    def refusal_gen(self) -> int:
        with self._lock:
            return self._refusal_gen

    def request_headers(self, *, content_type: bool = True) -> tuple[dict[str, str], int]:
        """Headers for one Solwyn API request, and that request's dispatch generation."""
        with self._lock:
            project = self._pinned_id if self._pinned_id is not None else self.selector
            dispatched_at = self._generation
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            PROJECT_HEADER: project,
        }
        if content_type:
            headers["Content-Type"] = "application/json"
        return headers, dispatched_at

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Call ``listener`` (outside the lock) after a refusal or a recovery."""
        with self._lock:
            self._listeners.append(listener)

    # ── mutations ──────────────────────────────────────────────────────

    def _effect_locked(self, refusal: Refusal) -> Effect:
        if (
            refusal.code in _RELAXABLE_CODES
            and refusal.posture == "alert_only"
            and self._ever_valid
        ):
            return "allow_unmetered"
        return "block"

    def refuse(
        self,
        refusal: Refusal,
        dispatched_at: int,
        *,
        from_validation: bool = False,
        own_gen: int | None = None,
    ) -> bool:
        """Record a refusal; True when the stored refusal changed.

        Tightening (or an equally strong refusal with new details) always
        applies. Relaxing ``block`` to ``allow_unmetered`` applies only when
        the answer was dispatched after the stored refusal was recorded
        (``dispatched_at >= refusal_gen``). ``from_validation`` marks an
        ``/auth/key`` answer: it keeps the regular retry cadence, while a new
        401-class refusal from any other request schedules an immediate
        revalidation, so an alert-only project relaxes promptly.

        Every independent refusal advances the ordering (``refusal_gen``),
        even one identical to the stored refusal, so no answer dispatched
        before it can relax or clear it. Only the body of the request whose own
        401 status recorded the stored refusal (``own_gen``) coalesces with it.
        """
        message, field = refusal_message(refusal, self.selector)
        with self._lock:
            changed, log_first, listeners = self._record_locked(
                refusal,
                message,
                field,
                dispatched_at,
                from_validation=from_validation,
                own_gen=own_gen,
            )
        if log_first:
            logger.error("solwyn.configuration_error.%s: %s", refusal.code, message)
        for listener in listeners:
            listener()
        return changed

    def _record_locked(
        self,
        refusal: Refusal,
        message: str,
        field: str,
        dispatched_at: int,
        *,
        from_validation: bool,
        own_gen: int | None = None,
    ) -> tuple[bool, bool, list[Callable[[], None]]]:
        """Apply one refusal under the lock: (changed, log it, listeners)."""
        now = self._clock()
        if from_validation:
            self._next_validate_at = now + REFUSED_RETRY_S
        effect = self._effect_locked(refusal)
        stored = self._refusal
        own = own_gen is not None and own_gen == self._refusal_gen
        if stored is not None:
            relaxes = stored.effect == "block" and effect == "allow_unmetered"
            if relaxes and dispatched_at < self._refusal_gen:
                return False, False, []
            identical = (stored.code, stored.message, stored.effect, stored.posture) == (
                refusal.code,
                message,
                effect,
                refusal.posture,
            )
            if own and (identical or stored.effect == effect):
                # This request's own status recorded the stored refusal: its
                # body may only add details, never advance the ordering.
                if not identical:
                    self._refusal = StoredRefusal(
                        code=refusal.code,
                        message=message,
                        field=field,
                        effect=effect,
                        posture=refusal.posture,
                    )
                return not identical, False, []
            if identical:
                # An independent confirmation: same state, newer ordering.
                return False, False, self._advance_locked(refusal, from_validation, now)
        self._refusal = StoredRefusal(
            code=refusal.code,
            message=message,
            field=field,
            effect=effect,
            posture=refusal.posture,
        )
        self._state = "refused"
        listeners = self._advance_locked(refusal, from_validation, now)
        log_first = refusal.code not in self._logged_codes
        if log_first:
            self._logged_codes.add(refusal.code)
        return True, log_first, listeners

    def _advance_locked(
        self, refusal: Refusal, from_validation: bool, now: float
    ) -> list[Callable[[], None]]:
        """Move the refusal ordering: nothing dispatched before may relax it."""
        self._generation += 1
        self._refusal_gen = self._generation
        if not from_validation:
            immediate = refusal.code in _RELAXABLE_CODES
            self._next_validate_at = now if immediate else now + REFUSED_RETRY_S
        return list(self._listeners)

    def observe_status(self, response: httpx.Response) -> None:
        """Response hook: act on a 401 status before its body is read.

        Any data-plane 401 records a blocking ``api_key_invalid`` (posture
        ``unknown``) at once, unless a block is already stored: a body that
        stalls, times out or is cancelled can then never fail open, and that
        body, however late, never relaxes it. A 401 answer to ``/auth/key`` is
        the revalidation whose classified body may relax a stored refusal, so
        it does not block at its status; its body is guarded instead.
        """
        if response.status_code != 401:
            return
        request = response.request
        request.extensions[_STATUS_401_EXTENSION] = True
        if request.url.path.endswith(AUTH_KEY_PATH):
            response.stream = _ValidationBodyGuard(response.stream, self)
            return
        refusal = Refusal(code="api_key_invalid")
        message, field = refusal_message(refusal, self.selector)
        with self._lock:
            stored = self._refusal
            if stored is not None and stored.effect == "block":
                # Already blocking, with possibly more precise details: keep
                # them, but advance the ordering (an independent refusal).
                log_first = False
                listeners = self._advance_locked(refusal, False, self._clock())
            else:
                _changed, log_first, listeners = self._record_locked(
                    refusal, message, field, self._generation, from_validation=False
                )
            request.extensions[_REFUSAL_GEN_EXTENSION] = self._refusal_gen
        if log_first:
            logger.error("solwyn.configuration_error.%s: %s", refusal.code, message)
        for listener in listeners:
            listener()

    def note_refusal(self, exc: BaseException, dispatched_at: int) -> bool:
        """Record the key or project refusal ``exc`` carries; True when it was one.

        A 401 whose body never arrived was already recorded at its status, so
        it counts as a refusal (never an outage) here too.
        """
        refusal = classify_exception(exc)
        if refusal is not None:
            self.refuse(refusal, dispatched_at, own_gen=_own_refusal_gen(exc))
            return True
        return saw_401(exc)

    def apply_auth_key(
        self, status: int, content: bytes, dispatched_at: int
    ) -> tuple[ValidationOutcome, AuthKeyResponse | None]:
        """Fold one ``/auth/key`` answer into the binding."""
        if status == 200:
            try:
                auth = AuthKeyResponse.model_validate_json(content)
            except ValidationError:
                self.validation_unavailable()
                return "unavailable", None
            if auth.scope != "full":
                self.refuse(Refusal(code="read_only_key"), dispatched_at, from_validation=True)
                return "refused", auth
            return self._validation_succeeded(auth, dispatched_at), auth
        refusal = classify(status, content)
        if refusal is None:
            self.validation_unavailable()
            return "unavailable", None
        self.refuse(refusal, dispatched_at, from_validation=True)
        return "refused", None

    def _validation_succeeded(self, auth: AuthKeyResponse, dispatched_at: int) -> ValidationOutcome:
        with self._lock:
            if dispatched_at != self._generation:
                # Something changed since dispatch (typically a refusal): this
                # answer is stale. The gate revalidates later.
                return "fenced"
            # Retained authority (held leases, cached allows) belongs to the
            # project it was granted for: a recovery or a different pinned id
            # (for example after an outage and a rename) invalidates it.
            recovered = self._state == "refused" or self._pinned_id != auth.project_id
            self._pinned_id = auth.project_id
            self._last_auth = auth
            self._state = "valid"
            self._ever_valid = True
            self._refusal = None
            self._generation += 1
            listeners = list(self._listeners) if recovered else []
        for listener in listeners:
            listener()
        return "valid"

    def validation_unavailable(self) -> None:
        """No answer: keep the state and schedule the next automatic attempt."""
        with self._lock:
            retry = UNVALIDATED_RETRY_S if self._state == "unvalidated" else REFUSED_OUTAGE_RETRY_S
            self._next_validate_at = self._clock() + retry

    # ── gate and exit check ────────────────────────────────────────────

    def revalidation_due(self) -> bool:
        with self._lock:
            return self._state != "valid" and self._clock() >= self._next_validate_at

    def begin_flight(self, loop: asyncio.AbstractEventLoop | None = None) -> tuple[_Flight, bool]:
        """Join the in-flight validation, or start one; True when this caller leads."""
        with self._lock:
            if self._flight is not None:
                return self._flight, False
            flight = _Flight(loop)
            self._flight = flight
            return flight, True

    def end_flight(self, flight: _Flight) -> None:
        with self._lock:
            if self._flight is flight:
                self._flight = None
        flight.finish()

    def admit(self) -> StoredRefusal | int:
        """The stored refusal while refused, else the generation to fence on."""
        with self._lock:
            if self._state == "refused" and self._refusal is not None:
                return self._refusal
            return self._generation

    def exit_verdict(self, gate_generation: int) -> ExitVerdict:
        """Whether a result decided since the gate may still be returned."""
        with self._lock:
            if self._state == "refused":
                return "refused"
            if self._generation != gate_generation:
                return "rerun"
            return "keep"

    def note_unmetered_call(self, refusal: StoredRefusal) -> None:
        """Every call let through unmetered logs at error (spec section 6.4)."""
        logger.error(
            "solwyn.api_key_refused: code=%s; Solwyn cannot meter this call and lets it "
            "run because the project is alert-only. %s",
            refusal.code,
            refusal.message,
        )
