"""One client's binding to its Solwyn project (spec section 6.2).

A ``ProjectBinding`` is shared by the budget enforcer, the metadata reporter
and the exit drain of one client. It owns:

- the header every request sends (``Solwyn-Project``: the pinned ``proj_`` id
  once ``GET /api/v1/auth/key`` answered 200, the configured selector before);
- the validation state (``unvalidated``, ``valid``, ``refused``) and the stored
  refusal with its effect (``block`` or ``allow_unmetered``);
- the generation fence: every request captures ``d = generation`` when it
  builds its headers, and every mutation takes ``d``. A validation success
  applies only when nothing changed since its dispatch; a refusal may always
  tighten, but may relax only when it was dispatched after the stored refusal
  was recorded. So a delayed response never weakens or clears a newer refusal.

The binding is sans-I/O and thread-safe. The enforcers perform the
``/auth/key`` request and call back into it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from solwyn._api_key_errors import Posture, Refusal, classify, refusal_message
from solwyn._constants import PROJECT_HEADER
from solwyn._types import AuthKeyResponse
from solwyn.exceptions import ApiKeyRefusedError

logger = logging.getLogger(__name__)

BindingState = Literal["unvalidated", "valid", "refused"]
Effect = Literal["block", "allow_unmetered"]
ValidationOutcome = Literal["valid", "refused", "unavailable", "fenced"]
ExitVerdict = Literal["keep", "refused", "rerun"]

# Retry cadence of the gate's automatic validation (spec section 6.2).
UNVALIDATED_RETRY_S = 30.0
REFUSED_RETRY_S = 60.0

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


class ProjectBinding:
    """Header, validation state and refusal fence for one client."""

    def __init__(self, api_key: str, selector: str) -> None:
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

    def refuse(self, refusal: Refusal, dispatched_at: int) -> bool:
        """Record a refusal under the fence; True when the stored refusal changed."""
        message, field = refusal_message(refusal, self.selector)
        with self._lock:
            effect = self._effect_locked(refusal)
            stored = self._refusal
            if stored is not None:
                weaker = stored.effect == "block" and effect == "allow_unmetered"
                if weaker and dispatched_at < self._refusal_gen:
                    # An answer older than the stored refusal carries staler
                    # server state: it may never loosen it.
                    return False
                if (stored.code, stored.message, stored.effect, stored.posture) == (
                    refusal.code,
                    message,
                    effect,
                    refusal.posture,
                ):
                    return False
            self._refusal = StoredRefusal(
                code=refusal.code,
                message=message,
                field=field,
                effect=effect,
                posture=refusal.posture,
            )
            self._state = "refused"
            self._generation += 1
            self._refusal_gen = self._generation
            self._next_validate_at = self._clock() + REFUSED_RETRY_S
            first_for_code = refusal.code not in self._logged_codes
            self._logged_codes.add(refusal.code)
            listeners = list(self._listeners)
        if first_for_code:
            logger.error("solwyn.configuration_error.%s: %s", refusal.code, message)
        for listener in listeners:
            listener()
        return True

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
                self.refuse(Refusal(code="read_only_key"), dispatched_at)
                return "refused", auth
            return self._validation_succeeded(auth, dispatched_at), auth
        refusal = classify(status, content)
        if refusal is None:
            self.validation_unavailable()
            return "unavailable", None
        self.refuse(refusal, dispatched_at)
        return "refused", None

    def _validation_succeeded(self, auth: AuthKeyResponse, dispatched_at: int) -> ValidationOutcome:
        with self._lock:
            if dispatched_at != self._generation:
                # Something changed since dispatch (typically a refusal): this
                # answer is stale. The gate revalidates later.
                return "fenced"
            recovered = self._state == "refused"
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
            retry = UNVALIDATED_RETRY_S if self._state == "unvalidated" else REFUSED_RETRY_S
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
