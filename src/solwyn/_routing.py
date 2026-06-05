"""Pure, sans-I/O provider selection (the Router / SelectionPolicy seam).

A ``SelectionPolicy`` is a pure function over an ordered list of
``ProviderCandidate`` snapshots plus a ``RoutingRequest``: it returns the
candidates in attempt order, with all unavailable targets dropped. Returning a
*list* (not a single provider) is what lets latency/cost policies slot in later
as drop-in reorderings — they never touch I/O, circuit breakers, or budget.

Critical correctness rule: ``circuit_breaker.admit()`` MUTATES
breaker state (an OPEN-but-recovery-eligible breaker flips to HALF_OPEN). The
router therefore orders purely on the non-mutating ``breaker_state`` /
``recovery_eligible`` snapshots captured into each candidate. It NEVER calls
``admit()`` or otherwise mutates a breaker — that happens exactly once, on
the single candidate actually being attempted, outside this module.

No httpx, no provider SDKs, no logging: this is a leaf module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

from solwyn._types import CircuitState, ProviderName

if TYPE_CHECKING:
    # Import-cycle guard: ``_registry`` imports nothing here, but referencing
    # ProviderRuntime at runtime would pull _registry -> providers in. With
    # ``from __future__ import annotations`` the dataclass field annotation is a
    # string and is never evaluated at runtime, so the type-only import suffices.
    from solwyn._registry import ProviderRuntime


@dataclass(frozen=True)
class ProviderCandidate:
    """An immutable, non-mutating snapshot of one provider link for ordering.

    ``breaker_state`` and ``recovery_eligible`` are read from the breaker WITHOUT
    mutating it (unlike ``admit()``). ``translatable`` is a FORWARD-LOOKING
    routing seam, currently ALWAYS ``True`` (fix [H]): the design eager-aborts the
    WHOLE chain on an untranslatable feature at the first cross-provider hop
    (``UntranslatableRequestError``), so an untranslatable request is never DEMOTED
    in routing — it fails loud before dispatch. The field (and the ``not
    translatable`` sort term in ``_health_key``) are retained as an intentional
    seam — like ``price_hint`` — so a future per-target translatability predicate
    could rank a translatable target ahead of an untranslatable one within a
    health tier without touching the dispatch loop. Do NOT delete it.
    ``price_hint`` is populated from a SERVER-provided relative-price signal
    — the SDK never computes price. ``latency_p50`` is the observed p50 latency
    (ms) for this provider, ``None`` until enough samples.
    """

    runtime: ProviderRuntime
    breaker_state: CircuitState
    recovery_eligible: bool
    translatable: bool
    price_hint: float | None = None
    latency_p50: float | None = None


class RoutingRequest(BaseModel):
    """Inputs a policy may consider when ordering candidates (additive-only)."""

    model_config = ConfigDict(extra="forbid")

    requested_provider: ProviderName
    estimated_input_tokens: int = 0


class SelectionPolicy(Protocol):
    """Pure, side-effect-free ordering of candidates into attempt order."""

    def order(
        self, candidates: list[ProviderCandidate], req: RoutingRequest
    ) -> list[ProviderCandidate]:
        """Return runtimes in attempt order; ``[]`` => all targets unavailable."""
        ...


_STATE_PRIORITY = {
    CircuitState.CLOSED: 0,
    CircuitState.HALF_OPEN: 1,
    CircuitState.OPEN: 2,
}


def _usable(candidates: list[ProviderCandidate]) -> list[ProviderCandidate]:
    """Drop OPEN breakers that are not recovery-eligible (shared health filter).

    Every policy reuses THIS one filter so health filtering can never drift: a
    candidate that is OPEN-not-eligible is unavailable and must never be
    attempted, regardless of how a policy then orders the survivors.
    """
    return [
        c for c in candidates if c.breaker_state is not CircuitState.OPEN or c.recovery_eligible
    ]


def _health_key(candidate: ProviderCandidate) -> tuple[int, bool]:
    """Shared health-tier sort key: CLOSED < HALF_OPEN < recovery-eligible OPEN.

    The secondary ``not translatable`` term sinks an untranslatable target below
    a translatable one WITHIN the same state tier (a translatable target is
    always preferable when both are equally healthy). Reused by every policy so
    none can promote an untranslatable target above a translatable one in the
    same usable set, nor jump an unhealthier candidate ahead of a healthier one.

    Forward-looking seam (fix [H]): ``translatable`` is currently ALWAYS ``True``
    (see ``ProviderCandidate``) because untranslatable requests eager-abort the
    chain rather than being demoted here, so this term is presently a
    no-op. It is retained so a future per-target translatability predicate slots
    in with zero dispatch changes.
    """
    return (_STATE_PRIORITY[candidate.breaker_state], not candidate.translatable)


class HealthBasedPolicy:
    """v1 default policy.

    Keeps configured (input) order within a tier, ranking CLOSED before
    HALF_OPEN before recovery-eligible OPEN, and never promotes an untranslatable
    target above a translatable one in the same state tier. OPEN breakers that
    are not recovery-eligible are dropped.

    PURE: orders on the non-mutating ``breaker_state`` / ``recovery_eligible``
    snapshots only — never calls ``admit()`` or mutates a breaker.
    """

    def order(
        self, candidates: list[ProviderCandidate], req: RoutingRequest
    ) -> list[ProviderCandidate]:
        # sorted() is stable, so configured order is preserved within a tier.
        return sorted(_usable(candidates), key=_health_key)


class LatencyPolicy:
    """Latency-aware drop-in policy (pure; sans-I/O).

    Applies the SAME health filtering as ``HealthBasedPolicy`` (drop
    OPEN-not-eligible; CLOSED before HALF_OPEN before recovery-eligible OPEN;
    untranslatable sinks below translatable in a tier), then within that usable
    health-ordered set prefers the LOWER observed p50 latency. A candidate whose
    ``latency_p50`` is ``None`` (not enough samples yet) sorts AFTER any
    candidate with a known p50 — unknown latency never jumps the queue.

    PURE drop-in: only REORDERS the usable set. Never calls ``admit()``,
    never mutates a breaker, does no I/O, and computes no price. Swapping this in
    for ``HealthBasedPolicy`` changes routing order with zero dispatch changes.
    """

    def order(
        self, candidates: list[ProviderCandidate], req: RoutingRequest
    ) -> list[ProviderCandidate]:
        # Stable secondary sort by latency keeps the health tier dominant: a
        # candidate with a known low p50 still never outranks a healthier tier.
        # (None p50 -> +inf sentinel so it sinks below every known p50.)
        return sorted(
            _usable(candidates),
            key=lambda c: (
                *_health_key(c),
                c.latency_p50 if c.latency_p50 is not None else float("inf"),
            ),
        )


class CostPolicy:
    """Cost-aware drop-in policy (pure; sans-I/O; NO price math).

    Applies the SAME health filtering as ``HealthBasedPolicy``, then within that
    usable health-ordered set prefers the LOWER ``price_hint``. ``price_hint`` is
    a SERVER-provided RELATIVE price signal (from ``BudgetCheckResponse``); the
    SDK never computes, derives, or arithmetically combines price — it only reads
    the hint and orders by it. A candidate whose ``price_hint`` is ``None`` sorts
    AFTER any candidate with a known hint. If NO candidate carries a hint (the
    server has not provided one yet), this falls back to the plain
    ``HealthBasedPolicy`` order so behaviour is identical to today.

    PURE drop-in: only REORDERS the usable set. Never calls ``admit()``,
    never mutates a breaker, does no I/O, and does no price arithmetic.
    """

    def order(
        self, candidates: list[ProviderCandidate], req: RoutingRequest
    ) -> list[ProviderCandidate]:
        usable = _usable(candidates)
        if not any(c.price_hint is not None for c in usable):
            # Server has fed no price hints yet — identical to health order.
            return sorted(usable, key=_health_key)
        # Stable secondary sort by the server hint keeps the health tier dominant.
        # (None hint -> +inf sentinel so it sinks below every known hint.)
        return sorted(
            usable,
            key=lambda c: (
                *_health_key(c),
                c.price_hint if c.price_hint is not None else float("inf"),
            ),
        )
