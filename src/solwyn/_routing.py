"""Pure, sans-I/O provider selection (§4.2 — the Router / SelectionPolicy seam).

A ``SelectionPolicy`` is a pure function over an ordered list of
``ProviderCandidate`` snapshots plus a ``RoutingRequest``: it returns the
candidates in attempt order, with all unavailable targets dropped. Returning a
*list* (not a single provider) is what lets latency/cost policies slot in later
as drop-in reorderings — they never touch I/O, circuit breakers, or budget.

Critical correctness rule (§4.2): ``circuit_breaker.can_proceed()`` MUTATES
breaker state (an OPEN-but-recovery-eligible breaker flips to HALF_OPEN). The
router therefore orders purely on the non-mutating ``breaker_state`` /
``recovery_eligible`` snapshots captured into each candidate. It NEVER calls
``can_proceed()`` or otherwise mutates a breaker — that happens exactly once, on
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
    mutating it (unlike ``can_proceed()``). ``translatable`` is the per-target
    request-translation predicate: always ``True`` in P1; P2 supplies the real
    per-target value. ``price_hint`` is populated later (P5) from a
    SERVER-provided relative-price signal — the SDK never computes price.
    """

    runtime: ProviderRuntime
    breaker_state: CircuitState
    recovery_eligible: bool
    translatable: bool
    price_hint: float | None = None


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


class HealthBasedPolicy:
    """v1 default policy.

    Keeps configured (input) order within a tier, ranking CLOSED before
    HALF_OPEN before recovery-eligible OPEN, and never promotes an untranslatable
    target above a translatable one in the same state tier. OPEN breakers that
    are not recovery-eligible are dropped.

    PURE: orders on the non-mutating ``breaker_state`` / ``recovery_eligible``
    snapshots only — never calls ``can_proceed()`` or mutates a breaker.
    """

    def order(
        self, candidates: list[ProviderCandidate], req: RoutingRequest
    ) -> list[ProviderCandidate]:
        prio = {
            CircuitState.CLOSED: 0,
            CircuitState.HALF_OPEN: 1,
            CircuitState.OPEN: 2,
        }
        usable = [
            c for c in candidates if c.breaker_state is not CircuitState.OPEN or c.recovery_eligible
        ]
        # sorted() is stable, so configured order is preserved within a tier.
        return sorted(usable, key=lambda c: (prio[c.breaker_state], not c.translatable))
