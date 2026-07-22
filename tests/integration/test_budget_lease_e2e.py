"""End-to-end budget-lease behaviour against a LIVE core API (PJ-2, Task S4).

Every test here names the DoD item it pins. The unit suites already own the two
items that need a clock the wire cannot provide:

* DoD 2 (a renewal never rides the hot path) — ``tests/unit/test_budget_lease.py
  ::TestLeaseRenewal::test_a_slow_renewal_never_delays_admission``
* DoD 8 (expired is not exhausted; a concurrent burst cannot overrun) —
  ``tests/unit/test_lease_state.py::TestOutageLadderExpiredLease
  ::test_expired_is_not_exhausted`` and ``::TestConcurrentBurst
  ::test_concurrent_burst_cannot_overrun``

...so they are deliberately NOT duplicated here.

Harness notes:

* Each test provisions its OWN account+project (``provision_project``) and hands
  the lease back before it finishes (``enforcer.close()`` surrenders, raw tests
  POST ``/lease/surrender``). No test depends on another's leftovers, and no
  test shares the session project — a held claim is counted against the
  project's budget while it is held, so a stray lease would starve a later test.
* Wire traffic is counted by wrapping the enforcer's own ``httpx.Client`` in a
  recording transport (``_WireCounter``): the real request still goes to the
  live API, and the count is the SDK's actual round-trips, not a proxy for them.
* Two tests reach for enforcer privates on purpose. ``_build_renewal`` +
  ``_renew_lease`` is the SAME code the async renewal worker runs; calling it
  inline is the only way to observe five successive renewals without waiting
  five ~15s refresh intervals. ``_lease.state_for`` reads the ledger the ladder
  decides from — the numbers the server sent, never numbers the SDK computed.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from conftest import (
    Credentials,
    ProjectCredentials,
    budget_status,
    burn_budget,
    provision_project,
)

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode, LeaseGrantRequest, ProviderName
from solwyn.budget import (
    _LEASE_PATH,
    _LEASE_RENEW_PATH,
    _LEASE_SURRENDER_PATH,
    BudgetEnforcer,
)

# A token-priced model the API knows: leases are token-denominated, so an
# unpriced or unit-priced model would make the run lease-INELIGIBLE by design.
MODEL = "gpt-5.5"
CHECK_PATH = "/api/v1/budgets/check"


def _run_id() -> str:
    return f"lease-e2e-{uuid.uuid4().hex[:12]}"


class _WireCounter(httpx.BaseTransport):
    """Counts control-plane round-trips while still performing them."""

    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport()
        self._lock = threading.Lock()
        self.paths: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.paths.append(request.url.path)
        return self._inner.handle_request(request)

    def count(self, path: str) -> int:
        with self._lock:
            return sum(1 for seen in self.paths if seen == path)

    def close(self) -> None:
        self._inner.close()


EnforcerFactory = Callable[..., tuple[BudgetEnforcer, _WireCounter]]


@pytest.fixture
def make_lease_enforcer() -> Iterator[EnforcerFactory]:
    """Build sync enforcers whose wire traffic is counted; close them on teardown."""
    created: list[BudgetEnforcer] = []

    def _make(
        credentials: Credentials,
        *,
        budget_mode: BudgetMode = BudgetMode.ALERT_ONLY,
        fail_open: bool = True,
        holder_id: str | None = None,
        **config: Any,
    ) -> tuple[BudgetEnforcer, _WireCounter]:
        enforcer = BudgetEnforcer(
            api_url=credentials.api_url,
            api_key=credentials.api_key,
            budget_mode=budget_mode,
            fail_open=fail_open,
            holder_id=holder_id,
            **config,
        )
        counter = _WireCounter()
        # Swap the transport, not the client contract: the enforcer keeps its
        # own timeout posture and every lease/check/surrender call is recorded.
        enforcer._http.close()
        enforcer._http = httpx.Client(transport=counter, timeout=5.0)
        created.append(enforcer)
        return enforcer, counter

    yield _make
    for enforcer in created:
        enforcer.close()


@pytest.fixture
def lease_project(api_url: str) -> ProjectCredentials:
    """A fresh alert_only project with ample budget, owned by this test alone."""
    return provision_project(api_url, name="sdk-lease", budget_limit=20.0, budget_mode="alert_only")


def _admit(
    enforcer: BudgetEnforcer,
    run_id: str,
    *,
    estimated_input_tokens: int = 100,
    estimated_output_bound: int = 1000,
    call_id: str | None = None,
) -> Any:
    """One run-scoped admission through the real ladder."""
    return enforcer.check_budget(
        estimated_input_tokens=estimated_input_tokens,
        model=MODEL,
        provider="openai",
        agent_run_id=run_id,
        call_id=call_id or str(uuid.uuid4()),
        estimated_output_bound=estimated_output_bound,
    )


def _join_renewals(enforcer: BudgetEnforcer, timeout: float = 10.0) -> None:
    """Wait for every dispatched renewal worker to land."""
    deadline = time.monotonic() + timeout
    for thread in list(enforcer._renewal_threads):
        thread.join(timeout=max(0.0, deadline - time.monotonic()))


def _grant_raw(
    credentials: Credentials,
    run_id: str,
    holder_id: str,
    *,
    fail_open: bool = True,
) -> dict[str, Any]:
    """A grant issued with the SDK's exact request bytes, outside any enforcer."""
    request = LeaseGrantRequest(
        agent_run_id=run_id,
        holder_id=holder_id,
        model=MODEL,
        provider=ProviderName.OPENAI,
        fail_open=fail_open,
        estimated_input_tokens=100,
    )
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        r = http.post(
            _LEASE_PATH,
            json=request.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        r.raise_for_status()
        body = r.json()
    if not isinstance(body, dict):
        pytest.fail(f"lease grant returned non-object JSON: {body!r}")
    return body


def _renew_raw(
    credentials: Credentials,
    lease_id: str,
    holder_id: str,
    generation: int,
    *,
    spent_tokens: int = 0,
) -> httpx.Response:
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        return http.post(
            _LEASE_RENEW_PATH,
            json={
                "lease_id": lease_id,
                "holder_id": holder_id,
                "generation": generation,
                "spent_tokens": spent_tokens,
            },
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )


def _surrender_raw(
    credentials: Credentials, lease_id: str, holder_id: str, generation: int
) -> dict[str, Any]:
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        r = http.post(
            _LEASE_SURRENDER_PATH,
            json={
                "lease_id": lease_id,
                "holder_id": holder_id,
                "generation": generation,
            },
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        r.raise_for_status()
        return dict(r.json())


def _send_confirm(credentials: Credentials, confirm_request: Any) -> None:
    """Deliver a confirm the SDK built (the reporter's exact wire bytes)."""
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        r = http.post(
            "/api/v1/budgets/confirm",
            json=confirm_request.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        r.raise_for_status()


@pytest.mark.integration
class TestLeaseBoundedRoundTrips:
    @pytest.mark.integration
    def test_n_calls_bounded_wire_traffic(
        self, lease_project: ProjectCredentials, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 1: N=20 run-scoped calls cost ONE grant and zero per-call checks."""
        enforcer, wire = make_lease_enforcer(lease_project)
        run_id = _run_id()

        for _ in range(20):
            result = _admit(enforcer, run_id)
            assert result.allowed is True
            assert result.lease_id is not None, (
                "a covered admission must draw on lease authority, not the per-call path"
            )
            assert result.reservation_id is None

        assert wire.count(_LEASE_PATH) == 1, f"expected exactly one grant: {wire.paths}"
        assert wire.count(_LEASE_RENEW_PATH) <= 1, f"renewal storm on the wire: {wire.paths}"
        assert wire.count(CHECK_PATH) == 0, (
            f"a covered run must not pay per-call checks: {wire.paths}"
        )

    @pytest.mark.integration
    def test_mid_run_hard_deny_stops(
        self, api_url: str, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 1 (runaway-run cap rule): a mid-run server deny stops the run at once.

        The lease still holds local authority when the deny lands — the next
        admission must refuse anyway, rather than spending the remainder.
        """
        credentials = provision_project(
            api_url, name="sdk-lease-middeny", budget_limit=100.0, budget_mode="hard_deny"
        )
        enforcer, wire = make_lease_enforcer(
            credentials, budget_mode=BudgetMode.HARD_DENY, fail_open=False
        )
        run_id = _run_id()

        first = _admit(enforcer, run_id)
        assert first.allowed is True
        assert first.lease_id is not None
        state = enforcer._lease.state_for(run_id)
        assert state is not None
        granted = state.granted_tokens
        assert granted > 0

        # Real server-side spend, far past limit + this lease's own claim, so the
        # run's pool (free headroom + its own outstanding claim) goes negative.
        burn_budget(credentials, input_tokens=1_000_000, output_tokens=4_000_000)

        # Deplete to ~90% on lease authority alone: that crosses the 75% renewal
        # trigger, and the renewal is what meets the server's deny.
        per_call = granted // 10 * 3
        for _ in range(3):
            drawn = _admit(enforcer, run_id, estimated_output_bound=per_call)
            assert drawn.allowed is True
            assert drawn.lease_id is not None, "depletion must come from the lease, not checks"
        _join_renewals(enforcer)

        assert wire.count(_LEASE_RENEW_PATH) >= 1, f"no renewal was sent: {wire.paths}"
        # ~10% of the grant is still unspent local authority: 1100 tokens would
        # fit comfortably. The refusal below is therefore the server's verdict,
        # not local exhaustion.
        assert granted - 3 * per_call > 1100

        final = _admit(enforcer, run_id)
        assert final.allowed is False, (
            "the admission after a mid-run hard deny must refuse, not spend the remainder"
        )

    @pytest.mark.integration
    def test_alert_only_past_cap_continues_with_warning(
        self, api_url: str, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 3 (live half): alert_only past its cap keeps running, with a warning."""
        credentials = provision_project(
            api_url, name="sdk-lease-alert", budget_limit=0.05, budget_mode="alert_only"
        )
        burn_budget(credentials, input_tokens=200_000, output_tokens=200_000)

        enforcer, wire = make_lease_enforcer(
            credentials, budget_mode=BudgetMode.ALERT_ONLY, fail_open=True
        )
        run_id = _run_id()

        for _ in range(3):
            result = _admit(enforcer, run_id)
            assert result.allowed is True
            assert result.warning is not None, (
                "an alert_only call past the cap must carry the customer's warning"
            )

        state = enforcer._lease.state_for(run_id)
        assert state is not None
        assert state.granted_tokens == 0, (
            "a project past its cap should hand back a zero-token grant, not a lease"
        )
        assert wire.count(_LEASE_PATH) == 1, f"the zero-token grant must not repeat: {wire.paths}"
        assert wire.count(CHECK_PATH) == 3, (
            f"a zero-token grant sends every call to the per-call path: {wire.paths}"
        )


@pytest.mark.integration
class TestLeasePartitioning:
    @pytest.mark.integration
    def test_two_holders_cannot_exceed_single_holder_bound(
        self, api_url: str, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 4: two holders on one run hold no more authority than one holder.

        The single-holder bound is measured first on the same project (a lone
        holder gets the whole pool). Two SDK clients then hammer ONE run
        concurrently, settling every admitted call so the server's float
        accounting sees the drawdown it would see in production. What is pinned
        is the server's PARTITION: at every instant, the token authority the two
        holders jointly hold is at most what one holder alone was granted.

        (Total lease-funded tokens over the life of the run is deliberately NOT
        the assertion — see the xfail below: settled-within-claim spend never
        reaches the counters, so lifetime drawdown is unbounded today.)
        """
        credentials = provision_project(
            api_url, name="sdk-lease-pair", budget_limit=10.0, budget_mode="hard_deny"
        )

        solo = _grant_raw(credentials, _run_id(), "solo-holder")
        solo_bound = solo["granted_tokens"]
        assert solo_bound > 0
        _surrender_raw(credentials, solo["lease_id"], "solo-holder", solo["generation"])

        run_id = _run_id()
        input_tokens, output_bound = 2_000, 8_000
        per_call = input_tokens + output_bound
        enforcers = [
            make_lease_enforcer(
                credentials,
                budget_mode=BudgetMode.HARD_DENY,
                fail_open=False,
                holder_id=f"pair-{index}-{uuid.uuid4().hex[:8]}",
            )[0]
            for index in (0, 1)
        ]
        admitted = [0, 0]

        def drive(index: int) -> None:
            enforcer = enforcers[index]
            drawn = 0
            for _ in range(25):
                call_id = str(uuid.uuid4())
                result = _admit(
                    enforcer,
                    run_id,
                    estimated_input_tokens=input_tokens,
                    estimated_output_bound=output_bound,
                    call_id=call_id,
                )
                if not (result.allowed and result.lease_id is not None):
                    continue
                drawn += per_call
                # Settle exactly what was reserved: the lease-tagged confirm is
                # the only writer of settled tokens, so skipping it would let a
                # renewal re-grant float the server still believes is unspent.
                _send_confirm(
                    credentials,
                    enforcer.build_confirm_request(
                        model=MODEL,
                        token_details=TokenDetails(
                            input_tokens=input_tokens, output_tokens=output_bound
                        ),
                        provider="openai",
                        call_id=call_id,
                        lease_id=result.lease_id,
                    ),
                )
            admitted[index] = drawn

        threads = [threading.Thread(target=drive, args=(index,)) for index in (0, 1)]
        stop_sampling = threading.Event()
        joint_peak = 0

        def sample_joint_authority() -> None:
            nonlocal joint_peak
            while not stop_sampling.is_set():
                held = 0
                for enforcer in enforcers:
                    state = enforcer._lease.state_for(run_id)
                    if state is not None and state.lease_id is not None:
                        held += state.granted_tokens
                joint_peak = max(joint_peak, held)
                time.sleep(0.01)

        sampler = threading.Thread(target=sample_joint_authority, daemon=True)
        sampler.start()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        stop_sampling.set()
        sampler.join(timeout=10)
        for thread in threads:
            assert not thread.is_alive(), "a holder thread did not finish"

        assert sum(admitted) > 0, "neither holder ever admitted on lease authority"
        assert joint_peak > 0, "the sampler never observed a live lease"
        assert joint_peak <= solo_bound, (
            "two holders jointly held more token authority "
            f"({joint_peak}) than a single holder's bound ({solo_bound})"
        )

    @pytest.mark.integration
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "OPEN PJ-2 FINDING (S4/S5 report, contract mismatch #1): a lease-tagged "
            "confirm whose spend lands INSIDE the claim never reaches the enforcement "
            "counters, and the renewal recycles the same claim "
            "(new_outstanding = max(rem_g, claim_g+1)) instead of converting the "
            "settled part into counted spend. A run therefore keeps drawing fresh "
            "grants past a hard cap with the control plane fully reachable — measured "
            "live at $15.00 of settled spend on a $10.00 hard_deny project, with the "
            "project's reported usage pinned at $9.99999 throughout. The only repair "
            "is the 300s reconcile pass, and only once the calls' metadata events are "
            "durable in cost_events (the reconcile floor is durable spend + lease "
            "outstanding). Delete this xfail (do NOT loosen the assertion) when core's "
            "renewal algebra counts settled float."
        ),
    )
    def test_lifetime_drawdown_stays_within_the_granted_bound(
        self, api_url: str, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 4 (budget-safety half): one run cannot outdraw the pool it was sized from.

        Every admitted call is settled with a lease-tagged confirm carrying
        exactly the tokens it reserved — the production settlement path. The
        tokens a run can draw over its whole life must therefore stay inside the
        authority the server sized from the project's headroom.
        """
        credentials = provision_project(
            api_url, name="sdk-lease-drawdown", budget_limit=10.0, budget_mode="hard_deny"
        )
        holder = f"solo-bound-{uuid.uuid4().hex[:8]}"
        solo = _grant_raw(credentials, _run_id(), holder)
        solo_bound = solo["granted_tokens"]
        _surrender_raw(credentials, solo["lease_id"], holder, solo["generation"])

        enforcer, _wire = make_lease_enforcer(
            credentials, budget_mode=BudgetMode.HARD_DENY, fail_open=False
        )
        run_id = _run_id()
        input_tokens, output_bound = 2_000, 8_000
        drawn = 0
        for _ in range(60):
            call_id = str(uuid.uuid4())
            result = _admit(
                enforcer,
                run_id,
                estimated_input_tokens=input_tokens,
                estimated_output_bound=output_bound,
                call_id=call_id,
            )
            if not (result.allowed and result.lease_id is not None):
                continue
            drawn += input_tokens + output_bound
            _send_confirm(
                credentials,
                enforcer.build_confirm_request(
                    model=MODEL,
                    token_details=TokenDetails(
                        input_tokens=input_tokens, output_tokens=output_bound
                    ),
                    provider="openai",
                    call_id=call_id,
                    lease_id=result.lease_id,
                ),
            )

        assert drawn > 0
        assert drawn <= solo_bound, (
            f"the run drew {drawn} lease-funded tokens against a pool sized at {solo_bound}"
        )


@pytest.mark.integration
class TestLeaseReclaim:
    @pytest.mark.integration
    def test_idle_share_reclaim(
        self, lease_project: ProjectCredentials, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 5: a holder that reports no burn watches its share shrink."""
        enforcer, _wire = make_lease_enforcer(lease_project)
        run_id = _run_id()
        call_id = str(uuid.uuid4())

        first = _admit(enforcer, run_id, call_id=call_id)
        assert first.lease_id is not None
        # Genuinely idle: hand the triggering call's reservation back so the
        # renewals report neither spend nor in-flight demand.
        enforcer.release_reservation(call_id)

        state = enforcer._lease.state_for(run_id)
        assert state is not None
        shares = [state.share_remaining_tokens]
        for _ in range(5):
            request = enforcer._build_renewal(
                run_id,
                model=MODEL,
                provider="openai",
                fallback_providers=[],
                fallback_models=[],
            )
            assert request is not None
            assert request.spent_tokens == 0
            assert request.reserved_tokens == 0
            enforcer._renew_lease(run_id, request, [MODEL])
            state = enforcer._lease.state_for(run_id)
            assert state is not None
            shares.append(state.share_remaining_tokens)

        assert shares[-1] < shares[0], (
            f"an idle holder's headroom share never shrank across renewals: {shares}"
        )
        assert all(later <= earlier for earlier, later in zip(shares, shares[1:], strict=False)), (
            f"an idle holder's share grew back between renewals: {shares}"
        )

    @pytest.mark.integration
    def test_surrender_releases(
        self, lease_project: ProjectCredentials, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 6a: close() hands the float back — the server's ceiling returns to 0."""
        enforcer, wire = make_lease_enforcer(lease_project)
        run_id = _run_id()

        result = _admit(enforcer, run_id)
        assert result.lease_id is not None

        held = budget_status(lease_project)
        assert held["outage_overspend_ceiling_usd"] > 0.0, (
            "a live lease must show its share as outage authority on the budget status"
        )
        assert held["current_usage"] > 0.0

        enforcer.close()

        assert wire.count(_LEASE_SURRENDER_PATH) == 1, f"close() did not surrender: {wire.paths}"
        released = budget_status(lease_project)
        assert released["outage_overspend_ceiling_usd"] == 0.0, (
            f"the surrendered lease still holds outage authority: {released}"
        )
        assert released["current_usage"] < held["current_usage"], (
            "the surrendered claim was never returned to the project's counters"
        )

    @pytest.mark.integration
    def test_expiry_reclaims_vanished_holder(self, api_url: str) -> None:
        """DoD 6b: a holder that vanishes without surrendering is swept at its deadline.

        SLOW BY CONSTRUCTION (~2 minutes). The wire's shortest path to expiry is
        the high-variance shrink: four renewals with a wildly uneven burn report
        push the server's CV past 1.0, which halves lease_length to 60s. The
        server deadline is issued_at + lease_length + refresh_interval, so the
        float comes back ~75s after the last renewal — and only once a sweep
        runs, which every lease op for the project triggers (the probe grant
        below is that op).
        """
        credentials = provision_project(
            api_url, name="sdk-lease-expiry", budget_limit=20.0, budget_mode="alert_only"
        )
        holder = "vanishing-holder"
        lease = _grant_raw(credentials, _run_id(), holder)
        generation = lease["generation"]

        # Uneven burn reports: three near-silent intervals then one huge one.
        for spent in (1, 1, 1, 1_000_000):
            response = _renew_raw(
                credentials, lease["lease_id"], holder, generation, spent_tokens=spent
            )
            response.raise_for_status()
            body = response.json()
            generation = body["generation"]
        if body["lease_length_s"] >= 120.0:
            pytest.fail(
                "high-variance reporting did not shrink lease_length; the wait below "
                f"would exceed the test's bound: {body}"
            )
        deadline_s = body["lease_length_s"] + body["refresh_interval_s"]

        # The holder now vanishes: no surrender, no further renewal.
        held = budget_status(credentials)
        assert held["outage_overspend_ceiling_usd"] > 0.0

        wait_until = time.monotonic() + deadline_s + 60.0
        reclaimed = False
        while time.monotonic() < wait_until:
            time.sleep(10)
            # Any lease op for the project runs the lazy sweep; the probe is
            # surrendered immediately so it contributes nothing to the ceiling.
            probe_holder = f"sweep-probe-{uuid.uuid4().hex[:8]}"
            probe = _grant_raw(credentials, _run_id(), probe_holder)
            if probe.get("lease_id"):
                _surrender_raw(credentials, probe["lease_id"], probe_holder, probe["generation"])
            if budget_status(credentials)["outage_overspend_ceiling_usd"] == 0.0:
                reclaimed = True
                break

        assert reclaimed, (
            "the vanished holder's float was never reclaimed within "
            f"{deadline_s + 60.0:.0f}s: {budget_status(credentials)}"
        )

    @pytest.mark.integration
    def test_lost_renewal_response_replays_identically(
        self, lease_project: ProjectCredentials
    ) -> None:
        """DoD 7: a renewal whose response was lost replays, advancing state once."""
        holder = f"lost-response-{uuid.uuid4().hex[:8]}"
        lease = _grant_raw(lease_project, _run_id(), holder)
        lease_id, generation = lease["lease_id"], lease["generation"]

        first = _renew_raw(lease_project, lease_id, holder, generation)
        first.raise_for_status()
        # The holder never saw that response: it sends the SAME renewal again.
        second = _renew_raw(lease_project, lease_id, holder, generation)
        second.raise_for_status()

        assert second.json() == first.json(), (
            "a replayed renewal must return the stored response byte-for-byte: "
            f"{first.json()} vs {second.json()}"
        )
        assert first.json()["generation"] == generation + 1

        # State advanced EXACTLY once: echoing the successor is fenceable, so the
        # server was at g+1 (not g+2) after the replay.
        third = _renew_raw(lease_project, lease_id, holder, first.json()["generation"])
        third.raise_for_status()
        assert third.json()["generation"] == generation + 2

        released = _surrender_raw(lease_project, lease_id, holder, third.json()["generation"])
        assert released["released_tokens"] >= 0
