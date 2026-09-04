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
from contextlib import suppress
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
    estimated_input_tokens: int = 100,
) -> dict[str, Any]:
    """A grant issued with the SDK's exact request bytes, outside any enforcer."""
    request = LeaseGrantRequest(
        agent_run_id=run_id,
        holder_id=holder_id,
        model=MODEL,
        provider=ProviderName.OPENAI,
        fail_open=fail_open,
        estimated_input_tokens=estimated_input_tokens,
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
    def test_alert_only_past_cap_continues_with_reservation(
        self, api_url: str, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 3 (live half): alert_only past its cap keeps running, reservation-funded.

        Since core PR #387 (2026-09-01) an alert_only project is never refused a
        project-level hold: the check answers allowed with a reservation id and
        a signed, non-positive remaining budget, and carries NO warning (the
        SDK's "Budget limit reached" warning is built only from a server
        refusal, which scoped hard caps can still produce). The lease path still
        hands back a zero-token grant, so every call rides the per-call check.
        """
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
            assert result.reservation_id is not None, (
                "an alert_only call past the cap is approved WITH a hold, not refused"
            )
            assert result.lease_id is None, "past the cap the call is reservation-funded"
            assert result.remaining_budget is not None and result.remaining_budget <= 0
            assert result.warning is None, (
                "the server approves the overshoot, so the SDK has no refusal to warn about"
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
        holders jointly hold is at most the independently measured run pool.

        Lifetime drawdown is a DIFFERENT invariant and belongs to the test
        below, which now holds live: a run cannot outdraw the pool it was sized
        from. Getting there took a fix on each side — core converting settled
        float into counted spend, and the ledger netting the drawdown a
        renewal's sizing never saw — and the strict xfail that carried this
        test's assertion through both is what forced them.
        """
        credentials = provision_project(
            api_url, name="sdk-lease-pair", budget_limit=100.0, budget_mode="hard_deny"
        )

        # At gpt-5.5's $30/M worst-case rate, $100 is a ~3.3M-token
        # pool. A 1M-token demand spans 8 lease intervals (8M tokens), so
        # the solo probe saturates the ratio bound and exposes the whole pool.
        solo = _grant_raw(
            credentials,
            _run_id(),
            "solo-holder",
            estimated_input_tokens=1_000_000,
        )
        solo_pool = solo["headroom_share_tokens"]
        assert solo_pool > 0
        _surrender_raw(credentials, solo["lease_id"], "solo-holder", solo["generation"])

        run_id = _run_id()
        # Bootstrap grants are 400k tokens; 25 x 20k reservations cross the
        # 75% renewal threshold while remaining far inside the roomy pool.
        input_tokens, output_bound = 5_000, 15_000
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
        driver_errors: list[Exception | None] = [None, None]
        initial_generations = [0, 0]
        both_holders_ready = threading.Event()
        overlap_barrier = threading.Barrier(2, action=both_holders_ready.set)
        resume_drivers = threading.Event()
        both_holders_renewed = threading.Event()
        renewed_barrier = threading.Barrier(2, action=both_holders_renewed.set)
        resume_after_renewal = threading.Event()

        def drive(index: int) -> None:
            enforcer = enforcers[index]
            drawn = 0

            def admit_and_settle(call_id: str) -> None:
                nonlocal drawn
                result = _admit(
                    enforcer,
                    run_id,
                    estimated_input_tokens=input_tokens,
                    estimated_output_bound=output_bound,
                    call_id=call_id,
                )
                if not (result.allowed and result.lease_id is not None):
                    return
                drawn += per_call
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
                        lease_claim_token=result.lease_claim_token,
                    ),
                )

            first_call_id = str(uuid.uuid4())
            try:
                first = _admit(
                    enforcer,
                    run_id,
                    estimated_input_tokens=input_tokens,
                    estimated_output_bound=output_bound,
                    call_id=first_call_id,
                )
                if not (first.allowed and first.lease_id is not None):
                    raise AssertionError(
                        f"holder {index} did not receive lease authority on its first admission"
                    )
                first_state = enforcer._lease.state_for(run_id)
                if first_state is None or first_state.lease_id is None:
                    raise AssertionError(f"holder {index} installed no first lease")
                initial_generations[index] = first_state.generation
                # Both first admissions (and therefore both grant responses)
                # must be installed before either holder settles or advances.
                # This creates a guaranteed overlap window for the assertion
                # below instead of hoping a periodic sampler catches one.
                overlap_barrier.wait(timeout=180)
                resume_drivers.wait(timeout=180)
                drawn += per_call
                _send_confirm(
                    credentials,
                    enforcer.build_confirm_request(
                        model=MODEL,
                        token_details=TokenDetails(
                            input_tokens=input_tokens,
                            output_tokens=output_bound,
                        ),
                        provider="openai",
                        call_id=first_call_id,
                        lease_id=first.lease_id,
                        lease_claim_token=first.lease_claim_token,
                    ),
                )

                renewed = False
                for _ in range(24):
                    call_id = str(uuid.uuid4())
                    admit_and_settle(call_id)
                    _join_renewals(enforcer)
                    state = enforcer._lease.state_for(run_id)
                    if (
                        not renewed
                        and state is not None
                        and state.lease_id is not None
                        and state.generation > initial_generations[index]
                    ):
                        renewed = True
                        renewed_barrier.wait(timeout=180)
                        resume_after_renewal.wait(timeout=180)
                if not renewed:
                    raise AssertionError(f"holder {index} never completed a renewal")
            except Exception as exc:
                driver_errors[index] = exc
                both_holders_ready.set()
                both_holders_renewed.set()
                with suppress(Exception):
                    overlap_barrier.abort()
                with suppress(Exception):
                    renewed_barrier.abort()
                resume_drivers.set()
                resume_after_renewal.set()
                return

            admitted[index] = drawn

        threads = [threading.Thread(target=drive, args=(index,)) for index in (0, 1)]
        for thread in threads:
            thread.start()
        initial_joint_authority = 0
        renewed_joint_authority = 0
        renewed_overlap_complete = False
        try:
            assert both_holders_ready.wait(timeout=180), (
                f"holders did not reach the forced overlap: {driver_errors}"
            )
            assert driver_errors == [None, None]
            initial_states = [enforcer._lease.state_for(run_id) for enforcer in enforcers]
            assert all(
                state is not None
                and state.lease_id is not None
                and state.share_remaining_tokens > 0
                for state in initial_states
            )
            initial_joint_authority = sum(
                state.share_remaining_tokens for state in initial_states if state is not None
            )
            resume_drivers.set()

            assert both_holders_renewed.wait(timeout=180), (
                f"holders did not reach the renewed overlap: {driver_errors}"
            )
            assert driver_errors == [None, None]
            renewed_states = [enforcer._lease.state_for(run_id) for enforcer in enforcers]
            assert all(
                state is not None
                and state.lease_id is not None
                and state.generation > initial_generations[index]
                and state.share_remaining_tokens > 0
                for index, state in enumerate(renewed_states)
            )
            renewed_joint_authority = sum(
                state.share_remaining_tokens for state in renewed_states if state is not None
            )
            renewed_overlap_complete = True
        finally:
            resume_drivers.set()
            resume_after_renewal.set()
            with suppress(Exception):
                overlap_barrier.abort()
            if not renewed_overlap_complete:
                with suppress(Exception):
                    renewed_barrier.abort()

        for thread in threads:
            thread.join(timeout=180)
        for thread in threads:
            assert not thread.is_alive(), "a holder thread did not finish"
        assert driver_errors == [None, None]

        assert sum(admitted) > 0, "neither holder ever admitted on lease authority"
        assert initial_joint_authority <= solo_pool, (
            "two holders jointly held more token authority "
            f"({initial_joint_authority}) than the run pool ({solo_pool})"
        )
        assert renewed_joint_authority <= solo_pool, (
            "two renewed holders jointly held more token authority "
            f"({renewed_joint_authority}) than the run pool ({solo_pool})"
        )

    @pytest.mark.integration
    def test_zero_token_second_holder_falls_back_to_authoritative_verdict(
        self, api_url: str, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 4: a valid zero-token lease defers to the per-call verdict."""
        credentials = provision_project(
            api_url, name="sdk-lease-tight-pair", budget_limit=10.0, budget_mode="hard_deny"
        )
        run_id = _run_id()
        first_holder = f"tight-first-{uuid.uuid4().hex[:8]}"
        first = _grant_raw(credentials, run_id, first_holder, fail_open=False)
        assert first["granted_tokens"] > 0

        enforcer, wire = make_lease_enforcer(
            credentials,
            budget_mode=BudgetMode.HARD_DENY,
            fail_open=False,
            holder_id=f"tight-second-{uuid.uuid4().hex[:8]}",
        )
        try:
            result = _admit(enforcer, run_id)
            state = enforcer._lease.state_for(run_id)

            assert state is not None
            assert state.lease_id is not None
            assert state.generation > 0
            assert state.granted_tokens == 0
            assert state.share_remaining_tokens == 0
            assert result.allowed is False
            assert result.lease_id is None
            assert wire.count(_LEASE_PATH) == 1
            assert wire.count(CHECK_PATH) == 1
        finally:
            enforcer.close()
            _surrender_raw(
                credentials,
                first["lease_id"],
                first_holder,
                first["generation"],
            )

    @pytest.mark.integration
    def test_lifetime_drawdown_stays_within_the_granted_bound(
        self, api_url: str, make_lease_enforcer: EnforcerFactory
    ) -> None:
        """DoD 4 (budget-safety half): one run cannot outdraw the pool it was sized from.

        Every admitted call is settled with a lease-tagged confirm carrying
        exactly the tokens it reserved — the production settlement path. The
        tokens a run can draw over its whole LIFE (not just per window) must
        therefore stay inside the authority the server sized from the project's
        headroom, however many renewals it takes.

        This is the regression pin for the settled-float leak (core 8bbeb805):
        before that fix, settled-within-claim spend never became counted spend
        and the renewal recycled the same claim forever, so this same drive took
        $15.00 out of a $10.00 hard_deny cap with the control plane fully up.

        Why the calls are ALL OUTPUT: the pool is denominated in tokens at the
        declared set's WORST-CASE rate. A cheaper mix (some input tokens) buys
        legitimately more tokens than that bound, so the token comparison below
        would be wrong for a reason that has nothing to do with the invariant.
        Spending only output tokens makes the actual rate the worst-case rate,
        which is what lets this test state the bound in tokens — the SDK never
        computes a price.
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
        output_bound = 10_000
        attempts = 60
        drawn = 0
        admitted_calls = 0
        for _ in range(attempts):
            call_id = str(uuid.uuid4())
            result = _admit(
                enforcer,
                run_id,
                estimated_input_tokens=0,
                estimated_output_bound=output_bound,
                call_id=call_id,
            )
            if not (result.allowed and result.lease_id is not None):
                continue
            drawn += output_bound
            admitted_calls += 1
            _send_confirm(
                credentials,
                enforcer.build_confirm_request(
                    model=MODEL,
                    token_details=TokenDetails(input_tokens=0, output_tokens=output_bound),
                    provider="openai",
                    call_id=call_id,
                    lease_id=result.lease_id,
                    lease_claim_token=result.lease_claim_token,
                ),
            )

        assert drawn > 0
        assert admitted_calls < attempts, (
            "the run never ran out of lease authority — a pool that funds every "
            "attempt cannot show that the drawdown is bounded at all"
        )
        assert drawn <= solo_bound, (
            f"the run drew {drawn} lease-funded tokens against a pool sized at {solo_bound}"
        )
        # The cap held server-side too: the counter now carries the settled
        # spend (that is the fix), so this is a real statement rather than a
        # float placeholder sitting where the spend should be.
        status = budget_status(credentials)
        assert status["current_usage"] <= status["budget_limit"]


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
        enforcer.release_reservation(
            call_id,
            lease_claim_token=first.lease_claim_token,
        )

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
        enforcer, _wire = make_lease_enforcer(lease_project)
        run_id = _run_id()

        result = _admit(enforcer, run_id)
        assert result.lease_id is not None

        held = budget_status(lease_project)
        assert held["outage_overspend_ceiling_usd"] > 0.0, (
            "a live lease must show its share as outage authority on the budget status"
        )
        assert held["current_usage"] > 0.0

        enforcer.close()

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
