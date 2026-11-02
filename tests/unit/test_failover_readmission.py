"""Under hard_deny a failover hop is re-admitted at its own price before it is sent.

The per-call check prices the admitted (provider, model). Settlement prices
the hop that serves, so a failover to a pricier hop could settle past its
reservation. Each differing hop therefore gets its own per-call check with its
own output bound; a denial refuses the call before the hop's provider is
called. Real openai clients serialize onto in-memory transports, and a stub
control plane prices each model at its own synthetic rate.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from conftest import VALID_PROJECT_ID

from solwyn._lease import DEFAULT_OUTPUT_BOUND
from solwyn.exceptions import BudgetExceededError, OutputBoundUnenforceableError
from solwyn.testing import FakeControlPlane

openai = pytest.importorskip("openai")

MESSAGES = [{"role": "user", "content": "hi"}]
CHEAP = "cheap-model"
PRICEY = "pricey-model"
GROQ_URL = "https://api.groq.com/openai/v1"

# Synthetic USD-per-token cards (not live rates): the pricey hop's output is
# about 16x the cheap one's, like gpt-4o against gpt-4o-mini.
CHEAP_2 = "cheap-model-2"
_RATES = {
    CHEAP: (0.15 / 1_000_000, 0.6 / 1_000_000),
    CHEAP_2: (0.15 / 1_000_000, 0.6 / 1_000_000),
    PRICEY: (2.5 / 1_000_000, 10.0 / 1_000_000),
}

_CHAT_BODY = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": PRICEY,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _PricedPlane(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Prices each check at its own model's rate against one budget.

    A check reserves input plus ``estimated_output_bound`` at that model's
    rates; a confirm frees its reservation and charges the actual tokens. In
    hard_deny a reservation over the remainder is refused. Everything else
    (receipts, events) goes to the zero-network double.
    """

    def __init__(self, *, budget_limit: float, mode: str = "hard_deny") -> None:
        self.budget_limit = budget_limit
        self.mode = mode
        self.checks: list[dict[str, Any]] = []
        self.confirms: list[dict[str, Any]] = []
        self.held: dict[str, float] = {}
        self.spent = 0.0
        self.plane = FakeControlPlane()

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = _RATES[model]
        return input_tokens * input_rate + output_tokens * output_rate

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/budgets/check"):
            return self._check(json.loads(request.content))
        if request.method == "POST" and path.endswith("/budgets/confirm"):
            body = json.loads(request.content)
            self.confirms.append(body)
            self.held.pop(body["reservation_id"], None)
            details = body["token_details"]
            self.spent += self._cost(
                body["model"], details["input_tokens"], details["output_tokens"]
            )
            return httpx.Response(204)
        return self.plane.transport.handle_request(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self.handle_request(request)

    def _check(self, body: dict[str, Any]) -> httpx.Response:
        self.checks.append(body)
        reservation = self._cost(
            body["model"], body["estimated_input_tokens"], body["estimated_output_bound"]
        )
        remaining = self.budget_limit - self.spent - sum(self.held.values())
        allowed = self.mode != "hard_deny" or reservation <= remaining
        reservation_id = f"res_{len(self.checks)}"
        if allowed:
            self.held[reservation_id] = reservation
        return httpx.Response(
            200,
            json={
                "allowed": allowed,
                "remaining_budget": remaining,
                "reservation_id": reservation_id if allowed else None,
                "mode": self.mode,
                "budget_limit": self.budget_limit,
                "current_usage": self.budget_limit - remaining,
                "project_id": VALID_PROJECT_ID,
                **({} if allowed else {"denied_by_period": "monthly"}),
            },
        )


class _Provider:
    """One provider endpoint: answers with ``status`` and records each body."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(
                self.status, json={"error": {"message": "x", "type": "x", "code": None}}
            )
        return httpx.Response(200, json=_CHAT_BODY)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)

    def client(self, base_url: str | None = None, *, is_async: bool = False) -> Any:
        if is_async:
            return openai.AsyncOpenAI(
                api_key="sk-test",
                base_url=base_url,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.async_handler)),
                max_retries=0,
            )
        return openai.OpenAI(
            api_key="sk-test",
            base_url=base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            max_retries=0,
        )


def _wrap(
    stub: _PricedPlane, primary: Any, fallback: Any, *, is_async: bool = False, **options: Any
) -> Any:
    from solwyn.client import AsyncSolwyn, Solwyn

    wrapper = AsyncSolwyn if is_async else Solwyn
    return wrapper(
        primary,
        api_key=stub.plane.api_key,
        api_url=stub.plane.api_url,
        control_plane_transport=stub,
        budget_check_cache_ttl=0,
        lease_enabled=False,
        fallback=[(fallback, PRICEY)],
        **options,
    )


def _setup(
    budget_limit: float, *, mode: str = "hard_deny", primary_status: int = 429
) -> tuple[_PricedPlane, _Provider, _Provider]:
    return (
        _PricedPlane(budget_limit=budget_limit, mode=mode),
        _Provider(primary_status),
        _Provider(),
    )


# 4096 output tokens: about $0.0025 on the cheap hop, about $0.041 on the pricey one.
_TIGHT = 0.01
_ROOMY = 0.10


@pytest.mark.unit
class TestFailoverReadmission:
    def test_pricier_hop_that_does_not_fit_is_refused_before_it_is_called(self) -> None:
        stub, primary, fallback = _setup(_TIGHT)
        with (
            _wrap(stub, primary.client(), fallback.client(GROQ_URL)) as solwyn,
            pytest.raises(BudgetExceededError),
        ):
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert len(primary.bodies) == 1
        assert fallback.bodies == []
        assert [(check["model"], check["estimated_output_bound"]) for check in stub.checks] == [
            (CHEAP, DEFAULT_OUTPUT_BOUND),
            (PRICEY, DEFAULT_OUTPUT_BOUND),
        ]
        assert stub.checks[1]["provider"] == "groq"
        assert "fallback_models" not in stub.checks[1] or stub.checks[1]["fallback_models"] == []

    def test_hop_that_fits_is_reserved_at_its_own_price_and_capped(self) -> None:
        stub, primary, fallback = _setup(_ROOMY)
        with _wrap(stub, primary.client(), fallback.client(GROQ_URL)) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP, PRICEY]
        assert len(fallback.bodies) == 1
        assert fallback.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND
        # The served hop settles its own reservation; the cheap one was closed
        # out at zero tokens under a fresh call id.
        settled = {confirm["reservation_id"]: confirm for confirm in stub.confirms}
        assert set(settled) == {"res_1", "res_2"}
        close_out, served = settled["res_1"], settled["res_2"]
        assert close_out["model"] == CHEAP
        assert close_out["token_details"]["output_tokens"] == 0
        assert served["model"] == PRICEY
        assert served["provider"] == "groq"
        assert close_out["call_id"] != served["call_id"]
        assert stub.held == {}

    @pytest.mark.asyncio
    async def test_async_pricier_hop_that_does_not_fit_is_refused(self) -> None:
        stub, primary, fallback = _setup(_TIGHT)
        solwyn = _wrap(
            stub,
            primary.client(is_async=True),
            fallback.client(GROQ_URL, is_async=True),
            is_async=True,
        )
        try:
            with pytest.raises(BudgetExceededError):
                await solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
        finally:
            await solwyn.close()

        assert len(primary.bodies) == 1
        assert fallback.bodies == []
        assert [check["model"] for check in stub.checks] == [CHEAP, PRICEY]

    @pytest.mark.asyncio
    async def test_async_hop_that_fits_is_reserved_at_its_own_price(self) -> None:
        stub, primary, fallback = _setup(_ROOMY)
        solwyn = _wrap(
            stub,
            primary.client(is_async=True),
            fallback.client(GROQ_URL, is_async=True),
            is_async=True,
        )
        try:
            await solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
        finally:
            await solwyn.close()

        assert [check["model"] for check in stub.checks] == [CHEAP, PRICEY]
        assert fallback.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND
        assert {confirm["reservation_id"] for confirm in stub.confirms} == {"res_1", "res_2"}

    def test_hop_denial_receipt_is_attributed_to_the_hop(self) -> None:
        stub, primary, fallback = _setup(_TIGHT)
        solwyn = _wrap(stub, primary.client(), fallback.client(GROQ_URL))
        reported: list[Any] = []
        original = solwyn._solwyn_reporter.report
        with (
            patch.object(
                solwyn._solwyn_reporter,
                "report",
                side_effect=lambda event: (reported.append(event), original(event)),
            ),
            solwyn,
            pytest.raises(BudgetExceededError),
        ):
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        denials = [event for event in reported if event.status.value == "budget_denied"]
        assert len(denials) == 1
        assert denials[0].model == PRICEY
        assert denials[0].attempt_index == 1
        assert denials[0].is_provider_fallback is True

    def test_ambiguous_previous_attempt_keeps_its_hold(self) -> None:
        # A 5xx may have been billed: with failover_idempotency="always" the
        # call still fails over, but the cheap reservation is not closed out.
        stub, primary, fallback = _setup(_ROOMY, primary_status=500)
        with _wrap(
            stub, primary.client(), fallback.client(GROQ_URL), failover_idempotency="always"
        ) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP, PRICEY]
        assert [confirm["reservation_id"] for confirm in stub.confirms] == ["res_2"]
        assert set(stub.held) == {"res_1"}

    def test_alert_only_fails_over_without_a_second_check(self) -> None:
        stub, primary, fallback = _setup(_TIGHT, mode="alert_only")
        with _wrap(stub, primary.client(), fallback.client(GROQ_URL)) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP]
        assert len(fallback.bodies) == 1
        assert "max_tokens" not in fallback.bodies[0]

    def test_lease_funded_call_fails_over_inside_its_lease(self) -> None:
        # The lease is valued at the worst-case rate of its declared chain, so
        # a failover inside the chain cannot spend past it: no per-call check.
        import solwyn

        plane = FakeControlPlane()
        primary, fallback = _Provider(429), _Provider()
        wrapped = plane.wrap(primary.client(), fallback=[(fallback.client(GROQ_URL), PRICEY)])
        with wrapped, solwyn.run("lease-failover"):
            wrapped.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert plane.checks == []
        assert len(plane.lease_grants) == 1
        assert len(fallback.bodies) == 1


@pytest.mark.unit
def test_cost_routed_first_hop_is_readmitted_at_its_own_price() -> None:
    # CostPolicy sends the call to the fallback first. The check priced the
    # primary, so the routed hop is re-admitted before it is sent.
    from solwyn._routing import CostPolicy

    plane = FakeControlPlane(price_hints={"openai": 3.0, "groq": 1.0})
    primary, fallback = _Provider(), _Provider()
    with plane.wrap(
        primary.client(),
        lease_enabled=False,
        fallback=[(fallback.client(GROQ_URL), PRICEY)],
        selection_policy=CostPolicy(),
    ) as solwyn:
        solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

    assert [(check.provider.value, check.model) for check in plane.checks] == [
        ("openai", CHEAP),
        ("groq", PRICEY),
    ]
    assert primary.bodies == []
    assert len(fallback.bodies) == 1


@pytest.mark.unit
class TestDecisionsUnderHardDeny:
    """Decisions takes no output-cap parameter, so hard_deny cannot hold it."""

    _QUESTIONS = [{"type": "predicate", "instructions": "Is it a greeting?"}]

    def _client(self, provider: _Provider, *, is_async: bool = False) -> Any:
        return provider.client(is_async=is_async)

    def test_decisions_is_refused_before_the_provider(self) -> None:
        provider, plane = _Provider(), FakeControlPlane()
        with (
            plane.wrap(self._client(provider), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.decisions.create(model="gpt-5.5", input="hi", questions=self._QUESTIONS)

        assert caught.value.reason == "no_output_cap"
        assert caught.value.provider == "openai"
        assert provider.bodies == []

    @pytest.mark.asyncio
    async def test_async_decisions_is_refused_before_the_provider(self) -> None:
        provider, plane = _Provider(), FakeControlPlane()
        async with plane.wrap_async(
            self._client(provider, is_async=True), lease_enabled=False
        ) as solwyn:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                await solwyn.decisions.create(
                    model="gpt-5.5", input="hi", questions=self._QUESTIONS
                )

        assert caught.value.reason == "no_output_cap"
        assert provider.bodies == []

    def test_lease_funded_decisions_is_refused_too(self) -> None:
        import solwyn

        provider, plane = _Provider(), FakeControlPlane()
        wrapped = plane.wrap(self._client(provider))
        with wrapped, solwyn.run("decisions"), pytest.raises(OutputBoundUnenforceableError):
            wrapped.decisions.create(model="gpt-5.5", input="hi", questions=self._QUESTIONS)

        assert provider.bodies == []


@pytest.mark.unit
class TestReadmissionAfterLeaseRefusal:
    """Finding 1: a run-scoped call whose lease grant is refused takes a per-call
    check, which claims its call id in the lease ledger. The hop's re-check
    must not trip over that claim, and must leave nothing reserved under it."""

    @pytest.mark.parametrize(
        ("status", "code"), [(409, "lease_holder_cap_exceeded"), (503, "lease_unavailable")]
    )
    def test_failover_after_a_refused_grant_is_readmitted(self, status: int, code: str) -> None:
        import solwyn

        plane = FakeControlPlane()
        primary, fallback = _Provider(429), _Provider()
        wrapped = plane.wrap(primary.client(), fallback=[(fallback.client(GROQ_URL), PRICEY)])
        with wrapped, plane.refuse_leases(status=status, code=code), solwyn.run("refused"):
            wrapped.chat.completions.create(model=CHEAP, messages=MESSAGES)
            ledger = wrapped._solwyn_budget._lease
            claimed = dict(ledger._call_claims)
            reserved = dict(ledger._call_index)

        assert [check.model for check in plane.checks] == [CHEAP, PRICEY]
        assert len(fallback.bodies) == 1
        # The primary's claim is the ledger's replay tombstone; it owns no
        # reservation, so nothing is left reserved after the call.
        assert len(claimed) == 1
        assert reserved == {}


@pytest.mark.unit
class TestReadmissionAfterAnAmbiguousAttempt:
    """Finding 2: after a possibly billed attempt, the next attempt is
    re-admitted even when its provider and model match, so two possibly
    billed generations never share one reservation."""

    def _wrap_same_identity(self, stub: _PricedPlane, primary: Any, again: Any) -> Any:
        from solwyn.client import Solwyn

        return Solwyn(
            primary,
            api_key=stub.plane.api_key,
            api_url=stub.plane.api_url,
            control_plane_transport=stub,
            budget_check_cache_ttl=0,
            lease_enabled=False,
            fallback=[(again, CHEAP)],
            failover_idempotency="always",
        )

    def test_same_identity_hop_after_an_ambiguous_attempt_is_readmitted(self) -> None:
        stub, primary, again = _PricedPlane(budget_limit=_ROOMY), _Provider(500), _Provider()
        with self._wrap_same_identity(stub, primary.client(), again.client()) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP, CHEAP]
        assert len(again.bodies) == 1
        # The possibly billed hold stays; the hop settles its own reservation.
        assert [confirm["reservation_id"] for confirm in stub.confirms] == ["res_2"]
        assert set(stub.held) == {"res_1"}

    def test_same_identity_hop_that_does_not_fit_beside_the_hold_is_refused(self) -> None:
        # One cheap reservation (about $0.0025) fits a $0.004 cap; two do not.
        stub, primary, again = _PricedPlane(budget_limit=0.004), _Provider(500), _Provider()
        with (
            self._wrap_same_identity(stub, primary.client(), again.client()) as solwyn,
            pytest.raises(BudgetExceededError),
        ):
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert len(primary.bodies) == 1
        assert again.bodies == []


@pytest.mark.unit
class TestHardDenyAllowsAreNeverCached:
    """Finding 4: a cached allow carries no reservation, so under hard_deny it
    would admit a second generation that nothing funds."""

    def _wrap_cached(self, stub: _PricedPlane, primary: Any, fallback: Any | None = None) -> Any:
        from solwyn.client import Solwyn

        options: dict[str, Any] = {}
        if fallback is not None:
            options["fallback"] = [(fallback, PRICEY)]
        return Solwyn(
            primary,
            api_key=stub.plane.api_key,
            api_url=stub.plane.api_url,
            control_plane_transport=stub,
            budget_check_cache_ttl=60,
            lease_enabled=False,
            **options,
        )

    def test_shared_fallback_is_checked_for_each_call(self) -> None:
        stub, primary, fallback = _setup(_ROOMY)
        with self._wrap_cached(stub, primary.client(), fallback.client(GROQ_URL)) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
            solwyn.chat.completions.create(model=CHEAP_2, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP, PRICEY, CHEAP_2, PRICEY]
        assert len(fallback.bodies) == 2
        served = [c for c in stub.confirms if c["model"] == PRICEY]
        assert len({confirm["reservation_id"] for confirm in served}) == 2

    def test_repeated_call_is_checked_each_time(self) -> None:
        stub, provider = _PricedPlane(budget_limit=_ROOMY), _Provider()
        with self._wrap_cached(stub, provider.client()) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP, CHEAP]
        assert len({confirm["reservation_id"] for confirm in stub.confirms}) == 2

    def test_alert_only_allows_are_still_cached(self) -> None:
        stub, provider = _PricedPlane(budget_limit=_ROOMY, mode="alert_only"), _Provider()
        with self._wrap_cached(stub, provider.client()) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP]


@pytest.mark.unit
class TestLeaseFundedAmbiguousFailover:
    """Finding 5: two possibly billed generations must not share one lease
    claim. Under hard_deny the possibly billed claim is consumed at its floor,
    and the next attempt is re-admitted with its own per-call check."""

    def test_next_attempt_after_a_possibly_billed_lease_claim_is_readmitted(self) -> None:
        import solwyn

        plane = FakeControlPlane()
        primary, fallback = _Provider(500), _Provider()
        wrapped = plane.wrap(
            primary.client(),
            fallback=[(fallback.client(GROQ_URL), PRICEY)],
            failover_idempotency="always",
        )
        with wrapped, solwyn.run("ambiguous") as run_id:
            wrapped.chat.completions.create(model=CHEAP, messages=MESSAGES)
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            spent = state.spent_tokens_since_report
            reserved = state.reserved_tokens

        assert len(plane.lease_grants) == 1
        # The fallback paid its own per-call check at its own bound ...
        assert [(check.provider.value, check.model) for check in plane.checks] == [("groq", PRICEY)]
        assert plane.checks[0].estimated_output_bound == DEFAULT_OUTPUT_BOUND
        # ... and settles that reservation, not the lease.
        assert len(plane.confirms) == 1
        assert plane.confirms[0].reservation_id is not None
        assert plane.confirms[0].lease_id is None
        # The possibly billed primary's claim stays spent at its floor.
        assert reserved == 0
        assert spent >= DEFAULT_OUTPUT_BOUND
        assert len(fallback.bodies) == 1

    def test_alert_only_lease_failover_is_unchanged(self) -> None:
        import solwyn

        plane = FakeControlPlane(mode="alert_only")
        primary, fallback = _Provider(500), _Provider()
        wrapped = plane.wrap(
            primary.client(),
            fallback=[(fallback.client(GROQ_URL), PRICEY)],
            failover_idempotency="always",
        )
        with wrapped, solwyn.run("ambiguous-alert"):
            wrapped.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert plane.checks == []
        assert len(plane.confirms) == 1
        assert plane.confirms[0].lease_id is not None


TOGETHER_URL = "https://api.together.xyz/v1"
LLAMA = "llama-3.1-8b-instant"
GPT_OSS = "openai/gpt-oss-20b"


@pytest.mark.unit
class TestLeaseCoversProviderModelPairs:
    """Finding 6: a lease covers the (provider, model) pairs it declared, as
    core prices them, not bare model names."""

    def test_declared_model_on_an_undeclared_provider_takes_a_per_call_check(self) -> None:
        import solwyn

        plane = FakeControlPlane()
        primary, fallback = _Provider(), _Provider()
        wrapped = plane.wrap(
            primary.client(GROQ_URL), fallback=[(fallback.client(TOGETHER_URL), GPT_OSS)]
        )
        with wrapped, solwyn.run("pairs"):
            # Declares (groq, llama) and (together, gpt-oss).
            wrapped.chat.completions.create(model=LLAMA, messages=MESSAGES)
            # (groq, gpt-oss) was never declared, though the name gpt-oss was.
            wrapped.chat.completions.create(model=GPT_OSS, messages=MESSAGES)

        assert len(plane.lease_grants) == 1
        assert [(check.provider.value, check.model) for check in plane.checks] == [
            ("groq", GPT_OSS)
        ]
        assert len(primary.bodies) == 2

    def test_declared_pair_stays_lease_funded(self) -> None:
        import solwyn

        plane = FakeControlPlane()
        primary, fallback = _Provider(), _Provider()
        wrapped = plane.wrap(
            primary.client(GROQ_URL), fallback=[(fallback.client(TOGETHER_URL), GPT_OSS)]
        )
        with wrapped, solwyn.run("pairs-same"):
            wrapped.chat.completions.create(model=LLAMA, messages=MESSAGES)
            wrapped.chat.completions.create(model=LLAMA, messages=MESSAGES)

        assert len(plane.lease_grants) == 1
        assert plane.checks == []


@pytest.mark.unit
class TestHopChecksAreFreshWithTheDefaultCache:
    """Sol r1 B1: with the DEFAULT cache config, every hop re-check is a fresh,
    reservation-backed check, never an allow served from the cache."""

    def _wrap_default(self, stub: _PricedPlane, primary: Any, fallback: Any, **kw: Any) -> Any:
        from solwyn.client import AsyncSolwyn, Solwyn

        wrapper = AsyncSolwyn if kw.pop("is_async", False) else Solwyn
        return wrapper(
            primary,
            api_key=stub.plane.api_key,
            api_url=stub.plane.api_url,
            control_plane_transport=stub,
            lease_enabled=False,
            fallback=[(fallback, PRICEY)],
        )

    def test_two_calls_sharing_a_fallback_each_reserve_it(self) -> None:
        # Two $0.041 hops do not fit a $0.05 cap together: the second refuses.
        stub, primary, fallback = _setup(0.05)
        with self._wrap_default(stub, primary.client(), fallback.client(GROQ_URL)) as solwyn:
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
            with pytest.raises(BudgetExceededError):
                solwyn.chat.completions.create(model=CHEAP_2, messages=MESSAGES)

        assert [check["model"] for check in stub.checks] == [CHEAP, PRICEY, CHEAP_2, PRICEY]
        assert len(fallback.bodies) == 1

    @pytest.mark.asyncio
    async def test_async_two_calls_sharing_a_fallback_each_reserve_it(self) -> None:
        stub, primary, fallback = _setup(0.05)
        solwyn = self._wrap_default(
            stub,
            primary.client(is_async=True),
            fallback.client(GROQ_URL, is_async=True),
            is_async=True,
        )
        try:
            await solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
            with pytest.raises(BudgetExceededError):
                await solwyn.chat.completions.create(model=CHEAP_2, messages=MESSAGES)
        finally:
            await solwyn.close()

        assert [check["model"] for check in stub.checks] == [CHEAP, PRICEY, CHEAP_2, PRICEY]
        assert len(fallback.bodies) == 1


@pytest.mark.unit
class TestPerCallOnlyChecksBypassTheCache:
    """A hop re-check neither reads nor writes the allow cache, whatever mode a
    cached entry has (parity with TS #26)."""

    def _enforcer(self, plane: FakeControlPlane) -> Any:
        from solwyn.budget import BudgetEnforcer

        return BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport, cache_ttl=60)

    def _check(self, enforcer: Any, **kw: Any) -> Any:
        return enforcer.check_budget(estimated_input_tokens=1, model=PRICEY, provider="groq", **kw)

    def test_a_cached_allow_never_answers_a_per_call_only_check(self) -> None:
        from contextlib import closing

        plane = FakeControlPlane(mode="alert_only")
        with closing(self._enforcer(plane)) as enforcer:
            self._check(enforcer)
            fresh = self._check(enforcer, per_call_only=True)

        assert len(plane.checks) == 2
        assert fresh.reservation_id is not None

    def test_a_per_call_only_allow_is_never_cached(self) -> None:
        from contextlib import closing

        plane = FakeControlPlane(mode="alert_only")
        with closing(self._enforcer(plane)) as enforcer:
            self._check(enforcer, per_call_only=True)
            self._check(enforcer)

        assert len(plane.checks) == 2


QWEN_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


@pytest.mark.unit
class TestLocalRefusalLeavesNoHold:
    """Sol r1 B4: a hard_deny refusal made before any provider request must
    free the per-call hold its check took, or the next valid call is denied."""

    def _wrap_single(self, stub: _PricedPlane, client: Any, *, is_async: bool = False) -> Any:
        from solwyn.client import AsyncSolwyn, Solwyn

        wrapper = AsyncSolwyn if is_async else Solwyn
        return wrapper(
            client,
            api_key=stub.plane.api_key,
            api_url=stub.plane.api_url,
            control_plane_transport=stub,
            lease_enabled=False,
        )

    _QUESTIONS = [{"type": "predicate", "instructions": "Is it a greeting?"}]

    def test_decisions_refusal_frees_its_hold(self) -> None:
        # One cheap 4096-token reservation is about $0.0025; a $0.003 cap fits
        # exactly one, so a stranded hold would deny the chat call after it.
        stub, provider = _PricedPlane(budget_limit=0.003), _Provider()
        with self._wrap_single(stub, provider.client()) as solwyn:
            with pytest.raises(OutputBoundUnenforceableError):
                solwyn.decisions.create(model=CHEAP, input="hi", questions=self._QUESTIONS)
            # The zero-usage confirm rides the reporter queue; deliver it.
            solwyn._solwyn_reporter._drain_confirms()
            solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)

        assert len(provider.bodies) == 1
        assert stub.held == {}
        refused = [c for c in stub.confirms if c["reservation_id"] == "res_1"]
        assert len(refused) == 1
        assert refused[0]["token_details"]["output_tokens"] == 0

    @pytest.mark.asyncio
    async def test_async_decisions_refusal_frees_its_hold(self) -> None:
        stub, provider = _PricedPlane(budget_limit=0.003), _Provider()
        solwyn = self._wrap_single(stub, provider.client(is_async=True), is_async=True)
        try:
            with pytest.raises(OutputBoundUnenforceableError):
                await solwyn.decisions.create(model=CHEAP, input="hi", questions=self._QUESTIONS)
            await solwyn._solwyn_reporter._drain_confirms()
            await solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
        finally:
            await solwyn.close()

        assert len(provider.bodies) == 1
        assert stub.held == {}

    def test_output_bound_refusal_on_chat_frees_its_hold(self) -> None:
        # Qwen with default thinking and no budget is refused before send.
        stub, qwen = _PricedPlane(budget_limit=0.003), _Provider()
        with self._wrap_single(stub, qwen.client(QWEN_URL)) as solwyn:
            with pytest.raises(OutputBoundUnenforceableError):
                solwyn.chat.completions.create(model=CHEAP, messages=MESSAGES)
            solwyn._solwyn_reporter._drain_confirms()
            solwyn.chat.completions.create(
                model=CHEAP, messages=MESSAGES, extra_body={"enable_thinking": False}
            )

        assert len(qwen.bodies) == 1
        assert stub.held == {}
