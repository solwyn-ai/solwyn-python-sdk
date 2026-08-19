"""Double-backed counterpart of the live wrapper budget-denial scenario."""

from __future__ import annotations

import pytest

from solwyn._types import CallStatus
from solwyn.exceptions import BudgetExceededError
from solwyn.testing import FakeControlPlane


class _ProviderDispatchMustNotRun:
    def create(self, **_kwargs: object) -> object:
        raise AssertionError("provider dispatch occurred after hard denial")


class _Chat:
    def __init__(self) -> None:
        self.completions = _ProviderDispatchMustNotRun()


class _OpenAIShape:
    def __init__(self) -> None:
        self.chat = _Chat()

    def with_options(self, **_kwargs: object) -> _OpenAIShape:
        return self


_OpenAIShape.__module__ = "openai._client"
_OpenAIShape.__name__ = "OpenAI"


@pytest.mark.unit
def test_denial_raises_before_provider_dispatch_without_provider_network() -> None:
    plane = FakeControlPlane(
        budget_limit=0.05,
        current_usage=0.10,
        remaining_budget=-0.05,
    )
    client = plane.wrap(_OpenAIShape(), lease_enabled=False)

    try:
        with pytest.raises(BudgetExceededError) as captured:
            client.chat.completions.create(
                model="solwyn-test/deny",
                messages=[],
            )
    finally:
        client.close()

    assert captured.value.budget_limit == pytest.approx(0.05)
    assert captured.value.current_usage > captured.value.budget_limit
    assert captured.value.budget_period == "monthly"
    assert len(plane.checks) == 1
    assert plane.confirms == []
    assert len(plane.ingested) == 1
    assert plane.ingested[0].status is CallStatus.BUDGET_DENIED
