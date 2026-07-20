"""E2E: a server-side hard_deny budget stops the call BEFORE the provider."""

from __future__ import annotations

import pytest
from conftest import Credentials, WireRecorder
from fake_provider import FakeProviderServer

from solwyn._types import CallStatus
from solwyn.exceptions import BudgetExceededError

MESSAGES = [{"role": "user", "content": "Say hello in five words."}]


@pytest.mark.integration
class TestBudgetDenial:
    """hard_deny project over limit -> BudgetExceededError, provider untouched."""

    @pytest.mark.integration
    def test_denial_raises_without_calling_provider(
        self,
        make_wrapped_client,
        fake_provider: FakeProviderServer,
        hard_denied_credentials: Credentials,
    ) -> None:
        client = make_wrapped_client(credentials=hard_denied_credentials)
        recorder = WireRecorder().attach(client)

        with pytest.raises(BudgetExceededError) as exc_info:
            client.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        # The provider endpoint was never contacted.
        assert fake_provider.request_count == 0
        # No spend was settled.
        assert recorder.settlements == []
        # The denial itself was reported for dashboard accuracy.
        assert len(recorder.events) == 1
        assert recorder.events[0].status == CallStatus.BUDGET_DENIED
        # The error carries the server's budget state.
        assert exc_info.value.budget_limit == pytest.approx(0.05)
        assert exc_info.value.current_usage > exc_info.value.budget_limit
