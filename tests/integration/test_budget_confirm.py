"""Integration tests for budget check → settlement round-trip.

Settlement rides the reporter now (PJ-1): the enforcer builds the confirm
sans-I/O via ``build_confirm_request`` and the reporter's ``_send_confirm``
performs the blocking POST. There is no ``enforcer.confirm_cost``.
"""

from __future__ import annotations

import pytest
from conftest import SAMPLE_TOKEN_DETAILS

from solwyn.budget import BudgetEnforcer
from solwyn.reporter import MetadataReporter


@pytest.mark.integration
class TestBudgetConfirmRoundTrip:
    """Check then settle: the full reservation lifecycle through the reporter."""

    @pytest.mark.integration
    def test_confirm_with_valid_reservation(
        self, budget_enforcer: BudgetEnforcer, metadata_reporter: MetadataReporter
    ) -> None:
        # Arrange — get a reservation
        result = budget_enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-5.5",
            provider="openai",
        )
        assert result.reservation_id is not None

        # Act — build the confirm sans-I/O on the enforcer, then settle it via
        # the reporter's confirm sender (best-effort, should not raise).
        confirm = budget_enforcer.build_confirm_request(
            reservation_id=result.reservation_id,
            model="gpt-5.5",
            token_details=SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id="call_integration_confirm_valid",
        )
        metadata_reporter._send_confirm(confirm)

    @pytest.mark.integration
    def test_confirm_invalid_reservation_does_not_raise(
        self, budget_enforcer: BudgetEnforcer, metadata_reporter: MetadataReporter
    ) -> None:
        """Settlement is best-effort — a bad reservation_id logs, doesn't raise."""
        confirm = budget_enforcer.build_confirm_request(
            reservation_id="res_nonexistent_000",
            model="gpt-5.5",
            token_details=SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id="call_integration_confirm_invalid",
        )
        metadata_reporter._send_confirm(confirm)
