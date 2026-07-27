"""R5: post-success bookkeeping must never destroy a paid response.

Helper-level tests. End-to-end coverage (response survives an adapter raise)
lives in the settlement-parity-style tests added by the next task.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from solwyn._token_details import TokenDetails
from solwyn.client import (
    _extract_usage_fail_soft,
    _safe_extract_region,
    _safe_extract_service_tier,
)


def _runtime(adapter: MagicMock) -> MagicMock:
    rt = MagicMock()
    rt.adapter = adapter
    rt.sdk_client = MagicMock()
    return rt


@pytest.mark.unit
class TestExtractUsageFailSoft:
    def test_provider_reported_usage_passes_through(self) -> None:
        adapter = MagicMock()
        reported = TokenDetails(input_tokens=10, output_tokens=5)
        adapter.extract_usage.return_value = reported
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(
            _runtime(adapter), object(), estimated_input_tokens=99
        )

        assert result is reported
        assert result.is_estimated is False

    def test_adapter_estimate_preferred_when_present(self) -> None:
        # Mirrors the existing inline behavior: a non-None estimate REPLACES
        # the extracted details (compat provider omitted its usage block).
        adapter = MagicMock()
        adapter.extract_usage.return_value = TokenDetails()
        estimated = TokenDetails(input_tokens=99, output_tokens=0, is_estimated=True)
        adapter.estimate_missing_usage.return_value = estimated

        result = _extract_usage_fail_soft(
            _runtime(adapter), object(), estimated_input_tokens=99
        )

        assert result is estimated

    def test_extract_usage_raise_degrades_to_adapter_estimate(self) -> None:
        adapter = MagicMock()
        adapter.extract_usage.side_effect = RuntimeError("unexpected shape")
        estimated = TokenDetails(input_tokens=42, is_estimated=True)
        adapter.estimate_missing_usage.return_value = estimated

        result = _extract_usage_fail_soft(
            _runtime(adapter), object(), estimated_input_tokens=42
        )

        assert result is estimated

    def test_both_raises_degrade_to_synthetic_estimate(self) -> None:
        adapter = MagicMock()
        adapter.extract_usage.side_effect = RuntimeError("boom")
        adapter.estimate_missing_usage.side_effect = ValueError("boom too")

        result = _extract_usage_fail_soft(
            _runtime(adapter), object(), estimated_input_tokens=123
        )

        assert result.input_tokens == 123
        assert result.output_tokens == 0
        assert result.is_estimated is True

    def test_raise_then_none_estimate_degrades_to_synthetic(self) -> None:
        adapter = MagicMock()
        adapter.extract_usage.side_effect = RuntimeError("boom")
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(
            _runtime(adapter), object(), estimated_input_tokens=7
        )

        assert result.input_tokens == 7
        assert result.is_estimated is True


@pytest.mark.unit
class TestSafeExtractRegionAndTier:
    def test_region_raise_degrades_to_none(self) -> None:
        adapter = MagicMock()
        adapter.extract_region.side_effect = AttributeError("no client attr")
        assert _safe_extract_region(_runtime(adapter)) is None

    def test_region_passthrough(self) -> None:
        adapter = MagicMock()
        adapter.extract_region.return_value = "us-east-1"
        assert _safe_extract_region(_runtime(adapter)) == "us-east-1"

    def test_tier_raise_degrades_to_none(self) -> None:
        adapter = MagicMock()
        adapter.extract_service_tier.side_effect = KeyError("service_tier")
        assert _safe_extract_service_tier(_runtime(adapter), object()) is None

    def test_tier_passthrough(self) -> None:
        adapter = MagicMock()
        adapter.extract_service_tier.return_value = "priority"
        assert _safe_extract_service_tier(_runtime(adapter), object()) == "priority"
