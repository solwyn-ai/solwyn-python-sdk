"""Tests for ProviderAdapter protocol and registry functions.

Uses a local stub adapter to verify the protocol interface — concrete adapters
(OpenAI, Anthropic, Google) are tested separately in their own files.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from solwyn._token_details import TokenDetails
from solwyn.providers._accumulator import StreamUsageAccumulator
from solwyn.providers._protocol import ProviderAdapter

# ---------------------------------------------------------------------------
# Stub adapter — a minimal conforming implementation for protocol testing
# ---------------------------------------------------------------------------


class _NoOpAccumulator:
    """Minimal accumulator for protocol testing."""

    def observe(self, chunk: object) -> None:
        pass

    def finalize(self) -> TokenDetails:
        return TokenDetails()

    def get_service_tier(self) -> str | None:
        return None


class _StubAdapter:
    """Minimal concrete adapter that satisfies the ProviderAdapter protocol."""

    @property
    def name(self) -> str:
        return "stub"

    def detect_client(self, client: Any) -> bool:
        return type(client).__name__ == "StubClient"

    def detect_model(self, model: str) -> bool:
        return model.startswith("stub-")

    def extract_usage(self, response: Any) -> TokenDetails:
        return TokenDetails()

    def extract_service_tier(self, response: Any) -> str | None:
        return None

    def extract_region(self, client: Any) -> str | None:
        return None

    def prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return dict(kwargs)

    def create_stream_accumulator(self) -> StreamUsageAccumulator:
        return _NoOpAccumulator()


class _IncompleteAdapter:
    """Class missing required methods — should NOT satisfy the protocol."""

    @property
    def name(self) -> str:
        return "incomplete"

    # Missing: detect_client, detect_model, extract_usage,
    # extract_service_tier, extract_region, prepare_streaming,
    # create_stream_accumulator


class _NoRegionAdapter:
    """Implements everything EXCEPT extract_region — must NOT satisfy the protocol.

    Pins extract_region as a required protocol member: region is part of the
    cost-attribution contract (Bedrock pricing is per model AND region), so an
    adapter cannot silently opt out of answering the question.
    """

    @property
    def name(self) -> str:
        return "no-region"

    def detect_client(self, client: Any) -> bool:
        return False

    def detect_model(self, model: str) -> bool:
        return False

    def extract_usage(self, response: Any) -> TokenDetails:
        return TokenDetails()

    def extract_service_tier(self, response: Any) -> str | None:
        return None

    def prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return dict(kwargs)

    def create_stream_accumulator(self) -> StreamUsageAccumulator:
        return _NoOpAccumulator()


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderAdapterProtocol:
    def test_conforming_class_satisfies_protocol(self) -> None:
        adapter = _StubAdapter()
        assert isinstance(adapter, ProviderAdapter)

    def test_incomplete_class_fails_protocol_check(self) -> None:
        adapter = _IncompleteAdapter()
        assert not isinstance(adapter, ProviderAdapter)

    def test_plain_object_fails_protocol_check(self) -> None:
        assert not isinstance(object(), ProviderAdapter)

    def test_name_property_returns_string(self) -> None:
        adapter = _StubAdapter()
        assert adapter.name == "stub"

    def test_detect_model_returns_bool(self) -> None:
        adapter = _StubAdapter()
        assert adapter.detect_model("stub-v1") is True
        assert adapter.detect_model("gpt-4") is False

    def test_detect_client_returns_bool(self) -> None:
        adapter = _StubAdapter()

        class StubClient:
            pass

        assert adapter.detect_client(StubClient()) is True
        assert adapter.detect_client(object()) is False

    def test_extract_usage_returns_token_details(self) -> None:
        adapter = _StubAdapter()
        result = adapter.extract_usage(object())
        assert isinstance(result, TokenDetails)

    def test_extract_service_tier_returns_none(self) -> None:
        adapter = _StubAdapter()
        assert adapter.extract_service_tier(object()) is None

    def test_extract_region_returns_none(self) -> None:
        adapter = _StubAdapter()
        assert adapter.extract_region(object()) is None

    def test_adapter_without_extract_region_fails_protocol_check(self) -> None:
        assert not isinstance(_NoRegionAdapter(), ProviderAdapter)

    def test_prepare_streaming_returns_dict(self) -> None:
        adapter = _StubAdapter()
        result = adapter.prepare_streaming({"stream": True})
        assert isinstance(result, dict)

    def test_create_stream_accumulator_returns_accumulator(self) -> None:
        adapter = _StubAdapter()
        acc = adapter.create_stream_accumulator()
        assert isinstance(acc, _NoOpAccumulator)
        assert isinstance(acc, StreamUsageAccumulator)
        # Verify accumulator satisfies the observe/finalize interface
        acc.observe(object())
        result = acc.finalize()
        assert isinstance(result, TokenDetails)
        assert acc.get_service_tier() is None


# ---------------------------------------------------------------------------
# Registry tests (registry functions patched to avoid loading real adapters)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderRegistry:
    """Tests for registry functions in solwyn.providers.

    Registry globals are patched directly so these tests have no dependency
    on OpenAI, Anthropic, or Google adapters being implemented yet.
    """

    @pytest.fixture
    def stub(self) -> _StubAdapter:
        return _StubAdapter()

    @pytest.fixture(autouse=True)
    def patch_registry(self, stub: _StubAdapter) -> Any:
        """Inject stub adapter into registry globals, bypassing _ensure_loaded."""
        import solwyn.providers as providers_mod

        adapters = [stub]
        adapter_by_name = {"stub": stub}
        with (
            patch.object(providers_mod, "_ADAPTERS", adapters),
            patch.object(providers_mod, "_ADAPTER_BY_NAME", adapter_by_name),
        ):
            yield

    def test_get_adapter_by_name_returns_adapter(self, stub: _StubAdapter) -> None:
        from solwyn.providers import get_adapter_by_name

        result = get_adapter_by_name("stub")
        assert result is stub

    def test_get_adapter_by_name_raises_for_unknown(self) -> None:
        from solwyn.providers import get_adapter_by_name

        with pytest.raises(ValueError, match="unknown"):
            get_adapter_by_name("unknown_provider")

    def test_get_adapter_for_model_returns_adapter(self, stub: _StubAdapter) -> None:
        from solwyn.providers import get_adapter_for_model

        result = get_adapter_for_model("stub-v1")
        assert result is stub

    def test_get_adapter_for_model_raises_for_unknown(self) -> None:
        from solwyn.providers import get_adapter_for_model

        with pytest.raises(ValueError, match="completely-unknown-model"):
            get_adapter_for_model("completely-unknown-model")

    def test_get_adapter_for_client_returns_adapter(self, stub: _StubAdapter) -> None:
        from solwyn.providers import get_adapter_for_client

        class StubClient:
            pass

        result = get_adapter_for_client(StubClient())
        assert result is stub

    def test_get_adapter_for_client_raises_for_unknown(self) -> None:
        from solwyn.providers import get_adapter_for_client

        with pytest.raises(ValueError, match="UnknownClient"):

            class UnknownClient:
                pass

            get_adapter_for_client(UnknownClient())
