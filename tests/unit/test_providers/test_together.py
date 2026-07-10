"""Tests for the first-class Together provider adapter."""

from __future__ import annotations

from typing import Any

import pytest

from solwyn.providers.openai_compatible import OpenAICompatibleAdapter
from solwyn.providers.together import TogetherAdapter


def _make_client(module_path: str, class_name: str, *, base_url: str | None = None) -> Any:
    """Return a client with controlled type metadata and optional base URL."""
    FakeClient = type(class_name, (), {"__module__": module_path})
    client = FakeClient()
    if base_url is not None:
        client.base_url = base_url  # type: ignore[attr-defined]
    return client


def _adapter() -> TogetherAdapter:
    return TogetherAdapter()


@pytest.mark.unit
class TestTogetherAdapterDetection:
    @pytest.mark.parametrize(
        ("module_path", "class_name"),
        [
            ("together", "Together"),
            ("together", "AsyncTogether"),
            ("together.client", "Together"),
            ("together.async_client", "AsyncTogether"),
        ],
    )
    def test_detects_native_together_clients(self, module_path: str, class_name: str) -> None:
        assert _adapter().detect_client(_make_client(module_path, class_name)) is True

    @pytest.mark.parametrize("class_name", ["Client", "AsyncClient", "Completions"])
    def test_rejects_non_client_objects_from_together_modules(self, class_name: str) -> None:
        assert _adapter().detect_client(_make_client("together.resources", class_name)) is False

    @pytest.mark.parametrize("module_path", ["other", "not_together.client"])
    @pytest.mark.parametrize("class_name", ["Together", "AsyncTogether"])
    def test_rejects_together_named_clients_from_other_modules(
        self, module_path: str, class_name: str
    ) -> None:
        assert _adapter().detect_client(_make_client(module_path, class_name)) is False

    def test_preserves_openai_client_detection_at_together_host(self) -> None:
        client = _make_client(
            "openai._client",
            "OpenAI",
            base_url="https://api.together.xyz/v1",
        )
        assert _adapter().detect_client(client) is True


@pytest.mark.unit
def test_together_adapter_identity() -> None:
    adapter = _adapter()
    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.name == "together"
    assert adapter.dialect == "openai"
