"""Opt-in pytest fixtures for denial-path budget enforcement tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NoReturn

import pytest

from solwyn.client import Solwyn
from solwyn.testing._plane import FakeControlPlane

__all__ = ["solwyn_control_plane", "solwyn_test_client"]


class _DenialOnlyCompletions:
    """Fail loudly if a denial-path test reaches provider dispatch."""

    def create(self, **_kwargs: object) -> NoReturn:
        raise AssertionError(
            "solwyn_test_client is denial-only; provider dispatch must not be reached"
        )


class _DenialOnlyChat:
    def __init__(self) -> None:
        self.completions = _DenialOnlyCompletions()


class _DenialOnlyOpenAI:
    """Minimal OpenAI-shaped sentinel, not a provider response fake."""

    def __init__(self) -> None:
        self.chat = _DenialOnlyChat()

    def with_options(self, **_kwargs: object) -> _DenialOnlyOpenAI:
        return self


_DenialOnlyOpenAI.__module__ = "openai._client"
_DenialOnlyOpenAI.__name__ = "OpenAI"


@pytest.fixture
def solwyn_control_plane() -> FakeControlPlane:
    """Return an independent in-process control plane for each test."""
    return FakeControlPlane()


@pytest.fixture
def solwyn_test_client(solwyn_control_plane: FakeControlPlane) -> Iterator[Solwyn]:
    """Yield a denial-only wrapper backed by ``solwyn_control_plane``."""
    client = solwyn_control_plane.wrap(_DenialOnlyOpenAI(), lease_enabled=False)
    try:
        yield client
    finally:
        client.close()
