"""Deterministic, in-process test doubles for Solwyn's control plane."""

from solwyn.testing._plane import MAGIC_MODELS, FakeControlPlane

__all__ = ["FakeControlPlane", "MAGIC_MODELS"]
