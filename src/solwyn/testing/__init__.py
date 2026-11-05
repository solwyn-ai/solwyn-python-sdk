"""FakeControlPlane routes Solwyn control-plane traffic without network I/O.

It preserves real wire shapes and Pydantic models but does not mock provider traffic.
It performs no pricing; the API owns pricing.
"""

from solwyn.testing._plane import MAGIC_MODELS, FakeBudgetRule, FakeControlPlane

__all__ = ["FakeBudgetRule", "FakeControlPlane", "MAGIC_MODELS"]
