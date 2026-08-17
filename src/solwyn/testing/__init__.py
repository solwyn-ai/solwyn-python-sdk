"""The testing package uses zero network traffic.

It preserves real wire shapes and Pydantic models. It performs no pricing; the
API owns pricing.
"""

from solwyn.testing._plane import MAGIC_MODELS, FakeControlPlane

__all__ = ["FakeControlPlane", "MAGIC_MODELS"]
