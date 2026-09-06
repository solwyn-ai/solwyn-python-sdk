"""Literal per-context pins over the full reviewed rule ledger.

Any rule addition, removal, or field change in a context's reachable set moves
that context's digest, forcing a reviewed literal update here. Use
``uv run python scripts/diff_surface_rules.py`` to see the rule-level delta.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from solwyn._surfaces import DIALECT_BY_PROVIDER, SURFACE_RULES, SurfaceContext

ROOT = Path(__file__).parents[2]

DECLARED_CONTEXT_DIGESTS: dict[tuple[str, str, str, str], str] = {
    ("openai", "openai", "openai_sdk", "sync"): (
        "sha256:0e7496c626906ffca14e7c1f47eee45c5c3c9ecb6bba1dfeadfd41c455d2b239"
    ),  # 3192 rules
    ("openai", "openai", "openai_sdk", "async"): (
        "sha256:0e7496c626906ffca14e7c1f47eee45c5c3c9ecb6bba1dfeadfd41c455d2b239"
    ),  # 3192 rules
    ("azure_openai", "openai", "openai_sdk", "sync"): (
        "sha256:fa8062afa1ac0a2778619400e5a988dac003d9eaa6bef797a86ff1f37b84f307"
    ),  # 3190 rules
    ("azure_openai", "openai", "openai_sdk", "async"): (
        "sha256:fa8062afa1ac0a2778619400e5a988dac003d9eaa6bef797a86ff1f37b84f307"
    ),  # 3190 rules
    ("openai_compatible", "openai", "openai_sdk", "sync"): (
        "sha256:d35901d8061f5f5decb764d1e7ece1293f48ded188405a5c98a5189554c85c1f"
    ),  # 3186 rules
    ("openai_compatible", "openai", "openai_sdk", "async"): (
        "sha256:d35901d8061f5f5decb764d1e7ece1293f48ded188405a5c98a5189554c85c1f"
    ),  # 3186 rules
    ("together", "openai", "openai_sdk", "sync"): (
        "sha256:d35901d8061f5f5decb764d1e7ece1293f48ded188405a5c98a5189554c85c1f"
    ),  # 3186 rules
    ("together", "openai", "openai_sdk", "async"): (
        "sha256:d35901d8061f5f5decb764d1e7ece1293f48ded188405a5c98a5189554c85c1f"
    ),  # 3186 rules
    ("together", "openai", "native_together", "sync"): (
        "sha256:b97e9a39d2b9f423b4fee53d7b58d139c80cbbaf4cdea8a5a593e59b975dd2c0"
    ),  # 1365 rules
    ("together", "openai", "native_together", "async"): (
        "sha256:b97e9a39d2b9f423b4fee53d7b58d139c80cbbaf4cdea8a5a593e59b975dd2c0"
    ),  # 1365 rules
    ("anthropic", "anthropic", "anthropic_sdk", "sync"): (
        "sha256:ea555e4c42c2b77d57feb7e6ce32ef3036d3f7e6f7d692eb39328e4a77d9578c"
    ),  # 2145 rules
    ("anthropic", "anthropic", "anthropic_sdk", "async"): (
        "sha256:9bc9e9083b1144a74252f0bda46d5cb891f1b347f9e81b8cf3c6dc6252abcec7"
    ),  # 2148 rules
    ("google", "google", "google_genai", "sync"): (
        "sha256:3e14882f135fa3930ddccbae45f9dc8a705038083c354e89fb0118af234d93ae"
    ),  # 412 rules
    ("google", "google", "google_genai", "async"): (
        "sha256:3ea181ad2bf325752ff760091136dbad0daaf5947f2feedcfff7b14ca6a4e1f3"
    ),  # 216 rules
    ("google", "google", "google_generativeai", "sync"): (
        "sha256:bbd3ee7bcf5c33d6153e3b7203de3a8e738454a6a4cbe6d002a89bdc961118d0"
    ),  # 24 rules
    ("bedrock", "bedrock", "bedrock_boto3", "sync"): (
        "sha256:37e60b8a2961fc808927c06bc0bb68221cc7adf1b3a0f7e32f16c44a1b12d789"
    ),  # 34 rules
    ("bedrock", "bedrock", "bedrock_aioboto3", "async"): (
        "sha256:37e60b8a2961fc808927c06bc0bb68221cc7adf1b3a0f7e32f16c44a1b12d789"
    ),  # 34 rules
}


def _reachable_rows(context: SurfaceContext) -> list[dict[str, object]]:
    return sorted(
        (
            rule.to_data()
            for rule in SURFACE_RULES
            if any(selector.specificity(context) is not None for selector in rule.selectors)
        ),
        key=lambda row: (str(row["surface"]), str(row["id"])),
    )


def _digest(rows: list[dict[str, object]]) -> str:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _capture_module() -> ModuleType:
    path = ROOT / "scripts" / "capture_surface_inventory.py"
    spec = importlib.util.spec_from_file_location("capture_surface_inventory_context_pins", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load inventory capture script at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("context_tuple", sorted(DECLARED_CONTEXT_DIGESTS))
def test_context_rule_set_matches_the_reviewed_literal_digest(
    context_tuple: tuple[str, str, str, str],
) -> None:
    # Arrange
    provider, dialect, client_shape, mode = context_tuple
    context = SurfaceContext(
        provider=provider,
        dialect=dialect,
        client_shape=client_shape,
        mode=mode,
    )

    # Act
    rows = _reachable_rows(context)

    # Assert
    assert rows, context
    assert _digest(rows) == DECLARED_CONTEXT_DIGESTS[context_tuple]


@pytest.mark.unit
def test_every_rule_is_reachable_from_at_least_one_declared_context() -> None:
    # Arrange
    reachable: set[str] = set()

    # Act
    for provider, dialect, client_shape, mode in DECLARED_CONTEXT_DIGESTS:
        context = SurfaceContext(
            provider=provider,
            dialect=dialect,
            client_shape=client_shape,
            mode=mode,
        )
        reachable.update(str(row["id"]) for row in _reachable_rows(context))
    all_ids = {rule.rule_id for rule in SURFACE_RULES}

    # Assert
    assert reachable == all_ids, sorted(all_ids - reachable)[:20]


@pytest.mark.unit
def test_declared_digest_contexts_match_capture_shape_registry() -> None:
    # Arrange
    capture = _capture_module()
    capture_contexts = {
        (
            spec.provider,
            DIALECT_BY_PROVIDER[spec.provider],
            spec.client_shape,
            spec.mode,
        )
        for spec in capture._SHAPES
    }

    # Act
    pinned_contexts = set(DECLARED_CONTEXT_DIGESTS)

    # Assert
    assert pinned_contexts == capture_contexts
