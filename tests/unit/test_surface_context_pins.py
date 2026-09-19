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
        "sha256:ceb5482f58a7fc134d5a96174ef86756fd3096395ba7e211b7ed51cd1573c745"
    ),  # 3905 rules
    ("openai", "openai", "openai_sdk", "async"): (
        "sha256:ceb5482f58a7fc134d5a96174ef86756fd3096395ba7e211b7ed51cd1573c745"
    ),  # 3905 rules
    ("azure_openai", "openai", "openai_sdk", "sync"): (
        "sha256:9a28ca94070bfe3df66d1ebdb4466c1c13aa6af0b15193f764ef9d0fe7499c5a"
    ),  # 3903 rules
    ("azure_openai", "openai", "openai_sdk", "async"): (
        "sha256:9a28ca94070bfe3df66d1ebdb4466c1c13aa6af0b15193f764ef9d0fe7499c5a"
    ),  # 3903 rules
    ("openai_compatible", "openai", "openai_sdk", "sync"): (
        "sha256:093b6caa36f16f8acc83586252d61d073912039c1daf15dfeaff9372201c5a21"
    ),  # 3899 rules
    ("openai_compatible", "openai", "openai_sdk", "async"): (
        "sha256:093b6caa36f16f8acc83586252d61d073912039c1daf15dfeaff9372201c5a21"
    ),  # 3899 rules
    ("together", "openai", "openai_sdk", "sync"): (
        "sha256:093b6caa36f16f8acc83586252d61d073912039c1daf15dfeaff9372201c5a21"
    ),  # 3899 rules
    ("together", "openai", "openai_sdk", "async"): (
        "sha256:093b6caa36f16f8acc83586252d61d073912039c1daf15dfeaff9372201c5a21"
    ),  # 3899 rules
    ("together", "openai", "native_together", "sync"): (
        "sha256:ed870fe92bc951dc1b8256d99ec24de59c131c663dc659445a60d6f22c9434d1"
    ),  # 1464 rules
    ("together", "openai", "native_together", "async"): (
        "sha256:ed870fe92bc951dc1b8256d99ec24de59c131c663dc659445a60d6f22c9434d1"
    ),  # 1464 rules
    ("anthropic", "anthropic", "anthropic_sdk", "sync"): (
        "sha256:ea555e4c42c2b77d57feb7e6ce32ef3036d3f7e6f7d692eb39328e4a77d9578c"
    ),  # 2145 rules
    ("anthropic", "anthropic", "anthropic_sdk", "async"): (
        "sha256:9bc9e9083b1144a74252f0bda46d5cb891f1b347f9e81b8cf3c6dc6252abcec7"
    ),  # 2148 rules
    ("google", "google", "google_genai", "sync"): (
        "sha256:00ade156e456cf5b49c4ea27f988c9afa6250eb7e7bdf36f0c82e7554d932036"
    ),  # 416 rules
    ("google", "google", "google_genai", "async"): (
        "sha256:66f07c2e650c313cac62a97054a4b45f3d56614b17b802de42b1a944314195cb"
    ),  # 218 rules
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
