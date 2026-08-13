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
        "sha256:152d7eed2760556a691626c44afca4e5d4b75cb8fc75ab2543aae653e14ab870"
    ),  # 3169 rules
    ("openai", "openai", "openai_sdk", "async"): (
        "sha256:152d7eed2760556a691626c44afca4e5d4b75cb8fc75ab2543aae653e14ab870"
    ),  # 3169 rules
    ("azure_openai", "openai", "openai_sdk", "sync"): (
        "sha256:a7a1cbe2920d240d5e254ab3b63108dc2456b0dee8285c421831d16eb55e3624"
    ),  # 3167 rules
    ("azure_openai", "openai", "openai_sdk", "async"): (
        "sha256:a7a1cbe2920d240d5e254ab3b63108dc2456b0dee8285c421831d16eb55e3624"
    ),  # 3167 rules
    ("openai_compatible", "openai", "openai_sdk", "sync"): (
        "sha256:a7a1cbe2920d240d5e254ab3b63108dc2456b0dee8285c421831d16eb55e3624"
    ),  # 3167 rules
    ("openai_compatible", "openai", "openai_sdk", "async"): (
        "sha256:a7a1cbe2920d240d5e254ab3b63108dc2456b0dee8285c421831d16eb55e3624"
    ),  # 3167 rules
    ("together", "openai", "openai_sdk", "sync"): (
        "sha256:a7a1cbe2920d240d5e254ab3b63108dc2456b0dee8285c421831d16eb55e3624"
    ),  # 3167 rules
    ("together", "openai", "openai_sdk", "async"): (
        "sha256:a7a1cbe2920d240d5e254ab3b63108dc2456b0dee8285c421831d16eb55e3624"
    ),  # 3167 rules
    ("together", "openai", "native_together", "sync"): (
        "sha256:c9fec947345de3ab372cb45905606004c808f9d0fce1576e58f59b2cd8dce9f5"
    ),  # 1365 rules
    ("together", "openai", "native_together", "async"): (
        "sha256:c9fec947345de3ab372cb45905606004c808f9d0fce1576e58f59b2cd8dce9f5"
    ),  # 1365 rules
    ("anthropic", "anthropic", "anthropic_sdk", "sync"): (
        "sha256:0ea288da03cb3695b9ad9c3156e50f2399f28683644fc470cdcd88d9e6db65c6"
    ),  # 1301 rules
    ("anthropic", "anthropic", "anthropic_sdk", "async"): (
        "sha256:8f0773dc0c8a233ea928a151e74eb3ab59899fb0561579ff323b512dce44acd5"
    ),  # 1304 rules
    ("google", "google", "google_genai", "sync"): (
        "sha256:8a213cab989f33c7fc10c281cddaee0773efe447fe35ec37d598d2e103b99acb"
    ),  # 392 rules
    ("google", "google", "google_genai", "async"): (
        "sha256:80a163721aff65981d3410e7e2212e63f872d3d283edb35d02eb9290ea21aa46"
    ),  # 206 rules
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
