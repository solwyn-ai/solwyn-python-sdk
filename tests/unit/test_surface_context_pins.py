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
        "sha256:a019e77d8996cf59c8f7ba7738619402b34489b46d475bb0326900ef76424183"
    ),  # 3173 rules
    ("openai", "openai", "openai_sdk", "async"): (
        "sha256:a019e77d8996cf59c8f7ba7738619402b34489b46d475bb0326900ef76424183"
    ),  # 3173 rules
    ("azure_openai", "openai", "openai_sdk", "sync"): (
        "sha256:57b2308e231074e49a4bee4dcecd10a3a5425f99ce3c6b3870277a1925153fc9"
    ),  # 3171 rules
    ("azure_openai", "openai", "openai_sdk", "async"): (
        "sha256:57b2308e231074e49a4bee4dcecd10a3a5425f99ce3c6b3870277a1925153fc9"
    ),  # 3171 rules
    ("openai_compatible", "openai", "openai_sdk", "sync"): (
        "sha256:84b0d2c5c5ce662ad3011a283211b51a8957f7e31d54b21d4c2beaa3cb9370ef"
    ),  # 3167 rules
    ("openai_compatible", "openai", "openai_sdk", "async"): (
        "sha256:84b0d2c5c5ce662ad3011a283211b51a8957f7e31d54b21d4c2beaa3cb9370ef"
    ),  # 3167 rules
    ("together", "openai", "openai_sdk", "sync"): (
        "sha256:84b0d2c5c5ce662ad3011a283211b51a8957f7e31d54b21d4c2beaa3cb9370ef"
    ),  # 3167 rules
    ("together", "openai", "openai_sdk", "async"): (
        "sha256:84b0d2c5c5ce662ad3011a283211b51a8957f7e31d54b21d4c2beaa3cb9370ef"
    ),  # 3167 rules
    ("together", "openai", "native_together", "sync"): (
        "sha256:b97e9a39d2b9f423b4fee53d7b58d139c80cbbaf4cdea8a5a593e59b975dd2c0"
    ),  # 1365 rules
    ("together", "openai", "native_together", "async"): (
        "sha256:b97e9a39d2b9f423b4fee53d7b58d139c80cbbaf4cdea8a5a593e59b975dd2c0"
    ),  # 1365 rules
    ("anthropic", "anthropic", "anthropic_sdk", "sync"): (
        "sha256:73ca7a41af6b0d56031f61b1d49827e9fed4dbc29ffcb314390b02558b2230b5"
    ),  # 1391 rules
    ("anthropic", "anthropic", "anthropic_sdk", "async"): (
        "sha256:dd7a772040126a4fbfc3a86f6831dba045396150abefd7470c51a16aea2ca82e"
    ),  # 1394 rules
    ("google", "google", "google_genai", "sync"): (
        "sha256:b8edd003fedf6cea1de1f5ef8b66c0c28593606bce50cd52a1d2d6bc1ea52318"
    ),  # 392 rules
    ("google", "google", "google_genai", "async"): (
        "sha256:5be9e299858993ab5f2954941b9a5188584ae538b99ef4294b00980d480e05cc"
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
