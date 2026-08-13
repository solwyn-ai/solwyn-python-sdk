#!/usr/bin/env python3
"""Validate an expanded surface contract and embed its deterministic payload.

The committed Python payload in ``src/solwyn/_surfaces.py`` is the canonical
rule ledger. Maintain it through this review workflow; do not check in a second
expanded JSON source of truth:

1. Bootstrap the ignored, editable review artifact from the current embedded
   contract::

       uv run python scripts/export_surface_contract.py

2. Edit exact rows in
   ``build/surface_contract/surface-classification.json``. Review additions and
   changes against current provider-inventory evidence, preserving stable rule
   IDs and exact selectors, modes, attribute shapes, and capability scopes.
3. Validate the expanded rows, canonicalize their ordering, and replace the
   committed Python payload::

       uv run python scripts/embed_surface_rules.py \
           --input build/surface_contract/surface-classification.json

4. Re-export from the new canonical payload and prove payload identity plus
   provider/report compatibility::

       make check-surface-contract
       make check-provider-surfaces
       uv run python scripts/export_surface_contract.py --check \
           --reports-dir build/provider_surface_inventory

Step 3 prints the rule-level delta against the installed payload and refuses
stale exports (``--allow-stale`` to override) and rule removals
(``--allow-removals``). ``make check-surface-contract`` is now edit-preserving:
a differing expanded contract FAILS the check and is left on disk untouched.

``--check`` is verification-only: it exits nonzero when the supplied expanded
contract would change the Python payload, and never writes or applies edits.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solwyn._surfaces import (
    CONTRACT_VERSION,
    AttributeShape,
    CapabilityScope,
    SurfaceCondition,
    SurfaceKind,
    SurfaceRule,
    SurfaceSelector,
    SurfaceSource,
    UsageBasis,
    payload_fingerprint,
)

ROOT = Path(__file__).parents[1]
DEFAULT_INPUT = ROOT / "build" / "surface_contract" / "surface-classification.json"
DEFAULT_SOURCE = ROOT / "src" / "solwyn" / "_surfaces.py"
BEGIN_MARKER = "# BEGIN GENERATED SURFACE RULE PAYLOAD"
END_MARKER = "# END GENERATED SURFACE RULE PAYLOAD"
EXPECTED_UNKNOWN_POLICY = {
    "kind": SurfaceKind.UNKNOWN.value,
    "policy_action": "posture",
    "dispatch_action": "posture",
    "acknowledgment": "exact_observed_terminal_only",
}


def _optional_enum(enum_type: type[Any], value: object) -> Any | None:
    return None if value is None else enum_type(value)


def _canonical_expanded_row(row: Mapping[str, Any]) -> dict[str, Any]:
    canonical = dict(row)
    canonical["selectors"] = sorted(
        row["selectors"],
        key=lambda selector: (
            selector["provider"] or "",
            selector["dialect"] or "",
            selector["client_shape"] or "",
            selector["mode"] or "",
        ),
    )
    canonical["expected_attribute_shapes"] = sorted(
        row["expected_attribute_shapes"],
        key=lambda shape: (shape["descriptor_category"], shape["return_shape"]),
    )
    return canonical


def _validated_rules(contract: Mapping[str, Any]) -> tuple[SurfaceRule, ...]:
    required = {"contract_version", "schema_version", "unknown_policy", "rules"}
    if not required <= set(contract) <= required | {"source_payload_fingerprint"}:
        raise RuntimeError("surface contract root fields are invalid")
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("surface contract version is unsupported")
    if contract.get("schema_version") != CONTRACT_VERSION:
        raise RuntimeError("surface contract schema is unsupported")
    if contract.get("unknown_policy") != EXPECTED_UNKNOWN_POLICY:
        raise RuntimeError("surface contract unknown policy is invalid")
    rows = contract.get("rules")
    if not isinstance(rows, list):
        raise RuntimeError("surface contract rules must be a list")

    rules: list[SurfaceRule] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise RuntimeError("surface contract rule must be an object")
        row: dict[str, Any] = raw_row
        try:
            rule = SurfaceRule(
                rule_id=str(row["id"]),
                surface=str(row["surface"]),
                selectors=tuple(
                    SurfaceSelector(
                        provider=selector["provider"],
                        dialect=selector["dialect"],
                        client_shape=selector["client_shape"],
                        mode=selector["mode"],
                    )
                    for selector in row["selectors"]
                ),
                kind=SurfaceKind(row["kind"]),
                source=SurfaceSource(row["source"]),
                expected_shapes=tuple(
                    AttributeShape(
                        descriptor_category=shape["descriptor_category"],
                        return_shape=shape["return_shape"],
                    )
                    for shape in row["expected_attribute_shapes"]
                ),
                usage_basis=_optional_enum(UsageBasis, row["usage_basis"]),
                acknowledgment_token=row["acknowledgment_token"],
                capability_scope=_optional_enum(CapabilityScope, row["capability_scope"]),
                condition=_optional_enum(SurfaceCondition, row["condition"]),
                reason=row["reason"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid expanded surface contract rule") from exc
        segments = rule.rule_id.split(".")
        if (
            len(segments) not in {4, 5}
            or segments[0] != "surface"
            or segments[2] != rule.kind.value
        ):
            raise RuntimeError(
                f"rule id {rule.rule_id!r} does not encode its kind ({rule.kind.value})"
            )
        if rule.to_data() != _canonical_expanded_row(row):
            raise RuntimeError(f"rule {rule.rule_id!r} does not match validated export")
        rules.append(rule)

    ordered = tuple(sorted(rules, key=lambda rule: (rule.surface, rule.rule_id)))
    if len({rule.rule_id for rule in ordered}) != len(ordered):
        raise RuntimeError("surface contract contains duplicate rule ids")
    return ordered


def rule_delta(
    new_rules: tuple[SurfaceRule, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (removed, added, changed) rule ids vs the installed payload."""

    from solwyn._surfaces import SURFACE_RULES

    current = {rule.rule_id: rule for rule in SURFACE_RULES}
    incoming = {rule.rule_id: rule for rule in new_rules}
    removed = tuple(sorted(current.keys() - incoming.keys()))
    added = tuple(sorted(incoming.keys() - current.keys()))
    changed = tuple(
        sorted(
            rule_id
            for rule_id in current.keys() & incoming.keys()
            if current[rule_id].to_data() != incoming[rule_id].to_data()
        )
    )
    return removed, added, changed


def _selector_key(selector: SurfaceSelector) -> tuple[str, str, str, str]:
    return (
        selector.provider or "",
        selector.dialect or "",
        selector.client_shape or "",
        selector.mode or "",
    )


def compact_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated compact representation decoded by ``_surfaces``."""

    compact_rules = []
    for rule in _validated_rules(contract):
        compact_rules.append(
            [
                rule.rule_id,
                rule.surface,
                [
                    [
                        selector.provider,
                        selector.dialect,
                        selector.client_shape,
                        selector.mode,
                    ]
                    for selector in sorted(rule.selectors, key=_selector_key)
                ],
                rule.kind.value,
                rule.source.value,
                [
                    [shape.descriptor_category, shape.return_shape]
                    for shape in sorted(rule.expected_shapes)
                ],
                rule.usage_basis.value if rule.usage_basis is not None else None,
                rule.acknowledgment_token,
                rule.capability_scope.value if rule.capability_scope is not None else None,
                rule.condition.value if rule.condition is not None else None,
                rule.reason,
            ]
        )
    return {"rules": compact_rules, "schema_version": CONTRACT_VERSION}


def encode_contract(contract: Mapping[str, Any]) -> str:
    """Return a deterministic base85/zlib payload for an expanded contract."""

    canonical = json.dumps(
        compact_contract(contract),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.b85encode(zlib.compress(canonical, level=9)).decode("ascii")


def _render_payload(encoded: str) -> str:
    chunks = [encoded[index : index + 100] for index in range(0, len(encoded), 100)]
    lines = "".join(f'    "{chunk}"\n' for chunk in chunks)
    return f"{BEGIN_MARKER}\n_GENERATED_SURFACE_RULE_PAYLOAD = (\n{lines})\n{END_MARKER}"


def rewrite_source(source: str, encoded: str) -> str:
    """Replace exactly one generated payload block in Python source text."""

    if source.count(BEGIN_MARKER) != 1 or source.count(END_MARKER) != 1:
        raise RuntimeError("surface source must contain one generated payload block")
    before, marked = source.split(BEGIN_MARKER, 1)
    _old_payload, after = marked.split(END_MARKER, 1)
    return f"{before}{_render_payload(encoded)}{after}"


def decode_source_payload(source: str) -> dict[str, Any]:
    """Decode the compact JSON payload from a marked surface source file."""

    if source.count(BEGIN_MARKER) != 1 or source.count(END_MARKER) != 1:
        raise RuntimeError("surface source must contain one generated payload block")
    _before, marked = source.split(BEGIN_MARKER, 1)
    payload_block, _after = marked.split(END_MARKER, 1)
    try:
        parsed = ast.parse(payload_block.strip())
        statement = parsed.body[0]
        if (
            len(parsed.body) != 1
            or not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id != "_GENERATED_SURFACE_RULE_PAYLOAD"
        ):
            raise RuntimeError("invalid generated surface payload assignment")
        encoded = ast.literal_eval(statement.value)
        if not isinstance(encoded, str):
            raise RuntimeError("generated surface payload must be a string")
        decoded = json.loads(zlib.decompress(base64.b85decode(encoded.encode("ascii"))))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("invalid generated surface payload") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("generated surface payload must decode to an object")
    return decoded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument("--allow-removals", action="store_true")
    return parser


def main() -> int:
    import solwyn

    installed = Path(solwyn.__file__).resolve()
    expected = (ROOT / "src" / "solwyn" / "__init__.py").resolve()
    if installed != expected:
        print(f"solwyn resolves to {installed}, not this checkout ({expected}); run via uv run")
        return 1
    args = _parser().parse_args()
    contract = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise RuntimeError("surface contract root must be an object")
    stamp = contract.get("source_payload_fingerprint")
    if stamp != payload_fingerprint() and not args.allow_stale:
        print(
            "stale expanded contract: source_payload_fingerprint "
            f"{stamp!r} does not match the installed payload; re-export on this "
            "branch or pass --allow-stale to override"
        )
        return 1
    removed, added, changed = rule_delta(_validated_rules(contract))
    if not args.check:
        for label, ids in (("removed", removed), ("added", added), ("changed", changed)):
            for rule_id in ids:
                print(f"{label}: {rule_id}")
        if removed and not args.allow_removals:
            print(f"refusing to remove {len(removed)} rule(s); pass --allow-removals")
            return 1
    existing = args.source.read_text(encoding="utf-8")
    if args.check:
        return 0 if decode_source_payload(existing) == compact_contract(contract) else 1
    rewritten = rewrite_source(existing, encode_contract(contract))
    args.source.write_text(rewritten, encoding="utf-8")
    print(args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
