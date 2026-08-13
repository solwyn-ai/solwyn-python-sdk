#!/usr/bin/env python3
"""Export the deterministic contextual provider capability contract.

``--check`` is safe after hand-editing: an existing file that differs from the
embedded payload FAILS the check and is left untouched; a missing file is
bootstrapped. The bare (no ``--check``) invocation always overwrites — it is
the bootstrap step and will destroy local edits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from solwyn._surfaces import (
    SURFACE_RULES,
    AttributeShape,
    SurfaceContext,
    SurfaceSource,
    payload_fingerprint,
    resolve_surface_rule,
    surface_contract_data,
)

DEFAULT_OUTPUT = (
    Path(__file__).parents[1] / "build" / "surface_contract" / "surface-classification.json"
)
_DIALECT_BY_PROVIDER = {
    "anthropic": "anthropic",
    "azure_openai": "openai",
    "bedrock": "bedrock",
    "google": "google",
    "openai": "openai",
    "openai_compatible": "openai",
    "together": "openai",
}


def render_contract() -> str:
    """Render the Python-owned contract with stable ordering and a final newline."""

    contract: dict[str, Any] = surface_contract_data()
    contract["source_payload_fingerprint"] = payload_fingerprint()
    return json.dumps(contract, indent=2, sort_keys=True) + "\n"


def write_contract(path: Path = DEFAULT_OUTPUT) -> None:
    """Create the destination directory and write the current contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_contract(), encoding="utf-8")


def compare_report_contract(report: dict[str, Any], *, label: str) -> tuple[str, ...]:
    """Return every missing or shape-incompatible raw-surface classification."""

    provider = str(report["provider"])
    context = SurfaceContext(
        provider=provider,
        dialect=_DIALECT_BY_PROVIDER[provider],
        client_shape=str(report["client_shape"]),
        mode=str(report["mode"]),
    )
    mismatches: list[str] = []
    for observation in report["observations"]:
        path = str(observation["path"])
        rule = resolve_surface_rule(
            context=context,
            path=path,
            source=SurfaceSource.RAW,
        )
        if rule is None:
            mismatches.append(f"{label}: no reviewed raw rule for {path!r} in {context}")
            continue
        shape = AttributeShape(
            descriptor_category=str(observation["descriptor_category"]),
            return_shape=str(observation["return_shape"]),
        )
        if not rule.accepts_shape(shape):
            mismatches.append(
                f"{label}: rule {rule.rule_id!r} rejects observed shape {shape} for {path!r}"
            )
    return tuple(mismatches)


def compare_report_directory(path: Path) -> tuple[str, ...]:
    """Check every generated full report, failing closed when none are present."""

    report_paths = sorted(path.glob("*.json"))
    if not report_paths:
        return (f"no provider surface reports found in {path}",)
    mismatches: list[str] = []
    observed_by_context: dict[SurfaceContext, set[str]] = {}
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mismatches.extend(compare_report_contract(report, label=report_path.name))
        provider = str(report["provider"])
        context = SurfaceContext(
            provider=provider,
            dialect=_DIALECT_BY_PROVIDER[provider],
            client_shape=str(report["client_shape"]),
            mode=str(report["mode"]),
        )
        observed_by_context.setdefault(context, set()).update(
            str(observation["path"]) for observation in report["observations"]
        )
    _print_dead_rule_advisories(observed_by_context)
    return tuple(mismatches)


def _print_dead_rule_advisories(
    observed_by_context: dict[SurfaceContext, set[str]],
) -> None:
    """Advisory only: rules resolvable for a captured context but never observed."""

    dead: list[str] = []
    for rule in SURFACE_RULES:
        applicable = [
            context
            for context in observed_by_context
            if resolve_surface_rule(
                context=context,
                path=rule.surface,
                source=SurfaceSource.RAW,
            )
            is rule
        ]
        if applicable and all(
            rule.surface not in observed_by_context[context] for context in applicable
        ):
            dead.append(rule.rule_id)
    for rule_id in sorted(dead)[:20]:
        print(f"advisory: unobserved rule {rule_id}")
    if len(dead) > 20:
        print(f"advisory: {len(dead) - 20} more unobserved rules")


def generate_and_check(
    output: Path = DEFAULT_OUTPUT,
    *,
    reports_dir: Path | None = None,
) -> tuple[str, ...]:
    """Verify (or bootstrap) the ledger without destroying local edits.

    An existing file that differs from the embedded payload is preserved and
    reported as a mismatch — it is either a hand-edit awaiting
    ``embed_surface_rules.py`` or a stale export from another payload. A
    missing file is bootstrapped so fresh CI runners can proceed.
    """

    rendered = render_contract()
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            return (
                f"{output} differs from the embedded payload; run "
                "scripts/embed_surface_rules.py --input <file> to apply edits, "
                "or delete the file and re-export to regenerate",
            )
    else:
        write_contract(output)
    if reports_dir is None:
        return ()
    return compare_report_directory(reports_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="also verify every generated full provider report against the reviewed contract",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.check:
        report_mismatches = generate_and_check(
            args.output,
            reports_dir=args.reports_dir,
        )
        if report_mismatches:
            for report_mismatch in report_mismatches:
                print(report_mismatch)
            return 1
        return 0
    write_contract(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
