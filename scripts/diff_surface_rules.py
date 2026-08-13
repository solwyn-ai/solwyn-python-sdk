#!/usr/bin/env python3
"""Print rule-level differences between two embedded surface payloads.

Usage::

    uv run python scripts/diff_surface_rules.py            # HEAD vs working tree
    uv run python scripts/diff_surface_rules.py main       # main vs working tree
    uv run python scripts/diff_surface_rules.py v0.1 HEAD  # two refs

Paste the output into the PR description of any payload-touching change.
Reads git refs via ``git show``; never writes anything.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = "src/solwyn/_surfaces.py"
BEGIN_MARKER = "# BEGIN GENERATED SURFACE RULE PAYLOAD"
END_MARKER = "# END GENERATED SURFACE RULE PAYLOAD"
WORKTREE = "WORKTREE"


def payload_rows(source_text: str) -> dict[str, list[object]]:
    """Decode the marked payload block into ``{rule_id: compact_row}``."""

    block = source_text.split(BEGIN_MARKER, 1)[1].split(END_MARKER, 1)[0]
    encoded = "".join(
        line.strip().strip('"') for line in block.splitlines() if line.strip().startswith('"')
    )
    payload = json.loads(zlib.decompress(base64.b85decode(encoded.encode("ascii"))))
    if not isinstance(payload, dict):
        raise RuntimeError("surface payload root must be an object")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise RuntimeError("surface payload schema version is unsupported")
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise RuntimeError("surface payload rules must be a list")

    rows: dict[str, list[object]] = {}
    for row in rules:
        if not isinstance(row, list) or not row or not isinstance(row[0], str) or not row[0]:
            raise RuntimeError(
                "surface payload rule must be a nonempty list with a nonempty string id"
            )
        rule_id = row[0]
        if rule_id in rows:
            raise RuntimeError(f"surface payload contains duplicate rule id: {rule_id}")
        rows[rule_id] = row
    return rows


def classify(
    old: dict[str, list[object]],
    new: dict[str, list[object]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    removed = tuple(sorted(old.keys() - new.keys()))
    added = tuple(sorted(new.keys() - old.keys()))
    changed = tuple(
        sorted(rule_id for rule_id in old.keys() & new.keys() if old[rule_id] != new[rule_id])
    )
    return removed, added, changed


def _source_at(ref: str) -> str:
    if ref == WORKTREE:
        return (ROOT / SOURCE).read_text(encoding="utf-8")
    return subprocess.run(
        ["git", "show", "--end-of-options", f"{ref}:{SOURCE}"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_ref", nargs="?", default="HEAD")
    parser.add_argument("new_ref", nargs="?", default=WORKTREE)
    args = parser.parse_args()
    removed, added, changed = classify(
        payload_rows(_source_at(args.old_ref)),
        payload_rows(_source_at(args.new_ref)),
    )
    for label, ids in (("removed", removed), ("added", added), ("changed", changed)):
        for rule_id in ids:
            print(f"{label}: {rule_id}")
    print(f"summary: {len(removed)} removed, {len(added)} added, {len(changed)} changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
