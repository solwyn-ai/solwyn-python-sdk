#!/usr/bin/env python3
"""Export the deterministic contextual provider capability contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solwyn._surfaces import surface_contract_data

DEFAULT_OUTPUT = Path(__file__).parents[1] / "docs" / "contracts" / "surface-classification.json"


def render_contract() -> str:
    """Render the Python-owned contract with stable ordering and a final newline."""

    return json.dumps(surface_contract_data(), indent=2, sort_keys=True) + "\n"


def write_contract(path: Path = DEFAULT_OUTPUT) -> None:
    """Create the destination directory and write the current contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_contract(), encoding="utf-8")


def compare_contract(path: Path = DEFAULT_OUTPUT) -> str | None:
    """Return a stable drift message, or ``None`` when the file matches."""

    if not path.exists() or path.read_text(encoding="utf-8") != render_contract():
        return f"surface contract drift: {path}"
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.check:
        mismatch = compare_contract(args.output)
        if mismatch is not None:
            print(mismatch)
            return 1
        return 0
    write_contract(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
