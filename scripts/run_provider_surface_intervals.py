#!/usr/bin/env python3
"""Run a CI batch with a fresh provider environment and reports per interval."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).parents[1]


def _run(command: list[str], env: dict[str, str]) -> bool:
    print(f"$ {shlex.join(command)}", flush=True)
    try:
        return subprocess.run(command, cwd=ROOT, env=env).returncode == 0
    except OSError as exc:
        print(f"Could not run command: {exc}", flush=True)
        return False


def run_intervals(intervals: list[dict[str, str]], output_dir: Path) -> int:
    failures: list[str] = []
    for row in intervals:
        family, interval = row["family"], row["interval"]
        label = f"{family}/{interval}"
        reports = output_dir / family / interval
        failed_checks: list[str] = []
        env = {
            **os.environ,
            "AWS_EC2_METADATA_DISABLED": "true",
            "SOLWYN_SURFACE_CANARY_FAMILY": family,
        }
        print(f"::group::{label} ({row['packages']})", flush=True)
        try:
            # Nothing installed for one interval can satisfy another interval's imports.
            with TemporaryDirectory(prefix="provider-surface-") as temporary_dir:
                venv = Path(temporary_dir) / "venv"
                python = str(venv / "bin" / "python")
                setup_commands = [
                    [sys.executable, "-m", "venv", str(venv)],
                    [python, "-m", "pip", "install", "-e", "."],
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "pytest>=8.0",
                        "pytest-asyncio>=0.24",
                        "pyyaml>=6.0",
                        row["packages"],
                    ],
                ]
                for command in setup_commands:
                    if not _run(command, env):
                        failed_checks.append("setup")
                        break
                if not failed_checks and not _run(
                    [
                        python,
                        "scripts/capture_surface_inventory.py",
                        "--check",
                        "--interval",
                        interval,
                        "--family",
                        family,
                        "--output-dir",
                        str(reports / "inventory"),
                    ],
                    env,
                ):
                    failed_checks.append("capture")
                # Mirror the old always() steps, even after setup or capture fails.
                if not _run(
                    [
                        python,
                        "scripts/export_surface_contract.py",
                        "--check",
                        "--output",
                        str(reports / "surface_contract/surface-classification.json"),
                        "--reports-dir",
                        str(reports / "inventory"),
                    ],
                    env,
                ):
                    failed_checks.append("classification")
                if not _run(
                    [
                        python,
                        "-m",
                        "pytest",
                        "tests/unit/test_surface_canary.py",
                        "-v",
                        "--tb=short",
                    ],
                    env,
                ):
                    failed_checks.append("canary")
        finally:
            print("::endgroup::", flush=True)
        if failed_checks:
            failures.append(f"{label} ({', '.join(failed_checks)})")
        print(f"Interval result: {label}: {'FAIL' if failed_checks else 'PASS'}", flush=True)
    print(
        f"Provider surface intervals: {len(intervals) - len(failures)} passed, "
        f"{len(failures)} failed",
        flush=True,
    )
    if failures:
        print(f"::error::Failed provider surface intervals: {'; '.join(failures)}", flush=True)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals-json", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/provider_surface_inventory"))
    args = parser.parse_args()
    return run_intervals(json.loads(args.intervals_json), args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
