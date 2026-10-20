#!/usr/bin/env python3
"""Run a CI batch with a fresh provider environment and reports per interval."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).parents[1]
PIP_TIMEOUT_SECONDS = 600
CHECK_TIMEOUT_SECONDS = 300


def _run(command: list[str], env: dict[str, str], *, timeout: int) -> bool:
    print(f"$ {shlex.join(command)}", flush=True)
    try:
        with subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True) as process:
            try:
                return process.wait(timeout=timeout) == 0
            except subprocess.TimeoutExpired:
                # pip may spawn build backends; terminate the whole group before cleanup.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
    except OSError as exc:
        print(f"Could not run command: {exc}", flush=True)
        return False


def _check(
    stage: str,
    command: list[str],
    env: dict[str, str],
    failed_checks: list[str],
    *,
    timeout: int = CHECK_TIMEOUT_SECONDS,
) -> bool:
    try:
        if _run(command, env, timeout=timeout):
            return True
    except subprocess.TimeoutExpired:
        print(f"{stage} timed out after {timeout} seconds", flush=True)
        failed_checks.append(f"{stage} (timeout)")
        return False
    failed_checks.append(stage)
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
                    ([sys.executable, "-m", "venv", str(venv)], CHECK_TIMEOUT_SECONDS),
                    ([python, "-m", "pip", "install", "-e", "."], PIP_TIMEOUT_SECONDS),
                    (
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
                        PIP_TIMEOUT_SECONDS,
                    ),
                ]
                for command, timeout in setup_commands:
                    if not _check("setup", command, env, failed_checks, timeout=timeout):
                        break
                if not failed_checks:
                    _check(
                        "capture",
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
                        failed_checks,
                    )
                # Mirror the old always() steps, even after setup or capture fails.
                _check(
                    "classification",
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
                    failed_checks,
                )
                _check(
                    "canary",
                    [
                        python,
                        "-m",
                        "pytest",
                        "tests/unit/test_surface_canary.py",
                        "-v",
                        "--tb=short",
                    ],
                    env,
                    failed_checks,
                )
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
