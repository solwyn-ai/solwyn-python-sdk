"""Consolidated CI must isolate intervals and finish every check after failures."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.unit


def _runner() -> ModuleType:
    path = ROOT / "scripts" / "run_provider_surface_intervals.py"
    spec = importlib.util.spec_from_file_location("provider_surface_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load provider surface runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("event", ["pull_request", "push", "schedule", "workflow_dispatch"])
@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="the GitHub Actions catalog step requires bash and jq",
)
def test_ci_batches_preserve_every_selected_catalog_interval(event: str, tmp_path: Path) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    catalog_step = next(
        step
        for step in workflow["jobs"]["provider-surface-matrix"]["steps"]
        if step.get("id") == "catalog"
    )
    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-e", "-c", catalog_step["run"]],
        cwd=ROOT,
        env={**os.environ, "GITHUB_EVENT_NAME": event, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=True,
    )
    matrix = json.loads(output.read_text().removeprefix("matrix=").strip())
    batches = matrix["include"]
    assert 1 <= len(batches) <= 3
    actual = [row for batch in batches for row in batch["intervals"]]
    catalog = json.loads((ROOT / "tests/provider_surface_intervals.json").read_text())["include"]
    expected = [row for row in catalog if event != "schedule" or row["interval"] == "latest"]
    assert sorted(actual, key=lambda row: (row["family"], row["interval"])) == sorted(
        expected, key=lambda row: (row["family"], row["interval"])
    )
    assert len({len(batch["intervals"]) for batch in batches}) == 1
    for row in expected:
        assert f"{row['family']}/{row['interval']}" in result.stdout


@pytest.mark.parametrize(
    "failure", [None, "venv", "install", "dependencies", "capture", "classification", "canary"]
)
def test_runner_isolates_intervals_and_continues_after_failures(
    failure: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    calls: list[tuple[list[str], dict[str, str]]] = []

    def execute(
        command: list[str], *, cwd: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess:
        assert cwd == ROOT
        calls.append((command, env))
        fail = (
            (failure == "venv" and command[1:3] == ["-m", "venv"])
            or (failure == "install" and "pip" in command)
            or (failure == "dependencies" and "--upgrade" in command)
            or (failure == "capture" and "scripts/capture_surface_inventory.py" in command)
            or (failure == "classification" and "scripts/export_surface_contract.py" in command)
            or (failure == "canary" and "pytest" in command)
        )
        # Both intervals fail when requested, so the final error must name both.
        return subprocess.CompletedProcess(command, 1 if fail else 0)

    # Avoid package downloads and inject process failures at the external boundary.
    monkeypatch.setattr(runner.subprocess, "run", execute)
    rows = [
        {"family": "openai", "interval": "floor", "packages": "openai==2.0.0"},
        {"family": "openai", "interval": "latest", "packages": "openai>=2.0"},
    ]
    output_dir = tmp_path / "reports"
    result = runner.run_intervals(rows, output_dir)
    output = capsys.readouterr().out

    assert result == (1 if failure else 0)
    assert output.count("::group::") == output.count("::endgroup::") == 2
    assert "openai/floor" in output
    assert "openai/latest" in output
    if failure:
        errors = [line for line in output.splitlines() if line.startswith("::error::Failed")]
        assert len(errors) == 1
        assert "openai/floor" in errors[0] and "openai/latest" in errors[0]

    venvs = [command[-1] for command, _ in calls if command[1:3] == ["-m", "venv"]]
    assert len(venvs) == len(set(venvs)) == 2
    checks = [command for command, _ in calls if "scripts/export_surface_contract.py" in command]
    canaries = [(command, env) for command, env in calls if "pytest" in command]
    assert len(checks) == len(canaries) == 2
    for row, venv, check, (canary, env) in zip(rows, venvs, checks, canaries, strict=True):
        python = str(Path(venv) / "bin/python")
        reports = output_dir / row["family"] / row["interval"]
        assert check == [
            python,
            "scripts/export_surface_contract.py",
            "--check",
            "--output",
            str(reports / "surface_contract/surface-classification.json"),
            "--reports-dir",
            str(reports / "inventory"),
        ]
        assert canary == [
            python,
            "-m",
            "pytest",
            "tests/unit/test_surface_canary.py",
            "-v",
            "--tb=short",
        ]
        assert env["SOLWYN_SURFACE_CANARY_FAMILY"] == row["family"]
        assert env["AWS_EC2_METADATA_DISABLED"] == "true"
        if failure != "venv":
            assert ([python, "-m", "pip", "install", "-e", "."], env) in calls
        if failure not in {"venv", "install"}:
            assert (
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
                env,
            ) in calls
        if failure not in {"venv", "install", "dependencies"}:
            assert (
                [
                    python,
                    "scripts/capture_surface_inventory.py",
                    "--check",
                    "--interval",
                    row["interval"],
                    "--family",
                    row["family"],
                    "--output-dir",
                    str(reports / "inventory"),
                ],
                env,
            ) in calls


def test_unavailable_interval_python_does_not_abort_remaining_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _runner()

    def unavailable(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("interpreter unavailable")

    monkeypatch.setattr(runner.subprocess, "run", unavailable)
    rows = [
        {"family": "openai", "interval": "floor", "packages": "openai==2.0.0"},
        {"family": "anthropic", "interval": "floor", "packages": "anthropic==0.50.0"},
    ]
    assert runner.run_intervals(rows, tmp_path) == 1
    output = capsys.readouterr().out
    assert "openai/floor" in output and "anthropic/floor" in output
    assert output.count("::endgroup::") == 2
