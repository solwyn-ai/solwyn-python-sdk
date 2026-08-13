"""Tests for the payload rule-diff tool."""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import zlib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]


def _diff_module() -> ModuleType:
    path = ROOT / "scripts" / "diff_surface_rules.py"
    spec = importlib.util.spec_from_file_location("diff_surface_rules", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_with_rows(rows: list[list[object]]) -> str:
    payload = {"rules": rows, "schema_version": 1}
    encoded = base64.b85encode(
        zlib.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    ).decode("ascii")
    chunks = "".join(f'    "{encoded[i : i + 100]}"\n' for i in range(0, len(encoded), 100))
    return (
        "x = 1\n# BEGIN GENERATED SURFACE RULE PAYLOAD\n"
        f"_GENERATED_SURFACE_RULE_PAYLOAD = (\n{chunks})\n"
        "# END GENERATED SURFACE RULE PAYLOAD\n"
    )


@pytest.mark.unit
def test_diff_classifies_added_removed_and_changed_rows() -> None:
    # Arrange
    diff = _diff_module()
    old = _source_with_rows([["rule.a", "a"], ["rule.b", "b"], ["rule.c", "c"]])
    new = _source_with_rows([["rule.a", "a"], ["rule.b", "B2"], ["rule.d", "d"]])

    # Act
    removed, added, changed = diff.classify(diff.payload_rows(old), diff.payload_rows(new))

    # Assert
    assert removed == ("rule.c",)
    assert added == ("rule.d",)
    assert changed == ("rule.b",)


@pytest.mark.unit
def test_payload_rows_reads_the_committed_payload() -> None:
    # Arrange
    diff = _diff_module()
    source = (ROOT / "src" / "solwyn" / "_surfaces.py").read_text(encoding="utf-8")

    # Act
    rows = diff.payload_rows(source)

    # Assert
    assert len(rows) > 5000
    assert all(rule_id.startswith("surface.") for rule_id in rows)


@pytest.mark.unit
def test_source_at_rejects_option_shaped_ref_without_writing(tmp_path: Path) -> None:
    # Arrange
    diff = _diff_module()
    output_prefix = tmp_path / "leaked"
    leaked_target = tmp_path / f"leaked:{diff.SOURCE}"
    leaked_target.parent.mkdir(parents=True)
    error = None

    # Act
    try:
        diff._source_at(f"--output={output_prefix}")
    except subprocess.CalledProcessError as exc:
        error = exc

    # Assert
    assert error is not None
    assert not leaked_target.exists()


@pytest.mark.unit
def test_payload_rows_rejects_duplicate_rule_ids() -> None:
    # Arrange
    diff = _diff_module()
    source = _source_with_rows([["rule.a", "first"], ["rule.a", "second"]])

    # Act / Assert
    with pytest.raises(RuntimeError, match=r"duplicate.*rule\.a"):
        diff.payload_rows(source)
