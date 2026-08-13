"""Tests for deterministic embedded surface-contract regeneration."""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
import zlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from solwyn._surfaces import surface_contract_data

ROOT = Path(__file__).parents[2]


def _embed_module() -> ModuleType:
    path = ROOT / "scripts" / "embed_surface_rules.py"
    assert path.exists(), f"missing deterministic payload encoder: {path}"
    spec = importlib.util.spec_from_file_location("embed_surface_rules", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load surface payload encoder at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stamped(contract: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    stamped = dict(contract)
    stamped["source_payload_fingerprint"] = fingerprint
    return stamped


@pytest.mark.unit
def test_embed_rejects_a_stale_source_fingerprint(tmp_path: Path) -> None:
    # Arrange
    from solwyn._surfaces import payload_fingerprint

    contract = _stamped(surface_contract_data(), "sha256:" + "0" * 64)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text(
        "x = 1\n# BEGIN GENERATED SURFACE RULE PAYLOAD\nP = ()\n"
        "# END GENERATED SURFACE RULE PAYLOAD\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "embed_surface_rules.py"),
        "--input",
        str(contract_path),
        "--source",
        str(source),
    ]

    # Act
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # Assert
    assert result.returncode != 0
    assert "stale" in (result.stdout + result.stderr)

    # Arrange
    good = _stamped(surface_contract_data(), payload_fingerprint())
    contract_path.write_text(json.dumps(good), encoding="utf-8")

    # Act
    ok = subprocess.run(command, capture_output=True, text=True, check=False)

    # Assert
    assert ok.returncode == 0


@pytest.mark.unit
def test_embed_requires_allow_removals_when_rules_disappear() -> None:
    # Arrange
    embed = _embed_module()
    contract: dict[str, Any] = surface_contract_data()
    reduced = dict(contract)
    reduced["rules"] = contract["rules"][:-1]

    # Act
    removed, added, changed = embed.rule_delta(embed._validated_rules(reduced))

    # Assert
    assert len(removed) == 1
    assert added == ()
    assert changed == ()


@pytest.mark.unit
def test_embed_cli_requires_allow_removals_before_rewriting(tmp_path: Path) -> None:
    # Arrange
    from solwyn._surfaces import payload_fingerprint

    contract: dict[str, Any] = surface_contract_data()
    removed_ids = sorted(row["id"] for row in contract["rules"][-2:])
    reduced = dict(contract)
    reduced["rules"] = contract["rules"][:-2]
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(_stamped(reduced, payload_fingerprint())),
        encoding="utf-8",
    )
    source_path = tmp_path / "_surfaces.py"
    original_source = (
        "before\n"
        "# BEGIN GENERATED SURFACE RULE PAYLOAD\n"
        "_GENERATED_SURFACE_RULE_PAYLOAD = (\n"
        '    "old"\n'
        ")\n"
        "# END GENERATED SURFACE RULE PAYLOAD\n"
        "after\n"
    )
    source_path.write_text(original_source, encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "embed_surface_rules.py"),
        "--input",
        str(contract_path),
        "--source",
        str(source_path),
    ]
    expected_removed_lines = [f"removed: {rule_id}" for rule_id in removed_ids]

    # Act
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    source_after_refusal = source_path.read_text(encoding="utf-8")
    allowed = subprocess.run(
        [*command, "--allow-removals"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Assert
    assert refused.returncode != 0
    assert [
        line for line in refused.stdout.splitlines() if line.startswith("removed: ")
    ] == expected_removed_lines
    assert "pass --allow-removals" in refused.stdout
    assert source_after_refusal == original_source
    assert allowed.returncode == 0, allowed.stderr
    assert [
        line for line in allowed.stdout.splitlines() if line.startswith("removed: ")
    ] == expected_removed_lines
    assert source_path.read_text(encoding="utf-8") != original_source


@pytest.mark.unit
def test_validated_rules_rejects_an_id_that_lies_about_kind() -> None:
    # Arrange
    embed = _embed_module()
    contract: dict[str, Any] = surface_contract_data()
    row = dict(contract["rules"][0])
    segments = row["id"].split(".")
    segments[2] = "blocked" if segments[2] != "blocked" else "metadata"
    row["id"] = ".".join(segments)
    tampered = dict(contract)
    tampered["rules"] = [row] + contract["rules"][1:]

    # Act / Assert
    with pytest.raises(RuntimeError, match="does not encode its kind"):
        embed._validated_rules(tampered)


@pytest.mark.unit
def test_surface_payload_encoder_canonicalizes_equivalent_input_order() -> None:
    # Arrange
    embed = _embed_module()
    contract: dict[str, Any] = json.loads(json.dumps(surface_contract_data()))
    reordered: dict[str, Any] = json.loads(json.dumps(contract))
    reordered["rules"].reverse()
    for row in reordered["rules"]:
        row["selectors"].reverse()
        row["expected_attribute_shapes"].reverse()
    assert reordered["rules"] != contract["rules"]
    assert any(len(row["selectors"]) > 1 for row in reordered["rules"])
    assert any(len(row["expected_attribute_shapes"]) > 1 for row in reordered["rules"])

    # Act
    first = embed.encode_contract(contract)
    second = embed.encode_contract(reordered)
    decoded = json.loads(zlib.decompress(base64.b85decode(first.encode("ascii"))))

    # Assert
    assert second == first
    assert decoded == embed.compact_contract(contract)


@pytest.mark.unit
def test_surface_payload_encoder_rejects_tampered_derived_fields() -> None:
    # Arrange
    embed = _embed_module()
    contract: dict[str, Any] = json.loads(json.dumps(surface_contract_data()))
    original = contract["rules"][0]["dispatch_action"]
    contract["rules"][0]["dispatch_action"] = "refuse" if original != "refuse" else "return"

    # Act / Assert
    with pytest.raises(RuntimeError, match="does not match validated export"):
        embed.encode_contract(contract)


@pytest.mark.unit
def test_surface_payload_encoder_rejects_tampered_unknown_policy() -> None:
    # Arrange
    embed = _embed_module()
    contract: dict[str, Any] = json.loads(json.dumps(surface_contract_data()))
    contract["unknown_policy"]["acknowledgment"] = "wildcard"

    # Act / Assert
    with pytest.raises(RuntimeError, match="unknown policy is invalid"):
        embed.encode_contract(contract)


@pytest.mark.unit
def test_surface_payload_rewriter_changes_only_the_marked_block() -> None:
    # Arrange
    embed = _embed_module()
    source = (
        "before\n"
        "# BEGIN GENERATED SURFACE RULE PAYLOAD\n"
        "_GENERATED_SURFACE_RULE_PAYLOAD = (\n"
        '    "old"\n'
        ")\n"
        "# END GENERATED SURFACE RULE PAYLOAD\n"
        "after\n"
    )

    # Act
    rewritten = embed.rewrite_source(source, "new-payload")

    # Assert
    assert rewritten == (
        "before\n"
        "# BEGIN GENERATED SURFACE RULE PAYLOAD\n"
        "_GENERATED_SURFACE_RULE_PAYLOAD = (\n"
        '    "new-payload"\n'
        ")\n"
        "# END GENERATED SURFACE RULE PAYLOAD\n"
        "after\n"
    )


@pytest.mark.unit
def test_surface_payload_cli_generates_then_checks_temporary_source(tmp_path: Path) -> None:
    # Arrange
    from solwyn._surfaces import payload_fingerprint

    input_path = tmp_path / "surface-classification.json"
    source_path = tmp_path / "_surfaces.py"
    contract = _stamped(surface_contract_data(), payload_fingerprint())
    input_path.write_text(json.dumps(contract), encoding="utf-8")
    stale_source = (
        "before\n"
        "# BEGIN GENERATED SURFACE RULE PAYLOAD\n"
        '_GENERATED_SURFACE_RULE_PAYLOAD = "old"\n'
        "# END GENERATED SURFACE RULE PAYLOAD\n"
        "after\n"
    )
    source_path.write_text(stale_source, encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "embed_surface_rules.py"),
        "--input",
        str(input_path),
        "--source",
        str(source_path),
    ]

    # Act
    stale_check = subprocess.run(
        [*command, "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    source_after_stale_check = source_path.read_text(encoding="utf-8")
    generated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    synchronized_check = subprocess.run(
        [*command, "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Assert
    assert stale_check.returncode == 1, stale_check.stderr
    assert source_after_stale_check == stale_source
    assert generated.returncode == 0, generated.stderr
    assert synchronized_check.returncode == 0, synchronized_check.stderr
    assert '"old"' not in source_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_surface_payload_check_accepts_equivalent_json_from_alternate_zlib(
    tmp_path: Path,
) -> None:
    # Arrange
    from solwyn._surfaces import payload_fingerprint

    embed = _embed_module()
    contract = _stamped(surface_contract_data(), payload_fingerprint())
    canonical_json = json.dumps(
        embed.compact_contract(contract),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    alternate_encoded = base64.b85encode(zlib.compress(canonical_json, level=1)).decode("ascii")
    assert alternate_encoded != embed.encode_contract(contract)

    input_path = tmp_path / "surface-classification.json"
    input_path.write_text(json.dumps(contract), encoding="utf-8")
    source_path = tmp_path / "_surfaces.py"
    source = embed.rewrite_source(
        "before\n# BEGIN GENERATED SURFACE RULE PAYLOAD\nold\n"
        "# END GENERATED SURFACE RULE PAYLOAD\nafter\n",
        alternate_encoded,
    )
    source_path.write_text(source, encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "embed_surface_rules.py"),
        "--input",
        str(input_path),
        "--source",
        str(source_path),
        "--check",
    ]

    # Act
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    # Assert
    assert result.returncode == 0, result.stderr
    assert source_path.read_text(encoding="utf-8") == source


@pytest.mark.unit
def test_committed_surface_payload_decodes_to_the_canonical_compact_contract() -> None:
    # Arrange
    embed = _embed_module()
    source_path = ROOT / "src" / "solwyn" / "_surfaces.py"
    committed_source = source_path.read_text(encoding="utf-8")

    # Act
    decoded_payload = embed.decode_source_payload(committed_source)

    # Assert
    assert decoded_payload == embed.compact_contract(surface_contract_data())


@pytest.mark.unit
def test_surface_contract_make_gate_checks_embedded_payload() -> None:
    # Arrange
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("check-surface-contract:", 1)[1].split("\n\n", 1)[0]

    # Act / Assert
    assert "uv run python scripts/export_surface_contract.py --check" in target
    assert (
        "uv run python scripts/embed_surface_rules.py "
        "--input build/surface_contract/surface-classification.json --check"
    ) in target
