"""Keep the documented provider versions tied to the tested dependency set."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.unit
def test_readme_supported_provider_versions_match_extras_intervals_and_latest() -> None:
    # Arrange: the lock owns dev versions; inventory owns the isolated async lane.
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    intervals = json.loads(
        (ROOT / "tests/provider_surface_intervals.json").read_text(encoding="utf-8")
    )["include"]
    fingerprints = json.loads(
        (ROOT / "tests/provider_surface_fingerprints.json").read_text(encoding="utf-8")
    )["fingerprints"]
    locked = {
        package["name"]: package["version"]
        for package in tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
        if "version" in package
    }
    extra_by_family = {
        "openai": "openai",
        "anthropic": "anthropic",
        "together": "together",
        "google-genai": "google",
        "google-generativeai": "dev",  # Legacy SDK; bring your own, no install extra.
        "bedrock": "bedrock",
    }
    expected = {}
    for family in {row["family"] for row in intervals}:
        package = "boto3" if family == "bedrock" else family
        floor = tuple(
            requirement
            for requirement in extras[extra_by_family[family]]
            if requirement.startswith(f"{package}>=")
        )
        assert len(floor) == 1, family
        breakpoints = tuple(
            row["packages"]
            for row in intervals
            if row["family"] == family and row["interval"] != "latest"
        )
        latest = (f"{package}=={locked[package]}",)
        inventory_package = "aioboto3" if family == "bedrock" else package
        inventory_versions = {
            distribution["version"]
            for row in fingerprints
            if row["structural_interval"] == "latest"
            for distribution in row["distributions"]
            if distribution["name"] == inventory_package
        }
        assert len(inventory_versions) == 1, family
        inventory_version = inventory_versions.pop()
        if family == "bedrock":
            latest += (f"aioboto3=={inventory_version}",)
        else:
            assert inventory_version == locked[package], family
        expected[family] = (floor, breakpoints, latest)

    # Act: parse the actual table, preserving duplicates and every cell value.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = re.search(
        r"^### Supported provider SDK versions\n(.*?)(?=^#{1,3} |\Z)",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section is not None, "README must document supported provider SDK versions"
    documented = {}
    for line in section.group(1).splitlines():
        if not line.startswith("| `"):
            continue
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        assert len(cells) == 4, line
        family = cells[0].strip("`")
        assert family not in documented, f"duplicate provider row: {family}"
        documented[family] = tuple(tuple(re.findall(r"`([^`]+)`", cell)) for cell in cells[1:])

    # Assert: missing/extra providers, floor drift, and stale latest pins all fail.
    assert documented == expected
