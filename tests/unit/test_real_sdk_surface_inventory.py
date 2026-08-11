"""Mandatory real-SDK inventory tests for the offline surface observer."""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[2]
FINGERPRINT_PATH = ROOT / "tests" / "provider_surface_fingerprints.json"
INTERVAL_CATALOG = ROOT / "tests" / "provider_surface_intervals.json"

MANDATORY_SHAPES = {
    "anthropic_async",
    "anthropic_sync",
    "azure_openai_async",
    "azure_openai_sync",
    "bedrock_aioboto3_async",
    "bedrock_boto3_sync",
    "google_genai_async",
    "google_genai_sync",
    "google_generativeai_sync",
    "openai_compatible_async",
    "openai_compatible_sync",
    "openai_native_async",
    "openai_native_sync",
    "openai_together_async",
    "openai_together_sync",
    "together_native_async",
    "together_native_sync",
}
OPENAI_SHAPES = {
    "azure_openai_async",
    "azure_openai_sync",
    "openai_compatible_async",
    "openai_compatible_sync",
    "openai_native_async",
    "openai_native_sync",
    "openai_together_async",
    "openai_together_sync",
}


def _sample_report(*, distribution_version: str = "2.53.0") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "shape_key": "openai_native_sync",
        "client_shape": "openai_sdk",
        "provider": "openai",
        "mode": "sync",
        "variant": "native",
        "structural_interval": "latest",
        "distribution": {"name": "openai", "version": distribution_version},
        "distributions": [{"name": "openai", "version": distribution_version}],
        "socket_attempts": 0,
        "namespaces": ["responses"],
        "service_model_operations": [],
        "observations": [
            {
                "path": "responses",
                "descriptor_category": "cached_property",
                "return_shape": "resource",
                "source": "public_attribute",
            },
            {
                "path": "responses.create",
                "descriptor_category": "function",
                "return_shape": "callable",
                "source": "public_attribute",
            },
        ],
    }


def _capture_module() -> ModuleType:
    path = ROOT / "scripts" / "capture_surface_inventory.py"
    spec = importlib.util.spec_from_file_location("capture_surface_inventory", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load inventory capture script at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.unit
def test_shape_registry_covers_every_mandatory_real_sdk_client() -> None:
    capture = _capture_module()

    assert set(capture.shape_keys()) == MANDATORY_SHAPES
    assert capture.import_name_for_shape("google_genai_sync") == "google.genai"
    assert capture.import_name_for_shape("google_generativeai_sync") == "google.generativeai"


@pytest.mark.unit
def test_family_selection_expands_to_exact_client_variants() -> None:
    capture = _capture_module()

    assert set(capture.shape_keys_for_families({"openai"})) == OPENAI_SHAPES


@pytest.mark.unit
def test_surface_fingerprint_is_structural_and_version_independent() -> None:
    # Arrange
    capture = _capture_module()
    first_report = _sample_report(distribution_version="2.53.0")
    second_report = _sample_report(distribution_version="2.54.0")

    # Act
    first = capture.fingerprint_report(first_report)
    second = capture.fingerprint_report(second_report)

    # Assert
    assert first["structure_sha256"] == second["structure_sha256"]
    assert first["distributions"] != second["distributions"]
    assert first["observation_count"] == 2
    assert first["namespace_count"] == 1
    assert first["service_model_operation_count"] == 0


@pytest.mark.unit
def test_surface_fingerprint_changes_when_the_public_graph_changes() -> None:
    # Arrange
    capture = _capture_module()
    first_report = _sample_report()
    second_report = _sample_report()
    second_report["observations"][1]["return_shape"] = "resource"

    # Act
    first = capture.fingerprint_report(first_report)
    second = capture.fingerprint_report(second_report)

    # Assert
    assert first["structure_sha256"] != second["structure_sha256"]


@pytest.mark.unit
def test_surface_fingerprint_rejects_any_socket_attempt() -> None:
    # Arrange
    capture = _capture_module()
    report = _sample_report()
    report["socket_attempts"] = 1

    # Act / Assert
    with pytest.raises(RuntimeError, match="socket_attempts must be zero"):
        capture.fingerprint_report(report)


@pytest.mark.unit
def test_fingerprint_manifest_update_preserves_unrelated_rows(tmp_path: Path) -> None:
    # Arrange
    capture = _capture_module()
    fingerprint_path = tmp_path / "fingerprints.json"
    unrelated = _sample_report()
    unrelated["shape_key"] = "anthropic_sync"
    unrelated["client_shape"] = "anthropic_sdk"
    unrelated["provider"] = "anthropic"
    capture.update_fingerprint_manifest(fingerprint_path, [unrelated])
    updated = _sample_report(distribution_version="2.54.0")

    # Act
    capture.update_fingerprint_manifest(fingerprint_path, [updated])

    # Assert
    manifest = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert [row["shape_key"] for row in manifest["fingerprints"]] == [
        "anthropic_sync",
        "openai_native_sync",
    ]
    assert manifest["fingerprints"][1]["distributions"] == [{"name": "openai", "version": "2.54.0"}]


@pytest.mark.unit
def test_compare_fingerprints_reports_missing_and_structural_drift(tmp_path: Path) -> None:
    # Arrange
    capture = _capture_module()
    fingerprint_path = tmp_path / "fingerprints.json"
    expected = _sample_report()
    missing = _sample_report()
    missing["shape_key"] = "anthropic_sync"
    missing["client_shape"] = "anthropic_sdk"
    missing["provider"] = "anthropic"
    capture.update_fingerprint_manifest(fingerprint_path, [expected])
    drifted = _sample_report(distribution_version="2.54.0")
    drifted["observations"][1]["return_shape"] = "resource"

    # Act
    mismatches = capture.compare_fingerprints(
        fingerprint_path,
        [drifted, missing],
    )

    # Assert
    assert mismatches == (
        "missing fingerprint: anthropic_sync@latest",
        "fingerprint drift: openai_native_sync@latest",
    )


@pytest.mark.unit
def test_compare_fingerprints_can_require_every_manifest_report(tmp_path: Path) -> None:
    # Arrange
    capture = _capture_module()
    fingerprint_path = tmp_path / "fingerprints.json"
    present = _sample_report()
    missing = _sample_report()
    missing.update(
        shape_key="anthropic_sync",
        client_shape="anthropic_sdk",
        provider="anthropic",
    )
    capture.update_fingerprint_manifest(fingerprint_path, [present, missing])

    # Act
    mismatches = capture.compare_fingerprints(
        fingerprint_path,
        [present],
        require_all=True,
    )

    # Assert
    assert mismatches == ("missing report: anthropic_sync@latest",)


@pytest.mark.unit
def test_load_reports_rejects_duplicate_shape_interval_keys(tmp_path: Path) -> None:
    # Arrange
    capture = _capture_module()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = _sample_report()
    (reports_dir / "one.json").write_text(json.dumps(report), encoding="utf-8")
    (reports_dir / "two.json").write_text(json.dumps(report), encoding="utf-8")

    # Act / Assert
    with pytest.raises(RuntimeError, match="duplicate report: openai_native_sync@latest"):
        capture.load_reports(reports_dir)


@pytest.mark.unit
def test_select_reports_rejects_incomplete_family_interval_slice() -> None:
    # Arrange
    capture = _capture_module()
    report = _sample_report()

    # Act / Assert
    with pytest.raises(
        RuntimeError,
        match="missing report: openai_compatible_sync@latest",
    ):
        capture.select_reports(
            [report],
            selected={"openai_native_sync", "openai_compatible_sync"},
            structural_interval="latest",
        )


@pytest.mark.unit
def test_select_reports_requires_each_shape_in_every_imported_interval() -> None:
    # Arrange
    capture = _capture_module()
    latest_native = _sample_report()
    floor_compatible = _sample_report()
    floor_compatible.update(
        shape_key="openai_compatible_sync",
        provider="openai_compatible",
        variant="generic_compatible",
        structural_interval="floor",
    )

    # Act / Assert
    with pytest.raises(
        RuntimeError,
        match="missing report: openai_compatible_sync@latest",
    ):
        capture.select_reports(
            [latest_native, floor_compatible],
            selected={"openai_native_sync", "openai_compatible_sync"},
            structural_interval=None,
        )


@pytest.mark.unit
def test_select_reports_rejects_an_unfiltered_partial_catalog() -> None:
    # Arrange
    capture = _capture_module()

    # Act / Assert
    with pytest.raises(RuntimeError, match="missing report:"):
        capture.select_reports(
            [_sample_report()],
            selected=None,
            structural_interval=None,
        )


@pytest.mark.unit
def test_select_reports_accepts_an_interval_owned_by_one_family() -> None:
    # Arrange
    capture = _capture_module()
    reports = []
    for shape_key, mode in (("anthropic_async", "async"), ("anthropic_sync", "sync")):
        report = _sample_report(distribution_version="0.80.0")
        report.update(
            shape_key=shape_key,
            client_shape="anthropic_sdk",
            provider="anthropic",
            mode=mode,
            structural_interval="anthropic-parse",
            distribution={"name": "anthropic", "version": "0.80.0"},
            distributions=[{"name": "anthropic", "version": "0.80.0"}],
        )
        reports.append(report)

    # Act
    selected = capture.select_reports(
        reports,
        selected=None,
        structural_interval="anthropic-parse",
    )

    # Assert
    assert {_report["shape_key"] for _report in selected} == {
        "anthropic_async",
        "anthropic_sync",
    }


@pytest.mark.unit
def test_inventory_run_writes_full_artifact_before_reporting_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    capture = _capture_module()
    fingerprint_path = tmp_path / "fingerprints.json"
    output_dir = tmp_path / "artifacts"
    expected = _sample_report()
    capture.update_fingerprint_manifest(fingerprint_path, [expected])
    drifted = _sample_report(distribution_version="2.54.0")
    drifted["observations"][1]["return_shape"] = "resource"
    monkeypatch.setattr(capture, "capture_all", lambda **_kwargs: (drifted,))

    # Act
    result = capture.run_inventory(
        output_dir=output_dir,
        fingerprint_path=fingerprint_path,
        structural_interval="latest",
        selected={"openai_native_sync"},
        check=True,
    )

    # Assert
    artifact_path = output_dir / "openai_native_sync--latest.json"
    assert result.paths == (artifact_path,)
    assert result.mismatches == ("fingerprint drift: openai_native_sync@latest",)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == drifted


@pytest.mark.unit
def test_inventory_run_retains_reports_captured_before_a_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    capture = _capture_module()
    output_dir = tmp_path / "artifacts"

    async def capture_shape(spec: Any, structural_interval: str) -> dict[str, Any]:
        if spec.key == "anthropic_sync":
            raise RuntimeError("later shape failed")
        report = _sample_report()
        report.update(
            shape_key=spec.key,
            client_shape=spec.client_shape,
            provider=spec.provider,
            mode=spec.mode,
            variant=spec.variant,
            structural_interval=structural_interval,
        )
        return report

    monkeypatch.setattr(capture, "_capture_shape", capture_shape)

    # Act / Assert
    with pytest.raises(RuntimeError, match="later shape failed"):
        capture.run_inventory(
            output_dir=output_dir,
            fingerprint_path=tmp_path / "fingerprints.json",
            structural_interval="latest",
            selected={"anthropic_async", "anthropic_sync"},
            check=True,
        )
    assert (output_dir / "anthropic_async--latest.json").exists()


@pytest.mark.unit
def test_namespace_discovery_fails_instead_of_truncating_at_max_depth() -> None:
    # Arrange
    capture = _capture_module()

    class ProviderResource:
        __module__ = "openai.resources.synthetic"

        def __init__(self, child: object | None = None) -> None:
            self._child = child

        @property
        def nested(self) -> object | None:
            return self._child

    root = ProviderResource(ProviderResource(ProviderResource(ProviderResource())))

    # Act / Assert
    with pytest.raises(capture.SurfaceInspectionError) as caught:
        capture._discover_namespaces(root, max_depth=2)
    assert caught.value.path == "nested.nested.nested"
    assert caught.value.stage == "depth_exhaustion"


@pytest.mark.unit
def test_namespace_discovery_allows_a_terminal_at_max_depth() -> None:
    # Arrange
    capture = _capture_module()

    class ProviderResource:
        __module__ = "openai.resources.synthetic"

        def __init__(self, child: object | None = None) -> None:
            self._child = child

        @property
        def nested(self) -> object | None:
            return self._child

    root = ProviderResource(ProviderResource(ProviderResource()))

    # Act
    namespaces = capture._discover_namespaces(root, max_depth=2)

    # Assert
    assert namespaces == ("nested", "nested.nested")


@pytest.mark.unit
def test_offline_guard_denies_an_attempted_socket_connection() -> None:
    capture = _capture_module()

    with (
        capture._deny_socket_access() as counter,
        pytest.raises(capture.OfflineViolationError),
    ):
        socket.create_connection(("127.0.0.1", 9))

    assert counter.attempts == 1


@pytest.mark.unit
@pytest.mark.parametrize("shape_key", sorted(MANDATORY_SHAPES))
def test_real_sdk_inventory_is_offline_deterministic_and_content_free(
    shape_key: str,
) -> None:
    capture = _capture_module()
    pytest.importorskip(capture.import_name_for_shape(shape_key))

    first = capture.capture_all(structural_interval="latest", selected={shape_key})
    second = capture.capture_all(structural_interval="latest", selected={shape_key})

    assert second == first
    assert {item["shape_key"] for item in first} == {shape_key}
    for report in first:
        assert report["schema_version"] == 1
        assert report["structural_interval"] == "latest"
        assert report["distribution"]["name"]
        assert report["distribution"]["version"]
        assert report["client_shape"]
        assert report["mode"] in {"sync", "async"}
        assert report["provider"]
        assert report["socket_attempts"] == 0
        observations = report["observations"]
        assert observations == sorted(observations, key=lambda item: item["path"])
        assert len({item["path"] for item in observations}) == len(observations)
        assert all(
            set(item) == {"path", "descriptor_category", "return_shape", "source"}
            for item in observations
        )
        serialized = json.dumps(report, sort_keys=True)
        assert "sk-test" not in serialized
        assert "testing" not in serialized


@pytest.mark.unit
@pytest.mark.parametrize(
    ("shape_key", "expected_paths"),
    [
        (
            "openai_native_sync",
            {
                "responses.create",
                "audio.speech.create",
                "videos.create_and_poll",
                "post",
                "with_options",
            },
        ),
        (
            "anthropic_sync",
            {"messages.stream", "messages.parse", "messages.batches.create"},
        ),
        (
            "together_native_sync",
            {
                "rerank.create",
                "code_interpreter.execute",
                "evals.create",
                "videos.create",
            },
        ),
        (
            "google_genai_sync",
            {"models.generate_content", "aio.models.generate_content"},
        ),
        ("google_generativeai_sync", {"generate_content"}),
        ("bedrock_boto3_sync", {"invoke_model", "converse", "apply_guardrail"}),
        ("bedrock_aioboto3_async", {"invoke_model", "converse", "apply_guardrail"}),
    ],
)
def test_latest_inventories_include_mandatory_nested_capabilities(
    shape_key: str,
    expected_paths: set[str],
) -> None:
    capture = _capture_module()
    pytest.importorskip(capture.import_name_for_shape(shape_key))
    report = capture.capture_all(structural_interval="latest", selected={shape_key})[0]

    assert expected_paths <= {row["path"] for row in report["observations"]}


@pytest.mark.unit
@pytest.mark.parametrize(
    "shape_key",
    ["bedrock_boto3_sync", "bedrock_aioboto3_async"],
)
def test_bedrock_inventory_unions_public_and_service_model_operations(shape_key: str) -> None:
    capture = _capture_module()
    pytest.importorskip(capture.import_name_for_shape(shape_key))
    report = capture.capture_all(structural_interval="latest", selected={shape_key})[0]

    rows = {item["path"]: item for item in report["observations"]}
    assert rows["invoke_model"]["source"] == "public_attribute"
    assert rows["invoke_model"]["return_shape"] == "callable"
    assert report["service_model_operations"] == sorted(report["service_model_operations"])
    assert "invoke_model" in report["service_model_operations"]


@pytest.mark.unit
def test_committed_latest_fingerprints_match_the_installed_sdk_set() -> None:
    # Arrange
    capture = _capture_module()
    selected = {
        shape_key
        for shape_key in capture.shape_keys()
        if _module_available(capture.import_name_for_shape(shape_key))
    }
    bedrock_shapes = set(capture.shape_keys_for_families({"bedrock"}))
    if not bedrock_shapes <= selected:
        selected -= bedrock_shapes

    if not selected:
        pytest.skip("provider SDKs are not installed")

    reports = capture.capture_all(
        structural_interval="latest",
        selected=selected,
    )

    # Act
    mismatches = capture.compare_fingerprints(FINGERPRINT_PATH, reports)

    # Assert
    assert mismatches == ()


@pytest.mark.unit
def test_structural_interval_catalog_covers_floor_breakpoints_and_latest() -> None:
    catalog = json.loads(INTERVAL_CATALOG.read_text(encoding="utf-8"))
    rows = catalog["include"]
    by_family = {
        family: {row["interval"] for row in rows if row["family"] == family}
        for family in {
            "anthropic",
            "bedrock",
            "google-genai",
            "google-generativeai",
            "openai",
            "together",
        }
    }

    assert catalog["schema_version"] == 1
    assert all({"floor", "latest"} <= intervals for intervals in by_family.values())
    assert {
        "openai-videos",
        "openai-skills",
        "openai-admin",
        "openai-content-provenance",
    } <= by_family["openai"]
    assert "anthropic-parse" in by_family["anthropic"]
    assert {"google-file-search", "google-webhooks"} <= by_family["google-genai"]
    assert {"bedrock-async-invoke", "bedrock-bidirectional", "bedrock-count-tokens"} <= (
        by_family["bedrock"]
    )
    for row in rows:
        assert row["packages"]
        if row["family"] == "bedrock":
            assert row["packages"].startswith("aioboto3")
            assert "botocore" not in row["packages"]


@pytest.mark.unit
def test_fingerprint_manifest_covers_every_catalog_interval_and_client_shape() -> None:
    # Arrange
    capture = _capture_module()
    rows = json.loads(INTERVAL_CATALOG.read_text(encoding="utf-8"))["include"]
    manifest = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    fingerprint_rows = manifest["fingerprints"]
    expected_keys = {
        (shape_key, matrix_row["interval"])
        for matrix_row in rows
        for shape_key in capture.shape_keys_for_families({matrix_row["family"]})
    }

    # Act
    actual_keys = {(row["shape_key"], row["structural_interval"]) for row in fingerprint_rows}

    # Assert
    assert manifest["schema_version"] == 1
    assert actual_keys == expected_keys
    assert len(actual_keys) == len(fingerprint_rows)
    assert all(row["distributions"] for row in fingerprint_rows)
    assert all(row["namespace_count"] >= 0 for row in fingerprint_rows)
    assert all(row["observation_count"] > 0 for row in fingerprint_rows)
    assert all(len(row["structure_sha256"]) == 64 for row in fingerprint_rows)


@pytest.mark.unit
def test_provider_inventory_ci_is_least_privilege_and_catalog_driven() -> None:
    # Arrange
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow_data = yaml.safe_load(workflow)
    checkout_steps = [
        step
        for job in workflow_data["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    inventory_steps = workflow_data["jobs"]["provider-surface-inventory"]["steps"]
    upload_steps = [
        step
        for step in inventory_steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]

    # Act / Assert
    assert "provider-surface-matrix:" in workflow
    assert "provider-surface-inventory:" in workflow
    assert "tests/provider_surface_intervals.json" in workflow
    assert workflow_data["permissions"] == {"contents": "read"}
    assert checkout_steps
    assert all(step.get("with", {}).get("persist-credentials") is False for step in checkout_steps)
    assert "ruff check src/ tests/ scripts/" in workflow
    assert "ruff format --check src/ tests/ scripts/" in workflow
    assert "--output-dir build/provider_surface_inventory" in workflow
    assert "tests/fixtures/provider_surface_inventory" not in workflow
    assert upload_steps == [
        {
            "name": "Upload provider surface inventory",
            "if": "always()",
            "uses": "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
            "with": {
                "name": "provider-surface-${{ matrix.family }}-${{ matrix.interval }}",
                "path": "build/provider_surface_inventory",
                "if-no-files-found": "error",
                "retention-days": 7,
            },
        }
    ]

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "capture-provider-surfaces:" in makefile
    assert "check-provider-surfaces:" in makefile
    assert "ruff check src/ tests/ scripts/" in makefile
    assert "ruff format --check src/ tests/ scripts/" in makefile
    assert "uv run --extra dev --with 'aioboto3>=13.0'" in makefile
    assert "--output-dir build/provider_surface_inventory" in makefile
    assert "tests/fixtures/provider_surface_inventory" not in makefile
