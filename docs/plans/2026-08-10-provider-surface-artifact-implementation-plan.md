# Reviewable Provider Surface Evidence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace 78 committed raw SDK surface inventories with a compact structural fingerprint manifest while preserving full downloadable CI evidence and exact drift failures.

**Architecture:** `scripts/capture_surface_inventory.py` continues to capture complete deterministic reports, but writes them to ignored `build/provider_surface_inventory/`. Pure helpers canonicalize each report into a version-independent structural digest and maintain `tests/provider_surface_fingerprints.json`. CI checks the compact manifest and uploads the complete reports with `actions/upload-artifact` even when the check fails.

**Tech Stack:** Python 3.11+, pytest, JSON/SHA-256, GitHub Actions, Make, existing real provider SDK test harness.

---

### Task 1: Define the compact fingerprint contract

**Files:**
- Modify: `tests/unit/test_real_sdk_surface_inventory.py`
- Modify: `scripts/capture_surface_inventory.py`

**Step 1: Write the failing digest tests**

Add tests built from a small synthetic report rather than a real SDK:

```python
@pytest.mark.unit
def test_surface_fingerprint_is_structural_and_version_independent() -> None:
    capture = _capture_module()
    report = _sample_report(distribution_version="2.53.0")

    first = capture.fingerprint_report(report)
    second = capture.fingerprint_report(_sample_report(distribution_version="2.54.0"))

    assert first["structure_sha256"] == second["structure_sha256"]
    assert first["distributions"] != second["distributions"]
    assert first["observation_count"] == 2
    assert first["namespace_count"] == 1
```

Add a second test that changes one observation's return shape and asserts the digest changes.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/unit/test_real_sdk_surface_inventory.py -k fingerprint -v
```

Expected: FAIL because `fingerprint_report` does not exist.

**Step 3: Implement the minimal canonical digest**

Add `hashlib` and these helpers:

```python
FINGERPRINT_SCHEMA_VERSION = 1


def _structural_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "shape_key": report["shape_key"],
        "client_shape": report["client_shape"],
        "provider": report["provider"],
        "mode": report["mode"],
        "variant": report["variant"],
        "structural_interval": report["structural_interval"],
        "namespaces": report["namespaces"],
        "service_model_operations": report["service_model_operations"],
        "observations": report["observations"],
    }


def fingerprint_report(report: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        _structural_payload(report), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "shape_key": report["shape_key"],
        "client_shape": report["client_shape"],
        "provider": report["provider"],
        "mode": report["mode"],
        "variant": report["variant"],
        "structural_interval": report["structural_interval"],
        "distributions": report["distributions"],
        "namespace_count": len(report["namespaces"]),
        "observation_count": len(report["observations"]),
        "service_model_operation_count": len(report["service_model_operations"]),
        "structure_sha256": hashlib.sha256(canonical).hexdigest(),
    }
```

**Step 4: Run the focused tests and verify GREEN**

Run the same focused command. Expected: PASS.

### Task 2: Maintain and compare a compact manifest

**Files:**
- Modify: `tests/unit/test_real_sdk_surface_inventory.py`
- Modify: `scripts/capture_surface_inventory.py`

**Step 1: Write failing manifest tests**

Add temporary-path tests proving:

- updating one interval replaces only matching `(shape_key, structural_interval)` rows;
- rows are sorted by shape and interval;
- comparison ignores distribution-version-only changes;
- comparison reports missing and structurally drifted rows exactly;
- loading duplicate keys raises `RuntimeError` rather than silently picking one.

The public helpers under test are:

```python
update_fingerprint_manifest(path: Path, reports: Collection[Mapping[str, Any]]) -> None
compare_fingerprints(path: Path, reports: Collection[Mapping[str, Any]]) -> tuple[str, ...]
load_reports(path: Path) -> tuple[dict[str, Any], ...]
```

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/unit/test_real_sdk_surface_inventory.py -k 'manifest or compare_fingerprints or load_reports' -v
```

Expected: FAIL because the manifest helpers do not exist.

**Step 3: Implement minimal manifest behavior**

Use this manifest shape:

```json
{
  "schema_version": 1,
  "fingerprints": []
}
```

Key rows by `(shape_key, structural_interval)`. Compare every structural field and digest while excluding only `distributions`. Preserve unrelated rows on refresh. Render with `indent=2`, `sort_keys=True`, and one trailing newline.

`load_reports` reads `*.json`, validates each top-level report is an object with the capture schema, and rejects duplicate keys.

**Step 4: Verify GREEN**

Run the same focused command. Expected: PASS.

### Task 3: Always write complete reports before checking

**Files:**
- Modify: `tests/unit/test_real_sdk_surface_inventory.py`
- Modify: `scripts/capture_surface_inventory.py`

**Step 1: Write failing artifact-flow tests**

Add tests that monkeypatch `capture_all` with synthetic reports and invoke a new orchestration helper:

```python
run_inventory(
    output_dir=tmp_path / "artifacts",
    fingerprint_path=tmp_path / "fingerprints.json",
    structural_interval="latest",
    selected={"openai_native_sync"},
    check=True,
)
```

Assert that a structurally mismatched check returns mismatch text and still writes the full report to `artifacts/openai_native_sync--latest.json`.

Add a reports-import test proving existing report files can refresh the compact manifest without provider SDK imports.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/unit/test_real_sdk_surface_inventory.py -k 'run_inventory or reports_import' -v
```

Expected: FAIL because the orchestration helper and CLI source option do not exist.

**Step 3: Implement report writing and CLI behavior**

Replace committed-fixture defaults with:

```python
ROOT = Path(__file__).parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "build" / "provider_surface_inventory"
DEFAULT_FINGERPRINT_PATH = ROOT / "tests" / "provider_surface_fingerprints.json"
```

Use stable artifact filenames: `<shape_key>--<structural_interval>.json`.

Add `--fingerprints PATH` and `--reports-dir PATH`. `--reports-dir` loads already-generated reports and skips capture. Without it, capture reports and write them to `--output-dir` before checking or refreshing. `--check` compares; default mode updates the manifest.

**Step 4: Verify GREEN**

Run the focused artifact-flow tests. Expected: PASS.

### Task 4: Change the repository contract and CI evidence flow

**Files:**
- Modify: `tests/unit/test_real_sdk_surface_inventory.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `Makefile`

**Step 1: Replace fixture-contract tests with failing fingerprint/CI tests**

Replace `test_committed_latest_fixtures_match_the_installed_sdk_set` and `test_every_catalog_interval_has_one_versioned_fixture_per_client_shape` with tests that assert:

- `tests/provider_surface_fingerprints.json` has exactly one row for every catalog-family shape and interval;
- every row has a 64-character lowercase SHA-256 digest and nonnegative counts;
- CI passes `--output-dir build/provider_surface_inventory`;
- CI uploads that directory using `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02` (`v4.6.2`) with `if: always()`, a matrix-qualified name, `if-no-files-found: error`, and bounded retention;
- Make refresh/check targets name fingerprints and leave full reports under `build/`.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/unit/test_real_sdk_surface_inventory.py -k 'catalog_interval or provider_inventory_ci or committed_fingerprints' -v
```

Expected: FAIL because the manifest and upload step do not exist.

**Step 3: Update workflow and Make targets**

After the check step, add:

```yaml
      - name: Upload provider surface inventory
        if: always()
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
        with:
          name: provider-surface-${{ matrix.family }}-${{ matrix.interval }}
          path: build/provider_surface_inventory
          if-no-files-found: error
          retention-days: 7
```

Pass `--output-dir build/provider_surface_inventory` to the check. Update Make descriptions from fixtures to fingerprints and use the same output directory.

**Step 4: Keep targeted tests RED until the manifest exists**

Do not weaken the exact catalog-coverage assertion. The next task creates the manifest from the existing raw reports.

### Task 5: Import fingerprints, then delete raw JSON

**Files:**
- Create: `tests/provider_surface_fingerprints.json`
- Delete: `tests/fixtures/provider_surface_inventory/*.json`

**Step 1: Build the compact manifest from the currently committed reports**

Run:

```bash
uv run python scripts/capture_surface_inventory.py \
  --reports-dir tests/fixtures/provider_surface_inventory \
  --fingerprints tests/provider_surface_fingerprints.json
```

Expected: one compact manifest containing all 78 accepted shape/interval rows.

**Step 2: Inspect the size and coverage before deletion**

Run:

```bash
wc -l tests/provider_surface_fingerprints.json
du -h tests/provider_surface_fingerprints.json
uv run pytest tests/unit/test_real_sdk_surface_inventory.py -k 'committed_fingerprints or catalog_interval' -v
```

Expected: the exact coverage tests PASS and the manifest is far smaller than the 21 MB raw corpus.

**Step 3: Remove the raw reports from Git**

Delete exactly the tracked JSON files under `tests/fixtures/provider_surface_inventory/`. The deletion is recoverable from commit `b5d4a6c` and must not touch other fixture directories.

**Step 4: Prove no repository code expects committed raw reports**

Run:

```bash
rg -n "tests/fixtures/provider_surface_inventory|compare_fixtures|write_fixtures|FIXTURE_DIR" .
```

Expected: no runtime/test references; documentation may mention the deleted path only as migration history.

### Task 6: Align the coverage-strict plan

**Files:**
- Modify: `docs/plans/2026-08-07-coverage-strict-mode-plan.md`

**Step 1: Update U0 and downstream references**

Replace requirements for committed raw fixtures with:

- compact committed structural fingerprints;
- full local/CI artifact reports;
- U1 consumption through freshly generated or downloaded reports;
- drift checks that ignore version-only changes but fail on graph changes.

Update U0 files, approach, test scenarios, verification, U1 inputs, U4 reuse language, risk mitigations, verification contract, and definition of done consistently.

**Step 2: Search for stale architecture claims**

Run:

```bash
rg -n "committed fixtures|persist.*fixture|U0.*fixtures|version-labelled fixtures" docs/plans/2026-08-07-coverage-strict-mode-plan.md
```

Expected: no stale requirement to commit full raw reports.

### Task 7: Verify, review, commit, and push

**Files:**
- All files changed by Tasks 1-6

**Step 1: Run targeted tests**

```bash
uv run pytest tests/unit/test_real_sdk_surface_inventory.py tests/unit/test_surface_observer.py -v
```

Expected: PASS.

**Step 2: Run the local latest fingerprint check**

```bash
make check-provider-surfaces
```

Expected: PASS and complete reports written only under ignored `build/provider_surface_inventory/`.

**Step 3: Run full repository verification**

```bash
make test
make check
```

Expected: all unit tests, Ruff lint/format, and Mypy pass.

**Step 4: Review the final PR diff**

Review against base `3dbe93ed6eb4bd9ef48db82adea94bc318566d65`. Confirm the generated corpus is gone, the remaining diff is human-reviewable, and no actionable findings remain.

**Step 5: Commit named files and push**

Create one implementation commit whose subject communicates the outcome, then push `codex/coverage-strict-02-inventories` to `origin`.
