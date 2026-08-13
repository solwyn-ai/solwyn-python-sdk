---
title: Provider Surface Inventory Artifact Design
type: refactor
date: 2026-08-10
artifact_contract: ce-design/v1
status: approved
---

# Provider Surface Inventory Artifact Design

## Context

Coverage-strict U0 observes real provider SDK client surfaces at supported dependency floors, known structural breakpoints, and the latest compatible release. The first implementation committed one pretty-printed raw JSON document for every client-shape and interval pair. That produced 78 files, roughly 21 MB, and more than 700,000 added lines. The corpus is generated evidence, not product source, and its repeated full SDK trees make ordinary code review impractical.

The approved replacement preserves the drift gate while moving bulky evidence out of Git.

## Decision

Git stores one compact fingerprint manifest. Each row identifies a client shape and structural interval and records:

- provider, client shape, mode, variant, and structural interval;
- the installed provider distribution versions observed when the row was accepted;
- counts for namespaces, observations, and service-model operations;
- a SHA-256 digest of canonical structural data.

The structural digest excludes installed distribution versions. A new provider release with an unchanged public surface therefore remains green, while any added, removed, or shape-changed surface fails the gate. Accepted version metadata remains available for audit but is informational during structural comparison.

Full inventory reports are generated into an ignored artifact directory. CI uploads that directory for every provider/interval matrix job, including failed checks. Reviewers can inspect or download the exact normalized reports that produced a fingerprint mismatch without making those reports permanent repository history.

## Data Flow

1. The interval catalog selects a provider family and dependency interval.
2. CI installs that interval's provider packages.
3. The capture script constructs genuine clients with fake credentials under the offline guard.
4. The script writes complete deterministic reports to the configured artifact directory.
5. The script derives canonical structural fingerprints and compares them with the committed manifest.
6. CI uploads the complete reports with a short retention period and preserves a failing exit status when fingerprints differ.

Local refresh uses the same capture path. It updates the compact fingerprint manifest and leaves complete reports in the ignored artifact directory for inspection.

## Canonical Fingerprint

The digest input contains only fields that describe the public pre-call graph:

- schema version;
- provider, client shape, mode, variant, shape key, and structural interval;
- ordered namespace paths;
- ordered observation rows with path, descriptor category, return shape, and source;
- ordered Bedrock service-model operations where applicable.

Canonical JSON uses sorted keys and compact separators. Distribution names and versions are excluded from the digest so version-only upgrades do not create false drift.

## Error Handling

- Capture or offline-guard failures remain hard failures and do not update the accepted manifest.
- Check mode writes full reports before comparing fingerprints so failed CI runs retain evidence.
- Missing, duplicate, unexpected, or structurally changed fingerprint rows fail with exact shape and interval identifiers.
- Refresh mode replaces only rows selected by the current family/interval invocation and preserves unrelated accepted rows, allowing matrix slices and local targeted refreshes.
- Artifact-upload failure is visible but must not convert a fingerprint failure into success.

## Repository Changes

- Delete `tests/fixtures/provider_surface_inventory/*.json`.
- Add a compact `tests/provider_surface_fingerprints.json` manifest.
- Ignore the local full-report artifact directory.
- Update the capture script, Make targets, tests, CI workflow, and the coverage-strict implementation plan to use fingerprints plus generated artifacts.
- Keep `tests/provider_surface_intervals.json` as the source of matrix coverage.

## Testing

- Unit-test canonical digest stability and distribution-version exclusion.
- Unit-test manifest refresh and exact mismatch reporting with temporary directories.
- Prove check mode writes complete reports even when a fingerprint differs.
- Prove the compact manifest covers every catalog interval and mandatory client shape.
- Keep genuine-SDK offline, determinism, mandatory-capability, and least-privilege workflow checks.
- Run the targeted surface tests, full unit suite, lint, format, and type checks before delivery.

## Non-Goals

- Git LFS or compressed raw snapshots. Both retain an opaque, review-hostile artifact as repository state.
- Weakening mandatory provider-shape or structural-interval coverage.
- Classifying provider capabilities in U0; policy classification remains U1.
- Changing Solwyn runtime behavior or Cloud API contracts.
