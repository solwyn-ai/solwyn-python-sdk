# Red surface canary / provider drift runbook

A `provider-surface-inventory` matrix job (or the Monday scheduled run) is red.
An unclassified provider surface is NOT a user emergency: at runtime it resolves
to `unknown` and follows `on_unmetered` (default `warn`). Red CI is the gate
doing its job.

## 1. Identify what drifted

- Open the failing job and download its artifact (`provider-surface-<family>-<interval>`).
- Read the failing step, in order of information:
  - `capture --check` → `fingerprint drift: <shape>@<interval>` — structure changed.
  - `export --check --reports-dir` → `no reviewed raw rule for '<path>' ...` — the actionable line: an exact new/changed path.
  - canary → `SurfaceCanaryError ... during shape_drift|unknown_classification` — a reviewed rule no longer matches reality.

## 2. Classify

- Pick the kind using the Initial Classification Policy table in
  `docs/plans/2026-08-07-coverage-strict-mode-plan.md` (blocked > unsupported >
  unmetered_spend > metadata/infrastructure; when unsure, `unmetered_spend`
  with an exact acknowledgment token).

## 3. Apply (the five-step loop)

1. `uv run python scripts/export_surface_contract.py`
2. Edit exact rows in `build/surface_contract/surface-classification.json`.
3. `uv run python scripts/embed_surface_rules.py --input build/surface_contract/surface-classification.json`
   (prints the rule delta; requires `--allow-removals` for deletions).
4. `uv run python scripts/export_surface_contract.py` (refreshes the expanded
   artifact's `source_payload_fingerprint` from the new embedded payload).
5. `make check-surface-contract && make check-provider-surfaces`

## 4. Refresh evidence + pins

- `make capture-provider-surfaces` (refreshes latest fingerprints only).
- Update the per-context digests in `tests/unit/test_surface_context_pins.py`
  and, if the OpenAI graph moved, the README's `OPENAI_STRICT_FINGERPRINT`
  (review the delta first — never paste blind).
- Run `make check && make test-unit && make check-surface-contract && make check-provider-surfaces`.
  Continue to the PR checklist only after it passes.

## 5. PR checklist

- Paste `uv run python scripts/diff_surface_rules.py <PR-base-ref>` output in
  the PR. Use the branch or ref the PR targets; use `origin/main` only when the
  PR actually targets main.
- One sentence per new rule saying why its kind is right.
- Never disable, skip, or `--ignore` the canary to unblock a release; the
  publish workflow already excludes it.
