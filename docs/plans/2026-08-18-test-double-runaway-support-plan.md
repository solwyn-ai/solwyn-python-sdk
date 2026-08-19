# Test-Double Runaway-Protection Support: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `main`'s runaway-protection stack (#66, #70, #73) into the `test-double` branch and extend `FakeControlPlane` so run-control directives, denial receipts, and aggregate-replay behavior are simulatable, recordable, and contract-checked — with no live server.

**Architecture:** Task 1 lands the merge and re-threads the branch's transport seam through every new `httpx.Client` construction site main added (the branch's own seam-conformance grep test is the done-signal). Tasks 2–3 extend the sans-I/O `_plane.py` verdict machine: a run-scoped `stop_run()` trigger emits `RunControlDirective` v1 on all three channels (check / lease grant / lease renew) gated on the request's `run_directive_version` opt-in — the exact template `failover_directive_version` already uses — and the ingest handler gains receipt recording plus scriptable rejection bodies. Task 4 extends the shared contract pack (both lanes: double + live). Task 5 adds chaos-marked game-day recipes and docs.

**Tech Stack:** Python ≥3.11, httpx, Pydantic v2 — unchanged; no new dependencies.

**Specs this plan argues from:**
- `docs/plans/2026-08-07-test-double-plan.md` (the shipped double; tenets carried forward) — in the main checkout at `~/dev/repos/solwyn-ai/solwyn-python-sdk/docs/plans/`
- `docs/plans/2026-08-07-runaway-protection-plan.md` (the merged feature's design) — same location
- The merged code itself on `main` @ `e991a91` is authoritative where it and the plan docs disagree.

## Global Constraints (carried from the test-double plan; all still binding)

- **No pricing, ever.** Denials and receipts are scripted/recorded, never priced.
- **Real wire contract only.** Every response built through vendored `solwyn._types` models, serialized `model_dump(mode="json", exclude_none=...)`; every request parsed through the vendored request models (`extra="forbid"`).
- **Privacy invariant unchanged.** `solwyn.testing` stays OFF the content-privileged allowlist; no privacy banner; no `print(`; receipts are content-free by construction (bounded structural labels only).
- `RuntimeError`, never `assert`, for runtime invariants (AST test + ruff S101).
- Pydantic v2 idioms; mypy strict over `src/solwyn/`.
- Scenario windows are request-counted, never wall-clock; all state instance-scoped.
- Precedence order unchanged: transport-level → endpoint refusals → verdict triggers → default allow. **New tier placement:** the `stop_run` verdict sits with the verdict triggers, but a stop beats `deny_next` for the same request (a server kill is authoritative over an ordinary deny — mirrors `budget.py` treating a matched directive as authoritative even over `allowed: true`).
- Every task ends green on `make check && make test`.
- Pre-launch: breaking changes acceptable.

## Background: what merged and what the gap is

`main` @ `e991a91` contains three PRs the merge-base (`1ea7eb0`, #65) lacks:

- **#66** — agent-framework wrappers; renamed wrapper internals into the `_solwyn_` namespace (`_config` → `_solwyn_config`, etc.). New exports `RunHandle`, `create_run`, `start_run`.
- **#70** — client-side velocity detection (`_velocity.py`), process-global run-termination registry (`_run_control.py`), 7 new env-mapped `velocity_*` config knobs, stream abort hooks.
- **#73** — the wire contract this plan simulates:
  - `run_directive_version: Literal["1"] | None` on `BudgetCheckRequest`, `LeaseGrantRequest`, `LeaseRenewRequest` (SDK always sends `"1"`).
  - `RunControlDirective{version: "1", action: "terminate", agent_run_id, reason≤64}` as optional `run_control` on `BudgetCheckResponse` **and** `LeaseGrantResponse` (grant + renew share the model).
  - Denial-receipt fields on `MetadataEvent` (ride `/metadata/ingest`): `deny_source` (7-value `DenySource` literal), `deny_reason≤64`, `denied_by_period≤32`, `estimated_output_bound≤100M`, `velocity_flags` (≤8 `VelocityFlag`), `receipt_aggregate_count 1..100M`, `receipt_pricing_input_tokens≤100M` (validator: requires `receipt_aggregate_count`).
  - Aggregate replay: dropped denial receipts fold by pricing-compatible identity and replay later as fresh `BUDGET_DENIED` events with `deny_source="aggregate_replay"`, summed quantities, unioned flags, and 100M-unit fragment splitting.
  - Index-aware ingest rejection parsing: `{"rejected": [{"index", "code", "model", "message"}]}` with EXACT / LEGACY / MALFORMED disposition tiers.
  - **Breaking:** `RunStoppedError` no longer subclasses `BudgetExceededError`; it derives from `SolwynError` with ctor `(agent_run_id, reason, source)`.

**The double today** (this branch) has zero `run_control` / receipt / velocity support. What it does have: the `run_stopped` denial period, `solwyn-test/deny-stopped` and `solwyn-test/runaway` magic models, and the `failover_directive_version` opt-in gating pattern — the direct precedents everything below extends. **No new endpoints arrived**, so the unmatched-endpoint tripwire stays quiet; the gap is field-level and behavioral.

---

### Task 1: Merge `main` into `test-double`; re-thread the transport seam

**Files:**
- Merge: `git merge main` in the `test-double` worktree (conflicts: `CHANGELOG.md`, `src/solwyn/_lifecycle.py`, `src/solwyn/client.py`, `src/solwyn/reporter.py`, `tests/unit/test_public_exports.py`)
- Modify post-merge: `src/solwyn/budget.py`, `src/solwyn/reporter.py`, `src/solwyn/_lifecycle.py` (route every new construction site through the seam factories), `src/solwyn/testing/_plane.py` (semantic fixes only, see Step 4)
- Test: existing `tests/unit/test_transport_seam.py` conformance grep is the acceptance test — do not weaken it

**Interfaces:**
- Consumes: the branch's existing seam — stored `self._transport`; factories `_new_http_client(timeout)`, `_new_async_http_client(timeout)`, `_new_exit_http_client()` on all four components; `control_plane_transport=` on `Solwyn`/`AsyncSolwyn`.
- Produces: a merged tree where `make check && make test` is green and *every* `httpx.Client(`/`httpx.AsyncClient(` in `src/solwyn/` lives inside a factory. Tasks 2–5 build on this tree and on the now-present vendored models (`RunControlDirective`, `DenySource`, `VelocityFlag`, receipt fields).

**Per-file resolution strategy:**

| File | Strategy |
|---|---|
| `CHANGELOG.md` | Keep both sides' entries under the unreleased heading; branch (testing) entries after main's runaway entries. |
| `reporter.py` | Take main's logic (fold state, `_build_receipt_replay_event`, index-aware rejection parsing) wholesale; re-apply the branch's seam: `transport:` kwarg, stored `_transport`, factories; convert main's four construction sites (sync/async init, fork-repair, exit) to factory calls. |
| `_lifecycle.py` | Take main's exit flush (receipt_fold_state, `_dispose_exit_event`, final aggregate attempt); the exit clients (surrender + flush) must come from the owning component's `_new_exit_http_client()`, as the branch already does — port that pattern onto main's new code, not the other way around. |
| `client.py` | Take main's velocity gates, receipt threading, and #66 `_solwyn_` renames; re-apply the branch's `control_plane_transport=` kwarg and thread it to enforcer + reporter through the renamed internals. |
| `test_public_exports.py` | Union: main's `RunTermination`/`run_termination`/`current_run_terminated`/`clear_run_termination`/`RunHandle`/`create_run`/`start_run` + the branch's `solwyn.testing` export pins. |

- [ ] **Step 1: Start the merge and resolve the five conflicts per the table**

```bash
cd /Users/christian/dev/repos/solwyn-ai/solwyn-python-sdk/.worktrees/test-double
git merge main   # do NOT commit until Step 5
```

- [ ] **Step 2: Run the seam conformance test; expect it to fail on main's new bare construction sites**

Run: `uv run pytest tests/unit/test_transport_seam.py -v`
Expected: FAIL listing bare `httpx.Client(`/`AsyncClient(` occurrences outside the factories (main added fork-repair, exit, and fold-drain sites in `budget.py`, `reporter.py`, `_lifecycle.py`).

- [ ] **Step 3: Route every flagged site through the component factories**

Pattern (identical to the branch's existing seam commits):

```python
# was (main):            httpx.Client(timeout=5.0)
# becomes:               self._new_http_client()
# was (main, surrender): httpx.Client(timeout=_SURRENDER_TIMEOUT_S)
# becomes:               self._new_http_client(timeout=_SURRENDER_TIMEOUT_S)
# was (main, exit paths in _lifecycle): httpx.Client(timeout=5.0)
# becomes:               <owning component>._new_exit_http_client()
```

Re-run the conformance test until it passes.

- [ ] **Step 4: Fix the semantic (non-textual) conflicts**

1. `RunStoppedError` is no longer a `BudgetExceededError`. Grep `tests/unit/testing_double/` and `src/solwyn/testing/` for `BudgetExceededError` used where the deny period is `run_stopped` (the `solwyn-test/deny-stopped` tests, gameday recipes, contract pack callers) and switch those expectations to `RunStoppedError` where the SDK now raises it.
2. `_TestingSolwyn`/`_TestingAsyncSolwyn` in `_plane.py` override wrapper hooks (`_intercepted_call` / `_media_call`). #66 renamed wrapper internals into the `_solwyn_` namespace — verify the overridden hook names still exist on the merged wrapper classes and rename the overrides if they moved.
3. The new autouse fixture `_reset_process_run_control` (tests/unit/conftest.py) now isolates the process-global stop registry for every unit test, including `tests/unit/testing_double/` — no action needed, but confirm it survived the merge; the double's run-scoped tests depend on it.
4. Hermetic wrap: `_hermetic_config_options` enumerates `SolwynConfig.model_fields` dynamically, so the 7 new `velocity_*` knobs are pinned automatically. Add one assertion to the existing hermetic-wrap test: exporting `SOLWYN_VELOCITY_MODE=deny` in the test env must NOT change a wrapped client's velocity mode.

- [ ] **Step 5: Full gate, then commit the merge**

```bash
make check && make test
git add -A && git commit   # keep the default merge message; append a body line:
# "Re-threads the control-plane transport seam through the runaway stack's
#  new client construction sites (fork-repair, exit, fold-drain)."
```

### Task 2: Run-control directive surface on `FakeControlPlane`

**Files:**
- Modify: `src/solwyn/testing/_plane.py` (triggers, verdict chain, check/grant/renew handlers, `MAGIC_MODELS`)
- Modify: `src/solwyn/testing/__init__.py` (no new exports — triggers are methods)
- Test: `tests/unit/testing_double/test_plane_run_control.py` (new)

**Interfaces:**
- Consumes: post-merge vendored models `RunControlDirective`, and `run_directive_version` on the three request models; the plane's existing `_evaluate_chain`, `_handle_check`, `_handle_lease_grant`, `_handle_lease_renew`, `_ScenarioWindow`.
- Produces (Tasks 4–5 rely on these exact names):
  - `FakeControlPlane.stop_run(agent_run_id: str, *, reason: str = "manual_kill") -> None` — from the next request onward, any check/grant/renew naming that run is denied `denied_by_period="run_stopped"`, with a `run_control` v1 terminate directive attached iff the request opted in (`run_directive_version == "1"`). Echoes the *raw* requested run id. Beats `deny_next` for the same request. Reason is first-writer-wins per run (a second `stop_run` for an already-stopped run raises `RuntimeError` — scripting two reasons for one run is a test bug, fail loud).
  - `FakeControlPlane.clear_stop(agent_run_id: str) -> None` — subsequent requests for that run get ordinary verdicts (the SDK's epoch-fencing decides client-side whether its own sticky clears).
  - `FakeControlPlane.misroute_stops(*, requests: int | None = None) -> AbstractContextManager[None]` — while active, directives (and the run_stopped denial's directive id) are emitted with agent_run_id `"solwyn-test-misrouted-run"` instead of the requested id: deterministic contract-drift simulation for the `_MisroutedControlDenial` breaker-success path.
  - Magic model `solwyn-test/kill` in `MAGIC_MODELS`: first check per `(plane, agent_run_id)` allows; every later check/grant/renew for that run behaves exactly as if `stop_run(run_id)` had been called (reason `"manual_kill"`). Run-scoped: raises `RuntimeError` outside a run context, same as `solwyn-test/runaway`.
  - Recording: `stopped_runs: dict[str, str]` (run id → reason) exposed read-only; cleared by `reset_recording()`? **No** — stops are scenario state, not traffic; `reset_recording()` must NOT clear them (add a test pinning that).

**Behavioral fidelity notes for the implementer (from the merged SDK, verify against `budget.py` post-merge):**
- Response shape for a stopped run on **check**: `allowed=False, reservation_id=None (omitted), mode="hard_deny", denied_by_period="run_stopped"`, plus `run_control` iff opted in. Serialization stays conditional exclude-none per the existing check rules.
- On **grant/renew**: the stopped-lease shape — lease block omitted entirely, `allowed=False, denied_by_period="run_stopped", mode="hard_deny"`, plus `run_control` iff opted in (grant and renew responses share `LeaseGrantResponse`; renew's opt-in field is on `LeaseRenewRequest`).
- A request *without* `agent_run_id` never receives a directive (there is no run to guard).
- Do **not** simulate `allowed: true` + directive — core never sends it; the SDK's normalization of that pathological shape is covered by main's unit tests with mocks.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/unit/testing_double/test_plane_run_control.py"""
import httpx
import pytest

from solwyn.testing import FakeControlPlane

# reuse this module's local helpers from test_plane_check.py's pattern:
# _check_payload(**overrides) builds a valid BudgetCheckRequest dict,
# _auth(plane) builds the Authorization header.


@pytest.mark.unit
def test_stop_run_attaches_v1_directive_to_opted_in_check() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        body = http.post(
            "/api/v1/budgets/check",
            json=_check_payload(agent_run_id="run-7", run_directive_version="1"),
            headers=_auth(plane),
        ).json()
    assert body["allowed"] is False
    assert body["denied_by_period"] == "run_stopped"
    assert body["run_control"] == {
        "version": "1",
        "action": "terminate",
        "agent_run_id": "run-7",
        "reason": "manual_kill",
    }
    assert "reservation_id" not in body


@pytest.mark.unit
def test_stop_run_without_opt_in_denies_but_omits_directive() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        body = http.post(
            "/api/v1/budgets/check",
            json=_check_payload(agent_run_id="run-7", run_directive_version=None),
            headers=_auth(plane),
        ).json()
    assert body["allowed"] is False
    assert body["denied_by_period"] == "run_stopped"
    assert "run_control" not in body


@pytest.mark.unit
def test_stop_beats_deny_next_and_other_runs_stay_allowed() -> None:
    plane = FakeControlPlane()
    plane.deny_next(5, period="monthly")
    plane.stop_run("run-7")
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        stopped = http.post(
            "/api/v1/budgets/check",
            json=_check_payload(agent_run_id="run-7", run_directive_version="1"),
            headers=_auth(plane),
        ).json()
        other = http.post(
            "/api/v1/budgets/check",
            json=_check_payload(agent_run_id="run-8", run_directive_version="1"),
            headers=_auth(plane),
        ).json()
    assert stopped["denied_by_period"] == "run_stopped"
    assert other["denied_by_period"] == "monthly"  # deny_next still consumed
    assert "run_control" not in other


@pytest.mark.unit
def test_second_stop_run_with_different_reason_raises() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7", reason="manual_kill")
    with pytest.raises(RuntimeError):
        plane.stop_run("run-7", reason="velocity:repeat_size")


@pytest.mark.unit
def test_clear_stop_restores_ordinary_allows() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")
    plane.clear_stop("run-7")
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        body = http.post(
            "/api/v1/budgets/check",
            json=_check_payload(agent_run_id="run-7", run_directive_version="1"),
            headers=_auth(plane),
        ).json()
    assert body["allowed"] is True
    assert "run_control" not in body
```

Add the enforcer-level and lease-channel tests in the same file:
- a real `BudgetEnforcer` on `plane.transport`: after `stop_run`, a check for that run returns a denied result and marks the SDK's run registry (`solwyn.current_run_terminated()` truthy inside the run context) — this proves the double's bytes drive the real directive machinery end to end;
- a granted lease for `run-7`, then `stop_run("run-7")`: the next renew response carries the stopped shape + directive, the SDK drops the lease, and `plane.lease_surrenders` records the surrender;
- `misroute_stops()`: with a real enforcer, the misrouted directive must NOT open the control-plane breaker (assert breaker still closed after 3 misrouted responses) and must NOT mark the registry;
- `solwyn-test/kill` via `plane.wrap(...)` + the run context: first call succeeds, second raises `RunStoppedError` (NOT `BudgetExceededError`), and `plane.stopped_runs` records the run.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/unit/testing_double/test_plane_run_control.py -v` → `AttributeError: stop_run` / unknown magic model.

- [ ] **Step 3: Implement**

In `_plane.py`:
- state: `self._stopped_runs: dict[str, str]` under the existing lock; `self._kill_seen_runs: set[str]` for the magic model (mirror `runaway`'s per-run memory).
- `stop_run` / `clear_stop` / `stopped_runs` property per the Interfaces block.
- verdict chain: in `_evaluate_chain`, before the `deny_next` consultation, resolve the request's `agent_run_id`; if in `_stopped_runs`, produce the run-stopped verdict carrying `(run_id, reason)`.
- handlers: `_handle_check` builds `RunControlDirective(version="1", action="terminate", agent_run_id=<raw or misrouted>, reason=<reason>)` and sets `run_control` only when the parsed request's `run_directive_version == "1"`; same in `_handle_lease_grant` / `_handle_lease_renew` on the stopped shape. `exclude_none` serialization needs no change.
- `misroute_stops`: a new `_ScenarioWindow.kind` consumed at directive-build time (verdict tier — it rewrites the directive id, nothing else).
- `MAGIC_MODELS` gains `solwyn-test/kill`; `_evaluate_chain` branches beside `runaway`: first sighting of `(run)` records into `_kill_seen_runs` and allows; later sightings behave as `stop_run`.

- [ ] **Step 4: Run the new tests, then the full gate** — `uv run pytest tests/unit/testing_double/ -v && make check && make test`.

- [ ] **Step 5: Commit**

```bash
git add src/solwyn/testing tests/unit/testing_double/test_plane_run_control.py
git commit -m "feat(testing): run-control directive surface on FakeControlPlane"
```

### Task 3: Denial receipts and scriptable ingest rejections

**Files:**
- Modify: `src/solwyn/testing/_plane.py` (`_handle_ingest`, recording accessors, `reject_ingest` window)
- Test: `tests/unit/testing_double/test_plane_receipts.py` (new)

**Interfaces:**
- Consumes: post-merge `MetadataEvent` receipt fields (parsing is already model-driven, so validation of `deny_source` / bounds / the pricing-basis-requires-aggregate validator comes free — a receipt the models reject already 422s).
- Produces (Tasks 4–5 rely on these exact names):
  - `FakeControlPlane.denial_receipts` (property) → `list[MetadataEvent]`: recorded ingest events with `deny_source is not None`, in arrival order.
  - `FakeControlPlane.aggregate_replays` (property) → `list[MetadataEvent]`: the subset with `deny_source == "aggregate_replay"`.
  - `FakeControlPlane.reject_ingest(*, indices: Sequence[int] | None = None, code: str = "invalid_tags", count: int | None = None, malformed: bool = False, requests: int | None = 1) -> AbstractContextManager[None]` — scripts the next `requests` ingest responses:
    - `indices=[...]` → EXACT mode: 202 with one full `{"index", "code", "model", "message"}` entry per index (model/message echoed from the batch entry at that index).
    - `count=n` (no indices) → LEGACY mode: 202 with `n` rejection entries carrying `code`/`model`/`message` but **no** `index` key.
    - `malformed=True` → 202 with `"rejected"` set to a non-list (`{"rejected": "corrupt"}`) — the reporter's MALFORMED fail-open path.
    - Exactly one of `indices` / `count` / `malformed` must be given; otherwise `RuntimeError`.
    - Rejected events are still **recorded** (`plane.ingested` keeps them) — the scripting shapes the response body only; a test asserts what the SDK *did* about the rejection, not what the plane discarded.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/unit/testing_double/test_plane_receipts.py"""
import httpx
import pytest

from solwyn.testing import FakeControlPlane

# _receipt_event(**overrides): local helper building a valid BUDGET_DENIED
# MetadataEvent dict (fresh uuid4 call_id, deny_source="server",
# deny_reason="manual_kill", denied_by_period="run_stopped",
# estimated_output_bound=4096, velocity_flags=[]).


@pytest.mark.unit
def test_denial_receipt_is_accepted_and_recorded() -> None:
    plane = FakeControlPlane()
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        resp = http.post(
            "/api/v1/metadata/ingest",
            json=[_receipt_event()],
            headers=_auth(plane),
        )
    assert resp.status_code == 202
    assert resp.json() == {"ingested": 1, "rejected": []}
    (receipt,) = plane.denial_receipts
    assert receipt.deny_source == "server"
    assert receipt.denied_by_period == "run_stopped"


@pytest.mark.unit
def test_aggregate_replay_receipt_with_null_pricing_basis_is_accepted() -> None:
    plane = FakeControlPlane()
    event = _receipt_event(
        deny_source="aggregate_replay",
        receipt_aggregate_count=17,
        input_tokens=123_456,
        # COARSE aggregate: no receipt_pricing_input_tokens key at all
    )
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        resp = http.post(
            "/api/v1/metadata/ingest", json=[event], headers=_auth(plane)
        )
    assert resp.status_code == 202
    (replay,) = plane.aggregate_replays
    assert replay.receipt_aggregate_count == 17
    assert replay.receipt_pricing_input_tokens is None


@pytest.mark.unit
def test_pricing_basis_without_aggregate_count_is_rejected() -> None:
    plane = FakeControlPlane()
    event = _receipt_event(receipt_pricing_input_tokens=500)  # no aggregate count
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        resp = http.post(
            "/api/v1/metadata/ingest", json=[event], headers=_auth(plane)
        )
    assert resp.status_code == 422  # vendored-model validator, mirrored server-side


@pytest.mark.unit
def test_reject_ingest_exact_mode_names_indices() -> None:
    plane = FakeControlPlane()
    batch = [_receipt_event(), _receipt_event(), _receipt_event()]
    with plane.reject_ingest(indices=[1], code="invalid_tags"):
        with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
            body = http.post(
                "/api/v1/metadata/ingest", json=batch, headers=_auth(plane)
            ).json()
    assert body["ingested"] == 2
    (entry,) = body["rejected"]
    assert entry["index"] == 1 and entry["code"] == "invalid_tags"
    assert set(entry) == {"index", "code", "model", "message"}


@pytest.mark.unit
def test_reject_ingest_legacy_mode_omits_index() -> None:
    plane = FakeControlPlane()
    with plane.reject_ingest(count=1):
        with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
            body = http.post(
                "/api/v1/metadata/ingest",
                json=[_receipt_event(), _receipt_event()],
                headers=_auth(plane),
            ).json()
    (entry,) = body["rejected"]
    assert "index" not in entry
```

Add a reporter-level test in the same file: a real `MetadataReporter` on `plane.transport` whose denied event meets `reject_ingest(indices=[0], ...)` must fold and, after a subsequent clean delivery cycle, replay — the replayed event lands in `plane.aggregate_replays` with `receipt_aggregate_count >= 1` and a fresh `call_id`. (Drive the cycle with the reporter's public flush/close API; consult `tests/unit/test_reporter_receipt_folding.py` from the merge for the cycle-success mechanics.)

- [ ] **Step 2: Run to verify failure** — `AttributeError: denial_receipts` / `reject_ingest`.

- [ ] **Step 3: Implement** — properties filter `self.ingested` under the lock; `reject_ingest` is a new `_ScenarioWindow.kind` consumed inside `_handle_ingest` after parsing (so recording and dedup still happen), shaping only the 202 body; argument exclusivity enforced with `RuntimeError`.

- [ ] **Step 4: Run + gate** — `uv run pytest tests/unit/testing_double/ -v && make check && make test`.

- [ ] **Step 5: Commit**

```bash
git add src/solwyn/testing tests/unit/testing_double/test_plane_receipts.py
git commit -m "feat(testing): denial-receipt recording and scriptable ingest rejections"
```

### Task 4: Contract pack extension (both lanes)

**Files:**
- Modify: `src/solwyn/testing/contract.py`
- Modify: `tests/unit/testing_double/test_contract_against_double.py`
- Modify: `tests/integration/test_contract_against_live.py`

**Interfaces:**
- Consumes: Task 2's directive emission; Task 3's receipt acceptance; the live lane's existing `StoppableRun` helper (`tests/integration/test_live_contract.py`) for stopping a real run.
- Produces:
  - `contract.assert_run_control_contract(http: httpx.Client, api_key: str, *, stopped_run_id: str) -> None` — caller stops the run first (double lane: `plane.stop_run(run_id)`; live lane: `StoppableRun`'s stop POST), then the pack asserts: opted-in check → `allowed is False`, `denied_by_period == "run_stopped"`, `run_control == {"version": "1", "action": "terminate", "agent_run_id": stopped_run_id, "reason": <non-empty str ≤64>}`; non-opted-in check → same denial, **no** `run_control` key; a check for a different run id → no `run_control` key.
  - `contract.assert_receipt_ingest_contract(http: httpx.Client, api_key: str) -> None` — a fully-populated denial receipt and a coarse aggregate replay (aggregate count, summed tokens, no pricing basis) each → `202 {"ingested": 1, "rejected": []}`.
  - The stop-side effect itself (dashboard JWT POST) stays out of the pack — it is server-state manipulation, injected by each lane, exactly like the pack's existing lane-specific setup.

- [ ] **Step 1: Write the two assertion functions in `contract.py`** following the existing functions' style (raw `httpx` requests, exact-key assertions, no SDK client involvement — the pack tests wire bytes, not SDK behavior).

- [ ] **Step 2: Wire the double lane** — `test_contract_against_double.py` gains two tests calling the new functions over `FakeControlPlane().transport` (stop via `plane.stop_run`). Run: `uv run pytest tests/unit/testing_double/test_contract_against_double.py -v` → PASS.

- [ ] **Step 3: Wire the live lane** — `test_contract_against_live.py` gains the same two calls; the run-control one provisions and stops a run via the `StoppableRun` pattern and skips (like its siblings) when the live API is absent. The receipt test refuses non-loopback control planes, matching main's new live receipt test.

- [ ] **Step 4: Gate + commit**

```bash
make check && make test
git add src/solwyn/testing/contract.py tests/unit/testing_double tests/integration
git commit -m "test(sdk): extend the shared contract pack with run-control and receipt assertions"
```

### Task 5: Game-day recipes, docs, changelog

**Files:**
- Modify: `tests/unit/testing_double/test_gameday_recipes.py`
- Modify: `README.md` ("Testing your budget enforcement" section), `tests/CLAUDE.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: Tasks 2–4 public API.
- Produces: four new `chaos`-marked, CI-runnable recipes; docs that cannot rot (every README snippet mirrored in a test, per the branch's existing convention).

- [ ] **Step 1: Write the four recipes** (each a full wrap-level scenario using `plane.wrap(...)`, the run context, and magic/programmatic triggers):

1. **Operator kill mid-run, durable through an outage:** allow one call → `plane.stop_run(run_id)` → next call raises `RunStoppedError` → `with plane.outage():` a further call still raises `RunStoppedError` (sticky replay, never fail-open) → assert the denial receipt in `plane.denial_receipts` has `deny_source in {"server", "sticky_replay"}` and `deny_reason == "manual_kill"`.
2. **Kill lands on the lease channel:** run under an active lease → `stop_run` → renewal carries the directive → assert `plane.lease_surrenders` grew and subsequent calls deny per-call with `denied_by_period == "run_stopped"`.
3. **Receipt loss folds and replays:** force a receipt drop with `plane.reject_ingest(...)` (or queue pressure), then a clean cycle → assert exactly one event in `plane.aggregate_replays` whose `input_tokens` equals the sum of the dropped receipts' and whose `velocity_flags` is the sorted union.
4. **Local velocity stop defers to nothing:** wrap with `velocity_mode="deny"`, `velocity_repeat_count=2`; two identical calls trip `repeat_size`; assert `RunStoppedError` with `deny_reason == "velocity:repeat_size"` receipt (`deny_source == "local_velocity"`), and that continued plane allows do NOT un-stop the run (server allows never clear a local stop).

- [ ] **Step 2: Run them** — `uv run pytest tests/unit/testing_double/test_gameday_recipes.py -m chaos -v` → PASS.

- [ ] **Step 3: Docs** — README gains a "Simulating an operator kill" recipe (stop_run + RunStoppedError, ~10 lines) and one sentence on `denial_receipts`; `tests/CLAUDE.md`'s double-vs-live section notes run-control/receipt parity now rides the contract pack; CHANGELOG entry under unreleased.

- [ ] **Step 4: Gate + commit**

```bash
make check && make test
git add README.md CHANGELOG.md tests/CLAUDE.md tests/unit/testing_double
git commit -m "test(sdk): runaway-protection game-day recipes; testing docs"
```

## Out of scope

- Simulating the SDK's client-side machinery (velocity detector, fold tables, epoch fencing) — the double simulates the *server*; the SDK's own machinery is the thing under test.
- The `allowed: true` + directive pathological shape (server never sends it; unit-mocked on main).
- Fragment-splitting (>100M) recipes — the split happens client-side and is unit-tested on main; the double only needs to accept the resulting ≤100M fragments, which Task 3's model-driven parsing already does.
- Any core/server change (none needed — core's run-stop endpoint and receipt ingest already shipped).
- Velocity config presets or a fake clock.

## Decisions flagged for review

1. **`stop_run` re-stop raises** instead of silently keeping the first reason (the SDK's first-writer-wins is about concurrent *server* races; in a scripted test, two reasons for one run is a mistake).
2. **`reset_recording()` does not clear stops** — stops are scenario state like `deny_next`, not recorded traffic; `clear_stop`/`clear_denials` are the reset channel.
3. **`misroute_stops` ships in v1** — contract-drift-in-a-can; it is the only way to CI-test the "drift must not open the fleet breaker" invariant deterministically.
