# Runaway Protection (Detect → Terminate → Prove) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute one PR-stage per branch; each stage is independently shippable and reviewable.

**Goal:** Ship the flagship runaway-protection program — a content-free velocity engine that flags retry storms and agent ping-pong (Detect), a server-pushed `terminate_run` directive that stops a flagged run within one call or one lease renewal (Terminate), and denial receipts that make every deny durably attributable so the dashboard can show "what the cap saved you" (Prove).

**Architecture:** Three deliverables riding machinery that already shipped. Detection is a new sans-I/O `_velocity.py` module fed per-call from the client's interception path (lengths, counts, timestamps, run ids only — never content). Termination is a new versioned `run_control` directive on the existing budget-check and lease-grant/renew responses (same pattern as `FailoverDirective`), enforced through the existing run-scoped sticky-denial cache plus a new process-wide run-control registry that powers a cooperative stop flag and mid-stream abort. Receipts enrich the already-emitted `BUDGET_DENIED` MetadataEvents with deny-source attribution and ride the existing reporter, with a fold-on-drop aggregate so outage-window denials still leave a durable trace.

**Tech Stack:** Python 3.12+ (`StrEnum`, `Literal`), Pydantic v2 (`ConfigDict(extra="forbid")`, `@model_validator`, wrap serializers), httpx, pytest. Core API: FastAPI + Postgres + Redis (see Part 4). Dashboard: React (see Part 4).

## Global Constraints (binding — every task inherits these)

- **Privacy invariant:** never capture, log, or transmit prompt/response content. Detection signals are lengths, counts, timestamps, and identifiers ONLY. No content fingerprinting in v1. CI-enforced by `tests/unit/test_privacy_firewall.py` (path-based scan of all non-allowlisted modules — new modules are covered automatically).
- **The API owns all pricing.** The SDK never computes cost. "Avoided spend" is priced server-side from estimated token counts.
- Business logic sans-I/O; client classes are thin I/O wrappers. New sans-I/O modules follow the `_lease.py` precedent (no locks, no I/O, caller serializes).
- Pydantic v2 only; wire models use `extra="forbid"`; conditionally-omitted response fields carry explicit `None` defaults (directive-v1 responses serialize exclude-none server-side).
- Runtime invariants use `raise RuntimeError(...)`, never `assert` (enforced by `tests/unit/test_no_production_asserts.py`).
- PJ-2 lease design is decided and binding (`~/dev/repos/solwyn-ai/audits/solwyn-budget-lease-design.html`). This plan rides the renewal channel; it does not redesign it. Core tenet holds: **Solwyn unreachability never blocks a customer call** — only customer-chosen verdicts (and now an explicit human/server kill) stop anything.
- Wire-contract changes deploy **API first**, SDK second (repo convention; see `Modality` field note at `src/solwyn/_types.py:122-127`).
- Every wire-model field change updates the frozen field-set pins in `tests/unit/test_contract_snapshot.py` (e.g. `EXPECTED_CHECK_RESPONSE_FIELDS`, `EXPECTED_METADATA_FIELDS`, asserted at `tests/unit/test_contract_snapshot.py:237-318`).
- Pre-launch, zero customers: breaking changes are fine; optimize for the right design.
- Quality gate per PR: `make check` (lint + format + typecheck) and `make test` green; `make test-integration` for stages touching live contracts (needs core API at localhost:8080).

---

## Part 0 — Verified starting points (inherited claims vs. what the repo actually says)

Every claim from the project brief was re-verified against the repo. Two needed correction; the corrections change the receipts design.

| # | Claim (inherited) | Verdict | Evidence |
|---|---|---|---|
| 1 | Check requests already carry model, provider, estimated tokens, tags, run id | **Confirmed** | `_build_check_request` returns `BudgetCheckRequest(estimated_input_tokens=…, model=…, provider=…, agent_run_id=…, tags=…)` — `src/solwyn/budget.py:263-274`; wire model `src/solwyn/_types.py:391-470` |
| 2 | Server-directive pattern already works | **Confirmed** | `FailoverDirective` (`src/solwyn/_types.py:382-388`), applied per-call under `_breaker_lock` by `_apply_failover_tuning_directive` (`src/solwyn/_base.py:620-677`); client opts in via `failover_directive_version="1"` (`src/solwyn/budget.py:273`) |
| 3 | Sticky per-run denial replay exists | **Confirmed** | Run-scoped sticky store: `_cache_response` (`src/solwyn/budget.py:328-336`, `_run_hard_deny_responses` OrderedDict, LRU-bounded at `_MAX_STICKY_RUN_DENIALS=128`, `budget.py:82`); replay on unreachable: `_build_prior_hard_deny_unavailable_result` (`budget.py:343-375`); lease-path exclusion of sticky-denied runs: `_lease_path_applies` (`budget.py:542-556`) |
| 4 | "The hard-deny path currently emits nothing durable" | **CORRECTED** | Every deny surfaced to a caller ALREADY emits a `CallStatus.BUDGET_DENIED` MetadataEvent: sync chat `src/solwyn/client.py:1285-1309`, sync media `client.py:1059-1089`, async twins `client.py:2158`, `client.py:2365`. What's actually missing: (a) the event is **source-blind** — no field says whether the deny was a live server verdict, a sticky replay, local enforcement, or lease exhaustion, and it carries no `denied_by_period`, no estimated output bound; (b) **delivery dies exactly when sticky replays happen** — sticky replays fire only when the control plane is unreachable (`budget.py:343-386`), and the event-ingest channel is deliberately not breaker-guarded with bounded retry then counted drop (`src/solwyn/reporter.py:1257-1296`, `reporter_max_send_attempts=5` at `src/solwyn/config.py:124`), so outage-window deny events are typically lost. Receipts = attribution + outage-durable delivery, not new emission. |
| 5 | `BudgetCheckResponse` has no termination field | **Confirmed** | `src/solwyn/_types.py:473-507` — fields end at `failover_directive`; nothing run-control shaped anywhere in the SDK (`grep -r "terminate\|runaway" src/` is empty) |
| 6 | Settlement rides `reporter.report_settlement(confirm, event)` off-thread, at-least-once | **Confirmed** | `src/solwyn/reporter.py:1318-1334` (sync), `reporter.py:1497` (async); confirms are breaker-guarded and HELD (never dropped) on breaker-open (`reporter.py:1210-1255`) — a delivery posture receipts will borrow in PR-6 |
| 7 | (memory) run-cap denials come from a server-side `runaway_run` budget rule | **Confirmed in core repo** | `denied_by_period="agent_run"` is emitted when a `RunawayRunConfig` rule arms a per-run counter (`core: api/src/solwyn_api/services/budget_aggregator.py:2189-2217`; rule model `api/src/solwyn_api/schemas/budget_rules.py:138-165`; configured via JWT-only `PUT /api/v1/projects/{id}/budget`, `api/src/solwyn_api/routers/budgets.py:1815-1835`). The SDK already maps that value to RUN-scoped sticky state (`budget.py:328-336`). **Consequence: termination denials reuse `denied_by_period="agent_run"` and the sticky machinery needs zero changes.** Two core caveats (Part 4): the label has a second producer (`scoped_budget` with `scope="agent_run"`, `budget_aggregator.py:2315`), and the run cap only arms on `HARD_DENY` projects (`budget_aggregator.py:2189`) — a kill directive must be enforced at the router edge so it also works for `alert_only` projects. Note: team memory's `budgets.py:643` pointer for the check serializer is stale — the real site is `budgets.py:1166-1203`. |
| 8 | Lease renewal cadence supports the "one renewal" guarantee | **Confirmed** | Renew-ahead at ~75% depletion or `refresh_interval_s` (seconds-order, server-jittered; `lease_length_s` minutes-order) per the binding lease spec; renewal fires off-thread from admission (`budget.py:1523-1530`, `budget.py:1606-1655`). Caveat found: renewals are triggered BY admissions — a run with a live lease making **no further calls** (e.g. one long stream) never renews, so a lone in-flight stream may not learn of termination until its next call. Documented as a bounded limitation in D2. |

Additional load-bearing facts verified for this plan:

- The four deny paths all converge on `BudgetCheckResult(allowed=False)` and the client's single `if not budget.allowed:` gate — so receipt attribution is a plumbing change on `BudgetCheckResult` (an internal model, `budget.py:115-139`, NOT a wire model), threaded into the existing event builders.
- A lease-channel authoritative deny already feeds the unchanged sticky machinery via `_lease_deny_result` (`budget.py:766-781`) — the exact precedent for feeding a `terminate_run` directive into stickiness.
- Renewal responses are processed off the caller's thread with close-epoch fencing (`_finish_renewal`, `budget.py:1025-1068`) — directive processing must ride the same fence.
- Stream wrappers settle exactly-once with whatever usage was observed on `close()` (`src/solwyn/stream.py:117-128`, `stream.py:237-251`) — mid-stream abort can reuse this settle-then-raise machinery verbatim.
- `MetadataEvent` and `BudgetCheckRequest` wire shapes are frozen by field-set pins in `tests/unit/test_contract_snapshot.py:237-318`; live server drift is covered by `tests/integration/test_live_contract.py`.
- Uncounted-mode telemetry (`_note_uncounted_admission`, `budget.py:891-922`) is the house pattern for loud-on-entry, rate-limited-while-persisting warnings (1/30s) — velocity warnings copy it.

---

## Part 1 — Settled design decisions

### D1. Detection: which rules, whose thresholds, what posture

**Decision: detection runs client-side, in a new sans-I/O `VelocityMonitor`, fed at call-interception time. Three rules ship in v1, warn-first by default, with local deny as opt-in. Thresholds are client-configured (SolwynConfig + env vars) in v1; server-side rule config is explicitly deferred.**

Evidence-driven reasoning:

1. **The server cannot see per-call cadence for exactly the runs that matter.** Under PJ-2, run-scoped token traffic takes the lease path (`budget.py:1296-1323`): calls draw down a local grant and the server hears only aggregate renewals every `refresh_interval_s`. A retry storm inside a leased run is invisible to server-side per-call rules. The client sees every call. (The server still gets every MetadataEvent within ~`reporter_flush_interval=5.0`s — good enough for dashboards and human kill decisions, not for first-line detection.)
2. **Privacy is the moat.** Competitors detecting loops server-side must ingest prompts to find "near-identical calls." Solwyn detects on the client over estimated sizes (already length-derived ints — `estimate_tokens_from_length`, fed from `estimate_content_length`, `client.py:1243-1248`), timestamps, and run ids, and transmits only verdicts. `_velocity.py` stays OUTSIDE the content-privileged allowlist by construction; the path-based firewall test covers it automatically.
3. **Zero added latency.** A sans-I/O ring-buffer update per call is O(window) with tiny constants; no wire round-trip.

**Rules v1** (names are wire-stable identifiers; they appear in logs, `velocity_flags`, and receipts):

| Rule | Fires when | Default thresholds | May terminate in `deny` mode? |
|---|---|---|---|
| `repeat_size` (retry storm) | ≥ N calls in the trailing W seconds, same run + same model, each estimated input size within ±max(8 tokens, 2%) of the current call's size | N=5, W=60s | **Yes** |
| `monotonic_growth` (agent ping-pong) | Last K calls in the run each strictly larger than the previous, latest ≥ G× the first of the streak, median inter-call gap < 30s | K=8, G=3.0 | **Yes** |
| `rate_acceleration` (burst anomaly) | Calls in last 60s ≥ F absolute AND ≥ A× the prior 60s window | F=30, A=3.0 | **No — warn-only always** (fan-out false-positive risk; see Risks) |

**False-positive posture:** `velocity_mode` config = `"off" | "warn" | "deny"`, default `"warn"`. Warn tier: one WARNING log per (run, rule) per 30s (copies the `_UNCOUNTED_WARN_INTERVAL_S` pattern, `budget.py:87`), plus `velocity_flags` on the call's MetadataEvent (PR-4) so dashboards see flagged runs. Deny tier (opt-in): `repeat_size`/`monotonic_growth` trips terminate the run locally via the run-control registry → `RunTerminatedError`. `rate_acceleration` never denies regardless of mode. Warn-first-then-deny escalation across the fleet is a server policy decision made on ingested flags (human or core-side policy) — pushed back via the `terminate_run` directive, keeping the kill authoritative and auditable server-side.

**Who configures:** v1 = client (`SolwynConfig` fields + `SOLWYN_*` env vars; defaults above). Server-pushed rule tuning is deferred but the seam is reserved: the `run_control` directive is versioned exactly so a v2 can carry rule config the way `FailoverDirective` carries failover tuning (`_base.py:620-677` precedent). Deferred because it needs dashboard rule-editing UI and adds nothing to the demo; client defaults + server kill button cover the launch story.

**Scope:** detection is run-scoped only (`agent_run_id` present, i.e. inside `solwyn.run()`). Unscoped traffic has no loop identity to key on; the `_auto-{sdk_instance_id}` server-side synthesis is not visible client-side. Media calls feed the same per-run window (a retrying image loop is still a loop).

### D2. Termination: wire shape, enforcement, streaming, caller experience

**Decision: a versioned `run_control` directive object on `BudgetCheckResponse` AND `LeaseGrantResponse` (grant + renew share the model), enforced by (a) feeding the existing run-scoped sticky machinery unchanged, (b) a new process-wide run-control registry powering a cooperative stop flag, and (c) a between-chunks abort seam in the stream wrappers. Callers get BOTH a typed `RunTerminatedError` and a cooperative flag API.**

Exact wire shape: see Part 2. Key semantics:

- **Directive ≠ just a deny.** A terminate response carries BOTH `allowed=false, mode=hard_deny, denied_by_period="agent_run"` (so the UNCHANGED sticky machinery at `budget.py:328-336` run-scopes it, and outage replay at `budget.py:343-375` preserves it) AND the `run_control` directive (which adds the reason, the typed exception, the stop-flag trigger for in-flight streams/loops, and a versioned seam for future actions). Reusing `denied_by_period="agent_run"` is deliberate: it is the value the server's existing `runaway_run` rule already emits, and it avoids the latent global-cache-poisoning path (an unknown period value falls through to the GLOBAL `_last_hard_deny_response` at `budget.py:338-340`, which would deny all traffic — exactly what a run kill must not do).
- **Termination overrides `alert_only`.** A dashboard kill is an explicit human action, not budget policy; the server sends `mode=hard_deny` on terminate responses even for alert-only projects. (SDK needs no special casing — it honors the response mode.)
- **Processing points (SDK):** (1) `check_budget` sync/async after `BudgetCheckResponse.model_validate` (`budget.py:1415`, `budget.py:1997`); (2) grant install `_install_grant` (`budget.py:1070-1107`); (3) renewal completion `_finish_renewal` (`budget.py:1025-1068`) — all before/with sticky-feeding, under the existing close-epoch fences (a fenced-stale response's directive is discarded; the client is closing). On directive receipt: mark the run-control registry, feed a synthesized run-scoped hard deny (reuse the `_lease_deny_result` pattern, `budget.py:766-781`), and drop the run's lease locally (`_lease.drop_if_current`). Server-side, the kill flag makes the next renewal come back as an authoritative deny carrying the directive (Part 4, C-2) — the SDK's existing renewal-deny handling (`_finish_renewal` → `_lease_deny_result`, `budget.py:1066-1067`) plus the new directive processing covers it; a renewal 404/409 after any server-side lease cleanup is already handled defensively (`budget.py:739-747`).
- **Un-kill is server-owned:** a later live ALLOW for the run clears run sticky state (existing behavior, `budget.py:307-310`) and clears the registry's server-sourced entry. Locally-sourced terminations (velocity deny mode) are NOT cleared by server allows — the local policy owns them; `solwyn.clear_run_termination(run_id)` is the escape hatch.
- **Stop guarantee:** per-call path — the very next `check_budget` gets the live deny+directive; every later call denies via sticky without waiting on the server. Lease path — the next renewal (≤ `refresh_interval_s`, seconds-order, fired off-thread at ~75% depletion per the binding spec) delivers the directive; the next admission after that denies. Net: **≤ 1 call or ≤ 1 renewal**, as targeted. Documented limitation: a run whose only in-flight work is one long stream and which makes no further calls triggers no renewals (renewals are admission-driven, `budget.py:1523-1530`), so its stream aborts only if something else in-process learns of the kill; worst-case exposure is bounded by that call's own `max_tokens`.
- **Streaming abort contract:** wrappers get an `abort_check: Callable[[], Exception | None] | None` seam, evaluated at each iteration head — between chunks, before pulling the next chunk from the provider. On trip: `close()` first (settles the observed partial usage exactly-once through the existing settlement path — `stream.py:117-128` — and closes the inner stream, which for providers that honor close stops server-side generation: that is the actual money saved), then raise the exception the seam returned. Chunks already yielded stay with the caller; partially consumed streams settle honestly at observed usage.
- **Caller experience: both.** (1) `RunTerminatedError(SolwynError)` — deliberately NOT a `BudgetExceededError` subclass: agent-loop code that catches/retries budget denials must not swallow a kill. Carries `agent_run_id`, `reason` (bounded str: `"manual_kill"`, `"runaway_velocity"`, `"velocity:repeat_size"`, …), `source` (`"server" | "local_velocity"`). Raised pre-flight on the next call and mid-stream at chunk boundaries. (2) Cooperative flag: `solwyn.current_run_terminated() -> bool` (ambient run via the `_run.py` contextvar) and `solwyn.run_termination(run_id) -> RunTermination | None` for orchestrator loops that want to exit gracefully between steps instead of catching exceptions.

### D3. Receipts: schema, sources, delivery, ingest needs

**Decision: receipts are the existing `BUDGET_DENIED` MetadataEvents, enriched with attribution fields (not a new event type or endpoint), plus a fold-on-drop aggregate for outage windows. Every deny source emits; delivery is the reporter's existing at-least-once event channel, upgraded so a receipt that would be dropped folds into a bounded in-memory aggregate re-emitted after recovery.**

- **Why enrich, not invent:** Part 0 #4 — emission already exists at all four deny sites with a per-call `call_id` join key; the server already dedups ingest on `(project_id, call_id, attempt_index)` (`core: api/migrations/versions/0001_initial.py:2236-2245`). Inventing a parallel receipt pipe would duplicate the reporter's queue/retry/shutdown machinery for no additional guarantee.
- **The server side of "Prove" substantially exists already** (verified in core; full citations in Part 4): ingest splits denied spend into a dedicated `cost_events.denied_cost` column (`api/src/solwyn_api/services/metadata_ingest.py:2054-2059`), a per-run `agent_run.total_denied_cost` aggregate is maintained on ingest, and a `SavingsSummary{saved_cost, saved_input_tokens, blocked_call_count}` already rides `/costs` and `/costs/total` (`api/src/solwyn_api/routers/costs.py:236-239`, `costs.py:1343-1361`) — the dashboard even renders a "~$X saved" callout (`dashboard/src/pages/projects/detail/costs.tsx:271-308`). **So the receipts program is: SDK attribution + outage durability + core storage of the new fields + per-event exposure — NOT building an avoided-spend aggregation.** One pricing nuance: today a denied event carries `input_tokens=est_in, output_tokens=0` (`client.py:1294-1295`), so `denied_cost` prices the avoided INPUT only; `estimated_output_bound` lets core additionally price an "up to" band (C-3), keeping the headline conservative.
- **Deny sources emitted** (the `deny_source` vocabulary, set on `BudgetCheckResult` by each builder and threaded into the event):
  - `server` — live cloud hard deny (`_build_result_from_response`, `budget.py:465-477`)
  - `sticky_replay` — outage replay of a prior hard deny (`_build_prior_hard_deny_unavailable_result`, `budget.py:343-375`)
  - `local_enforcement` — fail-closed local math while unreachable (`_build_local_enforcement_result`, `budget.py:489-538`)
  - `lease_exhausted` — lease-ladder deny at true headroom exhaustion (`_result_from_admission`, `budget.py:783-809`)
  - `local_velocity` — velocity deny-mode termination (client gate, PR-2)
  - `run_terminated` — deny of a call into an already-terminated run (registry gate)
  - `aggregate_replay` — synthetic post-outage aggregate receipt (PR-6)
- **Receipt payload** (all optional, None-skipped on the wire; exact field specs in Part 2): `deny_source`, `deny_reason` (echoes `denied_by_period`, the velocity rule, or the directive reason), `denied_by_period`, `estimated_output_bound` (the call's effective output cap, already computed pre-check at `client.py:1261-1267` — with `input_tokens` (already carried, `client.py:1294`) this lets the server price the AVOIDED call: input estimate + output bound at the denied model's card; the SDK still never computes cost), `velocity_flags` (also attachable to SUCCESS events so flagged-but-allowed calls are dashboard-visible), `receipt_aggregate_count` (PR-6 aggregates only).
- **Delivery guarantee:** healthy plane — existing at-least-once ingest (bounded retry, FIFO, drops counted+warned). Outage — PR-6: an event with `status=BUDGET_DENIED` that reaches a lossy disposition (retry exhaustion, overflow eviction, sealed-delivery refusal) folds into a bounded per-`(agent_run_id, deny_source)` aggregate (count, summed estimated tokens, first/last timestamps, ≤256 keys) instead of vanishing; the flush loop re-emits aggregates as synthetic `aggregate_replay` receipt events after the next successful send. This honors "every deny leaves a durable trace" without a durable on-disk spool (rejected for settlements in PJ-2; same posture here).
- **What the API must ingest:** the new optional MetadataEvent fields (its mirror model is `extra="forbid"` — deploys FIRST), storage columns, and an aggregation query for the dashboard figure. Part 4 specifies.

### D4. Sequencing: SDK-only first, then wire, then dashboard

- **SDK-only, no core dependency (land immediately):** PR-1 (detector, warn-only), PR-2 (run-control registry + local deny mode + typed error + cooperative flag), PR-3 (streaming abort seam). After PR-3 the LOCAL demo works: a retry loop in a `solwyn.run()` scope gets flagged, killed client-side in deny mode, and the deny is visible as `BUDGET_DENIED` events + warnings.
- **Wire + core (API deploys first):** PR-4 (receipt fields on events; core C-1 ingest+storage), PR-5 (run_control directive; core C-2 kill state + directive attach + kill endpoint).
- **Dashboard:** D-1 (kill button), D-2 (saved-you figure + flagged-runs surfacing) — Part 4.
- **Durability tail:** PR-6 (fold-on-drop receipt aggregates; core already accepts the fields from C-1).
- **Smallest slice that produces THE demo** ("detected the loop, killed it from the dashboard, here's the receipt"): PR-1 + PR-2 + PR-4 + PR-5 + C-1 + C-2 + D-1 + the C-3 saved-you endpoint stub reading BUDGET_DENIED rows. PR-3 and PR-6 harden the story (mid-stream kill, outage-proof receipts) but are not demo-blocking.

---

## Part 2 — Wire-contract deltas (exact; every one requires a matching Core API change, API deploys first)

New shared type (SDK `src/solwyn/_types.py`; mirrored in core's shared models):

```python
class RunControlDirective(BaseModel):
    """Versioned server order for the active agent run (v1: terminate only)."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["1"]
    action: Literal["terminate"]
    agent_run_id: str = Field(..., max_length=AGENT_RUN_ID_MAX_LENGTH)
    reason: str = Field(..., max_length=64)  # e.g. "manual_kill", "runaway_velocity"
```

`agent_run_id` is an echo guard: the SDK applies the directive only when it matches the request's run id (a misrouted directive must not kill the wrong run).

| Model | Field added | Type / default | When omitted | Notes |
|---|---|---|---|---|
| `BudgetCheckResponse` (`_types.py:473`) | `run_control` | `RunControlDirective \| None = None` | Omitted (server serializes exclude-none for directive-v1 requests) unless the run is kill-requested | Terminate responses ALSO carry `allowed=false, mode="hard_deny", denied_by_period="agent_run"` |
| `LeaseGrantResponse` (`_types.py:688`) | `run_control` | `RunControlDirective \| None = None` | Same | Rides grant AND renewal responses (one model serves both, `budget.py:1084`, `budget.py:1711`) |
| `BudgetCheckRequest` (`_types.py:391`) | `run_directive_version` | `Literal["1"] \| None = None` | Omitted when None (existing None-skipping serializer, `_types.py:396-412`) | Runtime always sends `"1"` (mirrors `failover_directive_version`, `budget.py:273`) |
| `LeaseGrantRequest` (`_types.py:510`) | `run_directive_version` | `Literal["1"] \| None = None` | Same | |
| `LeaseRenewRequest` (`_types.py:574`) | `run_directive_version` | `Literal["1"] \| None = None` | Same | The renewal channel is the ≤-one-renewal kill path |
| `MetadataEvent` (`_types.py:251`) | `deny_source` | `Literal["server","sticky_replay","local_enforcement","lease_exhausted","local_velocity","run_terminated","aggregate_replay"] \| None = None` | Omitted on non-deny events (None-skipping serializer, `_types.py:261-272`) | Receipt attribution |
| `MetadataEvent` | `deny_reason` | `str \| None = None`, `max_length=64` | Omitted when None | Rule name / directive reason / period echo |
| `MetadataEvent` | `denied_by_period` | `str \| None = None`, `max_length=32` | Omitted when None | Mirrors the check-response vocabulary |
| `MetadataEvent` | `estimated_output_bound` | `int \| None = None`, `ge=0` | Omitted when None | Lets the server price the avoided call's output side |
| `MetadataEvent` | `velocity_flags` | `list[str] \| None = None`, each item `max_length=32`, list `max_length=8` | Omitted when None | On flagged calls (SUCCESS or BUDGET_DENIED) |
| `MetadataEvent` | `receipt_aggregate_count` | `int \| None = None`, `ge=1` | Omitted when None | PR-6 synthetic aggregates only; when set, `input_tokens` is the SUMMED estimate over the folded window |

Explicitly NOT added in v1: `velocity_flags` on `BudgetCheckRequest` (run traffic is leased, so check-side flags would rarely be seen; the event channel already delivers flags to the server within ~5s — revisit only if core grows a synchronous auto-kill policy), and any rule-config payload in `run_control` (v2 seam, reserved by the version field).

SDK-side pins to update in the same PRs: `EXPECTED_CHECK_REQUEST/RESPONSE_FIELDS`, `EXPECTED_METADATA_FIELDS`, lease request/response field sets in `tests/unit/test_contract_snapshot.py:237-318`; live drift coverage extends `tests/integration/test_live_contract.py`.

---

## Part 3 — SDK execution stages (PR-sized, tests per stage)

### PR-1: Content-free velocity detector (warn-only)

**Files:**
- Create: `src/solwyn/_velocity.py`
- Modify: `src/solwyn/config.py` (new fields + env mapping, after `lease_output_bound_default`, `config.py:96`)
- Modify: `src/solwyn/_base.py` (`_SolwynBase.__init__`: construct monitor; `_base.py:533-598`)
- Modify: `src/solwyn/client.py` (feed monitor in `_intercepted_call` sync `client.py:1226-1248` / async `client.py:2302`, and `_media_call` sync `client.py:992-1022` / async `client.py:2102`)
- Create: `tests/unit/test_velocity.py`
- Modify: `tests/unit/test_config.py`

**Interfaces (produced, relied on by PR-2/PR-4):**
- `VelocityMonitor(config: VelocityConfig)` — thread-safe (own lock), fork-reset via `_reset_after_fork_in_child` (clears state, fresh lock; registered by `_SolwynBase` which already calls `register_fork_reset(self)` at `_base.py:598`).
- `VelocityMonitor.observe(*, run_id: str, estimated_input_tokens: int, model: str, now: float) -> tuple[str, ...]` — returns the tuple of rule names that fired for THIS call (empty tuple when clean). Pure state update + evaluation; no I/O, no logging (caller logs).
- `VelocityConfig` NamedTuple: `mode: Literal["off","warn","deny"]`, `repeat_count: int`, `repeat_window_s: float`, `growth_streak: int`, `growth_factor: float`, `accel_floor_per_min: int`, `accel_factor: float`.
- Config fields on `SolwynConfig`: `velocity_mode: Literal["off","warn","deny"] = "warn"`, `velocity_repeat_count: int = Field(default=5, ge=2)`, `velocity_repeat_window_s: float = Field(default=60.0, gt=0)`, `velocity_growth_streak: int = Field(default=8, ge=3)`, `velocity_growth_factor: float = Field(default=3.0, gt=1.0)`, `velocity_accel_floor_per_min: int = Field(default=30, ge=1)`, `velocity_accel_factor: float = Field(default=3.0, gt=1.0)`. Env vars: `SOLWYN_VELOCITY_MODE`, `SOLWYN_VELOCITY_REPEAT_COUNT`, etc. (extend the mapping at `config.py:145-159`).

**Design constraints for the implementer:**
- Per-run state: `deque[tuple[float, int, str]]` of `(now, estimated_input_tokens, model)`, `maxlen=64`; run map is an `OrderedDict` LRU bounded at 128 runs (mirrors `_MAX_STICKY_RUN_DENIALS`, `budget.py:82`). Both bounds are module constants, not config.
- Rule evaluation exactly as specified in D1's table. `repeat_size` tolerance: `abs(size - current) <= max(8, 0.02 * current)`. `rate_acceleration` compares count(last 60s) to count(60-120s ago) and requires the absolute floor — and is evaluated but NEVER included in the deny-eligible subset (that subset is a module constant `DENY_ELIGIBLE_RULES = frozenset({"repeat_size", "monotonic_growth"})` consumed by PR-2).
- The monitor never sees kwargs, messages, or any string other than `model` and `run_id` (both structural identifiers). Signature takes scalars only — this is the privacy boundary.
- Client wiring (per call, run-scoped only): after `est_in` is computed and before `check_budget`, `flags = self._velocity.observe(run_id=agent_run[0], estimated_input_tokens=est_in, model=requested_model, now=time.monotonic()) if agent_run[0] is not None and self._config.velocity_mode != "off" else ()`. Warn path in a tiny client helper `_warn_velocity(run_id, flags)`: WARNING log `"velocity.flagged: rule=%s run=%s"` rate-limited per (run, rule) to 1/30s — rate-limit state lives inside the monitor (`should_warn(run_id, rule, now) -> bool`) so both client classes share it.
- Thread the returned `flags` through to the metadata-event call sites as a local now (used by PR-4; until then it feeds only the warn path).

**Steps:**

- [ ] **Step 1: failing tests for the three rules + bounds + fork reset** (`tests/unit/test_velocity.py`). Representative cases (write all of these):

```python
def _mon(**over):
    cfg = VelocityConfig(mode="warn", repeat_count=5, repeat_window_s=60.0,
                         growth_streak=8, growth_factor=3.0,
                         accel_floor_per_min=30, accel_factor=3.0)._replace(**over)
    return VelocityMonitor(cfg)

def test_repeat_size_fires_on_fifth_near_identical_call():
    m = _mon()
    for i in range(4):
        assert m.observe(run_id="r1", estimated_input_tokens=1000, model="gpt-5", now=float(i)) == ()
    assert "repeat_size" in m.observe(run_id="r1", estimated_input_tokens=1005, model="gpt-5", now=4.0)

def test_repeat_size_ignores_other_models_and_stale_window():
    m = _mon()
    for i in range(4):
        m.observe(run_id="r1", estimated_input_tokens=1000, model="gpt-5", now=float(i))
    assert m.observe(run_id="r1", estimated_input_tokens=1000, model="claude-x", now=4.0) == ()
    assert m.observe(run_id="r1", estimated_input_tokens=1000, model="gpt-5", now=200.0) == ()

def test_monotonic_growth_needs_streak_factor_and_tight_cadence():
    m = _mon()
    sizes = [100, 220, 500, 900, 1500, 2200, 3000, 4000]  # 8 strictly growing, 40x
    flags = ()
    for i, s in enumerate(sizes):
        flags = m.observe(run_id="r1", estimated_input_tokens=s, model="gpt-5", now=i * 2.0)
    assert "monotonic_growth" in flags

def test_monotonic_growth_not_fired_for_slow_human_paced_chat():
    m = _mon()
    for i, s in enumerate([100, 220, 500, 900, 1500, 2200, 3000, 4000]):
        flags = m.observe(run_id="r1", estimated_input_tokens=s, model="gpt-5", now=i * 120.0)
    assert flags == ()  # median gap 120s >= 30s

def test_rate_acceleration_requires_floor_and_factor():
    m = _mon()
    # prior window: 5 calls; current window: 35 calls -> floor met, 7x prior
    for i in range(5):
        m.observe(run_id="r1", estimated_input_tokens=10 + i, model="m", now=float(i))
    flags = ()
    for i in range(35):
        flags = m.observe(run_id="r1", estimated_input_tokens=500 + 20 * i, model=f"m{i%3}", now=70.0 + i)
    assert "rate_acceleration" in flags

def test_runs_are_isolated_and_lru_bounded():
    m = _mon()
    for i in range(4):
        m.observe(run_id="r1", estimated_input_tokens=1000, model="g", now=float(i))
    assert m.observe(run_id="r2", estimated_input_tokens=1000, model="g", now=4.0) == ()
    for n in range(200):  # exceed the 128-run bound; r1 must be evicted, not grow unbounded
        m.observe(run_id=f"bulk{n}", estimated_input_tokens=1, model="g", now=300.0)
    assert m.run_count() <= 128

def test_should_warn_rate_limits_per_run_rule():
    m = _mon()
    assert m.should_warn("r1", "repeat_size", now=0.0) is True
    assert m.should_warn("r1", "repeat_size", now=10.0) is False
    assert m.should_warn("r1", "monotonic_growth", now=10.0) is True
    assert m.should_warn("r1", "repeat_size", now=31.0) is True

def test_fork_reset_clears_state_and_replaces_lock():
    m = _mon()
    m.observe(run_id="r1", estimated_input_tokens=1, model="g", now=0.0)
    old_lock = m._lock
    m._reset_after_fork_in_child()
    assert m.run_count() == 0 and m._lock is not old_lock
```

- [ ] **Step 2:** `pytest tests/unit/test_velocity.py -v` → FAIL (module missing).
- [ ] **Step 3:** implement `src/solwyn/_velocity.py` to the interface above (module docstring must state the privacy posture: scalars and identifiers only, no content, no logging). Deque/OrderedDict + `threading.Lock`; rule evaluation as one pass over the run's window at observe time; `statistics.median` for the gap check.
- [ ] **Step 4:** `pytest tests/unit/test_velocity.py -v` → PASS.
- [ ] **Step 5: failing config tests** in `tests/unit/test_config.py`: defaults (`velocity_mode == "warn"`, thresholds per D1), env-var loading (`SOLWYN_VELOCITY_MODE=deny`, `SOLWYN_VELOCITY_REPEAT_COUNT=9`), rejection of invalid mode (`extra="forbid"`-style `ValidationError`). Then implement the `SolwynConfig` fields + env mapping. Run: `pytest tests/unit/test_config.py -v` → PASS.
- [ ] **Step 6: wiring tests** (extend `tests/unit/test_client.py`, using the repo's existing fake-provider fixtures): a run-scoped loop of 5 identical-size calls logs exactly one `velocity.flagged` WARNING (caplog), `velocity_mode="off"` logs none, unscoped calls never feed the monitor. Implement wiring in all four interception sites + `_SolwynBase.__init__` monitor construction. Run: `pytest tests/unit/test_client.py -k velocity -v` → PASS.
- [ ] **Step 7:** `make check && make test` (privacy firewall + no-assert tests sweep the new module automatically). Commit: `feat(sdk): add content-free velocity detector (warn-only)`.

### PR-2: Run-control registry, `RunTerminatedError`, cooperative flag, local deny mode

**Files:**
- Create: `src/solwyn/_run_control.py`
- Modify: `src/solwyn/exceptions.py` (new error class)
- Modify: `src/solwyn/budget.py` (`_cache_response` allow branch clears server-sourced terminations, `budget.py:307-314`)
- Modify: `src/solwyn/client.py` (termination gates + deny-mode wiring at the four interception sites)
- Modify: `src/solwyn/__init__.py` (export `RunTerminatedError`, `current_run_terminated`, `run_termination`, `clear_run_termination`, `RunTermination`)
- Modify: `src/solwyn/_lifecycle.py` (register the module-level registry holder for fork reset)
- Create: `tests/unit/test_run_control.py`

**Interfaces (produced):**

```python
# _run_control.py (module-level, process-wide; bounded OrderedDict + Lock)
@dataclass(frozen=True)
class RunTermination:
    reason: str                      # bounded upstream; never content
    source: Literal["server", "local_velocity"]
    at_monotonic: float

def mark_terminated(run_id: str, *, reason: str, source: ...) -> None   # idempotent; first writer wins
def run_termination(run_id: str) -> RunTermination | None
def clear_termination_if(run_id: str, *, source: ...) -> None           # used by the allow-clears-server rule
def clear_run_termination(run_id: str) -> None                          # public escape hatch (any source)
def current_run_terminated() -> bool                                    # ambient run via _run.current_run()

# exceptions.py
class RunTerminatedError(SolwynError):
    def __init__(self, *, agent_run_id: str, reason: str, source: str) -> None
```

Registry bound: 256 entries, LRU eviction (constant `_MAX_TERMINATED_RUNS = 256`). Fork behavior: entries SURVIVE fork (a terminated run is terminated in the child too); only the lock is replaced (a `_RunControlState` holder object registered via `register_fork_reset`).

**Client gate semantics (implement identically at all four interception sites):**
1. **Pre-check gate:** if `run_termination(run_id)` is non-None with `source="local_velocity"` → emit a `BUDGET_DENIED` event (existing builder; receipt fields arrive in PR-4) and raise `RunTerminatedError` without calling `check_budget` (local kills have no server un-kill to discover).
2. **Deny-mode marking:** if PR-1's `flags` intersect `DENY_ELIGIBLE_RULES` and `velocity_mode == "deny"` → `mark_terminated(run_id, reason=f"velocity:{rule}", source="local_velocity")`, then fall into gate 1's raise for this same call (the flagged call itself never reaches the provider).
3. **Post-check gate:** after `check_budget` returns, if `run_termination(run_id)` is non-None (any source; a server-sourced entry appears in PR-5, and a live ALLOW has already had its chance to clear it via `_cache_response`) → emit the `BUDGET_DENIED` event and raise `RunTerminatedError` instead of proceeding; when `budget.allowed` is also False, prefer `RunTerminatedError` over `BudgetExceededError` whenever the registry has the run.

**Steps:**

- [ ] **Step 1: failing registry tests** (`tests/unit/test_run_control.py`): mark/read/idempotent-first-writer-wins; `clear_termination_if(source="server")` leaves local entries; LRU bound at 256; `current_run_terminated()` True inside a `solwyn.run()` scope whose id was marked, False outside; fork reset preserves entries and swaps the lock.
- [ ] **Step 2:** run → FAIL. Implement `_run_control.py` + `RunTerminatedError` + exports. Run → PASS.
- [ ] **Step 3: failing enforcer test** (extend `tests/unit/test_budget.py`): after `mark_terminated(run, source="server")` (simulating PR-5), a live ALLOW response for that run processed through `_cache_response(response, agent_run_id=run)` clears the server-sourced entry; a local entry survives the same allow. Implement the one-line clear in the allow branch (`budget.py:307-314`). Run → PASS.
- [ ] **Step 4: failing client tests** (extend `tests/unit/test_client.py`): (a) `velocity_mode="deny"` + 5 identical-size calls in a run → 5th call raises `RunTerminatedError`, the provider fake records only 4 calls, a `BUDGET_DENIED` event was reported for the 5th; (b) subsequent calls in the same run raise immediately without touching `check_budget` (assert via enforcer spy); (c) `velocity_mode="warn"` never raises; (d) `rate_acceleration` alone never terminates even in deny mode; (e) async twins of (a)-(b). Implement the gates. Run → PASS.
- [ ] **Step 5:** `make check && make test`. Commit: `feat(sdk): run-control registry, RunTerminatedError, and velocity deny mode`.

### PR-3: Mid-stream cooperative abort

**Files:**
- Modify: `src/solwyn/stream.py` (both wrappers)
- Modify: `src/solwyn/client.py` (wherever `SyncStreamWrapper`/`AsyncStreamWrapper` are constructed — pass the abort seam for run-scoped calls)
- Modify: `tests/unit/test_stream.py`, `tests/unit/test_stream_failover.py` (constructor signature)

**Interfaces:**
- Both wrappers gain keyword-only `abort_check: Callable[[], Exception | None] | None = None`. Evaluated at each iteration head (before pulling the next chunk). Non-None return → `close()` (settle observed usage exactly-once + close inner stream) then `raise` that exception. The client passes `lambda: _stream_abort_exception(run_id)` for run-scoped calls, where the helper returns a `RunTerminatedError` when `run_termination(run_id)` is set, else None. Unscoped calls pass None (zero behavior change).

Sync loop shape (async twin mirrors it):

```python
def __iter__(self) -> Iterator[Any]:
    try:
        for chunk in self._stream:
            if self._abort_check is not None:
                abort_exc = self._abort_check()
                if abort_exc is not None:
                    self.close()          # settles partial usage exactly-once, closes inner
                    raise abort_exc
            self._accumulator.observe(chunk)
            ...
```

(The check runs after a chunk is received but before it is yielded; a blocking provider read cannot be interrupted cooperatively, so this is the tightest safe granularity. The observed-but-unyielded chunk IS included in settlement — the accumulator saw it — which is correct: the provider generated and will bill it.)

**Steps:**

- [ ] **Step 1: failing wrapper tests** (`tests/unit/test_stream.py`): (a) abort_check armed after chunk 2 of 5 → iteration raises exactly the returned exception at the next boundary; per the loop shape, chunk 3 is pulled from the inner stream but neither observed nor yielded (assert yielded == 2, accumulator observed exactly chunks 1-2, no chunk 4+ was pulled), `on_complete` fired exactly once with the observed partial usage, and the inner stream's `close()` was called; (b) abort_check returning None throughout → identical behavior to today (settled on exhaustion, all chunks yielded); (c) `abort_check=None` → untouched legacy behavior; (d) abort on FIRST iteration → zero chunks yielded, settle fires with zero usage; (e) async twins of (a)-(d).
- [ ] **Step 2:** run → FAIL. Implement both wrappers. Run → PASS.
- [ ] **Step 3: failing client integration test** (extend `tests/unit/test_client.py`): streaming call inside a run; mid-consumption, `mark_terminated(run_id, reason="manual_kill", source="server")`; next chunk boundary raises `RunTerminatedError`; the settlement (confirm+event) reflects partial usage via the existing `report_settlement` path. Implement client wiring. Run → PASS.
- [ ] **Step 4:** `make check && make test` (privacy firewall re-verifies stream.py's no-content posture automatically). Commit: `feat(sdk): cooperative mid-stream abort for terminated runs`.

### PR-4: Denial receipts — attribution fields end-to-end ⚠ requires core C-1 deployed first

**Files:**
- Modify: `src/solwyn/_types.py` (`MetadataEvent` + 6 fields per Part 2)
- Modify: `src/solwyn/budget.py` (`BudgetCheckResult` + `deny_source`/`deny_reason`/`denied_by_period` internal fields, `budget.py:115-139`; set them in `_build_result_from_response` `budget.py:407-477`, `_build_prior_hard_deny_unavailable_result` `budget.py:343-375`, `_build_local_enforcement_result` `budget.py:489-538`, `_result_from_admission` `budget.py:783-809`)
- Modify: `src/solwyn/_base.py` (`_build_metadata_event` accepts the receipt params, `_base.py:801-871`)
- Modify: `src/solwyn/client.py` (four deny sites pass attribution + `estimated_output_bound` — hoist the bound already computed at `client.py:1261-1267` into a local; attach `velocity_flags` to the flagged call's SUCCESS/BUDGET_DENIED event; run-terminated/local-velocity gates from PR-2 set their sources)
- Modify: `tests/unit/test_contract_snapshot.py` (`EXPECTED_METADATA_FIELDS`)
- Modify: `tests/unit/test_budget.py`, `tests/unit/test_client.py`
- Modify: `tests/integration/test_live_contract.py` (denied-event ingest accepted live)

**Interfaces:** `BudgetCheckResult.deny_source/deny_reason/denied_by_period: str | None = None` (internal model — NOT wire; no snapshot pin). `_build_metadata_event(..., deny_source=None, deny_reason=None, denied_by_period=None, estimated_output_bound=None, velocity_flags=None)`.

**Steps:**

- [ ] **Step 1: failing wire tests**: snapshot pin update for `EXPECTED_METADATA_FIELDS`; serializer tests — a SUCCESS event without receipt fields dumps byte-identical keys to today (None-skipping), a BUDGET_DENIED event with `deny_source="sticky_replay"` dumps exactly the set fields.
- [ ] **Step 2: failing attribution tests** (`tests/unit/test_budget.py`): each builder stamps its source — live hard deny → `("server", period, period)`; unreachable-after-deny → `"sticky_replay"` with the preserved response's period; fail-closed local math → `"local_enforcement"`; lease-ladder deny → `"lease_exhausted"`.
- [ ] **Step 3:** implement types + builders. Run → PASS.
- [ ] **Step 4: failing client tests**: hard-denied chat call's reported event carries `deny_source="server"`, `estimated_output_bound` equal to the value passed to `check_budget`, and the run id; a velocity-flagged ALLOWED call's SUCCESS event carries `velocity_flags=("repeat_size",)`; a PR-2 local kill's event carries `deny_source="local_velocity"`, `deny_reason="velocity:repeat_size"`. Async twins. Implement client threading. Run → PASS.
- [ ] **Step 5:** `make check && make test`; then against a local core with C-1 deployed: `make test-integration` (new live test: POST a denied event with all receipt fields → 202 with zero per-event rejections). Commit: `feat(sdk): denial receipts — deny-source attribution on BUDGET_DENIED events`.

### PR-5: `terminate_run` directive on the wire ⚠ requires core C-2 deployed first

**Files:**
- Modify: `src/solwyn/_types.py` (`RunControlDirective`; `run_control` on `BudgetCheckResponse` + `LeaseGrantResponse`; `run_directive_version` on the three request models — exact specs in Part 2)
- Modify: `src/solwyn/budget.py` (send opt-in `run_directive_version="1"` in `_build_check_request` `budget.py:263-274`, `_build_grant_request` `budget.py:628-648`, `claim_renewal_request` call sites; process directives — new `_apply_run_control(directive, agent_run_id)` on `_BudgetEnforcerBase` called from both `check_budget` bodies after validation (`budget.py:1415-1434`, `budget.py:1997-2016`), from `_install_grant` (`budget.py:1070-1107`), and from `_finish_renewal` (`budget.py:1025-1068`) on the unfenced path only)
- Modify: `src/solwyn/client.py` (deny sites raise `RunTerminatedError` when the registry holds the run — already generic from PR-2's post-check gate; verify no site bypasses it)
- Modify: `tests/unit/test_contract_snapshot.py` (five pin updates), `tests/unit/test_budget.py`, `tests/unit/test_budget_lease.py`, `tests/integration/test_live_contract.py`

**`_apply_run_control` semantics (sans-I/O, on the base enforcer):**
1. Ignore unless `directive.action == "terminate"` and `directive.agent_run_id == agent_run_id` (echo guard; mismatches log one WARNING `run_control.directive_run_mismatch` with both ids — ids are structural, never content).
2. `_run_control.mark_terminated(agent_run_id, reason=directive.reason, source="server")`.
3. Feed stickiness with the response itself when it is a deny (the normal terminate response already carries `allowed=false, denied_by_period="agent_run"` and flows through `_cache_response` / `_lease_deny_result` unchanged); when a directive arrives on an ALLOW-shaped response (defensive: server bug), synthesize the run-scoped denial exactly as `_lease_deny_result` does (`budget.py:766-781`) so stickiness never depends on server shape discipline.
4. Drop the run's lease: `self._lease.drop_if_current(agent_run_id, lease_id=<current>, generation=<current>)` under `_state_lock`. (The server's kill flag independently turns the next grant/renewal into an authoritative deny carrying the directive — C-2; any post-kill renewal 404/409 is already handled at `budget.py:739-747`.)

**Steps:**

- [ ] **Step 1: failing wire tests**: pins; `RunControlDirective` round-trip; requests dump `run_directive_version": "1"` from the runtime builders; a check response WITH `run_control` parses; one WITHOUT (exclude-none) parses unchanged.
- [ ] **Step 2: failing enforcer tests** (`tests/unit/test_budget.py` + `tests/unit/test_budget_lease.py`): (a) check response `allowed=false, denied_by_period="agent_run"` + matching directive → registry marked (source=server), result denied, sticky replay preserves the deny during a subsequent simulated outage; (b) mismatched `agent_run_id` in the directive → NOT applied, one warning; (c) renewal response carrying the directive (via `_finish_renewal`) → registry marked, lease dropped, next admission for the run takes the legacy path and a live check re-decides; (d) fenced/late renewal response with a directive → discarded with the fence (no registry write); (e) directive on an allow-shaped response → synthesized run-scoped sticky deny; (f) later live ALLOW for the run → sticky cleared AND server-sourced registry entry cleared (un-kill; PR-2 Step 3 machinery); async twins where paths diverge.
- [ ] **Step 3:** implement. Run → PASS.
- [ ] **Step 4: end-to-end client test**: fake control plane returns terminate on the 6th check of a run → 6th call raises `RunTerminatedError` (not `BudgetExceededError`), its `BUDGET_DENIED` receipt carries `deny_source="server"`, `deny_reason="manual_kill"`; an in-flight stream in the same run (PR-3 seam) aborts at its next chunk. Run → PASS.
- [ ] **Step 5:** `make check && make test`; with core C-2 up: `make test-integration` (live: create run, kill via the C-2 endpoint, next check carries the directive). Commit: `feat(sdk): server-pushed terminate_run directive (check + lease channels)`.

### PR-6: Outage-durable receipts (fold-on-drop aggregates)

**Files:**
- Modify: `src/solwyn/reporter.py` (`_ReporterBase`: fold table + drain; hook the three lossy dispositions for BUDGET_DENIED events — retry exhaustion in the batch-failure path, overflow eviction in `_enqueue_owned` (`reporter.py:312-345`), sealed-delivery refusal in `_move_event_to_queue` (`reporter.py:347-359`) and the enqueue gates)
- Modify: `tests/unit/test_reporter.py`, `tests/unit/test_reporter_retry.py`, `tests/unit/test_reporter_exit_flush.py`

**Interfaces / semantics:**
- `_fold_receipt(event: MetadataEvent) -> None`: keyed by `(agent_run_id or "", deny_source or "")`; aggregate = `{count, summed_input_tokens, summed_output_bound, first_ts, last_ts, model, provider}`; bounded 256 keys — when full, increment a single overflow counter folded into drop accounting (`_count_drop("event", "receipt_fold_overflow")`) so even the fold's own bound is visible.
- Drain: at the top of each flush cycle whose previous cycle sent successfully (and once during close's final flush), non-empty folds convert to synthetic events — fresh `call_id`, `status=BUDGET_DENIED`, `deny_source="aggregate_replay"`, `receipt_aggregate_count=count`, `input_tokens=summed`, `estimated_output_bound=summed_output_bound`, `timestamp=now`, run id/model/provider from the fold — enqueued through the normal event path. At-least-once composes: if the synthetic event itself dies, it folds again (count preserved by re-folding the aggregate, not double-counting).
- Only `status == CallStatus.BUDGET_DENIED` events fold; everything else keeps today's counted-drop behavior. Aggregates fold into aggregates (sum counts) — `aggregate_replay` events that die re-fold under their original run key.

**Steps:**

- [ ] **Step 1: failing reporter tests**: (a) a BUDGET_DENIED event exhausting `max_send_attempts` folds instead of dropping (drop counter NOT incremented for it; fold table holds count=1 with its token sum); (b) 100 denied events for one run during a simulated dead plane → one fold entry, count=100; (c) recovery → next successful flush emits exactly one `aggregate_replay` event with `receipt_aggregate_count=100` and the summed tokens; (d) overflow eviction of a denied event folds; a SUCCESS event still drop-counts; (e) close() drains folds within the shutdown deadline (extend `test_reporter_exit_flush.py`); (f) async twins.
- [ ] **Step 2:** run → FAIL. Implement. Run → PASS.
- [ ] **Step 3:** `make check && make test`. Commit: `feat(sdk): fold-on-drop denial receipts survive control-plane outages`.

---

## Part 4 — Core API + dashboard execution stages

Cross-repo work in `~/dev/repos/solwyn-ai/core` (`api/` = FastAPI customer API, `dashboard/` = React customer SPA). All paths below are relative to that repo. Investigated 2026-08-07; every citation verified against the working tree.

### What already exists server-side (do not rebuild)

| Capability | Where |
|---|---|
| Denied-spend storage: `cost_events.denied_cost numeric(12,6)` beside `actual_cost`; denied events store `actual_cost=0.0`, avoided amount in `denied_cost` | `api/migrations/versions/0001_initial.py:1896-1946`; split logic `api/src/solwyn_api/services/metadata_ingest.py:2054-2059`; canonical rule in the `0002_admin_plane.py:14-18` docstring |
| Ingest dedup ledger `(project_id, call_id, attempt_index)`, append-only | `api/migrations/versions/0001_initial.py:2236-2245`, grants at `0001_initial.py:1095-1096` |
| Per-run denied aggregate `agent_run.total_denied_cost`, upserted on ingest | `api/src/solwyn_api/db/models.py:398`; upsert `metadata_ingest.py:2488-2502` |
| "What the cap saved you": `SavingsSummary{saved_cost, saved_input_tokens, blocked_call_count}` on `GET /costs` and `/costs/total` (`SUM(denied_cost)` — never `SUM(actual_cost)` filtered to denied, which is always $0) | `api/src/solwyn_api/routers/costs.py:236-239`, `costs.py:1343-1361`; also `AttestationSpendSummary.denied_spend_usd` (`api/src/solwyn_api/routers/attestations.py:259-300`) |
| Dashboard rendering of both: per-run "Lifetime denied cost" column + red row tint, and a "~$X saved" `SavingsCallout` on the Costs tab | `dashboard/src/pages/projects/detail/agents.tsx:182-191`, `agents.tsx:237-256`; `dashboard/src/pages/projects/detail/costs.tsx:271-308` |
| `runaway_run` budget rule (spend cap per run, optional call-rate/window pair) with full dashboard editor; rule pause/snooze lifecycle endpoints | `api/src/solwyn_api/schemas/budget_rules.py:138-165`; `dashboard/src/components/alert-form.tsx:371-475`; `dashboard/src/pages/alerts/rule-drawer.tsx:174-221` |
| Directive plumbing precedent: check handler attaches `FailoverDirective` and serializes `exclude_none` only for opted-in requests, via a hand-rolled `JSONResponse` + include-set | `api/src/solwyn_api/routers/budgets.py:1166-1203`; `FailoverDirective` in `shared/src/solwyn_shared/models.py:393-399` |

What does NOT exist: any run lifecycle/kill state (`agent_run` is a pure telemetry aggregate with no status column, `api/src/solwyn_api/db/models.py:339-407`), any run-mutating endpoint (`agent_runs.py` has exactly two GETs, `api/src/solwyn_api/routers/agent_runs.py:309-311`, `agent_runs.py:402-404`; the generated spec pins `post?: never`, `dashboard/src/api/generated/paths.ts:1704-1745`), any kill/terminate vocabulary anywhere in API or SPA, and any directive on lease responses (`api/src/solwyn_api/schemas/leases.py:199-260` has none).

### Server-side traps this plan must design around (found in investigation)

1. **The check serializer prunes fields by include-set, not None-ness** (`budgets.py:1195-1203`): a new `run_control` field must be added to the same removal logic `failover_directive` uses, or legacy SDKs (no opt-in) get a key their `extra="forbid"` mirror rejects.
2. **The failover-directive attach is wrapped in a bare `except Exception` that silently degrades to "no directive"** (`budgets.py:1181-1191`). A terminate signal must NOT ride that handler — a kill that silently fails open is the worst outcome. `run_control` gets its own attach block that fails loud (500 the check rather than drop a kill; a killed run's check erroring is acceptable, spending is not).
3. **Lease renew replays byte-identical stored responses** from `budget_leases.last_response_json` (`api/src/solwyn_api/db/models.py:727`; replay at `api/src/solwyn_api/services/budget_lease_service.py:1318`; written at `budget_lease_service.py:1359`, `budget_lease_service.py:2082`). A directive computed inside the service would be frozen stale into replays. Therefore the kill check + directive attach happens at the ROUTER edge, applied to every outgoing lease response — replayed or fresh.
4. **The `runaway_run` counter only arms on HARD_DENY projects** (`budget_aggregator.py:2189`) and `denied_by_period="agent_run"` has a second producer (`scoped_budget` with `scope="agent_run"`, `budget_aggregator.py:2315`). The kill flag is therefore its own router-edge check, independent of the rules engine, so it works on `alert_only` projects too.
5. **One unknown field 422s an entire 1000-event ingest batch** (`extra="forbid"` on `shared/src/solwyn_shared/models.py:251` + `additionalProperties:false` in `shared/openapi.json`). API-first deploy is mandatory, and the shared-vs-SDK parity gate must open a pending window per new field (`shared/tests/test_sdk_contract_parity.py:30-39` — the `_PENDING_SDK_*` sets).
6. **`GET /costs/events` hides the avoided amount today**: `CostEventResponse.cost = float(row.actual_cost)` (`costs.py:1150`) renders every denial as `$0.00`; `denied_cost` is not in the response model (`costs.py:291-330`). Per-event receipts need this exposed. Related: `group_by=status` can never return `budget_denied` because grouping reuses `spend_conditions` which excludes denials (`costs.py:1443-1446`).
7. **Dashboard codegen is a CI gate**: any endpoint/schema change requires `make openapi` (regenerates `shared/openapi.json`) then `npm run codegen` in `dashboard/` (`dashboard/package.json:11`); `scripts/check-codegen.mjs:32-35` fails CI on staleness.
8. **No index on `cost_events.status`** and the table is range-partitioned on `timestamp` — every new denial-only query must carry `project_id` + a bounded window (`metadata_ingest.py:1936-1946` documents the failure mode). Retention is 365 days (`api/src/solwyn_api/services/partition_retention.py:82-90`) — a lifetime "saved you" number needs a rollup before partitions drop (out of scope; the per-run `total_denied_cost` aggregate survives).

### Detection interplay with the server's existing runaway machinery

Core already runs a `runaway_run` pipeline — but it is **post-ingest, alert-only, spend/call-count based**, which is complementary to (not duplicated by) this plan's client-side velocity rules:

- The check-path side of the rule is a **spend cap** (denies at `spend_threshold_usd`, `budget_aggregator.py:2189-2217`).
- The alerting side (`_RunawayCandidate` + `RunawayRunNotificationData`, `budget_aggregator.py:387-462`; evaluated by `check_runaway_runs`, `budget_aggregator.py:4514`, driven post-commit from ingest) fires per-run safety ALERTS, optionally on a call-rate/window pair (`call_threshold` / `window_minutes` on `RunawayRunConfig`, `schemas/budget_rules.py:161-165`). It runs minutes-lagged behind ingest and never denies.

Client rules add what neither side can: **pre-spend, per-call, real-time** signals (including inside leased runs the server never sees per-call), plus size-shape rules (`repeat_size`, `monotonic_growth`) that need call-by-call granularity. Division of labor: client detects and (opt-in) locally denies; server keeps spend-cap denial + alerting; humans (or future core policy) convert alerts + `velocity_flags` into `terminate_run` directives. C-1/C-2 do not touch the existing runaway pipeline.

### Coordination with the budget-pressure program (parallel planning session, same channel)

Both programs add fields through the same check-response directive gate (`budgets.py:1166-1203`) and the same parity pending-set mechanism (`shared/tests/test_sdk_contract_parity.py`). Agreed composition rules:

- `terminate_run` rides its OWN opt-in marker (`run_directive_version` — this plan); budget-pressure rides its own. **Neither program bumps `failover_directive` to version "2".**
- Namespaces are disjoint: this plan owns `run_control`, `run_directive_version`, `deny_source`, `deny_reason`, `estimated_output_bound`, `velocity_flags`, `receipt_aggregate_count`; budget-pressure owns `budget_pressure`, `budget_pressure_version`, `chain_price_hints`, `pressure_degraded`, `is_endpoint_fallback`, `endpoint_label`, and the `FailoverReason` value `"budget_pressure"`. Neither reuses the other's names.
- Budget-pressure REPLACES the never-emitted `price_hints` on `BudgetCheckResponse` with chain-aligned `chain_price_hints` — nothing in this plan builds on `price_hints`, so no conflict.
- Whichever program's core PR lands second rebases the `response_fields` include-set block and the parity pending-sets over the first.
- Coordination closed, confirmed by the budget-pressure session 2026-08-07: it attaches its own signals fail-soft-but-loud (dedicated error metrics, not the silent `budgets.py:1181-1191` handler) and stamps lease responses at the router edge post-replay via `model_copy` — consistent with this plan's C-2 approach; `denied_by_period` vocabulary reused with existing meaning only.

### C-1: Ingest accepts + stores receipt fields (opens the API-first window for SDK PR-4)

**Files:**
- Modify: `shared/src/solwyn_shared/models.py` (`MetadataEvent` + the six receipt fields from Part 2, exact same specs — `deny_source`, `deny_reason`, `denied_by_period`, `estimated_output_bound`, `velocity_flags`, `receipt_aggregate_count`)
- Modify: `shared/tests/test_sdk_contract_parity.py` (add the six names to a new `_PENDING_SDK_METADATA_RECEIPT_FIELDS` window, pattern at `test_sdk_contract_parity.py:30-39`), `shared/tests/test_contract_snapshot.py` (field-set pins)
- Create: `api/migrations/versions/0007_denial_receipts.py` — `cost_events` + `deny_source varchar(32) NULL`, `deny_reason varchar(64) NULL`, `denied_by_period varchar(32) NULL`, `estimated_output_bound integer NULL`, `velocity_flags jsonb NULL`, `receipt_aggregate_count integer NULL`; `agent_run` + `velocity_flags jsonb NULL` (merged rule-name set for run-level "flagged" surfacing)
- Modify: `api/src/solwyn_api/db/models.py` (ORM mirrors on `CostEvent` ~`models.py:441-530` and `AgentRun` ~`models.py:339-407`)
- Modify: `api/src/solwyn_api/services/metadata_ingest.py` (persist the new columns in `_store_cost_events_bulk` `metadata_ingest.py:1026-1270` AND the per-event fallback `metadata_ingest.py:2148+`; union `velocity_flags` into the `agent_run` upsert at `metadata_ingest.py:2488-2502`; `receipt_aggregate_count >= 1` events settle `denied_cost` from the SUMMED `input_tokens` exactly like ordinary denials — no special pricing)
- Regenerate: `make openapi`
- Tests: shared round-trip + parity window; ingest tests asserting all six columns persist, a legacy event without the fields ingests unchanged, per-event rejection indexes unaffected, and the agent_run flags union

**Steps:**
- [ ] Failing shared-model tests (field pins + serializer None-skipping parity with the SDK's Part 2 spec) → implement models + parity window → PASS.
- [ ] Migration + ORM; failing ingest tests → implement persistence → PASS.
- [ ] `make openapi`; core quality gate; deploy. Commit: `feat(api): ingest + store denial-receipt attribution fields`.

### C-2: Run kill state, terminate endpoint, and the `run_control` directive (opens the window for SDK PR-5)

**Files:**
- Modify: `shared/src/solwyn_shared/models.py` (`RunControlDirective` per Part 2, next to `FailoverDirective` `models.py:393-399`; `BudgetCheckResponse.run_control` `models.py:476-506`; `BudgetCheckRequest.run_directive_version` `models.py:459-465` area), `shared/src/solwyn_shared/__init__.py` export, parity/snapshot tests (new pending windows for the request fields)
- Modify: `api/src/solwyn_api/schemas/leases.py` (`LeaseGrantResponse.run_control: RunControlDirective | None = None` — the module's own convention note at `leases.py:9-13` covers it; grant/renew request models gain `run_directive_version`)
- Create: `api/migrations/versions/0008_run_kill_state.py` — `agent_run` + `kill_requested_at timestamptz NULL`, `kill_reason varchar(64) NULL`, `killed_by varchar(36) NULL` (durable record; composite-PK row `(project_id, id)` exists for any run the dashboard can display, since rows are created by ingest)
- Modify: `api/src/solwyn_api/constants.py` (new `RedisKeys.run_kill(project_id, agent_run_id)` — hot-path flag, TTL 7 days; precedent `constants.py:27`, `constants.py:179-180`)
- Modify: `api/src/solwyn_api/routers/agent_runs.py` — new `POST /api/v1/projects/{project_id}/agent-runs/{run_id}/terminate`, JWT-only via `require_budget_write` (`api/src/solwyn_api/dependencies.py:418`; same gate class as `PUT /budget`, `budgets.py:1815-1826`), body `{reason?: str<=64}` defaulting to `"manual_kill"`; 404 if the run row doesn't exist for the project; writes the three columns + sets the Redis flag + emits an audit event (audit machinery: `routers/audit_events.py`); idempotent (re-kill refreshes reason/TTL, 200)
- Modify: `api/src/solwyn_api/routers/budgets.py` — check handler: when `body.agent_run_id` is set, one Redis `GET run_kill` (only for run-scoped checks; sub-ms beside the existing counter ops). If killed: override the decision to `allowed=false, mode=hard_deny, denied_by_period="agent_run"` BEFORE serialization, and attach `run_control` (only when `body.run_directive_version == "1"`; otherwise the include-set removes the key exactly like `failover_directive`, `budgets.py:1195`). The attach is its OWN code path — never inside the failover-directive `except Exception` degrade (`budgets.py:1181-1191`); a failure here fails the request loud.
- Modify: lease router edge (`budgets.py:1647-1737`): after the service returns (fresh OR replayed — this is the replay-staleness fix), the same Redis check; if killed, replace the response with the deny shape (`allowed=false`, lease fields absent, snapshot fields intact — deny precedent `budget_lease_service.py:1965-1971`) + `run_control` when the request opted in. Grant, renew, and surrender-adjacent paths all covered by one helper.
- Regenerate `make openapi`.
- Tests (api unit + integration): kill → next check denied WITH directive for an opted-in request; same for lease grant and renew, including a REPLAYED renew (idempotent retry) carrying a fresh directive; `alert_only` project still denied (router-edge override); legacy request without `run_directive_version` gets the deny but NO `run_control` key (no 422 on old SDK mirrors); non-existent run 404s; endpoint requires JWT + budget-write (API key → 403); audit event emitted; re-kill idempotent.

**Steps:**
- [ ] Shared models + parity windows (failing tests first) → PASS.
- [ ] Migration + Redis key + terminate endpoint (failing router tests first) → PASS.
- [ ] Check + lease router-edge enforcement (failing tests incl. the replayed-renew case) → PASS.
- [ ] `make openapi`; quality gate; deploy. Commit: `feat(api): run kill state + terminate_run directive on check and lease responses`.

### C-3: Per-event receipt exposure + "up to" band (after C-1)

**Files:**
- Modify: `api/src/solwyn_api/routers/costs.py` — `CostEventResponse` (`costs.py:291-330`) gains `denied_cost: float`, `deny_source: str | None`, `deny_reason: str | None`; populate from the row (`costs.py:1150` area). `SavingsSummary` (`costs.py:236-239`) gains `saved_cost_upper: float | None`: computed by pricing a synthetic `TokenDetails(input=saved_input_tokens, output=SUM(estimated_output_bound))` through the existing pure pricing math (`PricingService._calculate_cost`, `api/src/solwyn_api/services/pricing_service.py:100`; note `estimate_cost`'s 1:1 assumption at `pricing_service.py:863-890` is exactly why the bound-aware band uses the pure function instead). `None` when no denied row carried a bound. Headline `saved_cost` stays the conservative input-estimate figure.
- Tests: savings query returns the band only when bounds exist; denied event rows expose `denied_cost`/`deny_source`; the `SUM(denied_cost)`-not-`actual_cost` rule pinned (regression guard for the documented trap, `0002_admin_plane.py:14-18`).
- Regenerate `make openapi`.

**Steps:** failing tests → implement → PASS → deploy. Commit: `feat(api): per-event denial receipts + avoided-spend upper band`.

### D-1: Dashboard "Kill run" button (after C-2)

**Files:**
- Regenerate: `npm run codegen` in `dashboard/` (pulls the new terminate path; `scripts/check-codegen.mjs:32-35` gates staleness)
- Modify: `dashboard/src/api/hooks/use-agent-runs.ts` — first mutation in this file: `useTerminateRun(projectId, runId)` via `useSWRMutation` sharing the run key so `trigger()` revalidates the detail (key-sharing pattern documented at `dashboard/src/api/hooks/use-budget.ts:12-15`, mutation shape at `use-budget.ts:30-43`; POST-action precedent `dashboard/src/pages/alerts/rule-drawer.tsx:204-210`)
- Modify: `dashboard/src/pages/projects/detail/agents.tsx` — *Kill run* button in the `RunDetail` expanded panel (`agents.tsx:46-142`), destructive-confirm modal following `dashboard/src/components/emergency-revoke-dialog.tsx:17-60` (danger banner + explicit confirm; type-to-confirm not required — a kill is reversible-by-support and stops spend), Sonner toast on success, `Killed` badge on the row from `kill_requested_at`
- Tests: extend `dashboard/tests/pages/agents.test.tsx` (MSW handler for the terminate POST; confirm-flow → request fired → row shows Killed; error path toasts) and `dashboard/tests/api/use-agent-runs.test.ts` (hook contract)

**Steps:** codegen → failing component/hook tests → implement → PASS. Commit: `feat(dashboard): kill-run action on the agent-runs view`.

### D-2: Flagged-run surfacing + receipts presentation (after C-1, C-3)

**Files:**
- Modify: `dashboard/src/api/hooks/use-agent-runs.ts` + `dashboard/src/pages/projects/detail/agents.tsx` — `Flagged` badge on runs whose `velocity_flags` is non-empty (from C-1's `agent_run.velocity_flags`; new field arrives via codegen), tooltip lists rule names; this is the "detected the loop" half of the demo
- Modify: `dashboard/src/components/cost-events-table.tsx` — denied rows show the avoided amount from the new `denied_cost` field with an "avoided" affordance instead of today's forced `$0.00` (`cost-events-table.tsx:476-478`), plus `deny_source` as a small badge (`sticky replay`, `server`, `local velocity`, …)
- Modify: `dashboard/src/pages/projects/detail/costs.tsx` — `SavingsCallout` (`costs.tsx:271-308`) gains the "up to ~$Y" band when `saved_cost_upper` is present; keep the `blocked_call_count <= 0` hide rule
- Note for implementer: Recharts `ResponsiveContainer` latches its first jsdom measurement — any new chart needs an explicit parent height and a browser check (`dashboard/CLAUDE.md:106-108`); prefer non-chart presentation here (badges, callout copy)
- Tests: `dashboard/tests/pages/agents.test.tsx` (flag badge), `dashboard/tests/components/` cost-events-table cases (denied row shows avoided amount + source), costs page callout band case

**Steps:** codegen → failing tests → implement → PASS. Commit: `feat(dashboard): flagged runs, denial receipts, and avoided-spend band`.

---

## Part 5 — Sequencing & demo slices

```
SDK:   PR-1 ──► PR-2 ──► PR-3 ─────────────────────────► PR-6
                  │
Core:             │   C-1 (ingest fields) ──► C-3 (per-event exposure + band)
                  │     │
SDK:              └──► PR-4 (needs C-1 deployed)
Core:                 C-2 (kill state + directive + endpoint)
SDK:                          └──► PR-5 (needs C-2 deployed)
Dash:                 D-1 (kill button, needs C-2) · D-2 (flags + receipts UI, needs C-1/C-3)
```

Within each repo the stages are sequential; across repos, C-1 can start immediately (it only mirrors Part 2's field specs) and C-2 in parallel with SDK PR-2/PR-3. Every wire window follows the same dance: shared model + parity pending window → API deploy → SDK PR closes the window.

- **Local demo (SDK only, after PR-2/PR-3):** velocity deny mode catches a scripted retry loop on the 5th call, raises `RunTerminatedError`, aborts a live stream mid-consumption; receipts visible as BUDGET_DENIED events + warnings.
- **Flagship demo (after PR-4/PR-5 + C-1/C-2 + D-1):** scripted ping-pong run shows a `Flagged` badge in the dashboard (velocity_flags on ingested events), operator clicks *Kill run*, the run's next check/renewal carries `terminate_run`, the SDK raises `RunTerminatedError` within one call, and the run's denied calls appear with `deny_source="server"` alongside the already-shipping per-run "Lifetime denied cost" column — "detected the loop, killed it from the dashboard, here's the receipt."
- **Full story (add C-3 + D-2 + PR-6):** per-event receipts show the avoided amount and deny source, the existing "~$X saved" callout gains its "up to $Y" band, and receipts survive control-plane outages via fold-on-drop aggregates.

## Part 6 — Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| False positives on legitimate fan-out (N similar-size calls at once trips `repeat_size`) | Wrongly killed runs → trust damage | Warn-first default; deny is opt-in config; tight size tolerance (±max(8, 2%)) + same-model requirement; `rate_acceleration` permanently warn-only; per-(run,rule) 30s warn cooldown; receipts make every trip auditable so thresholds can be tuned before anyone enables deny |
| A terminate response with a novel `denied_by_period` value would poison the GLOBAL deny cache (`budget.py:338-340`) | One killed run blocks all traffic | Contract pins the terminate deny to `denied_by_period="agent_run"` (Part 2); PR-5 test (b)/(e) assert run-scoping; core C-2 test asserts the emitted period value |
| Wire drift between repos | 422s stranding checks/renewals | API deploys first (both repos' convention); field-set pins in `test_contract_snapshot.py`; live drift in `test_live_contract.py` extended per PR |
| Directive races the close fence (late renewal delivering a kill during client shutdown) | Stale directive mutating a closing enforcer | Directive processing sits inside the existing close-epoch fences (`budget.py:1036-1044`); PR-5 test (d) pins discard-on-fence |
| Mid-stream abort mis-settling (raise via the error path would zero out real usage) | Broken spend accounting | Contract is settle-then-raise through `close()` (exactly-once `_settled` guard, `stream.py:65-82`); PR-3 tests pin partial-usage settlement |
| Receipt fold hides individual deny records during outages | Coarser forensic detail | Aggregates preserve count + token sums + window bounds per (run, source); individual receipts still flow whenever the plane is up; drop accounting exposes even fold overflow |
| Detector lock contention on hot multi-threaded sync clients | Latency on the interception path | O(window≤64) work under a dedicated lock (never `_state_lock`); observe() does no I/O and no logging; if profiling ever shows contention, per-run sharding is a drop-in |
| Two SDK instances (holders) in one run each see only half the traffic | Split loops evade client thresholds | Known v1 limitation; flags from every holder still land on ingested events, so server-side/dashboard correlation catches it; server-side velocity over ingest is future work (out of scope) |
| A run's only activity is one long stream → no admissions → no renewals → kill not learned | Termination delayed up to the stream's own `max_tokens` | Documented (D2); bounded by the call's output cap; acceptable v1 |
| A kill directive silently failing open server-side (the failover-directive attach precedent swallows exceptions, `core budgets.py:1181-1191`) | Operator believes the run is dead while it keeps spending | C-2 mandates a separate fail-loud attach path for `run_control`; test pins that an attach failure errors the check rather than dropping the directive |
| Lease-renew idempotent replay serves a stale response without the directive (`last_response_json`, `core db/models.py:727`) | Kill misses its ≤-one-renewal window | C-2 applies the kill check + directive at the router edge on every outgoing lease response, replayed or fresh; test covers the replayed-renew case |
| New MetadataEvent fields 422 whole 1000-event batches against an undeployed API | Telemetry blackout | API-first ordering enforced by the shared parity gate (`core shared/tests/test_sdk_contract_parity.py:30-39` pending windows) + SDK PRs 4/5 blocked on C-1/C-2 deploys |
| Killed-run state outliving usefulness (Redis flag TTL 7d vs durable columns) | A revived run id after 7 days would not be re-denied by the flag | Durable columns remain the audit record; the SDK's sticky state covers the process lifetime; run ids are ephemeral by design (fresh UUID per `solwyn.run()` scope) — accepted |

## Part 7 — Out of scope (explicit)

- Content fingerprinting/similarity hashing for "near-identical" detection (sizes only in v1; any future fingerprint must live inside the `_privacy.py` allowlist and never leave the process).
- Server-pushed velocity RULE configuration (v2 seam reserved via the directive version field).
- `velocity_flags` on `BudgetCheckRequest` / lease renewals (event channel carries the signal; revisit only for synchronous server auto-kill).
- New `run_control` actions beyond `terminate` (pause, degrade-model ladders — idea 5 territory).
- Identity-aggregated budgets (separate market gap, separate program).
- Durable on-disk receipt spool (rejected for settlements in PJ-2; process-lifetime best-effort + fold aggregates is the posture).
- Cross-process/multi-holder server-side velocity detection.
- TypeScript SDK parity (follows this design in solwyn-ts-sdk after the Python contract settles).
- Framework integrations / kill-switch ergonomics for LangChain etc. (idea 2's program).
- Any adjacent refactors (e.g. client.py deny-site deduplication beyond what the gates require).
- Server-side auto-kill policy (core auto-terminating on ingested velocity flags without a human click) — the directive channel supports it later; v1 keeps the kill decision human.
- Lifetime "saved you" rollups beyond the 365-day `cost_events` retention (`core partition_retention.py:82-90`); the per-run `agent_run.total_denied_cost` aggregate already survives partition drops.
- Fixing the pre-existing `daily_spend_by_project` matview quirk (its `event_count` includes denials, `core 0001_initial.py:2370-2378`) — noted for the core team, not this program's work.
- A deep-linkable per-run dashboard route (`/runs/:runId` — run detail is an expanded table row today, `dashboard agents.tsx:264-308`); the kill button works from the existing panel.
