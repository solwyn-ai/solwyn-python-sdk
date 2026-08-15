# Graduated Budget Pressure + Cost-Aware Degrade Ladder — Execution Plan

> **For agentic workers:** execute stage-by-stage with the superpowers:executing-plans or
> subagent-driven-development skill. Each stage is one PR with its own test cycle. Stages 0–2
> are SDK-only and land in any order after their listed prerequisites; SDK Stage 3 is blocked
> on core PR C1 in the cross-repo section (core repo: `/Users/christian/dev/repos/solwyn-ai/core/`).

**Goal:** Make the server the broadcaster of a budget-pressure signal and per-entry price
hints, and make the SDK shed load within customer-declared comfort bounds — activating the
already-shipped-but-inert `CostPolicy` and never letting a degraded call go untagged.

**Architecture:** Utility demand response. The server owns the signal (pressure tier +
relative price hints, riding existing check/lease responses — zero new round-trips). The edge
owns a *bounded* response: pure reordering of the customer's own declared chain, applied in
`_routing.py`, gated by one config knob and a per-call override. The SDK never computes
price. Prerequisite folded in: per-endpoint health identity, so pressure routing rests on
correct per-endpoint breaker/latency/price state.

**Tech stack:** Existing — Pydantic v2 (`extra="forbid"`), httpx, sans-I/O `_base.py`/
`_routing.py`/`_lease.py`, path-based privacy firewall.

---

## Global constraints (binding, from the project brief + repo invariants)

- The API owns all pricing. The SDK acts on server-sent hints/tiers only. No client-side price tables.
- Wire changes follow the directive-v1 exclude-none posture; both sides are `extra="forbid"`, so **API deploys first** for every new field (the repo's stated rule — `_types.py:72-74`, `_types.py:123-128`; the exclude-none posture is stated at `_types.py:539-548`).
- Same-dialect failover stays native passthrough; cross-dialect stays the minimal translation subset, fail loud. Degrade-to-cheaper respects dialect boundaries via the `translatable` seam (Stage 1).
- Privacy invariant unchanged: only `_privacy.py` + `providers/_translation/` touch content. New routing code sees keys/booleans, never content.
- Business logic sans-I/O in `_base.py`/`_routing.py`. `RuntimeError`, never `assert`. Pydantic v2 only.
- PJ-2 lease design is binding: pressure rides the existing grant/renew responses; the lease protocol is not touched.
- Pre-launch: breaking `CostPolicy`'s silent fallback and re-shaping the never-sent `price_hints` field are both allowed.

---

## Verified starting points (re-checked 2026-08-07; all anchors re-verified against both repos 2026-08-14)

Every claim from the project brief held, and the plumbing is *further along* than the brief assumed:

| Claim | Verified at |
|---|---|
| `CostPolicy.order()` falls back to health order when no hints | `src/solwyn/_routing.py:186-220` (branch at 208-211) |
| README documents the inert state | `README.md:542` |
| `price_hints` / `failover_directive` slots on `BudgetCheckResponse` | `src/solwyn/_types.py:550-560` |
| Check requests send the element-aligned chain | request model `src/solwyn/_types.py:491-499`, builder `src/solwyn/budget.py:246-280`, client sends `self._runtimes[1:]` at `src/solwyn/client.py:1513-1514` (async: 2627-2628) |
| Server-directive application pattern (lock, suppression log) | `src/solwyn/_base.py:1325-1382` |
| Same-name health-domain limitation | `README.md:225` |

**Beyond the brief — already built, waiting for the server:**

- `_SolwynBase.update_price_hints()` stores hints (`_base.py:1439-1447`); the client applies `result.price_hints` after every check at **four** call sites (`client.py:1303-1304`, `1550-1551`, `2447-2448`, `2652-2653`); `_select_candidates` snapshots hints onto candidates (`_base.py:1464-1488`); a one-time "CostPolicy inactive" warning fires when it degrades (`_base.py:155-170`, `1492-1497`). **CostPolicy self-activates on the first non-null hint with zero SDK changes.** The capability-negotiation seam the brief asked about already exists end to end.
- Plumbing tests exist: `tests/unit/test_routing.py:278-425` (CostPolicy ordering laws incl. scale invariance), `tests/unit/test_routing_policy_swap.py:278+` (server hint → result → candidate → reorder).
- The renewal request builder already accepts an optional chain re-declaration (`_lease.py:879-908`, request model `_types.py:677-692`) — the hook Stage 3 needs.

**Identity collapse mechanics (the prerequisite bug), precisely:**

- One breaker per *name*: `_base.py:1474` (`self._get_circuit_breaker(runtime.adapter.name)`); latency windows and price hints also name-keyed (`_base.py:1414-1422`, `1488`).
- `is_provider_fallback = rt.entry.provider != primary.entry.provider` (`client.py:1638`) drives *both* wire labeling *and* sanitization (`_build_hop_kwargs` branch select `client.py:633`, endpoint-scoped stripping only in the cross-provider branch `client.py:670-723`; `prepare_streaming(cross_provider=...)` `client.py:1668-1671`). Two same-name entries therefore share health state, are labeled model fallbacks of each other, and skip sanitization between hops — exactly as README:225 says.
- `ProviderName` is a **closed StrEnum** (`_types.py:62-97`) used on every wire model. This kills the "enforced explicit naming" option (see Q3): two Azure resources have no second legal name to take.

---

## Chosen design

**The tier ladder** (server-sent tier; SDK behavior bounded by the customer's declared chain):

| Tier | Server signals at | SDK response (within comfort bounds) |
|---|---|---|
| *(absent)* | < 80% | Configured policy, untouched |
| `elevated` | ~80% | Fallback entries reorder cost-first among themselves; the front-of-order candidate keeps its slot |
| `critical` | ~95% | Full cost-first: the cheapest usable declared entry serves first, displacing the primary |
| *(deny)* | 100% | Existing hard-deny path, unchanged |

- Health always dominates price: pressure reorders **within** health tiers using the existing `_health_key` (`_routing.py:110-125`); it never promotes an unhealthier or untranslatable candidate, never introduces an entry that isn't in the configured chain.
- If the user already selected `CostPolicy`, `elevated`/`critical` add nothing but the event tags (they are already cost-first) — consistent by construction.
- Every call checked under pressure is tagged, even when the response level is `off`.

**Comfort bounds:** one config knob (max degradation the customer permits) + one per-call override:

```
budget_pressure_response: "off" | "reorder" | "prefer_cheaper"   (default "reorder")
solwyn_degrade: bool  (per-call kwarg; False → "off" for this call, True → "prefer_cheaper" ceiling)
```

The declared chain is the hard bound at every level — pressure can only prefer entries the
customer already declared acceptable by putting them in the chain.

**Signal transport:** call-scoped, no new global mutable state. The tier arrives on the
`BudgetCheckResult` for *this* call (check path: mapped from the response; lease path: the
ledger stores the last grant/renew signal per run and stamps it onto the synthesized result).
The client threads it into `RoutingRequest`. No cross-call tier state in `_SolwynBase` at all
— staleness and locking questions vanish.

---

## Settled questions

### Q1 — Who owns the thresholds? **Server-defined tiers. The edge never maps utilization.**

- Evidence: the SDK already has the vocabulary for "the server resolves which period binds" — `denied_by_period` (`_types.py:539-548`); a raw utilization number is ill-defined when daily, agent-run, and tag periods coexist, and the server is the only party that knows which one is under pressure.
- Tuning thresholds (80/95, per-plan overrides, hysteresis) without an SDK redeploy is the whole point of "server owns the signal"; edge-mapped thresholds would fragment behavior across SDK versions in one fleet.
- Demand-response analogy holds exactly: the utility broadcasts the event tier, not its reserve margin; the thermostat doesn't recompute grid state.
- The response carries `period` and `utilization` as **observability payload only** — the SDK logs/tags them, never branches on them. The tier→behavior mapping is the edge's bounded response, constrained by `budget_pressure_response`.
- Hysteresis is server-side (recommended to core: enter elevated ≥80% / exit <75%; enter critical ≥95% / exit <90%) so ordering can't thrash around a boundary. The SDK needs no debouncing because the tier is call-scoped.

### Q2 — Comfort bounds. **The declared chain is the bound; one knob sets how far pressure may go; per-call escape hatch; no cheaper entry → loud no-op.**

- Declaring an entry in `fallback=` already means "this (provider, model) may serve my calls" — failover serves it today without asking. Pressure routing only re-prefers within that same set, so no new declaration surface (chain annotations, per-run manifests) is needed for v1. Rejected: per-entry annotations (`degrade_ok=True`) — extra config surface with no v1 semantics the knob doesn't cover; revisit if customers ask for mixed chains.
- Knob default `"reorder"`: reordering *fallbacks* changes only which already-acceptable entry serves when the front-of-order fails — mild, and the out-of-box behavior demos the feature. Displacing the customer's chosen primary model (`"prefer_cheaper"`) is a quality decision only the customer should make, so it is opt-in. **[Flagged for review — see Decisions for Christian]**
- Per-call override `solwyn_degrade` follows the exact `solwyn_idempotent` precedent (popped before dispatch, `client.py:1586-1592`) — lets a team mark quality-critical calls undegradable inside an otherwise-enrolled app.
- No cheaper entry declared (single-entry chain, or no hints yet): the transform is a no-op — mirrors `CostPolicy`'s existing no-hints branch (`_routing.py:208-211`). The event still carries `budget_pressure`, so the dashboard can show "pressure signaled, nothing declared to shed to" — that nudge is the product surface for getting chains declared.
- Media surfaces: no failover chain exists by design (`README.md:231`) — tier is tagged on media events, routing untouched.

### Q3 — Health-domain fix shape. **Default per-endpoint identity (derived), not enforced explicit naming.**

- Enforced explicit naming is structurally impossible: identity on the wire is the closed `ProviderName` enum (`_types.py:62-97`) and an override may only relabel *within* the known names of the same dialect (`_registry.py:40-73`). Two Azure resources are both, correctly, `azure_openai` — there is no second name to demand.
- Design: `ProviderRuntime` grows a derived, process-local `endpoint_key`. Entries with the same adapter name **and** the same endpoint fingerprint share a key (the legitimate same-endpoint model-fallback case keeps one health domain); same name + different fingerprint get distinct keys. Fingerprint = normalized `base_url` (scheme/host/port, trailing-slash-insensitive) when the client exposes one, else Bedrock's `client.meta.region_name`+endpoint, else `id(sdk_client)`. Duck-typed, no provider imports, no content.
- Blast radius, enumerated:
  - **Local keying (Stage 0, no wire change):** breakers (`_base.py:1474`), latency windows (`client.py:1840` → `_base.py:1414`), price-hint lookup (`_base.py:1488`), the walk's `failed_providers` double-count guard (`client.py:1622`, `1751-1753`), `ProviderUnavailableError.attempted` lists.
  - **Sanitization (Stage 0):** the dispatch computes `is_endpoint_fallback = rt.endpoint_key != primary.endpoint_key` and feeds *that* to `_build_hop_kwargs` and `prepare_streaming` — a same-name different-endpoint hop now takes the existing same-dialect cross-provider branch (`client.py:670-723`; since PR #64 it includes a Google-target early return at 693-699 that also strips endpoint-scoped keys): endpoint-scoped keys stripped, target `default_params` re-applied, output-cap key normalized. Wire labeling is untouched in Stage 0.
  - **Breaker reports (interim):** two same-name endpoints produce two breakers but `BreakerStateReport.provider` is one enum slot (`_types.py:187-201`). Stage 0 reports the *most severe* state per name (OPEN > HALF_OPEN > CLOSED) with summed counts — documented interim, made precise by `endpoint_label` in Stage 3.
  - **Failover labeling (Stage 3, wire):** events gain optional `is_endpoint_fallback: bool | None` and `endpoint_label: str | None` (label = `"{name}#{chain_index}"` — never a base_url, which can embed internal hostnames). `is_provider_fallback` keeps its exact current wire meaning (served *name* ≠ requested *name*), so nothing existing drifts.
- False-split risk (same endpoint spelled two ways) costs a duplicated breaker domain — mild; false-merge (today's bug) costs mislabeled fallbacks and skipped sanitization that can 4xx a hop (`README.md:225`). Normalization plus preferring `base_url` over object identity keeps false-splits rare.

### Q4 — Loud tagging. **Tier on every event of a pressured call; a distinct failover reason when pressure changed the serving entry; confirm carries the tier so settled spend is attributable.**

- `MetadataEvent` (`_types.py:304-432`) gains `budget_pressure: Literal["elevated","critical"] | None` and `pressure_degraded: bool | None`; `FailoverReason` (`_types.py:107-112`) gains `BUDGET_PRESSURE = "budget_pressure"`. All None-defaulted — the None-dropping serializer (`_types.py:314-325`) keeps every unpressured event's wire bytes unchanged.
- Tagging rules:
  - `budget_pressure=<tier>` on **every** event built for a call whose check result carried a tier — success, error, and budget_denied — regardless of `budget_pressure_response` (visibility even when the customer opted out of action).
  - `pressure_degraded=True` + `failover_reason=BUDGET_PRESSURE` on the served hop when (a) a tier was active with response level ≥ reorder, (b) the served candidate preceded the counterfactual normal-order first pick in the pressure ordering, and (c) that counterfactual pick was usable and had not already been attempted-and-failed this walk. If the primary errored and the walk fell to the cheap entry, that stays `PRIMARY_ERROR` — pressure gets credit only when pressure, not failure, moved the call. The counterfactual is one extra pure `policy.order()` call, computed only when a tier is active.
- `BudgetConfirmRequest` (`_types.py:817-922`) gains `budget_pressure` with the same literal, so every settled spend row is pressure-attributable on the dashboard without a join through events.
- The 100% tier needs nothing new: `CallStatus.BUDGET_DENIED` events already exist (`client.py:1307-1324`).

### Q5 — Sequencing. **Three SDK-only PRs land now behind the existing seam; the contract stages are a strict API-first lockstep (core C1 → SDK 3 → core C2 → core C3).**

- The SDK can land, fully tested, before core ships anything: Stage 0 (endpoint identity), Stage 1 (translatability predicate), Stage 2 (pressure machinery — inert because no response ever carries a tier, exactly as `CostPolicy` is inert today, and the hint plumbing already self-activates: `client.py:1303-1304` → `_base.py:1488` → `_routing.py:212-220`).
- Both wire directions are `extra="forbid"` (`_types.py:529`, and core "forbids unknown keys" per `_types.py:647-649`), so lockstep is forced in *both* directions: the SDK may not **send** a new request/event field until core accepts it, and core may not **emit** a new response field except gated on the SDK's opt-in marker. Hence the marker pattern (`budget_pressure_version="1"`, mirroring `failover_directive_version` at `_types.py:513-516` / `budget.py:279`).
- Exact order: **core C1** (accept markers + new event/confirm/breaker fields) → **SDK 3** (send marker, parse new response fields, populate labels/tags, grow live-contract tests) → **core C2** (emit hints + pressure, gated on the marker) → **core C3** (dashboard). At C2 deploy time, every already-deployed SDK ≥ Stage 3 activates with zero redeploy — the credibility story ("`CostPolicy` went live without a client release") is the sequencing's own proof.
- The lockstep is not just convention — it is *enforced* in the core repo: `shared/tests/test_sdk_contract_parity.py` pins the SDK's vendored wire models against `solwyn_shared` and requires any server-leading field to be registered in an explicit `_PENDING_SDK_*` set naming the SDK change that closes it (`core/shared/tests/test_sdk_contract_parity.py:13-37`). C1 arms those pending sets; SDK Stage 3 closes them.

---

## Wire-contract spec (exact; all response fields exclude-none with explicit `None` defaults per repo convention)

**New shared model (SDK vendored copy in `_types.py`, lock-step with core):**

```python
PressureTier = Literal["elevated", "critical"]

class BudgetPressure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tier: PressureTier
    period: str = Field(..., max_length=64)   # same vocabulary as denied_by_period ("daily", "agent_run", ...)
    utilization: float = Field(..., ge=0)     # observability only; the SDK never branches on it
```

Omission semantics: the whole `budget_pressure` object is **absent** below the elevated
threshold and on servers that predate the feature. Absent = no pressure. The SDK treats the
two identically (there is no "unknown" state to represent).

**Request models** (`BudgetCheckRequest`, `LeaseGrantRequest`, `LeaseRenewRequest`): add

```python
budget_pressure_version: Literal["1"] | None = Field(default=None)   # opt-in marker, None-omitted
```

**`BudgetCheckResponse`:** add `budget_pressure: BudgetPressure | None = None`; **replace**
`price_hints: dict[ProviderName, float] | None` with

```python
chain_price_hints: list[float | None] | None = Field(default=None)
```

element-aligned with `[request.(provider, model)] + zip(request.fallback_providers,
request.fallback_models)`; length MUST equal `1 + len(fallback_providers)` (SDK validates,
discards-with-warning on mismatch); values are relative, dimensionless, scale-free (the
ordering-invariance law is already tested at `test_routing.py:355`); `None` element = server
can't price that entry. Rationale for replacing the dict: it is keyed by `ProviderName` and
therefore cannot distinguish two same-name endpoints (Q3) **or** two models on one provider —
and the 95% tier is precisely about model-level preference. The dict was never emitted by any
server, so this is a zero-cost pre-launch correction. **[Flagged for review — Decisions]**

**`LeaseGrantResponse`:** add `budget_pressure: BudgetPressure | None = None` and
`chain_price_hints: list[float | None] | None = None`, aligned to the chain **as sent in the
triggering grant/renew request**; omitted when that request carried no chain. (The SDK makes
renewals always re-declare the chain — the builder already takes it, `_lease.py:879-908`.)

**`MetadataEvent`:** add `budget_pressure: PressureTier | None`, `pressure_degraded: bool | None`,
`is_endpoint_fallback: bool | None`, `endpoint_label: str | None` (max_length 80). `FailoverReason`
adds `BUDGET_PRESSURE = "budget_pressure"`.

**`BudgetConfirmRequest`:** add `budget_pressure: PressureTier | None`.

**`BreakerStateReport`:** add `endpoint_label: str | None` (max_length 80).

---

## Stages

### Stage 0 — PR 1 (SDK-only): per-endpoint health identity

**Files:** modify `src/solwyn/_registry.py`, `src/solwyn/_base.py`, `src/solwyn/client.py`,
`README.md:225`; create `tests/unit/test_endpoint_identity.py`.

**Produces (later stages consume):** `ProviderRuntime.endpoint_key: str` and the dispatch-local
`is_endpoint_fallback` boolean.

1. `_registry.py`: add `_endpoint_fingerprint(client) -> str` (normalized `base_url` →
   bedrock `meta.region_name`+endpoint → `f"obj:{id(client)}"`; duck-typed, content-free) and
   compute `endpoint_key` during `build_runtimes`: `adapter.name` when the name is unique in
   the chain or all same-name entries share a fingerprint; else `f"{name}#{chain_index}"`
   with same-fingerprint entries sharing the first-seen key (model fallback on one endpoint
   keeps one domain).
2. `_base.py`: key `_circuit_breakers`, `_latency_windows`, and the candidate hint lookup by
   `endpoint_key` (constructor loop `_base.py:826-829`, `_get_circuit_breaker` call site
   `_base.py:1474`, `record_latency`/`observed_p50` callers). `_get_breaker_snapshots` merges
   per name for the wire: most severe state, summed counts (interim until Stage 3).
3. `client.py`: compute `is_endpoint_fallback = rt.endpoint_key != primary.endpoint_key` in
   both walks (sync `client.py:1638` region, async `~2734`); pass it (not the name-based
   flag) to `_build_hop_kwargs` and `prepare_streaming`; key `failed_providers` and
   `attempted` by `endpoint_key`; wire event fields unchanged this stage.
4. README:225: rewrite the limitation paragraph to the new behavior (present tense, per the
   snapshot-not-history convention); note the one remaining wire-labeling gap until Stage 3.

**Tests:** two same-name compat entries (different base_urls) get distinct breakers — opening
one leaves the other CLOSED; a hop between them strips `extra_headers`/`stream_options` and
re-applies the target's `default_params` (the README:225 4xx scenario); same client object
twice (model fallback) keeps one shared breaker and native passthrough (no stripping);
trailing-slash/case base_url variants share a fingerprint; breaker report for a collided name
carries the most severe state. Run `make check && make test`.

Commit: `fix(sdk): per-endpoint health domains and hop sanitization for same-name chain entries`

### Stage 1 — PR 2 (SDK-only): translatability predicate (the routing seam's "fix [H]")

**Why in scope:** the moment hints arrive, cost-first ordering can promote a *cross-dialect*
entry to the front. If the request can't ride the translation subset, `_build_hop_kwargs`
raises and **aborts the whole chain before dispatch** (`client.py:1650-1678`) — a call that
would have succeeded on the primary now fails because we tried to save money. The
`ProviderCandidate.translatable` field and the `not translatable` sort term exist precisely
for this (`_routing.py:39-47`, `110-125`) and are currently hardwired `True`.

**Files:** modify `src/solwyn/providers/_translation/` (new probe module), `src/solwyn/_base.py`
(`_select_candidates` signature), `src/solwyn/client.py` (both walks); extend
`tests/unit/test_routing.py`, translation tests.

1. `providers/_translation/probe.py`: `def can_translate(source_dialect: str, target_dialect: str, kwargs: Mapping[str, object], *, is_streaming: bool) -> bool` —
   structural inspection only (key presence, `tools`/`response_format`/streaming shape); same
   checks `to_canonical` + the tool-stream gate (`client.py:772-779`) would fail on, without
   raising and without touching values. Content-privileged module: allowed to see the dict,
   must never log or store it. **Uncertain → return `True`** (fail into today's eager-abort
   behavior, never silently demote a translatable target).
2. `_base._select_candidates` gains keyword-only `translatable: Callable[[ProviderRuntime], bool] | None = None`
   (None → all True); populates the candidate field (`_base.py:1481`).
3. Both dispatch walks pass a closure: same-dialect targets → `True`; cross-dialect →
   `can_translate(primary.adapter.dialect, rt.adapter.dialect, kwargs, is_streaming=...)`.
   The eager-abort in `_build_hop_kwargs` stays as the authoritative backstop.

**Tests:** a tool-using streamed request sinks the cross-dialect candidate below same-dialect
ones under `CostPolicy` even when it is cheapest (existing sink law `test_routing.py:102`
extended); probe returns `True` on unknown shapes; probe never raises; dispatch still
eager-aborts if an untranslatable target is reached anyway; privacy-firewall test still green
(probe lives inside the allowlist path).

Commit: `feat(sdk): per-target translatability predicate feeding the routing seam`

### Stage 2 — PR 3 (SDK-only): pressure machinery, inert behind the seam

**Files:** modify `src/solwyn/_types.py` (SDK-side models: `PressureTier`, `BudgetPressure`,
event/confirm fields, `FailoverReason.BUDGET_PRESSURE`), `src/solwyn/_routing.py`,
`src/solwyn/_base.py`, `src/solwyn/budget.py` (`BudgetCheckResult.budget_pressure`),
`src/solwyn/config.py`, `src/solwyn/client.py`; create `tests/unit/test_budget_pressure_routing.py`.

1. `config.py`: `budget_pressure_response: Literal["off", "reorder", "prefer_cheaper"] = "reorder"`
   + env `SOLWYN_BUDGET_PRESSURE_RESPONSE` in the field map.
2. `_routing.py`: `RoutingRequest` gains `pressure_tier: PressureTier | None = None` (the
   docstring already declares the dataclass additive-only, `_routing.py:61-68`); new pure
   function:

   ```python
   def apply_budget_pressure(
       ordered: list[ProviderCandidate],
       *,
       tier: PressureTier | None,
       response_level: Literal["off", "reorder", "prefer_cheaper"],
   ) -> list[ProviderCandidate]:
   ```

   Laws: no-op when tier is None, level is "off", or no candidate carries a hint (mirror of
   `_routing.py:208-211`); `elevated` (or `critical` capped at level "reorder") re-sorts
   positions 1..n by `(*_health_key(c), hint-or-inf)` leaving position 0 fixed; `critical`
   at level "prefer_cheaper" re-sorts the whole list by the same key. Stable sort keeps the
   policy's order for ties; health tier and translatability stay dominant by key construction;
   the function only reorders — never adds, never drops.
3. `_base._select_candidates`: after `self._policy.order(...)` (`_base.py:1491`), apply
   `apply_budget_pressure(ordered, tier=req.pressure_tier, response_level=<effective level>)`
   before the identity filter. The price-signal gate (`_base.py:1462-1471`) also computes hints
   when a tier is active, not only when the policy declares `uses_price_signal`.
4. `budget.py`: `BudgetCheckResult` gains `budget_pressure: BudgetPressure | None = None`
   (populated in Stage 3; the cache path will carry it from the cached response so a ≤5s-TTL
   cached allow keeps its tier).
5. `client.py` (both walks + media path): pop `solwyn_degrade` beside `solwyn_idempotent`
   (`client.py:1586-1592`); resolve the effective response level (call override beats
   config); pass `pressure_tier=budget.budget_pressure.tier if ... else None` into
   `RoutingRequest` (`client.py:1597-1602`); when a tier is active, compute the counterfactual
   normal order once and thread `(tier, degraded, reason)` into `_build_metadata_event` /
   `_build_error_event` / the settlement confirm per the Q4 rules; log one WARNING per client
   per tier escalation (`budget pressure {tier}: {period} at {utilization:.0%}, response={level}`),
   following the suppression-log pattern (`_base.py:1378-1382`).
6. Event/confirm fields are None-defaulted → the serializer drops them; they cannot populate
   until a server sends a tier, so **no wire bytes change from this PR** (assert exactly that
   in tests).

**Tests (pure, table-driven where possible):** ordering laws above; off/override gating;
single-entry chain no-op; tier tagged on events even at level "off"; `BUDGET_PRESSURE` reason
only when the counterfactual pick was usable and unfailed (PRIMARY_ERROR wins when the primary
errored); serialized unpressured event/confirm byte-identical to before; config env parsing.
Extend `test_wire_model_tags.py`-style serialization checks.

Commit: `feat(sdk): graduated budget-pressure routing behind the server-signal seam`

### Stage 3 — PR 4 (SDK, blocked on core PR C1): speak the contract

**Files:** modify `src/solwyn/_types.py` (request markers; `chain_price_hints` replacing
`price_hints`; `LeaseGrantResponse` additions; `BreakerStateReport.endpoint_label`),
`src/solwyn/budget.py`, `src/solwyn/_lease.py`, `src/solwyn/_base.py`, `src/solwyn/client.py`;
extend `tests/integration/test_live_contract.py`.

1. Requests: `_build_check_request` (`budget.py:246-280`) and the lease grant/renew builders
   set `budget_pressure_version="1"`; renewals always re-declare the current chain (call-site
   change; the builder already accepts it). The vendored-model edits in this PR are what
   close core's armed parity pending-sets (`core/shared/tests/test_sdk_contract_parity.py:13-37`)
   — core's suite fails loudly until the entries are removed, so land the core-side
   pending-set removal in lockstep.
2. Responses: map `budget_pressure` + `chain_price_hints` in `_build_result_from_response`
   (`budget.py:427-500`). The internal carrier changes shape with it — this PR owns **all**
   the existing price-hint plumbing sites:
   - `BudgetCheckResult.price_hints: dict[str, float] | None` (`budget.py:143`) becomes
     `chain_price_hints: list[float | None] | None` (positional; the enforcer knows the chain
     it sent but not the client's endpoint keys).
   - The four client apply-sites (`client.py:1303-1304`, `1550-1551`, `2447-2448`,
     `2652-2653`) zip the positional list against `self._runtimes` and call
     `update_price_hints({runtime.endpoint_key: hint})`; `update_price_hints`
     (`_base.py:1439-1447`) keeps its signature but its docstring re-keys to endpoint keys
     (`_base.py:1488` already reads by the Stage 0 key).
   - Length mismatch → warn once, discard hints, never guess.
   - Update the plumbing tests that pin the old dict shape
     (`tests/unit/test_routing_policy_swap.py:278+`).
3. Lease path: `LeaseLedger` stores the last `BudgetPressure` and hint list per run when a
   grant/renew response applies (`apply_grant_response`, called under `_state_lock` from
   `budget.py:1070-1077` on renew, `budget.py:698` on grant — sans-I/O storage fits the ledger's contract); `_check_lease`
   stamps them onto the synthesized `BudgetCheckResult`, and the client's existing
   `if budget.price_hints is not None: update_price_hints(...)` seam (`client.py:1550-1551`)
   keeps working unchanged for leased runs.
4. Populate `is_endpoint_fallback` / `endpoint_label` on events and `endpoint_label` on
   breaker reports (removing the Stage 0 merge-worst interim); same-name endpoint hops keep
   `is_provider_fallback=False` on the wire but now carry the endpoint truth.
5. Live-contract tests (pattern: `TestLiveFailoverDirectiveContract`,
   `test_live_contract.py:300-322`): marker-carrying check gets `budget_pressure` omitted on
   a low-utilization project; elevated/critical fixture projects (core C2 item 5) return the
   block with `tier`/`period`/`utilization`; `chain_price_hints` length equals `1 + len(fallback_providers)`
   and round-trips `None` elements; grant and renew responses carry the same blocks; ingest
   accepts events with every new field populated; confirm with `budget_pressure` settles.

Commit: `feat(sdk): budget-pressure + chain-price-hint wire contract (directive-v1 posture)`

### Stage 4 — PR 5 (SDK, blocked on core PR C2): end-to-end activation + docs

1. Integration tests driving the ladder live: a call on the elevated fixture serves the
   cheaper declared entry under `prefer_cheaper` and tags it; `CostPolicy` selected + hints
   present → no inactive-warning, cost order served; 100% still hard-denies.
2. `README.md`: delete the line-542 inert paragraph; document the ladder, the knob, the
   per-call override, and the tagging contract; config-table rows for the new knob; update
   `src/solwyn/CLAUDE.md` module map lines that this plan changed (present-tense snapshot).
3. CHANGELOG entry; verify `_warn_cost_policy_inactive_once` still fires for hint-less
   servers and its message mentions the server-side activation story.

Commit: `docs(sdk): budget pressure ladder + CostPolicy activation`

---

## Cross-repo execution: Core API + dashboard

Investigated directly in `/Users/christian/dev/repos/solwyn-ai/core/` (paths below are
relative to that repo). Ground facts the stages build on:

- **The directive-v1 mechanism is exactly as the SDK assumes, and it lives in the router.**
  `api/src/solwyn_api/routers/budgets.py:1122-1209` (`budget_check`): the request's
  `failover_directive_version` opt-in selects `exclude_none=opted_in_to_directive` on
  `model_dump` and gates the directive via a `response_fields` include-set; non-opted
  requests get the byte-stable legacy response (1198-1201). New optional response fields are
  backward-safe **only** through this gate — that is where `budget_pressure` and
  `chain_price_hints` attach or get stripped.
- **Entitlement gating has a worked pattern:** `get_tier_policy(auth.subscription_tier)` →
  `policy.capabilities.failover_entitlement` (`budgets.py:1178-1183`), with
  `services/tier_policy.py` as Python source of truth bridged to the dashboard mocks by the
  committed `shared/tier-capabilities.json` (two-sided drift tests, per that file's header).
- **All the numbers pressure needs are already in scope at the check decision.**
  `services/budget_aggregator.py:_check_budget_impl`: the allow return (3600-3611) has
  `primary_new_usd` / `budget_limit`, and the Lua result is the *minimum remaining across all
  periods* (3521-3528); per-period previous/new usage — including the sub-period read at
  3553-3562 and scoped/run counters via `reserve_observations` — already feeds the
  threshold-crossing machinery (`_check_thresholds`, 3538-3574). Pressure is a pure function
  of values this method already holds; no new counter reads.
- **Lease grant/renew have the same numbers at their response build:**
  `services/budget_lease_service.py:2530-2538` (`LeaseGrantResponse(... budget_limit,
  current_usage=micros_to_usd(current_micros))`), periods context at 1476-1484; `grant` at
  845, `renew` at 1007.
- **The price-hint source exists:** `services/pricing_service.py:
  worst_case_token_rate_micros_per_mtok` (793-842) already computes a conservative scalar
  rate per (provider, model) — context brackets probed at both ends, Bedrock regional
  overrides folded, worst service-tier multiplier applied — returning `None` for
  unit-priced/unknown/zero-rate models. That `None` maps 1:1 onto `chain_price_hints`'
  per-element `None` semantics.
- **Event persistence + surfacing paths are known:** ingested events map failover fields at
  `services/metadata_ingest.py:2122-2131` and `2404-2413` into `CostEvent` columns
  (`db/models.py:600-605`); the dashboard renders them in
  `dashboard/src/components/provider-failover-feed.tsx:90-114` (requested provider +
  `failover_reason`) and filters them in `cost-events-table.tsx:164` (`model_fallback`
  option), fed by `routers/costs.py:979` (`list_cost_events`) / `1356` (`query_costs`).
  SPA types are generated (`make openapi` → `shared/openapi.json` →
  `dashboard/src/api/generated/paths.ts`; CI `make openapi-check`).
- **Confirmed absence:** no code under `api/src` populates `price_hints` — the field exists
  only on the shared model (`shared/src/solwyn_shared/models.py:522`). The inert-slot story
  checks out server-side too.

### Core PR C1 — accept the contract (unblocks SDK Stage 3)

**Files:** `shared/src/solwyn_shared/models.py`, `shared/tests/test_sdk_contract_parity.py`,
`shared/tests/test_contract_snapshot.py`, `api/src/solwyn_api/services/metadata_ingest.py`,
`api/src/solwyn_api/db/models.py` + new `api/migrations/versions/`, `api/src/solwyn_api/routers/budgets.py`
(confirm + breaker-report acceptance), `api/tests/unit/routers/test_budgets.py`.

1. `shared` models: add `BudgetPressure` + `PressureTier`; add `budget_pressure_version`
   markers to `BudgetCheckRequest` (beside `failover_directive_version`, models.py:488),
   `LeaseGrantRequest`, `LeaseRenewRequest`; **replace** `price_hints` (models.py:522) with
   `chain_price_hints` per the wire spec; add `budget_pressure`/`chain_price_hints` to
   `BudgetCheckResponse` and `LeaseGrantResponse`; add the event/confirm/breaker fields and
   the `FailoverReason.BUDGET_PRESSURE` value.
2. Arm the parity pending sets (`test_sdk_contract_parity.py:13-37` pattern; the sets
   already carry other features' entries — `"tags"`, `"parent_agent_run_id"` — add
   alongside, don't replace) naming SDK Stage 3 as the closer; refresh the contract
   snapshot test.
3. Ingest: persist `budget_pressure` / `pressure_degraded` / `is_endpoint_fallback` /
   `endpoint_label` — new nullable `CostEvent` columns (migration), mapping added at the two
   ingest sites (2122-2131, 2404-2413). Confirm handler accepts `budget_pressure`; breaker
   report route accepts `endpoint_label`. No emission anywhere yet.
4. Tests: router accepts marker-carrying requests byte-stably; ingest round-trips the new
   fields; migration up/down; `make openapi` regenerated (response models unchanged so far
   for the dashboard).

### Core PR C2 — emit the signal (activates every deployed SDK ≥ Stage 3)

**Files:** `api/src/solwyn_api/services/budget_aggregator.py`,
`api/src/solwyn_api/services/budget_lease_service.py`, `api/src/solwyn_api/services/pricing_service.py`,
`api/src/solwyn_api/routers/budgets.py`, `api/src/solwyn_api/services/tier_policy.py` +
regenerated `shared/tier-capabilities.json`, tests.

1. **Hints:** add a per-element variant beside `worst_case_token_rate_micros_per_mtok`
   (`pricing_service.py:793-842`) — same card logic, but returning
   `list[int | None]` for an ordered `[(provider, model), ...]` chain instead of the max.
   Normalize to relative (divide by the minimum positive rate) so the wire value stays
   dimensionless; unknown/unit-priced entries → `None`. Attach in the `budget_check`
   directive block (`budgets.py:1172-1209`), gated on `budget_pressure_version`, aligned to
   `[body.provider/model] + zip(body.fallback_providers, body.fallback_models)`.
2. **Pressure tier:** compute in `_check_budget_impl` where per-period usage already exists
   (allow path 3521-3574; also set on the alert-only-deny response so alert-mode projects
   feel pressure before their soft cap): utilization per period from the same observations
   the threshold machinery reads; tier = highest tier any period crosses; `period` label
   reuses the `denied_by_period` vocabulary. Hysteresis (enter 80/exit 75, enter 95/exit 90)
   needs last-signaled state: one Redis key per (project, period) with the existing period
   TTL helper (`get_period_ttl`, aggregator:1676). Aggregator sets `budget_pressure` on
   `BudgetCheckResponse`; the router's `response_fields` gate strips it for non-opted
   requests exactly like `failover_directive`.
   **Failure posture (explicit, learned from the directive attach handler):** the existing
   directive attach swallows exceptions with only a log line (`budgets.py:1187-1197`) — a
   serialization bug there ships silently. Pressure and hints keep the *fail-soft* half of
   that posture (omitting the block degrades to normal-tier / health-order routing, which is
   the safe default and honors never-block) but NOT the silent half: each attach failure
   increments a dedicated counter metric (`budget_pressure_attach_errors_total`,
   `chain_price_hints_attach_errors_total`) beside the WARNING, so a shipped bug is visible
   on ops dashboards, not just in grepped logs.
3. **Lease parity — attach at the router edge, never inside the service.** The lease
   service persists the last grant/renew response verbatim and replays it byte-identically
   when a holder retries (`db/models.py:776-778` `last_response_json`; replay paths
   `budget_lease_service.py:891-902` and `1035-1038`). A pressure tier or hint baked into that
   blob goes stale on every replay. So: extract the tier computation into a helper shared
   with item 2 (one hysteresis-state namespace), and have the lease **routers**
   (`budgets.py:1647` grant, `1686` renew) stamp `budget_pressure` + `chain_price_hints`
   onto the service's response via `model_copy(update=...)` after it returns — fresh on
   replays, gated on the request's `budget_pressure_version`, and the persisted blob stays
   byte-identical for the fencing contract. Hints align to the chain in the triggering
   request (the router has the body); omitted when the request re-declared none. Same
   fail-soft-plus-metric posture as item 2.
4. **Entitlement:** add a `budget_pressure` capability to `tier_policy.py`, regenerate
   `tier-capabilities.json` (its two-sided drift tests force the dashboard mocks to follow).
   Unentitled plans simply never get the block — plan gating is emission-side.
5. **Fixtures for the SDK live suite:** projects pinned at elevated/critical utilization
   (the hard-deny fixture precedent exists — SDK `tests/integration/test_live_contract.py:586`).
6. Explicit non-coupling: customer threshold alert rules (percent thresholds,
   `budgets.py:168-180`) stay a separate notification feature; pressure tiers are fixed 80/95
   server defaults in v1. Making tiers per-project-configurable can reuse the `BudgetRule`
   grammar later — out of scope now.

### Core PR C3 — dashboard visibility + comfort-bounds surface

**Files:** `dashboard/src/pages/projects/detail/budget.tsx`, `providers.tsx`,
`dashboard/src/components/budget-gauge.tsx`, `cost-events-table.tsx`,
`provider-failover-feed.tsx`, `feature-gate.tsx` (pattern), `routers/budgets.py`
(`BudgetStatusResponse`, :234-255 / `budget_status`, :1752), `routers/costs.py` (:979, :1356),
regenerated `paths.ts` in both SPA pipelines.

1. **Pressure state:** extend `BudgetStatusResponse` with the current
   tier/period/utilization (computed the same way as C2, read-only) → pressure chip beside
   the `BudgetGauge` on `budget.tsx`, tier-gated via the `feature-gate` pattern.
2. **Degradation attribution:** expose the new event columns through `list_cost_events` /
   `query_costs`; add a "Budget pressure" filter option beside `model_fallback`
   (`cost-events-table.tsx:164`), a degraded badge on rows with `pressure_degraded`, and a
   `budget_pressure` reason label in `provider-failover-feed.tsx` (90-114). A degraded run is
   then attributable from either surface — the SDK-side tagging rules (Q4) guarantee the
   fields are present whenever degradation happened.
3. **Comfort-bounds visibility (not declaration):** the comfort bound itself is SDK config by
   design (Q2 — the declared chain + one knob, both in code, like every routing knob per SDK
   `README.md:422`). The dashboard's job is the nudge: on the budget page, when pressure was
   signaled in the selected window but **zero** events carry `pressure_degraded`, render
   "pressure was signaled; no cheaper fallback entries were declared/allowed to act" —
   computable from cost events alone, no new persistence.
4. `npm run codegen` both SPAs; `make openapi-check` green.

### Parallel-plan coordination (runaway-protection session)

Both this plan and the runaway-protection plan extend the same channel: new opt-in-gated
optional fields on check/lease responses, built in the same router block
(`budgets.py:1172-1209`) and the same shared models file — and the aggregator already hosts
runaway machinery (`_RunawayCandidate` etc., `budget_aggregator.py:392-467`) adjacent to
where C2's pressure computation lands. Non-collision rules adopted here, to be mirrored
there:

- Field namespaces are disjoint: this plan claims `budget_pressure`, `budget_pressure_version`,
  `chain_price_hints`, `pressure_degraded`, `is_endpoint_fallback`, `endpoint_label`, and the
  `FailoverReason` value `budget_pressure`. A terminate directive should claim `terminate_run*`
  names under its **own** opt-in marker — neither plan bumps `failover_directive` to "2"
  (versioning the tuning directive would couple two unrelated capabilities' rollouts).
- Whichever core PR lands second rebases the router's `response_fields` include/exclude block
  and the parity pending-sets; each session's pending-set entries name their own SDK closer.
- **Resolved 2026-08-07:** the runaway plan now exists
  (`docs/plans/2026-08-07-runaway-protection-plan.md`) and adopted these rules with no
  conflicts. Its claims, for the record: `run_control`, `run_directive_version` (own opt-in
  marker; `failover_directive` stays `"1"`), `deny_source`, `deny_reason`,
  `estimated_output_bound`, `velocity_flags`, `receipt_aggregate_count`; it reuses
  `denied_by_period="agent_run"` with its existing meaning. This plan reuses the same
  vocabulary for `BudgetPressure.period` with the same meaning and repurposes nothing.
  Two traps it surfaced are folded in above: the silent directive attach handler
  (C2 item 2's fail-soft-plus-metric posture) and the byte-identical lease-replay blob
  (C2 item 3's router-edge attach).
- **Update 2026-08-14 (re-verification):** the runaway program's first rungs have since
  shipped — core #329 (per-run runaway hard caps) and #330 (stop agent runs + budget-authority
  fencing) landed as `runaway_run` threshold rules and run-stop machinery, **not** as wire
  directives; the SDK consumed them in #53 (`denied_by_period="run_stopped"`,
  `RunStoppedError`). None of the runaway plan's reserved wire names (`run_control`,
  `run_directive_version`, `deny_source`, `velocity_flags`, …) exist in either repo yet, and
  the check/lease/confirm wire contract is byte-stable since 2026-08-07, so the non-collision
  rules stand unchanged. These merges are what shifted the aggregator/lease-service line
  anchors (corrected above). One naming caveat for whoever lands the runaway wire fields:
  `estimated_output_bound` already exists as a shipped SDK-internal kwarg (`client.py:1519` →
  `budget.py:619`), which the responses-metering plan also reuses — the runaway plan's
  proposed wire field of the same name must be reconciled with it.
- The other plans present at planning time (`test-double`, `coverage-strict-mode`,
  eco-admission) only *consume* the contract — the test double will need a scenario update
  once `budget_pressure` ships (noted in its "scriptable price_hints" scenario list). Since
  then the untracked-surface program shipped end-to-end (SDK #54–#64, core #337–#338): it
  added the `UntrackedSurfaceReport` advisory wire model + endpoint — fully disjoint from
  this plan's field namespace, but the reason most SDK `_types.py`/`_base.py`/`client.py`
  anchors moved. The new `docs/plans/2026-08-13-responses-metering-plan.md` declares zero
  wire-contract changes yet reshapes the `client.py` dispatch seams (`_surface` parameter,
  primary-only walk) — whichever of it and SDK Stage 2 lands second re-verifies the
  `client.py` anchors.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tier flapping around 80/95 thrashes ordering | Server-side hysteresis (core C2 item 2); tier is call-scoped so the SDK holds no flappable state |
| Cost-first promotes an untranslatable cross-dialect entry → self-inflicted chain abort | Stage 1 predicate sinks it within its health tier; probe defaults to `True` on uncertainty so behavior never gets *worse* than today's eager-abort |
| Silent-feeling quality degradation damages trust | Primary displacement is opt-in (`prefer_cheaper`); every pressured call tagged; per-call `solwyn_degrade=False`; one-line WARNING on tier escalation |
| `extra="forbid"` lockstep breakage (either direction) | Marker-gated emission + API-deploys-first ordering (C1 → SDK 3 → C2); the core parity test's pending-set mechanism forces the window closed; live-contract tests assert both presence and omission |
| Parallel runaway-protection plan edits the same router block / shared models | Disjoint field namespaces + separate opt-in markers (coordination section); second-lander rebases the `response_fields` block and parity pending-sets |
| Dashboard/API type drift from new response fields | Generated-types pipeline is CI-enforced (`make openapi-check`, both SPAs' `codegen:check`) — C2/C3 regenerate as part of the PR |
| Endpoint fingerprint false-split (one endpoint spelled two ways) | base_url normalization; cost is a duplicate breaker domain (mild) vs today's false-merge (hop-4xx + wrong health) |
| Chain/hints misalignment (server bug, chain drift mid-run) | SDK validates length, discards-with-warning, never guesses; renewals re-declare the chain so lease hints always align to a request the SDK sent |
| Pressure staleness on leased runs between renewals | Bounded by the renewal cadence the lease design already guarantees (75% depletion / refresh deadline); check-path calls refresh every call; accepted, documented |
| Stage 0 breaker-report merge hides one endpoint's state | Interim only; most-severe-state merge is conservative (never hides an outage); precise per-endpoint rows land with `endpoint_label` in Stage 3 |

## Out of scope (explicit)

- Admin SPA/admin-plane surfacing (customer dashboard only; the admin plane reads replicas and needs nothing here).
- Per-project configurable pressure thresholds (v1 is fixed 80/95 server defaults; the `BudgetRule` grammar is the future home — core C2 item 6).
- Output-bound tightening under pressure (clamping `max_tokens` at critical) — a plausible future rung, not in v1; the SDK does not mutate caller caps.
- Any lease-protocol change (PJ-2 binding), runaway/terminate directives (idea 1's program), server-governed entitlement for the pressure knob (plan-gating is emission-side), latency-policy changes, client-side price tables (never), cross-provider failover for media surfaces (excluded by design, `README.md:231`), per-entry chain annotations (`degrade_ok=`), TS-SDK port of all of the above (tracked in the port decisions doc; the wire spec here is the artifact it will consume).

## Decisions for Christian

1. **Default comfort bound** — plan recommends `budget_pressure_response="reorder"` (act on
   fallback order out of the box; primary displacement opt-in). Alternative: default
   `"prefer_cheaper"` for the stronger out-of-box story, accepting silent-primary-swap risk
   (loudly tagged). Flip is a one-line default + docs.
2. **`price_hints` wire shape** — plan replaces the never-emitted `dict[ProviderName, float]`
   with chain-aligned `chain_price_hints: list[float | None]` (needed for per-model 95%-tier
   preference and same-name endpoints). This shape also lands in the Core API and the future
   TS SDK, so it's the one to veto early if you want the dict kept alongside instead.

---

## Verification Summary

Fact-checked 2026-08-14 against SDK `main` @ `1b09d62` and core `main` @ `254aeeba`
(the plan's 2026-08-07 baselines were SDK `5afd803` / core `fa0e8336`). Roughly 110
verifiable claims checked: every `file:line` anchor in both repos, plus each behavioral,
structural, and namespace claim.

**Confirmed (no change needed):**

- Every behavioral and architectural claim holds. The plan is entirely unexecuted: none of
  `budget_pressure`, `chain_price_hints`, `pressure_degraded`, `is_endpoint_fallback`,
  `endpoint_label`, `endpoint_key`, `PressureTier`, `solwyn_degrade`, `apply_budget_pressure`,
  or `can_translate` exists in either repo's code.
- The check/lease/confirm/event wire contract is byte-stable since 2026-08-07 in both repos
  (core's `shared` models diff contains only the unrelated `UntrackedSurfaceReport`).
- SDK `_routing.py`, `_lease.py`, `_registry.py`, `tests/unit/test_routing.py`, and
  `tests/unit/test_routing_policy_swap.py` are byte-identical to the plan's baseline — all
  their anchors were already correct.
- Core's `budget_check` directive/`response_fields` block is byte-identical to baseline
  (only line-shifted); `price_hints` is still declared server-side and populated nowhere.

**Corrected (~75 line anchors, all drift, no behavior changes):**

- SDK `_base.py` grew ~800 lines (guarded-resource/untracked-surface PRs #58/#59/#63/#64):
  e.g. `update_price_hints` 734-742 → 1439-1447, breaker keying 769 → 1474, directive
  application 620-677 → 1325-1382. `client.py` anchors shifted +41 to +265 (four hint-apply
  sites now 1303/1550/2447/2652; sync walk region 1373 → 1638). `_types.py` anchors after
  ~line 126 shifted +53 (`MetadataEvent` 251-379 → 304-432). README's inert-CostPolicy
  paragraph 424 → 542.
- Core: aggregator anchors shifted ~+135 and lease-service anchors +122/+282 (runaway hard
  caps #329, stop-run #330); router anchors +6 (`budget_check` 1116-1203 → 1122-1209);
  shared-model anchors +29 (`price_hints` 493 → 522); ingest sites, `db/models.py`,
  `costs.py`, and dashboard-feeding routes re-anchored; parity-test span widened to 13-37.
- Two claims tightened beyond renumbering: the directive-v1 exclude-none posture is stated at
  `_types.py:539-548` (not at the `_types.py:72-74`/`123-128` API-deploys-first comments),
  and the cross-provider sanitization branch gained a Google-target early return in PR #64
  (noted inline at the two places that cite the branch).

**Substantive updates (world changed since 2026-08-07):**

- Parallel-plan coordination section extended: runaway program partially shipped (core
  #329/#330 as `runaway_run` rules + run-stop, SDK #53 as `denied_by_period="run_stopped"`)
  with no wire-name collisions; untracked-surface program shipped end-to-end (disjoint);
  responses-metering plan (2026-08-13) added — zero wire changes, but it will re-drift
  `client.py` anchors; `estimated_output_bound` naming clash flagged for the runaway plan.
- Core C1 note added: the parity pending sets already carry other features' entries
  (`"tags"`, `"parent_agent_run_id"`) — append, don't replace.

**Unverifiable:** none.
