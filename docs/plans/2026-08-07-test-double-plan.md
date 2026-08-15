# `solwyn.testing` — Deterministic Control-Plane Test Double: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an in-process control-plane double (`solwyn.testing.FakeControlPlane`) with deterministic scenario triggers, so any team can exercise `BudgetExceededError` handling, fail-open posture, lease fallback ladders, and exit-flush behavior in CI with zero network and zero credentials — and so Solwyn's own test suite can run most control-plane behavior tests without a live API.

**Architecture:** A sans-I/O scenario state machine (`_plane.py`) fronted by one dual sync/async httpx transport (`_transport.py`) that speaks the real wire contract by constructing and serializing the SDK's own vendored Pydantic models (directive-v1 exclude-none). It is injected through a new first-class `transport` seam threaded through `BudgetEnforcer`, `MetadataReporter`, `Solwyn`/`AsyncSolwyn`, **and the `_lifecycle` exit paths** (which today build their own bare `httpx.Client`s and would silently escape any injected transport).

**Tech Stack:** Python ≥3.11, httpx (`httpx.BaseTransport`/`httpx.AsyncBaseTransport` — already a core dep), Pydantic v2 (already a core dep). **No new dependencies**; the module ships inside the wheel (`[tool.hatch.build.targets.wheel] packages = ["src/solwyn"]`, pyproject.toml).

## Global Constraints

- **No pricing, ever.** The double never converts tokens to dollars. Denials are *declarative* (scripted triggers), and all dollar display fields (`budget_limit`, `current_usage`, `remaining_budget`) are scripted numbers. This is the hard boundary from the rejected LocalBudget sibling idea. The only local cost math in the repo stays the flat-rate outage placeholder (`DEFAULT_COST_PER_TOKEN`, `budget.py:56`).
- **Real wire contract only.** Every response the double emits is built through the vendored models in `_types.py` and serialized `model_dump(mode="json", exclude_none=True)` — the directive-v1 exclude-none convention the live API uses (per `_types.py:486-495` doc and `tests/integration/test_live_contract.py`). Every request the double receives is validated through the vendored *request* models (`extra="forbid"` catches drift). A double that bypasses serialization tests nothing.
- **Privacy invariant unchanged.** The double sees only control-plane wire payloads (token counts, model names, tags) — content-free by design. `solwyn.testing` is NOT added to the content-privileged allowlist in `tests/unit/test_privacy_firewall.py`.
- **Sans-I/O split.** Scenario/verdict logic lives in a pure module; the transport is a thin I/O shim (repo convention: `_base.py` pattern, root `CLAUDE.md`).
- **`RuntimeError`, never `assert`** for runtime invariants (enforced by `tests/unit/test_no_production_asserts.py`).
- Pydantic v2 idioms only; `extra="forbid"` on new models.
- Never import provider SDKs in `src/`.
- Marker convention per `tests/CLAUDE.md`: category marker first (`unit` for double tests — they mock all external deps; the already-registered, currently unused `chaos` marker, pyproject.toml:81, for game-day recipes).
- Every PR passes `make check` and `make test` (unit suite, CI-run). Integration (`make test-integration`) unaffected until Task 4.
- Pre-launch: breaking config/constructor changes are acceptable (memory: zero customers).

---

## 1. Verified ground truth (re-checked against the repo)

Claims inherited from the project brief, re-verified. Two corrections and one load-bearing discovery.

| Claim | Verdict | Evidence |
|---|---|---|
| `BudgetMode` has exactly `ALERT_ONLY` and `HARD_DENY`; no dry-run | **Confirmed** | `src/solwyn/_types.py:46-50` |
| `SolwynConfig`: 31 fields, 24 env-mapped, no offline/noop knob | **Confirmed** | 31 fields at `src/solwyn/config.py:36-127`; env map of 24 at `config.py:135-160` |
| `api_key` is the only required field | **Confirmed with nuance** | `api_key` is the only required *constructor* field (`config.py:36`), but `_check_chain` requires ≥1 provider entry (`config.py:199-204`); the wrapper auto-builds that list from the wrapped client (`client.py:762-763`), so from the user's seat the claim holds |
| Control-plane breaker: 3 failures / 60 s | **Corrected: 3 failures / 30 s** | `control_plane_failure_threshold: int = 3` and `control_plane_recovery_timeout: float = 30.0` (`config.py:102-103`); 60 s is the *provider* breaker default (`config.py:78`). Breaker built with `success_threshold=1`, `name="control-plane"` at `_base.py:588-593` |
| Exit flush 5 s + lease surrender 2 s | **Confirmed** | `reporter_shutdown_deadline: float = 5.0` (`config.py:127`); `_EXIT_SURRENDER_BUDGET_S = 2.0` (`_lifecycle.py:114`) |
| Sticky per-run denial cache | **Confirmed** | `_run_hard_deny_responses` (`budget.py:190`), populated on `denied_by_period == "agent_run"` (`budget.py:328-336`), capped at 128 (`budget.py:82`); tag denials deliberately non-sticky (`budget.py:324-326`) |
| Live wire-contract test defines the fidelity spec | **Confirmed** | `tests/integration/test_live_contract.py` pins every behavior-bearing shape (enumerated in §2.1) |

**Verified against the real control plane** (`/Users/christian/dev/repos/solwyn-ai/core/`, the FastAPI customer API at `core/api/src/solwyn_api/`; reference only — no server work in this plan):

- **`localhost:8080` is the real server, and no mock exists to reuse.** The SDK integration suite's target is `make smoke-api-dev` (`core/Makefile:146-150`): the genuine `solwyn_api.main:app` under uvicorn with Loops/Stripe env-blanked — it needs the full DB+Redis stack. Core ships no fixture server, seeded harness, or dev double; its own tests use in-process FastAPI test clients, which the SDK cannot reuse without taking core + FastAPI + Redis as dependencies. **Decision: the double *coexists* with the smoke server** — double for CI and customers, smoke server as the live integration lane and fidelity ground truth. Nothing is replaced, and **no server-side change is needed** (the ask list to core is empty).
- **Check serialization is *conditionally* exclude-none** — a fact invisible from SDK-side tests: `budget_check` serializes `exclude_none=opted_in_to_directive` (`core/api/src/solwyn_api/routers/budgets.py:1166-1202`); a request without `failover_directive_version` gets the byte-stable legacy nullable shape. The SDK always opts in (`budget.py:273`), but the double mirrors the conditionality so raw-bytes tests of both shapes stay honest.
- **Lease responses are *unconditionally* exclude-none** — declared at the router: `response_model_exclude_none=True` (`budgets.py:1646-1650`), which is what makes the lease block vanish on ineligible/deny shapes.
- **The four lease refusal codes are a documented cross-repo contract**, not incidental: `core/api/src/solwyn_api/schemas/errors.py:73-85` spells out the SDK's dispatch semantics per code; the wire shape is `{"detail": {"code": ..., "message": <fixed public string>}}` built by `_lease_http_error` with the message map at `budgets.py:1599-1621`.
- **Check's refusal surface is richer than the SDK-side tests show**: 422 unknown-model (`UnknownModelErrorDetail`, `budgets.py:1129-1151`), 503 counter-floor-unreadable (`budgets.py:1152-1160`), 404 project-not-found (`budgets.py:1161-1164`), 429 rate-limit with Retry-After (`budgets.py:1105-1108`). The SDK treats all of these as outage-posture triggers (`budget.py:1397-1408`) — behavior worth a deterministic trigger (§3, `refuse_checks`).
- **Confirm idempotency is server-real, two ways**: a replayed confirm whose reservation is gone but settled returns an idempotent 204 via the settled marker (`budgets.py:1340-1364`); a truly unknown/expired reservation is a 404 (`budgets.py:1365-1368`), which the reporter classifies as a terminal DROP — worth a trigger (§3, `expire_reservations`).
- **Ingest**: 202 always for well-formed batches — even all-rejected ones — with per-event `{"ingested": n, "rejected": [...]}` dispositions (`core/api/src/solwyn_api/routers/costs.py:411-424, 2141-2143`).

**Load-bearing discovery — the exit paths bypass any injected client.** The interpreter-exit lease surrender builds its own `httpx.Client(timeout=5.0)` at `_lifecycle.py:254`, and the exit flush builds another at `_lifecycle.py:472`. The fork reset does the same (`budget.py:1248`, `reporter.py:768/1424`). A `MockTransport` swapped onto `enforcer._http` (today's private-seam pattern, e.g. `tests/unit/test_control_plane_breaker.py:86`) never covers those paths — and "exit-flush behavior in CI" is one of this project's named goals. **Consequence:** the injection seam must be a *stored transport* that every client construction site consults, not a one-off client replacement. Task 1 exists because of this.

**Full control-plane surface the double must speak** (every URL the SDK constructs):

| Endpoint | Caller | Success shape the SDK reads |
|---|---|---|
| `POST /api/v1/budgets/check` | enforcer (`budget.py:1385`) | 200, `BudgetCheckResponse` (exclude-none) |
| `POST /api/v1/budgets/confirm` | reporter (`reporter.py:1230,1881`), exit flush (`_lifecycle.py:469`) | 204 empty; replayed `call_id` still 204 |
| `POST /api/v1/metadata/ingest` | reporter (`reporter.py:1277,1927`), exit flush (`_lifecycle.py:470`) | 202, `{"ingested": n, "rejected": []}` |
| `POST /api/v1/budgets/lease` | enforcer (`budget.py:60`) | 200, `LeaseGrantResponse` |
| `POST /api/v1/budgets/lease/renew` | enforcer (`budget.py:61`) | 200, `LeaseGrantResponse`, generation+1 |
| `POST /api/v1/budgets/lease/surrender` | enforcer (`budget.py:62`), exit hook (`_lifecycle.py:257`) | 200, `{"released_tokens": int}`, idempotent |
| `POST /api/v1/projects/{id}/providers/breaker-reports` | reporter (`reporter.py:1187,1844`) | 2xx sink |
| `GET /health` | tooling convention (`tests/integration/conftest.py:62`) | 200 |

---

## 2. Settled design questions

### 2.1 Fidelity boundary — what the double simulates faithfully, and what it refuses to

**Evidence.** Two sources, now cross-checked: the SDK's live contract test (what the SDK *reads*) and the real endpoint implementations in `core/api/` (what the server *writes* — citations in §1). `test_live_contract.py` is the SDK's own definition of "the shapes behavior keys on": check allow omits `denied_by_period` entirely (exclude-none, lines 266-275); deny carries the exact literals `"monthly"` / `"agent_run"` / `"tag"` that drive sticky-vs-non-sticky caching (lines 277-343, keyed on `budget.py:324-336`); allow carries a v1 `failover_directive` with a real boolean (lines 240-260); lease grants carry the full 8-key lease block or omit it entirely on ineligible/deny (lines 359-368, 534-569); refusals are classified **by status alone** — 409 `lease_holder_cap_exceeded`, 404 `lease_not_found`, 409 `lease_generation_conflict`, 503 `lease_unavailable` (lines 572-615 and `budget.py:693-764`); renew echoes generation+1 (line 631); surrender returns `{"released_tokens"}` and is idempotent (lines 652-661); confirm is 204, replay-safe on `call_id`, and 422s unless exactly one settlement key is present (lines 693-761); read-only key is a 403 with `detail.code == "read_only_key"` that the SDK treats as "plane responded" (`_read_only_key.py`, `reporter.py:1241`). The reporter additionally classifies confirm/ingest outcomes into SENT / HELD / RETRY / DROPPED off status and breaker state (`reporter.py:1210-1299`).

**Faithful (in scope):**
- All 8 endpoints above, request-validated through the vendored request models, response-serialized through the vendored response models mirroring core's serialization exactly: check exclude-none **iff** the request opted into the directive (`core budgets.py:1166-1202`), lease responses exclude-none **always** (`core budgets.py:1646-1650`).
- Check verdicts: allow with fresh `reservation_id`; deny with the three period literals; alert-only deny (response `mode` drives `budget.py:443-463`); directive v1 echo gated on the request's `failover_directive_version` opt-in; scriptable `price_hints`.
- Check refusals (scripted, canned to core's real shapes): 422 unknown-model detail, 429 with Retry-After, 503 string-detail (`core budgets.py:1105-1164`) — all of which the SDK routes to its outage posture (`budget.py:1397-1408`).
- Confirm: 204; per-`call_id` replay dedup (recorded once) plus settled-marker idempotency for replayed reservations (`core budgets.py:1340-1364`); 404 for expired/unknown reservations (`core budgets.py:1365-1368`, the reporter's terminal-DROP path); 422 on both/neither settlement key (mirrors the model validator server-side, so raw-bytes tests still hold).
- Ingest: 202 with an accurate `{"ingested": n, "rejected": []}` body (the reporter parses it — `_log_ingest_rejections`, `reporter.py:1283`).
- Leases: grant with a complete, internally consistent lease block (`generation` starts at 1, `refresh_interval_s < lease_length_s`, posture echoing the plane's `mode` and the request's `fail_open` — the §4 outage-ladder inputs); renew with generation fencing (echo G → G+1; wrong G → 409 `lease_generation_conflict`); unknown lease → 404 `lease_not_found`; scriptable 409 holder-cap and 503 `lease_unavailable` refusals — all four codes in core's documented `{"detail": {"code", "message"}}` shape (`core schemas/errors.py:73-85`, `core budgets.py:1599-1621`); ineligible (`zero_rate_model`) with the lease block omitted; deny with block omitted + `denied_by_period`; `final_grant` wind-down flag; idempotent surrender (`released_tokens: 0` on replay, never 404 — `core budgets.py:1746-1747`).
- Transport-level failure scenarios: outage (`httpx.ConnectError`), slow response (deadline/exit-flush pressure), read-only-key 403.
- Breaker interplay: nothing simulated — the *real* client-side breaker reacts to the double's failures exactly as to a live outage (3 failures → OPEN, `config.py:102`). Tests that need fast recovery pass the existing `control_plane_recovery_timeout` config knob; no fake clock in v1.

**Explicitly out (too much fidelity = a second server):**
- Pricing of any kind: no token→dollar math, no per-model rates, no computed `current_usage`. Deny is triggered, never computed. Dollar fields are plane attributes the test may set.
- Auth/signup/projects CRUD, budget-rule PUTs, period rollover, the 900 s abandoned-reservation sweep, server-side lease reclaim/float accounting, cross-process/cross-instance state, dashboards.
- Server-side jitter (`refresh_interval_s` is exactly what the test configured — determinism beats realism here).

The anti-drift hinge between the double and the live API is Task 4's shared contract-assertion pack: one set of assertion functions runs against the double in every CI run and against the live API in the integration lane. If core's wire drifts, the live lane fails; if the double drifts, the CI lane fails against the same assertions.

### 2.2 Injection point — explicit object + first-class transport kwarg; no env mode

**Evidence.** (a) All four component constructors hard-code their clients: `budget.py:1233`, `reporter.py:697/1383`; fork-reset re-creates them (`budget.py:1248`); the exit paths create more (`_lifecycle.py:254,472`). (b) The existing test pattern overwrites the private `_http` (`test_control_plane_breaker.py:86`) — precisely the monkeypatching the constraints reject, and it misses fork/exit. (c) `httpx.MockTransport` implements both `handle_request` and `handle_async_request`, so one transport object can serve `Solwyn` and `AsyncSolwyn` and the sync exit clients. (d) `Solwyn.__init__`'s `**config_kwargs` feed `SolwynConfig` (`extra="forbid"`, `client.py:762-771`), so the seam must be an explicitly named parameter, not a config field — an httpx transport is a live object, not env-mappable config, and doesn't belong in a Pydantic settings model.

**Decision:** three layers, each independently usable:

1. **Core seam (Task 1):** keyword-only `transport:` on `BudgetEnforcer`/`AsyncBudgetEnforcer`/`MetadataReporter`/`AsyncMetadataReporter`, and keyword-only `control_plane_transport:` on `Solwyn`/`AsyncSolwyn` that threads it through. Components *store* the transport and route **every** client construction through one private factory (`_new_http_client()`), including fork-reset; `_lifecycle` builds its exit clients via the owning component's factory instead of bare `httpx.Client(...)`. Default `None` preserves today's behavior exactly. The injected object must implement both sync and async transport interfaces (as `httpx.MockTransport` and the plane's transport do) — documented on the parameter.
2. **The double (Tasks 2-3):** `FakeControlPlane` exposes `.transport` plus `.api_key` / `.api_url`.
3. **Sugar:** `plane.wrap(inner_client, **kwargs) -> Solwyn` / `plane.wrap_async(...) -> AsyncSolwyn` — one-line setup. `solwyn.testing` may import core; core never imports `solwyn.testing`.

**Zero-network guarantee:** the transport handles requests in process (no sockets can be opened), the plane's `api_url` is `http://control-plane.invalid` (RFC 2606 — any code path that escapes the seam fails DNS instantly rather than touching a real host), and the plane's deterministic `api_key` (`sk_proj_` + 64 hex, matching `_validation.py:14`) authenticates nowhere.

**Rejected — `SOLWYN_MODE=test` env/config mode:** a mode string can leak into prod config; a config-constructed double leaves the test with no object reference for triggers and recorded-traffic assertions; and process-global state is exactly what breaks two independently scripted planes in one process. The explicit object is also the discoverable one: it appears in the README, in `solwyn.testing.__all__`, and in the type of the `control_plane_transport` parameter. A pytest fixture (Task 5) recovers the "zero boilerplate" ergonomics an env var would have offered.

**Naming — judgment call flagged for review:** the brief says `TestControlPlane`, but pytest collects any `Test*`-prefixed class visible in a test module's namespace, so `from solwyn.testing import TestControlPlane` would emit `PytestCollectionWarning` in every customer test file (the well-known Starlette `TestClient` papercut). Plan uses **`FakeControlPlane`**. Revert in Task 2 if you prefer the brief's name despite the warning noise.

### 2.3 Trigger design — programmatic scenarios as the foundation, magic model names as sugar, magic tags rejected

**Evidence.** Tags are not a neutral carrier: a check with `tags is not None` skips the lease path *and* the allow cache entirely (`budget.py:1298-1302, 1328`), so a magic *tag* would silently reroute the very machinery under test — a deny triggered by tag could never exercise the lease ladder. Model names, by contrast, ride every path untouched: check (`_types.py:417`), lease grant (`_types.py:545`), confirm (`_types.py:804`), and the live contract test itself already uses a magic model (`"no-such-model-for-leases"` → `zero_rate_model`, `test_live_contract.py:534-549`) — precedent inside the fidelity spec. Meanwhile, plane-state scenarios (outage, slow settlement, lease 503) are not per-call properties and cannot be expressed by any per-call token; and deny-then-outage-then-recovery composition needs ordered imperative control.

**Decision:**
- **Programmatic API is the source of truth** (`deny_next`, `deny_run`, `outage()`, `slow()`, `refuse_leases()`, `read_only()` — signatures in §3). Deterministic composition via ordered calls and context managers; windows measured in **request counts, never wall-clock seconds**, so scenarios are exactly reproducible.
- **Magic model names are per-call-verdict sugar** implemented on top, under the reserved prefix `solwyn-test/` (table in §3.1). Stripe-faithful: the magic value rides the same field a real value would, deny paths raise before any provider call (so deny tests need no provider mock at all), and teams can hardcode them in fixtures without holding a plane reference.
- **Magic tags: rejected** for the routing evidence above. (Plane-side, `tags` in a check still work normally — `solwyn-test/deny-tag` exists to exercise the non-sticky tag-denial branch — the rejection is only of tags *as the trigger channel*.)
- **Parallel-worker determinism:** all state is instance-scoped (no module globals, no env mutation, no wall clock in verdict logic); pytest-xdist workers are separate processes and separate plane instances; two planes in one process share nothing. Magic-model verdicts are stateless per call except `runaway`, which is keyed per `(plane instance, agent_run_id)`.
- **Precedence** when scenarios overlap, matching real-world layering: transport-level (outage, slow) → endpoint refusals (lease 503/409, read-only 403) → verdict triggers (deny_next, deny_run, magic models) → default allow. Documented on the class.

### 2.4 Dogfooding — what moves onto the double, what stays live

**Evidence.** The integration suite (18 modules) is gated behind a live API at `localhost:8080` and skipped otherwise (`tests/integration/conftest.py:58-65`); CI runs unit only (`tests/CLAUDE.md`). Inspection splits the suite by *what the assertions are about*:

**Moves to the double (becomes CI-run, `unit`-marked, in `tests/unit/testing_double/`):** tests whose assertions are about **SDK behavior given a server response** — `test_budget_check.py` (allow shape, fail-open posture — its "unreachable" case literally points at `http://localhost:1` today, `test_budget_check.py:44`, which `plane.outage()` replaces deterministically), `test_budget_confirm.py`, `test_async_budget.py`, the SDK-behavior halves of `test_metadata_ingest.py`/`test_async_metadata_ingest.py`, and the denial-path behavior of `test_e2e_budget_denial.py` (deny raises before the provider is called — no provider fixture needed on the double). Migration = port assertions, not delete: the original files are removed only where the double-backed copy asserts strictly more (Task 4 lists per-file dispositions).

(Prior art, settled in §1: the live target is core's real `smoke-api-dev` server, and core ships no mock to reuse — the double *coexists* with it rather than replacing or wrapping anything, and the ask list to core is empty.)

**Stays live (genuinely needs the real API):**
- `test_live_contract.py` — **entirely.** It is the fidelity spec; running it against the double would be circular.
- Everything asserting **server-side** behavior: real reservation settlement moving `current_usage` (`conftest.budget_status`, lease e2e reclaim observations in `test_budget_lease_e2e.py`), server-side confirm replay dedup, the holder-cap firing organically, signup/provisioning flows, the entitlement-ceiling behavior (`tests/CLAUDE.md`).
- The `test_e2e_*` wrapper harness keeps a live lane: it exists to prove the full pipeline over real HTTP on both sides. Task 4 adds double-backed *counterparts* for its scenario coverage rather than moving it.

**New coverage only the double makes possible (CI-run for the first time):** control-plane outage mid-run, lease-503 fallback ladder, slow-settlement vs. the 5 s exit-flush deadline (`config.py:127`) and 2 s surrender budget (`_lifecycle.py:114`), breaker-open HELD confirms, sticky run denial across an outage. These become the `chaos`-marked game-day recipes.

---

## 3. Public API of `solwyn.testing`

```
src/solwyn/testing/
  __init__.py       # exports: FakeControlPlane, MAGIC_MODELS, __all__
  _plane.py         # FakeControlPlane + sans-I/O verdict/state machine
  _transport.py     # _FakePlaneTransport(httpx.BaseTransport, httpx.AsyncBaseTransport)
  _wire.py          # request parsing + exclude-none response serialization via _types models
  pytest_plugin.py  # opt-in fixtures (Task 5)
```

```python
class FakeControlPlane:
    """In-process Solwyn control-plane double. Deterministic, zero-network.

    Simulates wire behavior only: denials are scripted, never priced —
    the real API owns all pricing. Thread-safe (reporter flush threads and
    lease renewal workers hit the transport concurrently).
    """

    def __init__(
        self,
        *,
        mode: BudgetMode | str = BudgetMode.HARD_DENY,
        budget_limit: float = 100.0,
        current_usage: float = 0.0,
        remaining_budget: float | None = None,   # default: limit - usage
        project_id: str = "proj_fake",
        failover_tuning_allowed: bool = True,
        price_hints: dict[str, float] | None = None,
        lease_eligible: bool = True,
        granted_tokens: int = 200_000,
        refresh_interval_s: float = 30.0,
        lease_length_s: float = 90.0,
    ) -> None: ...

    # ── wiring ────────────────────────────────────────────────────────
    api_key: str          # deterministic "sk_proj_" + 64 hex (passes _validation)
    api_url: str          # "http://control-plane.invalid" (RFC 2606 — can never resolve)
    transport: httpx.BaseTransport  # also implements AsyncBaseTransport; hand to
                                    # control_plane_transport= or any httpx client

    def wrap(self, provider_client: object, **solwyn_kwargs: Any) -> Solwyn: ...
    def wrap_async(self, provider_client: object, **solwyn_kwargs: Any) -> AsyncSolwyn: ...
    # wrap/wrap_async pass api_key/api_url/control_plane_transport and default
    # budget_check_cache_ttl=0 (deterministic per-call checks; the allow cache
    # otherwise swallows reservation assertions — precedent: integration
    # conftest.py:382-384). Caller kwargs override everything.

    # ── scenario triggers (programmatic; composable; request-counted) ─
    def deny_next(self, n: int = 1, *, period: str = "monthly") -> None: ...
    def deny_run(self, agent_run_id: str) -> None: ...        # denied_by_period="agent_run"
    def clear_denials(self) -> None: ...
    def outage(self, *, requests: int | None = None) -> AbstractContextManager[None]: ...
        # httpx.ConnectError on every control-plane request; count-bounded or
        # context-bounded ("with plane.outage(): ..." → recovery on exit)
    def slow(self, seconds: float, *, path: str = "/api/v1/budgets/confirm",
             requests: int | None = None) -> AbstractContextManager[None]: ...
        # time.sleep in sync handler, asyncio.sleep in async handler
    def refuse_leases(self, *, status: int = 503, code: str = "lease_unavailable",
                      requests: int | None = None) -> AbstractContextManager[None]: ...
        # also usable as (status=409, code="lease_holder_cap_exceeded")
    def read_only(self, *, requests: int | None = None) -> AbstractContextManager[None]: ...
        # 403 {"detail": {"code": "read_only_key", ...}} on write endpoints
    def refuse_checks(self, *, status: int = 503, requests: int | None = None,
                      retry_after: int | None = None) -> AbstractContextManager[None]: ...
        # canned check refusals in core's real shapes: 422 unknown-model detail,
        # 429 (+Retry-After header), 503 string detail (core budgets.py:1105-1164)
    def expire_leases(self) -> None: ...   # next renew → 404 lease_not_found (forces re-grant)
    def expire_reservations(self) -> None: ...
        # next confirms → 404 "Reservation not found or expired"
        # (core budgets.py:1365-1368; the reporter's terminal-DROP accounting path)

    # ── recorded wire traffic (parsed through the REAL request models) ─
    checks: list[BudgetCheckRequest]
    confirms: list[BudgetConfirmRequest]
    ingested: list[MetadataEvent]
    lease_grants: list[LeaseGrantRequest]
    lease_renewals: list[LeaseRenewRequest]
    lease_surrenders: list[LeaseSurrenderRequest]
    breaker_reports: list[dict[str, Any]]
    unmatched_requests: list[tuple[str, str]]   # (method, path) that 404'd
    def reset_recording(self) -> None: ...
```

Display fields (`budget_limit`, `current_usage`, `remaining_budget`) are plain mutable attributes — a test scripts spend progression by assignment, never by pricing.

### 3.1 Magic model names (`MAGIC_MODELS`)

| Model | Check verdict | Exercises |
|---|---|---|
| `solwyn-test/deny` | deny, `denied_by_period="monthly"`, plane `mode` | `BudgetExceededError` + global sticky deny (`budget.py:340`) |
| `solwyn-test/deny-alert` | deny with response `mode="alert_only"` | allowed-with-warning path (`budget.py:443-463`) |
| `solwyn-test/deny-tag` | deny, `denied_by_period="tag"` | non-sticky tag denial (`budget.py:324-326`) |
| `solwyn-test/runaway` | first check per `agent_run_id` allows; later checks for that run deny with `"agent_run"` | runaway-run denial + per-run sticky cache (`budget.py:328-336`) |
| `solwyn-test/lease-ineligible` | lease grant → `eligible=False`, `ineligible_reason="zero_rate_model"`; checks allow | legacy-path fallback (`test_live_contract.py:534-549` shape) |

Prefix `solwyn-test/` is reserved; unknown `solwyn-test/*` names raise `RuntimeError` from the plane (fail loud beats silently allowing a typo'd scenario).

---

## 4. Tasks

Five PR-sized tasks. Each ends green on `make check && make test`.

### Task 1: Control-plane transport seam (core refactor, no new package)

**Files:**
- Modify: `src/solwyn/budget.py` (BudgetEnforcer `__init__` ~:1210-1252, AsyncBudgetEnforcer equivalents)
- Modify: `src/solwyn/reporter.py` (both reporters' `__init__`/fork-reset, ~:660-700, :768, :1346-1424)
- Modify: `src/solwyn/client.py` (`Solwyn.__init__` :724, `AsyncSolwyn.__init__` :1844 — new kwarg, threaded through :781-811 / :1894-1907)
- Modify: `src/solwyn/_lifecycle.py` (exit surrender :254, exit flush :472 — build via component factory)
- Modify: `tests/unit/test_control_plane_breaker.py` (`_make_enforcer` :74-87 — constructor injection replaces `enforcer._http =` overwrite)
- Test: `tests/unit/test_transport_seam.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `BudgetEnforcer(..., transport: httpx.BaseTransport | None = None)`, same on `AsyncBudgetEnforcer` / `MetadataReporter` / `AsyncMetadataReporter`; `Solwyn(..., control_plane_transport: httpx.BaseTransport | None = None)`, same on `AsyncSolwyn`; private `_new_http_client()` (sync components) / `_new_async_http_client()` (async components) and `_new_exit_http_client() -> httpx.Client` (all components; used by `_lifecycle`). Tasks 2-5 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/unit/test_transport_seam.py — every control-plane construction site honors an injected transport."""
import httpx
import pytest

from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY
from solwyn.budget import BudgetEnforcer
from solwyn.reporter import MetadataReporter


class _Recorder:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/api/v1/budgets/check":
            return httpx.Response(200, json=ALLOW_BUDGET_RESPONSE)
        if request.url.path == "/api/v1/budgets/confirm":
            return httpx.Response(204)
        return httpx.Response(202, json={"ingested": 1, "rejected": []})


@pytest.mark.unit
def test_enforcer_constructor_transport_serves_check() -> None:
    rec = _Recorder()
    enforcer = BudgetEnforcer(
        "http://control-plane.invalid", VALID_API_KEY,
        transport=httpx.MockTransport(rec.handler),
    )
    try:
        result = enforcer.check_budget(
            estimated_input_tokens=10, model="gpt-5.5", provider="openai"
        )
    finally:
        enforcer.close()
    assert result.allowed is True
    assert rec.paths == ["/api/v1/budgets/check"]


@pytest.mark.unit
def test_fork_reset_rebuilds_client_on_same_transport() -> None:
    rec = _Recorder()
    enforcer = BudgetEnforcer(
        "http://control-plane.invalid", VALID_API_KEY,
        transport=httpx.MockTransport(rec.handler),
    )
    try:
        enforcer._reset_after_fork_in_child()  # simulated fork
        enforcer.check_budget(estimated_input_tokens=10, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()
    assert rec.paths == ["/api/v1/budgets/check"]


@pytest.mark.unit
def test_exit_client_factory_uses_injected_transport() -> None:
    rec = _Recorder()
    reporter = MetadataReporter(
        "http://control-plane.invalid", VALID_API_KEY,
        transport=httpx.MockTransport(rec.handler),
    )
    try:
        exit_client = reporter._new_exit_http_client()
        resp = exit_client.post("http://control-plane.invalid/api/v1/metadata/ingest", json=[])
        exit_client.close()
    finally:
        reporter.close()
    assert resp.status_code == 202
    assert rec.paths == ["/api/v1/metadata/ingest"]
```

Add async twins (`AsyncBudgetEnforcer` / `AsyncMetadataReporter` with the same `MockTransport`), and a `Solwyn`-level test that `control_plane_transport=` reaches both `_budget._http` and `_reporter._http` (assert via one wrapped `check_budget` + one `report_settlement` flush landing in `rec.paths`).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_transport_seam.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'transport'`.

- [ ] **Step 3: Implement the seam**

Pattern (sync enforcer; mirror on the other three components):

```python
# budget.py — BudgetEnforcer.__init__ signature gains:
transport: httpx.BaseTransport | None = None,
# body:
self._transport = transport
self._http = self._new_http_client()

def _new_http_client(self) -> httpx.Client:
    """Every sync control-plane client is built here (init, fork, exit)."""
    return httpx.Client(timeout=5.0, transport=self._transport)

def _new_exit_http_client(self) -> httpx.Client:
    """Client for the _lifecycle exit paths (their own 5.0 timeout today)."""
    return httpx.Client(timeout=5.0, transport=self._transport)
```

Fork-reset bodies replace `httpx.Client(timeout=5.0)` with `self._new_http_client()` (`budget.py:1248`, reporter equivalents). Async components: `_new_async_http_client() -> httpx.AsyncClient` with `transport=self._transport` (type the stored attribute `httpx.BaseTransport | httpx.AsyncBaseTransport | None`; document that injected transports must implement both interfaces, as `httpx.MockTransport` does — the async reporter's exit path is drained by a *sync* client). `_lifecycle.py:254` becomes `client = holder._new_exit_http_client()`; `_lifecycle.py:472` becomes `client = base._new_exit_http_client()`. `client.py`: `control_plane_transport: httpx.BaseTransport | None = None` keyword-only on both wrappers, passed as `transport=` to enforcer + reporter (`client.py:781-811`, `:1894-1907`). Reporter timeout stays 10.0 in its own `_new_http_client`.

Migrate `test_control_plane_breaker.py:_make_enforcer` to pass `transport=httpx.MockTransport(transport.handler)` instead of overwriting `enforcer._http` (`:86`) — proof the public seam covers the existing suite's needs.

- [ ] **Step 4: Guard the seam against regression**

Add to `tests/unit/test_transport_seam.py` a conformance test in the repo's `test_no_production_asserts.py` style: grep `src/solwyn/` for `httpx.Client(` / `httpx.AsyncClient(` occurrences and assert every hit is inside a `_new_*http_client` factory (allowlist: `budget.py`, `reporter.py` factory bodies). A future construction site that bypasses the seam fails CI, not a customer's test.

- [ ] **Step 5: Run full gate**

Run: `make check && make test`
Expected: PASS, including the migrated breaker tests.

- [ ] **Step 6: Commit**

```bash
git add src/solwyn/budget.py src/solwyn/reporter.py src/solwyn/client.py src/solwyn/_lifecycle.py tests/unit/test_transport_seam.py tests/unit/test_control_plane_breaker.py
git commit -m "feat(sdk): first-class control-plane transport seam (incl. fork + exit paths)"
```

### Task 2: `FakeControlPlane` core — check / confirm / ingest, scenarios, recording

**Files:**
- Create: `src/solwyn/testing/__init__.py`, `_plane.py`, `_transport.py`, `_wire.py`
- Test: `tests/unit/testing_double/test_plane_check.py`, `test_plane_confirm_ingest.py`, `test_plane_scenarios.py`, `test_plane_wrap.py`

**Interfaces:**
- Consumes: Task 1's `control_plane_transport=` kwarg; vendored models from `solwyn._types`.
- Produces: `FakeControlPlane` per §3 (constructor, `api_key`/`api_url`/`transport`, `wrap`/`wrap_async`, `deny_next`/`deny_run`/`clear_denials`/`outage`/`slow`/`read_only`/`refuse_checks`/`expire_reservations`, recording lists, `reset_recording`), `MAGIC_MODELS`. Lease endpoints 501 until Task 3.

- [ ] **Step 1: Write the failing tests** (representative cores; one per module)

```python
# test_plane_check.py
@pytest.mark.unit
def test_default_check_allows_with_reservation_and_v1_directive() -> None:
    plane = FakeControlPlane()
    enforcer = BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport)
    try:
        result = enforcer.check_budget(
            estimated_input_tokens=100, model="gpt-5.5", provider="openai"
        )
    finally:
        enforcer.close()
    assert result.allowed is True
    assert result.reservation_id is not None
    assert result.failover_tuning_allowed is True
    # the double parsed the SDK's real wire bytes:
    assert plane.checks[0].failover_directive_version == "1"


@pytest.mark.unit
def test_deny_next_omits_denied_by_period_on_later_allow() -> None:
    # Wire-shape fidelity: allow responses must carry NO denied_by_period key.
    plane = FakeControlPlane()
    plane.deny_next(1, period="monthly")
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        denied = http.post("/api/v1/budgets/check", json=_check_payload(),
                           headers=_auth(plane)).json()
        allowed = http.post("/api/v1/budgets/check", json=_check_payload(),
                            headers=_auth(plane)).json()
    assert denied["allowed"] is False and denied["denied_by_period"] == "monthly"
    assert allowed["allowed"] is True and "denied_by_period" not in allowed


# test_plane_scenarios.py
@pytest.mark.unit
def test_outage_trips_real_breaker_then_recovers() -> None:
    plane = FakeControlPlane()
    enforcer = BudgetEnforcer(
        plane.api_url, plane.api_key, transport=plane.transport,
        control_plane_breaker=CircuitBreaker(
            failure_threshold=3, recovery_timeout=0.05, success_threshold=1,
            name="control-plane",
        ),
    )
    try:
        with plane.outage():
            for _ in range(3):
                assert enforcer.check_budget(
                    estimated_input_tokens=10, model="gpt-5.5", provider="openai"
                ).allowed is True  # fail-open posture
        time.sleep(0.06)  # recovery window elapses
        result = enforcer.check_budget(
            estimated_input_tokens=10, model="gpt-5.5", provider="openai"
        )
    finally:
        enforcer.close()
    assert result.reservation_id is not None  # live check again


# test_plane_wrap.py
@pytest.mark.unit
def test_magic_deny_model_raises_before_provider_call() -> None:
    openai = pytest.importorskip("openai")
    plane = FakeControlPlane()
    inner = openai.OpenAI(base_url="http://provider.invalid", api_key="sk-unused")
    client = plane.wrap(inner)
    try:
        with pytest.raises(BudgetExceededError):
            client.chat.completions.create(
                model="solwyn-test/deny", messages=[{"role": "user", "content": "hi"}]
            )
    finally:
        client.close()
    # zero provider network: deny raised pre-flight, provider.invalid never resolved
    assert [c.model for c in plane.checks] == ["solwyn-test/deny"]
```

`test_plane_confirm_ingest.py` pins: confirm → 204; replayed `call_id` → 204 recorded once; `plane.expire_reservations()` → 404 and the reporter's terminal-DROP accounting; both/neither settlement key (raw bytes) → 422; ingest → 202 `{"ingested": n, "rejected": []}`; `plane.slow(seconds=..., path=".../confirm")` makes `reporter.close(timeout=0.2)` return within its deadline with the confirm counted, not hung (the exit-flush pressure scenario). `test_plane_check.py` additionally pins the serialization conditionality: a raw check *without* `failover_directive_version` gets the legacy nullable shape (null-valued keys present), an opted-in check gets exclude-none (`core budgets.py:1166-1202`); and `plane.refuse_checks(status=429, retry_after=1)` drives the SDK to its outage posture.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/unit/testing_double/ -v` → `ModuleNotFoundError: solwyn.testing`.

- [ ] **Step 3: Implement**

`_plane.py`: `FakeControlPlane` holds a `threading.Lock`-guarded scenario state (deque of pending denials, active outage/slow/read-only windows with remaining-request counters, per-run runaway memory, seen-`call_id` set) and the verdict function `handle(method, path, body) -> _PlaneResponse | _RaiseTransportError` — pure, no I/O, no wall clock. `_wire.py`: parse requests via `BudgetCheckRequest.model_validate` etc. (a request the vendored models reject → 422, mirroring the server); serialize via `response_model.model_dump(mode="json", exclude_none=True)` (`RuntimeError` if a handler ever emits a non-model). `_transport.py`: one class implementing `handle_request` (sleeps with `time.sleep`) and `handle_async_request` (`await asyncio.sleep`), both delegating to `plane.handle`; outage raises `httpx.ConnectError("solwyn.testing: scripted outage")`. `wrap`/`wrap_async` per §3. Unknown paths → 404 + `unmatched_requests` append; lease paths → 501 with a "Task 3" message so a lease-enabled run fails loud, not weird (until Task 3, `wrap()` also defaults `lease_enabled=False`, removed in Task 3).

- [ ] **Step 4: Run + privacy gate** — `uv run pytest tests/unit/testing_double/ -v` then `uv run pytest tests/unit/test_privacy_firewall.py -v` (the new package must pass the firewall *without* joining the allowlist).

- [ ] **Step 5: Full gate + commit**

```bash
make check && make test
git add src/solwyn/testing tests/unit/testing_double
git commit -m "feat(testing): FakeControlPlane — deterministic in-process control-plane double"
```

### Task 3: Lease surface on the plane

**Files:**
- Modify: `src/solwyn/testing/_plane.py`, `_wire.py`, `__init__.py`
- Test: `tests/unit/testing_double/test_plane_lease.py`

**Interfaces:**
- Consumes: Task 2's plane internals; `LeaseGrantRequest/LeaseRenewRequest/LeaseSurrenderRequest/LeaseGrantResponse` from `_types.py`.
- Produces: the three lease endpoints live; `refuse_leases()`, `expire_leases()`, `solwyn-test/lease-ineligible`; `wrap()` stops defaulting `lease_enabled=False`.

- [ ] **Step 1: Write the failing tests** — pin, against a real `BudgetEnforcer` on the plane's transport, the live-contract shapes: fresh grant carries all 8 lease-block keys with `generation == 1`, `refresh_interval_s < lease_length_s`, posture echoing plane `mode` + request `fail_open`; renew echoing G returns G+1, wrong G → 409 `lease_generation_conflict`, unknown id → 404 `lease_not_found`; surrender → `{"released_tokens": > 0}` then idempotent `{"released_tokens": 0}`; `refuse_leases()` → run drops to the legacy per-call path (`plane.checks` grows — the refusal-is-not-an-outage classification, `budget.py:693-764`); deny grant (via `deny_next`) omits the lease block and feeds sticky denial; `with plane.outage():` after a held grant exercises the §4 ladder (drawdown continues until share exhaustion). End-to-end: `plane.wrap(...)` + `solwyn.run(...)` + magic models drive grant → renew (force with a small `refresh_interval_s` and a scripted `expire_leases()`) → surrender-on-close, asserted from `plane.lease_grants/lease_renewals/lease_surrenders`.

- [ ] **Step 2: Run to verify failure** — lease endpoints answer 501.

- [ ] **Step 3: Implement** — per-`(agent_run_id, holder_id)` lease records (id `lse_fake<n>`, monotonically issued; generation counter; `final_grant=False` until `expire_leases`), grant/renew/surrender handlers building `LeaseGrantResponse` through the model, fencing by comparing the echoed generation, surrender idempotency via a released set. No jitter, no float accounting, no reclaim — the *client-side* ladder in `_lease.py` is the thing under test.

- [ ] **Step 4-5: Gate + commit**

```bash
make check && make test
git add src/solwyn/testing tests/unit/testing_double/test_plane_lease.py
git commit -m "feat(testing): lease grant/renew/surrender surface on FakeControlPlane"
```

### Task 4: Contract parity pack + dogfooding migration + chaos recipes

**Files:**
- Create: `src/solwyn/testing/contract.py` (assertion pack callable against *any* plane — double or live)
- Create: `tests/unit/testing_double/test_contract_against_double.py`
- Create: `tests/integration/test_contract_against_live.py`
- Create: `tests/unit/testing_double/test_gameday_recipes.py` (`chaos`-marked, still CI-runnable — deterministic)
- Migrate: `tests/integration/test_budget_check.py`, `test_budget_confirm.py`, `test_async_budget.py` → double-backed unit twins; delete the originals **only** where every assertion was SDK-side (per-file audit in the PR description); `test_e2e_budget_denial.py` gains a double-backed twin, live original kept.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `contract.assert_check_contract(http: httpx.Client, api_key: str)`, `assert_confirm_contract(...)`, `assert_lease_contract(...)` — plain functions taking a ready client, so both lanes share one source of truth. This is the anti-drift hinge from §2.1: same assertions, two lanes, drift on either side fails its lane.

- [ ] **Step 1:** Write `contract.py` by extracting the *shape* assertions from `test_live_contract.py` (allow omits `denied_by_period`; deny literals; directive booleans; lease-block key presence/absence; refusal status codes; surrender idempotency) into parametrized functions. The live test file keeps its server-state-dependent parts (burns, `budget_status`, holder-cap loop) unchanged and additionally calls the shared pack.
- [ ] **Step 2:** `test_contract_against_double.py` runs the pack over `FakeControlPlane().transport` (unit-marked). Run: `uv run pytest tests/unit/testing_double/test_contract_against_double.py -v` → PASS.
- [ ] **Step 3:** Migrate the three integration modules; each ported test's double twin must assert a superset (e.g. `test_fail_open_allows_on_bad_url` becomes `plane.outage()` + additionally asserting the breaker opened after 3 failures).
- [ ] **Step 4:** Game-day recipes (`chaos`): deny → outage → recovery with sticky denial preserved across the outage (`budget.py:343-375`); slow confirm vs `reporter_shutdown_deadline`; lease refusal ladder; breaker-open HELD confirms then drain on recovery.
- [ ] **Step 5:** Gate + commit

```bash
make check && make test && uv run pytest tests/ -m chaos -v
git add src/solwyn/testing/contract.py tests/
git commit -m "test(sdk): shared control-plane contract pack; move SDK-behavior integration tests onto the double"
```

### Task 5: DX finish — docs, pytest fixtures, changelog

**Files:**
- Modify: `README.md` (new "Testing your budget enforcement" section), `CHANGELOG.md`
- Create: `src/solwyn/testing/pytest_plugin.py`
- Test: `tests/unit/testing_double/test_pytest_plugin.py` (via `pytest.Pytester`)
- Modify: `tests/unit/test_public_exports.py` (pin `solwyn.testing.__all__`)

**Interfaces:**
- Consumes: Tasks 2-3 public API.
- Produces: opt-in fixtures `solwyn_control_plane` (fresh plane per test) and `solwyn_test_client` (plane + wrapped stub); enabled via `pytest_plugins = ["solwyn.testing.pytest_plugin"]` — deliberately **not** auto-registered via the `pytest11` entry point (installing the SDK must not mutate anyone's pytest environment).

- [ ] **Step 1:** README section (see Documentation plan, §5) — every snippet executed as a doctest-style unit test or mirrored verbatim in `test_pytest_plugin.py` so the docs cannot rot.
- [ ] **Step 2:** `pytest_plugin.py` (~30 lines: two fixtures, teardown closes wrapped clients).
- [ ] **Step 3:** CHANGELOG entry under the next unreleased version (memory: CLAUDE.md is a snapshot; history goes to CHANGELOG only).
- [ ] **Step 4:** Gate + commit

```bash
make check && make test
git add README.md CHANGELOG.md src/solwyn/testing/pytest_plugin.py tests/
git commit -m "docs(testing): README testing guide + opt-in pytest fixtures for FakeControlPlane"
```

---

## 5. Documentation plan (part of the deliverable — this is a DX feature)

- **README section "Testing your budget enforcement"** (Task 5), structured as three copy-paste recipes, shortest first:
  1. *Test your deny handler* — magic model, 8 lines, no provider mock needed (deny raises pre-flight): `FakeControlPlane().wrap(OpenAI(...))` + `pytest.raises(BudgetExceededError)`.
  2. *Test your fail-open posture* — `with plane.outage():`, provider mocked with the reader's own tooling (respx shown), assert the call proceeded + warning surfaced.
  3. *Game day* — deny → outage → recovery composition, pointing at the `chaos` recipes in-repo for the full ladder.
  Plus the magic-model table (§3.1), the precedence rule (§2.3), and one explicit boundary paragraph: *"The double never prices anything — the API owns pricing. Scripted denials test your handling, not your budget math."*
- **Module docstrings** on `solwyn.testing.__init__` (the contract in three sentences: zero network, real wire shapes, no pricing) and on every public method (each scenario names the SDK behavior it exercises, with the config knob that pairs with it — e.g. `outage()` ↔ `control_plane_failure_threshold`/`control_plane_recovery_timeout`).
- **`tests/CLAUDE.md`** gains a short "double vs live" section: what belongs in `tests/unit/testing_double/`, what must stay live (§2.4 rules), and that `test_live_contract.py` + `contract.py` are the paired fidelity spec.
- **CHANGELOG**: one entry per shipped PR, rolled up at release.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Double drifts from the live API → green CI, lying tests | Shared contract pack (Task 4) runs the same assertions against both; `test_live_contract.py` stays authoritative and untouched-in-spirit; double responses are *constructed through* the vendored models, so SDK-side model drift breaks the double loudly at build time |
| A client construction site bypasses the transport seam (future code) | Task 1 Step 4 conformance test greps `src/` for bare `httpx.Client(`/`AsyncClient(` outside the factories — same enforcement pattern as `test_no_production_asserts.py` |
| Scope creep into pricing ("just compute usage from confirms…") | Global constraint + module docstring non-goal + no token→dollar code path exists to extend; display fields are attributes, and the README boundary paragraph sets user expectations |
| `slow()` in async tests blocks the event loop via a sync handler | Dual-interface transport sleeps with `asyncio.sleep` on the async path (Task 2 Step 3) — this is why the plane ships its own transport class instead of raw `MockTransport` |
| Sync `slow()` deliberately blocks reporter flush threads | Intended (it's the exit-flush pressure scenario) — documented on the method; count-bounded windows prevent a forgotten `slow()` from hanging a whole suite |
| Concurrent access (renewal daemon threads, flush threads) corrupts plane state | Single lock around the verdict state machine; recording lists appended under the same lock; covered by an existing-style thread-safety test in Task 2 |
| Breaker recovery needs 30 s wall clock in tests | Recipes pass the existing env-mapped `control_plane_recovery_timeout` knob (`config.py:103`); injectable clock explicitly deferred (out of scope) |
| `Test*` class name triggers pytest collection warnings in customer suites | Named `FakeControlPlane` (§2.2) — flagged for review |
| Users run `solwyn.testing` in production | Harmless by construction (nothing global; a plane only affects clients explicitly wired to it); the `.invalid` `api_url` makes accidental real traffic impossible rather than dangerous |

## 7. Out of scope (explicit)

- Any local pricing, budget math, or "no cloud account" enforcement mode (rejected sibling idea — hard boundary).
- A `BudgetMode.DRY_RUN` / `SOLWYN_MODE=test` config mode.
- Provider-side fakes for customers (`tests/integration/fake_provider.py` remains internal test infra; customers bring respx/vcr/their own mocks — README shows the pairing).
- Simulating: auth/signup/projects CRUD, budget-rule configuration, period rollover, server-side sweeps/reclaim/float accounting, entitlement tiers, cross-process shared plane state.
- Injectable clock / time-travel for breaker & lease expiry (config knobs suffice for v1; revisit if recipes get sleep-heavy).
- Auto-registered pytest plugin (`pytest11` entry point) — opt-in only.
- TS SDK port of the double (design deliberately ports — sans-I/O verdict core + fetch-layer shim — but that's the other repo's plan).
- Wholesale deletion of the integration suite — only the Task 4 per-file audited migrations.
- Any change to the core repo (verified: none needed — §1). Core stays the reference and the live lane's server; if a future fidelity gap ever needs a server-side contract fixture, that becomes a separately-filed small ask, not staged work here.

## 8. Decisions flagged for review

1. **Class name `FakeControlPlane`** instead of the brief's `TestControlPlane` — avoids `PytestCollectionWarning` in every customer test file that imports it (§2.2). Say the word to keep `TestControlPlane`.
2. **`wrap()` defaults `budget_check_cache_ttl=0`** — deterministic per-call checks in tests, diverging from the production default of 5 s (precedent: the integration harness does exactly this, `tests/integration/conftest.py:382-384`, because allow-cache hits carry no reservation and silently skip settlement). Overridable per call; flagged because it makes test behavior differ from prod defaults by design.
