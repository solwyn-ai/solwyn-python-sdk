# Changelog

All notable changes to the Solwyn Python SDK are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions are
derived from git tags (hatch-vcs).

## [Unreleased]

## [0.7.0] - 2026-09-05

Every telemetry event for a lease-funded call now names the budget lease that
funded it, so Solwyn Cloud counts a call's spend once between its metadata
landing and its confirmation settling, including when the confirmation is
lost. Budget leases are also released early: the SDK hands a lease back the
moment it stops using it — a run scope that exits, a renewal answered
ineligible or denied, an unreadable response, a stopped run — instead of
leaving the reservation standing until the server's deadline. The testing
double catches up with the live ingest contract's duplicate lane, and the
surface canary admits the namespaces the September provider SDK releases added.
Wire-contract changes are API-first: Solwyn Cloud accepts `lease_id` before
this SDK releases, and early release adds no new field or route — only new
calls to the existing surrender endpoint. Ships #81.

### Added

- **Leases are handed back the moment the SDK stops using them.** A budget lease
  the SDK drops used to keep its float reserved on the server until the lease
  deadline (~2 minutes) plus whatever sweep followed — money nobody could spend,
  and a hard-deny sibling run refused against a counter that still included it.
  The SDK now surrenders a lease as soon as it knows the authority is dead: a
  renewal answered ineligible or denied, a lease response whose lease block
  cannot be read, and a server run-stop directive. It does NOT surrender on a
  404/409 renewal (there is nothing to release, or a live successor that would
  refuse it) or on a client-side expiry (the next admission simply re-grants).
  Every release is sent off the caller's thread and is best-effort: a refusal is
  a debug log and never retried, and the server's own deadline remains the
  backstop. A re-grant for the same run is fenced behind the release of the
  lease it replaces, so the server can never see the replacement before the
  release of what it replaced.
- **Run-scope exit surrenders the run's leases.** `with solwyn.run(...)` now
  ends the reservation it held: leaving the block — including on an exception —
  and `RunHandle.finish()` release that run's leases, so the budget is back when
  the run is over instead of at `close()` or the server deadline. `activate()`
  scopes on a `create_run()` handle deliberately do not release, since a
  detached identity is re-entered by design. Work that outlives the scope with
  the run id still bound is unaffected beyond paying one grant on its next call.
- **`MetadataEvent.lease_id`.** Every telemetry event for a lease-funded call
  now names the budget lease that funded it, the same id its confirm carries,
  on success and error events alike. Solwyn Cloud uses it to keep a call's
  spend counted once between its metadata landing and its confirmation
  settling, including when the confirmation is lost. Wire-contract change,
  API-first: Solwyn Cloud accepts the field before this release. No behaviour
  change for reservation-funded, cache-hit, fail-open, uncounted or denied
  calls, whose events carry no lease id and whose wire bytes are unchanged.

### Changed

- **A surrender gets 3 seconds and one retry.** Releasing a lease is not a cheap
  request — the server takes the project lock, drains pending targets, loads the
  ingested term, runs the release and commits — and the previous 1-second bound
  timed out under load, leaving the float to expiry. The per-request timeout is
  now 3 seconds with exactly one retry, and only on a timeout: the route is
  idempotent, so an attempt that may never have been read is worth repeating,
  while a refusal is an answer. Transport failures are unchanged (the shared
  control-plane breaker owns them), and the interpreter-exit budget stays at its
  own 2 seconds. A `close()` against an unresponsive control plane can now cost
  up to 3 seconds rather than 1; it still bounds the whole drain, not each lease.
- **Surface canary admits the September provider SDK namespaces.** openai 3.8
  (`safety.alerts`), anthropic 1.4 (`beta.organization` admin tree,
  `beta.webhooks.parse_unverified`, raw and streaming variants of
  `messages.batches.results`) and google-genai 2.22 (`environments.files`) each
  added surfaces the reviewed rule ledger did not know, so every `latest`
  inventory lane and the real-SDK fingerprint test went red. All 795 new paths
  are classified by precedent: resource namespaces, read/write admin and file
  operations as `unmetered_spend` with exact acknowledgment tokens, and the
  raw-response wrappers as `unmetered_spend` at `raw_response` scope, exactly
  like `admin.organization`, `beta.skills` and `environments`. Latest
  fingerprints, per-context digests and the README strict fingerprint (now
  audited against `openai==3.8.0`) are refreshed. Runtime behaviour for
  existing surfaces is unchanged; before this release these paths resolved to
  `unknown` and followed `on_unmetered`.
- **The lock tracks the latest openai again.** CrewAI (via `instructor`) pins
  `openai<3` and `jiter<0.16`, which had frozen the universal lock at openai
  2.x while CI's real-SDK lanes audit the latest release. `[tool.uv]
  override-dependencies` relaxes those two pins for lock resolution only; the
  opt-in CrewAI smoke lane is where a real CrewAI-versus-openai-3 break would
  surface. Dev-tooling only; nothing changes for installed packages.
- **`FakeControlPlane` reports ingest dedup skips in `duplicates[]`.** The
  double's `/metadata/ingest` 202 body is now the three-lane shape the live API
  has answered with since 2026-09-01: `{"ingested", "rejected", "duplicates"}`,
  where each duplicate names the submitted index, the probe that hit
  (`legacy_key` for the timestamp-plus-instance key, checked first, or
  `call_id`) and the echoed call id, and the three lanes partition the batch. A
  replayed batch used to answer `{"ingested": 0, "rejected": []}`; tests that
  pinned that two-key body must add the lane. The reporter never read it, so
  SDK behaviour is unchanged. The shared `solwyn.testing.contract` receipt pack
  now pins the three-key body in both lanes.

## [0.6.0] - 2026-08-24

The SDK grows from a budget gate into a control plane for agent runs. Failover
is cost-aware: every budget check opts into server price hints, and
`CostPolicy` serves the cheapest healthy provider for exactly the call that was
priced. Runaway runs stop from either side — a server-pushed `terminate_run`
directive or the content-free local velocity detector raises `RunStoppedError`,
aborts streams at the next chunk boundary, and leaves a denial receipt that
survives control-plane outages. Native OpenAI and Azure OpenAI Responses calls
are metered; wrappers pass `isinstance`, so LangChain/LangGraph, CrewAI, and
OpenAI Agents admit them; strict coverage controls can refuse unmetered surfaces
before provider I/O; Anthropic 1.x clients on `httpx2` are supported; and
`solwyn.testing.FakeControlPlane` exercises all of it with zero network.
Wire-contract changes are API-first: Solwyn Cloud accepts every field below
before this SDK releases. Ships #53–#55, #57–#67, #70, #73, #77, and #78.

### Added

- **`CostPolicy` now consumes server price hints.** Every budget check opts
  into `price_hints_version: "1"`, and the request-scoped hints apply only to
  the call they were priced for. When the policy serves a cheaper healthy
  provider ahead of a healthy primary, the resulting failover metadata reports
  `FailoverReason.COST_ROUTED` — only when the served provider's hint is
  strictly below the primary's (or the primary is unhinted), so an injected
  policy that fronts a pricier provider is still labelled `circuit_open`.
- **A first-class, zero-network control-plane test double now exercises budget
  enforcement through production wire models.** The injected transport seam is
  shared by normal operation, fork recovery, interpreter-exit delivery, and
  lease surrender. `FakeControlPlane` scripts transport and endpoint failures,
  magic-model verdicts, request recording, reservations, and the complete lease
  lifecycle without pricing provider work. A reusable
  `solwyn.testing.contract` pack is dogfooded against both the double and the
  live API. SDK-behavior integration cases now run on `FakeControlPlane` in the
  zero-network CI lane, while live integration retains authentication, routing,
  persistence, and pricing-catalog checks as deployed server-state coverage.
  Deterministic CI game-day recipes cover denial, outage, breaker, reporter, and
  lease recovery ladders. The README adds copy/paste enforcement recipes and
  `solwyn.testing.pytest_plugin` provides explicitly opt-in, function-scoped
  plane and denial-client fixtures—never automatic pytest registration.
- **The control-plane double now simulates runaway protection end to end.**
  `FakeControlPlane.stop_run(run_id)` denies every later check, lease grant, and
  lease renewal for that run with a version 1 `run_control` terminate directive
  whenever the request opted in, `clear_stop(run_id)` lifts it, and the
  `solwyn-test/kill` magic model scripts the same kill from a model name.
  `misroute_stops()` echoes directives for the wrong run, so server contract
  drift is testable deterministically. `denial_receipts` and
  `aggregate_replays` expose the recorded content-free denial evidence, and
  `reject_ingest(...)` scripts index-aware, legacy, and malformed ingest
  rejection bodies that drive the SDK's receipt fold and replay path. The
  shared `solwyn.testing.contract` pack gained `assert_run_control_contract`
  and `assert_receipt_ingest_contract`, and the CI game-day recipes cover an
  operator kill surviving an outage, a kill landing on the lease renewal
  channel, receipt loss folding into one aggregate replay, and a local velocity
  stop that no server allow can lift.
- **Provider wrappers now pass `isinstance`-shaped framework admission.**
  `Solwyn(client)` and `AsyncSolwyn(client)` report the wrapped provider class
  through `__class__` while `type(wrapper)` remains the truthful Solwyn class.
  Public attribute writes and deletes forward to the provider client;
  `copy.copy` and `copy.deepcopy` preserve one shared enforcement handle; and
  pickling fails with guidance to construct a fresh wrapper in the target
  process. Public passthrough reads still cross the capability guard, so known
  untracked and newly observed leaves keep the existing warn-once default (or
  the configured strict/allow posture) instead of silently acquiring coverage.
- **Detached run identities support begin/end framework callbacks and
  cross-task activation.** `start_run(...)` exposes the scoped `RunHandle`
  lifecycle, while `create_run(...)` snapshots a stable run ID, parent, and
  inherited tags without changing the current context. Its reusable
  `handle.activate()` scope binds that identity around provider work in another
  task or thread, and `finish()` fails loud while an activation remains live.
- **OpenAI Agents, LangChain/LangGraph, and CrewAI have admitted integration
  recipes and offline real-framework smokes.** OpenAI Agents uses a wrapped
  `AsyncSolwyn` default client plus recipe-local model/provider adapters; its
  locked/current smoke covers Chat Completions retries, streaming, handoffs,
  function-tool turns, and budget denial, but no
  `solwyn.integrations.agents` module ships. `solwyn[langchain]` adds the
  content-free `SolwynRunScopeHandler`; the exact docs/test raw-response shim
  admits basic non-streaming `invoke`/`ainvoke` and two-node LangGraph calls.
  `solwyn[crewai]` adds the content-free structural `SolwynEventListener`;
  native LiteLLM remains attribution-only with zero Solwyn enforcement, while
  the narrowly tested sync plain-text custom-`BaseLLM` recipe crosses a wrapped
  client. CrewAI/LiteLLM run in an isolated dependency lane, and scheduled
  smoke jobs re-resolve current framework releases to expose churn.
- **Native OpenAI and Azure OpenAI Responses calls are now budget-metered.**
  Sync and async `responses.create(...)`, `responses.parse(...)`, and new-response
  `responses.stream(...)` helper calls use one primary-only path with
  Responses-compatible effective defaults: chat-only defaults are stripped,
  caller arguments win, and budget preflight and provider dispatch see the same
  mapping. `create(stream=True)` and completed helper streams settle
  provider-reported usage from the terminal event; entered streams abandoned
  before terminal usage settle a length-based estimate, with lease-backed
  abandoned calls floored at the reserved authority. Non-streaming create/parse calls
  with omitted or zeroed Responses usage now use that same explicitly marked
  request-length estimate, and lease-backed calls likewise retain their full
  reservation rather than settling as exact zero. Effective `background=True`
  requests fail loud because queued responses expose no create-time usage, and
  parse refuses effective streaming before budget or provider I/O. To keep
  preflight and the serialized wire request identical, metering-critical
  `extra_body` overrides (`model`, `input`, `instructions`,
  `max_output_tokens`, and `stream`) are rejected; vendor-specific extensions
  still pass through unchanged. The helper's
  existing-response retrieval overload remains raw because it creates no new
  spend. Azure admission is explicit through an Azure-only
  `CompatProfile.supports_responses` capability flag; all other compatible
  profiles retain their guarded raw Responses managers.
- **Explicit provider identity pins bypass auto-detection.** Passing
  `provider="openai"` (or a provider in a fallback 4-tuple) selects that named
  adapter without inspecting `base_url`, enabling native OpenAI Responses
  metering behind corporate gateways and local proxies. Pins do not translate
  dialects or replace SDK clients: wrapper construction validates the actual
  provider-client family and sync/async mode, rejects mismatches with
  `ConfigurationError(field="client")`, and reports unknown names against the
  `provider` field.
- **Strict pre-call coverage controls and local coverage manifests.** Clients
  now classify and guard the reachable public provider-client graph. Set
  `on_unmetered="raise"` / `SOLWYN_ON_UNMETERED=raise` to refuse untracked or
  unknown capabilities before provider I/O; `warn` remains the compatibility
  default and `allow` is the explicit unrestricted posture. The exported
  `UntrackedSpendSurfaceError` identifies the exact refused capability, while
  `coverage(client)` returns a deterministic, content-free, sans-I/O audit
  report with literal bidirectional expectations.
- **Exact acknowledgment escape controls ship with strict posture.**
  `acknowledge_untracked` and the comma-delimited
  `SOLWYN_ACKNOWLEDGE_UNTRACKED` accept reviewed terminal capability tokens
  only. Namespaces, wildcards, inapplicable tokens, tracked leaves, blocked
  leaves, and unsupported leaves are rejected; the conditional token-billed
  TTS exception uses `audio.speech.create:gpt-4o-mini-tts`.
- **Privacy-safe advisory reporting identifies untracked provider-client
  surfaces.** It is ON by default. Provider-call paths schedule fire-and-forget
  delivery through the reporter's background thread/task, never on the budget
  hot path; shutdown may make a deadline-bounded best-effort final attempt, and
  send failures remain silent. External payloads contain only a structural
  dotted surface path; bounded provider/client-shape, sync/async, rule, scope,
  and posture fields; approximate occurrence counts and first/last timestamps;
  and random SDK-instance/report identifiers. They never contain model names,
  request arguments, prompts, or responses. Only unacknowledged `warn`/`allow`
  observations are reported; `raise` refusals and acknowledged escapes are
  not. Reporting adds neither a budget check nor a cost event, so it does not
  meter or budget-enforce these calls; counts are bounded-overcount signals,
  not billing truth, and dashboard absence is not comprehensive usage evidence.
  Core derives the project from authentication on its project-implicit route.
  Set `report_untracked_surfaces=False` or
  `SOLWYN_REPORT_UNTRACKED_SURFACES=false` to disable optional external
  advisory egress without changing local `on_unmetered`
  `warn`/`allow`/`raise` behavior.
- **Agent-run stops raise `RunStoppedError`.** The public exception inherits
  directly from `SolwynError`, not `BudgetExceededError`, so an agent loop's
  budget-denial handler cannot swallow and retry an explicit stop. It carries
  only the structural `agent_run_id`, `reason`, and `source`; server/operator
  stops use `server`, while deny-eligible local velocity rules use
  `local_velocity`. The bounded process-wide registry is exposed cooperatively
  through `current_run_terminated()`, `run_termination(run_id)`,
  `clear_run_termination(run_id)`, and immutable `RunTermination` values.
  Exact reasons remain a 256-entry LRU that never guesses from fingerprints, so
  churn cannot false-stop an unrelated or new run. A stopped run may be
  forgotten after LRU eviction if the control plane does not reaffirm it—the
  fixed-memory tradeoff required to preserve exact answers. Active stream handles
  retain their immutable first stop independently of registry eviction until
  release. These controls govern future dispatch and do not preempt a
  non-streaming request already in flight. Streams stop at the next raw
  provider-chunk boundary: that chunk is pulled and discarded, prior usage is
  settled exactly once as a partial success, the provider stream is closed, and
  the original error remains terminal.
- **Content-free run velocity detection is configurable and bounded.** Seven
  `velocity_*` / `SOLWYN_VELOCITY_*` settings control mode, repeat-size,
  monotonic-growth, and rate-acceleration thresholds. Only repeat-size and
  monotonic-growth are deny-eligible; rate acceleration remains advisory.
  State contains scalar token counts, timestamps, and structural identifiers
  only, with 128×64 detailed history and fixed-memory conservative suppression.
- **Denied-call receipts carry structural attribution.** `deny_source`,
  `deny_reason`, `denied_by_period`, `estimated_output_bound`, `velocity_flags`,
  `receipt_aggregate_count`, and `receipt_pricing_input_tokens` are optional,
  content-free metadata fields. Unavoidable receipt losses fold by
  pricing-compatible identity — which includes the denial reason and period,
  so a run's `run_stopped` and `monthly` evidence never merge — and replay in
  per-field 100-million-unit-safe aggregates after delivery recovers,
  carrying reason, period, and the union of velocity flags. Aggregates
  preserve exact token/media totals, optional media quantity presence, and
  the original per-call input-token pricing basis. Each run holds a bounded
  budget of exact-pricing fold keys; past it (or with the shared table full)
  receipts fold into one coarse per-run aggregate whose null pricing basis
  marks it unpriceable — counts and attribution survive instead of the
  receipt being refused. Terminal receipt losses are counted at receipt
  weight, never per event, so an aggregate can never understate its
  cardinality. A run-stop directive echoed for the wrong run is treated as
  server contract drift on every channel: it credits the shared
  control-plane breaker, logs a distinct ERROR, and degrades that one call
  to the outage posture instead of opening the breaker fleet-wide.
- **Anthropic 1.x clients are supported on their `httpx2` HTTP stack.** Core
  still never imports `httpx2`: an exception is admitted as an httpx2 transport
  error only when its ancestry is identity-proven against the loaded module, and
  a per-hop timeout is delivered in the provider client's own HTTP stack — an
  httpx `Timeout` for httpx1-based clients, a native `httpx2.Timeout` for
  Anthropic 1.x whose constructor is identity-proven against both the loaded
  module export and the client's own timeout instance (so Anthropic's
  `x-stainless-read-timeout` header carries the numeric read bound; if the
  proof fails the granular four-tuple is used and only that header degrades).
  Anthropic 1.0's promoted-stable `files` and `skills` surfaces are
  classified as unmetered spend, so they keep the ordinary `on_unmetered`
  posture instead of resolving as unknown surface drift; the provider-surface
  canary pins them at the `anthropic-stable-files-skills` interval
  (`anthropic==1.0.0`).

### Changed

- **The budget allow-cache is now bounded and hint-aware.** The single cached
  allow slot is now a bounded 16-entry LRU keyed by provider, model, fallback
  chain, and modality, with the existing `budget_check_cache_ttl` window
  (default 5 s, `SOLWYN_BUDGET_CHECK_CACHE_TTL`). A cache hit replays only the
  hints belonging to its own entry. `CostPolicy` warns once only when a check
  carries no hints (`null`), not when the server explicitly clears them with
  `{}`.
- **Breaking only for consumers of private wrapper attributes: Solwyn-owned
  state now lives under `_solwyn_*`.** Names such as the former wrapper
  `_client`, `_budget`, and `_reporter` no longer expose Solwyn internals;
  non-prefixed names belong to the wrapped provider client. Public Solwyn APIs
  are unchanged, and the partition prevents collisions with provider SDK
  private state after type-transparent framework admission.
- **Breaking (pre-launch): native OpenAI and Azure OpenAI Responses create,
  parse, and stream leaves are no longer acknowledgeable unmetered
  capabilities.** Remove those leaves from `acknowledge_untracked`; they are
  now tracked and rejected as acknowledgment tokens. Acknowledgment examples
  continue to use the still-unmetered `responses.retrieve` leaf.
- **Untracked capabilities now warn once by default instead of passing
  silently.** Guarded namespaces keep descendants inside the same posture
  decision. Strict mode is a cooperative pre-call guard, not a sandbox:
  retained raw clients, private wrapper state, explicitly acknowledged scoped
  raw escapes, and native behavior on returned provider objects remain outside
  its enforcement boundary.
- **Identifier-clean provider paths that exceed advisory wire limits keep the
  configured local posture.** `warn` logs once and forwards, while `allow`
  forwards silently; an over-length, over-depth, or non-ASCII path is counted
  locally and never sent externally. Malformed public paths (empty, private, or
  non-identifier segments) still fail closed.
- **`run_stopped` denials now stay sticky only for their run.** Per-call and
  lease denials use the same run-scoped classification as `agent_run`, so a
  control-plane outage cannot replay a dashboard stop against unrelated run
  ids. `BudgetCheckResult.denied_by_period` now carries the Cloud/lease label
  through client error construction; ordinary hard-deny errors expose the
  actual period and fall back to `unknown` only when no label was supplied.
- **Breaking for anyone passing `inf`: failover timeout bounds must now be
  finite numbers.** `failover_total_timeout` and `failover_hop_read_timeout`
  reject booleans, `NaN`, and `±inf` at construction; `float("inf")` was
  previously accepted and produced an unbounded failover window or hop read
  bound. `failover_hop_read_timeout` must still be positive, while a zero
  `failover_total_timeout` remains accepted as the intentional zero-deadline
  semantics.

### Fixed

- **A Responses stream-helper entry failure is classified instead of always
  blaming the provider.** The SDK manager sends its request in `__enter__`,
  after the candidate walk, so that failure now runs the same
  `classify_exception` dispositions the walk uses: a request-shaped 4xx records
  no circuit-breaker failure and reports no `possibly_succeeded`, while
  failover and post-send-ambiguous errors keep the breaker verdict. Previously a
  repeated 400 on `responses.stream(...)` could open the breaker and block all
  traffic to a healthy provider. Failures after the provider stream opens are
  unchanged.
- **Compatible-provider Responses streams without readable usage hold their
  lease floor.** A streaming settlement that falls back to the adapter's own
  length estimate now settles as unmeasured, so a run lease trues up at its
  reserved bound instead of re-lending output allowance the response already
  spent. The estimated token details are still reported as telemetry. Chat
  streaming is unaffected.
- **Closing a Responses stream helper before entering it releases its
  reservation.** No provider request was ever dispatched, so the wrapper no
  longer ships a confirm for phantom spend, records no circuit-breaker success,
  and reports no latency sample; it still forwards `close()` to the SDK
  manager.
- **`extra_body={"stream": ...}` is refused on metered Responses calls.**
  The OpenAI SDK merges `extra_body` after named arguments, so an entry there
  would desync Solwyn's streaming mode from the dispatched request and lose the
  call's metered spend.
- **Anthropic 1.x transport failures are classified instead of failing fast.**
  An `httpx2` transport error is now dispositioned exactly like its httpx1 twin
  and at the same position — ahead of any attached HTTP status: provably
  pre-send connect and pool failures fail over, while read, write, and protocol
  failures stay post-send ambiguous and are re-raised rather than retried
  against another provider. Family recognition reads the `httpx2` module and
  class names statically, so its lazy module `__getattr__` can never raise out
  of the classifier.
- **The `solwyn.testing` contract probe now opts into price hints and
  validates the served hint map.** It sends the SDK's production
  `price_hints_version: "1"`, and requires every provider key to be a known
  `ProviderName` (an unknown key makes the whole check response unparseable, so
  the SDK would fail open with no reservation) and every hint to be a finite
  JSON number; `{}` and `null` remain valid statements.

### Removed

- **The client-wide price-hint store and update APIs are removed.**
  `Solwyn.update_price_hints` and `AsyncSolwyn.update_price_hints` no longer
  exist; price hints are request-scoped and supplied by the server's budget
  response.

### Known limitation

- **Lease-backed `solwyn.run()` calls carry no price hints.** `CostPolicy`
  keeps configured provider order for those calls until lease grants carry
  hints.

## [0.5.0] - 2026-08-07

Spend attribution becomes first-class: clients carry default tags, nested run
scopes inherit and refine them key by key, one immutable per-call snapshot
feeds both budget admission and spend events, and nested runs report their
immediate parent run. Wire-contract changes are API-first: Solwyn Cloud
accepts every field below before this SDK releases. Ships #50.

### Added

- **Clients accept default spend tags through `tags={...}` or `SOLWYN_TAGS`.**
  The environment value uses comma-separated `key=value` entries and splits
  each entry at the first `=`. Values containing commas use the constructor
  mapping instead. Run-scope tags override client defaults key by key, and
  per-call `solwyn_tags=` values have the highest precedence.
- **Budget admission observes the call's tags, and tag-cap denials stay
  selector-scoped.** A tagged call's `/budgets/check` carries the same
  immutable tag snapshot its spend event reports. A denial with
  `denied_by_period == "tag"` denies only traffic matching that selector — it
  never creates project-wide or run-wide sticky denial authority. Tagged calls
  always take a fresh budget check, bypassing the project allow cache and the
  run lease path, because tag selectors can carry independent caps.
- **Nested runs report their immediate parent.** Spend events from a nested
  `solwyn.run(...)` scope carry `parent_agent_run_id` — the raw id of the
  immediately enclosing run; root runs omit the field. Parent context restores
  after normal exits and exceptions, including across concurrent asyncio tasks.
- **`solwyn.current_run_context()` returns the active `RunContext(id, name,
  tags)`.** The tag mapping is a defensive copy, so caller mutation cannot
  change the active scope. `RunContext` and the attribution bounds
  (`TAGS_MAX_KEYS`, `TAG_KEY_MAX_LENGTH`, `TAG_VALUE_MAX_LENGTH`) are package
  exports.

### Changed

- **Nested `solwyn.run(...)` scopes inherit tags additively by default.** A
  child keeps all non-conflicting outer tags and overwrites only keys it
  supplies itself. `inherit_tags=False` starts a fresh tag scope, and exiting
  either form restores the exact outer context.
- **A combined tag map over 10 keys clamps without aborting the live LLM
  call.** The event keeps per-call keys first, then active-scope keys, then
  client-default keys, preserving insertion order within each layer and the
  highest-precedence value on conflicts. Each overflowing capture emits one
  `SolwynTagsClampedWarning`, exported from the package root; lower-priority
  excess tags are dropped from that event while the provider request continues.
- **Tag keys and values containing NUL are rejected eagerly.** This mirrors the
  control plane's storage and selector constraints, so invalid tags cannot turn
  budget checks into control-plane outage failures.

## [0.4.0] - 2026-07-28

The control plane comes off the hot path: run-scoped token leases replace the
per-call budget check, all settlement rides the reporter off the caller's thread,
and spend telemetry is delivered at least once across outages, forks, and
interpreter exit. Failover timeouts split the failover window from each hop's
read bound, post-success bookkeeping is fail-soft, and per-call and background
work is trimmed — the `openai` extra no longer installs `tiktoken`.
Wire-contract changes are API-first: Solwyn Cloud accepts every field below
before this SDK releases. Ships #36–#46; includes pre-launch breaking changes to
the enforcer and confirm-builder APIs, noted under *Changed*.

### Changed

- **The SDK now does less work on per-call and background hot paths while
  preserving routing behavior and wire-model validation.** Provider
  circuit-breaker reports are sent only when state changes, with a full refresh
  controlled by `breaker_report_heartbeat` /
  `SOLWYN_BREAKER_REPORT_HEARTBEAT` (default
  `60.0` seconds): a failed send remains due, `close()` attempts one final
  forced full snapshot within the shared shutdown deadline (an already-active
  cycle can consume that deadline before the send starts), and an idle reporter
  tick spawns no work when nothing is due.
  Default routing policies now declare whether they need latency medians or
  price hints, so unused signals are not computed; compatible streams stop
  accumulating response length once provider usage latches while continuing to
  extract usage and service-tier details; and trivial primary passthrough uses
  a narrow fresh-copy kwargs path. `RoutingRequest` is now a frozen dataclass.
  The unused `TokenizerManager` path and the `tiktoken` dependency were removed,
  so the `openai` extra no longer installs `tiktoken`. Validated Pydantic
  construction remains: a 100,000-iteration benchmark found `model_construct`
  was `2.905373 µs` slower per logical call, so it was not adopted.
- **The chain deadline no longer caps an in-flight hop's read.**
  `failover_total_timeout` (default `30.0`) is now purely the FAILOVER WINDOW:
  it bounds the budget pre-flight, each hop's connect/pool slice, Retry-After
  sleeps, and advancement between hops. Reads are bounded separately by the new
  `failover_hop_read_timeout`, so a call may now run up to roughly one failover
  window plus one hop read timeout — window expiry still gates advancement
  BETWEEN hops, so at most one hop per call can consume the full read bound.
  Previously a slow-but-connected provider was cut at 30 seconds and re-raised
  as `APITimeoutError`; because a read timeout is post-send ambiguous and
  re-raises WITHOUT failing over under the default
  `failover_idempotency="safe"`, that cut bought no failover — it only
  converted legitimately slow generations (reasoning models, large
  `max_tokens`) into ambiguous spend. Clients exposing `with_options` (openai,
  anthropic, every OpenAI-compatible provider, together) now receive a granular
  `httpx.Timeout` — connect/pool from the window slice, read/write from the hop
  read bound. Callers who prefer fast ambiguous failure lower
  `failover_hop_read_timeout`.
- **`failover_total_timeout` no longer bounds a google hop's pre-send phase.**
  google-genai supports only a single whole-request timeout — it cannot split
  connect from read — so Solwyn gives a google hop the read bound as its
  whole-request timeout. A google pre-send hang (TCP/TLS connect, pool wait)
  can therefore block up to `failover_hop_read_timeout` and exhaust the
  failover window without ever failing over. Lower `failover_hop_read_timeout`
  to tighten that guarantee; see the README's *Failover timeouts* section.
- **`check_budget` gained `call_id` and `estimated_output_bound`** (both
  keyword-only, both optional). `call_id` keys the in-process lease
  reservation the call draws down so settlement can true it up against actual
  usage — the same id the confirm and the metadata event already join on, now
  threaded one step earlier. `estimated_output_bound` is the conservative
  output half of that reservation: the `Solwyn` wrapper derives the largest
  effective cap across every configured provider hop after applying
  per-call-over-entry-over-global precedence and provider-specific aliases; a
  direct enforcer caller can supply the already-resolved scalar. A call with
  no explicit cap reserves `lease_output_bound_default` instead. A caller that
  supplies neither still admits — the enforcer mints a `call_id` and falls
  back to the default bound — so the signature remains additive for direct
  enforcer users. A direct caller that intends to settle or release a
  lease-funded call must supply and retain its own `call_id`; an anonymous
  synthetic reservation can only age out on the 900-second sweep.
- **`build_confirm_request` is now fully keyword-only and its settlement key
  is exclusive.** `reservation_id` moved from a required positional to an
  optional keyword and gained a sibling, `lease_id`: a confirm settles EITHER
  a per-call reservation or a lease, never both, matching the API's new
  exactly-one-of validator. Building a lease-keyed confirm also trues the
  call's local reservation up from its bound to actual token usage, so the
  in-memory remainder and the server's float move on the same event. Lease
  authority wins if a caller somehow passes both keys — the authority the call
  actually drew down is the one that must be settled. A lease-keyed confirm
  must also echo the process-local `lease_claim_token`; omitting it raises
  `RuntimeError` before local authority is mutated.
- **`BudgetCheckResult` gained `lease_id` and `lease_claim_token`.** `lease_id`
  is set when the call was admitted on lease authority instead of a per-call
  reservation. `lease_claim_token` is the opaque capability for that exact
  local call-ID claim; it is excluded from serialization and never reaches
  Solwyn Cloud. Direct enforcer callers thread whichever of `reservation_id`
  / `lease_id` is populated into `build_confirm_request`, plus
  `lease_claim_token` when `lease_id` is present. Exactly one settlement key
  is ever non-`None`. A lease-funded error path echoes the same token to
  `release_reservation`.

- **`call_id` now pins the canonical UUID text form** on both wires that carry
  it — `BudgetConfirmRequest.call_id` and `MetadataEvent.call_id` — matching
  what the API has required since its idempotency ledger landed
  (`^[0-9a-f]{8}-...$`, max 36 chars). The top-level wrapper has only ever
  emitted `str(uuid.uuid4())`, so nothing it sends changes. Direct
  `check_budget` callers must provide the canonical form when a run-scoped
  text call participates in leasing; non-run, media, and lease-disabled calls
  preserve the legacy behavior and do not validate an otherwise-unused
  descriptive value. A drifted wire id now fails at the seam that built it
  instead of arriving as a 422 that loses the settlement it was carrying.
  `call_id` is durable spend identity — the API's cost-event ledger dedups on
  it — and it is the join key between an event and its confirm, so both halves
  answer to one shape.

- **Non-streaming settlement moved off the caller's hot path.** Sync and async
  chat completions and the whole media lifecycle (embeddings, images, audio,
  video) previously settled a reservation with a BLOCKING `confirm_cost` POST on
  the caller's thread AFTER the provider had already answered — the response was
  withheld until Solwyn's `/budgets/confirm` round-trip returned. Settlement now
  builds the confirm sans-I/O via `build_confirm_request` and enqueues it with
  its metadata event as one ordered item through
  `reporter.report_settlement(confirm, event)` — the exact path streaming
  `on_complete` already used. The caller gets the provider response without
  waiting on any Solwyn round-trip.
- **`BudgetEnforcer.confirm_cost` / `AsyncBudgetEnforcer.confirm_cost` are
  removed** (pre-launch, breaking). The reporter owns settlement delivery via
  `_send_confirm`, and it already carries the consecutive-confirm-failure
  ERROR-escalation convention (after 10 consecutive failures, logs at ERROR).
- **A shared control-plane circuit breaker discovers a Solwyn outage once, not
  once per call.** One `CircuitBreaker(name="control-plane")` per client guards
  BOTH the `/budgets/check` POST (budget enforcer) and the `/budgets/confirm`
  POST (reporter). A streak of consecutive failures against Solwyn's own API
  opens the breaker so the next check/confirm short-circuits — applying the
  configured posture (fail-open / local enforcement) or dropping the confirm
  instantly — instead of paying the per-request timeout again. A read-only-key
  response means Solwyn RESPONDED, so it records success; the control-plane
  breaker is never a provider breaker (excluded from breaker reports).
- **The budget pre-flight timeout dropped to 1.0s by default.** The
  `/budgets/check` POST gates the caller's hot path, so its per-request timeout
  is now a short `budget_check_timeout=1.0` (was 5s); the control-plane breaker
  caps repeated discovery of an outage.

- **The async reporter auto-starts its flush loop on first enqueue.** The sync
  `MetadataReporter` is live from construction (its daemon flush thread starts in
  `__init__`), but `AsyncMetadataReporter` previously flushed only after an
  explicit `start()` / `async with AsyncSolwyn`. Constructed without `async
  with`, it queued events AND budget-confirm settlements silently until
  `close()` — and confirms are settlement data, so server-side spend tracking
  drifted. `report()`, `report_confirm()`, and `report_settlement()` now start
  the flush loop on the first enqueue when a running event loop is present;
  called with no running loop, the event stays queued and a single warning per
  reporter instance is logged (enqueue never raises — it is on the LLM call
  path, so fail-loud was not an option there). `start()` is now idempotent (a
  second call reuses the live flush task instead of orphaning it and resetting
  the shutdown event), and `start()` after `close()` raises `RuntimeError` —
  restarting a closed reporter is a programming error. Enqueue after `close()`
  is dropped and counted (`closed_enqueue`), matching the sync reporter.
- **Spend telemetry is now delivered at least once.** Confirms, settlements, and
  metadata batches used to be dropped on their first failed send. They are now
  retried with bounded exponential backoff (`reporter_max_send_attempts`,
  `reporter_retry_backoff_base`, `reporter_retry_backoff_cap`). A transient
  failure — `httpx.TransportError` or HTTP 408/429/5xx — is retried; every other
  status is terminal, so a poison item can never wedge the queue head. The
  server dedups duplicate sends via an idempotency ledger, so the SDK retries
  freely and never does client-side dedup.
- **A breaker-open flush HOLDS settlements instead of dropping them.** When the
  shared control-plane breaker refuses admission, the confirm is now kept for a
  later cycle (it previously dropped the confirm at DEBUG, silently losing
  acknowledged spend for the duration of a Solwyn outage). Breaker accounting is
  unchanged: a refusal is not an attempt, so it touches neither the breaker nor
  the consecutive-confirm-failure counter.
- **A settlement's metadata event now survives its confirm.** When a settlement's
  confirm is terminally rejected or exhausts its retries, the paired event is
  still handed to the ingest queue — `cost_events` ingest is the durable spend
  truth and must not be lost because the confirm failed. An overflow-evicted
  settlement ships its event the same way; a settlement rejected after
  `close()` counts BOTH halves (`settlement_confirm.closed_enqueue` and
  `event.closed_enqueue`).
- **Confirm and settlement queues drain strictly FIFO.** A head that fails
  transiently is requeued and PARKS its queue for the cycle — later spend can
  no longer be confirmed or ingested ahead of earlier acknowledged spend, and
  an outage burns one retry attempt per cycle instead of one per queued item.
- **Undeliverable spend is counted and loudly logged, never silently dropped.**
  Queue overflow, retry exhaustion, terminal statuses, per-event rejections
  inside a 202 ingest body, shutdown-deadline expiry, and enqueue-after-close
  all increment a counter now readable via the new `dropped_counts` property. The first drop logs immediately; after that at
  most one aggregated `reporter.spend_events_dropped` WARNING per 60s, so a
  sustained outage reports loss without flooding logs.
- **Queued spend survives interpreter exit.** A process that exits WITHOUT
  calling `close()` previously discarded everything still queued. A single
  `atexit` hook now flushes each live reporter (sync via `close()`, async over a
  temporary sync client since no event loop exists at exit), and async reporters
  additionally arm a GC-only `weakref.finalize` covering the constructed-queued-
  then-garbage-collected case — GC-only (`atexit` disabled on the finalizer)
  because weakref's own exit hook runs BEFORE the SDK's and would drain live
  reporters without accounting; a genuine GC drain logs each loss
  (`lifecycle.gc_flush_dropped`) since the collected reporter's
  `dropped_counts` no longer exists. The exit drain is a true wall-clock bound:
  it runs on a daemon worker joined at the deadline — closing a sync HTTP
  client does not reliably interrupt a blocked read, so no close()-based abort
  is trusted — with pop-and-claim ownership mirroring the reporter's close
  seal: a timed-out join claims the worker's in-hand items and sweeps the
  queues (counted `shutdown_deadline`), so a slow-drip response (invisible to
  httpx's per-socket-op timeouts) cannot hold process exit and a worker
  unblocking late never double-counts. Lossy dispositions publish atomically
  with their ownership release (under the same lock), so the seal can never
  land between the two and let the drain return with an item accounted
  nowhere while process exit kills the worker. Exit delivery is
  accountable per item: every popped
  confirm and event reports a sent/failed/expired disposition and every failure
  is counted. Exit confirms ride the control-plane breaker's admission — a
  known-down control plane refuses them (counted `exit_breaker_open`) after at
  most one recovery probe — while metadata ingest is never breaker-gated, so
  settlement events and standalone events still get their deadline-bounded
  ingest attempt on the way out.
- **Reporters, budget enforcers, and circuit breakers are fork-safe.** Threads,
  locks, and HTTP clients do not survive `fork()`, so a forked child inherited a
  dead flush thread and never delivered its settlements. A single
  `os.register_at_fork` hook (skipped on platforms without fork) now rebuilds
  locks, swaps in fresh HTTP clients (abandoning — never closing — the parent's
  sockets), and relaunches the sync flush thread on the child's next enqueue.
  Breaker health state is deliberately inherited; queued items duplicated by
  fork are deliberately kept, since the server dedups.
- **`close()` is bounded by one shutdown deadline.** `close(timeout=...)` (sync
  and async; default `reporter_shutdown_deadline`) now shares a single monotonic
  deadline across the thread/task join, the final flush, and the breaker-report
  cycle. Against a black-holed control plane, shutdown no longer pays a serial
  per-request timeout chain across every queued item — work still queued when
  the deadline is reached is counted `shutdown_deadline` and dropped. The
  deadline is a true wall-clock bound: httpx timeouts cap individual socket
  operations, not total response time, so the final flush runs off the closing
  thread (sync: a daemon worker joined at the deadline; async: a task the
  deadline cancels) and a slow-drip response cannot hold `close()` open. At the
  deadline `close()` (sync AND async) seals delivery and takes final ownership
  of ALL undelivered spend: items a stuck flush thread still holds mid-POST and
  enqueues racing the final drain are counted before `close()` returns, never
  requeued into a queue nothing drains. An async `close()` cancelled mid-flush
  keeps the atexit hook and GC finalizer armed as the last delivery path — they
  detach only once close actually completes — and a cancelled drain REQUEUES its
  in-hand item under the ownership gate rather than writing it off: the rescue
  paths can only retry what is in the queues, so a completing close's seal
  counts the requeued item while a cancelled close leaves it deliverable. Even
  a zero-deadline close awaits its cancelled drains' cleanup so every in-hand
  item is back in its queue for the seal before rescue detaches.

### Added

- **`failover_hop_read_timeout` (default `600.0`) bounds each hop's
  read/write**, decoupled from the failover window. `600.0` matches the
  openai/anthropic SDK's read/write default, so a wrapped call's read/write
  bound never fires earlier than the unwrapped SDK's would; connect/pool
  instead track the shrinking failover window. It is constructor-only
  (deliberately no `SOLWYN_*` env var) and server-governed: on a plan without
  the failover-tuning entitlement a custom value is suppressed back to `600.0`,
  warned once per client. Values must be greater than zero.
- **Building a Bedrock client with an unbounded botocore read timeout now warns
  at build time.** Solwyn cannot bound a Bedrock hop per call (boto3 has no
  per-call timeout override), so the caller's botocore
  `Config(read_timeout=...)` governs — and `read_timeout=None` is the one shape
  neither Solwyn nor botocore will ever bound, letting a single stuck Converse
  read hang the call indefinitely. Detection stays zero-import and defensive:
  an unreadable or absent config reads as bounded.
- **Run-scoped token leases replace the per-call `/budgets/check` on the hot
  path.** A call carrying an `agent_run_id` used to pay a blocking budget
  round-trip to Solwyn before every provider request. The enforcer now takes
  ONE server-granted lease per run (`POST /api/v1/budgets/lease`), draws it
  down in memory, renews it ahead of need off the caller's thread
  (`/budgets/lease/renew`), and hands it back when the run is done
  (`/budgets/lease/surrender`) — the DHCP shape, with the same reason for
  being: authority is delegated for a bounded window so the client can keep
  serving without asking permission per call. The lease is denominated in
  TOKENS, never dollars; the server folds price into `granted_tokens` and the
  SDK still never computes cost. Admission is a sans-I/O state machine
  (`_lease.py`) shared by the sync and async enforcers: each call atomically
  reserves `estimated_input + output_bound` out of the granted remainder under
  one lock, and settlement trues that reservation up to actual usage
  (overshoot may drive the remainder negative — the next renewal nets it out).
  A reservation stranded by an error path that never settles is swept at 900s.
  Renewals are ALWAYS asynchronous — a thread for the sync enforcer, a task
  for the async one — and are driven by admission (≥75% depleted, or the
  refresh deadline passed) rather than by a timer, so an idle run costs
  nothing: breaker-guarded, jittered, one in flight at a time, with 1s→30s
  full-jitter backoff. Every grant carries a `generation`; an advancing
  generation is required before it installs, and a renewal additionally
  applies only while its originating lease id and generation still match, so
  a slow response can neither rewind the ledger nor mutate a replacement
  lease. A same-lease renewal lands net of spend settled after its request
  snapshot and every still-live reservation in both the granted and
  headroom-share pools, so the replacement generation never recreates
  authority already spent or committed. A grant that arrives
  `final_grant=True` logs a wind-down warning; a deny verdict at grant or
  renewal feeds the existing sticky-deny machinery unchanged.
- **The lease admission ladder never lets a Solwyn outage block a customer
  call.** Every deny the lease path can produce traces to a customer-chosen
  verdict, never to unreachability. With the granted remainder exhausted and
  the control plane believed UP, the call falls back to today's per-call
  check — an empty wallet is not a refusal, and the server stays
  authoritative. With the plane unreachable, the call draws instead on
  `headroom_share_tokens`, the run's apportioned slice of real remaining
  headroom: admitted with a warning, still metered. Only when that share is
  genuinely exhausted does the CUSTOMER's own mode decide — `hard_deny`
  denies, `alert_only` admits and warns. Past the lease deadline with the
  plane still down, the grant's `posture.on_unreachable` (the client's own
  configured `fail_open`, echoed back by the server so the grant is one
  self-describing artifact) picks the floor: `fail_open` admits UNCOUNTED,
  warning once on entry to the episode and then at most 1/30s, tallying every
  such call for the next successful renewal to report; `local_enforce` meters
  against the freshest known share remainder and denies at the customer's mode
  when it is spent. Structured refusals are absorbed rather than propagated: a
  `409 lease_holder_cap_exceeded` marks the run lease-ineligible (legacy path),
  `404 lease_not_found` / `409 lease_generation_conflict` drop the local lease
  so the next admission asks for a fresh grant, and `503 lease_unavailable`
  parks lease attempts for 30s while the legacy path — which has its own
  fail-open — carries the traffic. Non-run calls, non-text modalities, calls
  carrying `estimated_media`, and calls whose model or fallback chain is
  outside the lease's declared set take the legacy per-call path automatically.
- **A held lease is handed back at close AND at interpreter exit.** The
  enforcer's `close()` first seals the lifecycle, captures every unacknowledged
  renewal delta, and atomically drains the ledger before waiting or doing I/O.
  It then surrenders every captured lease best-effort on one short deadline.
  A grant or renewal landing after that fence can never repopulate local state:
  its returned authority is surrendered instead, together with any spend its
  predecessor snapshot did not report. The shipped `atexit` machinery performs
  the same DHCPRELEASE-style cleanup for a process that exits without closing,
  letting the server re-lend float immediately instead of waiting out the lease
  deadline. Exit surrenders ride the control-plane breaker's admission exactly
  like exit confirms (a known-down plane refuses them instantly) and run on a
  daemon worker joined at a wall-clock deadline, so an unreachable Solwyn can
  never hold up process exit. Unlike queued spend, nothing here is worth
  waiting for — an unsurrendered lease expires on the server anyway.
- **New lease knobs (with `SOLWYN_*` env vars):** `lease_enabled`
  (`SOLWYN_LEASE_ENABLED`, default `True`) is the kill switch — `False` routes
  every call back to the per-call check path, with no lease state built at all;
  `lease_output_bound_default` (`SOLWYN_LEASE_OUTPUT_BOUND_DEFAULT`, default
  `4096`) bounds the output half of a reservation for a call that declares no
  `max_tokens`-family cap, and defaults to the ledger's own constant so the two
  can never drift.
- **New reporter delivery knobs (with `SOLWYN_*` env vars):**
  `reporter_max_send_attempts` (`SOLWYN_REPORTER_MAX_SEND_ATTEMPTS`, default
  `5`), `reporter_retry_backoff_base` (`SOLWYN_REPORTER_RETRY_BACKOFF_BASE`,
  default `1.0`), `reporter_retry_backoff_cap`
  (`SOLWYN_REPORTER_RETRY_BACKOFF_CAP`, default `60.0`), and
  `reporter_shutdown_deadline` (`SOLWYN_REPORTER_SHUTDOWN_DEADLINE`, default
  `5.0`).
- **`MetadataReporter.dropped_counts` / `AsyncMetadataReporter.dropped_counts`**
  expose undeliverable spend as a `{"kind.reason": count}` mapping, with kinds
  `confirm` / `settlement_confirm` / `event` and reasons `overflow`,
  `retry_exhausted`, `terminal_status`, `ingest_rejected`, `shutdown_deadline`,
  `exit_breaker_open`, and `closed_enqueue`.
- **New config knobs (with `SOLWYN_*` env vars):**
  `budget_check_timeout` (`SOLWYN_BUDGET_CHECK_TIMEOUT`, default `1.0`),
  `control_plane_failure_threshold`
  (`SOLWYN_CONTROL_PLANE_FAILURE_THRESHOLD`, default `3`), and
  `control_plane_recovery_timeout`
  (`SOLWYN_CONTROL_PLANE_RECOVERY_TIMEOUT`, default `30.0`).
- **`CircuitBreaker` gained a `name` label** (default `"provider"`) that
  distinguishes health domains in the transition log lines
  (e.g. `Circuit breaker [control-plane] opened due to failures`).

### Fixed

- **A connect/pool timeout wrapped by the provider SDK now fails over.** Both
  openai and anthropic wrap the ENTIRE `httpx.TimeoutException` family in
  `APITimeoutError`, so a provably pre-send `ConnectTimeout`/`PoolTimeout`
  reached classification wearing the same class name as a post-send read
  timeout — and was classified post-send ambiguous, which under the default
  `failover_idempotency="safe"` re-raises without ever trying the next
  candidate. A stalled primary therefore took the whole call down instead of
  failing over. `APITimeoutError` is now classified by its chained cause, the
  same way `APIConnectionError` already was: a pre-send httpx cause fails over,
  read/write/unknown/no-cause stays ambiguous. This matters most now that the
  connect slice is short and deadline-derived (above) — it is the bound that
  fires first on a stalled provider.
- **A media call whose failover window expired during the budget pre-flight no
  longer starts provider I/O.** With connect and read decoupled (above), the
  expired path's floor connect slice is satisfied by a warm pooled connection,
  after which the hop could read for the full hop read bound — so whether an
  out-of-window media call escaped came down to pool state. Both the sync and
  async media paths now gate on window expiry before dispatch and release the
  reservation, matching the chat walk.
- **Server-pushed failover tuning is now snapshotted once per call.** The
  directive writer mutates the config's tuning fields under the breaker lock,
  while the dispatch path re-read them unlocked mid-call — so a call could
  observe a torn mix of old and new tuning (e.g. the new total timeout with the
  old idempotency mode). Each call now captures one immutable `FailoverTuning`
  snapshot under that same lock and consumes only that.
- **Lease call IDs are fenced for one bounded call lifecycle instead of the
  client's entire lifetime.** Every lease-participating outcome — including a
  cold-start grant and a dynamic legacy fallback — claims the reconciliation
  id before I/O. A second owner reusing it during the 900-second retention
  window raises `RuntimeError` before a second drawdown. Claims expire from a
  heap-backed map, and each later reuse receives a new opaque token, so a stale
  owner cannot re-enter, settle, or release its successor after an ABA-style
  id reuse.
- **Renewal and shutdown races no longer lose spend or resurrect authority.**
  Renewal results are fenced to their originating lease generation;
  post-snapshot settlements and in-flight bounds carry into the successor
  generation exactly once. Close seals and drains state atomically, late
  initial grants are surrendered rather than installed, and async close waits
  for or cancels renewal tasks within the same shared deadline.
- **Output-cap alias precedence now matches dispatch on every provider hop.**
  Per-call caps beat provider-entry and global defaults even when one layer
  uses `max_tokens` and another uses `max_completion_tokens`, including
  pre-`gpt-5` OpenAI/Azure models, OpenAI-compatible failovers, and
  cross-dialect translation. Lease reservations therefore cannot silently use
  a lower-priority, larger cap.
- **A refused `HALF_OPEN` control-plane follower now records fail-open usage.**
  When another call owns the recovery probe, an allowed call without usable
  lease authority enters the same uncounted episode and tally as every other
  control-plane outage path, so its debt reaches the next successful renewal.
- **Breaker-report writes now recognize read-only keys.** The breaker-report
  POST was the only Cloud write not routed through the one-time read-only-key
  diagnostic (#32): with a key that can read budget checks but not write, each
  5-second reporter cycle emitted a `reporter.breaker_send_failed` warning per
  provider — the exact noise the diagnostic exists to collapse. A read-only 403
  on the breaker path now logs `solwyn.configuration_error.read_only_key` once
  per process and ends the cycle instead of posting the remaining doomed
  snapshots; all other breaker-send failures still warn per provider.

### Documentation

- **`BudgetCheckResponse.denied_by_period` relaxation is intentional and now
  documented + test-enforced.** The field moved from required-nullable to
  defaulted in #33 without a stated rationale. Verified against the live API:
  directive-v1 check responses (the only wire this SDK requests — every check
  opts in via `failover_directive_version: "1"`) are serialized exclude-none,
  so ALLOW responses omit the key entirely; restoring `Field(...)` would
  reject every live allow response, and requiring it on deny would misread the
  server's legitimate period-less deny edge case as a validation failure
  (which fails open). The posture is pinned by unit tests
  (`test_contract_snapshot.py`) and, because a defaulted field can no longer
  fail loudly on server drift, the deny-side shapes the SDK keys on
  (`denied_by_period`, including the run-scoped `"agent_run"` literal, and
  `failover_directive.failover_tuning_allowed`) are now verified against the
  live API by `tests/integration/test_live_contract.py`.

### Reliability

- **Post-success bookkeeping is now fail-soft.** Usage/region/tier extraction
  in sync/async non-streaming and media success blocks, and region extraction
  in streaming pre-wrapper and error-event paths, now degrade on adapter raise:
  usage degrades to an estimate (`is_estimated=true`), while region/tier degrade
  to `None` (omitted from wire) — instead of destroying the paid provider
  response. (`_translation.normalize_response(...)` still raises loudly per
  contract.) (R5).
- **An UNMEASURABLE call settles its lease at the reserved bound.** When every
  usage read for a call raises, the synthetic fallback carries the pre-flight
  input estimate and no output at all. `build_confirm_request` now takes
  `usage_unmeasured` (keyword-only, default `False`): the wire confirm is
  unchanged — the honest `is_estimated=true` under-measure the cloud
  reconciles — but the local `LeaseLedger.true_up` settles that call at its
  reserved bound instead of crediting the untouched-looking output allowance
  back. Refunding it would re-lend authority a paid response already consumed
  and let later admissions exceed the run's hard token cap (R5).
- **Budget check distinguishes unparseable 2xx bodies from transport outages.**
  When Solwyn returns a 2xx status with an unreadable response body, it now
  logs `budget.check_response_unreadable` at ERROR level and records breaker
  success (server contract drift), distinct from transport-level failures that
  record breaker failure. The distinction is implemented in both sync and async
  enforcers (R6).

---

## [0.3.0] - 2026-07-16

Run-scoped budget enforcement, explicit customer tags, and server-governed
failover tuning. Wire-contract changes are API-first: Solwyn Cloud accepts every
field below before this SDK releases.

### Added

- **Run scopes enforce per-run caps.** Every pre-flight check made inside
  `solwyn.run(...)` now carries the active `agent_run_id`, so Cloud can enforce a
  cap scoped to a single run. Run-scoped checks bypass the SDK's global allow
  cache — a cached allow for one run can never authorize another — and a run
  denial (`denied_by_period == "agent_run"`) sticks to that run alone rather than
  replacing global budget state, so one capped run cannot deny an unrelated one.
  Cloud usage visibility stays asynchronous: with the defaults, the conservative
  transition/leak upper bound after a cap is crossed is ~10 seconds (the 5-second
  global allow-cache TTL plus the reporter's 5-second flush interval). (#29)
- **Bounded customer tags.** `solwyn.run(name, tags={...})` and the reserved
  per-call `solwyn_tags={...}` keyword carry explicit customer metadata for
  grouping and export — never derived from prompts or responses. Per-call keys
  shallow-merge over run tags. The merged mapping admits at most 10 keys, keys of
  1–64 characters, and values of 0–256 characters; a mapping that exceeds any
  bound is REJECTED, never silently truncated, so attribution is never quietly
  wrong. `solwyn_tags=` is stripped before provider dispatch (pass it as a call
  argument, not in `default_params`). Mappings are copied at scope entry and call
  start, so later caller mutation cannot change attribution. (#31)
- **Circuit breaker state reporting.** The reporter now piggybacks a
  `BreakerStateReport` snapshot per provider — state, failure/success counts,
  snapshot time, and the bounded `sdk_instance_id` — onto its existing flush
  cycle, making SDK-local provider health visible server-side instead of trapped
  in-process. Reporting is per-instance and self-limiting: it engages only once a
  project id has been learned from a Cloud response, and a snapshot failure is
  logged by exception type and skipped, never raised onto the call path. The
  breaker remains authoritative and entirely process-local — only its state is
  reported, and nothing Cloud returns steers it. Governed by the new
  `breaker_reporting_enabled` config field (env var
  `SOLWYN_BREAKER_REPORTING_ENABLED`), default enabled; set it false to opt out
  of breaker snapshots entirely. (#30)
- **Failover tuning entitlements.** Budget checks opt into a versioned
  `FailoverDirective` (`version: "1"`), whose `failover_tuning_allowed` flag
  governs exactly seven customer-configurable fields: `failover_total_timeout`,
  `failover_idempotency`, `same_provider_retries`,
  `circuit_breaker_recovery_timeout_jitter`,
  `circuit_breaker_failure_threshold`, `circuit_breaker_recovery_timeout`, and
  `circuit_breaker_success_threshold`. Enforcement is ADVISORY: when tuning is
  not entitled, plan defaults are applied (retuning existing breakers in place)
  without changing provider order and without failing calls; absent policy
  delivery leaves behavior unchanged. Provider entries and routing policy are
  deliberately outside the boundary. (#33)
- **OpenAI prompt cache-write usage.** The OpenAI extractor now reads
  `cache_write_tokens` off the prompt token details (Chat Completions) and input
  token details (Responses API) — covering every OpenAI-compatible profile and
  the stream accumulators — and reports it as `cache_creation_5m_tokens`.
  OpenAI's cache writes are its provider-default/minimum 30-minute writes; they
  ride the dataset's existing 5m cache-write bucket to preserve the wire
  contract, rather than adding a per-TTL field. A response without the bucket
  reports 0 exactly as before. (#28)
- **Read-only project key diagnostics.** A Cloud 403 carrying the structured
  `{"detail": {"code": "read_only_key"}}` contract is recognized and logged once
  per process at ERROR level as an actionable configuration diagnostic, instead
  of surfacing as a generic budget-check failure. The match is exact — only the
  structured contract — and the response body is never exposed. (#32)
- **Preserved hard denies are logged.** When a budget check cannot reach Cloud
  while a prior authoritative hard deny is on record — the project-period deny,
  or a run's sticky deny — the denial is preserved instead of failing open, and
  the SDK now logs a WARNING on the `solwyn.budget` logger naming the usage and
  limit (`Cloud API unreachable; preserving prior hard deny: $99.50/$100.00
  used`). The warning is emitted on every affected call, not once per process,
  so a sustained outage under a hard deny stays visible for its whole duration;
  the `BudgetCheckResult.warning` field is unchanged. (#34)
- **`UnsupportedSurfaceError` exported from the package root.** `from solwyn
  import UnsupportedSurfaceError` now works and the class joins `__all__`,
  alongside every other exception. The deep import from `solwyn.exceptions`
  keeps working — the export is additive. (#34)

### Changed

- **`AGENT_RUN_ID_MAX_LENGTH` raised from 255 to 256.** The validation cap for
  `agent_run_id` is now 256 characters. The bound only widens, so no previously
  valid id becomes invalid; SDK-generated run ids are far shorter and unaffected.

---

## [0.2.0] - 2026-07-11

Non-text modality support — SDK merge 1. Wire-contract changes are API-first:
Solwyn Cloud accepts every field below before this SDK releases.

### Added

- **Embeddings are now budget-tracked.** `client.embeddings.create` (native
  OpenAI and every OpenAI-compatible profile — xAI, DeepSeek, Mistral, Qwen,
  Groq, Together, Fireworks, Perplexity, Azure OpenAI, OpenRouter, Z.ai, Ollama,
  vLLM, LM Studio, and the generic catch-all) and Google's
  `client.models.embed_content` are intercepted through a lean media-call
  lifecycle: estimate → budget check → primary-only provider call →
  extract/measure → confirm + report. Embeddings emit `modality="embedding"` on
  the budget check, the confirm, and the metadata event. There is no candidate
  walk, no cross-provider translation, and no model fallback for media surfaces
  (embedding vectors are not interchangeable across providers).
- **`modality` wire field.** `MetadataEvent`, `BudgetCheckRequest`, and
  `BudgetConfirmRequest` each carry `modality`
  (`Literal["text","image","audio","video","embedding"]`, default `"text"`).
  Chat calls ride the `"text"` default and are behaviorally unchanged; media
  surfaces set the surface's modality. The server's pricing-card unit — not this
  label alone — selects the billing basis (core bills text tokens until
  per-modality cards land).

### Changed

- **Estimation fallback extended to embeddings.** When an embeddings response
  reports no usable usage block, billable input is derived from a length-based
  estimate off `input=` (str / list[str] via provider char-ratio; list[int] /
  list[list[int]] counted directly as pre-tokenized ids), flagged
  `token_details.is_estimated = true`, never a silent zero. Google's
  `EmbedContentResponse` currently exposes no `usage_metadata`, so Google
  embeddings are estimator-driven in practice; the usage-extraction path stays
  forward-compatible if Google adds it. An unobservable quantity settles as
  `None` — never a real $0 price.
- **Warn-once pass-through posture for unshipped spend surfaces.** Recognized but
  not-yet-intercepted billable surfaces (OpenAI-dialect `images` / `audio` /
  `videos`; Google `generate_images` / `generate_videos`) warn exactly once per
  process, then pass through untracked. The latch is per-process (not
  per-instance), so the create-a-client-per-request pattern stays quiet. Truly
  unrelated resources (files, moderations, batches, …) pass through silently.

### Fixed

- **Bedrock `start_async_invoke` now fails loud.** It joins `invoke_model` /
  `invoke_model_with_response_stream` in raising `ConfigurationError` instead of
  bypassing budget tracking — a video-scale spend surface whose usage lands
  out-of-band in S3 was a silent budget hole. Use Converse, or call the
  unwrapped client for deliberately untracked calls.

---

Non-text modality support — SDK merge 2. Wire-contract changes are API-first:
Solwyn Cloud accepts every field below before this SDK releases.

### Added

- **Image generation is now budget-tracked.** `client.images.generate` and
  `client.images.edit` (native OpenAI and every OpenAI-compatible profile,
  including Together) are intercepted through the media-call lifecycle, emitting
  `modality="image"`. Native gpt-image reports token usage carrying per-modality
  buckets (`image_input_tokens` / `image_output_tokens`, documented subsets of
  input/output); compat, dall-e, and FLUX endpoints report no usage, so the
  billable basis is the request-derived per-image `MediaUsage`
  (`n` → `image_count`, `size` → `resolution`, `quality` → `quality`). Both bases
  ride the confirm when observable and the server's pricing-card unit picks
  (token card vs per-image card). The operation (generate vs edit) rides a
  private marker that is stripped before the provider call; the customer
  `prompt=` / `image=` / `mask=` bytes are never read.
- **Google `generate_images` (imagen) is now budget-tracked.**
  `client.models.generate_images` is intercepted through the same lifecycle,
  emitting `modality="image"`. imagen exposes no usage, so the request-derived
  per-image quantity is the sole billable basis: `config.number_of_images`
  (read duck-typed from both the config-object and dict shapes, defaulting to 1
  per the API contract) → `image_count`. An exact count, never a silent $0.
- **Google `usageMetadata` per-modality token buckets.** The Google chat
  extractor now maps the `prompt_tokens_details` (input side) and
  `candidates_tokens_details` (output side) `ModalityTokenCount` lists: IMAGE
  buckets → `image_input_tokens` / `image_output_tokens`; AUDIO buckets →
  `audio_input_tokens` / `audio_output_tokens`. This lets a
  `generate_content` call to a token-billed image model (e.g. gemini-3-pro-image)
  carry its image-output tokens so the server prices them at the image rate.
  Duck-typed with None-safety throughout; a response without per-modality details
  leaves every field at 0 exactly as before.
- **`MediaUsage` wire type + `estimated_media` + image token keys (wire window
  2).** `MediaUsage` — the non-token billable quantities and variant selectors a
  per-unit-priced surface bills on (`image_count`, `generation_count`,
  `video_seconds`, `audio_seconds`, `input_characters`, `resolution`, `quality`)
  — is vendored inline (the SDK is self-contained) and carried on
  `MetadataEvent.media_usage` and `BudgetConfirmRequest.media_usage`;
  `BudgetCheckRequest.estimated_media` carries the request-derived pre-flight
  quantity so the budget check prices a precise per-unit cost. `TokenDetails`
  gains `image_input_tokens` / `image_output_tokens`. Every quantity is `None`
  (never a zero-as-default) when unobservable, routing a known-unit card with a
  missing quantity to the server's unpriced lane rather than a real $0.

### Changed

- **`images` and Google `generate_images` graduated from warn-once to
  intercepted.** Both leave the unshipped-spend-surface warn-once set — a metered
  surface must not advertise itself as untracked (superseding the merge-1
  warn-once posture entry for those two surfaces). Remaining warn-once
  pass-throughs: OpenAI-dialect `audio` / `videos` and Google
  `generate_videos`.

---

Non-text modality support — SDK merge 3. Audio surfaces reuse the wire contract
shipped in merge 2 (`modality="audio"`, `MediaUsage.audio_seconds` /
`input_characters`, `TokenDetails.audio_input_tokens`) — no new wire fields.

### Added

- **Audio transcription is now budget-tracked.**
  `client.audio.transcriptions.create` (native OpenAI and every
  OpenAI-compatible profile, including Groq whisper) is intercepted through the
  media-call lifecycle, emitting `modality="audio"`. One extractor discriminates
  on `usage.type`: the token-billed models (the gpt-4o-transcribe family) settle
  on their audio-input token bucket (`audio_input_tokens`), while whisper reports
  an integer whole-second count (`usage.seconds`, present on any JSON
  `response_format`) that rides `MediaUsage.audio_seconds` — the fractional
  top-level `duration` field is never read. A non-JSON `response_format`
  (`text` / `srt` / `vtt`) returns a plain string with no usage, so the call is
  tracked UNPRICED with a one-time hint to pass a JSON `response_format` for
  priced tracking. The audio file bytes are never read.
- **Text-to-speech is now budget-tracked.** `client.audio.speech.create` (native
  OpenAI and every OpenAI-compatible profile) is intercepted through the same
  lifecycle, emitting `modality="audio"`. TTS responses carry no usage, so the
  sole billable basis is the request's `input` text length, measured in the
  privacy firewall as an exact character count (`MediaUsage.input_characters`)
  that rides both the pre-flight budget check and the settled confirm; the input
  text itself is never read, logged, or transmitted (tts-1 / tts-1-hd price from
  this character count server-side). Token-billed TTS models (`gpt-4o-mini-tts`
  and its dated snapshots) publish no usage of any kind — their audio-output
  tokens are unobservable, so by product decision those calls are warned once per
  process and passed through UNTRACKED rather than settled at an estimated or
  silent $0.

### Changed

- **`audio` graduated from warn-once to intercepted.** The `audio` attribute now
  returns intercepting machinery — its `transcriptions` and `speech` sub-surfaces
  are metered — leaving the unshipped-spend-surface warn-once set (a metered
  surface must not advertise itself as untracked, superseding the merge-1
  warn-once posture entry for `audio`). Its `translations` sub-surface stays
  recognized-but-untracked and warns once per process. Remaining warn-once
  pass-throughs: OpenAI-dialect `videos` and Google `generate_videos`.

---

Non-text modality support — SDK merge 4. Video surfaces reuse the wire contract
shipped in merge 2 (`modality="video"`, `MediaUsage.video_seconds` /
`resolution`) — no new wire fields.

### Added

- **Google `generate_videos` (veo) is now budget-tracked.**
  `client.models.generate_videos` is intercepted through the media-call
  lifecycle, emitting `modality="video"`. Video generation is asynchronous — the
  call returns a long-running operation carrying no usage — so billing settles at
  INITIATION from request params: `config.duration_seconds` (read duck-typed from
  both the config-object and dict shapes) at `config.resolution` →
  `MediaUsage.video_seconds` / `resolution`, always marked `is_estimated=true`.
  The pre-flight budget check prices a precise per-second cost, so an oversized
  request is denied before the provider is called; the estimate is a deliberate,
  conservative over-count (the provider does not charge for failed or blocked
  generations). google-genai publishes no default duration, so an absent
  `duration_seconds` stays `None` (tracked unpriced) rather than a guessed value.
  The customer `prompt=` and any seed image bytes are never read; the returned
  operation object passes through untouched — callers poll it themselves.
- **OpenAI `videos.create` (Sora) is now budget-tracked.** `client.videos.create`
  is intercepted through the same lifecycle, emitting `modality="video"`. Sora
  returns an async video job carrying no usage, so billing likewise settles at
  INITIATION from the top-level request params: `seconds` (a digit string per the
  API reference, or an int duck-typed) → `MediaUsage.video_seconds`, and `size`
  normalized to a resolution LABEL — `min(width, height) + "p"`, so
  `1280x720`/`720x1280` → `720p`, `1792x1024`/`1024x1792` → `1024p`,
  `1920x1080`/`1080x1920` → `1080p` — matched against the server's per-second
  variant grid (an unparseable size passes through raw so the server fails loud on
  a miss rather than mispricing). `is_estimated` is always `true`. OpenAI's API
  reference documents stable defaults (`seconds` `"4"`, `size` `720x1280`), so an
  omitted param settles the documented value — billing what the provider applies
  is faithful — while a present-but-garbage duration stays `None` (tracked
  unpriced) rather than a guessed value. The customer `prompt=` and reference-image
  bytes are never read; the returned video job passes through untouched — callers
  poll it themselves. Sora is OpenAI-only, so `videos.create` on an
  OpenAI-compatible client fails loud with `UnsupportedSurfaceError`.

### Changed

- **`videos` and Google `generate_videos` graduated from warn-once to
  intercepted.** Both leave the unshipped-spend-surface warn-once set — a metered
  surface must not advertise itself as untracked (superseding the merge-1
  warn-once posture entries and the trailing merge-2/merge-3 mentions of these two
  surfaces). The Google dialect now carries no unshipped media surface, and the
  OpenAI-dialect warn-once set holds only `translations` (reached via
  `client.audio.translations`).
