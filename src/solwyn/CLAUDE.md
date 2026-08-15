# SDK Source

## Module Map

- `client.py` — `Solwyn` (sync) + `AsyncSolwyn` (async) wrappers
- `_base.py` — shared sans-I/O logic (budget request construction, metadata formatting, guarded surface resolution and posture enforcement)
- `_surface_graph.py` — offline, operation-free observer for provider client capability graphs; raises `SurfaceInspectionError` on inaccessible declared namespaces or cycles
- `_surfaces.py` — sole sans-I/O owner of contextual surface classifications, applicability, shapes, usage basis, and JSON-ready contract data
- `_coverage.py` — frozen deterministic coverage reports and bidirectional literal expectations; structural metadata only, with no provider operations or network I/O
- `_registry.py` — binds caller SDK clients to provider adapters. Omitted `provider=` uses ordered type/base-url detection; an explicit primary or fallback provider is an identity assertion that bypasses detection and selects the named adapter. It never translates dialects or creates provider clients. `_SolwynBase` then validates the named provider/dialect against the actual module/class-derived SDK shape and, for pins, the known sync/async client mode before resolving any surface
- `budget.py` — `BudgetEnforcer` / `AsyncBudgetEnforcer` with cloud API check + local fallback. Settlement lives in the reporter, NOT here: `build_confirm_request` (sans-I/O, keyword-only) constructs the confirm; there is no `confirm_cost`. The `/budgets/check` POST rides the injected control-plane breaker. Run-scoped, token-billed calls meet the LEASE branch first (`_check_lease`): a sticky hard deny keeps the run on the per-call path, otherwise `_lease.admit()` decides and the enforcer performs only the I/O it prescribes — a blocking `POST /budgets/lease` grant on the caller's `budget_check_timeout` when there is no usable lease (one in-flight grant per run; losers take the legacy path rather than storm or wait), an ALWAYS-async renewal (daemon thread / task) at 75% depletion or the refresh deadline, and a best-effort surrender on `close()` and at interpreter exit. Every lease call rides `_control_plane_breaker` exactly like the check, and a REFUSAL (409 cap, 503 lease_unavailable, malformed body) credits the breaker and routes the run to the legacy path — only transport failures are an outage. An authoritative grant/renew denial feeds the UNCHANGED sticky-deny machinery; `BudgetCheckResult.lease_id` is the settlement key that later trues the reservation up — except for a call whose usage was never measurable (`build_confirm_request(usage_unmeasured=True)`, the client's fail-soft synthetic tier), which settles AT its reserved bound so a paid response's spent output allowance is never re-lent past the run's cap. The enforcer also owns the two things the sans-I/O ledger cannot: the §8 uncounted-mode telemetry (`_note_uncounted_admission` — one WARNING on ENTRY to a fail-open uncounted episode, then at most 1/30s while it persists; an installed grant ends the episode, both the cold-start and the expiry-ladder admissions feed it) and the fork identity — `_reset_after_fork_in_child` mints a FRESH `holder_id`, because the client's `_sdk_instance_id` survives a fork and the server releases a same-(project, run, holder) active lease as stale, so a child re-granting under the parent's id would kill the parent's live lease and start a churn
- `_lease.py` — sans-I/O lease ledger (PJ-2): the §4 outage ladder, atomic reservation math, generation fencing, renewal/backoff bookkeeping. Takes NO locks and does NO I/O — the caller serializes it (the sync enforcer's `_state_lock`, the async enforcer's event loop) and supplies a monotonic `now`
- `circuit_breaker.py` — local-only circuit breaker state machine. `name` distinguishes health domains in logs (provider breakers vs the shared `"control-plane"` breaker)
- `reporter.py` — `MetadataReporter` (thread queue) / `AsyncMetadataReporter` (create_task). Owns settlement delivery via `report_settlement` → `_send_confirm` (guarded by the shared control-plane breaker) and the consecutive-confirm-failure ERROR escalation. Breaker snapshots POST only on state/count change since the last successful send, with a `breaker_report_heartbeat` full refresh (default 60s) and an unconditional final shutdown send; idle ticks spawn no breaker worker. Delivery is AT-LEAST-ONCE: bounded retry with backoff (`reporter_max_send_attempts` / `reporter_retry_backoff_base` / `reporter_retry_backoff_cap`), breaker-open HOLDS confirms/settlements (never drops), confirm/settlement queues drain strictly FIFO (a transiently-failed head requeues and PARKS its queue for the cycle), a dropped/exhausted/overflow-evicted settlement confirm still ships its metadata event (ingest is the durable spend truth; the server dedups via its `(project_id, call_id, attempt_index)` ledger), every unavoidable drop is counted in `dropped_counts` — a lost settlement pair counts BOTH halves, and per-event rejections inside a 202 ingest body count `event.ingest_rejected` — and warned from the drop path (first drop immediately, then rate-limited), and `close(timeout=...)` runs the whole shutdown chain under ONE monotonic deadline (`reporter_shutdown_deadline`). The deadline is a true wall-clock bound (httpx timeouts cap socket ops, not total response time): the final flush runs off the closing thread — sync on a daemon worker joined at the deadline, async as a task the deadline cancels. Both closes end with `_seal_delivery` (base-class ownership gate: `_ownership_lock` + `_delivery_closed` + sync `_in_hand`): enqueues, drain requeues, and sync drain pops are ownership-gated — a sync drain pops-and-claims atomically via `_pop_in_hand`/`_pop_batch_in_hand` so the seal can never miss an item between pop and claim — and a timed-out join's in-hand items plus racing enqueues are counted before close returns. Async close sets `_close_completed` (and detaches the GC finalizer) only when it FINISHES — a cancelled close leaves the lifecycle rescue armed. An async drain cancelled mid-send REQUEUES its in-hand item under the ownership gate (`_requeue_cancelled`) instead of counting it: rescue can only retry what is IN the queues, a completing close's seal counts the requeued item, and only a seal-refused requeue counts directly. A zero-deadline close still awaits its cancelled drains' cleanup so every in-hand item is requeued for the seal first. `max_queue_size` must be >= 1 (constructor + config both validate). Ingest (`_send_batch`) is deliberately NOT breaker-guarded — an ingest blip must not flip budget checks to fail-open
- `_lifecycle.py` — process-lifecycle wiring: lazy singleton atexit hook (flushes live reporters at interpreter exit with per-item sent/failed/expired accounting; the blocking exit drain is wall-clock-bounded by a daemon worker JOINED at the deadline — httpx timeouts cap socket ops, not total response time, and closing a sync client does NOT reliably interrupt a blocked read, so no close()-based abort is trusted; `_ExitOwnership` mirrors the reporter's seal: pops register in-hand atomically, every lossy disposition publishes UNDER the ownership lock (`resolve_counted` — a seal landing in a release→publish gap would return with the item accounted nowhere and the daemon worker may never run again; plain `resolve` is for sent releases only), a timed-out join seals (in-hand + swept queues counted `shutdown_deadline`), a late-unblocking worker resolves to owned=0 and never double-counts, and a post-deadline failure classifies "expired"; exit confirms ride breaker admission — at most ONE recovery probe, refused confirms counted `exit_breaker_open` — while events are never breaker-gated and always get their deadline-bounded ingest attempt), a GC-ONLY `weakref.finalize` for never-started async reporters (`finalizer.atexit = False` — weakref's own exit hook registers after ours, so LIFO would drain live reporters FIRST with no accounting; the atexit hook is the sole live-reporter exit owner, and a genuine GC drain accounts losses via the logger-backed `_gc_drop_counter`), and the `os.register_at_fork` registry — `CircuitBreaker`, enforcers, reporters, and `_SolwynBase` all implement `_reset_after_fork_in_child` (fresh locks/clients; the sync reporter's flush thread relaunches lazily on the next enqueue via `_ensure_thread`). Also home of `_is_retryable_exc`, the `_monotonic` clock seam for the exit drain, and `register_lease_holder` — enforcers holding PJ-2 leases are surrendered (DHCPRELEASE-style) AFTER the reporter flush, breaker-admission gated and bounded by their own small deadline; surrendered runs are discarded so `close()` + exit can never release the same lease twice
- `config.py` — `SolwynConfig` with env var loading (`SOLWYN_*` prefix)
- `exceptions.py` — `SolwynError` base, `BudgetExceededError`, `RunStoppedError`, `ProviderUnavailableError`, `ConfigurationError`
- `_privacy.py` — length-only prompt estimation. PRIVACY-CRITICAL: content-privileged allowlist with `providers/_translation/`
- `_run.py` — `ContextVar`-backed agent-run scope; `run(name, tags=...)` public entry point; `current_run()` preserves the public `(id, name)` tuple while a private copied snapshot carries bounded tags through deferred reporting
- `_types.py` — Pydantic models for API request/response contracts
- `_validation.py` — API key + project ID format validation
- `providers/` — extraction adapters (OpenAI, OpenAI-compatible, Anthropic, Google, Bedrock); `_translation/` is content-privileged for request/response translation only

## Coverage Contract Ownership

`src/solwyn/_surfaces.py` is the source of truth consumed by runtime guards,
coverage manifests, drift canaries, and the generated
`build/surface_contract/surface-classification.json`. That JSON is a short-lived
CI artifact for diagnosing every rule and applicability decision. It is not
committed; the Python rules remain authoritative, and cross-SDK consumption
remains deferred.

Keep namespaces and terminal leaves separate. Acknowledgments are exact,
applicable terminal tokens only and must never authorize a namespace or use a
wildcard. Wrapper ownership precedes raw paths, deliberate block precedes
unsupported, and unsupported precedes untracked posture. OpenAI-compatible
adapters keep video explicitly unsupported while native OpenAI video remains
tracked. The token-billed TTS condition is a distinct synthetic rule with token
`audio.speech.create:gpt-4o-mini-tts`.

Native OpenAI and Azure OpenAI `responses.create`, `responses.parse`, and
`responses.stream` are metered through the OpenAI-dialect
wildcard-plus-override contract: provider-specific rules use source `BOTH`,
while the wildcard keeps other compatible profiles at `unmetered_spend`. The
`responses.stream` new-response overload is metered; its `response_id` /
`starting_after` existing-response retrieval overload is a reviewed raw
pass-through with exact caller kwargs and no second budget check or settlement.
Every other Responses leaf, including beta/raw-response helpers, and all other
compatible profiles remain `unmetered_spend` at this milestone.
Native OpenAI behind a nonstandard gateway is admitted with
`provider="openai"`; the pin bypasses base-url detection but retains the same
surface-context validation and native Responses capability.

`on_unmetered="warn"` is the compatibility default; `raise` is strict mode and
`allow` is the explicit pass posture. Guarded provider resources forward
attributes only — protocol dunders (with/async-with) are not forwarded in ANY
posture, including allow. Strict mode guards cooperative public
pre-call access only. It is not a sandbox for callers that retain the raw
client, reach private wrapper state, acknowledge a scoped raw escape, or invoke
native behavior on provider-returned response/page/stream/job/operation
objects.

Advisory reporting for untracked provider-client surfaces is ON by default.
From provider-call paths, the SDK only schedules fire-and-forget delivery
through the reporter's background thread/task, never on the budget hot path.
Shutdown may wait within the shared deadline for active advisory work and a
best-effort final attempt; send failures remain silent. Each external payload
contains a structural dotted surface path; bounded execution-context and
classification fields (provider, client shape, sync/async mode, rule kind,
optional capability scope, and posture); approximate occurrence counts and
first/last timestamps; and random SDK-instance/report identifiers. Model names,
request arguments, prompts, and responses are never included. A public path that
is identifier-clean but structurally unreportable (deeper or longer than the
wire pattern allows, or non-ASCII) stays LOCAL-ONLY: it is warned and counted in
the process registry exactly like any other untracked surface, and never reaches
a reporter. Only a malformed path — an empty, private, or non-identifier
segment — fails closed with a content-free `RuntimeError`.

Only unacknowledged `warn`/`allow` observations enter advisory reporting;
`raise` refusals and acknowledged escapes do not. Reporting adds neither a
budget check nor a cost event, so it does not meter or budget-enforce these
calls. Counts are approximate, bounded-overcount signals rather than billing
truth, and absence from the dashboard is not comprehensive usage evidence.
Core derives the project from authentication on its project-implicit route.
Set `report_untracked_surfaces=False` or
`SOLWYN_REPORT_UNTRACKED_SURFACES=false` to disable this optional external
advisory egress without changing local `on_unmetered`
`warn`/`allow`/`raise` behavior.

Coverage fingerprints must be independently reviewed literals. Never derive a
value from the report under test and immediately pass it to `expect(...)`.
Changes to audit fields are bidirectional drift and require reviewing the full
entries before updating the literal.

### Changing a surface rule

Never hand-edit the payload block in `_surfaces.py`. The loop:

1. `uv run python scripts/export_surface_contract.py` — bootstrap the editable
   JSON (OVERWRITES any existing local edits — export first, edit second).
2. Edit exact rows in `build/surface_contract/surface-classification.json`.
   Compat-vs-native splits use wildcard-plus-override selectors: a
   `provider: null` rule for every openai-dialect provider, plus a
   higher-specificity exact-provider rules such as `provider: "openai"` and
   `provider: "azure_openai"` that win for admitted profiles (see the Responses
   rules). Keep rule ids stable; the id's third segment must equal the rule's
   kind.
3. `uv run python scripts/embed_surface_rules.py --input build/surface_contract/surface-classification.json`
   — validates, prints the rule delta, refuses stale exports (`--allow-stale`)
   and silent removals (`--allow-removals`).
4. `uv run python scripts/export_surface_contract.py` — refresh the expanded
   artifact and its `source_payload_fingerprint` from the newly embedded
   canonical payload before running any `--check` command.
5. Update the per-context digests in
   `tests/unit/test_surface_context_pins.py` (and
   the README's `OPENAI_STRICT_FINGERPRINT` if the OpenAI graph moved),
   then run
   `make check && make test-unit && make check-surface-contract && make check-provider-surfaces`.
   Continue to the PR output only after this passes.

Put `uv run python scripts/diff_surface_rules.py <PR-base-ref>` output in the
PR. Use the branch or ref the PR targets; use `origin/main` only when the PR
actually targets main.
Red canary? See `docs/surface-canary-runbook.md`.
Release forensics: any installed wheel reproduces its full ledger via
`python -c "from solwyn._surfaces import surface_contract_data; import json; print(json.dumps(surface_contract_data()))"`.

## Provider Adapter Notes

- **Anthropic**: `input_tokens` = base + `cache_read_input_tokens` + `cache_creation.ephemeral_5m_input_tokens` + `cache_creation.ephemeral_1h_input_tokens` (additive, base does NOT include cache); aggregate-only `cache_creation_input_tokens` falls back to the 5m bucket
- **OpenAI**: Two response shapes — Chat Completions (`prompt_tokens`/`completion_tokens`) vs Responses API (`input_tokens`/`output_tokens`). Detect via `hasattr(usage, 'prompt_tokens')`. Responses dispatch is duck-typed through `prepare_responses_call`: the leaf routes to `responses.create`, `responses.parse`, or the SDK's `responses.stream` manager helper; create streaming sets `stream=True` but never `stream_options`, parse refuses streaming before dispatch, and terminal events supply `event.response.usage` plus `event.response.service_tier`. Missing or zeroed usage on a Responses call is unmeasured usage, foreground or streaming: a 0/0 read becomes the synthetic request-length input estimate marked `is_estimated`, an adapter's own estimated tier (the compat streaming fallback) is kept as the better telemetry, and either way lease settlement holds the reserved bound rather than re-lending spent output allowance; chat's zero-usage semantics are unchanged. Because the OpenAI SDK applies `extra_body` after named arguments, Responses metering rejects `model`, `input`, `instructions`, `max_output_tokens`, or `stream` inside `extra_body` before preflight and provider dispatch; pass them as top-level arguments, while other vendor extensions remain untouched. The new-response stream-helper manager opens only on context entry, returns a Responses-specific wrapped inner event stream with SDK helpers forwarded, and settles a length estimate when an ENTERED stream is abandoned. The existing-response overload bypasses defaults and metering because it retrieves an already-created response rather than creating new spend
- **Google**: `output_tokens` = `candidates_token_count` + `thoughts_token_count`. Usage on `response.usage_metadata` not `response.usage`
- **Bedrock**: Converse responses are DICTS (mapping access, never getattr). `input_tokens` = `inputTokens` + `cacheReadInputTokens` + `cacheWriteInputTokens` (additive — AWS-documented formula). Cache-write TTL split via `usage.cacheDetails` (`[{inputTokens, ttl}]`); aggregate-only falls back to the 5m bucket. Streaming usage arrives in the terminal `metadata` event. Service tier = `serviceTier.type`, else `performanceConfig.latency`. Region from `client.meta.region_name` → `provider_region` (pricing is per model AND region). boto3 never imported; detection = module contains `botocore` + `meta.service_model.service_name == "bedrock-runtime"`. boto3 has no `with_options` — per-hop timeouts cannot be applied; the caller's botocore Config governs
- **OpenAI-compatible** (`openai_compatible.py`): one adapter class, one `CompatProfile` per provider (hosts/ports/model-prefixes/include_usage/supports_responses flags). `dialect="openai"`, distinct `name` per provider. `supports_responses` defaults false and is true only for Azure OpenAI; callers and every proxy/dispatcher seam use the shared `_responses_preparer` capability check, while native OpenAI remains supported through its missing-attribute default. Detection is by `base_url` host (Azure also by client class name; Ollama/vLLM/LM Studio by localhost port 11434/8000/1234; unknown hosts -> generic catch-all). Streaming usage tiers: standard `usage` (last non-None chunk, including nested terminal `event.response.usage`) -> Groq legacy `x_groq.usage` (raw dict!) -> length-based estimate marked `is_estimated=True`. `stream_options include_usage` is injected ONLY where documented-safe (xAI/Mistral/Perplexity reject it; OpenRouter deprecates it); unsupported profiles STRIP caller stream_options so failover hops don't 4xx. Azure skips injection when `data_sources` is present

## Client Proxy Patterns

- Proxy properties (`chat`, `messages`, `models`) use `@functools.cached_property`; Bedrock's `converse`/`converse_stream`/`invoke_model*` are plain methods on Solwyn/AsyncSolwyn (boto3 methods live on the client root, not a nested resource)
- `responses` is a conditional cached property: an adapter satisfying `_responses_preparer` (method exists and `supports_responses` is truthy, defaulting true when absent) returns a sync/async proxy whose `create`, `parse`, and new-response `stream` methods call `_intercepted_call(_surface="responses")` with the selected leaf. Native OpenAI and Azure OpenAI satisfy that condition; every other compatible profile retains its raw manager. The path is primary-only and builds one filtered effective-default mapping for both preflight and dispatch: chat-only defaults are stripped and caller kwargs win. Effective `background=True` fails loud, including OpenAI `extra_body` precedence; parse also refuses effective streaming (with the same structural precedence) before budget or provider dispatch. The new-response stream leaf returns a manager wrapper instead of the direct event-stream wrapper; context entry opens the SDK manager, wraps its inner stream for accumulation, and context exit reconciles exactly once before forwarding SDK cleanup — settling an entered stream, or releasing the reservation with no confirm, event, or breaker verdict when the manager never dispatched. Because that manager sends its request in `__enter__`, a failure there is classified by `classify_exception` exactly like a candidate-walk dispatch error: FAIL_FAST records no breaker failure, FAILOVER and POST_SEND_AMBIGUOUS do, and only the ambiguous case reports `possibly_succeeded`. Any failure after the provider stream opens keeps the established-stream verdict (breaker failure, possibly-succeeded). A `response_id` / `starting_after` stream call is returned directly from the selected SDK as reviewed raw retrieval, preserving manager identity and exact kwargs without defaults or metering. The async proxy preserves the SDK's synchronous manager-return shape with a deferred async bridge. Other providers and all remaining leaves resolve through the shared surface policy
- `_force_stream=True` is set by the Google proxy's `generate_content_stream` AND the Bedrock `converse_stream` method; `_intercepted_call` folds it into the dispatch-level `is_streaming` boolean, which (not the original flag) drives the served hop's stream-method selection — so cross-provider failover INTO Google/Bedrock streams via their dedicated methods and OUT via `stream=True`
- Per-provider dispatch quirks (stream kwarg vs dedicated method, Bedrock's `modelId` rename, Google's per-request HTTP bound) live on each adapter's `prepare_call`; `_sync_dispatch`/`_async_dispatch` are provider-agnostic. The Bedrock proxy renames boto3's `modelId` → internal `model` at interception; `BedrockAdapter.prepare_call` renames it back. The whole pipeline keys on `kwargs["model"]`
- Bedrock streaming shape: `converse_stream` returns `{"stream": EventStream, ...}` — the SERVED adapter's `unwrap_stream_source` hands `_wrap_stream` the INNER event stream, and the PRIMARY adapter's `wrap_stream_result` reshapes the wrapper to the caller dialect (boto3 dict for Bedrock callers, the bare wrapper for everyone else); early abandonment settles via `result["stream"].close()` or the wrapper's context manager (exactly once, via the `_settled` guard)
- Bedrock `invoke_model` fails loud (`ConfigurationError`) — usage is buried in a consume-once body with response content; silent pass-through would be a budget bypass
- ALL reservation settlement — streaming `on_complete` AND non-streaming chat/media success — fire-and-forgets via `reporter.report_settlement(confirm, event)` (confirm built sans-I/O by `budget.build_confirm_request`, enqueued with its metadata event as one ordered item). Never blocks the user thread; there is no blocking `confirm_cost`

## Thread Safety

- Sync `BudgetEnforcer` is thread-safe — mutable state guarded by `self._state_lock`. The `LeaseLedger` is sans-I/O and unlocked BY DESIGN: every ledger call (admission, grant application, true-up/release, request builders) happens under that same lock, and one admission must run inside ONE lock section or a concurrent burst can jointly overrun the grant. Never hold the lock across HTTP
- Async `AsyncBudgetEnforcer` does not need a lock — event loop serialization
- `MetadataReporter._in_flight` guarded by `self._in_flight_lock`
