# Changelog

All notable changes to the Solwyn Python SDK are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions are
derived from git tags (hatch-vcs).

## [Unreleased]

### Changed

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
  Queue overflow, retry exhaustion, terminal statuses, shutdown-deadline
  expiry, and enqueue-after-close all increment a counter now readable via the
  new `dropped_counts` property. The first drop logs immediately; after that at
  most one aggregated `reporter.spend_events_dropped` WARNING per 60s, so a
  sustained outage reports loss without flooding logs.
- **Queued spend survives interpreter exit.** A process that exits WITHOUT
  calling `close()` previously discarded everything still queued. A single
  `atexit` hook now flushes each live reporter (sync via `close()`, async over a
  temporary sync client since no event loop exists at exit), and async reporters
  additionally arm a `weakref.finalize` covering the constructed-queued-then-
  garbage-collected case. Exit delivery is accountable per item: every popped
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
  the deadline is reached is counted `shutdown_deadline` and dropped. At the
  deadline the sync `close()` takes final ownership of ALL undelivered spend:
  items a stuck flush thread still holds mid-POST and enqueues racing the final
  drain are counted before `close()` returns, never requeued into a queue
  nothing drains. An async `close()` cancelled mid-flush keeps the atexit hook
  and GC finalizer armed as the last delivery path — they detach only once
  close actually completes.

### Added

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
  `retry_exhausted`, `terminal_status`, `shutdown_deadline`,
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
