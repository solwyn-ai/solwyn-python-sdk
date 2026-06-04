# SDK Source

## Module Map

- `client.py` — `Solwyn` (sync) + `AsyncSolwyn` (async) wrappers
- `_base.py` — shared sans-I/O logic (budget request construction, metadata formatting)
- `budget.py` — `BudgetEnforcer` / `AsyncBudgetEnforcer` with cloud API check + local fallback
- `circuit_breaker.py` — local-only circuit breaker state machine
- `reporter.py` — `MetadataReporter` (thread queue) / `AsyncMetadataReporter` (create_task)
- `config.py` — `SolwynConfig` with env var loading (`SOLWYN_*` prefix)
- `tokenizer.py` — tiktoken + heuristic fallback
- `exceptions.py` — `SolwynError` base, `BudgetExceededError`, `ProviderUnavailableError`, `ConfigurationError`
- `_privacy.py` — length-only prompt estimation. PRIVACY-CRITICAL: content-privileged allowlist with `providers/_translation/`
- `_run.py` — `ContextVar`-backed agent-run scope; `run(name)` public entry point, `current_run()` read seam used by `_base.py`
- `_types.py` — Pydantic models for API request/response contracts
- `_validation.py` — API key + project ID format validation
- `providers/` — extraction adapters (OpenAI, Anthropic, Google); `_translation/` is content-privileged for request/response translation only

## Provider Adapter Notes

- **Anthropic**: `input_tokens` = base + `cache_read_input_tokens` + `cache_creation.ephemeral_5m_input_tokens` + `cache_creation.ephemeral_1h_input_tokens` (additive, base does NOT include cache); aggregate-only `cache_creation_input_tokens` falls back to the 5m bucket
- **OpenAI**: Two response shapes — Chat Completions (`prompt_tokens`/`completion_tokens`) vs Responses API (`input_tokens`/`output_tokens`). Detect via `hasattr(usage, 'prompt_tokens')`
- **Google**: `output_tokens` = `candidates_token_count` + `thoughts_token_count`. Usage on `response.usage_metadata` not `response.usage`

## Client Proxy Patterns

- Proxy properties (`chat`, `messages`, `models`) use `@functools.cached_property`
- `_force_stream=True` is set by the Google proxy's `generate_content_stream`; `_intercepted_call` folds it into the dispatch-level `is_streaming` boolean, which (not the original flag) drives the served hop's stream-method selection — so cross-provider failover INTO Google streams via `generate_content_stream` and OUT OF Google streams via `stream=True`
- Stream `on_complete` fire-and-forgets `confirm_cost` via `reporter.report_confirm()` — never blocks user thread

## Thread Safety

- Sync `BudgetEnforcer` is thread-safe — mutable state guarded by `self._state_lock`
- Async `AsyncBudgetEnforcer` does not need a lock — event loop serialization
- `MetadataReporter._in_flight` guarded by `self._in_flight_lock`
