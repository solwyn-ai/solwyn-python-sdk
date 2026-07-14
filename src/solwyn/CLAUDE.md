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
- `_run.py` — `ContextVar`-backed agent-run scope; `run(name, tags=...)` public entry point; `current_run()` preserves the public `(id, name)` tuple while a private copied snapshot carries bounded tags through deferred reporting
- `_types.py` — Pydantic models for API request/response contracts
- `_validation.py` — API key + project ID format validation
- `providers/` — extraction adapters (OpenAI, OpenAI-compatible, Anthropic, Google, Bedrock); `_translation/` is content-privileged for request/response translation only

## Provider Adapter Notes

- **Anthropic**: `input_tokens` = base + `cache_read_input_tokens` + `cache_creation.ephemeral_5m_input_tokens` + `cache_creation.ephemeral_1h_input_tokens` (additive, base does NOT include cache); aggregate-only `cache_creation_input_tokens` falls back to the 5m bucket
- **OpenAI**: Two response shapes — Chat Completions (`prompt_tokens`/`completion_tokens`) vs Responses API (`input_tokens`/`output_tokens`). Detect via `hasattr(usage, 'prompt_tokens')`
- **Google**: `output_tokens` = `candidates_token_count` + `thoughts_token_count`. Usage on `response.usage_metadata` not `response.usage`
- **Bedrock**: Converse responses are DICTS (mapping access, never getattr). `input_tokens` = `inputTokens` + `cacheReadInputTokens` + `cacheWriteInputTokens` (additive — AWS-documented formula). Cache-write TTL split via `usage.cacheDetails` (`[{inputTokens, ttl}]`); aggregate-only falls back to the 5m bucket. Streaming usage arrives in the terminal `metadata` event. Service tier = `serviceTier.type`, else `performanceConfig.latency`. Region from `client.meta.region_name` → `provider_region` (pricing is per model AND region). boto3 never imported; detection = module contains `botocore` + `meta.service_model.service_name == "bedrock-runtime"`. boto3 has no `with_options` — per-hop timeouts cannot be applied; the caller's botocore Config governs
- **OpenAI-compatible** (`openai_compatible.py`): one adapter class, one `CompatProfile` per provider (hosts/ports/model-prefixes/include_usage flag). `dialect="openai"`, distinct `name` per provider. Detection by `base_url` host (Azure also by client class name; Ollama/vLLM/LM Studio by localhost port 11434/8000/1234; unknown hosts -> generic catch-all). Streaming usage tiers: standard `usage` (last non-None chunk) -> Groq legacy `x_groq.usage` (raw dict!) -> length-based estimate marked `is_estimated=True`. `stream_options include_usage` injected ONLY where documented-safe (xAI/Mistral/Perplexity reject it; OpenRouter deprecates it); unsupported profiles STRIP caller stream_options so failover hops don't 4xx. Azure skips injection when `data_sources` present

## Client Proxy Patterns

- Proxy properties (`chat`, `messages`, `models`) use `@functools.cached_property`; Bedrock's `converse`/`converse_stream`/`invoke_model*` are plain methods on Solwyn/AsyncSolwyn (boto3 methods live on the client root, not a nested resource)
- `_force_stream=True` is set by the Google proxy's `generate_content_stream` AND the Bedrock `converse_stream` method; `_intercepted_call` folds it into the dispatch-level `is_streaming` boolean, which (not the original flag) drives the served hop's stream-method selection — so cross-provider failover INTO Google/Bedrock streams via their dedicated methods and OUT via `stream=True`
- Per-provider dispatch quirks (stream kwarg vs dedicated method, Bedrock's `modelId` rename, Google's per-request HTTP bound) live on each adapter's `prepare_call`; `_sync_dispatch`/`_async_dispatch` are provider-agnostic. The Bedrock proxy renames boto3's `modelId` → internal `model` at interception; `BedrockAdapter.prepare_call` renames it back. The whole pipeline keys on `kwargs["model"]`
- Bedrock streaming shape: `converse_stream` returns `{"stream": EventStream, ...}` — the SERVED adapter's `unwrap_stream_source` hands `_wrap_stream` the INNER event stream, and the PRIMARY adapter's `wrap_stream_result` reshapes the wrapper to the caller dialect (boto3 dict for Bedrock callers, the bare wrapper for everyone else); early abandonment settles via `result["stream"].close()` or the wrapper's context manager (exactly once, via the `_settled` guard)
- Bedrock `invoke_model` fails loud (`ConfigurationError`) — usage is buried in a consume-once body with response content; silent pass-through would be a budget bypass
- Stream `on_complete` fire-and-forgets reservation settlement via `reporter.report_settlement()` -- never blocks user thread

## Thread Safety

- Sync `BudgetEnforcer` is thread-safe — mutable state guarded by `self._state_lock`
- Async `AsyncBudgetEnforcer` does not need a lock — event loop serialization
- `MetadataReporter._in_flight` guarded by `self._in_flight_lock`
