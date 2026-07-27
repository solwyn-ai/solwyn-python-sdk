# Solwyn Python SDK

Budget enforcement, circuit breaking, and usage tracking for OpenAI, Anthropic, Google, and Amazon Bedrock LLM clients — plus any provider that speaks the OpenAI Chat Completions dialect (xAI, DeepSeek, Mistral, Qwen, Z.ai, Groq, Together, Fireworks, Perplexity, Azure OpenAI, OpenRouter, Ollama, vLLM, LM Studio, …).

[![CI](https://github.com/solwyn-ai/solwyn-python-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/solwyn-ai/solwyn-python-sdk/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/solwyn)](https://pypi.org/project/solwyn/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/solwyn)](https://pypi.org/project/solwyn/)
[![License](https://img.shields.io/github/license/solwyn-ai/solwyn-python-sdk)](LICENSE)

Solwyn wraps your existing LLM client. Calls go directly to the provider — the SDK only reports metadata (token counts, media quantities, latency, model name) to the Solwyn API. **Prompts and responses never leave your application.**

## Installation

```sh
pip install solwyn
```

Optional extras pin tested provider-SDK floors — `solwyn[openai]` (also enables tiktoken-based token estimation), `solwyn[anthropic]`, `solwyn[google]`, `solwyn[bedrock]` (convenience only — the SDK never imports boto3), `solwyn[together]` (Together SDK 2.0+), or `solwyn[all]`:

```sh
pip install solwyn[openai]
```

Other OpenAI-compatible endpoints (Groq, OpenRouter, vLLM, …) ride the `openai` extra. Together can use that path too; `solwyn[together]` supplies its native SDK instead.

## Quick Start

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(
    OpenAI(),
    api_key="sk_proj_...",
)

response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Hello!"}],
)

client.close()
```

Or use as a context manager:

```python
with Solwyn(OpenAI(), api_key="sk_proj_...") as client:
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

## Providers

### OpenAI

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(OpenAI(), api_key="sk_proj_...")
response = client.chat.completions.create(model="gpt-5.5", messages=[...])
```

### Anthropic

```python
from anthropic import Anthropic
from solwyn import Solwyn

client = Solwyn(Anthropic(), api_key="sk_proj_...")
response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[...])
```

### Google Gemini

```python
from google import genai
from solwyn import Solwyn

client = Solwyn(genai.Client(api_key="..."), api_key="sk_proj_...")
response = client.models.generate_content(model="gemini-3.5-flash", contents="Hello!")
```

### Amazon Bedrock

Wrap a `bedrock-runtime` boto3 client. Solwyn intercepts the [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) (`converse` / `converse_stream`), which works uniformly across every chat model Bedrock hosts — Anthropic Claude, Meta Llama, Mistral, Amazon Nova, Cohere, AI21, DeepSeek, and more. Auth stays entirely on your boto3 client (IAM credentials, profiles, roles, SigV4) — Solwyn never sees it.

```python
import boto3
from botocore.config import Config
from solwyn import Solwyn

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    # Recommended: let Solwyn own retries/failover instead of stacking
    # botocore's default retry layer (legacy mode retries up to 5 times).
    config=Config(retries={"total_max_attempts": 1}, read_timeout=60),
)

client = Solwyn(bedrock, api_key="sk_proj_...")
response = client.converse(
    modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
    inferenceConfig={"maxTokens": 1024},
)
```

Streaming preserves the boto3 contract (`response["stream"]`); usage settles from the stream's terminal `metadata` event:

```python
response = client.converse_stream(
    modelId="amazon.nova-pro-v1:0",
    messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
)
for event in response["stream"]:
    ...
```

If you stop consuming the stream early, call `response["stream"].close()` (or wrap iteration in `with response["stream"]:`) to settle the budget reservation — the same close obligation raw boto3's `EventStream` has. `close()` settles exactly once with whatever usage was observed and is safe to call repeatedly.

Notes:

- Model identity is reported exactly as you pass it — foundation-model ids, cross-region inference profiles (`us.` / `eu.` / `jp.` / `global.` …), or full ARNs — together with the client's region, because Bedrock pricing is keyed per model **and** region. Prompt-cache reads/writes (including the 1h-TTL tier via `usage.cacheDetails`) and the latency/service pricing tier are captured for exact repricing.
- `invoke_model` / `invoke_model_with_response_stream` / `start_async_invoke` raise `ConfigurationError` instead of bypassing budget tracking: their usage is buried in a consume-once body (or lands out-of-band in S3, for `start_async_invoke`) alongside response content. Use Converse, or call the unwrapped boto3 client for deliberately untracked calls.
- boto3 has no per-call timeout override, so neither the failover window nor `failover_hop_read_timeout` can shorten an in-flight Bedrock hop — set `read_timeout` in your botocore `Config` (building a client with `read_timeout=None` logs a warning; see [Failover timeouts](#failover-timeouts)).
- Async works with [aioboto3](https://github.com/terricain/aioboto3): `AsyncSolwyn(client)` inside `async with session.client("bedrock-runtime") as client`.
- Bedrock participates in cross-provider failover in both directions (e.g. Bedrock-Claude ⇄ direct Anthropic) via the same canonical translation subset as the other providers.

### Together AI

Solwyn supports the native Together SDK at `together>=2.0`. Install the convenience extra, then wrap the client directly:

```sh
pip install "solwyn[together]"
```

```python
from solwyn import Solwyn
from together import Together

client = Solwyn(Together(api_key="..."), api_key="sk_proj_...")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Pair sync and async client types: use `Solwyn` with `Together`, and `AsyncSolwyn` with `AsyncTogether`:

```python
from solwyn import AsyncSolwyn
from together import AsyncTogether

async with AsyncSolwyn(AsyncTogether(api_key="..."), api_key="sk_proj_...") as client:
    response = await client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

The optional extra is bring-your-own convenience only: Solwyn core never imports Together. An `openai.OpenAI` client pointed at Together's compatible endpoint remains supported as described below.

### OpenAI-compatible providers

Point an `openai.OpenAI` client at any OpenAI-compatible endpoint via `base_url` and wrap it as usual. Solwyn detects the provider from the URL, so budgets, per-agent attribution, failover, and the cost dashboard all see the *real* provider (e.g. `groq`), not "openai":

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(
    OpenAI(base_url="https://api.groq.com/openai/v1", api_key="gsk_..."),
    api_key="sk_proj_...",
)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Auto-detected providers:

| Provider | Detected from | Streaming usage |
|----------|--------------|-----------------|
| xAI (Grok) | `api.x.ai` | automatic (final chunk); `stream_options` is never sent — xAI rejects it |
| DeepSeek | `api.deepseek.com` | `include_usage` injected |
| Mistral | `api.mistral.ai` | `stream_options` never sent (strict validation); final-chunk usage or estimate |
| Qwen (DashScope compat) | `dashscope*.aliyuncs.com` | `include_usage` injected |
| Z.ai (`zai`) | `api.z.ai` | `include_usage` injected |
| Groq | `api.groq.com` | `include_usage` injected; legacy `x_groq.usage` also handled |
| Together AI | native `Together` / `AsyncTogether`, or `api.together.xyz` / `api.together.ai` | automatic (final chunk) |
| Fireworks | `api.fireworks.ai` | automatic (final chunk) |
| Perplexity (Sonar) | `api.perplexity.ai` | usage on streamed chunks; `stream_options` never sent |
| Azure OpenAI | `*.openai.azure.com` or `AzureOpenAI` client class | `include_usage` injected (skipped for "on your data" `data_sources` requests, which reject it) |
| OpenRouter | `openrouter.ai` | automatic (final chunk); `stream_options` is deprecated there |
| Ollama | `localhost:11434` | `include_usage` injected (older versions ignore it → estimate) |
| vLLM | `localhost:8000` | `include_usage` injected |
| LM Studio | `localhost:1234` | `include_usage` injected (pre-0.3.18 omits usage → estimate) |
| Anything else | any non-OpenAI `base_url` | generic `openai_compatible`; `stream_options` never sent |

For endpoints auto-detection can't name (e.g. vLLM on a non-default port), pass the provider explicitly — on the constructor for the primary, or as the 4th element of a fallback spec:

```python
client = Solwyn(
    OpenAI(base_url="http://gpu-box:8080/v1", api_key="-"),
    api_key="sk_proj_...",
    provider="vllm",
    fallback=[(OpenAI(base_url="https://openrouter.ai/api/v1", api_key="sk-or-..."), "openrouter/auto"),
              (other_client, "my-model", {}, "ollama")],
)
```

**Token accounting.** Budgets and attribution depend on accurate per-call usage, and "OpenAI-compatible" endpoints differ most in exactly that. Solwyn requests streaming usage only from providers where that's documented-safe, reads it from the final chunk where it arrives automatically, and — when a provider reports no usage at all (or reports an unparseable/zeroed block alongside real content) — falls back to a length-based estimate that is **explicitly marked** (`token_details.is_estimated = true` on the wire, plus a one-time SDK warning). Degraded accounting is loud and flagged, never silently zero.

The "never sent" entries above describe Solwyn's own injection policy. A `stream_options` you pass explicitly always reaches your configured provider untouched (drop-in contract); it is only stripped when a *failover hop* lands on a provider known to reject it.

**Pricing.** The SDK never computes cost. It reports the served `(provider, model)` verbatim — for OpenRouter that's the full model slug (e.g. `anthropic/claude-sonnet-4.5`) — and Solwyn Cloud's PricingService prices it. Models unknown to the catalog are surfaced as unpriced on the dashboard rather than silently costed at $0.

**Failover.** Compat providers participate fully in failover. Between two OpenAI-dialect providers (e.g. Groq → OpenRouter) requests pass through natively — tools, JSON mode, and streaming included (`max_completion_tokens` is rewritten to `max_tokens` for targets that need the legacy key). Per-call `extra_headers`/`extra_query`/`extra_body` are stripped on cross-provider hops — they're endpoint-scoped, authored for the original endpoint — though the fallback entry's own `default_params` versions still apply. Across dialects (e.g. Groq → Anthropic) the standard translation subset applies.

**Known limitation.** Circuit-breaker health, latency signals, and failover labeling key off the provider *name*. Two chain entries that resolve to the same name (two Azure resources, two unnamed gateways both detected as `openai_compatible`) share one health domain and are reported as model fallbacks of each other. For the same reason, a hop between same-name entries skips cross-provider request sanitization — `stream_options` stripping, the `max_completion_tokens` → `max_tokens` rewrite, and endpoint-scoped param stripping (`extra_headers`/`extra_query`/`extra_body`). A `stream_options` or gateway header you authored for the first endpoint reaches the second untouched and can 4xx there. Give distinct endpoints distinct provider identities where possible — explicit `provider=` on the constructor, or the 4th element of a fallback spec.

## Media surfaces

Beyond chat, Solwyn tracks the non-text surfaces that spend money. Each rides the same budget-check → provider call → confirm lifecycle as a chat call, tagged with its modality (`embedding`, `image`, `audio`, `video`) so Solwyn Cloud's PricingService prices it on the right card. There is no cross-provider failover for these surfaces — an embedding vector or a generated image isn't interchangeable across providers.

| Surface | OpenAI dialect (native + compatible) | Google (Gemini) |
|---------|--------------------------------------|-----------------|
| Embeddings | `client.embeddings.create` | `client.models.embed_content` |
| Images | `client.images.generate` / `client.images.edit` | `client.models.generate_images` (Imagen) |
| Audio — transcription | `client.audio.transcriptions.create` (incl. Groq whisper) | — |
| Audio — speech (TTS) | `client.audio.speech.create` | — |
| Video | `client.videos.create` (Sora) | `client.models.generate_videos` (Veo) |

Billable quantities are read from the response's usage block where it exists (gpt-image token buckets, whisper duration) and derived from the request where a provider reports none — image counts from `n=`, TTS character counts from `input=`, video seconds from the request. Whatever the SDK can't observe stays `None`, and the call is tracked **unpriced** rather than settled at a silent $0. Only lengths, counts, durations, and variant selectors are ever measured — never the media itself.

**Posture notes.**

- **Whisper needs a JSON `response_format` to be priced.** `whisper-1` reports its billable duration only under a JSON response format. A non-JSON `response_format` (`text` / `srt` / `vtt`) carries no usage, so the call is tracked unpriced with a one-time hint to pass `response_format="json"` (or `"verbose_json"`) for priced tracking.
- **`gpt-4o-mini-tts` is passed through untracked.** Token-billed TTS models publish no usage metadata, so their audio-output tokens are unobservable. Rather than settle a silent $0, the call passes through untracked (no budget check, no cost event) after a one-time warning.
- **`audio.translations` is passed through untracked.** The translations sub-surface isn't intercepted yet; it warns once, then passes through untracked.

## Async

```python
from openai import AsyncOpenAI
from solwyn import AsyncSolwyn

async with AsyncSolwyn(
    AsyncOpenAI(),
    api_key="sk_proj_...",
) as client:
    response = await client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

## Streaming

Pass `stream=True` as you normally would. Solwyn wraps the stream transparently and reports usage when it completes:

```python
stream = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Tagging Calls with Agent Runs

Wrap a unit of work with `solwyn.run(name, tags=...)` to attribute every LLM call inside it to a single agent run. The dashboard groups cost and latency by run, so you can see "this nightly batch cost $4.20." Tags are optional explicit customer metadata for grouping and export.

```python
import solwyn
from openai import OpenAI

client = solwyn.Solwyn(OpenAI(), api_key="sk_proj_...")

with solwyn.run("nightly-batch", tags={"team": "research", "env": "prod"}) as run_id:
    client.chat.completions.create(model="gpt-5.5", messages=[...])
    client.chat.completions.create(
        model="gpt-5.5",
        messages=[...],
        solwyn_tags={"env": "staging", "job": "backfill"},
    )
```

Use the reserved `solwyn_tags=` keyword as a call argument, not in `default_params`; it is removed before provider dispatch. Per-call keys shallow-merge over run tags. Tag mappings allow at most 10 string keys, keys must contain 1–64 characters, and string values may contain 0–256 characters. The SDK copies mappings at scope entry and call start, so later caller mutation cannot change attribution.

Works the same with `async with` and is safe across concurrent asyncio tasks — each task sees only its own active run. Calls made outside a `solwyn.run(...)` scope are still tracked; the API groups them into `_auto-{sdk_instance_id}-{YYYY-MM-DD}` using the event's UTC timestamp.

Do not open `solwyn.run(...)` inside an async generator. Python runs the consumer's `async for` body in the same context after a generator `yield`, so an inner generator scope would leak into customer code. The SDK rejects that pattern at scope entry. Open the scope in the consumer, or await the generator entirely inside an outer run scope.

Tasks created with `asyncio.create_task(...)` inside a run capture that task's context. If the task keeps making LLM calls after the `with` block exits, those calls are still attributed to the captured run id. Use `asyncio.TaskGroup` or await spawned tasks before leaving the scope when attribution must end with the block.

### ThreadPoolExecutor

`solwyn.run(...)` uses Python `contextvars`. Context propagates across asyncio tasks, but not into `ThreadPoolExecutor` workers. Use `solwyn.run_in_executor(...)` when submitting threaded work that should keep the active run tag:

```python
from concurrent.futures import ThreadPoolExecutor

with solwyn.run("nightly-batch"), ThreadPoolExecutor() as executor:
    future = solwyn.run_in_executor(executor, call_openai, prompt)
    result = future.result()
```

`run_in_executor(...)` returns the executor's `concurrent.futures.Future`, not an awaitable. In asyncio code, wrap it with `asyncio.wrap_future(future)`. If you submit directly to an executor, wrap the callable with `contextvars.copy_context().run(...)` yourself.

## Budget Enforcement

Set `budget_mode` to control spending:

```python
client = Solwyn(
    OpenAI(),
    api_key="sk_proj_...",
    budget_mode="hard_deny",
)
```

| Mode | Behavior |
|------|----------|
| `alert_only` | Log a warning when budget is exceeded (default) |
| `hard_deny` | Raise `BudgetExceededError` and block the call |

### Run-scoped leases

Token-billed calls inside `solwyn.run(...)` use budget leases by default. The first
eligible call requests a server grant; later calls reserve tokens from that grant
in memory, while renewal runs in the background at the refresh deadline or 75%
depletion. `close()` surrenders held leases. Non-run traffic, non-token/media
traffic, lease-ineligible runs or models, and clients with `lease_enabled=False`
keep using the per-call `/budgets/check` path.

Each reservation includes the input estimate plus the largest effective output
cap across the configured provider chain, including global defaults, provider
defaults, and Google/Bedrock nested cap fields. When a hop has no explicit cap,
`lease_output_bound_default` supplies that hop’s conservative output allowance.

During a control-plane outage, a live lease spends its remaining grant and then
its holder-specific headroom share. Exhausting both follows the customer’s
`budget_mode`: `hard_deny` blocks; `alert_only` proceeds with a warning. After a
lease expires, `fail_open=True` permits explicitly **uncounted** calls and tallies
them for the next successful renewal; `fail_open=False` enforces the last known
local bound. Uncounted episodes log `lease.uncounted_entry` immediately and
`lease.uncounted_continuing` at most every 30 seconds. Installing a fresh grant
ends the episode, so a later outage emits a new entry warning.

The global allow cache applies only to eligible legacy/non-run checks; it never
authorizes one run from another run’s state. Cloud usage reporting remains
asynchronous, so legacy cached work and the reporter flush interval can still
delay dashboard visibility.

```python
from solwyn import BudgetExceededError

try:
    response = client.chat.completions.create(model="gpt-5.5", messages=[...])
except BudgetExceededError as e:
    print(f"Budget limit: ${e.budget_limit}, usage: ${e.current_usage}")
```

## Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| `api_key` | `SOLWYN_API_KEY` | *required* | Solwyn project API key |
| `api_url` | `SOLWYN_API_URL` | `https://api.solwyn.ai` | Solwyn API endpoint |
| `fail_open` | `SOLWYN_FAIL_OPEN` | `True` | Allow LLM calls when Solwyn API is unreachable |
| `budget_mode` | `SOLWYN_BUDGET_MODE` | `alert_only` | Budget enforcement mode |
| `budget_check_cache_ttl` | `SOLWYN_BUDGET_CHECK_CACHE_TTL` | `5` | Allow-cache lifetime for eligible legacy/non-run checks |
| `budget_check_timeout` | `SOLWYN_BUDGET_CHECK_TIMEOUT` | `1.0` | Hot-path control-plane check/grant timeout in seconds |
| `lease_enabled` | `SOLWYN_LEASE_ENABLED` | `True` | Use in-memory token leases for eligible run-scoped calls |
| `lease_output_bound_default` | `SOLWYN_LEASE_OUTPUT_BOUND_DEFAULT` | `4096` | Output-token allowance when no configured provider hop has an explicit cap |
| `control_plane_failure_threshold` | `SOLWYN_CONTROL_PLANE_FAILURE_THRESHOLD` | `3` | Consecutive Solwyn API failures before local outage posture applies |
| `control_plane_recovery_timeout` | `SOLWYN_CONTROL_PLANE_RECOVERY_TIMEOUT` | `30.0` | Seconds before probing the Solwyn API after its breaker opens |

Failover and routing (`model=`, `fallback=`, `provider=`, `default_params=`, `selection_policy=`, and the failover tuning knobs) are configured in code only — they take client objects and policies, not strings. See [Provider Failover](https://docs.solwyn.ai/docs/sdk/guides/provider-failover) and [Configuration](https://docs.solwyn.ai/docs/sdk/guides/configuration).

`CostPolicy` is not yet active: the API does not send price hints yet, so selecting it currently falls back to health-based ordering (it logs a one-time warning when it does).

Use env vars to avoid passing credentials in code:

```sh
export SOLWYN_API_KEY="sk_proj_..."
```

```python
client = Solwyn(OpenAI())  # picks up from environment
```

### Failover timeouts

Failover is bounded by two independent timeouts. Like the rest of the failover tuning they are constructor-only (no `SOLWYN_*` env var) and server-governed: on a plan without the failover-tuning entitlement a custom value is replaced by the SDK default, warned once per client.

| Knob | Default | What it bounds |
|------|---------|----------------|
| `failover_total_timeout` | `30.0` | The **failover window** — the budget pre-flight, each hop's connect/pool slice, Retry-After sleeps, and advancement between hops |
| `failover_hop_read_timeout` | `600.0` | The **per-hop read/write bound** — how long one dispatched hop may spend reading a response |

The failover window deliberately does *not* cap a dispatched hop's read. A pre-send hang (connect, pool wait) is provably failover-safe, so it must fail inside the window; a read timeout is post-send *ambiguous* — the request may already have been served and billed — and under the default `failover_idempotency="safe"` it re-raises instead of failing over. Cutting a slow read at the failover window therefore buys no failover, only ambiguous spend.

`600.0` matches the openai/anthropic SDK defaults, so a wrapped call never times out earlier than the unwrapped SDK would. Because window expiry gates advancement *between* hops, at most one hop per call can consume the full read bound: worst-case wall clock is roughly one failover window plus one `failover_hop_read_timeout`.

Lower `failover_hop_read_timeout` if you would rather fail fast than wait out a slow generation (reasoning models, large `max_tokens`) — remembering that the fast failure is an ambiguous re-raise, not a failover:

```python
client = Solwyn(OpenAI(), api_key="sk_proj_...", failover_hop_read_timeout=120.0)
```

**google-genai limitation.** google-genai supports only a single whole-request timeout — it cannot split connect from read (a client-level `httpx.Timeout` via `client_args` is overridden per-request by the SDK itself). Solwyn therefore gives a google hop the read bound (`failover_hop_read_timeout`, default 600s) as its whole-request timeout. Consequence: a google **pre-send hang** (TCP/TLS connect, pool wait) is *not* bounded by `failover_total_timeout` — a single hung google hop can block up to the read bound and exhaust the failover window without ever failing over. (OS TCP timeouts typically cap a dead-host connect at ~1–2 minutes.) If you run google as primary with fallbacks and want a tighter failover guarantee, lower `failover_hop_read_timeout` — for google it bounds the entire request.

**Bedrock limitation.** boto3 has no per-call timeout override, so Solwyn cannot bound a Bedrock hop at all — the caller's botocore `Config(read_timeout=...)` governs. Building a Bedrock client whose botocore `Config` sets `read_timeout=None` logs a warning at build time: that is the one shape neither Solwyn nor botocore will bound.

## Error Handling

All SDK errors inherit from `SolwynError`:

| Exception | Raised when |
|-----------|-------------|
| `BudgetExceededError` | Budget exceeded in `hard_deny` mode |
| `ProviderUnavailableError` | Circuit breaker is open, or the failover chain is exhausted |
| `ConfigurationError` | Invalid API key format, invalid `provider=` override, or an untracked call surface (e.g. Bedrock `invoke_model`) |
| `UntranslatableRequestError` | A cross-provider failover hop cannot represent the request (structural labels only — never content) |
| `UntranslatableModelError` | No model mapping exists for a cross-provider failover hop |

Provider errors (e.g., `openai.RateLimitError`) pass through unmodified.

## Data Transparency

The SDK sends a `MetadataEvent` after each LLM call. This is everything it transmits:

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model name (e.g., `gpt-5.5`) |
| `provider` | `str` | Provider identifier (`openai`, `anthropic`, `google`, `bedrock`, `groq`, `openrouter`, …) |
| `modality` | `str` | Call modality (`text`, `image`, `audio`, `video`, `embedding`); `text` for chat, `embedding` for embeddings calls |
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `token_details` | `object` | Breakdown: cached, reasoning, audio, and image token buckets; `is_estimated` flags length-based estimates when a provider reports no usage |
| `media_usage` | `object \| None` | Non-token billable quantities for media calls — image counts, media durations in seconds, TTS character counts — plus `resolution`/`quality` variant selectors. Each quantity is `None` when the SDK can't observe it (never a zero-as-default), and the whole object is omitted for text/chat calls |
| `latency_ms` | `float` | Call duration in milliseconds |
| `status` | `str` | `success`, `error`, or `budget_denied` |
| `is_model_fallback` | `bool` | Whether the call was served by a same-provider entry in the `fallback=` chain after the primary model failed |
| `sdk_instance_id` | `str` | Per-process UUID for deduplication |
| `timestamp` | `datetime` | When the call completed (UTC) |
| `agent_run_id` | `str \| None` | Run id from the active `solwyn.run(...)` scope, if any. When omitted, the API creates `_auto-{sdk_instance_id}-{YYYY-MM-DD}` |
| `agent_run_name` | `str \| None` | Run name passed to `solwyn.run(...)`, if any |
| `provider_region` | `str \| None` | Cloud region of the serving endpoint (Bedrock — pricing is per model and region); omitted for other providers |
| `tags` | `object \| None` | Optional explicit customer-supplied tags from `solwyn.run(..., tags=...)` and `solwyn_tags=`. Never inferred from prompts or responses; omitted when empty or unset |

**The SDK never captures, logs, or transmits prompts or responses.** Explicit customer-supplied tags are outside this zero-content guarantee and are transmitted as provided. Prompt and response privacy is enforced by [structural tests](tests/unit/test_privacy_firewall.py) and the [privacy module](src/solwyn/_privacy.py).

## Release Compatibility

Wire-contract changes are API-first: Solwyn Cloud must accept new fields and enum values before an SDK release ships them. As of the current release line the Cloud API accepts the full wire contract — the `modality` discriminator, the `media_usage` quantities (image counts, media durations, character counts, and resolution/quality selectors), the image and audio `token_details` buckets, the Bedrock and OpenAI-compatible `provider` values, `provider_region`, bounded `tags`, `service_tier` on budget confirms, `token_details.is_estimated`, 2048-char model identifiers, and per-event ingest dispositions. Optional fields are omitted entirely (never `null`) when unset, so payloads for providers that don't use them are byte-identical to earlier releases.

## Requirements

Python 3.11+

## Contributing

```sh
make install          # install in dev mode
make install-hooks    # install pre-commit hook
make check            # lint + format + typecheck
make test             # run unit tests
```

## Links

- [Documentation](https://docs.solwyn.ai)
- [Solwyn Cloud](https://solwyn.ai) — Dashboard, alerts, and analytics
- [MPI.sh](https://mpi.sh) — LLM API pricing comparison

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
