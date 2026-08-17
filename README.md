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

Optional extras pin tested provider-SDK floors — `solwyn[openai]`, `solwyn[anthropic]`, `solwyn[google]`, `solwyn[bedrock]` (convenience only — the SDK never imports boto3), `solwyn[together]` (Together SDK 2.0+), or `solwyn[all]`:

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

For endpoints auto-detection can't name, pin the provider explicitly — on the constructor for the primary, or as the 4th element of a fallback spec:

```python
client = Solwyn(
    OpenAI(base_url="http://gpu-box:8080/v1", api_key="-"),
    api_key="sk_proj_...",
    provider="vllm",
    fallback=[(OpenAI(base_url="https://openrouter.ai/api/v1", api_key="sk-or-..."), "openrouter/auto"),
              (other_client, "my-model", {}, "ollama")],
)
```

`provider=` is an identity assertion, not a label applied after detection. It
bypasses `base_url` detection entirely and selects the named adapter. The pin
does not translate dialects, rewrite the endpoint, or synthesize a different
SDK client; construction still validates the actual client family and
sync/async mode, and raises `ConfigurationError(field="client")` for a mismatch.
Unknown provider names raise `ConfigurationError(field="provider")`. Fallback
provider pins follow the same rules.

This is useful for native OpenAI behind a corporate gateway or local proxy,
where an arbitrary `base_url` would otherwise look like a generic compatible
provider. Pinning `openai` preserves the native Responses surface and OpenAI
budget attribution:

```python
gateway = OpenAI(base_url="http://localhost:9999/v1", api_key="...")
client = Solwyn(gateway, api_key="sk_proj_...", provider="openai")

response = client.responses.create(
    model="gpt-5.5",
    input="Summarize the release notes.",
)
```

If that gateway omits or zeroes foreground Responses usage, `create` and
`parse` settle the request-length input estimate with
`token_details.is_estimated = true` instead of reporting exact `0/0`. The
unknown output remains zero in the estimate, and a lease-backed call keeps its
full reserved bound so that unseen output spend is not re-lent. Streaming and
the stream helper apply the same conservative policy when terminal usage is
missing.

**Token accounting.** Budgets and attribution depend on accurate per-call usage, and "OpenAI-compatible" endpoints differ most in exactly that. Solwyn requests streaming usage only from providers where that's documented-safe, reads it from the final chunk where it arrives automatically, and — when a provider reports no usage at all (or reports an unparseable/zeroed block alongside real content) — falls back to a length-based estimate that is **explicitly marked** (`token_details.is_estimated = true` on the wire; compatible-provider degradation also emits the existing one-time SDK warning where applicable). Degraded accounting is flagged, never silently zero.

The "never sent" entries above describe Solwyn's own injection policy. A `stream_options` you pass explicitly always reaches your configured provider untouched (drop-in contract); it is only stripped when a *failover hop* lands on a provider known to reject it.

**Pricing.** The SDK never computes cost. It reports the served `(provider, model)` verbatim — for OpenRouter that's the full model slug (e.g. `anthropic/claude-sonnet-4.5`) — and Solwyn Cloud's PricingService prices it. Models unknown to the catalog are surfaced as unpriced on the dashboard rather than silently costed at $0.

**Failover.** Compat providers participate fully in failover. Between two OpenAI-dialect providers (e.g. Groq → OpenRouter) requests pass through natively — tools, JSON mode, and streaming included (`max_completion_tokens` is rewritten to `max_tokens` for targets that need the legacy key). Per-call `extra_headers`/`extra_query`/`extra_body` are stripped on cross-provider hops — they're endpoint-scoped, authored for the original endpoint — though the fallback entry's own `default_params` versions still apply. Across dialects (e.g. Groq → Anthropic) the standard translation subset applies.

**Known limitation.** Circuit-breaker health, latency signals, and failover labeling key off the provider *name*. Two chain entries that resolve to the same name (two Azure resources, two unnamed gateways both detected as `openai_compatible`) share one health domain and are reported as model fallbacks of each other. For the same reason, a hop between same-name entries skips cross-provider request sanitization — `stream_options` stripping, the `max_completion_tokens` → `max_tokens` rewrite, and endpoint-scoped param stripping (`extra_headers`/`extra_query`/`extra_body`). A `stream_options` or gateway header you authored for the first endpoint reaches the second untouched and can 4xx there. Give distinct endpoints distinct provider identities where possible — explicit `provider=` on the constructor, or the 4th element of a fallback spec.

**Known limitation.** `solwyn_tags` is removed only on intercepted provider calls. On non-intercepted surfaces such as `client.files.create(...)`, it is passed through to the provider SDK, which can raise `TypeError` for the unexpected keyword. Call those surfaces without `solwyn_tags`.

## Media surfaces

Beyond chat, Solwyn tracks the non-text surfaces that spend money. Each rides the same budget-check → provider call → confirm lifecycle as a chat call, tagged with its modality (`embedding`, `image`, `audio`, `video`) so Solwyn Cloud's PricingService prices it on the right card. There is no cross-provider failover for these surfaces — an embedding vector or a generated image isn't interchangeable across providers.

| Surface | OpenAI dialect (native + compatible) | Google (Gemini) |
|---------|--------------------------------------|-----------------|
| Embeddings | `client.embeddings.create` | `client.models.embed_content` |
| Images | `client.images.generate` / `client.images.edit` | `client.models.generate_images` (Imagen) |
| Audio — transcription | `client.audio.transcriptions.create` (incl. Groq whisper) | — |
| Audio — speech (TTS) | `client.audio.speech.create` | — |
| Video | `client.videos.create` (Sora) | `client.models.generate_videos` (Veo) |
| Responses | Native OpenAI + Azure OpenAI: `client.responses.create` / `.parse` / `.stream` | — |

Billable quantities are read from the response's usage block where it exists (gpt-image token buckets, whisper duration) and derived from the request where a provider reports none — image counts from `n=`, TTS character counts from `input=`, video seconds from the request. Whatever the SDK can't observe stays `None`, and the call is tracked **unpriced** rather than settled at a silent $0. Only lengths, counts, durations, and variant selectors are ever measured — never the media itself.

**Posture notes.**

- **Whisper needs a JSON `response_format` to be priced.** `whisper-1` reports its billable duration only under a JSON response format. A non-JSON `response_format` (`text` / `srt` / `vtt`) carries no usage, so the call is tracked unpriced with a one-time hint to pass `response_format="json"` (or `"verbose_json"`) for priced tracking.
- **`gpt-4o-mini-tts` is untracked.** Token-billed TTS models publish no usage metadata, so their audio-output tokens are unobservable. Rather than settle a silent $0, the call follows the configured untracked posture (under the default `warn`, it passes through after a one-time warning with no budget check or cost event).
- **`audio.translations` is untracked.** The translations sub-surface isn't intercepted yet, so it follows the same configured posture.

## Strict coverage controls

- **OpenAI Responses:** Native OpenAI and Azure OpenAI
  `responses.create(...)`, `responses.parse(...)`, and the
  `responses.stream(...)` context-manager helper are budget-metered for sync
  and async clients. `create(stream=True)` is
  supported; streaming `parse` is not metered, so any effective streaming parse
  request is refused. The stream helper's new-response overload preserves the
  SDK's context-manager and `get_final_response()` behavior while settling
  terminal usage or a conservative estimate on early exit; a helper closed
  before it is ever entered sent no provider request, so it releases its
  reservation instead of settling. Foreground non-streaming
  calls likewise settle a conservative marked
  estimate when Responses usage is missing or zeroed; lease-backed calls hold
  the reserved bound because output usage is unobservable. Its existing-response
  retrieval overload (`response_id` / `starting_after`) creates no new spend,
  so it is a reviewed raw pass-through: no defaults, budget check, or duplicate
  settlement are applied. Every other Responses leaf, including beta and raw
  response helpers, remains guarded by `on_unmetered`.
  `background=True` create calls are refused because queued responses expose no
  create-time usage. Because the OpenAI SDK serializes `extra_body` after named
  arguments, metering-critical overrides for `model`, `input`, `instructions`,
  `max_output_tokens`, or `stream` are refused with
  `ConfigurationError(field="extra_body")`; pass those values as top-level
  Responses arguments instead. Other vendor-specific `extra_body` extensions
  pass through unchanged. Other OpenAI-compatible providers retain their raw
  Responses managers and follow the guarded unmetered posture.

Solwyn classifies the public pre-call capability graph of every supported
wrapped client. Tracked leaves are intercepted as usual. Resource namespaces
stay guarded so access to a parent never grants present or future descendants.
Known untracked leaves and newly observed leaves follow `on_unmetered`:

- `on_unmetered="warn"` logs once and permits the call (the compatibility default).
- `on_unmetered="raise"` refuses the call before provider I/O with
  `UntrackedSpendSurfaceError`. This is strict mode.
- `on_unmetered="allow"` permits the call without warning.

By default, unacknowledged `warn` and `allow` observations schedule structural
advisory POSTs to the project-implicit `/api/v1/untracked-surfaces` route. A
background reporter thread/task sends them immediately on first observation
and then at a 15-minute cadence; shutdown may make a deadline-bounded
best-effort final attempt. Send failures are silent. Payloads contain only the
dotted surface path, bounded provider/client-shape, sync/async, rule, scope,
and posture fields, approximate occurrence counts and first/last timestamps,
and random SDK-instance/report identifiers. Model names, request arguments,
prompts, and responses are never included. This is an approximate signal, not
billing truth, and it does not add a budget check or cost event.

Set `report_untracked_surfaces=False` or
`SOLWYN_REPORT_UNTRACKED_SURFACES=false` to opt out of advisory egress. This
does not change the local `on_unmetered` posture; warnings, allowed calls, and
strict refusals continue unchanged.

Set the posture in the constructor or with `SOLWYN_ON_UNMETERED=raise`:

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(
    OpenAI(),
    api_key="sk_proj_...",
    on_unmetered="raise",
    acknowledge_untracked={"responses.retrieve"},
)
```

Acknowledgments are narrow, deliberate exceptions to the posture. Each token
must name an applicable, observed terminal capability; it grants only that
leaf. Namespace tokens such as `responses` are invalid, as are wildcards,
typos, tracked leaves, blocked leaves, and unsupported leaves. Namespace
objects remain guarded after an acknowledgment, so `responses.retrieve` does not
authorize a future sibling. The equivalent comma-delimited environment
encoding is
`SOLWYN_ACKNOWLEDGE_UNTRACKED="responses.retrieve,audio.speech.create:gpt-4o-mini-tts"`.
The conditional token for token-billed TTS is exactly
`audio.speech.create:gpt-4o-mini-tts`; acknowledging ordinary
`audio.speech.create` does not cover that model-specific exception.

Provider applicability is explicit. Native OpenAI video is tracked through
`videos.create`; video on an OpenAI-compatible provider is unsupported and
raises `UnsupportedSurfaceError` before dispatch. An acknowledgment cannot
turn an unsupported adapter surface into a supported one.

#### Tested SDK version intervals

The surface contract is verified against pinned structural breakpoints per
provider (see `tests/provider_surface_intervals.json`): floor, named
breakpoints, and latest. Versions between tested breakpoints may expose
surfaces we have not classified; those resolve as `unknown` and follow
`on_unmetered`.

Use `solwyn.coverage(client)` to review the exact effective graph without
calling a provider operation. Coverage is computed locally and transmits
nothing. It reads structural client metadata only—never prompts, responses,
credentials, or request content. The report separates policy decisions from
dispatch behavior and includes provider-chain usage guarantees.

For CI, pin an independently reviewed literal fingerprint. This example is the
exhaustive strict, unacknowledged fingerprint exercised against
`openai==2.53.0` by this repository's real-client test:

```python
from openai import OpenAI
from solwyn import CoverageFingerprint, Solwyn, coverage

audit_client = Solwyn(
    OpenAI(),
    api_key="sk_proj_...",
    on_unmetered="raise",
)

OPENAI_STRICT_FINGERPRINT = CoverageFingerprint(
    guarded_namespaces="sha256:38de7d9d718f03bc61f4a24e24f131c1a018434fcb38eb5cb7371290fc72e074",
    tracked="sha256:586f19c33f350871240a3498fbfa255c9759bec35e1285a8fccfeb937ec68148",
    untracked="sha256:1a3192143f409c0e38edcee32232d411c40706fcddd7a7d729403d67690ffb2c",
    unknown="sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    scoped_escapes="sha256:6808a0f2ac290c9d4d1504b21b1c0ba98267636ced4234416b53533b29bb4073",
    blocked="sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    unsupported="sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    conditional="sha256:ce837f71d1fc97849872c5d0f86b0b1f26e1bc4e46a29c3b1b8004bf4b9bcb77",
    safe="sha256:9029368e5fa0a7bf4260cc782560c8ec9a53c948fc280102b1c3633eee5234c5",
)

report = coverage(audit_client)
report.expect(OPENAI_STRICT_FINGERPRINT)
```

Azure OpenAI exposes the same metered Responses trio, but its surrounding
capability graph is distinct. Audit and pin an Azure client independently; do
not reuse the native OpenAI fingerprint for Azure.

When a provider SDK changes, inspect `report.entries`, decide whether each
change is acceptable, and then paste a newly reviewed literal. Never approve a
report with a fingerprint derived from that same report in the assertion; that
would make the check tautological.

Strict mode is not a sandbox. It is a cooperative guard around the wrapper's
public pre-call surface. The following can bypass pre-call strict enforcement:
retaining the raw provider client, accessing private wrapper state,
acknowledging a scoped raw escape, or invoking native behavior on a returned
response, page, stream, job, or operation object. Keep those capabilities out
of code that relies on strict enforcement, or review their use explicitly.

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

client = solwyn.Solwyn(
    OpenAI(),
    api_key="sk_proj_...",
    tags={"environment": "prod", "service": "research"},
)

with solwyn.run("nightly-batch", tags={"team": "research", "env": "prod"}) as run_id:
    client.chat.completions.create(model="gpt-5.5", messages=[...])
    client.chat.completions.create(
        model="gpt-5.5",
        messages=[...],
        solwyn_tags={"env": "staging", "job": "backfill"},
    )
```

The constructor's `tags=` mapping supplies client defaults to every intercepted call. You can set the same defaults from the environment with comma-separated `key=value` entries:

```sh
export SOLWYN_TAGS="environment=prod,service=research"
```

The environment format splits each entry at its first `=`, so values may contain `=` but cannot contain commas. Use constructor `tags={"segment": "east,canary"}` when a value contains a comma.

Nested runs inherit outer tags additively. Inner tags overwrite only keys they reuse, so a sub-agent keeps its orchestrator's attribution without repeating it:

```python
with solwyn.run("orchestrator", tags={"team": "research", "workflow": "eval"}):
    with solwyn.run("critic", tags={"agent": "critic", "team": "safety"}):
        client.chat.completions.create(model="gpt-5.5", messages=[...])
        # Tags: team=safety, workflow=eval, agent=critic

    with solwyn.run("isolated", tags={"agent": "one-off"}, inherit_tags=False):
        client.chat.completions.create(model="gpt-5.5", messages=[...])
        # Tags: agent=one-off
```

Use the reserved `solwyn_tags=` keyword as a call argument, not in `default_params`; it is removed before provider dispatch. Precedence is client defaults, then the active run scope, then per-call tags, with the higher-precedence value kept on conflicts. Each supplied mapping allows at most 10 string keys, keys must contain 1–64 characters, and string values may contain 0–256 characters. Keys and values cannot contain NUL characters. The SDK validates and copies mappings at client or run creation and at call start, so invalid input fails eagerly and later caller mutation cannot change attribution.

The combined map can exceed 10 keys even when each supplied mapping is valid. In that case the SDK keeps 10 deterministically: per-call keys first, then active-scope keys, then client-default keys, preserving insertion order within each layer. It emits one `SolwynTagsClampedWarning` for the overflowing call and still dispatches the provider request; lower-priority excess tags are absent from that call's event.

Tagged calls always perform a fresh budget check because tag selectors can carry independent spending caps. They cannot use the project allow cache or the agent-run budget lease path. Setting constructor `tags=` therefore puts every intercepted call on the control-plane request path; leave client defaults unset when that attribution is not needed.

Use `solwyn.current_run_context()` to read the active `RunContext(id, name, tags)`. Its tag mapping is a fresh copy, so caller mutation cannot change the active scope. `solwyn.current_run()` returns the `(id, name)` pair.

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

## Testing your budget enforcement

`FakeControlPlane` exercises the production control-plane transport seam with
zero network traffic and the same request, response, and Pydantic wire models as
Solwyn Cloud. You create and own the plane, wrap your provider client with it,
and inspect its request recordings after the call. The provider remains your
responsibility: mock it normally whenever the test can reach provider dispatch.

The double never prices anything — the API owns pricing. Scripted denials test your handling, not your budget math.

### 1. Test a deny handler

The magic denial happens during preflight, so no provider request can occur.
The wrapper context manager closes the client even if the assertion fails.

<!-- test-double-snippet:deny-handler -->
```python
from openai import OpenAI
import pytest
from solwyn import BudgetExceededError
from solwyn.testing import FakeControlPlane

def test_deny_handler():
    plane = FakeControlPlane()
    with plane.wrap(OpenAI(api_key="test")) as client:
        with pytest.raises(BudgetExceededError):
            client.chat.completions.create(model="solwyn-test/deny", messages=[])
```

Magic models are reserved, deterministic verdict scripts:

| Model | Scripted control-plane behavior |
|-------|---------------------------------|
| `solwyn-test/deny` | Hard denial for the monthly period |
| `solwyn-test/deny-alert` | Monthly denial in `alert_only` mode, so dispatch proceeds with a warning |
| `solwyn-test/deny-tag` | Hard denial attributed to the `tag` period |
| `solwyn-test/deny-stopped` | Hard denial attributed to `run_stopped` |
| `solwyn-test/runaway` | First check per run is allowed; later checks are denied for `agent_run` |
| `solwyn-test/lease-ineligible` | Allow the call but make its run ineligible for a token lease |

For overlapping scripts, precedence is transport failure → endpoint refusal → verdict → allow.
An outage therefore tests unreachable posture without a
scripted denial leaking through, while a reachable endpoint refusal wins over
the normal verdict.

### 2. Test fail-open posture

Mock the provider separately—here with `respx`—and assert both that dispatch
proceeded and that the control-plane warning surfaced.

<!-- test-double-snippet:fail-open -->
```python
import logging
import httpx
import respx
from openai import OpenAI
from solwyn.testing import FakeControlPlane

def test_fail_open_provider_proceeds(caplog):
    plane = FakeControlPlane()
    provider = OpenAI(base_url="https://provider.test/v1", api_key="test")
    with respx.mock:
        route = respx.post("https://provider.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "gpt-5.5", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "served"},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            })
        )
        with (
            plane.wrap(provider, fail_open=True, lease_enabled=False) as client,
            caplog.at_level(logging.WARNING),
            plane.outage(),
        ):
            response = client.chat.completions.create(model="gpt-5.5", messages=[])
    assert route.called and response.choices[0].message.content == "served"
    assert "budget check failed" in caplog.text.lower()
```

### 3. Run a deny → outage → recovery game day

Compose scenarios on one caller-owned plane to prove that a known hard denial
is preserved during an outage and cleared only by a recovered allow verdict.

<!-- test-double-snippet:game-day -->
```python
import httpx
import pytest
import respx
from openai import OpenAI
from solwyn import BudgetExceededError
from solwyn.testing import FakeControlPlane

def test_deny_outage_recovery():
    plane = FakeControlPlane()
    provider = OpenAI(base_url="https://provider.test/v1", api_key="test")
    with respx.mock:
        route = respx.post("https://provider.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "gpt-5.5", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "served"},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            })
        )
        with plane.wrap(provider, fail_open=True, lease_enabled=False) as client:
            with pytest.raises(BudgetExceededError):
                client.chat.completions.create(model="solwyn-test/deny", messages=[])
            with plane.outage(), pytest.raises(BudgetExceededError):
                client.chat.completions.create(model="gpt-5.5", messages=[])
            recovered = client.chat.completions.create(model="gpt-5.5", messages=[])
    assert route.call_count == 1
    assert recovered.choices[0].message.content == "served"
```

See the in-repo
[`test_gameday_recipes.py`](tests/unit/testing_double/test_gameday_recipes.py)
for the full refusal, breaker, reporter, lease-drawdown, and recovery ladder.

### Opt-in pytest fixtures

Fixtures never auto-register. Enable them in the test module (or your own
`conftest.py`) and request both fixtures when you want to script and inspect the
same plane. `solwyn_test_client` is the normal `Solwyn` wrapper around a private
denial-only dispatch sentinel; it does not simulate provider responses.

<!-- test-double-snippet:pytest-fixtures -->
```python
import pytest
from solwyn import BudgetExceededError

pytest_plugins = ["solwyn.testing.pytest_plugin"]

def test_denial_fixture(solwyn_control_plane, solwyn_test_client):
    with pytest.raises(BudgetExceededError):
        solwyn_test_client.chat.completions.create(
            model="solwyn-test/deny", messages=[]
        )
    assert len(solwyn_control_plane.checks) == 1
```

## Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| `api_key` | `SOLWYN_API_KEY` | *required* | Solwyn project API key |
| `api_url` | `SOLWYN_API_URL` | `https://api.solwyn.ai` | Solwyn API endpoint |
| `tags` | `SOLWYN_TAGS` | `None` | Default spend tags for intercepted calls; env format is comma-separated `key=value` entries |
| `fail_open` | `SOLWYN_FAIL_OPEN` | `True` | Allow LLM calls when Solwyn API is unreachable |
| `budget_mode` | `SOLWYN_BUDGET_MODE` | `alert_only` | Budget enforcement mode |
| `budget_check_cache_ttl` | `SOLWYN_BUDGET_CHECK_CACHE_TTL` | `5` | Allow-cache lifetime for eligible legacy/non-run checks |
| `budget_check_timeout` | `SOLWYN_BUDGET_CHECK_TIMEOUT` | `1.0` | Hot-path control-plane check/grant timeout in seconds |
| `lease_enabled` | `SOLWYN_LEASE_ENABLED` | `True` | Use in-memory token leases for eligible run-scoped calls |
| `lease_output_bound_default` | `SOLWYN_LEASE_OUTPUT_BOUND_DEFAULT` | `4096` | Output-token allowance when no configured provider hop has an explicit cap |
| `on_unmetered` | `SOLWYN_ON_UNMETERED` | `warn` | Handle untracked or unknown pre-call capabilities with `warn`, `raise`, or `allow` |
| `report_untracked_surfaces` | `SOLWYN_REPORT_UNTRACKED_SURFACES` | `True` | Send optional structural advisory reports for unacknowledged `warn`/`allow` observations; set false to keep them local |
| `acknowledge_untracked` | `SOLWYN_ACKNOWLEDGE_UNTRACKED` | empty | Exact terminal capability tokens; env format is comma-delimited |
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

Failover is bounded by two independent timeouts. Both are constructor-only (no `SOLWYN_*` env var) and server-governed: on a plan without the failover-tuning entitlement a custom value is replaced by the SDK default, warned once per client.

| Knob | Default | What it bounds |
|------|---------|----------------|
| `failover_total_timeout` | `30.0` | The **failover window** — the budget pre-flight, each hop's connect/pool slice, Retry-After sleeps, and advancement between hops |
| `failover_hop_read_timeout` | `600.0` | The **per-hop read/write bound** — how long one dispatched hop may spend reading a response |

The failover window deliberately does *not* cap a dispatched hop's read. A pre-send hang (connect, pool wait) is provably failover-safe, so it must fail inside the window; a read timeout is post-send *ambiguous* — the request may already have been served and billed — and under the default `failover_idempotency="safe"` it re-raises instead of failing over. Cutting a slow read at the failover window therefore buys no failover, only ambiguous spend.

`600.0` matches the openai/anthropic SDK's read/write default, so a wrapped call's read/write bound never fires earlier than the unwrapped SDK's would — connect/pool instead track the shrinking failover window. Because window expiry gates advancement *between* hops, at most one hop per call can consume the full read bound: worst-case wall clock is roughly one failover window plus one `failover_hop_read_timeout`.

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
| `BudgetExceededError` | Cloud denies a budget check in `hard_deny` mode, or local enforcement denies while Cloud is unreachable and `fail_open=False` |
| `RunStoppedError` | A dashboard stop is learned for an agent run — on the next budget check for per-call traffic, or after lease renewal or re-grant for leased traffic |
| `ProviderUnavailableError` | Circuit breaker is open, or the failover chain is exhausted |
| `ConfigurationError` | Invalid API key format, invalid `provider=` pin/client pairing, or an untracked call surface (e.g. Bedrock `invoke_model`) |
| `UntrackedSpendSurfaceError` | Strict coverage posture refuses an unacknowledged untracked or unknown capability before provider I/O |
| `UnsupportedSurfaceError` | The selected provider adapter does not support an explicit Solwyn wrapper surface |
| `UntranslatableRequestError` | A cross-provider failover hop cannot represent the request (structural labels only — never content) |
| `UntranslatableModelError` | No model mapping exists for a cross-provider failover hop |

`RunStoppedError` is a `BudgetExceededError` subclass, so existing
`except BudgetExceededError` handlers continue to catch dashboard stops. A stop
does not interrupt requests already in flight or streams already returned.

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
