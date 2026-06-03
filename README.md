# Solwyn Python SDK

Budget enforcement, circuit breaking, and usage tracking for OpenAI, Anthropic, and Google LLM clients.

[![CI](https://github.com/solwyn-ai/solwyn-python/actions/workflows/ci.yml/badge.svg)](https://github.com/solwyn-ai/solwyn-python/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/solwyn)](https://pypi.org/project/solwyn/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/solwyn)](https://pypi.org/project/solwyn/)
[![License](https://img.shields.io/github/license/solwyn-ai/solwyn-python)](LICENSE)

Solwyn wraps your existing LLM client. Calls go directly to the provider — the SDK only reports metadata (token counts, latency, model name) to the Solwyn API. **Prompts and responses never leave your application.**

## Installation

```sh
pip install solwyn
```

For improved token estimation with OpenAI models:

```sh
pip install solwyn[openai]
```

## Quick Start

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(
    OpenAI(),
    api_key="sk_proj_...",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

client.close()
```

Or use as a context manager:

```python
with Solwyn(OpenAI(), api_key="sk_proj_...") as client:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

## Providers

### OpenAI

```python
from openai import OpenAI
from solwyn import Solwyn

client = Solwyn(OpenAI(), api_key="sk_proj_...")
response = client.chat.completions.create(model="gpt-4o", messages=[...])
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
response = client.models.generate_content(model="gemini-2.0-flash", contents="Hello!")
```

## Async

```python
from openai import AsyncOpenAI
from solwyn import AsyncSolwyn

async with AsyncSolwyn(
    AsyncOpenAI(),
    api_key="sk_proj_...",
) as client:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

## Streaming

Pass `stream=True` as you normally would. Solwyn wraps the stream transparently and reports usage when it completes:

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Tagging Calls with Agent Runs

Wrap a unit of work with `solwyn.run(name)` to attribute every LLM call inside it to a single agent run. The dashboard groups cost and latency by run, so you can see "this nightly batch cost $4.20."

```python
import solwyn
from openai import OpenAI

client = solwyn.Solwyn(OpenAI(), api_key="sk_proj_...")

with solwyn.run("nightly-batch") as run_id:
    client.chat.completions.create(model="gpt-4o", messages=[...])
    client.chat.completions.create(model="gpt-4o", messages=[...])
```

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

```python
from solwyn import BudgetExceededError

try:
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
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
| `fallback_model` | `SOLWYN_FALLBACK_MODEL` | `None` | Model name to retry with when the primary call fails (same provider, same client) |

Use env vars to avoid passing credentials in code:

```sh
export SOLWYN_API_KEY="sk_proj_..."
```

```python
client = Solwyn(OpenAI())  # picks up from environment
```

## Error Handling

All SDK errors inherit from `SolwynError`:

| Exception | Raised when |
|-----------|-------------|
| `BudgetExceededError` | Budget exceeded in `hard_deny` mode |
| `ProviderUnavailableError` | Circuit breaker is open |
| `ConfigurationError` | Invalid API key format |

Provider errors (e.g., `openai.RateLimitError`) pass through unmodified.

## Data Transparency

The SDK sends a `MetadataEvent` after each LLM call. This is everything it transmits:

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model name (e.g., `gpt-4o`) |
| `provider` | `str` | `openai`, `anthropic`, or `google` |
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `token_details` | `object` | Breakdown: cached, reasoning, audio tokens |
| `latency_ms` | `float` | Call duration in milliseconds |
| `status` | `str` | `success`, `error`, or `budget_denied` |
| `is_model_fallback` | `bool` | Whether the call used `fallback_model` after the primary model failed |
| `sdk_instance_id` | `str` | Per-process UUID for deduplication |
| `timestamp` | `datetime` | When the call completed (UTC) |
| `agent_run_id` | `str \| None` | Run id from the active `solwyn.run(...)` scope, if any. When omitted, the API creates `_auto-{sdk_instance_id}-{YYYY-MM-DD}` |
| `agent_run_name` | `str \| None` | Run name passed to `solwyn.run(...)`, if any |

**The SDK never captures, logs, or transmits prompts or responses.** This is enforced by [structural tests](tests/unit/test_privacy_firewall.py) and the [privacy module](src/solwyn/_privacy.py).

## Release Compatibility

Provider failover confirms include `provider` and `call_id` on `POST /api/v1/budgets/confirm` so Solwyn Cloud can price and deduplicate the served provider. Release this SDK only after Solwyn Cloud accepts those fields; deploy the Cloud API first.

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
