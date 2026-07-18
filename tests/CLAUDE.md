# Tests

## Structure

```
tests/
  unit/                   # All unit tests — mock all external deps
    conftest.py           # Shared fixtures + constants (VALID_API_KEY, ALLOW_BUDGET_RESPONSE)
    test_providers/       # Provider adapter tests
  integration/            # Real HTTP against the Solwyn API
    conftest.py           # Auto-bootstraps test user + project + API key; E2E wrapper harness
    fake_provider.py      # Local fake providers: OpenAI-compatible + anthropic-dialect servers (stdlib-only test infra)
    test_e2e_*.py         # Real Solwyn(OpenAI(...)) wrappers over the live pipeline
```

No `__init__.py` files anywhere. Import shared constants with `from conftest import ...` (absolute, not relative).

## Markers

Every test must have a category marker. Marker order on methods:

```python
@pytest.mark.unit       # 1. category
@pytest.mark.asyncio    # 2. execution mode
@pytest.mark.parametrize(...)  # 3. parametrize
```

Registered markers: `unit`, `integration`, `chaos`, `performance`, `stress`.

CI runs only `unit`. Integration is opt-in (PR label `run-integration` or `workflow_dispatch`).

## Running Integration Tests

```bash
# In ../core (smoke mode — Loops disabled so signup returns a token directly):
make db-setup && make smoke-api-dev

# In this repo:
uv run pytest tests/ -m integration -v
```

The integration conftest auto-creates a test user and project via the API. Set `SOLWYN_TEST_API_URL` (default `http://127.0.0.1:8080`), or `SOLWYN_TEST_API_KEY` + `SOLWYN_TEST_PROJECT_ID` to skip bootstrap.

Bootstrap needs the API's dev signup fast path. `make dev-api` with a Loops key configured runs verify-first signup (OTP emailed, stored hashed) — the suite then skips with guidance rather than failing. Use `make smoke-api-dev`, or provide `SOLWYN_TEST_API_KEY` (happy-path tests only; the hard-deny fixture always bootstraps its own project).

## E2E Wrapper Harness

`test_e2e_*.py` are the only tests that run the full interception pipeline (`client.py`, `_proxies.py`, `stream.py`, `providers/`) over real HTTP on both sides: a real `openai` client (dev dep) pointed at `fake_provider.FakeProviderServer`, wrapped in a real `Solwyn`/`AsyncSolwyn` talking to the live API.

- **Detection doubles as coverage**: an ephemeral-port server is detected as the `openai_compatible` catch-all; `start_on_conventional_port()` (1234/11434/8000) exercises the lmstudio/ollama/vllm port heuristic. Tests skip if all three ports are occupied by real services.
- **`WireRecorder`** (conftest) records `confirm_cost` / `report` / `report_settlement` payloads while delegating to the real implementations — wire assertions with live delivery preserved. Streamed calls WITH a reservation settle via `report_settlement`; errors and reservation-less streams still go through `report`.
- **Conventions**: wrapped clients are built with `budget_check_cache_ttl=0` (an allow-cache hit has no reservation_id → confirm silently skips) and happy paths use a model the API prices (`gpt-4o`; unpriced models are allowed without a reservation).
- **Budget denial** is server-driven: `hard_denied_credentials` creates a `hard_deny` project ($0.05 limit) and burns it with one large confirm, so a wrapper check raises `BudgetExceededError` before any provider call.
- **Failover-session seams**: `FakeProviderServer.fail_next(status, count=1, *, retry_after=None)` queues real error responses (optionally with a `Retry-After` header); `set_omit_usage()` strips usage from JSON and stream responses (the SDK must fall back to a length-based estimate with `is_estimated=True`); `drop_next_stream(after_chunks=N)` truncates the next stream without the HTTP/1.1 chunked terminator, so the client sees `httpx.RemoteProtocolError`, not clean EOF. All seams reset between tests via the autouse fixture.
- **Fallback servers**: session fixtures `fake_provider` (120/45 tokens), `fake_provider_fallback` (77/33), and `fake_provider_anthropic` (88/44, anthropic dialect — serves `POST /v1/messages`, rejects `stream:true` with 501, `base_url` without `/v1`). Distinct token counts prove which server served a call; the 4-tuple fallback form `(client, model, params, "groq")` makes attribution distinguishable within the same dialect. Streaming ends with a usage-bearing final chunk unless `set_omit_usage()` is active.
- **Entitlement ceiling**: signup-bootstrapped accounts are free tier, so the server's failover-tuning directive forces `same_provider_retries` (and other tuning fields) back to SDK defaults on every call — E2E tests can prove the suppression but NOT tuning-dependent behavior like Retry-After re-attempts (unit-only until the harness can provision a team/scale-tier account).

## Conventions

- Arrange-Act-Assert pattern with comments
- Mock at service boundaries, never internal methods
- All mocks must use `spec=` (catches renamed methods)
- Async mocking: `AsyncMock` for the async method, `MagicMock` for the response (httpx Response.json() is sync)
- Shared test constants in `conftest.py` — import, don't redefine
- `_mock_anthropic_client()` exists in two test files with DIFFERENT signatures — do NOT consolidate
- Mock response dicts must include all fields the API returns — no relying on Pydantic defaults
