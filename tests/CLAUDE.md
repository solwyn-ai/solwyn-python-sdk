# Tests

## Structure

```
tests/
  unit/                   # All unit tests — mock all external deps (one exception below)
    conftest.py           # Shared fixtures + constants (VALID_API_KEY, ALLOW_BUDGET_RESPONSE)
    test_providers/       # Provider adapter tests
    test_real_sdk_*.py    # Real-SDK smoke tests (the sanctioned exception — see below)
  integration/            # Real HTTP against the Solwyn API
    conftest.py           # Auto-bootstraps test user + project + API key; E2E wrapper harness
    fake_provider.py      # Local fake OpenAI-compatible server (stdlib-only test infra)
    test_e2e_*.py         # Real Solwyn(OpenAI(...)) wrappers over the live pipeline
```

No `__init__.py` files anywhere. Import shared constants with `from conftest import ...` (absolute, not relative).

## Real-SDK Smoke Tests

`tests/unit/test_real_sdk_*.py` are the ONE sanctioned exception to "unit tests
mock everything": they construct GENUINE provider SDK objects (openai,
anthropic, together, boto3 — test-only dev deps) to catch drift that synthetic
stubs can't (renamed classes, changed base_url shapes, botocore internals).
They stay hermetic — client construction and botocore `Stubber` only, never
network. Each SDK is gated by `pytest.importorskip` (`ModuleNotFoundError` →
clean skip, e.g. the wheel-only publish CI job), so the suite passes with no
provider SDKs installed. `src/` still never imports a provider SDK; detection
stays duck-typed. All OTHER unit tests must keep mocking.

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
- **Failover-session seams**: `FakeProviderServer.fail_next(status, count=N)` queues error responses; run two servers as primary + fallback clients; streaming always ends with a usage-bearing final chunk.

## Conventions

- Arrange-Act-Assert pattern with comments
- Mock at service boundaries, never internal methods
- All mocks must use `spec=` (catches renamed methods)
- Async mocking: `AsyncMock` for the async method, `MagicMock` for the response (httpx Response.json() is sync)
- Shared test constants in `conftest.py` — import, don't redefine
- `_mock_anthropic_client()` exists in two test files with DIFFERENT signatures — do NOT consolidate
- Mock response dicts must include all fields the API returns — no relying on Pydantic defaults
