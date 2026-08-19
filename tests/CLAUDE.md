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
    fake_provider.py      # Local fake providers: OpenAI chat + Responses and anthropic-dialect servers (stdlib-only test infra)
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

## Double vs Live Contract Tests

`tests/unit/testing_double/` owns deterministic control-plane behavior: scripted
verdicts and failures, recordings, reservation and lease state, wrapper fixture
ergonomics, and the shared contract pack running with zero network. Provider
dispatch remains separately mocked there. Real authentication, routing, storage,
pricing-catalog behavior, and deployed API integration stay in `tests/integration/`.
`tests/integration/test_live_contract.py` remains the live fidelity specification, and
`tests/integration/test_contract_against_live.py` is the file that runs `solwyn.testing.contract`
against the live API — that's the file to edit when the pack changes.

Run-control and denial-receipt parity rides that shared pack too:
`assert_run_control_contract` (the `run_control` v1 terminate directive on the
check and lease channels) and `assert_receipt_ingest_contract` (per-call
receipts and aggregate replays through `/metadata/ingest`) run in both lanes, so
a drifted directive or receipt field fails the double and the live API alike.
Each lane still scripts its own server state: the double calls
`plane.stop_run(...)`, the live lane POSTs the dashboard stop.

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

The integration conftest auto-creates a test user and project via the API. Set `SOLWYN_TEST_API_URL` (default `http://127.0.0.1:8080`), or `SOLWYN_TEST_API_KEY` + `SOLWYN_TEST_PROJECT_ID` to skip bootstrap. Direct `test_fake_provider.py` harness tests are the one exception: a targeted run has no control-plane dependency, so wire-shape coverage remains runnable while the API is offline.

Bootstrap needs the API's dev signup fast path. `make dev-api` with a Loops key configured runs verify-first signup (OTP emailed, stored hashed) — the suite then skips with guidance rather than failing. Use `make smoke-api-dev`, or provide `SOLWYN_TEST_API_KEY` (happy-path tests only; the hard-deny fixture always bootstraps its own project).

## E2E Wrapper Harness

`test_e2e_*.py` are the only tests that run the full interception pipeline (`client.py`, `_proxies.py`, `stream.py`, `providers/`) over real HTTP on both sides: a real `openai` client (dev dep) pointed at `fake_provider.FakeProviderServer`, wrapped in a real `Solwyn`/`AsyncSolwyn` talking to the live API.

- **Detection doubles as coverage**: an ephemeral-port server is detected as the `openai_compatible` catch-all; `start_on_conventional_port()` (1234/11434/8000) exercises the lmstudio/ollama/vllm port heuristic. Tests skip if all three ports are occupied by real services.
- **`WireRecorder`** (conftest) records budget preflights (`.budget_checks`), `report_settlement` payloads (`.settlements`), and `report` payloads (`.events`) while delegating to the real implementations — wire assertions with live delivery preserved. EVERY reservation-backed success — non-streaming AND streaming — settles via `report_settlement` (confirm + event as one `.settlements` tuple); denials, reservation-less successes, and errors still go through `report` (`.events`). There is no `confirm_cost` seam.
- **Conventions**: wrapped clients are built with `budget_check_cache_ttl=0` (an allow-cache hit has no reservation_id → confirm silently skips) and happy paths use a model the API prices (`gpt-5.5`; unpriced models are allowed without a reservation).
- **Budget denial** is server-driven: `hard_denied_credentials` creates a `hard_deny` project ($0.05 limit) and burns it with one large confirm, so a wrapper check raises `BudgetExceededError` before any provider call.
- **Provider routes and seams**: `FakeProviderServer` serves chat completions and OpenAI 2.53-compatible `/v1/responses` JSON/SSE (including structured parse output and terminal flat usage). Its direct real-SDK probe locks down OpenAI's wire precedence: `extra_body` replaces same-named Responses arguments. `fail_next(status, count=1, *, retry_after=None)` queues real error responses (optionally with a `Retry-After` header); `set_omit_usage()` strips usage from JSON and stream responses on both routes (the SDK must fall back to a length-based estimate with `is_estimated=True`); `drop_next_stream(after_chunks=N)` truncates the next stream without the HTTP/1.1 chunked terminator, so the client sees `httpx.RemoteProtocolError`, not clean EOF. All seams reset between tests via the autouse fixture.
- **Native pin E2E**: `test_e2e_responses.py` passes `provider="openai"` to real sync/async OpenAI clients at the ephemeral fake URL. This intentionally bypasses generic-compatible detection and proves check → native Responses dispatch → one settlement for create, create streaming, parse, and the stream-manager helper. Provider usage settles exactly when present; omitted non-stream create/parse usage settles a marked request-length estimate instead of exact zero.
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
