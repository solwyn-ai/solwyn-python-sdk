# Solwyn Python SDK

Drop-in wrapper for `openai`, `anthropic`, `google.generativeai`, and boto3 `bedrock-runtime` clients. Extracts token details, enforces budgets, handles failover — never computes cost (the API owns pricing).

## Commands

```bash
make check                                # full quality gate (lint + format + typecheck)
make test                                 # unit tests (~1.5s)
make test-integration                     # integration tests (needs API at localhost:8080)
make install                              # install in dev mode
make install-hooks                        # install pre-commit hook
```

## Architecture

```
Solwyn SDK (this repo)                   Solwyn Cloud API
┌──────────────────────┐                ┌─────────────────┐
│  your code           │  token counts  │  PricingService  │
│    ↓                 │  (no cost)     │  Budget state    │
│  Solwyn(OpenAI())    │ ──────────────>│  Cost dashboard  │
│    ↓                 │                └─────────────────┘
│  LLM provider <──────│── direct call
└──────────────────────┘

_SolwynBase          # Shared sans-I/O logic (config, token estimation, metadata)
  ├── Solwyn         # Sync: httpx.Client
  └── AsyncSolwyn    # Async: httpx.AsyncClient
```

## Rules

- NEVER capture, log, or transmit prompts or responses
- All business logic in `_base.py` (sans-I/O); client classes are thin I/O wrappers
- httpx for HTTP (already a transitive dep of openai/anthropic SDKs)
- Never import provider SDKs in core code — detection is duck-typed (the Bedrock adapter never imports boto3; `solwyn[bedrock]` is a convenience extra only)
- tiktoken is optional — always provide heuristic fallback
- Runtime invariants use `raise RuntimeError(...)`, not `assert` — Python's `-O` strips asserts. Enforced by `tests/unit/test_no_production_asserts.py`.
- Pydantic v2 only — `ConfigDict(...)`, `@model_validator`, `.model_dump()`. No v1 patterns.

## Key Conventions

- Pydantic models use `extra="forbid"` — catches typos and contract drift
- Response models (e.g. `BudgetCheckResponse`) use `Field(...)` for all fields the API returns — no silent defaults that mask contract changes
- Provider adapter registry lazy-loads concrete adapters on first call
- `check_budget(provider=...)` is required and keyword-only
- Consecutive confirm_cost failures are tracked — after 10, logs at ERROR level

## Privacy

`_privacy.py` and every module under `providers/_translation/` are the ONLY modules that touch customer prompt content (the content-privileged allowlist). Never log, store, or concatenate prompt text outside these modules. CI-enforced by `tests/unit/test_privacy_firewall.py` (path-based: the allowlist covers `_privacy.py` plus the whole `providers/_translation/` package).
