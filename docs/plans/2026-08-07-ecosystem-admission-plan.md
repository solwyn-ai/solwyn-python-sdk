# Ecosystem Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per the owner's standing workflow preference: subagents execute only; the main session reviews diffs and re-runs tests itself.

**Goal:** Make `isinstance(Solwyn(OpenAI()), openai.OpenAI)` true so injectable-client framework paths stop rejecting the wrapper, then ship thin framework glue (OpenAI Agents SDK recipe, LangChain/LangGraph callback, CrewAI listener) on the run-hierarchy/tags event contract.

**Architecture:** The wrapper becomes a wrapt-style transparent proxy — a `__class__` property that reports the wrapped client's class, plus attribute-set forwarding and a `_solwyn_` internal namespace — with zero provider imports (the lie is computed from the caller's own object). Framework adapters are attribution-only glue over the existing run-scope/tags seams; enforcement always rides the wrapped client. Adapters live in `solwyn/integrations/` behind extras; core never imports a framework.

**Tech Stack:** Python ≥3.11 (pyproject.toml:5), Pydantic v2, httpx, pytest. Frameworks (langchain-core, crewai, openai-agents) appear only in `solwyn/integrations/` modules and optional test jobs.

## Global Constraints

Copied from CLAUDE.md and the project brief — every task implicitly includes these:

**Historical anchor provenance (2026-08-13):** Unless a later dated note
explicitly says otherwise, every raw `path:line` reference in this recovered
plan was verified against commit
`78a73744be86838551aa5c3078583658cea56a0b` (`Coverage strict mode 9/9: harden
surface-contract enforcement (#63)`), not the current checkout. Symbol and path
names are authoritative. Before implementing any open task, resolve each
symbol's current location with `rg`; do not transplant the historical line
number. Dated 2026-08-15 Responses milestone notes describe current behavior by
symbol/path and do not repin historical lines.

- NEVER capture, log, or transmit prompts or responses. Content-privileged allowlist = `_privacy.py` + `providers/_translation/` only, CI-enforced by `tests/unit/test_privacy_firewall.py`.
- Never import provider SDKs in core code; detection and identity stay duck-typed against the caller's own object.
- All business logic sans-I/O in `_base.py`; client classes are thin I/O wrappers.
- Runtime invariants use `raise RuntimeError(...)`, never `assert` (enforced by `tests/unit/test_no_production_asserts.py`).
- Pydantic v2 only; models use `extra="forbid"`.
- Provider adapter registry order is load-bearing (`providers/__init__.py:35-49`).
- Framework adapters stay thin and contract-driven; the event contract is ours, the glue is disposable.
- Pre-launch: breaking the current wrapper class is fine when the replacement is the right design.
- Quality gate per task: `make check` && `make test` green before commit.

---

## 0. Verified ground truth

Every inherited claim was re-checked against the repo. Two corrections and one discovery below.

| Claim | Verdict | Evidence |
|---|---|---|
| Wrapper is rejected by type-checking frameworks; no identity override exists | **Re-confirmed 2026-08-13** | `Solwyn.__getattr__` at `src/solwyn/client.py:2071` (async: `client.py:3148`) — no longer a raw `getattr` passthrough: it routes through the guarded-surface resolver `_resolve_public_attribute` (`_base.py:1071`) added by the coverage strict-mode track; still no `__class__`/`__instancecheck__`/`__setattr__` override anywhere in `src/` (grep) |
| "Wrapper does not override `__instancecheck__`" | **Confirmed but technically imprecise** | `__instancecheck__` lives on the metaclass of the class being *checked against* (`openai.OpenAI`'s metaclass) — overriding it would require importing/patching the provider. The duck-typed lever is a `__class__` property on the wrapper, which CPython's `isinstance` consults as a fallback (the wrapt/unittest.mock mechanism). |
| Zero framework integrations exist | **Re-confirmed 2026-08-13** | grep for langchain/crewai/opentelemetry/otel/llama_index/autogen across src+tests+pyproject: one comment hit (an OpenTelemetry span-semantics comment, `tests/unit/test_run_context_manager.py:115`), no code; `src/solwyn/integrations/`, `docs/integrations/`, `tests/integration/frameworks/`, and `frameworks-smoke.yml` all still absent |
| "14-provider compat table" | **Corrected: 15 profiles = 14 named + 1 generic catch-all** | `providers/openai_compatible.py:121-222` — xai, deepseek, mistral, qwen, zai, groq, together, fireworks, perplexity, azure_openai, openrouter, ollama, vllm, lmstudio + `openai_compatible` catch-all |
| Seven integration seams sit ready | **Re-confirmed 2026-08-13** (enumerated in §2) | run scope `_run.py:343`; hierarchy fields `_types.py:340-361`; three-layer tag merge `_run.py:167-176`; per-call `solwyn_tags` pop `client.py:1252` (+3 sibling sites: `1470`, `2407`, `2597`); fire-and-forget reporter `reporter.py:775,1318`; compat table above; guarded `__getattr__` passthrough `client.py:2071`; `run_in_executor` `_run.py:309-340` |
| Run hierarchy + inherited tags just landed (PR #50) | **Confirmed** | commit `3eddfc0 feat(sdk): add inherited spend tags and run hierarchy (#50)`; `parent_agent_run_id` on the wire at `_types.py:349-356` |
| strands-agents#1103: users demand injectable pre-configured clients | **Confirmed externally** | The issue asks for "a fixed, injected `AsyncOpenAI` client to enable alternative interface-compatible clients" — an isinstance-shaped ask, exactly what the enabler unlocks |
| OpenAI Agents SDK ships `set_default_openai_client` and no budget primitive | **Confirmed externally** | openai-agents-python config docs; also confirms `set_default_openai_api("chat_completions")` exists (needed below) |
| **Discovery: OpenAI Responses coverage is partial** | **Milestone update 2026-08-15** | Native OpenAI and Azure OpenAI `responses.create(...)`, `responses.parse(...)`, and the new-response `responses.stream(...)` context-manager helper are now metered for sync and async clients through one budget-preflight and settlement lifecycle; `create(stream=True)` is included. Existing-response stream retrieval is a reviewed raw path with no duplicate charge. Beta/raw-response helpers, other Responses leaves, and non-Azure compatible profiles remain guarded `unmetered_spend`. The Agents SDK chat-completions pin is therefore optional when its path stays within the admitted native/Azure trio, while teams requiring strict coverage of every Responses path may still pin it. |
| **Discovery: internal attribute collision** | **Re-confirmed 2026-08-13, load-bearing** | The wrapper stores the wrapped client as `self._client` (`client.py:918`, async `client.py:2122`); openai-python's own client stores its httpx client as `self._client`. Once frameworks accept the wrapper as a real client, external code reading `client._client` gets the wrong object. Fixed by the `_solwyn_` namespace (Task 1). |

---

## 1. Settled questions

### Q1. Mechanism: transparent proxy vs runtime subclass vs hybrid → **transparent proxy (wrapt-style `__class__` lie)**

**How each scores on the required axes:**

| Axis | Proxy (`__class__` property) | Runtime subclass `type("X", (Solwyn, type(client)), {})` |
|---|---|---|
| `isinstance(w, openai.OpenAI)` | ✅ True (CPython `isinstance` falls back to `obj.__class__`; the mock/wrapt-proven path; Pydantic v2 arbitrary-type validation also passes because it calls `isinstance`) | ✅ True |
| `issubclass(type(w), openai.OpenAI)` / `type(w) is openai.OpenAI` | ❌ still False — accepted; no target framework is known to check this way (strands #1103, Agents SDK, pydantic all isinstance-shaped) | ✅ / ❌ |
| `__init__` bypass | Not needed — wrapper never runs provider `__init__`; provider methods reached via `__getattr__` are **bound to the real client**, so `self` inside them is always the real object | Must construct via `object.__new__`; provider class methods then bind `self` = wrapper: instance-attr **reads** fall through `__getattr__` to the real client but **writes** land on the wrapper → silently divergent state, per provider-SDK version |
| Attribute shadowing | Contained: only wrapper instance attrs shadow; Task 1 moves them all into `_solwyn_*` | Uncontained: full provider MRO merges with wrapper MRO; modern openai-python defines resources (`chat`, …) as class-level cached_properties — MRO races decided per SDK release |
| Metaclass / `__slots__` hazards | None | Real across 15+ client classes incl. dynamically generated botocore classes |
| Pickling | `pickle` reads `obj.__class__` → would masquerade as the provider class; fixed with explicit `__reduce_ex__` raising a guidance TypeError (provider clients are unpicklable anyway — httpx state — so this is drop-in parity) | Dynamic classes can't pickle by reference at all; same override needed |
| `__class__` reporting | `w.__class__` = provider class (what frameworks check); `type(w)` = `Solwyn` (what debuggers/`repr` show) — the split is documented | `type(w)` = per-client-class synthetic type; class-cache bookkeeping needed |
| Existing `__getattr__` interception + registry | Unchanged. Detection uses `type(client).__module__` (`providers/openai.py:380`, `openai_compatible.py:297`) and `type()` is immune to the lie — a double-wrapped client still fails detection loudly (`providers/__init__.py:83`); Task 2 adds a clearer guard | `__module__` of the synthetic class would confuse detection; more moving parts |

**Decision:** transparent proxy. The runtime subclass buys only `issubclass`/exact-`type` admission — which nothing in evidence requires — at the cost of an uncontrolled state-split hazard that re-opens with every provider SDK release. Hybrid ("proxy now, subclass later if a framework demonstrably issubclass-checks") remains available; nothing in the proxy design forecloses it.

**TS portability (notes only, per constraints):** JS `Proxy` around the real client with default `getPrototypeOf` gives `instanceof` admission for free (instanceof walks the target's prototype chain); the `_solwyn_` name-partition rule maps to `Symbol`-keyed internals, which is cleaner than a string prefix. The event-contract surface (§2) is identical. No Python-only tricks are load-bearing.

### Q2. Blast radius: what assumes the wrapper's own class identity

Verified inventory — everything the identity change touches:

1. **`self._client` name collision** (`client.py:918`, async `client.py:2122`) — collides with openai-python's private `_client` (httpx). → Task 1 renames all wrapper instance attrs to `_solwyn_*`. Full list to rename (re-inventoried 2026-08-13, now **20** — the coverage track added 3): 15 set in `_SolwynBase.__init__` (`_base.py:674-753`): `_config`, `_runtimes`, `_surface_context` (new), `_guard_lock` (new), `_guarded_resources` (new), `_requested_failover_tuning`, `_failover_tuning_suppression_logged`, `_sdk_instance_id`, `_policy`, `_signal_lock`, `_latency_windows`, `_last_price_hints`, `_circuit_breakers`, `_breaker_lock`, `_control_plane_breaker`; plus 5 set in the `client.py` subclass inits (`Solwyn.__init__` `client.py:900-993`, async mirror ~`client.py:2100s`): `_client`, `_adapter`, `_dialect`, `_budget`, `_reporter`. Scope: `_SolwynBase`, `Solwyn`, `AsyncSolwyn` only (NOT `_MaterializedStream` etc., NOT enforcer/reporter internals — those objects are never presented to frameworks). Precedent: `_GuardedResource` already uses the `_solwyn_` prefix for its own internals (`_solwyn_owner`/`_solwyn_raw`/`_solwyn_path`, `_base.py:643-644`) — those stay as-is; no collision with the rename.
2. **`__setattr__` divergence** — today `w.api_key = "x"` lands on the wrapper while the real client keeps the old key; requests then use the old key while reads return the new one. → Task 2 forwards non-`_solwyn_` sets to the wrapped client. `functools.cached_property` (7 proxies — `chat`/`embeddings`/`images`/`audio`/`videos`/`messages`/`models` — at `client.py:995-1090`, async mirror `client.py:2193-2280`) writes straight into the instance `__dict__`, bypassing `__setattr__` — proxies stay local, verified by test. (Re-confirmed 2026-08-13: no `__setattr__` exists anywhere in `src/` yet.)
3. **Registry detection** — uses `type(client)`, immune to the lie (see Q1 table). Double-wrap currently dies with a generic `ValueError` (`providers/__init__.py:83`; re-confirmed, no "already wrapped" guard exists yet); Task 2 adds an explicit `ConfigurationError("already wrapped")` via a `type()`-based check. New since plan: construction also validates every runtime context against the declared surface table — `_validate_surface_context` (`_base.py:601-611`, invoked at `_base.py:692`) raises `ConfigurationError` on undeclared provider/client-shape/mode pairings; the double-wrap guard slots in before adapter detection as planned and is unaffected.
4. **AST-based tests key on literal class names** — `tests/unit/test_stream_nonblocking.py:27,55` parse `client.py` for `ClassDef` named `Solwyn`/`AsyncSolwyn`. Constraint: keep both class names and module path. The plan does.
5. **Context-manager/close semantics** — `close()` (`client.py:2060-2064`, async `client.py:3136-3140`) closes reporter+budget but NOT the wrapped client. Note: `self._budget.close()` now also surrenders PJ-2 leases (`budget.py:1814`), so the forwarded provider close must run AFTER the full Solwyn shutdown chain. Post-admission, frameworks that `with client:` or `client.close()` expect provider sockets released. → Task 2 forwards close to the wrapped client after Solwyn shutdown, duck-typed and fail-soft.
6. **Escape-hatch/Responses milestone update (2026-08-15)** — `w.with_options(...)` and `w.copy()` (openai/anthropic) still return unwrapped clients under the shared `unmetered_spend` posture. Native OpenAI and Azure OpenAI `w.responses.create(...)`, `.parse(...)`, and the new-response `.stream(...)` helper are now the exceptions: sync/async foreground calls, including `create(stream=True)`, are intercepted and settled. Existing-response stream retrieval stays raw with no duplicate charge. Beta/raw-response access, other Responses leaves, and non-Azure compatible profiles remain guarded `unmetered_spend`. Under admission, those residual leaves warn or raise instead of escaping silently. Task 6's LangChain smoke test will assert that the enforced path is actually exercised per invoke. If a target framework routes model calls through `with_options`, escalate to the shared-core rewrap design sketched in Task 3 notes (deliberately not built in v1 — YAGNI until evidence).
7. **Streaming return type** — streaming returns `SyncStreamWrapper`/`AsyncStreamWrapper` (`stream.py`), not the provider's `Stream` class. Frameworks consume streams by iteration (duck-typed) — Agents SDK chat-completions streaming does. Task 5 will document this limitation and cover it with the Agents SDK streaming smoke test; making stream wrappers type-transparent is out of scope v1.
8. **No production code isinstance-checks the wrapper** — grep found `isinstance(x, Solwyn…)` only in exception-hierarchy tests. Nothing internal breaks when `w.__class__` lies, because internals hold `self` (type-based dispatch is never used on the wrapper).

### Q3. Framework order → **Agents SDK recipe → LangChain/LangGraph → CrewAI**

1. **OpenAI Agents SDK (first; recipe-only).** Directly unlocked by the enabler (`set_default_openai_client(AsyncSolwyn(AsyncOpenAI()))`); the SDK deliberately ships no budget primitive (vacuum); zero package code to maintain against the fastest-churning target (dropped openai v1 support in 4 days). Run scoping maps: whole run → `solwyn.run()`; per-agent + handoffs → `RunHooks.on_agent_start/on_agent_end/on_handoff` via the Task 4 handles. At the 2026-08-15 milestone, the recipe treats `set_default_openai_api("chat_completions")` as optional for native OpenAI and Azure OpenAI paths limited to the admitted create/parse/stream trio; strict full-coverage users may still pin because beta/raw-response helpers and other Responses leaves remain unmetered (§0 discovery).
2. **LangChain/LangGraph (second).** Largest install base; direct AgentBudget parity target. Subscribes only to **structural** callbacks: `on_chain_start`/`on_chain_end`/`on_chain_error` (`run_id`, `parent_run_id`, serialized class name) → run-scope handles. Enforcement rides the injected transparent client (`ChatOpenAI(root_client=Solwyn(...))`-style), which the enabler admits past pydantic validation.
3. **CrewAI (third; attribution-first).** CrewAI's model calls go through LiteLLM, not an injectable openai client, so v1 is honest: a `BaseEventListener` mapping crew/task boundaries to run scopes (never the `LLMCall*` events — those carry prompts/completions), plus a documented enforcement recipe (custom `BaseLLM` routed through a Solwyn-wrapped client). Ships last because its enforcement story is the weakest until CrewAI grows client injection.

What each adapter consumes from our contract is specified in §2; none of them ever reads token usage from the framework or emits spend — the wrapped client is the only reporter, so double-counting is structurally impossible.

### Q4. Packaging → **in-repo `solwyn/integrations/` + extras; Agents SDK recipes-only**

- `solwyn/integrations/<framework>.py` subpackage in this repo. Rationale: pre-launch velocity, adapters version-locked to the event contract they ride, one CI. Separate PyPI packages reconsidered post-launch if framework release cadence forces independent versioning — the module boundary makes that split mechanical later.
- Extras: `solwyn[langchain]`, `solwyn[crewai]` (floors set to what the smoke matrix passes against at merge; starting candidates `langchain-core>=0.3`, `crewai>=0.100`). Framework imports happen only inside the adapter module with a guidance `ImportError`. The Agents SDK integration is docs + a tested example — no shipped code, no extra.
- **Version-churn testing without importing frameworks in core:** two-tier. (a) Tasks 6-7 will exercise adapters against *protocol doubles* — tiny in-repo stand-ins mirroring the exact callback signatures — so `make test` never imports a framework (mirrors the sanctioned test-only-provider-SDK pattern, pyproject.toml:27,31 comments). (b) Task 5 will add `tests/integration/frameworks/` smoke tests against real latest frameworks in a scheduled + manually-triggered CI job (`frameworks-smoke.yml`), not in the PR gate. A smoke break will mean the framework moved — fix the thin adapter, never the contract.

---

## 2. The event-contract surface adapters rely on

This is the frozen surface framework glue may touch. Anything not listed is off-limits to adapters.

| Seam | Where | Adapters use it for |
|---|---|---|
| `solwyn.run(name, tags=..., inherit_tags=...)` context manager | `_run.py:343` | Whole-run scoping in recipes |
| **NEW** `solwyn.start_run(...) -> RunHandle` / `handle.finish()` (Task 4) | `_run.py` | Begin/end-shaped framework callbacks (chain start/end, agent hooks, crew events) |
| `agent_run_id` / `parent_agent_run_id` / `agent_run_name` wire fields | `_types.py:340-361` | Hierarchy: nested handles get parented automatically (`_run.py:246-247`) |
| Three-layer tag merge: per-call `solwyn_tags` > scope tags > client `tags=` | `_run.py:167-176`; kwarg pop `client.py:1252` / `1470` (async `2407` / `2597`) | Framework metadata → customer tags (bounded by `TAGS_MAX_KEYS` clamp, `_run.py:173-184`) |
| Budget check carries `agent_run_id` + tags | `client.py:1280-1291`; `_types.py:447-458` | Run scoping is not just attribution: PJ-2 leases are per-run, so per-agent/per-task scopes buy per-agent lease granularity |
| Fire-and-forget reporter | `reporter.py:775` (`report`), `reporter.py:1318` (`report_settlement`) | Guarantee: adapters add zero latency and zero I/O; they never touch the reporter directly |
| Compat table (14 named + catch-all) | `openai_compatible.py:121-223` | Recipes work for any OpenAI-compatible base_url, not just OpenAI |
| Guarded attribute passthrough + `on_unmetered` posture | `client.py:2071` → `_resolve_public_attribute` `_base.py:1071`; posture `_apply_untracked_posture` `_base.py:942-969`, warn latch `_warn_contextual_surface_once` `_base.py:197-255`; knob `config.py:59` | Non-intercepted framework touches resolve through the surface contract; untracked spend surfaces warn (default), raise (strict), or pass (`allow`) |
| `run_in_executor` | `_run.py:309-340` | Docs for threaded frameworks (ContextVar propagation) |

Adapter obligations (enforced by the Task 6 firewall extension): structural fields only; never read content-bearing callback params; never log call arguments; never import httpx; never construct `MetadataEvent`.

---

## 3. Tasks

Each task is one PR. Steps are checkboxed; every task ends with `make check && make test` and a commit.

### Task 1: `_solwyn_` internal namespace (pure rename)

**Files:**
- Modify: `src/solwyn/client.py`, `src/solwyn/_base.py`, `src/solwyn/_proxies.py`, `src/solwyn/stream.py` (references), tests that reach into wrapper internals (mechanical)
- Create: `tests/unit/test_wrapper_namespace.py`

**Interfaces:**
- Produces: every instance attribute of `_SolwynBase`/`Solwyn`/`AsyncSolwyn` is named `_solwyn_*` (e.g. `self._solwyn_client`, `self._solwyn_reporter`). Later tasks rely on the prefix as the local/forwarded partition rule.

- [ ] **Step 1: Write the failing AST guard test**

```python
"""Namespace hygiene: wrapper instance attributes must never collide with a
wrapped provider client's own attribute namespace (e.g. openai-python's
private `_client`). Everything the wrapper stores on itself lives under
`_solwyn_*`; all other names belong to the wrapped client."""
import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "solwyn"
_WRAPPER_CLASSES = {"_SolwynBase": "_base.py", "Solwyn": "client.py", "AsyncSolwyn": "client.py"}


def _self_attribute_stores(class_node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(class_node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)
    return names


@pytest.mark.unit
def test_wrapper_instance_attributes_use_solwyn_namespace() -> None:
    offenders: list[str] = []
    for cls_name, filename in _WRAPPER_CLASSES.items():
        tree = ast.parse((_SRC / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                offenders += [
                    f"{cls_name}.{attr}"
                    for attr in _self_attribute_stores(node)
                    if not attr.startswith("_solwyn_")
                ]
    assert offenders == [], f"non-namespaced wrapper attrs: {sorted(offenders)}"
```

- [ ] **Step 2: Run it — expect FAIL** listing all 20 current attrs (`Solwyn._client`, `_SolwynBase._config`, `_SolwynBase._surface_context`, …): `pytest tests/unit/test_wrapper_namespace.py -v`
- [ ] **Step 3: Mechanical rename** across the three classes and every reference (src + tests). `grep -rn "self\._client\b\|\._reporter\b\|\._budget\b" src tests` style sweeps until clean. No behavior change; docstrings in `client.py` module map (`src/solwyn/CLAUDE.md`) updated to the new names.
- [ ] **Step 4: Full suite passes**: `make check && make test` (the existing 71-file unit suite is the real regression net here; the heavy surface-posture tests — `test_unmetered_posture.py`, `test_surfaces.py` — churn only where they construct wrappers, never in the contract digests, which pin rules, not attribute names)
- [ ] **Step 5: Commit** `refactor(sdk): move wrapper internals into the _solwyn_ namespace`

### Task 2: Type transparency core

**Files:**
- Modify: `src/solwyn/client.py` (both classes), `src/solwyn/_registry.py` (double-wrap guard), `src/solwyn/CLAUDE.md`, `README.md`
- Create: `tests/unit/test_type_transparency.py`

**Interfaces:**
- Produces: `Solwyn`/`AsyncSolwyn` gain `__class__` (property), `__setattr__`, `__delattr__`, `__dir__`, `__reduce_ex__`, `__copy__`, `__deepcopy__`, `__repr__`; `build_runtimes` raises `ConfigurationError(field="client")` on an already-wrapped client. `type(w)` remains `Solwyn` — internal code and debuggers unaffected.

- [ ] **Step 1: Write the failing edge-case matrix** (mirror the import-or-skip pattern of `tests/unit/test_real_sdk_detection.py`; provider SDKs are sanctioned test-only deps). Core cases:

```python
import copy
import pickle

import pytest
from pydantic import BaseModel, ConfigDict

from solwyn import AsyncSolwyn, Solwyn
from solwyn.exceptions import ConfigurationError

openai = pytest.importorskip("openai")
# construction is offline: build_runtimes does no I/O; budget checks happen per call


def _wrapped() -> tuple:
    inner = openai.OpenAI(api_key="test-key", base_url="http://localhost:1")
    return Solwyn(inner, api_key=VALID_API_KEY), inner  # VALID_API_KEY from conftest


@pytest.mark.unit
def test_isinstance_reports_wrapped_class_and_wrapper_class() -> None:
    w, _ = _wrapped()
    assert isinstance(w, openai.OpenAI)          # the admission fix
    assert isinstance(w, Solwyn)                 # internals unaffected
    assert type(w) is Solwyn                     # debugger truth unchanged


@pytest.mark.unit
def test_mock_framework_type_gate_admits_wrapper() -> None:
    def framework_entry(client: object) -> str:
        if not isinstance(client, openai.OpenAI):
            raise TypeError("expected an openai.OpenAI client")
        return "admitted"
    w, _ = _wrapped()
    assert framework_entry(w) == "admitted"


@pytest.mark.unit
def test_pydantic_arbitrary_type_field_admits_wrapper() -> None:
    class FrameworkConfig(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        client: openai.OpenAI
    w, _ = _wrapped()
    assert FrameworkConfig(client=w).client is w


@pytest.mark.unit
def test_pickle_fails_loud_with_guidance() -> None:
    w, _ = _wrapped()
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(w)


@pytest.mark.unit
def test_copy_and_deepcopy_return_the_same_shared_wrapper() -> None:
    w, _ = _wrapped()
    assert copy.copy(w) is w
    assert copy.deepcopy(w) is w


@pytest.mark.unit
def test_public_attribute_sets_forward_to_wrapped_client() -> None:
    w, inner = _wrapped()
    w.timeout = 5.0
    assert inner.timeout == 5.0
    assert "timeout" not in vars(w)


@pytest.mark.unit
def test_internal_attribute_sets_stay_local() -> None:
    w, inner = _wrapped()
    w._solwyn_probe = "x"
    assert vars(w)["_solwyn_probe"] == "x"
    assert not hasattr(inner, "_solwyn_probe")


@pytest.mark.unit
def test_wrapping_a_wrapper_fails_loud() -> None:
    w, _ = _wrapped()
    with pytest.raises(ConfigurationError, match="already wrapped"):
        Solwyn(w, api_key=VALID_API_KEY)


@pytest.mark.unit
def test_dir_unions_wrapper_and_wrapped_names() -> None:
    w, _ = _wrapped()
    listing = dir(w)
    assert "chat" in listing and "close" in listing        # wrapper surface
    assert "with_options" in listing                        # wrapped surface


@pytest.mark.unit
def test_interception_properties_still_win() -> None:
    from solwyn._proxies import _SyncChatProxy
    w, _ = _wrapped()
    assert isinstance(w.chat, _SyncChatProxy)               # cached_property beats passthrough


@pytest.mark.unit
def test_cached_property_dict_write_bypasses_setattr_forwarding() -> None:
    w, inner = _wrapped()
    _ = w.chat
    assert "chat" in vars(w) and not isinstance(inner.chat, type(w.chat))
```

Plus mirrors: `AsyncSolwyn` + `openai.AsyncOpenAI`; `importorskip` variants for `anthropic.Anthropic` and google `genai.Client`; a bedrock variant using the existing duck-typed fake from `tests/unit/test_bedrock_client.py`; a close-forwarding test (wrapped client's `close` called once after Solwyn shutdown, fail-soft if absent).

- [ ] **Step 2: Run — expect FAIL** on every new dunder: `pytest tests/unit/test_type_transparency.py -v`
- [ ] **Step 3: Implement on both classes** (identical bodies; module-level constant shared):

```python
_SOLWYN_INTERNAL_PREFIX = "_solwyn_"

    @property
    def __class__(self) -> type:  # type: ignore[override]
        """Report the wrapped client's class so isinstance-based admission
        (frameworks, pydantic arbitrary-type validation) accepts the wrapper.
        type(self) still names the wrapper — debuggers see the truth."""
        return type(self._solwyn_client)

    def __getattr__(self, name: str) -> Any:
        if name.startswith(_SOLWYN_INTERNAL_PREFIX):
            raise AttributeError(name)          # never forward internals; guards recursion
        # UNCHANGED posture path — this is the current body at client.py:2071.
        # The guarded-surface resolver owns warn/raise/allow; do NOT regress
        # this to a raw getattr. Only the attr name changes (Task 1 rename).
        return self._resolve_public_attribute(
            self._solwyn_client, name=name, path=name, source=SurfaceSource.RAW
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith(_SOLWYN_INTERNAL_PREFIX):
            object.__setattr__(self, name, value)
            return
        setattr(self._solwyn_client, name, value)   # keep wrapper and client in agreement

    def __delattr__(self, name: str) -> None:
        if name.startswith(_SOLWYN_INTERNAL_PREFIX):
            object.__delattr__(self, name)
            return
        delattr(self._solwyn_client, name)

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | set(dir(self._solwyn_client)))

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError(
            "Solwyn clients hold live reporter/budget state and cannot be pickled; "
            "construct a fresh Solwyn(...) in the target process"
        )

    def __copy__(self) -> Solwyn:
        return self        # a spend-enforcement handle must not fork budget identity

    def __deepcopy__(self, memo: dict[int, Any]) -> Solwyn:
        return self        # frameworks deep-copy configs; shared handle is the correct copy

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._solwyn_client!r})"
```

  Notes: `close()` additionally calls `getattr(self._solwyn_client, "close", None)` (or `aclose` on async) fail-soft after reporter/budget shutdown. Double-wrap guard in `build_runtimes` (`_registry.py:134`), before adapter detection, using the lie-immune check: `if issubclass(type(primary_client), _SolwynBase): raise ConfigurationError("client is already wrapped by Solwyn — pass the raw provider client", field="client")` — imported lazily or via a marker classattr to avoid a cycle (implementer's choice; `getattr(type(client), "_solwyn_is_wrapper_type", False)` with the classattr set on `_SolwynBase` avoids the import entirely).
- [ ] **Step 4: Matrix + full suite pass**: `pytest tests/unit/test_type_transparency.py -v && make check && make test`
- [ ] **Step 5: Docs** — README "drop-in" section gains the isinstance claim + the `type(w)`-vs-`__class__` note + pickle/copy semantics; `src/solwyn/CLAUDE.md` client bullet updated (snapshot style, no history).
- [ ] **Step 6: Commit** `feat(sdk): type-transparent wrapper — isinstance(Solwyn(c), type(c)) is true`

### Task 3: Make the escape hatches loud (`responses`, `with_options`, `copy`) — **SUPERSEDED: delivered by the coverage strict-mode track (#54–#63)**

**Validated 2026-08-13.** The surface contract did what this task planned, with a stronger mechanism than the old warn-once map (`_UNSHIPPED_SPEND_SURFACES` and `_warn_unmetered_spend_surface_once` no longer exist). The bullets below record that validation point; the 2026-08-15 Responses-metering milestones are summarized in the final bullet:

- `responses` is a guarded **namespace** rule (`surface.responses.namespace.1727b9072678`, `dispatch_action="guard"`): access returns a cached `_GuardedResource` (`_base.py:637-662`); every leaf under it (`responses.create`, `.parse`, `.stream`, `.input_items.list`, …, plus the `beta.responses.*` mirror) is `unmetered_spend`. The warn fires at the **leaf**, not on namespace access — this task's original test sketch (warn on the `w.responses` attribute read itself) does not match the shipped behavior and was dropped.
- `with_options` and `copy` are top-level `unmetered_spend` rules with `capability_scope="client"` and exact acknowledgment tokens (`surface.with-options.unmetered_spend.90e313d4ca6b`, `surface.copy.unmetered_spend.2e101db23559`), covering the openai/azure_openai/together and anthropic client shapes. The value they return is still the raw unwrapped client — the documented cooperative-guard boundary (`src/solwyn/CLAUDE.md`).
- Posture knob: `on_unmetered: Literal["warn", "raise", "allow"] = "warn"` (`config.py:59`). Warn mode logs once per `(provider, client_shape, mode, rule_id, drift)` key via `_warn_contextual_surface_once` (`_base.py:197-255`, bounded at 512 keys; test reset seam kept the old name `_reset_unmetered_spend_warnings()`, now `_base.py:189-194`); `raise` refuses with `UntrackedSpendSurfaceError` — an `AttributeError` subclass so `hasattr`/`getattr(..., default)` feature-probes keep working (`exceptions.py:252`); `allow` passes silently. Behavior pinned by `tests/unit/test_unmetered_posture.py`.
- At the 2026-08-13 validation point, the Responses API remained unmetered and Task 5's `set_default_openai_api("chat_completions")` pin was required.
- Any future rule change here (e.g. metering Responses) goes through the surface-rule workflow in `src/solwyn/CLAUDE.md` (export → edit JSON → embed → re-pin the digests in `tests/unit/test_surface_context_pins.py`) — never hand-edit `_surfaces.py`.
- **Milestone update 2026-08-15:** native OpenAI and Azure OpenAI `responses.create`, `responses.parse`, and the new-response `responses.stream` helper now use metered source-`BOTH` rules; `create(stream=True)` is included. Existing-response stream retrieval remains raw with no duplicate charge. Beta/raw-response helpers, other leaves, and non-Azure compatible profiles remain guarded `unmetered_spend`, so the Agents SDK pin is optional for paths limited to the admitted trio and remains available for teams requiring strict full coverage.

- [x] No implementation work remains; this task closes with no PR.

**Deferred contingency (not v1):** if a target framework routes calls through `with_options` (the Task 6 smoke test will prove it by asserting one budget check per framework invoke), implement shared-core rewrap: `with_options`/`copy` overrides on the wrapper that wrap the returned SDK client in a companion view sharing `_solwyn_budget`/`_solwyn_reporter`/breakers (no second reporter thread, no second lease holder identity), with `close()` a no-op on shared resources. Design sketch recorded here so the escalation is a bounded task, not a redesign.

### Task 4: `RunHandle` — begin/end-shaped run scoping for adapters

**Files:**
- Modify: `src/solwyn/_run.py`, `src/solwyn/__init__.py` (+ `__all__`), `tests/unit/test_public_exports.py`
- Create: `tests/unit/test_run_handle.py`

**Interfaces:**
- Consumes: `_RunScope._enter`/`_exit` (class `_run.py:205`; `_enter` `_run.py:238`, `_exit` `_run.py:267`) — the machinery already is begin/end-shaped; this exposes it safely. (Re-confirmed 2026-08-13: `start_run`/`RunHandle` do not exist yet anywhere in `_run.py` or `__init__.py`.)
- Produces: `solwyn.start_run(name, tags=None, *, inherit_tags=True) -> RunHandle`; `RunHandle.run_id: str`; `RunHandle.finish() -> None`. Tasks 5-7 call exactly this.

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.unit
def test_start_run_scopes_events_until_finish() -> None:
    handle = solwyn.start_run("adapter-scope", tags={"team": "x"})
    ctx = solwyn.current_run_context()
    assert ctx.id == handle.run_id and ctx.tags == {"team": "x"}
    handle.finish()
    assert solwyn.current_run_context().id is None


@pytest.mark.unit
def test_nested_handles_thread_parent_run_id() -> None:
    outer = solwyn.start_run("outer")
    inner = solwyn.start_run("inner")
    # _capture_run_context is the event-build seam; parent must be the outer id
    assert _capture_run_context()[3] == outer.run_id
    inner.finish(); outer.finish()


@pytest.mark.unit
def test_double_finish_raises_runtime_error() -> None:
    handle = solwyn.start_run("once")
    handle.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        handle.finish()


@pytest.mark.unit
def test_non_lifo_finish_raises_runtime_error() -> None:
    outer = solwyn.start_run("outer"); inner = solwyn.start_run("inner")
    with pytest.raises(RuntimeError, match="LIFO"):
        outer.finish()
    inner.finish(); outer.finish()
```

  Plus an asyncio test: handle started inside a task stays in that task's context (mirror `test_run_context_manager.py` patterns).
- [ ] **Step 2: Run — expect FAIL** (`start_run` undefined)
- [ ] **Step 3: Implement** in `_run.py` (sans-I/O, no locks — same contract as `_RunScope`):

```python
class RunHandle:
    """Begin/end-shaped run scope for framework adapters whose boundaries
    arrive as separate callbacks (chain start/end, agent hooks, crew events).

    Same semantics as ``with solwyn.run(...):`` — ContextVar-backed, LIFO per
    context, nested handles are parented automatically. ``finish()`` must run
    in the same context that called ``start_run`` (see module docstring for
    thread/async-generator caveats).
    """

    def __init__(self, scope: _RunScope, run_id: str) -> None:
        self._scope = scope
        self._run_id = run_id
        self._finished = False

    @property
    def run_id(self) -> str:
        return self._run_id

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError(f"run handle {self._run_id!r} already finished")
        self._scope._exit()            # raises RuntimeError on non-LIFO exit (unchanged rule)
        self._finished = True


def start_run(
    name: str,
    tags: Mapping[str, str] | None = None,
    *,
    inherit_tags: bool = True,
) -> RunHandle:
    """Open a run scope without a with-block. The adapter-facing seam."""
    scope = _RunScope(name, tags, inherit_tags=inherit_tags)
    return RunHandle(scope, scope._enter())
```

  Export `RunHandle`, `start_run` from `solwyn/__init__.py`; extend `test_public_exports.py`.
- [ ] **Step 4: Pass + full suite**; **Step 5: Commit** `feat(sdk): RunHandle/start_run — begin-end run scoping for framework adapters`

### Task 5: OpenAI Agents SDK recipe (docs + tested example)

**Files:**
- Create: `docs/integrations/openai-agents.md`, `tests/integration/frameworks/__init__.py`, `tests/integration/frameworks/test_openai_agents_smoke.py`, `.github/workflows/frameworks-smoke.yml`
- Modify: `pyproject.toml` (a `frameworks` dependency-group for the smoke job only), `Makefile` (`test-frameworks` target), `README.md` (integrations pointer)

**Interfaces:**
- Consumes: Task 2 admission; native OpenAI and Azure OpenAI Responses create/parse/stream metering plus the residual guarded `unmetered_spend` posture for beta/raw-response helpers, other leaves, and non-Azure compatible profiles (Task 3, superseded); Task 4 `start_run`.
- Produces: the documented recipe below; the smoke-CI harness that Tasks 6-7 reuse.

- [ ] **Step 1: Write the recipe** (`docs/integrations/openai-agents.md`) — full worked example, this is the shipped artifact:

```python
from agents import Agent, RunHooks, Runner, set_default_openai_api, set_default_openai_client
from openai import AsyncOpenAI

import solwyn
from solwyn import AsyncSolwyn

client = AsyncSolwyn(AsyncOpenAI(), api_key="sk_proj_...")
set_default_openai_client(client)
# OPTIONAL at the 2026-08-15 milestone: native OpenAI and Azure OpenAI
# responses.create, responses.parse, and the new-response responses.stream()
# helper are metered, including create(stream=True). Uncomment the conservative
# pin when your Agents SDK path may use beta/raw-response or other Responses
# leaves and you require strict full coverage:
# set_default_openai_api("chat_completions")


class SolwynRunHooks(RunHooks):
    """Per-agent run scoping: each agent activation is a child solwyn run,
    so budgets, leases, and the dashboard see the agent hierarchy."""

    def __init__(self) -> None:
        self._handles: dict[str, solwyn.RunHandle] = {}

    async def on_agent_start(self, context, agent) -> None:
        self._handles[agent.name] = solwyn.start_run(f"agent:{agent.name}")

    async def on_agent_end(self, context, agent, output) -> None:
        handle = self._handles.pop(agent.name, None)
        if handle is not None:
            handle.finish()

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        # handoffs appear as sibling child-runs under the enclosing run scope;
        # tag the receiving agent's run at start instead of scoping here
        pass


async def main() -> None:
    with solwyn.run("triage-workflow", tags={"team": "support"}):
        result = await Runner.run(triage_agent, "...", hooks=SolwynRunHooks())
```

  **Milestone guidance (2026-08-15):** `set_default_openai_api("chat_completions")` is optional for native OpenAI and Azure OpenAI clients whose Agents SDK path stays within `responses.create`, `responses.parse`, or the new-response `responses.stream()` helper; `create(stream=True)` is included. It remains the conservative choice when strict coverage must include beta/raw-response access or other leaves. Non-Azure compatible Responses paths are still guarded and unmetered.

  Also documents: budget-denied behavior inside a run (`BudgetExceededError` surfaces from the model call), fallback config, and the streaming caveat (§1 Q2 item 7).
- [ ] **Step 2: Write the smoke test** — respx-stubbed OpenAI endpoint + stubbed control plane (reuse `conftest` doubles); it will assert: (a) `set_default_openai_client` accepts the wrapper (no type rejection), (b) exactly one budget check per model call, (c) settled events carry `agent_run_id` with the expected `agent:` names and `parent_agent_run_id` = the workflow run. Marked `@pytest.mark.framework_smoke`, excluded from `make test`.
- [ ] **Step 3: CI job** `frameworks-smoke.yml`: weekly cron + `workflow_dispatch`; installs `.[dev]` + the `frameworks` group at latest; runs `make test-frameworks`.
- [ ] **Step 4: Run the smoke locally** (`make test-frameworks`) — expect PASS; `make check && make test` still green (proves core untouched by framework deps).
- [ ] **Step 5: Commit** `docs(integrations): OpenAI Agents SDK recipe + frameworks smoke harness`

### Task 6: LangChain/LangGraph adapter

**Files:**
- Create: `src/solwyn/integrations/__init__.py` (empty, doc-comment only), `src/solwyn/integrations/langchain.py`, `tests/unit/test_integrations_langchain.py` (protocol doubles), `tests/integration/frameworks/test_langchain_smoke.py`, `docs/integrations/langchain.md`
- Modify: `pyproject.toml` (`langchain = ["langchain-core>=0.3"]` extra), `tests/unit/test_privacy_firewall.py` (integrations rules)

**Interfaces:**
- Consumes: Task 4 `start_run`/`RunHandle`.
- Produces: `SolwynRunScopeHandler(BaseCallbackHandler)` — attribution-only; enforcement rides the injected transparent client (documented in the same page).

- [ ] **Step 1: Extend the privacy firewall first (failing)** — new rules for `src/solwyn/integrations/**`: (a) may not **load** the content-bearing callback parameters (`inputs`, `outputs`, `prompts`, `messages`, `generations`, `response`, `chunk`, `text`, `content`) — signature binds them, body never reads them (AST: no `ast.Name` load of those ids); (b) no logging call may take an argument beyond literal strings + structural ids; (c) no `httpx` import; (d) no `MetadataEvent` construction. Add a fixture module violating each rule to prove the test bites.
- [ ] **Step 2: Protocol-double unit tests (failing)** — an in-repo `_FakeCallbackManager` that invokes the handler exactly the way langchain-core does (`on_chain_start(serialized, inputs, *, run_id, parent_run_id=None, **kw)`), asserting: scopes open/close around chain boundaries; nesting maps `parent_run_id` → solwyn parent; `on_chain_error` closes the scope; out-of-order end logs a structural warning instead of raising; handler never touches `inputs`/`outputs` (firewall covers statically; double also asserts no attribute access via a sentinel object that raises on any attribute read).
- [ ] **Step 3: Implement** `src/solwyn/integrations/langchain.py`:

```python
"""LangChain/LangGraph run-scope attribution. Attribution ONLY — budget
enforcement rides the Solwyn-wrapped client injected into the model class.
This module never reads prompts, messages, or outputs (CI-enforced)."""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "solwyn.integrations.langchain requires langchain-core; "
        "install with: pip install 'solwyn[langchain]'"
    ) from exc

import solwyn

logger = logging.getLogger(__name__)


class SolwynRunScopeHandler(BaseCallbackHandler):
    """Maps chain/graph-node boundaries to solwyn run scopes."""

    run_inline = True  # sync handler on async runs must not be thread-offloaded
                       # (run scopes are ContextVar-backed)

    def __init__(self, *, name_prefix: str = "langchain", tags: dict[str, str] | None = None):
        self._prefix = name_prefix
        self._tags = tags
        self._handles: dict[UUID, solwyn.RunHandle] = {}

    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: UUID,
                       parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        name = "chain"
        if isinstance(serialized, dict):
            raw = serialized.get("name")
            if isinstance(raw, str) and raw:
                name = raw
        self._handles[run_id] = solwyn.start_run(f"{self._prefix}:{name}", tags=self._tags)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id)

    def _finish(self, run_id: UUID) -> None:
        handle = self._handles.pop(run_id, None)
        if handle is None:
            return
        try:
            handle.finish()
        except RuntimeError:
            # Framework delivered callbacks out of LIFO order (parallel
            # branches sharing one context). Structural ids only.
            logger.warning("solwyn.integrations.langchain: non-LIFO scope close for %s", run_id)
```

- [ ] **Step 4: Unit tests + firewall pass**; `make check && make test` (core still framework-free — verify with `python -c "import solwyn"` in an env without langchain).
- [ ] **Step 5: Smoke test** (`test_langchain_smoke.py`, `framework_smoke` marker): real langchain + `ChatOpenAI` with the Solwyn-wrapped client injected (`root_client=`/`client=` fields — pydantic admission is the Task 2 win) + the handler; respx-stubbed endpoints; it will assert one budget check per invoke (this is the with_options-escape detector from Task 3's contingency) and events scoped to the chain's run. Cover: sync chain, async chain, two-node LangGraph graph.
- [ ] **Step 6: Docs** `docs/integrations/langchain.md` — enforcement (inject wrapped client) + attribution (handler) shown together; threading caveats (`run_in_executor`, `_run.py:309-340`).
- [ ] **Step 7: Commit** `feat(integrations): LangChain/LangGraph run-scope callback + solwyn[langchain] extra`

### Task 7: CrewAI listener

**Files:**
- Create: `src/solwyn/integrations/crewai.py`, `tests/unit/test_integrations_crewai.py`, `tests/integration/frameworks/test_crewai_smoke.py`, `docs/integrations/crewai.md`
- Modify: `pyproject.toml` (`crewai = ["crewai>=0.100"]` — pin the floor to what smoke passes at merge)

**Interfaces:**
- Consumes: Task 4 handles; Task 6 firewall rules (apply automatically by path).
- Produces: `SolwynEventListener(BaseEventListener)` mapping crew/task boundaries to run scopes. **Never** subscribes `LLMCallStartedEvent`/`LLMCallCompletedEvent`/tool events — CrewAI documents that those carry prompts and completions, which the firewall forbids us to receive.

- [ ] **Step 1: Protocol-double unit tests (failing)** — fake event bus with `.on(EventType)` decorator registration; assert: crew kickoff start/complete opens/closes an outer run; task started/completed opens/closes child runs (parented); failure events close scopes; listener registers zero content-bearing event types (introspect the fake bus's registry against a denylist).
- [ ] **Step 2: Implement** — same shape as Task 6 (import-guarded `crewai.utilities.events` names; handles keyed by event source ids; structural names like `crew:{crew_name}` / `task:{task_index}`; non-LIFO closes log-warn).
- [ ] **Step 3: Firewall + unit + `make check && make test` pass.**
- [ ] **Step 4: Smoke test** — minimal one-agent crew against stubbed endpoints; it will assert attribution and the enforcement caveat honestly: without the custom-LLM recipe the model call is NOT budget-checked. The smoke will pin that behavior as attribution-only so a future CrewAI client-injection feature flips the test loudly.
- [ ] **Step 5: Docs** `docs/integrations/crewai.md` — listener setup + the enforcement recipe (CrewAI custom `BaseLLM` routed through a Solwyn-wrapped OpenAI-compatible client), clearly labeled attribution-first.
- [ ] **Step 6: Commit** `feat(integrations): CrewAI run-scope event listener + solwyn[crewai] extra`

### Task 8: Docs closure

**Files:**
- Create: `docs/integrations/README.md` (framework matrix: what each integration gives — enforcement / attribution / hierarchy — and its churn posture)
- Modify: `README.md` (admission story: the isinstance fix + one-liner per framework), `CHANGELOG.md` (unreleased: enabler + integrations, incl. the `_solwyn_` rename and new warn-once surfaces as breaking-ish notes), `CLAUDE.md` (project instructions gain one snapshot line: integrations are attribution-only, content-free, firewall-enforced)

- [ ] **Step 1: Write matrix + README + changelog** (changelog carries the narrative; CLAUDE.md stays a snapshot — standing rule)
- [ ] **Step 2: `make check && make test`**; **Step 3: Commit** `docs: ecosystem admission — integration matrix and changelog`

---

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `type(w)` vs `w.__class__` split confuses debugging | `__repr__` names both; README section; `type(w) is Solwyn` documented as the truth channel |
| A framework mutates client state via attribute sets | `__setattr__` forwarding keeps wrapper and client in agreement (Task 2 tests) |
| Frameworks deep-copy configs holding the client (CrewAI does) | `__copy__`/`__deepcopy__` return the shared wrapper — budget identity must not fork; memo'd in §6 |
| `with_options`/`copy`/residual Responses leaves bypass budgets under injection | **Milestone update 2026-08-15:** native OpenAI and Azure OpenAI create/parse/new-response-stream are metered, including create-stream; existing-response retrieval is raw without duplicate charging. Beta/raw-response helpers, other leaves, and non-Azure compatible profiles stay under the guarded unmetered posture. Teams needing strict full coverage may still pin Agents SDK to chat completions; Task 6 smoke will be the live escape detector and the rewrap contingency is pre-designed. |
| Framework churn breaks adapters | Adapters will remain attribution-only over the §2 contract; Tasks 6-7 protocol-double unit tests will keep `make test` framework-free; Task 5 scheduled smoke CI will catch drift without blocking PRs; Agents SDK gets no shipped code at all |
| Content leaks through framework callbacks (CrewAI LLM events, LangChain inputs/outputs) | Adapters subscribe to structural events only; firewall extension statically forbids loading content params, logging args, and event construction (Task 6 Step 1) |
| ContextVar scoping misfires in exotic execution (threaded callbacks, parallel branches) | `run_inline = True`; non-LIFO closes degrade to a structural warning, never an exception into user pipelines; limitation matrix in docs; Task 6 smoke will cover sync/async/LangGraph |
| Double-wrap or wrapping a wrapper masquerading as a provider client | Task 2 will add a lie-immune `type()`-based guard that raises `ConfigurationError` |
| Pickle-based frameworks (multiprocessing spawn) reject the wrapper | Loud `TypeError` with guidance — parity with the underlying clients, which are unpicklable anyway |

## 5. Out of scope

- TypeScript SDK implementation (portability notes in §1 Q1 only; TS port decisions live in the solwyn-ts-sdk track).
- ~~Responses API interception beyond native `create`/`create(stream=True)` — including parse, the stream helper, beta/raw-response, other leaves, Azure, and compatible profiles.~~ **Superseded 2026-08-15:** native OpenAI and Azure OpenAI create/parse/new-response-stream are now metered, including create-stream. Beta/raw-response helpers, other leaves, and non-Azure compatible profiles remain out of scope.
- `with_options`/`copy` shared-core rewrap (pre-designed contingency, built only on smoke evidence).
- Type transparency for sub-objects (`chat.completions`, stream wrappers) — frameworks duck-type these.
- OTel exporter; LlamaIndex / AutoGen / Pydantic AI / Strands adapters (recipe backlog once #1103-style injection lands there).
- Any Solwyn Cloud API change — this project is SDK + docs only.

## 6. Judgment calls made without asking (flagged for review)

1. **`copy`/`deepcopy` will return the same shared wrapper** rather than raising. A spend-enforcement handle must not fork lease/reporter identity, and deep-copying frameworks (CrewAI) would otherwise be re-rejected at admission — the failure this project exists to fix. Alternative (fail loud) preserved in git if you disagree.
2. **`with_options`/`copy` and residual Responses leaves stay passthrough under the guarded posture in v1** instead of building interception/rewrap now. Native OpenAI and Azure OpenAI `responses.create`/`parse`/new-response `stream` are the metered exceptions as of 2026-08-15, including `create(stream=True)`; beta/raw-response helpers, other leaves, and non-Azure compatible profiles retain the warn/raise/allow posture. The planned smoke tests will make any real-world escape visible immediately; the rewrap design is written down and bounded.
3. **`close()` will forward to the wrapped client** (fail-soft) once Task 2 ships — drop-in semantics once frameworks own the client's lifecycle.
4. **Adapters will be attribution-only** — they will never read framework token usage or emit spend; the wrapped client will be the single reporting path (prevents double-count by construction).

---

## Verification Summary

**Validated 2026-08-13 against `main` @ `78a7374` ("Coverage strict mode 9/9", #63), after the coverage strict-mode track (#54–#63) landed.** ~45 verifiable claims (line refs, names, counts, behaviors) checked by three parallel codebase sweeps; every stale value was corrected in place above.

**Confirmed unchanged (the plan's premises hold):**
- No `__class__`/`__setattr__`/`__instancecheck__` override anywhere in `src/` — Task 2 is still needed and still unblocked.
- Zero framework integration code; `src/solwyn/integrations/`, `docs/integrations/`, `tests/integration/frameworks/`, `frameworks-smoke.yml`, and a `test-frameworks` Make target all absent — Tasks 5–8 still open.
- `start_run`/`RunHandle` absent — Task 4 still open. No double-wrap guard exists (`ValueError` at `providers/__init__.py:83`) — Task 2's guard still needed. `build_runtimes` at `_registry.py:134`.
- Registry order block `providers/__init__.py:35-49`; 15 compat profiles (14 named + catch-all); `reporter.py:775` / `reporter.py:1318`; `_run.py:238` `_enter`, `246-247` parenting, `309` `run_in_executor`, `343` `run()`; AST class-name pins at `test_stream_nonblocking.py:27,55`; `_SyncChatProxy`; `SyncStreamWrapper`/`AsyncStreamWrapper`; `_MaterializedStream`; commit `3eddfc0` (#50); `requires-python >= 3.11` (`pyproject.toml:5`); privacy-firewall, no-asserts, real-SDK-detection, public-exports, and bedrock-fake test files all present as described.

**Corrected in place (mostly line drift from the coverage track):**
- `__getattr__` moved to `client.py:2071` / `client.py:3148` and now routes through `_resolve_public_attribute` (`_base.py:1071`) — Task 2's sketch updated to preserve it.
- `self._client` at `client.py:918` / `client.py:2122`; wrapper attr inventory grew 17 → **20** (`_surface_context`, `_guard_lock`, `_guarded_resources`); base init moved to `_base.py:674-753`, subclass attrs to `client.py:900-993`.
- Proxy cached_properties (7) at `client.py:995-1090` / `2193-2280`; `close()` at `client.py:2060-2064` / `3136-3140` (budget close now also surrenders PJ-2 leases); `solwyn_tags` pops at `client.py:1252/1470/2407/2597`; budget check at `client.py:1280-1291`, `_types.py:447-458`; OpenAI detection at `providers/openai.py:380`; compat table `openai_compatible.py:121-223`; tag merge `_run.py:167-176`; `_RunScope` at `_run.py:205`, `_exit` at `267`; unit suite is **71** files (was cited as 60); pyproject SDK comments at lines 27 and 31; the lone grep hit is an OpenTelemetry comment.

**Superseded (1 task):** Task 3 in full — the coverage strict-mode track replaced the planned warn-once map with the surface contract: `responses` = guarded namespace whose leaves follow explicit rules; `with_options`/`copy` = scoped client escapes with acknowledgment tokens; `on_unmetered` warn/raise/allow (`config.py:59`); strict error `UntrackedSpendSurfaceError` (`exceptions.py:252`). §0's Responses row, §1 Q2 item 6, the §4 risk row, and judgment call 2 were updated to match. **Milestone update 2026-08-15:** native OpenAI and Azure OpenAI `responses.create`, `responses.parse`, and the new-response `responses.stream` helper are metered, including `create(stream=True)`; existing-response retrieval remains raw without duplicate charging. Beta/raw-response helpers, other leaves, and non-Azure compatible profiles remain guarded `unmetered_spend`. The Agents SDK chat-completions pin is optional for admitted native/Azure paths and remains appropriate for strict full-coverage deployments.

**New facts the remaining tasks must respect:**
- Construction now validates every runtime context against the declared surface table (`_validate_surface_context`, `_base.py:601-611`) — the Task 2 double-wrap guard is unaffected but sits alongside it.
- `_GuardedResource` already uses the `_solwyn_` prefix internally (`_base.py:643-644`) — precedent for Task 1, and those attrs are excluded from the rename.
- Attribute renames churn surface-posture tests only where they construct wrappers; the context digests in `test_surface_context_pins.py` pin rules, not attribute names, and do not move.

**Unverifiable:** none.
