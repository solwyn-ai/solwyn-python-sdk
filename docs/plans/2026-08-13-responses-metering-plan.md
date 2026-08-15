# OpenAI Responses API First-Class Metering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Meter the Responses API spend surfaces — `responses.create()` (non-streaming and `stream=True`), `responses.parse()`, and the `responses.stream()` helper — on native OpenAI and Azure OpenAI, with budget pre-flight, interception, and settlement via `reporter.report_settlement`; plus an explicit `provider=` pin that unlocks E2E coverage of the native path.

**Architecture:** A capability-conditional `responses` wrapper proxy (the `messages` cached_property precedent; native OpenAI in PR B, + Azure via a `CompatProfile` flag in PR C) routes `create()` — then `parse()` and the `stream()` helper — into the existing `_intercepted_call` pipeline under a new internal `_surface="responses"` branch: Responses-shaped estimation, `max_output_tokens` lease bound, primary-runtime-only candidate walk (no failover; filtered defaults merge), and a duck-typed `prepare_responses_call` adapter seam. Settlement, breakers, leases, and the stream wrapper are reused unchanged; the one accumulator learns to read usage off the terminal `response.completed` event. The surface contract flips `responses.create` to `metered` for native OpenAI via the export → edit JSON → embed → re-pin workflow.

**Tech Stack:** Python ≥3.11 (pyproject.toml:5 — `requires-python = ">=3.11"`), Pydantic v2, httpx, pytest (+respx where needed), uv.

**Spec:** Inlined below (§Scope). Commissioned from a direct prompt; no separate spec doc exists. Related context: `docs/plans/2026-08-07-ecosystem-admission-plan.md` (Task 3 superseded note, Task 5 recipe).

## Scope

**In (v1):**
- **PR A + PR B (core):** native OpenAI `responses.create(...)` non-streaming AND `responses.create(stream=True)`.
- **PR C (coverage completion):** `responses.parse(...)` (Task 6); the `responses.stream()` context-manager helper (Task 7); Azure OpenAI parity for all three metered leaves via a `CompatProfile` capability flag (Task 8); explicit `provider=` adapter pin + E2E coverage of the native path (Task 9).
- Every other compat profile (groq, together, etc.) keeps today's unmetered_spend warn posture — raw pass-through behavior byte-for-byte unchanged.
- Sync (`Solwyn`) and async (`AsyncSolwyn`) parity throughout.

**Out (v1, all stay unmetered_spend guarded):** `.retrieve/.delete/.cancel/.compact`, `input_items.*`, `input_tokens.count`, `with_raw_response.*`, `with_streaming_response.*`, `beta.responses.*`, cross-dialect failover, background/batch. `background=True` is REFUSED loud (see Design Decisions).

**No wire-contract changes:** the budget check sends `modality="text"` (Responses is token-billed chat-family; the API prices by model), settlement rides the existing confirm + metadata event shapes. No Cloud API work is required.

## Global Constraints

- All business logic sans-I/O in `_base.py`; client classes are thin I/O wrappers.
- Never import provider SDKs in core code — detection and dispatch stay duck-typed.
- Prompt content touched ONLY in `_privacy.py` + `providers/_translation/` (CI: `tests/unit/test_privacy_firewall.py`). The new estimator lives in `_privacy.py`. Length-only, never join/log/store content.
- Runtime invariants use `raise RuntimeError(...)`, never `assert` (CI: `test_no_production_asserts.py`).
- Pydantic v2 only.
- NEVER hand-edit the payload block in `src/solwyn/_surfaces.py`. Surface-rule changes go through the loop in `src/solwyn/CLAUDE.md` §"Changing a surface rule" (export → edit `build/surface_contract/surface-classification.json` → embed → refresh export → update pins) and the PR body must include `uv run python scripts/diff_surface_rules.py <PR-base-ref>` output.
- Every task ends green: `make check && make test-unit` (the contract-flip task additionally runs `make check-surface-contract && make check-provider-surfaces`, and `make test` for the real-SDK canary).
- One commit per task.
- **Ordering vs the ecosystem-admission plan:** the agreed batch order lands eco-admission Tasks 1–2 (the `_solwyn_` internal rename + type transparency) BEFORE this plan. This plan's code sketches were written against pre-rename names — if the rename has landed, substitute the renamed attributes throughout (`self._client` → `self._solwyn_client`, `self._adapter` → `self._solwyn_adapter`, `s._budget` → `s._solwyn_budget`, `s._reporter` → `s._solwyn_reporter`, and the proxy's `self._solwyn._client` → `self._solwyn._solwyn_client`). Cited `client.py`/`_base.py` line numbers will also have drifted; treat them as anchors, not addresses. The new proxy classes' own attrs (`_SyncResponsesProxy._solwyn`) are outside the rename's AST guard (it covers `_SolwynBase`/`Solwyn`/`AsyncSolwyn` only).

## Design Decisions (settled — do not re-litigate during execution)

1. **Capability-conditional cached_property** — `Solwyn.responses` / `AsyncSolwyn.responses` follow the `messages` precedent (`client.py:1057-1071`): an adapter that serves the responses surface (Decision 11's `_responses_preparer` — native OpenAI in PR B, + azure_openai in PR C Task 8) gets the interception proxy; every other provider falls through to `_resolve_public_attribute(self._client, name="responses", path="responses", source=RAW)`, which reproduces today's guarded/AttributeError behavior exactly. (PR B may land the condition as `adapter.name == "openai"` and Task 8 generalizes it to the helper — but implementing the helper from the start is preferred.)
2. **Rides `_intercepted_call`, not `_media_call`** — Responses is token-billed, streamable, and settles on provider usage; it needs the chat pipeline's stream wrapper, breaker/latency accounting, and lease bounds. A new keyword-only `_surface: str = "chat"` parameter branches the four seams that differ (estimation, output bound + fallback hints, candidate restriction + kwargs pass-through, dispatch). Everything else — deadline, breaker admission, Retry-After same-provider retry, error classification/events, settlement, `_wrap_stream` — is reused untouched.
3. **Primary-only candidate walk** — after `_select_candidates`, the responses branch keeps only the primary runtime. If the primary's breaker is OPEN the list is empty → existing `ProviderUnavailableError` + reservation release path. A failed primary call raises to the caller (no fallback hop): compat providers don't uniformly serve `/v1/responses` and no translation subset exists. `check_budget` receives empty `fallback_providers`/`fallback_models`.
4. **Filtered defaults merge for responses kwargs** — `default_params` DOES apply to responses calls (a user who configured `default_params={"temperature": 0.2}` expects it on every call; silently dropping it is debt). The responses branch merges global defaults + the primary entry's `default_params` UNDER the caller's kwargs (caller always wins), minus `solwyn_tags` (existing filter) and a chat-only strip set `_CHAT_ONLY_DEFAULT_KEYS = frozenset({"max_tokens", "max_completion_tokens", "stream_options"})` — those are the identified keys the Responses endpoint rejects. Any other incompatible default is NOT silently filtered: it reaches the provider and 400s loudly, exactly as it would on a raw client call (fail-loud over silent-drift; no allow-list to go stale). `_build_hop_kwargs` itself is not reused — it is failover-hop-shaped; the responses merge is a small local expression.
5. **Duck-typed `prepare_responses_call` adapter seam** — mirrors the `MediaSurfaceAdapter` pattern: NOT added to the `ProviderAdapter` protocol (its docstring pins `prepare_call` as the stable chat seam). `_sync_dispatch`/`_async_dispatch` grow a `surface` parameter; `surface="responses"` resolves the method via `getattr`, failing loud with `UnsupportedSurfaceError` when absent. No `stream_options` injection — the Responses API rejects that param and its streams always carry usage on the terminal event.
6. **One accumulator, two shapes** — `OpenAIStreamAccumulator.observe` additionally saves `event.response` when it carries usage (`response.completed` / `response.incomplete` / `response.failed` terminal events). `finalize()` and `get_service_tier()` already read `.usage` / `.service_tier` off the saved object, so a saved `Response` works unchanged. Chat chunks have no `.response` attribute; no cross-talk.
7. **`background=True` refused loud** — a queued background response returns no usage at create time; silent pass-through would settle $0 (a budget bypass). Precedent: Bedrock `invoke_model` fails loud with `ConfigurationError`. Poll-based settlement is future work.
8. **Contract flip = wildcard-plus-override** — the existing wildcard/azure/together `unmetered_spend` rules for `responses.create` keep their ids and lose only their `provider:"openai"` selectors; a new higher-specificity `provider:"openai"` metered rule wins for native (the `videos.create` pair precedent, per `src/solwyn/CLAUDE.md`). Native `responses.create`: `kind=metered, source=both, policy_action=track, dispatch_action=intercept, usage_basis=provider` (mirrors native `chat.completions.create`, rule `surface.chat-completions-create.metered.44de662a8c9b`). The `responses` namespace gets the same split: native rule `source=both` (wrapper cached_property + raw), compat rules unchanged at `source=raw`. In Task 4 (PR B) all other `responses.*` leaves and all `beta.responses.*` rules are untouched — the proxy's `__getattr__` forwards them with `source=RAW`, which their existing rules already cover. Tasks 6–8 (PR C) then apply this same wildcard-plus-override pattern to `responses.parse`, `responses.stream`, and the Azure selectors — each flip atomic with its own proxy wiring (Decision 9 applies per task).
9. **Proxy + contract flip are ATOMIC** — a metered rule with no proxy makes `w.responses.create` hit the generic resolver's `RuntimeError("metered surface reached generic resolver")` (`_base.py:1108-1109,1129-1130`); a proxy with no metered rule resolves no WRAPPER-source rule and mis-postures the call. They land in the same task/commit.
10. **Acknowledgment copy migrates ONCE, to `responses.retrieve`** — `responses.create` is the canonical `acknowledge_untracked` example in README.md (lines ~271-282) and is pinned verbatim by `test_coverage_manifest.py::test_readme_states_the_strict_coverage_and_trust_boundary_contract` (~line 904). After the flips, `create`/`parse`/`stream` are no longer applicable unmetered leaves on native OpenAI — so every example/test usage moves to `responses.retrieve`, which stays unmetered after ALL of this plan's tasks (do NOT use `responses.parse` as the target; Task 6 meters it). One move, done in Task 4, stable through PR C. Pre-launch, the breaking acknowledgment change is acceptable (zero customers).
11. **Azure rides a `CompatProfile` capability flag, not adapter identity** — a new `supports_responses: bool = False` field on `CompatProfile`, set `True` only for `azure_openai` in v1. The capability question "can this adapter serve the responses surface?" is answered by one helper (`_responses_preparer(adapter)`: the duck-typed `prepare_responses_call` present AND `getattr(adapter, "supports_responses", True)` truthy), used by BOTH the cached_property condition and the pipeline pre-check — so the proxy, the pre-check, and the dispatcher can never disagree, and a future compat provider gains metering by flipping one profile field + its contract rules. Generic/unknown compat profiles stay excluded by default.
12. **`responses.stream()` helper wraps the SDK's stream manager, settles exactly once** — a `_ResponsesStreamManagerWrapper` (sync + async) enters the SDK's `ResponseStreamManager`, wraps the inner event stream for accumulator-based settlement (the terminal `response.completed` event carries usage — same Task 2 machinery), and forwards helper attributes (`get_final_response`, `until_done`, …) to the inner stream untouched. Settlement fires exactly once under the existing `_settled`-style guard: on the terminal event when observed, else as a length-based estimate on context-manager exit/close (abandonment ≠ $0). Helper methods never re-settle.
13. **Explicit `provider=` pin is a real feature, not a test hook** — `Solwyn(client, provider="openai")` (and async) bypasses base_url detection and selects the named adapter directly, validated against the registry and re-validated by `_validate_surface_context`. Legitimate use: native OpenAI behind a corporate proxy/gateway base_url that detection would classify as generic-compat. The E2E harness uses exactly this pin to route its `FakeProviderServer` through the native adapter — no test-only seams in core.

## File Structure

| File | Change |
|---|---|
| `src/solwyn/_privacy.py` | Add `estimate_responses_content_length(kwargs)` (content-privileged, length-only) |
| `src/solwyn/_base.py` | Add `_responses_output_bound(kwargs, default_bound)` (sans-I/O) |
| `src/solwyn/providers/openai.py` | Add `OpenAIAdapter.prepare_responses_call`; extend `OpenAIStreamAccumulator.observe` |
| `src/solwyn/client.py` | `_surface` param on both `_intercepted_call`s; `surface` param on `_sync_dispatch`/`_async_dispatch`; `responses` cached_property on `Solwyn` + `AsyncSolwyn` |
| `src/solwyn/_proxies.py` | Add `_SyncResponsesProxy`, `_AsyncResponsesProxy` |
| `src/solwyn/_surfaces.py` | Regenerated payload only (embed script — never hand-edited) |
| `tests/unit/test_responses_estimation.py` | NEW — estimator + output-bound tests |
| `tests/unit/test_providers/test_openai.py` | `prepare_responses_call` tests |
| `tests/unit/test_stream_accumulators.py` | Terminal-event observe tests |
| `tests/unit/test_responses_call.py` | NEW — pipeline branch + proxy behavior tests (sync + async) |
| `tests/unit/test_surface_context_pins.py` | Openai-context digests re-pinned |
| `tests/unit/test_unmetered_posture.py`, `tests/unit/test_coverage_manifest.py` | Migrate `responses.create` example/acknowledgment usages to `responses.parse` |
| `README.md` | Acknowledgment example swap + `OPENAI_STRICT_FINGERPRINT` refresh + Responses feature row |
| `CHANGELOG.md`, `src/solwyn/CLAUDE.md`, `docs/plans/2026-08-07-ecosystem-admission-plan.md` | Docs updates |
| `src/solwyn/providers/openai_compatible.py` | PR C: `supports_responses` CompatProfile flag (azure_openai only) + compat `prepare_responses_call` (Task 8) |
| `src/solwyn/_proxies.py` (or `stream.py`) | PR C: `_ResponsesStreamManagerWrapper` sync/async (Task 7) |
| `src/solwyn/_registry.py`, `src/solwyn/client.py` | PR C: explicit `provider=` adapter pin (Task 9) |
| E2E harness (`FakeProviderServer` + its test suite) | PR C: `/v1/responses` fake endpoint + native-pin live-pipeline test (Task 9) |

---

## PR Grouping

**PR A — "Responses metering foundations" (Tasks 1–2).** Pure sans-I/O additions: estimator, output bound, adapter seam, accumulator extension. Zero behavior change for any existing call path; every addition is unit-tested and inert until PR B wires it. Rationale: keeps the reviewable surface of the risky PR small, and these pieces are independently correct/testable against the OpenAI wire shapes.

**PR B — "Meter native OpenAI responses.create" (Tasks 3–5).** Pipeline branch, proxies, surface-contract flip (with pins/fingerprints/copy migrations), and docs. Rationale: Decision 9 makes proxy + contract flip inseparable, and the pipeline branch is untestable end-to-end without the proxy; shipping the docs in the same PR keeps the README's pinned coverage claims true at every merge point. Task granularity inside the PR is preserved as one commit per task, each green.

**PR C — "Responses coverage completion" (Tasks 6–9).** `responses.parse`, the `responses.stream()` helper, Azure OpenAI parity via the `CompatProfile` capability flag, and the explicit `provider=` pin + E2E coverage. Rationale: each rides the machinery PR B proved out (same pipeline branch, same accumulator, same flip workflow), so they review as increments, and grouping them keeps digest re-pins to one more round. Task granularity stays one green commit per task.

Docs note: each PR keeps its own README/pinned-copy claims true at merge (the coverage-manifest test enforces this) — a docs-only trailing PR is impossible by construction.

---

### Task 1: Responses request estimation + output bound (sans-I/O)

**Files:**
- Modify: `src/solwyn/_privacy.py` (after `estimate_content_length`, ~line 693)
- Modify: `src/solwyn/_base.py` (near `_dialect_output_cap`, ~line 351)
- Test: `tests/unit/test_responses_estimation.py` (new)

**Interfaces:**
- Consumes: `_positive_output_cap` (`_base.py:333`), existing `estimate_tokens_from_length` (unchanged, used later by the caller).
- Produces: `estimate_responses_content_length(kwargs: dict[str, Any]) -> int` and `_responses_output_bound(kwargs: dict[str, object], default_bound: int) -> int` — Task 3 calls both by these exact names.

Why a separate estimator instead of extending `estimate_content_length`: that function is also called by `_media_call` (`client.py:1264`) where embeddings/TTS kwargs carry an `input` key that must NOT be counted there (embeddings pre-flight would double-shift). A dedicated recognizer keeps the change surface-local.

- [ ] **Step 1: Write the failing tests**

```python
"""Responses API request estimation + output bound (sans-I/O)."""

from __future__ import annotations

import pytest

from solwyn._base import _responses_output_bound
from solwyn._privacy import estimate_responses_content_length


@pytest.mark.unit
class TestEstimateResponsesContentLength:
    def test_string_input(self) -> None:
        assert estimate_responses_content_length({"input": "hello world"}) == 11

    def test_instructions_added(self) -> None:
        kwargs = {"input": "hi", "instructions": "be brief"}
        assert estimate_responses_content_length(kwargs) == 2 + 8

    def test_message_items_with_string_content(self) -> None:
        kwargs = {"input": [{"role": "user", "content": "abcd"}]}
        assert estimate_responses_content_length(kwargs) == 4

    def test_message_items_with_text_parts(self) -> None:
        kwargs = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "abc"},
                        {"type": "input_image", "image_url": "https://x/y.png"},
                    ],
                }
            ]
        }
        # image part carries no countable text
        assert estimate_responses_content_length(kwargs) == 3

    def test_function_call_output_items(self) -> None:
        kwargs = {"input": [{"type": "function_call_output", "call_id": "c1", "output": "12345"}]}
        assert estimate_responses_content_length(kwargs) == 5

    def test_no_recognizable_content(self) -> None:
        # prompt-template / previous_response_id calls may carry no input at all
        assert estimate_responses_content_length({"previous_response_id": "resp_1"}) == 0

    def test_garbage_shapes_degrade_to_zero_not_raise(self) -> None:
        assert estimate_responses_content_length({"input": [42, None, "x"]}) == 0
        assert estimate_responses_content_length({"input": {"not": "a list"}}) == 0


@pytest.mark.unit
class TestResponsesOutputBound:
    def test_reads_max_output_tokens(self) -> None:
        assert _responses_output_bound({"max_output_tokens": 512}, 4096) == 512

    def test_falls_back_to_default(self) -> None:
        assert _responses_output_bound({}, 4096) == 4096

    def test_ignores_garbage_values(self) -> None:
        assert _responses_output_bound({"max_output_tokens": True}, 4096) == 4096
        assert _responses_output_bound({"max_output_tokens": -5}, 4096) == 4096
        assert _responses_output_bound({"max_output_tokens": "big"}, 4096) == 4096
```

Note `test_garbage_shapes_degrade_to_zero_not_raise`: a non-list `input` returns whatever was counted so far (estimation is heuristic and must never destroy a call); string items inside the list (`"x"`) are NOT items with content and count 0 — only dict items are read.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_responses_estimation.py -v`
Expected: FAIL with `ImportError: cannot import name 'estimate_responses_content_length'`

- [ ] **Step 3: Implement the estimator in `_privacy.py`**

Place after `estimate_content_length` / `_google_prompt_text_length`:

```python
def estimate_responses_content_length(kwargs: dict[str, Any]) -> int:
    """Total character length of prompt content in a Responses API request.

    Walks ``instructions`` and ``input`` (a string, or a list of message /
    tool-output items whose text rides ``content`` — a string or a list of
    ``{"text": ...}`` parts — or ``output``) and sums string lengths WITHOUT
    concatenating them. The returned integer is safe to log — it is not
    reversible to prompt content. Unrecognized shapes contribute 0 (estimation
    is heuristic; it must never reject a call). Distinct from
    ``estimate_content_length`` on purpose: that recognizer also serves the
    media lifecycle, where an ``input`` key (embeddings, TTS) must NOT be
    counted as chat content.
    """
    total = 0
    instructions = kwargs.get("instructions")
    if isinstance(instructions, str):
        total += len(instructions)

    value = kwargs.get("input")
    if isinstance(value, str):
        return total + len(value)
    if not isinstance(value, list):
        return total
    for item in value:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        total += len(text)
        output = item.get("output")
        if isinstance(output, str):
            total += len(output)
    return total
```

- [ ] **Step 4: Implement the output bound in `_base.py`**

Place near `_dialect_output_cap`:

```python
def _responses_output_bound(kwargs: dict[str, object], default_bound: int) -> int:
    """Effective output reservation for one Responses API call.

    The Responses API caps output via ``max_output_tokens`` (the chat
    ``max_tokens`` family is not accepted on that endpoint, so
    ``_effective_output_bound`` cannot see this cap). An unbounded call
    contributes the configured conservative fallback, mirroring the chat path.
    """
    return _positive_output_cap(kwargs.get("max_output_tokens")) or default_bound
```

- [ ] **Step 5: Run the tests + privacy firewall**

Run: `uv run pytest tests/unit/test_responses_estimation.py tests/unit/test_privacy_firewall.py -v`
Expected: PASS (the firewall test is path-based; `_privacy.py` is allowlisted)

- [ ] **Step 6: Full gate + commit**

Run: `make check && make test-unit`

```bash
git add src/solwyn/_privacy.py src/solwyn/_base.py tests/unit/test_responses_estimation.py
git commit -m "feat: Responses API request estimation + max_output_tokens bound (sans-I/O)"
```

---

### Task 2: OpenAI adapter seam + stream accumulator terminal-event support

**Files:**
- Modify: `src/solwyn/providers/openai.py` (`OpenAIAdapter`, ~line 425; `OpenAIStreamAccumulator.observe`, ~line 496)
- Test: `tests/unit/test_providers/test_openai.py`, `tests/unit/test_stream_accumulators.py`

**Interfaces:**
- Consumes: `_extract_openai_usage`, `_extract_service_tier` (both already handle the Responses `Response` object).
- Produces: `OpenAIAdapter.prepare_responses_call(client, kwargs, *, is_streaming: bool) -> tuple[Callable[..., Any], dict[str, Any]]` — Task 3's dispatchers resolve it via `getattr(adapter, "prepare_responses_call", None)`. `OpenAIStreamAccumulator` unchanged in signature.

- [ ] **Step 1: Write the failing adapter tests** (in `tests/unit/test_providers/test_openai.py`)

```python
@pytest.mark.unit
class TestPrepareResponsesCall:
    def test_routes_to_responses_create(self) -> None:
        adapter = OpenAIAdapter()
        client = MagicMock()
        kwargs = {"model": "gpt-5.5", "input": "hi"}
        method, shaped = adapter.prepare_responses_call(client, kwargs, is_streaming=False)
        assert method is client.responses.create
        assert shaped == kwargs
        assert shaped is not kwargs  # defensive copy, caller dict never aliased

    def test_streaming_sets_stream_kwarg_without_stream_options(self) -> None:
        adapter = OpenAIAdapter()
        client = MagicMock()
        method, shaped = adapter.prepare_responses_call(
            client, {"model": "gpt-5.5", "input": "hi"}, is_streaming=True
        )
        assert shaped["stream"] is True
        # The Responses API rejects stream_options; usage always rides the
        # terminal response.completed event, so nothing must be injected.
        assert "stream_options" not in shaped
```

- [ ] **Step 2: Write the failing accumulator tests** (in `tests/unit/test_stream_accumulators.py`, `TestOpenAIStreamAccumulator`)

```python
    def test_responses_terminal_event_nested_usage(self) -> None:
        """Real Responses streams nest usage at event.response.usage (response.completed)."""
        acc = OpenAIStreamAccumulator()
        acc.observe(SimpleNamespace(type="response.output_text.delta", delta="hi"))
        acc.observe(
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=120,
                        output_tokens=45,
                        input_tokens_details=SimpleNamespace(cached_tokens=10),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=5),
                    ),
                    service_tier="flex",
                ),
            )
        )
        result = acc.finalize()
        assert result.input_tokens == 120
        assert result.output_tokens == 45
        assert result.cached_input_tokens == 10
        assert result.reasoning_tokens == 5
        assert acc.get_service_tier() == "flex"

    def test_responses_pre_terminal_events_with_usage_none_are_ignored(self) -> None:
        """response.created / in_progress carry a response whose usage is still None."""
        acc = OpenAIStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                type="response.created",
                response=SimpleNamespace(usage=None, service_tier=None),
            )
        )
        assert acc.finalize() == TokenDetails()
        assert acc.get_service_tier() is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_providers/test_openai.py -k ResponsesCall -v && uv run pytest tests/unit/test_stream_accumulators.py -k responses_terminal -v`
Expected: FAIL (`AttributeError: prepare_responses_call` / zero-usage finalize)

- [ ] **Step 4: Implement `prepare_responses_call`** (on `OpenAIAdapter`, after `prepare_call`)

```python
    def prepare_responses_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Responses API hop: streaming rides the ``stream=True`` kwarg.

        Duck-typed seam (mirrors ``prepare_media_call``): only the native
        OpenAI adapter serves it; the dispatcher fails loud when absent. No
        ``stream_options`` injection — the Responses API rejects that param and
        its streams always carry usage on the terminal ``response.completed``
        event. Returns a defensive COPY of kwargs; never reads content.
        """
        kwargs = dict(kwargs)
        if is_streaming:
            kwargs["stream"] = True
        return client.responses.create, kwargs
```

- [ ] **Step 5: Extend `OpenAIStreamAccumulator.observe`**

```python
    def observe(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage_chunk = chunk
            return
        # Responses API streams nest usage on the terminal events'
        # ``event.response`` (response.completed / .incomplete / .failed).
        # Saving the inner Response keeps finalize()/get_service_tier()
        # unchanged: both read ``.usage`` / ``.service_tier`` off the saved
        # object. Chat chunks carry no ``response`` attribute.
        response = getattr(chunk, "response", None)
        if response is not None and getattr(response, "usage", None) is not None:
            self._usage_chunk = response
```

- [ ] **Step 6: Run tests, verify the whole accumulator file still passes**

Run: `uv run pytest tests/unit/test_stream_accumulators.py tests/unit/test_providers/test_openai.py -v`
Expected: PASS (including the pre-existing chat-shape tests)

- [ ] **Step 7: Full gate + commit**

Run: `make check && make test-unit`

```bash
git add src/solwyn/providers/openai.py tests/unit/test_providers/test_openai.py tests/unit/test_stream_accumulators.py
git commit -m "feat: OpenAI adapter responses dispatch seam + terminal-event stream usage"
```

**→ Open PR A here** ("Responses metering foundations"). Body: note the additions are inert until the wiring PR; link this plan.

---

### Task 3: `_intercepted_call` responses branch + dispatch routing (sync + async)

**Files:**
- Modify: `src/solwyn/client.py` — `_sync_dispatch` (~1171), `_intercepted_call` (~1467), `_async_dispatch` (~2332), async `_intercepted_call` (~2594)
- Test: `tests/unit/test_responses_call.py` (new)

**Interfaces:**
- Consumes: `estimate_responses_content_length` (Task 1), `_responses_output_bound` (Task 1), `prepare_responses_call` (Task 2), `UnsupportedSurfaceError` (`exceptions.py`).
- Produces: `_intercepted_call(*, _force_stream=False, _surface="chat", **kwargs)` on both classes; `_sync_dispatch(..., surface="chat")` / `_async_dispatch(..., surface="chat")`. Task 4's proxies call `_intercepted_call(_surface="responses", **kwargs)`.

This task adds NO public exposure and NO contract change — the branch is exercised through `_intercepted_call` directly, so the surface canaries and pins are untouched and the task lands green on its own.

- [ ] **Step 1: Write the failing pipeline tests**

Build on the harness patterns in `tests/unit/test_client.py` (`_make_solwyn`, `_allow_budget_result`) — import them or replicate minimally; use `from conftest import VALID_API_KEY, VALID_PROJECT_ID` for shared constants.

```python
"""Responses-surface pipeline branch: estimation, bound, primary-only walk, settlement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from conftest import VALID_API_KEY, VALID_PROJECT_ID
from solwyn.client import Solwyn
from solwyn.exceptions import UnsupportedSurfaceError


def _mock_openai_responses_client():
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    response = MagicMock()
    response.usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=45,
        input_tokens_details=SimpleNamespace(cached_tokens=10),
        output_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )
    response.service_tier = "default"
    client.responses.create.return_value = response
    return client, response


def _allow_budget() -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id="res_1",
        project_id=VALID_PROJECT_ID,
        price_hints=None,
        failover_tuning_allowed=None,
    )


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __iter__(self):
        yield from self._events

    def close(self):
        self.closed = True


@pytest.mark.unit
class TestResponsesInterceptedCall:
    def test_dispatches_to_responses_create_and_settles(self) -> None:
        client, response = _mock_openai_responses_client()
        s = _make_solwyn(client)
        settlements = []
        s._reporter.report_settlement = lambda confirm, event: settlements.append((confirm, event))
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()) as check:
            result = s._intercepted_call(_surface="responses", model="gpt-5.5", input="hi there")
        # dispatch went to responses.create with untouched caller kwargs
        assert result is response
        assert client.responses.create.call_args.kwargs["input"] == "hi there"
        assert client.chat.completions.create.call_count == 0
        # settlement rode report_settlement with Responses-shape usage
        (confirm, event) = settlements[0]
        assert event.input_tokens == 120 and event.output_tokens == 45
        # budget check: no fallback hints, responses estimation, responses bound
        kw = check.call_args.kwargs
        assert kw["fallback_providers"] == [] and kw["fallback_models"] == []
        assert kw["estimated_input_tokens"] > 0  # "hi there" chars -> tokens

    def test_max_output_tokens_feeds_estimated_output_bound(self) -> None:
        client, _ = _mock_openai_responses_client()
        s = _make_solwyn(client)
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()) as check:
            s._intercepted_call(
                _surface="responses", model="gpt-5.5", input="x", max_output_tokens=512
            )
        assert check.call_args.kwargs["estimated_output_bound"] == 512

    def test_default_params_merge_filtered_for_responses(self) -> None:
        # Valid defaults apply; chat-only keys are stripped; caller wins.
        client, _ = _mock_openai_responses_client()
        s = _make_solwyn(
            client, default_params={"max_tokens": 100, "temperature": 0.5, "top_p": 0.9}
        )
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()):
            s._intercepted_call(_surface="responses", model="gpt-5.5", input="x", top_p=0.2)
        sent = client.responses.create.call_args.kwargs
        assert "max_tokens" not in sent          # chat-only key stripped
        assert sent["temperature"] == 0.5        # valid default merged
        assert sent["top_p"] == 0.2              # caller kwarg beats default
        assert "solwyn_tags" not in sent

    def test_provider_error_raises_without_failover(self) -> None:
        client, _ = _mock_openai_responses_client()
        s = _make_solwyn(client)
        err = _Status(429)  # transport error harness from test_client.py
        client.responses.create.side_effect = err
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()):
            with pytest.raises(_Status):
                s._intercepted_call(_surface="responses", model="gpt-5.5", input="x")

    def test_streaming_wraps_and_settles_on_terminal_event(self) -> None:
        client, _ = _mock_openai_responses_client()
        terminal = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=30,
                    output_tokens=12,
                    input_tokens_details=None,
                    output_tokens_details=None,
                ),
                service_tier="flex",
            ),
        )
        client.responses.create.return_value = _FakeStream(
            [SimpleNamespace(type="response.output_text.delta", delta="h"), terminal]
        )
        s = _make_solwyn(client)
        settlements = []
        s._reporter.report_settlement = lambda confirm, event: settlements.append((confirm, event))
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()):
            stream = s._intercepted_call(
                _surface="responses", model="gpt-5.5", input="x", stream=True
            )
        events = list(stream)
        assert events[-1] is terminal
        assert client.responses.create.call_args.kwargs["stream"] is True
        assert "stream_options" not in client.responses.create.call_args.kwargs
        (confirm, event) = settlements[0]
        assert event.input_tokens == 30 and event.output_tokens == 12
        assert event.service_tier == "flex"

    def test_non_openai_adapter_fails_loud_before_budget_check(self) -> None:
        # anthropic adapter has no prepare_responses_call; the capability
        # pre-check must refuse BEFORE check_budget so no reservation is ever
        # taken for a call that can never dispatch.
        client, _ = _mock_anthropic_client()  # from test_client.py harness
        s = _make_solwyn(client)
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()) as check:
            with pytest.raises(UnsupportedSurfaceError):
                s._intercepted_call(_surface="responses", model="claude-x", input="x")
        assert check.call_count == 0

    def test_abandoned_stream_settles_estimated_not_zero(self) -> None:
        # Caller closes the stream before the terminal response.completed event
        # arrives: settlement must ride the length-based estimate (marked
        # is_estimated), never a silent $0. Mirror the existing chat
        # abandoned-stream test's assertions (see test_client.py /
        # test_stream_failover.py early-close cases) for the exact fields.
        client, _ = _mock_openai_responses_client()
        client.responses.create.return_value = _FakeStream(
            [SimpleNamespace(type="response.output_text.delta", delta="h")]  # no terminal event
        )
        s = _make_solwyn(client)
        settlements = []
        s._reporter.report_settlement = lambda confirm, event: settlements.append((confirm, event))
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()):
            stream = s._intercepted_call(
                _surface="responses", model="gpt-5.5", input="x", stream=True
            )
        iterator = iter(stream)
        next(iterator)
        stream.close()
        assert len(settlements) == 1
        (_, event) = settlements[0]
        assert event.token_details.is_estimated is True
```

Mirror the dispatch/settlement/streaming tests for `AsyncSolwyn` (`@pytest.mark.asyncio`, `AsyncMock` for `responses.create`, async fake stream with `aclose`). Copy the local helpers rather than importing across test modules where the repo convention prefers duplication (see the `_mock_anthropic_client` note in `tests/CLAUDE.md`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_responses_call.py -v`
Expected: FAIL with `TypeError: _intercepted_call() got an unexpected keyword argument '_surface'`

- [ ] **Step 3: Implement the sync branch**

`_sync_dispatch` — add `surface: str = "chat"` keyword and route:

```python
        if surface == "responses":
            prepare = getattr(runtime.adapter, "prepare_responses_call", None)
            if prepare is None:
                raise UnsupportedSurfaceError(
                    surface="responses.create", provider=runtime.adapter.name
                )
            method, call_kwargs = prepare(
                client, cast("dict[str, Any]", kwargs), is_streaming=is_streaming
            )
            return method(**call_kwargs)
```

(The `with_options` timeout/max_retries application above the seam is shared and stays before this branch.)

`_intercepted_call` — signature `def _intercepted_call(self, *, _force_stream: bool = False, _surface: str = "chat", **kwargs: object) -> Any:` and four surgical branch points, preceded by a capability pre-check:

0. **Capability pre-check** — immediately after the primary runtime is resolved (`runtime = self._runtimes[0]`), BEFORE the budget check, so no reservation is ever taken for a call that can never dispatch (the dispatcher's own `getattr` check in `_sync_dispatch` stays as a backstop):

```python
        if _surface == "responses" and getattr(
            runtime.adapter, "prepare_responses_call", None
        ) is None:
            raise UnsupportedSurfaceError(
                surface="responses.create", provider=runtime.adapter.name
            )
```

1. **Estimation** (replaces the `metering_kwargs` block usage for responses; the legacy-google `metering_kwargs` shape check is chat-only, skip it):

```python
        if _surface == "responses":
            char_count = estimate_responses_content_length(cast("dict[str, Any]", kwargs))
        else:
            char_count = estimate_content_length(metering_kwargs)
```

2. **Budget check** — for responses pass `fallback_providers=[]`, `fallback_models=[]`, and `estimated_output_bound=_responses_output_bound(kwargs, self._config.lease_output_bound_default)` (the chat path keeps its existing `max(...)` expression).

3. **Candidate restriction + kwargs pass-through** — after the existing `candidates = self._select_candidates(...)` and idempotency filtering:

```python
        if _surface == "responses":
            # v1: native-OpenAI primary only. No translation subset exists for
            # the Responses request shape and compat /v1/responses support is
            # not uniform, so a failed primary raises instead of walking the
            # chain. Health filtering still applies: an OPEN primary breaker
            # leaves no candidates and takes the ProviderUnavailableError path.
            candidates = [c for c in candidates if c is primary]
```

   Inside the walk, the hop-kwargs build becomes surface-aware — `_build_hop_kwargs` and `prepare_streaming` are chat-shaped and are skipped for responses (Decision 4/5):

```python
                if _surface == "responses":
                    # Filtered defaults merge (Decision 4): defaults under caller
                    # kwargs, minus solwyn_tags and the chat-only strip set.
                    call_kwargs = {
                        key: value
                        for key, value in {**global_defaults, **rt.entry.default_params}.items()
                        if key != "solwyn_tags" and key not in _CHAT_ONLY_DEFAULT_KEYS
                    } | dict(kwargs)
                    served_model = requested_model
                else:
                    call_kwargs = _build_hop_kwargs(...)   # existing code
                    served_model = requested_model if is_primary else rt.entry.model
                    if is_streaming:
                        call_kwargs = rt.adapter.prepare_streaming(...)
```

4. **Dispatch** — thread `surface=_surface` into the `self._sync_dispatch(...)` call. Everything downstream (breaker verdicts, error events, `_wrap_stream`, settlement) is untouched; the streaming branch already builds the accumulator from the served adapter, which Task 2 taught the terminal-event shape.

- [ ] **Step 4: Run the sync tests**

Run: `uv run pytest tests/unit/test_responses_call.py -v -k "not asyncio"`
Expected: PASS

- [ ] **Step 5: Mirror in `AsyncSolwyn`** (`_async_dispatch` + async `_intercepted_call`, identical branch structure)

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/unit/test_responses_call.py tests/unit/test_client.py tests/unit/test_stream_failover.py -v`
Expected: PASS (chat pipeline behavior unchanged)

- [ ] **Step 7: Full gate + commit**

Run: `make check && make test-unit`

```bash
git add src/solwyn/client.py tests/unit/test_responses_call.py
git commit -m "feat: responses surface branch in the interception pipeline (sync + async)"
```

---

### Task 4: Responses proxies + surface-contract flip (ATOMIC)

**Files:**
- Modify: `src/solwyn/_proxies.py` (new proxy classes)
- Modify: `src/solwyn/client.py` (`responses` cached_property on both classes, near `messages` ~1057/~2251)
- Modify: `build/surface_contract/surface-classification.json` (transient, via scripts) → `src/solwyn/_surfaces.py` (regenerated payload)
- Modify: `tests/unit/test_surface_context_pins.py` (openai-context digests), `README.md` (acknowledgment example + `OPENAI_STRICT_FINGERPRINT`), `tests/unit/test_coverage_manifest.py` (pinned README claims), `tests/unit/test_unmetered_posture.py` (acknowledgment token usages)
- Test: `tests/unit/test_responses_call.py` (proxy section), existing posture/canary suites

**Interfaces:**
- Consumes: `_intercepted_call(_surface="responses", ...)` (Task 3), `_enforce_explicit_surface` / `_resolve_public_attribute` (`_base.py:1038/1071`), `ConfigurationError`.
- Produces: `Solwyn.responses` / `AsyncSolwyn.responses` cached properties; `_SyncResponsesProxy` / `_AsyncResponsesProxy` with `create(**kwargs)` and pass-through `__getattr__`.

- [ ] **Step 1: Write the failing proxy tests** (append to `tests/unit/test_responses_call.py`)

```python
@pytest.mark.unit
class TestResponsesProxy:
    def test_native_openai_create_is_intercepted(self) -> None:
        client, response = _mock_openai_responses_client()
        s = _make_solwyn(client)
        with patch.object(s._budget, "check_budget", return_value=_allow_budget()) as check:
            result = s.responses.create(model="gpt-5.5", input="hi")
        assert result is response
        assert check.call_count == 1  # budget pre-flight ran

    def test_native_openai_create_does_not_warn_unmetered(self, caplog) -> None:
        client, _ = _mock_openai_responses_client()
        s = _make_solwyn(client)
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            with patch.object(s._budget, "check_budget", return_value=_allow_budget()):
                s.responses.create(model="gpt-5.5", input="hi")
        assert not [r for r in foreground_records(caplog) if "unmetered" in r.message.lower()]

    def test_background_true_refused_loud(self) -> None:
        client, _ = _mock_openai_responses_client()
        s = _make_solwyn(client)
        with pytest.raises(ConfigurationError):
            s.responses.create(model="gpt-5.5", input="hi", background=True)
        assert client.responses.create.call_count == 0

    def test_other_leaves_stay_guarded_passthrough(self) -> None:
        client, _ = _mock_openai_responses_client()
        client.responses.retrieve.return_value = "raw-retrieved"
        s = _make_solwyn(client)
        # .retrieve is still an unmetered_spend leaf: warned, then served raw
        assert s.responses.retrieve("resp_1") == "raw-retrieved"

    def test_compat_provider_keeps_unmetered_posture(self, caplog) -> None:
        # Groq-shaped compat client: responses.create must remain a WARNED raw
        # pass-through, never budget-checked (v1 is native-only).
        client = MagicMock()
        client.__class__.__module__ = "openai._client"
        client.__class__.__name__ = "OpenAI"
        client.base_url = "https://api.groq.com/openai/v1"
        client.with_options.return_value = client
        client.responses.create.return_value = "compat-raw"
        s = _make_solwyn(client)
        with patch.object(s._budget, "check_budget") as check:
            with caplog.at_level(logging.WARNING, logger="solwyn._base"):
                assert s.responses.create(model="llama-x", input="hi") == "compat-raw"
        assert check.call_count == 0
        assert any("responses.create" in r.message for r in foreground_records(caplog))
```

Adjust the compat-client construction to match how `tests/unit/test_openai_compatible_client.py` builds a detected-compat mock (base_url attribute shape); reuse its helper if one exists.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_responses_call.py -k Proxy -v`
Expected: FAIL (`w.responses` resolves to the guarded raw resource; `create` is not intercepted; no budget check)

- [ ] **Step 3: Implement the proxies** (in `_proxies.py`, sync shown; async mirrors with `async def create` awaiting the async `_intercepted_call`)

```python
class _SyncResponsesProxy:
    """Proxy for client.responses that intercepts create() (native OpenAI only).

    ``client.responses.create()`` flows through ``_intercepted_call`` under the
    ``responses`` surface: budget pre-flight, primary-only dispatch, and
    settlement via ``reporter.report_settlement`` — non-streaming and
    ``stream=True`` alike. ``default_params`` applies under the caller's
    kwargs, minus the chat-only keys the Responses endpoint rejects
    (``max_tokens`` family, ``stream_options``); other incompatible defaults
    fail loud at the provider, as on a raw call.

    ``background=True`` is refused loud: a queued background response carries
    no usage at create time, so silent pass-through would settle $0 — the same
    budget-bypass posture that fails Bedrock ``invoke_model`` loud. Every other
    ``responses`` attribute (``parse``, ``stream``, ``retrieve``, ...) is
    resolved by the shared surface policy and keeps its unmetered_spend
    posture.
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn

    def create(self, **kwargs: Any) -> Any:
        self._solwyn._enforce_explicit_surface("responses.create", source=SurfaceSource.WRAPPER)
        if kwargs.get("background") is True:
            raise ConfigurationError(
                "responses.create(background=True) is not supported: a queued "
                "background response reports no usage at create time, so Solwyn "
                "cannot meter it. Use the raw client for background responses.",
                field="background",
            )
        return self._solwyn._intercepted_call(_surface="responses", **kwargs)

    def __getattr__(self, name: str) -> Any:
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.responses,
            name=name,
            path=f"responses.{name}",
            source=SurfaceSource.RAW,
        )
```

`client.py` cached properties (sync shown; async mirrors with `_AsyncResponsesProxy`):

```python
    @functools.cached_property
    def responses(self) -> Any:
        """Native OpenAI: client.responses.create() goes through interception.

        Conditional like ``messages``: only the native OpenAI adapter serves a
        metered responses seam in v1. Azure/compat/other-dialect clients fall
        through to the shared surface policy, preserving their guarded
        unmetered_spend (or absent-attribute) behavior unchanged. Cached: the
        adapter is fixed at construction.
        """
        if self._adapter.name == "openai":
            return _SyncResponsesProxy(self)
        return self._resolve_public_attribute(
            self._client,
            name="responses",
            path="responses",
            source=SurfaceSource.RAW,
        )
```

Import `ConfigurationError` in `_proxies.py`; add both proxy classes to the client imports.

- [ ] **Step 4: Flip the contract via the sanctioned loop** (from `src/solwyn/CLAUDE.md`; NEVER hand-edit `_surfaces.py`)

1. `uv run python scripts/export_surface_contract.py`
2. Edit `build/surface_contract/surface-classification.json`:
   - Rule `surface.responses-create.unmetered_spend.f323258fe692`: REMOVE the two `provider: "openai"` selectors (sync + async). Keep id, keep wildcard/azure/together selectors.
   - ADD `surface.responses-create.metered.<new-suffix>`: selectors `provider:"openai"` (sync + async, `dialect:"openai"`, `client_shape:"openai_sdk"`); `kind=metered, source=both, policy_action=track, dispatch_action=intercept, usage_basis=provider`; shapes `[{"descriptor_category":"function","return_shape":"callable"}]`. Mirror `surface.chat-completions-create.metered.44de662a8c9b` field-for-field. The id's third segment must equal `metered`.
   - Rule `surface.responses.namespace.<existing>`: REMOVE the two `provider:"openai"` selectors; ADD `surface.responses.namespace.<new-suffix>` for `provider:"openai"` with `source=both` and shapes including `{"descriptor_category":"cached_property","return_shape":"resource"}` (compare against the `chat` namespace rule `surface.chat.namespace.c22d6909b91f` and let step 5's canaries confirm the exact shape set).
   - Touch NOTHING else (no other leaves, no `beta.responses.*`).
3. `uv run python scripts/embed_surface_rules.py --input build/surface_contract/surface-classification.json` — review the printed rule delta: exactly the edits above, nothing more.
4. `uv run python scripts/export_surface_contract.py` (refresh the expanded artifact + fingerprint).
5. Update the `("openai", "openai", "openai_sdk", sync/async)` digests in `tests/unit/test_surface_context_pins.py` (only openai contexts should move — if azure/compat/together digests move, the JSON edit touched too much; STOP and re-check) and the README `OPENAI_STRICT_FINGERPRINT` block (~line 320).

**Caution on step 2's edit shape:** the instructions above assume the existing `unmetered_spend` rules carry per-provider selectors that can be removed individually. If the exported JSON instead shows a `provider: null` wildcard selector, do NOT force the removal — use the documented wildcard-plus-override pattern as-is (keep the wildcard, add only the higher-specificity `provider:"openai"` rules, per the `videos.create` pair in `src/solwyn/CLAUDE.md`). In either case the authorities are the embed script's printed rule delta and the only-openai-digests-move guardrail, not this plan's exact edit list.

- [ ] **Step 5: Migrate the acknowledgment example copy** (Decision 10)

- `README.md` ~271-282: swap `responses.create` → `responses.retrieve` (NOT `responses.parse` — Task 6 meters it; Decision 10) in the `acknowledge_untracked` example and the `SOLWYN_ACKNOWLEDGE_UNTRACKED` example; keep the "Namespace tokens such as `responses` are invalid" sentence.
- `tests/unit/test_coverage_manifest.py`: update the pinned README claims (~904-905) and the acknowledgment fixtures at ~179-184/~217 to `responses.retrieve` (keep `with_raw_response.responses.create` usages — that leaf is still unmetered).
- `tests/unit/test_unmetered_posture.py` ~505/512: this fixture acknowledges `responses.create` — move it to `responses.retrieve` (or a compat-provider context) so it still exercises a real unmetered leaf that SURVIVES PR C. Line ~578 (`with_raw_response.responses.create`) stays.
- `grep -rn "responses.create" README.md docs/ src/ tests/` and resolve every remaining reference that implies the leaf is unmetered on native OpenAI.

- [ ] **Step 6: Run the full verification chain**

Run: `make check && make test-unit && make check-surface-contract && make check-provider-surfaces && make test`
Expected: PASS. If a canary or pin fails, read `docs/surface-canary-runbook.md` — do not chase it by hand-editing `_surfaces.py`.

- [ ] **Step 7: Generate the PR diff artifact**

Run: `uv run python scripts/diff_surface_rules.py origin/main` (PR B targets main) — save the output for the PR body.

- [ ] **Step 8: Commit**

```bash
git add src/solwyn/_proxies.py src/solwyn/client.py src/solwyn/_surfaces.py \
        tests/unit/test_responses_call.py tests/unit/test_surface_context_pins.py \
        tests/unit/test_coverage_manifest.py tests/unit/test_unmetered_posture.py README.md
git commit -m "feat: meter native OpenAI responses.create (proxy + surface-contract flip)"
```

(`build/surface_contract/` stays uncommitted — it is a transient CI artifact; never commit the expanded JSON.)

---

### Task 5: Documentation sync

**Files:**
- Modify: `README.md` (feature/coverage section: add the Responses row — native OpenAI `responses.create` incl. `stream=True` is metered; `.parse`/`.stream()` helper/background remain guarded)
- Modify: `CHANGELOG.md` (feature entry + the acknowledgment-token breaking note: `responses.create` is no longer an acknowledgeable unmetered token on native OpenAI)
- Modify: `src/solwyn/CLAUDE.md`:
  - "Client Proxy Patterns": add one bullet — `responses` is a conditional cached_property (native OpenAI → interception proxy via `_intercepted_call(_surface="responses")`, primary-only, filtered defaults merge (chat-only keys stripped), `background=True` refused loud; others → shared surface policy).
  - "Provider Adapter Notes" OpenAI bullet: note `prepare_responses_call` (duck-typed, no stream_options) and terminal-event stream usage (`event.response.usage`).
  - "Coverage Contract Ownership": note native `responses.create` is metered via the wildcard-plus-override pattern; other `responses.*` leaves stay unmetered_spend.
  - Present-tense snapshot only — replace, don't narrate history (history goes to CHANGELOG).
- Modify: `docs/plans/2026-08-07-ecosystem-admission-plan.md` — add a dated update note (2026-08-13+) to: §0 discovery row (~line 41), §1 Q2 item 6 (~line 76), Task 5 recipe (~line 485), the §4 risk row (~line 635), and the Task 3 superseded summary (~line 675): native OpenAI `responses.create`/`create(stream=True)` is now metered, so the `set_default_openai_api("chat_completions")` pin is OPTIONAL for native OpenAI clients. (After PR C lands, `.parse` and the `.stream()` helper are metered too — update the note again then to drop the residual caveat; only `beta.responses.*`/raw-response paths remain unmetered.)

**Interfaces:** none (docs only).

- [ ] **Step 1: Make the edits above** (each is fully specified; no new claims — every statement must match what Tasks 1-4 shipped)
- [ ] **Step 2: Verify the pinned-copy tests still pass**

Run: `uv run pytest tests/unit/test_coverage_manifest.py -v`
Expected: PASS (README claims and manifest guidance stay in sync)

- [ ] **Step 3: Full gate + commit**

Run: `make check && make test-unit`

```bash
git add README.md CHANGELOG.md src/solwyn/CLAUDE.md docs/plans/2026-08-07-ecosystem-admission-plan.md
git commit -m "docs: responses metering coverage, adapter notes, and Agents SDK pin update"
```

**→ Open PR B here** ("Meter native OpenAI responses.create"). PR body must include the `diff_surface_rules.py` output from Task 4 Step 7.

---

### Task 6: Meter `responses.parse` (native OpenAI)

**Files:**
- Modify: `src/solwyn/providers/openai.py` (`prepare_responses_call` gains keyword-only `leaf: str = "create"`), `src/solwyn/_proxies.py` (proxy `parse()` method), `src/solwyn/client.py` (thread `_responses_leaf` through `_intercepted_call` → dispatch), `src/solwyn/_surfaces.py` (regenerated), pins/README per the Task 4 pattern
- Test: `tests/unit/test_responses_call.py` (parse section), `tests/unit/test_providers/test_openai.py`

**Interfaces:**
- Consumes: everything Tasks 1–4 built. Estimation is unchanged — `parse` carries the same `input`/`instructions` content shape (`text_format` is a schema class, not content; the estimator never reads it).
- Produces: `prepare_responses_call(client, kwargs, *, is_streaming, leaf="create")` routing `leaf="parse"` → `client.responses.parse`; `_intercepted_call(_surface="responses", _responses_leaf="parse", ...)`; proxy `parse(**kwargs)`.

- [ ] **Step 1: Failing tests** — parse is intercepted, budget-checked, and settles off the returned `ParsedResponse.usage` (same flat shape); `parse` + `stream=True` is refused with `ConfigurationError` (the SDK's parse is non-streaming; the `stream()` helper is Task 7); compat-provider `parse` stays a warned raw pass-through; strict-mode acknowledgment of `responses.parse` on native OpenAI is now REJECTED as inapplicable (it is no longer an unmetered leaf — mirrors the existing acknowledgment-validation tests).
- [ ] **Step 2: Implement** — extend the seam with `leaf` (default `"create"` keeps Tasks 2–4 behavior byte-identical); add the proxy method (`_enforce_explicit_surface("responses.parse", WRAPPER)` first, no `background` concern); thread `_responses_leaf` through both dispatchers.
- [ ] **Step 3: Contract flip** — wildcard-plus-override for `responses.parse` exactly as Task 4 Step 4 did for `create` (metered, `source=both`, `usage_basis=provider`); re-pin openai digests; the acknowledgment examples already point at `responses.retrieve` (Decision 10) and must NOT move again.
- [ ] **Step 4: Full verification chain** (`make check && make test-unit && make check-surface-contract && make check-provider-surfaces && make test`) **+ commit** `feat: meter native OpenAI responses.parse`

### Task 7: Meter the `responses.stream()` helper

**Files:**
- Modify: `src/solwyn/_proxies.py` (proxy `stream()` + `_ResponsesStreamManagerWrapper` sync/async — or `stream.py` if the wrapper reads better beside the stream wrappers), `src/solwyn/client.py` (responses branch: `leaf="stream"` dispatch returns a MANAGER, wrapped instead of `_wrap_stream`), `src/solwyn/_surfaces.py` (regenerated), pins/README
- Test: `tests/unit/test_responses_call.py` (stream-helper section)

**Interfaces:**
- Consumes: Decision 12 design; Task 2 accumulator (terminal `response.completed` events flow through the manager's inner stream unchanged); Task 6's `leaf` seam (`leaf="stream"` → `client.responses.stream`).
- Produces: proxy `stream(**kwargs)` returning a `_ResponsesStreamManagerWrapper` that is a drop-in for the SDK manager.

- [ ] **Step 1: Failing tests** — `with s.responses.stream(...) as events:` iterates the fake terminal-event stream and settles exactly once off `response.completed`; abandonment (exit the with-block before the terminal event) settles a length-based estimate, never $0; `get_final_response()` is forwarded to the inner stream and does NOT double-settle; budget pre-flight runs once before the manager opens; compat providers keep the warned raw helper.
- [ ] **Step 2: Implement** per Decision 12 — the wrapper owns a `_settled` guard; `__enter__` enters the inner manager and wraps its stream for accumulator observation; `__exit__`/`close` settle-if-unsettled and always close the inner manager; `__getattr__` forwards everything else.
- [ ] **Step 3: Contract flip** — `responses.stream` → metered for native OpenAI (same pattern); re-pin.
- [ ] **Step 4: Full verification chain + commit** `feat: meter the native OpenAI responses.stream() helper`

### Task 8: Azure OpenAI parity via `CompatProfile.supports_responses`

**Files:**
- Modify: `src/solwyn/providers/openai_compatible.py` (`supports_responses: bool = False` on `CompatProfile`, `True` for azure_openai; adapter `prepare_responses_call` mirroring the native one, refusing when the profile flag is off), `src/solwyn/_base.py` or `client.py` (`_responses_preparer(adapter)` helper per Decision 11; cached_property + pre-check both switch to it), `src/solwyn/_surfaces.py` (regenerated), pins/README
- Test: `tests/unit/test_responses_call.py` (azure section), `tests/unit/test_openai_compatible_client.py`

- [ ] **Step 1: Failing tests** — an Azure-shaped client (`AzureOpenAI` class-name detection) gets the interception proxy: `create`/`parse`/`stream()` budget-checked and settled with `provider="azure_openai"` attribution; groq/generic compat clients still get warned raw pass-through (the flag defaults off); the capability helper is the single condition (assert proxy presence and pre-check agree for every named profile).
- [ ] **Step 2: Implement** — profile flag + compat `prepare_responses_call` + the shared `_responses_preparer` helper (replacing any `adapter.name == "openai"` condition from PR B).
- [ ] **Step 3: Contract flip** — move the azure_openai selectors for `responses` namespace + `create`/`parse`/`stream` leaves to metered/`source=both` (same wildcard-plus-override pattern; together/wildcard rules untouched). Digest guardrail update: openai AND azure_openai contexts move — any OTHER moving digest means the JSON edit touched too much; STOP.
- [ ] **Step 4: Full verification chain + commit** `feat: meter the Responses API on Azure OpenAI via CompatProfile capability flag`

### Task 9: Explicit `provider=` pin + E2E coverage

**Files:**
- Modify: `src/solwyn/_registry.py` (`build_runtimes` accepts `provider: str | None`; named lookup bypasses detection), `src/solwyn/client.py` (constructor kwarg on both classes), README.md (pin section)
- Modify: the E2E harness — `FakeProviderServer` gains a `/v1/responses` endpoint (non-streaming + SSE with a terminal `response.completed` carrying usage)
- Test: `tests/unit/test_registry.py` (or the registry's existing test home), E2E suite

- [ ] **Step 1: Failing unit tests** — `Solwyn(openai_client, provider="openai")` selects the native adapter even with a localhost `base_url` (detection bypassed); unknown provider name → `ConfigurationError(field="provider")`; a pinned pairing still passes through `_validate_surface_context` (an undeclared client-shape/mode pin fails loud, not silently).
- [ ] **Step 2: Implement the pin** — named registry lookup before detection; document that the pin asserts provider identity, it does not translate dialects.
- [ ] **Step 3: E2E** — fake `/v1/responses` endpoint; live-pipeline test: real `openai.OpenAI(base_url=fake)` + `provider="openai"` pin → assert budget check → dispatch → settlement for `create`, `create(stream=True)`, `parse`, and the `stream()` helper.
- [ ] **Step 4: Full gate + commit** `feat: explicit provider pin + Responses E2E coverage`

**→ Open PR C here** ("Responses coverage completion"). PR body includes the Task 6–8 `diff_surface_rules.py` outputs.

---

## Known Limitations / Follow-ups (documented, not blocking)

- **`background=True`** refused; poll-based settlement (retrieve-time usage) is future work.
- **Non-Azure compat responses** (groq, together, generic, …) unmetered by design; each future provider is one `supports_responses` profile flip + its contract rules (Task 8 built the seam).
- **`beta.responses.*`** and `with_raw_response.*`/`with_streaming_response.*` stay unmetered_spend guarded.

## Self-Review (performed at plan time)

- **Spec coverage:** wrapper interception ✓ (Task 4), budget pre-flight ✓ (Task 3 branch point 2), settlement via `report_settlement` ✓ (reused chat settlement, asserted in Task 3 tests), streaming ✓ (Tasks 2+3), capability-gated provider scope ✓ (Decision 1/3/11, Task 4 compat test, Task 8), contract via sanctioned workflow ✓ (Task 4 Step 4, reused by Tasks 6–8), eco-admission doc impact ✓ (Task 5), PR grouping with rationale ✓ (3 PRs).
- **2026-08-13 scope expansion (owner request):** PR C (Tasks 6–9: `parse`, `stream()` helper, Azure, `provider=` pin + E2E) folded into v1; Decision 10's acknowledgment target changed `responses.parse` → `responses.retrieve` so the copy migrates once; Decision 4 changed from drop-all to filtered defaults merge. The PR C task sketches are design-verified but their cited SDK shapes (`ResponseStreamManager` surface, azure detection fixture names) should be re-confirmed against the installed SDKs at execution time.
- **Type consistency:** `estimate_responses_content_length` / `_responses_output_bound` / `prepare_responses_call(client, kwargs, *, is_streaming)` / `_intercepted_call(_surface=...)` / `_sync_dispatch(..., surface=...)` names match across Tasks 1→4.
- **Verified against the codebase (2026-08-13):** all cited line numbers, rule ids (`surface.responses-create.unmetered_spend.f323258fe692`, `surface.chat-completions-create.metered.44de662a8c9b`, `surface.videos-create.*` pair), the `messages` conditional precedent, the accumulator's existing (flat-usage-only) Responses tests, the README acknowledgment pins, and the openai-context digests in `test_surface_context_pins.py`.
