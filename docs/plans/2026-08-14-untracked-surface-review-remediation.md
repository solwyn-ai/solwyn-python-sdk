# Untracked-Surface Review Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the advisory-report hot loop and warn/allow regression, silence no-loop advisory activation, and pin the shipping wire/lifecycle behavior identified by review.

**Architecture:** Keep structural observation state in the existing reporter-owned ledger and fix invalid-wire throttling at that shared boundary so sync and async completion hooks both quiesce. Split runtime structural-path acceptance from wire eligibility: malformed paths still fail closed, while identifier-clean depth/length overflow follows local posture and is excluded from advisory state. Exercise the remaining boundaries through real `Solwyn`/`AsyncSolwyn` instances wherever attribution or notifier wiring matters.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx, pytest, unittest.mock, threading, asyncio.

---

### Task 1: Throttle validation-failed advisory keys

**Files:**
- Modify: `src/solwyn/reporter.py:201-283`
- Test: `tests/unit/test_untracked_reporter.py`

**Step 1: Write the failing regression test**

- Put an out-of-vocabulary `capability_scope` into a real reporter's `_UntrackedReportState`.
- Run `_flush_untracked_reports()` once with the network boundary patched.
- Assert that no POST is attempted, the key is no longer due during the 15-minute interval, and it becomes due at the interval boundary.

**Step 2: Verify the test fails for the expected reason**

Run:

```bash
uv run pytest tests/unit/test_untracked_reporter.py -k validation_failed -q
```

Expected: failure because `reports_due()` remains `True` after the empty build.

**Step 3: Implement the minimal shared-state fix**

- On `UntrackedSurfaceReport` validation failure, record `last_attempted_at[key] = now` under the state lock.
- Fence the update by `generation` so a captured pre-fork build cannot mutate child state.
- Do not advance `last_sent_occurrences`; a future compatible vocabulary may retry after 15 minutes.

**Step 4: Verify green**

Run the focused test command again and confirm it passes.

### Task 2: Pin runtime/wire vocabulary parity

**Files:**
- Test: `tests/unit/test_untracked_reporter.py`

**Step 1: Add parity tests**

- Assert every `CapabilityScope.value` is admitted by `UntrackedScope`.
- Exercise every current `_client_shape()` branch with lightweight client types and assert every result is admitted by `UntrackedClientShape`.

**Step 2: Run the focused contract tests**

```bash
uv run pytest tests/unit/test_untracked_reporter.py -k 'wire_model or runtime_structural_vocabularies' -q
```

Expected: pass on the current vocabulary and fail if either runtime vocabulary grows without its wire literal.

### Task 3: Preserve warn/allow traversal for wire-ineligible structural paths

**Files:**
- Modify: `src/solwyn/_surfaces.py:25-26,353-364,432-440`
- Modify: `src/solwyn/_base.py:175-337`
- Test: `tests/unit/test_unmetered_posture.py:1055-1099`
- Modify: `CHANGELOG.md:53-60`

**Step 1: Write failing traversal tests**

- Keep malformed, private, and non-ASCII paths pinned to the hard `RuntimeError` boundary.
- Add overlong single-segment and nine-segment identifier-clean paths under `warn` and `allow`.
- Assert provider access forwards; `warn` logs once; `allow` is silent; and the wire-ineligible terminal path enters neither the process observation registry nor the origin reporter state.

**Step 2: Verify red**

```bash
uv run pytest tests/unit/test_unmetered_posture.py -k 'wire_ineligible or invalid_wire_surface' -q
```

Expected: current code raises `RuntimeError: invalid public surface path` for depth/length overflow.

**Step 3: Split structural validity from wire eligibility**

- Add an internal predicate that raises for malformed/non-ASCII/private paths and returns whether an identifier-clean path fits the 128-character/eight-segment wire boundary.
- Keep `_validate_surface_path()` strict for embedded rules and acknowledgment tokens.
- Make runtime rule resolution return no rule for structurally valid overflow paths.
- Skip advisory registry/notifier bookkeeping for those paths.
- Retain a bounded local warn-once digest latch so `warn` stays log-and-forward without retaining or sending the over-limit path; `allow` remains silent.

**Step 4: Verify green and document the compatibility behavior**

Run the focused posture tests and add a changelog note that wire-ineligible structural paths forward under `warn`/`allow` but are not externally reported.

### Task 4: Silence async advisory activation outside an event loop

**Files:**
- Modify: `src/solwyn/reporter.py:1768-1780`
- Test: `tests/unit/test_reporter_exit_flush.py`

**Step 1: Write the failing regression test**

- Construct an `AsyncSolwyn` client outside an event loop with `on_unmetered="allow"`.
- Resolve an untracked surface and assert zero reporter/base warning records.
- Invoke the blocking lifecycle drain and assert the advisory POST still occurs.

**Step 2: Verify red**

```bash
uv run pytest tests/unit/test_reporter_exit_flush.py -k no_loop_advisory -q
```

Expected: failure due to `reporter.enqueue_without_event_loop` and `_warned_no_loop=True`.

**Step 3: Implement the minimal activation guard**

- Probe `asyncio.get_running_loop()` in `_notify_untracked_observation()`.
- Return silently when no loop exists; do not call `_ensure_started()` and do not consume the spend-event warning latch.
- Leave queued spend-event behavior unchanged.

**Step 4: Verify green**

Run the focused test again and the existing no-loop spend-event warning test.

### Task 5: Cover the default client/wire boundary and attribution

**Files:**
- Test: `tests/unit/test_untracked_reporter.py`

**Step 1: Strengthen the shipping-default client test**

- Use `Solwyn(...)` without overriding `on_unmetered` or `report_untracked_surfaces`.
- Resolve/call the untracked surface through the client.
- Assert one local warning, a background advisory POST, exact auth headers, and the exact report body key set.

**Step 2: Rewrite cross-client attribution through clients**

- Construct two `Solwyn` clients and trigger observations only through the first client's public surface.
- Coalesce immediate workers at the I/O scheduler boundary, then drain each real reporter.
- Assert only the originating reporter posts for both same-key and different-key cases.

**Step 3: Run focused tests**

```bash
uv run pytest tests/unit/test_untracked_reporter.py -k 'shipping_defaults or originating_observations' -q
```

Expected: pass on current wiring; fail under notifier/header/cross-client mutations.

### Task 6: Pin cadence, close coordination, and failing exit POST

**Files:**
- Test: `tests/unit/test_untracked_reporter.py`
- Test: `tests/unit/test_reporter_exit_flush.py`

**Step 1: Add the cadence lower bound**

- After a successful attempt and a new observation, assert not due at `+899.999s` and due at `+900s`.

**Step 2: Add sync/async close coordination tests**

- Start an advisory worker/task whose POST is held at a gate.
- Assert `close()` waits for the in-flight advisory within its deadline.
- With a zero/expired deadline and due state, assert close does not start a final advisory cycle.

**Step 3: Add failing exit-drain advisory coverage**

- Make the blocking advisory POST raise.
- Assert the attempt is cadence-throttled, no successful-send baseline advances, and no spend-drop count is emitted.

**Step 4: Run focused tests**

```bash
uv run pytest tests/unit/test_untracked_reporter.py tests/unit/test_reporter_exit_flush.py -q
```

Expected: all advisory reporter and lifecycle tests pass.

### Task 7: Full verification and final review

**Files:**
- Review all modified files above.

**Step 1: Run targeted posture/reporter/lifecycle tests**

```bash
uv run pytest tests/unit/test_unmetered_posture.py tests/unit/test_untracked_reporter.py tests/unit/test_reporter_exit_flush.py tests/unit/test_async_metadata_reporter.py -q
```

**Step 2: Run the full unit suite**

```bash
make test-unit
```

**Step 3: Run the quality gate**

```bash
make check
```

**Step 4: Inspect the final diff and branch state**

```bash
git diff --check
git status --short
git diff -- src/solwyn/reporter.py src/solwyn/_surfaces.py src/solwyn/_base.py CHANGELOG.md tests/unit/test_unmetered_posture.py tests/unit/test_untracked_reporter.py tests/unit/test_reporter_exit_flush.py
```

Confirm every review item has either a code fix or a mutation-targeted test, with the pre-existing `docs/plans/2026-08-13-untracked-surface-signal-plan.md` left untouched.
