# OpenAI Agents SDK

Solwyn can be installed as the OpenAI Agents SDK's default client. The Agents
SDK makes its normal OpenAI call; the `AsyncSolwyn` wrapper performs budget
admission, failover, settlement, and run attribution around that call.

Install the provider extra and the framework:

```sh
pip install "solwyn[openai]" openai-agents
```

## Complete example

This example pins Chat Completions as an explicit, tested model path, creates a
workflow run around `Runner.run(...)`, and records each agent activation as a
stable child run. The small model wrapper is important: Agents may execute
successive turns for the same agent in different asyncio tasks, so a context
opened by an agent-start hook cannot safely remain open until agent-end.

```python
import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Any

from agents import (
    Agent,
    Model,
    ModelProvider,
    MultiProvider,
    RunConfig,
    RunHooks,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

import solwyn
from solwyn import AsyncSolwyn


class RetrySafeAsyncSolwyn(AsyncSolwyn):
    """Keep Agents-managed retries on Solwyn's metered client path."""

    def with_options(self, **kwargs: object) -> "RetrySafeAsyncSolwyn":
        if kwargs != {"max_retries": 0}:
            raise RuntimeError(
                "This Agents recipe only supports with_options(max_retries=0); "
                "configure every other provider option on AsyncOpenAI before wrapping it"
            )
        # Solwyn already forces max_retries=0 on the raw provider SDK for every
        # hop. Returning this wrapper preserves that setting and metering.
        return self


client = RetrySafeAsyncSolwyn(
    AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    api_key=os.environ["SOLWYN_API_KEY"],
)
set_default_openai_client(client)

# This recipe deliberately uses the path covered by Solwyn's Agents smoke test.
set_default_openai_api("chat_completions")

# Optional: OpenAI Agents tracing is separate from Solwyn reporting. Disable it
# when prompts and responses must not be exported by the framework itself.
set_tracing_disabled(True)


class SolwynRunHooks(RunHooks[None]):
    """Create stable logical agent runs and nominate one for each model turn."""

    def __init__(self, model_provider: ModelProvider) -> None:
        # Agent object identity distinguishes two same-named agent instances.
        self._handles: dict[int, solwyn.RunHandle] = {}
        self._pending_handle: ContextVar[solwyn.RunHandle | None] = ContextVar(
            "solwyn_agents_pending_handle",
            default=None,
        )
        self.model_provider = SolwynModelProvider(
            model_provider,
            self._pending_handle,
        )

    @staticmethod
    def _completed() -> asyncio.Future[None]:
        completed = asyncio.get_running_loop().create_future()
        completed.set_result(None)
        return completed

    # These overrides intentionally use `def`, not `async def`. Agents evaluates
    # each call before passing the returned awaitable to gather_with_cancel().
    # In particular, on_llm_start must set the pending ContextVar in the model
    # turn's task; doing that inside an async hook body would isolate the write
    # in a separately scheduled hook task.
    def on_agent_start(
        self, context: Any, agent: Agent[None]
    ) -> asyncio.Future[None]:
        agent_key = id(agent)
        if agent_key in self._handles:
            raise RuntimeError("Agents started an already-active agent instance")
        self._handles[agent_key] = solwyn.create_run(f"agent:{agent.name}")
        return self._completed()

    def on_llm_start(
        self,
        context: Any,
        agent: Agent[None],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> asyncio.Future[None]:
        # system_prompt and input_items may contain customer content. They are
        # required hook parameters but are deliberately never read or retained.
        handle = self._handles.get(id(agent))
        if handle is None:
            raise RuntimeError("Agents model turn has no active Solwyn agent run")
        self._pending_handle.set(handle)
        return self._completed()

    def _finish_agent(self, agent: Agent[None]) -> None:
        agent_key = id(agent)
        handle = self._handles.get(agent_key)
        if handle is not None:
            handle.finish()
            self._handles.pop(agent_key)

    def on_agent_end(
        self, context: Any, agent: Agent[None], output: Any
    ) -> asyncio.Future[None]:
        # `output` may contain customer content. It is deliberately not read,
        # stored, logged, or sent to Solwyn.
        self._finish_agent(agent)
        return self._completed()

    def on_handoff(
        self,
        context: Any,
        from_agent: Agent[None],
        to_agent: Agent[None],
    ) -> asyncio.Future[None]:
        # The outgoing model activation has already closed. Finish its logical
        # run before Agents starts the receiver, making the agents siblings.
        self._finish_agent(from_agent)
        return self._completed()

    def discard_abandoned(self) -> None:
        # An error may bypass on_agent_end. Finish every inactive detached run;
        # finish() raises instead of silently clearing any still-active run.
        self._pending_handle.set(None)
        for agent_key, handle in list(self._handles.items()):
            handle.finish()
            self._handles.pop(agent_key)


class SolwynModel(Model):
    """Activate a stable agent run only while delegated provider work executes."""

    def __init__(
        self,
        delegate: Model,
        pending_handle: ContextVar[solwyn.RunHandle | None],
    ) -> None:
        self._delegate = delegate
        self._pending_handle = pending_handle
        # Agents invokes the same per-turn Model wrapper for every Runner-managed
        # retry attempt, so consume the pending handle once and retain it here.
        self._handle: solwyn.RunHandle | None = None
        self._attempt_active = False

    def _consume_handle(self) -> solwyn.RunHandle:
        if self._handle is not None:
            if self._pending_handle.get() is not None:
                raise RuntimeError(
                    "Agents reused a bound Solwyn model for a new model turn"
                )
            return self._handle
        handle = self._pending_handle.get()
        self._pending_handle.set(None)
        if handle is None:
            raise RuntimeError("Agents model call has no pending Solwyn agent run")
        self._handle = handle
        return handle

    def _start_attempt(self) -> solwyn.RunHandle:
        if self._attempt_active:
            raise RuntimeError(
                "Concurrent calls through one Solwyn Agents model are unsupported"
            )
        handle = self._consume_handle()
        self._attempt_active = True
        return handle

    def _finish_attempt(self) -> None:
        if not self._attempt_active:
            raise RuntimeError("Solwyn Agents model attempt state is inconsistent")
        self._attempt_active = False

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        # Arguments can contain customer content. Forward them unchanged and do
        # not inspect, log, store, concatenate, or derive anything from them.
        handle = self._start_attempt()
        try:
            with handle.activate():
                return await self._delegate.get_response(*args, **kwargs)
        finally:
            self._finish_attempt()

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        handle = self._start_attempt()
        try:
            with handle.activate():
                stream = self._delegate.stream_response(*args, **kwargs)
        except BaseException:
            self._finish_attempt()
            raise
        return SolwynModelStream(stream, handle, self._finish_attempt)

    def get_retry_advice(self, request: Any) -> Any:
        return self._delegate.get_retry_advice(request)

    async def close(self) -> None:
        await self._delegate.close()

    async def _cleanup_on_run_end(self, owner: object) -> None:
        await self._delegate._cleanup_on_run_end(owner)


class SolwynModelStream(AsyncIterator[Any]):
    """Activate one logical run around each provider stream operation."""

    def __init__(
        self,
        delegate: AsyncIterator[Any],
        handle: solwyn.RunHandle,
        finish_attempt: Callable[[], None],
    ) -> None:
        self._delegate = delegate
        self._handle = handle
        self._finish_attempt = finish_attempt
        self._closed = False
        self._pulling = False

    def __aiter__(self) -> "SolwynModelStream":
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        if self._pulling:
            raise RuntimeError(
                "Concurrent iteration of one Solwyn Agents stream is unsupported"
            )
        self._pulling = True
        try:
            # Never keep a Solwyn ContextVar activation open across a yield.
            with self._handle.activate():
                event = await anext(self._delegate)
        except StopAsyncIteration:
            self._pulling = False
            await self.aclose()
            raise
        except BaseException:
            self._pulling = False
            await self.aclose()
            raise
        self._pulling = False
        return event

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._pulling:
            raise RuntimeError(
                "Cannot close a Solwyn Agents stream while iteration is active"
            )
        try:
            aclose = getattr(self._delegate, "aclose", None)
            if callable(aclose):
                with self._handle.activate():
                    await aclose()
        finally:
            self._closed = True
            self._finish_attempt()


class SolwynModelProvider(ModelProvider):
    """Wrap every model while otherwise preserving provider behavior."""

    def __init__(
        self,
        delegate: ModelProvider,
        pending_handle: ContextVar[solwyn.RunHandle | None],
    ) -> None:
        self._delegate = delegate
        self._pending_handle = pending_handle

    def get_model(self, model_name: str | None) -> Model:
        return SolwynModel(
            self._delegate.get_model(model_name),
            self._pending_handle,
        )

    async def aclose(self) -> None:
        await self._delegate.aclose()


triage_agent = Agent[None](
    name="Triage",
    instructions="Classify the support request and return the next action.",
    model="gpt-4.1-mini",
)


async def main() -> None:
    hooks = SolwynRunHooks(MultiProvider())
    run_config = RunConfig(
        model_provider=hooks.model_provider,
        tracing_disabled=True,
        trace_include_sensitive_data=False,
    )
    try:
        with solwyn.run("triage-workflow", tags={"team": "support"}):
            await Runner.run(
                triage_agent,
                "A customer cannot sign in.",
                hooks=hooks,
                run_config=run_config,
            )
    finally:
        hooks.discard_abandoned()
        await hooks.model_provider.aclose()
        await client.close()


asyncio.run(main())
```

`create_run(...)` snapshots the workflow parent, inherited tags, and one logical
agent-run ID without changing the current context. Every model turn then
reactivates that same ID inside whichever task Agents chose for the provider
call. This preserves one budget/lease ledger for a multi-turn agent instead of
silently creating a new run per LLM turn.

OpenAI Agents' Runner-managed retry helper reuses one resolved `Model` object
for every attempt, so `SolwynModel` caches that turn's handle after consuming
the pending ContextVar. On retry attempts, Agents asks its OpenAI client for
`with_options(max_retries=0)`. Solwyn already disables provider-SDK retries on
every hop; `RetrySafeAsyncSolwyn` therefore returns the same metered wrapper for
that exact request. It refuses timeout changes, different retry counts, or
additional options instead of returning an unmetered raw client. Configure all
such OpenAI options on `AsyncOpenAI(...)` before wrapping it. The recipe does
not import or depend on OpenAI Agents' private retry internals.

The synchronous hook bodies are a current OpenAI Agents API adaptation, not a
general Python callback convention. Agents currently evaluates the hook call
before passing its returned awaitable to `gather_with_cancel()`. The model
wrapper consumes the pending handle immediately and blindly delegates model
arguments. The scheduled framework smoke test covers this evaluation-order
assumption and will fail if it changes.

## Runner-managed retries

To let Agents retry transient provider failures, add its public retry settings
to the same `RunConfig`:

```python
from agents import ModelRetrySettings, ModelSettings, retry_policies

run_config = RunConfig(
    model_provider=hooks.model_provider,
    model_settings=ModelSettings(
        retry=ModelRetrySettings(
            max_retries=2,
            policy=retry_policies.network_error(),
        )
    ),
    tracing_disabled=True,
    trace_include_sensitive_data=False,
)
```

Every actual attempt receives its own Solwyn budget preflight and metadata
event under the same stable logical agent run. A connection failure releases
its reservation and emits an error event; because it has no provider-reported
usage, only a later successful attempt produces a usage confirm.

## Chat Completions and Responses

The Agents SDK defaults to the Responses API. Current Solwyn releases meter
native OpenAI and Azure OpenAI `responses.create(...)`, `responses.parse(...)`,
and new-response `responses.stream(...)` calls, including
`create(stream=True)`. You may omit `set_default_openai_api(...)` to use that
default when your Agents path stays on those supported leaves.

This recipe still pins `"chat_completions"` because it is a deliberate,
portable path choice and the exact path exercised by the framework smoke test.
Beta/raw-response helpers, other Responses leaves, and non-Azure
OpenAI-compatible Responses paths remain subject to Solwyn's `on_unmetered`
posture. Use the pin when you require the recipe's tested coverage boundary.

## Budget denial

If the control plane denies the preflight in `hard_deny` mode, Solwyn raises
`BudgetExceededError` before calling OpenAI. The Agents runner does not convert
it to a framework-specific exception:

```python
from solwyn import BudgetExceededError

hooks = SolwynRunHooks(MultiProvider())
run_config = RunConfig(
    model_provider=hooks.model_provider,
    tracing_disabled=True,
    trace_include_sensitive_data=False,
)
try:
    with solwyn.run("triage-workflow", tags={"team": "support"}):
        result = await Runner.run(
            triage_agent,
            "...",
            hooks=hooks,
            run_config=run_config,
        )
except BudgetExceededError:
    # Apply your application's deny behavior, or omit this block to propagate.
    raise
finally:
    hooks.discard_abandoned()
```

Do not catch `BudgetExceededError` as a provider outage or retry it outside
Solwyn; that would defeat the hard budget decision.

## Provider fallback

Fallback stays in the Solwyn client configuration. This snippet continues from
the complete example above and reuses its `RetrySafeAsyncSolwyn` subclass. The
Agents SDK continues to see one default client while Solwyn selects a healthy
configured hop:

```python
import os

from agents import set_default_openai_api, set_default_openai_client
from openai import AsyncOpenAI

openrouter = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
client = RetrySafeAsyncSolwyn(
    AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    api_key=os.environ["SOLWYN_API_KEY"],
    fallback=[
        (openrouter, "openai/gpt-4.1-mini", {}, "openrouter"),
    ],
)
set_default_openai_client(client)
set_default_openai_api("chat_completions")
```

The fallback entry owns its client, model, optional default parameters, and
provider identity. Solwyn remains the only component that reports settlement,
so a successful fallback does not double-count the Agents run.

## Streaming

`Runner.run_streamed(...)` is supported on the pinned Chat Completions path:

```python
hooks = SolwynRunHooks(MultiProvider())
run_config = RunConfig(
    model_provider=hooks.model_provider,
    tracing_disabled=True,
    trace_include_sensitive_data=False,
)
try:
    with solwyn.run("triage-workflow", tags={"team": "support"}):
        result = Runner.run_streamed(
            triage_agent,
            "...",
            hooks=hooks,
            run_config=run_config,
        )
        async for event in result.stream_events():
            ...
finally:
    hooks.discard_abandoned()
```

Solwyn returns its own async stream wrapper rather than OpenAI's exact stream
class. The Agents SDK consumes streams by asynchronous iteration, which the
wrapper preserves and the smoke test verifies. Code that requires the
provider's exact stream type is not supported. Consume `stream_events()` to
completion so terminal usage, settlement, and `on_agent_end` all run.
