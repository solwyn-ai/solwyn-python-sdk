# LangChain and LangGraph

`SolwynRunScopeHandler` adds content-free chain and graph-node attribution.
Budget enforcement still comes from the Solwyn-wrapped OpenAI clients injected
into `ChatOpenAI`; the callback handler never makes provider or control-plane
requests.

Install the integration and OpenAI extras plus the model package:

```sh
pip install "solwyn[langchain,openai]" langchain-openai langgraph
```

## Admitted ChatOpenAI path

Current `langchain-openai` calls `client.with_raw_response.create(...)` for a
normal non-streaming invocation. Solwyn intentionally treats raw-response
helpers as untracked surfaces. The small content-blind compatibility leaves
below adapt that call back to Solwyn's metered `chat.completions.create(...)`
without inspecting or logging its arguments or result. `OpaqueParsedResult`
transiently holds the returned provider object only so `parse()` can hand that
same untouched object back to `ChatOpenAI`; it does not inspect, log, or persist
the object.

```python
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from solwyn import AsyncSolwyn, Solwyn
from solwyn.integrations.langchain import SolwynRunScopeHandler


class OpaqueParsedResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def parse(self) -> Any:
        return self._value


class SyncMeteredChatCompletions:
    def __init__(self, delegate: Any, handler: SolwynRunScopeHandler) -> None:
        self._delegate = delegate
        self._handler = handler

    @property
    def with_raw_response(self) -> "SyncMeteredChatCompletions":
        return self

    def create(self, *opaque_args: Any, **opaque_kwargs: Any) -> OpaqueParsedResult:
        with self._handler.activate_model_call():
            value = self._delegate.create(*opaque_args, **opaque_kwargs)
        return OpaqueParsedResult(value)


class AsyncMeteredChatCompletions:
    def __init__(self, delegate: Any, handler: SolwynRunScopeHandler) -> None:
        self._delegate = delegate
        self._handler = handler

    @property
    def with_raw_response(self) -> "AsyncMeteredChatCompletions":
        return self

    async def create(
        self, *opaque_args: Any, **opaque_kwargs: Any
    ) -> OpaqueParsedResult:
        with self._handler.activate_model_call():
            value = await self._delegate.create(*opaque_args, **opaque_kwargs)
        return OpaqueParsedResult(value)


handler = SolwynRunScopeHandler(tags={"framework": "langchain"})

sync_solwyn = Solwyn(
    OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    api_key=os.environ["SOLWYN_API_KEY"],
)
async_solwyn = AsyncSolwyn(
    AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    api_key=os.environ["SOLWYN_API_KEY"],
)

model = ChatOpenAI(
    model="gpt-4.1-mini",
    # The leaves are the only ordinary invoke/ainvoke call sites. The roots
    # prevent ChatOpenAI from constructing separate unmetered clients.
    client=SyncMeteredChatCompletions(sync_solwyn.chat.completions, handler),
    root_client=sync_solwyn,
    async_client=AsyncMeteredChatCompletions(
        async_solwyn.chat.completions,
        handler,
    ),
    root_async_client=async_solwyn,
    use_responses_api=False,
    include_response_headers=False,
)

prompt = ChatPromptTemplate.from_template("Answer this request: {question}")
chain = (prompt | model).with_config(
    {"run_name": "support-chain", "callbacks": [handler]}
)

with sync_solwyn:
    result = chain.invoke({"question": "..."})


async def invoke_async() -> None:
    async with async_solwyn:
        await chain.ainvoke({"question": "..."})
```

The supported recipe is deliberately exact:

- `invoke` and `ainvoke` on basic, non-streaming Chat Completions;
- invocation through a chain or graph configured with this handler, so every
  model callback has a mapped structural parent;
- `use_responses_api=False`;
- no `response_format` or structured-output helper;
- `include_response_headers=False`; and
- one model invocation per callback nomination. Batch/generate APIs are not
  admitted by this one-shot handoff.

Streaming, structured output, the Responses API, and raw root-client helpers
take different `langchain-openai` call paths and are not covered by this
recipe. Do not infer budget enforcement for those paths.

The handler fails closed when callback hierarchy is missing or inconsistent.
Do not call the configured `ChatOpenAI` directly: an unparented
`model.invoke(...)` / `model.ainvoke(...)` has no chain or graph nomination and
raises instead of silently losing attribution. Invoke the handler-bound chain
or graph shown above.

## LangGraph hierarchy

Use the same handler in the compiled graph's config:

```python
graph = builder.compile().with_config(
    {"run_name": "support-graph", "callbacks": [handler]}
)
```

LangGraph supplies explicit callback parent IDs. The handler creates detached
Solwyn identities keyed by those IDs, so a model call in `draft`, for example,
is attributed to `langchain:draft` with the graph run as its parent. It never
derives names or tags from model inputs or outputs.

## Context and threading

LangChain runs async terminal callbacks in shielded copied tasks. For that
reason the handler does not keep `start_run()` scopes open between callbacks.
It creates detached identities and activates one only around the provider call
made by the compatibility leaf. The one-shot nomination is isolated by asyncio
task lineage and is cleared through a shared locked cell when terminal
callbacks run in copied contexts.

`ContextVar` state is not inherited reliably by arbitrary worker threads. Keep
the model-start callback and compatibility-leaf call in the same task lineage.
If application code submits work to an executor, use `solwyn.run_in_executor`
or `contextvars.copy_context().run(...)` as described in `solwyn._run`.
