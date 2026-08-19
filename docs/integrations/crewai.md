# CrewAI

`SolwynEventListener` provides content-free crew and task identity mapping. It
is attribution infrastructure, not an enforcement hook: CrewAI's native
LiteLLM path does not call a Solwyn-wrapped provider client and therefore does
not perform a Solwyn budget check.

Install the listener integration:

```sh
pip install "solwyn[crewai]"
```

CrewAI 0.157.0 is the tested minimum. The scheduled framework smoke also tests
the current CrewAI release so event and `BaseLLM` drift fails visibly.

## Structural attribution listener

Instantiate one listener before kicking off crews:

```python
from solwyn.integrations.crewai import SolwynEventListener

listener = SolwynEventListener(tags={"framework": "crewai"})
result = crew.kickoff()
```

The listener subscribes to exactly these structural events:

- crew kickoff started, completed, and failed; and
- task started, completed, and failed.

It deliberately does not subscribe to `LLMCall*`, stream-chunk, or tool-usage
events. Those CrewAI events can carry prompts, completions, tool arguments, or
exceptions. The listener also never reads the payload object supplied with a
structural event. From each source it reads only the structural crew/task ID,
the crew name, and the crew's ordered task IDs.

The generated names are `crew:{crew_name}` and `task:{zero_based_index}`. A
task identity is parented explicitly to its crew identity. CrewAI 1.x dispatches
task callbacks through a thread pool, so the listener uses detached handles
instead of keeping `ContextVar` tokens open across callbacks. It pre-registers
the known task handles during the awaited crew-start callback, then validates
the later task-start transition. Duplicate, unknown, and out-of-order source
IDs fail closed; terminal crew events deterministically clean up any remaining
structural handles. Cleanup warnings contain only generated run IDs.

The same `Crew` object can be kicked off sequentially. If CrewAI starts the
next kickoff before its prior crew-terminal callback runs, the listener creates
a fresh crew/task hierarchy and absorbs the pending old terminal structurally.
Reuse while any prior task lifecycle is still live is ambiguous (including a
true concurrent kickoff), so the source is blocked and provider activation
fails closed instead of attributing the new call to the old task generation.

The listener by itself does not send a model-call event to Solwyn. With native
CrewAI/LiteLLM, a real provider call can succeed while Solwyn observes zero
budget checks, confirmations, or spend events. This is intentional and is
pinned by the offline framework smoke.

## Budget enforcement with a custom BaseLLM

To enforce a budget, route the provider call through a Solwyn-wrapped client
and activate the task identity only around that call. The admitted recipe below
is sync, non-streaming, plain-text Chat Completions without CrewAI tools or
structured output. To satisfy CrewAI's `BaseLLM` contract, it necessarily
adapts message content into the provider request and reads the returned text.
It forwards that content only for the provider call and return value; it never
logs or persists either side. The shipped listener itself remains entirely
content-free.

Install the OpenAI provider extra as well:

```sh
pip install "solwyn[crewai,openai]"
```

```python
import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from openai import OpenAI

from solwyn import Solwyn
from solwyn.integrations.crewai import SolwynEventListener


class SolwynCrewAILLM(BaseLLM):
    def __init__(
        self,
        *,
        model: str,
        client: Solwyn,
        listener: SolwynEventListener,
    ) -> None:
        super().__init__(model=model)
        self._client = client
        self._listener = listener

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        **opaque_kwargs: Any,
    ) -> str:
        if from_task is None:
            raise RuntimeError("CrewAI did not provide a structural task source")
        if tools or available_functions or opaque_kwargs.get("response_model"):
            raise RuntimeError(
                "This CrewAI recipe admits plain-text calls without tools or schemas"
            )

        provider_messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        with self._listener.activate(from_task):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=provider_messages,
            )
        return response.choices[0].message.content or ""


listener = SolwynEventListener(tags={"framework": "crewai"})
client = Solwyn(
    OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    api_key=os.environ["SOLWYN_API_KEY"],
    provider="openai",
    budget_mode="hard_deny",
)
llm = SolwynCrewAILLM(
    model="gpt-4.1-mini",
    client=client,
    listener=listener,
)

agent = Agent(
    role="Analyst",
    goal="Answer the assigned request",
    backstory="A concise research analyst",
    llm=llm,
    allow_delegation=False,
)
task = Task(
    description="Answer the supplied request",
    expected_output="A concise answer",
    agent=agent,
)
crew = Crew(
    name="SupportCrew",
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
)

with client:
    result = crew.kickoff()
```

For the model call above, Solwyn performs the normal pre-flight budget check
and records the call under `task:0`, whose parent is `crew:SupportCrew`. If the
listener has not observed a valid crew lifecycle, `activate(from_task)` raises
rather than placing spend under an unrelated or root run.

Do not infer coverage for async crews, streaming, tools, structured output,
planning model calls, knowledge/embedding providers, or direct use of the raw
OpenAI client. Those paths are outside this recipe and need their own admitted,
tested Solwyn-wrapped provider boundary.
