"""Offline smoke coverage against real LangChain, ChatOpenAI, and LangGraph."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
import respx
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import AsyncOpenAI, OpenAI

from solwyn import AsyncSolwyn, Solwyn
from solwyn.integrations.langchain import SolwynRunScopeHandler

from . import (
    CONTROL_PLANE_URL,
    MODEL,
    SOLWYN_API_KEY,
    FrameworkSmokeHarness,
    make_offline_openai_client,
    make_offline_openai_sync_client,
)


class _OpaqueParsedResult:
    """Minimum raw-response facade used by ChatOpenAI's ordinary invoke path."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def parse(self) -> Any:
        return self._value


class _SyncMeteredChatCompletions:
    """Route ChatOpenAI's raw-response leaf through Solwyn's metered create."""

    def __init__(self, delegate: Any, handler: SolwynRunScopeHandler) -> None:
        self._delegate = delegate
        self._handler = handler

    @property
    def with_raw_response(self) -> _SyncMeteredChatCompletions:
        return self

    def create(self, *opaque_args: Any, **opaque_kwargs: Any) -> _OpaqueParsedResult:
        with self._handler.activate_model_call():
            value = self._delegate.create(*opaque_args, **opaque_kwargs)
        return _OpaqueParsedResult(value)


class _AsyncMeteredChatCompletions:
    """Async companion to the content-blind sync compatibility leaf."""

    def __init__(self, delegate: Any, handler: SolwynRunScopeHandler) -> None:
        self._delegate = delegate
        self._handler = handler

    @property
    def with_raw_response(self) -> _AsyncMeteredChatCompletions:
        return self

    async def create(self, *opaque_args: Any, **opaque_kwargs: Any) -> _OpaqueParsedResult:
        with self._handler.activate_model_call():
            value = await self._delegate.create(*opaque_args, **opaque_kwargs)
        return _OpaqueParsedResult(value)


def _client_options() -> dict[str, Any]:
    return {
        "api_key": SOLWYN_API_KEY,
        "api_url": CONTROL_PLANE_URL,
        "provider": "openai",
        "budget_mode": "hard_deny",
        "budget_check_cache_ttl": 0,
        "lease_enabled": False,
        "breaker_reporting_enabled": False,
        "reporter_flush_interval": 60.0,
    }


def _make_clients(router: respx.MockRouter) -> tuple[Solwyn, AsyncSolwyn]:
    sync_client = Solwyn(make_offline_openai_sync_client(router), **_client_options())
    async_client = AsyncSolwyn(make_offline_openai_client(router), **_client_options())
    assert isinstance(sync_client, OpenAI)
    assert isinstance(async_client, AsyncOpenAI)
    return sync_client, async_client


def _make_model(
    sync_client: Solwyn,
    async_client: AsyncSolwyn,
    handler: SolwynRunScopeHandler,
) -> ChatOpenAI:
    model = ChatOpenAI(
        model=MODEL,
        client=_SyncMeteredChatCompletions(sync_client.chat.completions, handler),
        root_client=sync_client,
        async_client=_AsyncMeteredChatCompletions(async_client.chat.completions, handler),
        root_async_client=async_client,
        use_responses_api=False,
        include_response_headers=False,
    )
    assert model.root_client is sync_client
    assert model.root_async_client is async_client
    return model


def _success_events(harness: FrameworkSmokeHarness) -> list[dict[str, Any]]:
    return [event for event in harness.events if event["status"] == "success"]


def _assert_handler_clean(handler: SolwynRunScopeHandler) -> None:
    assert handler._handles == {}
    assert handler._pending_model_run_id is None


def _assert_exact_call_associations(
    harness: FrameworkSmokeHarness,
    events: list[dict[str, Any]],
) -> None:
    assert len(harness.budget_checks) == len(harness.confirms) == len(events)
    checks_by_run = {check["agent_run_id"]: check for check in harness.budget_checks}
    events_by_call = {event["call_id"]: event for event in events}
    confirms_by_call = {confirm["call_id"]: confirm for confirm in harness.confirms}
    assert len(checks_by_run) == len(events)
    assert len(events_by_call) == len(events)
    assert events_by_call.keys() == confirms_by_call.keys()
    for event in events:
        check = checks_by_run[event["agent_run_id"]]
        confirm = confirms_by_call[event["call_id"]]
        assert check["agent_run_id"] == event["agent_run_id"]
        assert confirm["call_id"] == event["call_id"]


@pytest.mark.framework_smoke
def test_langchain_sync_chain_enforces_once_and_attributes_outer_chain(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock)
    sync_client, async_client = _make_clients(respx_mock)
    handler = SolwynRunScopeHandler(tags={"framework": "langchain"})
    model = _make_model(sync_client, async_client, handler)
    chain = (RunnableLambda(lambda value: [HumanMessage(content=value)]) | model).with_config(
        {"run_name": "sync-chain", "callbacks": [handler]}
    )

    try:
        with caplog.at_level(logging.WARNING), sync_client:
            result = chain.invoke("Offline sync request")
    finally:
        asyncio.run(async_client.close())

    assert result.content == "This is a fake response."
    assert harness.model_call_count == 1
    assert len(harness.budget_checks) == 1
    assert len(harness.confirms) == 1
    events = _success_events(harness)
    assert len(events) == 1
    _assert_exact_call_associations(harness, events)
    assert events[0]["agent_run_name"] == "langchain:sync-chain"
    assert events[0]["agent_run_id"] == harness.budget_checks[0]["agent_run_id"]
    assert "parent_agent_run_id" not in events[0]
    assert events[0]["tags"] == {"framework": "langchain"}
    assert all("untracked surface" not in record.getMessage() for record in caplog.records)
    _assert_handler_clean(handler)


@pytest.mark.framework_smoke
async def test_langchain_async_chain_enforces_once_and_attributes_outer_chain(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock)
    sync_client, async_client = _make_clients(respx_mock)
    handler = SolwynRunScopeHandler(tags={"framework": "langchain"})
    model = _make_model(sync_client, async_client, handler)
    chain = (RunnableLambda(lambda value: [HumanMessage(content=value)]) | model).with_config(
        {"run_name": "async-chain", "callbacks": [handler]}
    )

    try:
        with caplog.at_level(logging.WARNING):
            async with async_client:
                result = await chain.ainvoke("Offline async request")
    finally:
        sync_client.close()

    assert result.content == "This is a fake response."
    assert harness.model_call_count == 1
    assert len(harness.budget_checks) == 1
    assert len(harness.confirms) == 1
    events = _success_events(harness)
    assert len(events) == 1
    _assert_exact_call_associations(harness, events)
    assert events[0]["agent_run_name"] == "langchain:async-chain"
    assert events[0]["agent_run_id"] == harness.budget_checks[0]["agent_run_id"]
    assert "parent_agent_run_id" not in events[0]
    assert events[0]["tags"] == {"framework": "langchain"}
    assert all("untracked surface" not in record.getMessage() for record in caplog.records)
    _assert_handler_clean(handler)


@pytest.mark.framework_smoke
def test_langgraph_two_nodes_enforce_each_model_invoke_under_graph_hierarchy(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock, model_calls=2)
    sync_client, async_client = _make_clients(respx_mock)
    handler = SolwynRunScopeHandler(tags={"framework": "langgraph"})
    model = _make_model(sync_client, async_client, handler)

    def draft(state: MessagesState) -> dict[str, list[Any]]:
        return {"messages": [model.invoke(state["messages"])]}

    def review(state: MessagesState) -> dict[str, list[Any]]:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", END)
    graph = builder.compile().with_config({"run_name": "support-graph", "callbacks": [handler]})

    try:
        with caplog.at_level(logging.WARNING), sync_client:
            graph.invoke({"messages": [HumanMessage(content="Offline graph request")]})
    finally:
        asyncio.run(async_client.close())

    assert harness.model_call_count == 2
    assert len(harness.budget_checks) == 2
    assert len(harness.confirms) == 2
    events = _success_events(harness)
    assert len(events) == 2
    _assert_exact_call_associations(harness, events)
    assert {event["agent_run_name"] for event in events} == {
        "langchain:draft",
        "langchain:review",
    }
    assert {event["agent_run_id"] for event in events} == {
        check["agent_run_id"] for check in harness.budget_checks
    }
    assert len({event["call_id"] for event in events}) == 2
    assert {event["call_id"] for event in events} == {
        confirm["call_id"] for confirm in harness.confirms
    }
    parent_ids = {event["parent_agent_run_id"] for event in events}
    assert len(parent_ids) == 1
    assert None not in parent_ids
    assert all(event["tags"] == {"framework": "langgraph"} for event in events)
    assert all("untracked surface" not in record.getMessage() for record in caplog.records)
    _assert_handler_clean(handler)
