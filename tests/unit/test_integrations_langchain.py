"""Protocol-double tests for the LangChain/LangGraph attribution handler."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from contextvars import copy_context
from types import ModuleType
from typing import Any
from uuid import UUID, uuid4

import pytest

import solwyn
from solwyn._run import _capture_run_context


class _FakeBaseCallbackHandler:
    raise_error = False
    run_inline = False


@pytest.fixture
def handler_type(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    """Load the optional integration against a tiny langchain-core protocol."""
    callbacks = ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = _FakeBaseCallbackHandler  # type: ignore[attr-defined]
    package = ModuleType("langchain_core")
    package.callbacks = callbacks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_core", package)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks)
    sys.modules.pop("solwyn.integrations.langchain", None)
    module = importlib.import_module("solwyn.integrations.langchain")
    return module.SolwynRunScopeHandler


class _UnreadableContent:
    """Fails if the handler inspects, formats, or otherwise touches content."""

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"content-bearing callback value was accessed through {name}")


class _FakeCallbackManager:
    """Calls the handler using langchain-core's callback signatures."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler

    def start(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str = "RunnableSequence",
    ) -> None:
        self.handler.on_chain_start(
            {"name": name},
            _UnreadableContent(),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def end(self, *, run_id: UUID, parent_run_id: UUID | None = None) -> None:
        self.handler.on_chain_end(
            _UnreadableContent(),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def error(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
    ) -> None:
        self.handler.on_chain_error(
            RuntimeError("structural-test-error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def chat_model_start(self, *, run_id: UUID, parent_run_id: UUID | None) -> None:
        self.handler.on_chat_model_start(
            {},
            _UnreadableContent(),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def llm_start(self, *, run_id: UUID, parent_run_id: UUID | None) -> None:
        self.handler.on_llm_start(
            {},
            _UnreadableContent(),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def model_end(self, *, run_id: UUID, parent_run_id: UUID | None) -> None:
        self.handler.on_llm_end(
            _UnreadableContent(),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def model_error(self, *, run_id: UUID, parent_run_id: UUID | None) -> None:
        self.handler.on_llm_error(
            RuntimeError("structural-model-error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )


def _assert_clean(handler: Any) -> None:
    assert handler._handles == {}
    assert handler._pending_model_run_id is None
    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)


@pytest.mark.unit
def test_chain_callbacks_create_detached_scope_and_model_call_consumes_it_once(
    handler_type: type[Any],
) -> None:
    handler = handler_type(tags={"source": "langchain"})
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    model_run_id = uuid4()

    manager.start(run_id=chain_run_id, name="SupportChain")
    assert solwyn.current_run_context().id is None
    manager.chat_model_start(run_id=model_run_id, parent_run_id=chain_run_id)

    with handler.activate_model_call():
        active = solwyn.current_run_context()
        assert active.name == "langchain:SupportChain"
        assert active.tags == {"source": "langchain"}

    assert solwyn.current_run_context().id is None
    assert handler._pending_model_run_id is None
    with (
        pytest.raises(RuntimeError, match="no pending model-call nomination"),
        handler.activate_model_call(),
    ):
        pass

    manager.model_end(run_id=model_run_id, parent_run_id=chain_run_id)
    manager.end(run_id=chain_run_id)
    _assert_clean(handler)


@pytest.mark.unit
def test_nested_parent_run_id_maps_to_solwyn_parent(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    parent_callback_id = uuid4()
    child_callback_id = uuid4()
    model_run_id = uuid4()

    manager.start(run_id=parent_callback_id, name="Graph")
    parent_solwyn_id = handler._handles[parent_callback_id].run_id
    manager.start(
        run_id=child_callback_id,
        parent_run_id=parent_callback_id,
        name="Node",
    )
    manager.chat_model_start(run_id=model_run_id, parent_run_id=child_callback_id)

    with handler.activate_model_call():
        child_snapshot = _capture_run_context()

    assert child_snapshot[1] == "langchain:Node"
    assert child_snapshot[3] == parent_solwyn_id
    manager.model_end(run_id=model_run_id, parent_run_id=child_callback_id)
    manager.end(run_id=child_callback_id, parent_run_id=parent_callback_id)
    manager.end(run_id=parent_callback_id)
    _assert_clean(handler)


@pytest.mark.unit
def test_chain_error_finishes_detached_handle(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    run_id = uuid4()
    manager.start(run_id=run_id)

    manager.error(run_id=run_id)

    _assert_clean(handler)


@pytest.mark.unit
@pytest.mark.parametrize("terminal", ["end", "error"])
def test_chain_terminal_clears_its_unconsumed_model_nomination_before_finish(
    handler_type: type[Any],
    terminal: str,
) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    model_run_id = uuid4()
    manager.start(run_id=chain_run_id)
    manager.chat_model_start(run_id=model_run_id, parent_run_id=chain_run_id)

    getattr(manager, terminal)(run_id=chain_run_id)

    _assert_clean(handler)
    nomination = handler._model_nomination.get()
    assert nomination is not None
    assert nomination._handle is None
    with (
        pytest.raises(RuntimeError, match="no pending model-call nomination"),
        handler.activate_model_call(),
    ):
        pass


@pytest.mark.unit
def test_chain_terminal_during_active_call_defers_until_activation_exits(
    handler_type: type[Any],
) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    model_run_id = uuid4()
    manager.start(run_id=chain_run_id)
    manager.chat_model_start(run_id=model_run_id, parent_run_id=chain_run_id)

    with handler.activate_model_call():
        copy_context().run(manager.end, run_id=chain_run_id)
        assert chain_run_id in handler._handles

    _assert_clean(handler)
    copy_context().run(
        manager.model_end,
        run_id=model_run_id,
        parent_run_id=chain_run_id,
    )
    _assert_clean(handler)


@pytest.mark.unit
def test_unknown_non_null_chain_parent_fails_closed(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)

    with pytest.raises(RuntimeError, match="unknown parent chain run id"):
        manager.start(run_id=uuid4(), parent_run_id=uuid4(), name="Orphan")

    _assert_clean(handler)


@pytest.mark.unit
def test_unknown_model_parent_fails_closed(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)

    with pytest.raises(RuntimeError, match="unknown parent chain run id"):
        manager.chat_model_start(run_id=uuid4(), parent_run_id=uuid4())

    _assert_clean(handler)


@pytest.mark.unit
def test_unparented_model_start_fails_closed(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)

    with pytest.raises(RuntimeError, match="requires a parent chain run id"):
        manager.chat_model_start(run_id=uuid4(), parent_run_id=None)

    _assert_clean(handler)


@pytest.mark.unit
def test_activate_model_call_without_nomination_fails_closed(
    handler_type: type[Any],
) -> None:
    handler = handler_type()

    with (
        pytest.raises(RuntimeError, match="no pending model-call nomination"),
        handler.activate_model_call(),
    ):
        pass

    _assert_clean(handler)


@pytest.mark.unit
def test_out_of_order_end_warns_and_deferred_parent_finishes_after_child(
    handler_type: type[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    parent_callback_id = uuid4()
    child_callback_id = uuid4()
    manager.start(run_id=parent_callback_id, name="Graph")
    manager.start(
        run_id=child_callback_id,
        parent_run_id=parent_callback_id,
        name="Node",
    )

    with caplog.at_level(logging.WARNING, logger="solwyn.integrations.langchain"):
        manager.end(run_id=parent_callback_id)

    assert parent_callback_id in handler._handles
    assert child_callback_id in handler._handles
    assert str(parent_callback_id) in caplog.text

    manager.end(run_id=child_callback_id, parent_run_id=parent_callback_id)

    _assert_clean(handler)


@pytest.mark.unit
def test_copied_terminal_context_clears_shared_unconsumed_nomination(
    handler_type: type[Any],
) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    model_run_id = uuid4()
    manager.start(run_id=chain_run_id)
    manager.chat_model_start(run_id=model_run_id, parent_run_id=chain_run_id)

    copy_context().run(
        manager.model_error,
        run_id=model_run_id,
        parent_run_id=chain_run_id,
    )

    assert handler._pending_model_run_id is None
    with (
        pytest.raises(RuntimeError, match="no pending model-call nomination"),
        handler.activate_model_call(),
    ):
        pass
    manager.error(run_id=chain_run_id)
    _assert_clean(handler)


@pytest.mark.unit
def test_terminal_callback_only_clears_matching_nomination(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    model_run_id = uuid4()
    manager.start(run_id=chain_run_id)
    manager.chat_model_start(run_id=model_run_id, parent_run_id=chain_run_id)

    manager.model_error(run_id=uuid4(), parent_run_id=chain_run_id)

    assert handler._pending_model_run_id == model_run_id
    with handler.activate_model_call():
        assert solwyn.current_run_context().id == handler._handles[chain_run_id].run_id
    manager.model_end(run_id=model_run_id, parent_run_id=chain_run_id)
    manager.end(run_id=chain_run_id)
    _assert_clean(handler)


@pytest.mark.unit
async def test_concurrent_task_nominations_are_isolated(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    first_chain = uuid4()
    second_chain = uuid4()
    manager.start(run_id=first_chain, name="First")
    manager.start(run_id=second_chain, name="Second")
    expected = {
        first_chain: handler._handles[first_chain].run_id,
        second_chain: handler._handles[second_chain].run_id,
    }
    ready = 0
    lock = asyncio.Lock()
    release = asyncio.Event()

    async def invoke(chain_run_id: UUID) -> str | None:
        nonlocal ready
        model_run_id = uuid4()
        manager.chat_model_start(run_id=model_run_id, parent_run_id=chain_run_id)
        async with lock:
            ready += 1
            if ready == 2:
                release.set()
        await release.wait()
        with handler.activate_model_call():
            active_id = solwyn.current_run_context().id
            await asyncio.sleep(0)
            assert solwyn.current_run_context().id == active_id
        manager.model_end(run_id=model_run_id, parent_run_id=chain_run_id)
        manager.end(run_id=chain_run_id)
        return active_id

    first_id, second_id = await asyncio.gather(invoke(first_chain), invoke(second_chain))

    assert first_id == expected[first_chain]
    assert second_id == expected[second_chain]
    _assert_clean(handler)


@pytest.mark.unit
def test_model_error_after_activation_exception_leaves_no_scope_or_nomination(
    handler_type: type[Any],
) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    model_run_id = uuid4()
    manager.start(run_id=chain_run_id)
    manager.chat_model_start(run_id=model_run_id, parent_run_id=chain_run_id)

    with pytest.raises(RuntimeError, match="provider failed"), handler.activate_model_call():
        assert solwyn.current_run_context().id is not None
        raise RuntimeError("provider failed")

    assert solwyn.current_run_context().id is None
    manager.model_error(run_id=model_run_id, parent_run_id=chain_run_id)
    manager.error(run_id=chain_run_id)
    _assert_clean(handler)


@pytest.mark.unit
def test_unconsumed_second_model_start_is_rejected_as_ambiguous(
    handler_type: type[Any],
) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    first_model_run = uuid4()
    manager.start(run_id=chain_run_id)
    manager.chat_model_start(run_id=first_model_run, parent_run_id=chain_run_id)

    with pytest.raises(RuntimeError, match="unconsumed model-call nomination"):
        manager.chat_model_start(run_id=uuid4(), parent_run_id=chain_run_id)

    manager.model_error(run_id=first_model_run, parent_run_id=chain_run_id)
    manager.error(run_id=chain_run_id)
    _assert_clean(handler)


@pytest.mark.unit
def test_on_llm_start_fallback_uses_same_content_blind_nomination_seam(
    handler_type: type[Any],
) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain_run_id = uuid4()
    model_run_id = uuid4()
    manager.start(run_id=chain_run_id)

    manager.llm_start(run_id=model_run_id, parent_run_id=chain_run_id)
    with handler.activate_model_call():
        assert solwyn.current_run_context().id == handler._handles[chain_run_id].run_id

    manager.model_end(run_id=model_run_id, parent_run_id=chain_run_id)
    manager.end(run_id=chain_run_id)
    _assert_clean(handler)


@pytest.mark.unit
def test_unknown_terminal_callback_is_a_noop(handler_type: type[Any]) -> None:
    handler = handler_type()
    manager = _FakeCallbackManager(handler)

    manager.end(run_id=uuid4())
    manager.error(run_id=uuid4())
    manager.model_end(run_id=uuid4(), parent_run_id=None)

    _assert_clean(handler)


@pytest.mark.unit
def test_handler_is_inline_for_model_start_task_lineage(handler_type: type[Any]) -> None:
    handler = handler_type()

    assert handler.run_inline is True
    assert handler.raise_error is True
