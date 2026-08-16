"""LangChain/LangGraph run-scope attribution.

Attribution only: budget enforcement rides the Solwyn-wrapped client injected
into the model class. This module binds content-bearing callback parameters but
never reads them. That boundary is enforced by the SDK privacy firewall.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "solwyn.integrations.langchain requires langchain-core; "
        "install with: pip install 'solwyn[langchain]'"
    ) from exc

import solwyn

logger = logging.getLogger(__name__)


class _ModelNomination:
    """One model callback's content-free, task-lineage activation handoff."""

    def __init__(
        self,
        model_run_id: UUID,
        chain_run_id: UUID,
        handle: solwyn.RunHandle,
    ) -> None:
        self._model_run_id = model_run_id
        self._chain_run_id = chain_run_id
        self._handle: solwyn.RunHandle | None = handle
        self._lock = Lock()

    @property
    def pending_run_id(self) -> UUID | None:
        with self._lock:
            if self._handle is None:
                return None
            return self._model_run_id

    def consume(self) -> tuple[solwyn.RunHandle | None, UUID]:
        with self._lock:
            handle = self._handle
            self._handle = None
            return handle, self._chain_run_id

    def clear_if_owned_by_chain(self, chain_run_id: UUID) -> bool:
        """Release this cell's handle when its owning chain terminates."""
        with self._lock:
            if chain_run_id != self._chain_run_id:
                return False
            self._handle = None
            return True

    def clear_if_matching(self, model_run_id: UUID) -> UUID | None:
        """Clear an unconsumed nomination and return its owning chain id."""
        with self._lock:
            if model_run_id != self._model_run_id:
                return None
            self._handle = None
            return self._chain_run_id


class SolwynRunScopeHandler(BaseCallbackHandler):
    """Map chain and graph-node model calls to detached Solwyn run identities."""

    run_inline = True
    raise_error = True

    def __init__(
        self,
        *,
        name_prefix: str = "langchain",
        tags: dict[str, str] | None = None,
    ) -> None:
        self._prefix = name_prefix
        self._tags = tags
        self._handles: dict[UUID, solwyn.RunHandle] = {}
        self._parents: dict[UUID, UUID | None] = {}
        self._children: dict[UUID, set[UUID]] = {}
        self._deferred: set[UUID] = set()
        self._state_lock = Lock()
        self._model_nomination: ContextVar[_ModelNomination | None] = ContextVar(
            f"solwyn_langchain_model_nomination_{id(self)}",
            default=None,
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create one detached identity without reading the chain input."""
        structural_name = "chain"
        if isinstance(name, str) and name:
            structural_name = name
        elif isinstance(serialized, dict):
            serialized_name = serialized.get("name")
            if isinstance(serialized_name, str) and serialized_name:
                structural_name = serialized_name

        with self._state_lock:
            if run_id in self._handles:
                raise RuntimeError("LangChain started a duplicate callback run id")
            parent_handle = self._handles.get(parent_run_id) if parent_run_id is not None else None
            if parent_run_id is not None and parent_handle is None:
                raise RuntimeError("LangChain chain callback has an unknown parent chain run id")
            if parent_handle is None:
                handle = solwyn.create_run(
                    f"{self._prefix}:{structural_name}",
                    tags=self._tags,
                )
            else:
                with parent_handle.activate():
                    handle = solwyn.create_run(
                        f"{self._prefix}:{structural_name}",
                        tags=self._tags,
                    )
            self._handles[run_id] = handle
            self._parents[run_id] = parent_run_id
            self._children[run_id] = set()
            if parent_handle is not None and parent_run_id is not None:
                self._children[parent_run_id].add(run_id)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Finish one detached identity without reading the chain output."""
        self._clear_chain_nomination(run_id)
        self._finish(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Finish one failed identity without inspecting the exception."""
        self._clear_chain_nomination(run_id)
        self._finish(run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Nominate the mapped parent scope without reading chat messages."""
        self._nominate(run_id, parent_run_id)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Nominate the mapped parent scope without reading prompts."""
        self._nominate(run_id, parent_run_id)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Clear only this model callback's unconsumed nomination."""
        self._clear_nomination(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Clear only this failed model callback's unconsumed nomination."""
        self._clear_nomination(run_id)

    @contextmanager
    def activate_model_call(self) -> Iterator[None]:
        """Activate the nominated chain/node identity for one provider call.

        The nomination is one-shot. A call without a mapped parent remains
        an error so a misconfigured model call cannot lose attribution.
        """
        nomination = self._model_nomination.get()
        if nomination is None:
            raise RuntimeError("LangChain has no pending model-call nomination")
        handle, chain_run_id = nomination.consume()
        if handle is None:
            raise RuntimeError("LangChain has no pending model-call nomination")
        try:
            with handle.activate():
                yield
        finally:
            self._retry_deferred(chain_run_id)

    @property
    def _pending_model_run_id(self) -> UUID | None:
        nomination = self._model_nomination.get()
        if nomination is None:
            return None
        return nomination.pending_run_id

    def _nominate(self, model_run_id: UUID, parent_run_id: UUID | None) -> None:
        if parent_run_id is None:
            raise RuntimeError("LangChain model callback requires a parent chain run id")
        with self._state_lock:
            handle = self._handles.get(parent_run_id)
        if handle is None:
            raise RuntimeError("LangChain model callback has an unknown parent chain run id")
        existing = self._model_nomination.get()
        if existing is not None and existing.pending_run_id is not None:
            raise RuntimeError(
                "LangChain delivered a second unconsumed model-call nomination; "
                "batch generate calls are unsupported"
            )
        self._model_nomination.set(_ModelNomination(model_run_id, parent_run_id, handle))

    def _clear_chain_nomination(self, chain_run_id: UUID) -> None:
        nomination = self._model_nomination.get()
        if nomination is not None:
            nomination.clear_if_owned_by_chain(chain_run_id)

    def _clear_nomination(self, model_run_id: UUID) -> None:
        nomination = self._model_nomination.get()
        if nomination is None:
            return
        chain_run_id = nomination.clear_if_matching(model_run_id)
        if chain_run_id is not None:
            self._retry_deferred(chain_run_id)

    def _finish(self, run_id: UUID) -> None:
        with self._state_lock:
            handle = self._handles.get(run_id)
            if handle is None:
                return
            if self._children[run_id]:
                self._defer(run_id)
                return
            try:
                handle.finish()
            except RuntimeError:
                self._defer(run_id)
                return
            parent_run_id = self._forget(run_id)
            self._drain_deferred_ancestors(parent_run_id)

    def _retry_deferred(self, run_id: UUID) -> None:
        with self._state_lock:
            if run_id not in self._deferred or self._children.get(run_id):
                return
            handle = self._handles.get(run_id)
            if handle is None:
                self._deferred.discard(run_id)
                return
            try:
                handle.finish()
            except RuntimeError:
                return
            parent_run_id = self._forget(run_id)
            self._drain_deferred_ancestors(parent_run_id)

    def _drain_deferred_ancestors(self, run_id: UUID | None) -> None:
        while run_id is not None and run_id in self._deferred:
            if self._children[run_id]:
                return
            handle = self._handles[run_id]
            try:
                handle.finish()
            except RuntimeError:
                return
            run_id = self._forget(run_id)

    def _defer(self, run_id: UUID) -> None:
        if run_id not in self._deferred:
            logger.warning(
                "solwyn.integrations.langchain: deferred structural scope close for %s",
                run_id,
            )
        self._deferred.add(run_id)

    def _forget(self, run_id: UUID) -> UUID | None:
        parent_run_id = self._parents.pop(run_id)
        self._handles.pop(run_id)
        self._children.pop(run_id)
        self._deferred.discard(run_id)
        if parent_run_id is not None and parent_run_id in self._children:
            self._children[parent_run_id].discard(run_id)
        return parent_run_id


__all__ = ["SolwynRunScopeHandler"]
