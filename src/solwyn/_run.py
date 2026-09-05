"""Agent-run scopes for context managers and begin/end-shaped adapters.

Binds an active run id/name plus optional explicit customer tags to a
``ContextVar`` for the duration of a scope. Cost events emitted inside the
scope carry a copied snapshot in the wire payload; events emitted outside are
unscoped. Tags are never inferred from prompts or responses.

The contextvar is the scope integration seam. Each intercepted call captures
an immutable-by-convention snapshot and threads it through deferred reporting.
``contextvars`` propagation guarantees correct scoping across asyncio tasks.
Threads and ``ThreadPoolExecutor`` workers do not
inherit the active run reliably; use ``run_in_executor(...)`` or
``contextvars.copy_context().run(...)`` when submitting threaded work.
Do not open a run scope inside an async generator; async generator yields
share the consumer's context and would leak the generator's run into the
consumer body.

``start_run(...)`` exposes the same scope machinery to framework adapters
whose boundaries arrive as separate callbacks. Its ``RunHandle.finish()``
must run in the same context that called ``start_run(...)``.

``create_run(...)`` instead snapshots a detached logical identity without
changing the current context. Its handle can bind that identity repeatedly in
different tasks or threads through short-lived ``activate()`` scopes.

This module never touches prompt or response content.
"""

from __future__ import annotations

import inspect
import sys
import unicodedata
import uuid
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future
from contextlib import AbstractAsyncContextManager, AbstractContextManager, suppress
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass
from threading import Lock
from types import FrameType, TracebackType
from typing import NamedTuple, ParamSpec, TypeVar

from solwyn._constants import (
    AGENT_RUN_NAME_MAX_LENGTH,
    TAG_KEY_MAX_LENGTH,
    TAG_VALUE_MAX_LENGTH,
    TAGS_MAX_KEYS,
)
from solwyn._lifecycle import _surrender_run
from solwyn.exceptions import SolwynTagsClampedWarning

_P = ParamSpec("_P")
_T = TypeVar("_T")
_RunContextSnapshot = tuple[
    str | None,
    str | None,
    dict[str, str] | None,
    str | None,
]


class RunContext(NamedTuple):
    """Public snapshot of the active agent-run scope."""

    id: str | None
    name: str | None
    tags: dict[str, str] | None


@dataclass(frozen=True)
class _RunIdentity:
    """Immutable logical identity reused by detached run activations."""

    run_id: str
    name: str
    tags: tuple[tuple[str, str], ...] | None
    parent_run_id: str | None

    def active_value(self) -> tuple[str, str, dict[str, str] | None]:
        """Return a fresh ContextVar payload for one activation."""
        tags = dict(self.tags) if self.tags is not None else None
        return (self.run_id, self.name, tags)


# Single contextvar holding either None or one run payload. Storing all values
# together makes the swap atomic across async task transitions.
_active_run: ContextVar[tuple[str, str, dict[str, str] | None] | None] = ContextVar(
    "solwyn_active_run", default=None
)

_DISALLOWED_NAME_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


@dataclass(frozen=True)
class _RunFrame:
    scope_id: int
    run_id: str
    parent_run_id: str | None
    token: Token[tuple[str, str, dict[str, str] | None] | None]
    prior_active: tuple[str, str, dict[str, str] | None] | None


_run_frames: ContextVar[tuple[_RunFrame, ...]] = ContextVar("solwyn_run_frames", default=())


def _new_run_id() -> str:
    """Generate a fresh agent_run_id.

    Format: ``run_`` + UUID4 canonical string.
    Comfortably under the 255-char wire cap.
    """
    return f"run_{uuid.uuid4()}"


def current_run() -> tuple[str | None, str | None]:
    """Return the active ``(agent_run_id, agent_run_name)`` or ``(None, None)``.

    Read at metadata-event build time. Returning a tuple (rather than the
    raw ``ContextVar`` value) lets callers unpack without a None-check.
    """
    active = _active_run.get()
    if active is None:
        return (None, None)
    return (active[0], active[1])


def current_run_context() -> RunContext:
    """Return the active run id, name, and a defensive copy of its tags."""
    active = _active_run.get()
    if active is None:
        return RunContext(None, None, None)
    tags = dict(active[2]) if active[2] is not None else None
    return RunContext(active[0], active[1], tags)


def _copy_tags(
    tags: object | None,
    *,
    parameter: str,
) -> dict[str, str] | None:
    """Validate and copy one explicit tag mapping without normalization."""
    if tags is None:
        return None
    if not isinstance(tags, Mapping):
        raise TypeError(f"{parameter} requires a mapping of string keys to string values")
    if len(tags) > TAGS_MAX_KEYS:
        raise ValueError(f"{parameter} allows at most {TAGS_MAX_KEYS} keys")

    copied: dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str):
            raise TypeError(f"{parameter} keys must be strings")
        if not key:
            raise ValueError(f"{parameter} keys must be non-empty")
        if len(key) > TAG_KEY_MAX_LENGTH:
            raise ValueError(f"{parameter} key exceeds max length {TAG_KEY_MAX_LENGTH}")
        if "\x00" in key:
            raise ValueError(f"{parameter} keys must not contain NUL characters")
        if not isinstance(value, str):
            raise TypeError(f"{parameter} values must be strings")
        if len(value) > TAG_VALUE_MAX_LENGTH:
            raise ValueError(f"{parameter} value exceeds max length {TAG_VALUE_MAX_LENGTH}")
        if "\x00" in value:
            raise ValueError(f"{parameter} values must not contain NUL characters")
        copied[key] = value
    return copied or None


def _capture_run_context(
    per_call_tags: object | None = None,
    *,
    default_tags: object | None = None,
) -> _RunContextSnapshot:
    """Capture the active run, immediate parent, and merged tag layers."""
    active = _active_run.get()
    run_id: str | None = None
    run_name: str | None = None
    parent_run_id: str | None = None
    defaults = _copy_tags(default_tags, parameter="client tags")
    scope_tags: dict[str, str] | None = None
    if active is not None:
        run_id, run_name, scope_tags = active
        frames = _run_frames.get()
        if frames:
            parent_run_id = frames[-1].parent_run_id
    override = _copy_tags(per_call_tags, parameter="solwyn_tags")

    merged: dict[str, str] = {}
    clamped = False
    for layer in (override, scope_tags, defaults):
        if layer is None:
            continue
        for key, value in layer.items():
            if key in merged:
                continue
            if len(merged) == TAGS_MAX_KEYS:
                clamped = True
                continue
            merged[key] = value

    if clamped:
        with suppress(Exception):
            warnings.warn(
                f"merged tags exceed {TAGS_MAX_KEYS} keys; lower-priority tags were dropped",
                SolwynTagsClampedWarning,
                stacklevel=2,
            )
    return (run_id, run_name, merged or None, parent_run_id)


def _name_has_disallowed_char(name: str) -> bool:
    return any(unicodedata.category(char) in _DISALLOWED_NAME_CATEGORIES for char in name)


def _validate_run_definition(
    name: object,
    tags: object | None,
    *,
    api_name: str,
) -> tuple[str, dict[str, str] | None]:
    """Validate one run name/tag definition without reading active context."""
    if not isinstance(name, str):
        raise TypeError(f"{api_name}(name) requires str, got {type(name).__name__}")
    if not name.strip():
        raise ValueError(f"{api_name}(name) requires a non-empty name")
    if _name_has_disallowed_char(name):
        raise ValueError(
            f"{api_name}(name) must not contain control characters, "
            "format, or line-separator characters"
        )
    if len(name) > AGENT_RUN_NAME_MAX_LENGTH:
        raise ValueError(
            f"{api_name}(name) exceeds max length {AGENT_RUN_NAME_MAX_LENGTH} (got {len(name)})"
        )
    copied_tags = _copy_tags(tags, parameter=f"{api_name}(tags)")
    return name, copied_tags


def _snapshot_run_identity(
    name: str,
    tags: Mapping[str, str] | None,
    *,
    inherit_tags: bool,
) -> _RunIdentity:
    """Create one logical identity from the current parent and tag context."""
    prior_active = _active_run.get()
    frames = _run_frames.get()
    parent_run_id = frames[-1].run_id if frames else None
    scope_tags = dict(tags or {})
    if inherit_tags and prior_active is not None and prior_active[2] is not None:
        for key, value in prior_active[2].items():
            scope_tags.setdefault(key, value)
    return _RunIdentity(
        run_id=_new_run_id(),
        name=name,
        tags=tuple(scope_tags.items()) or None,
        parent_run_id=parent_run_id,
    )


def _called_from_async_generator() -> bool:
    """Return True when ``run()`` is being entered inside an async generator."""
    frame: FrameType | None = sys._getframe(2)
    try:
        while frame is not None:
            if frame.f_code.co_flags & inspect.CO_ASYNC_GENERATOR:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


class _RunScope(AbstractContextManager[str], AbstractAsyncContextManager[str]):
    """Context manager returned by ``solwyn.run(name)``.

    Supports both ``with`` and ``async with``. Nested scopes inherit tags by
    default, with inner values winning only on conflicting keys.
    The outer scope is restored automatically on exit via ``ContextVar.reset``.
    """

    def __init__(
        self,
        name: str,
        tags: Mapping[str, str] | None = None,
        *,
        inherit_tags: bool = True,
    ) -> None:
        self._name, self._tags = _validate_run_definition(
            name,
            tags,
            api_name="solwyn.run",
        )
        self._inherit_tags = inherit_tags
        self._scope_id = id(self)

    def _enter(self) -> str:
        if _called_from_async_generator():
            raise TypeError(
                "solwyn.run(...) inside async generators is not supported; "
                "open the scope in the consumer or await the generator inside an outer scope"
            )
        prior_active = _active_run.get()
        frames = _run_frames.get()
        identity = _snapshot_run_identity(
            self._name,
            self._tags,
            inherit_tags=self._inherit_tags,
        )
        token = _active_run.set(identity.active_value())
        _run_frames.set(
            (
                *frames,
                _RunFrame(
                    scope_id=self._scope_id,
                    run_id=identity.run_id,
                    parent_run_id=identity.parent_run_id,
                    token=token,
                    prior_active=prior_active,
                ),
            )
        )
        return identity.run_id

    def _exit(self, *, require_same_context: bool = False) -> None:
        """Pop this scope's frame, and release the leases the run held.

        The release is the run's own (``frame.run_id``), never a nested or
        outer one, and it happens only where a frame is actually popped: an
        exit that raises has changed nothing to release. ``require_same_context``
        is ``RunHandle.finish()``'s path — it owns the release itself, after
        its handle lock, so nothing here fires twice for one run.
        """
        frames = _run_frames.get()
        if not frames:
            if require_same_context:
                raise RuntimeError(
                    "RunHandle.finish() must be called in the same context where "
                    "start_run() created it"
                )
            return
        frame = frames[-1]
        if frame.scope_id != self._scope_id:
            raise RuntimeError("solwyn.run scopes must exit in LIFO order")
        if require_same_context:
            try:
                _active_run.reset(frame.token)
            except ValueError as exc:
                raise RuntimeError(
                    "RunHandle.finish() must be called in the same context where "
                    "start_run() created it"
                ) from exc
            _run_frames.set(frames[:-1])
            return
        try:
            _active_run.reset(frame.token)
        except ValueError:
            # Async generator finalizers can run in a different Context than
            # __aenter__. A token cannot be reset from there, so avoid
            # surfacing a cleanup-time exception. Restore the prior value in
            # the finalizer context as a best effort; entering inside async
            # generators is rejected to prevent contaminating the consumer.
            _active_run.set(frame.prior_active)
        finally:
            _run_frames.set(frames[:-1])
        # The run is over — including when the block is leaving on an
        # exception. Hand its float back instead of holding it to the deadline.
        _surrender_run(frame.run_id)

    def __enter__(self) -> str:
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit()

    async def __aenter__(self) -> str:
        return self._enter()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit()


class RunHandle:
    """Lifecycle handle for framework adapter run identities.

    A handle from :func:`start_run` owns one ContextVar-backed scope and must
    finish in its creating context. A handle from :func:`create_run` owns a
    detached identity that can be bound through reusable :meth:`activate`
    scopes in different contexts before it is finished.
    """

    def __init__(
        self,
        scope: _RunScope | None,
        run_id: str,
        *,
        identity: _RunIdentity | None = None,
    ) -> None:
        if (scope is None) == (identity is None):
            raise RuntimeError("run handle requires exactly one scope or detached identity")
        self._scope = scope
        self._run_id = run_id
        self._identity = identity
        self._finished = False
        self._active_activations = 0
        self._state_lock = Lock()

    @property
    def run_id(self) -> str:
        """Return the stable id for this run scope."""
        return self._run_id

    def finish(self) -> None:
        """Finish this logical run, restoring a started scope when applicable.

        Finishing ends the run for both handle flavours, so both release the
        leases it held — outside the handle lock, and only once the finish has
        actually taken (a refused finish leaves the run running). An
        ``activate()`` scope deliberately does NOT: a detached identity is
        re-entered by design, and releasing at each activation would trade one
        blocking grant per binding for nothing.
        """
        with self._state_lock:
            if self._finished:
                raise RuntimeError(f"run handle {self._run_id!r} already finished")
            if self._active_activations:
                raise RuntimeError(
                    f"run handle {self._run_id!r} cannot finish while activations are still active"
                )
            if self._scope is not None:
                self._scope._exit(require_same_context=True)
            self._finished = True
        _surrender_run(self._run_id)

    def activate(self) -> _RunActivation:
        """Bind a detached logical identity in the caller's context."""
        if self._identity is None:
            raise RuntimeError("RunHandle.activate() requires a handle created by create_run()")
        return _RunActivation(self)

    def _reserve_activation(self) -> _RunIdentity:
        with self._state_lock:
            if self._finished:
                raise RuntimeError(f"run handle {self._run_id!r} already finished")
            if self._identity is None:
                raise RuntimeError("RunHandle.activate() requires a handle created by create_run()")
            self._active_activations += 1
            return self._identity

    def _release_activation(self) -> None:
        with self._state_lock:
            if self._active_activations <= 0:
                raise RuntimeError("run handle activation count is inconsistent")
            self._active_activations -= 1


class _RunActivation(AbstractContextManager[str], AbstractAsyncContextManager[str]):
    """One strict ContextVar binding of a detached logical run identity."""

    def __init__(self, handle: RunHandle) -> None:
        self._handle = handle
        self._scope_id = id(self)
        self._entered = False

    def _enter(self) -> str:
        frames = _run_frames.get()
        if any(frame.run_id == self._handle.run_id for frame in frames):
            raise RuntimeError(
                f"run handle {self._handle.run_id!r} is already active in this context"
            )
        identity = self._handle._reserve_activation()
        prior_active = _active_run.get()
        token: Token[tuple[str, str, dict[str, str] | None] | None] | None = None
        try:
            token = _active_run.set(identity.active_value())
            _run_frames.set(
                (
                    *frames,
                    _RunFrame(
                        scope_id=self._scope_id,
                        run_id=identity.run_id,
                        parent_run_id=identity.parent_run_id,
                        token=token,
                        prior_active=prior_active,
                    ),
                )
            )
        except BaseException:
            if token is not None:
                _active_run.reset(token)
            self._handle._release_activation()
            raise
        self._entered = True
        return identity.run_id

    def _exit(self) -> None:
        frames = _run_frames.get()
        matching_index = next(
            (index for index, frame in enumerate(frames) if frame.scope_id == self._scope_id),
            None,
        )
        if matching_index is None:
            raise RuntimeError(
                "RunHandle activation must exit in the same context where it was entered"
            )
        if matching_index != len(frames) - 1:
            raise RuntimeError("solwyn.run scopes must exit in LIFO order")
        frame = frames[-1]
        try:
            _active_run.reset(frame.token)
        except ValueError as exc:
            raise RuntimeError(
                "RunHandle activation must exit in the same context where it was entered"
            ) from exc
        _run_frames.set(frames[:-1])
        self._handle._release_activation()
        self._entered = False

    def __enter__(self) -> str:
        if self._entered:
            raise RuntimeError("RunHandle activation is already entered")
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit()

    async def __aenter__(self) -> str:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit()


def start_run(
    name: str,
    tags: Mapping[str, str] | None = None,
    *,
    inherit_tags: bool = True,
) -> RunHandle:
    """Open a run scope for begin/end-shaped framework callbacks."""
    scope = _RunScope(name, tags, inherit_tags=inherit_tags)
    return RunHandle(scope, scope._enter())


def create_run(
    name: str,
    tags: Mapping[str, str] | None = None,
    *,
    inherit_tags: bool = True,
) -> RunHandle:
    """Create a detached logical run without changing the current context.

    The returned handle snapshots its id, name, inherited tags, and parent at
    creation. Use ``with handle.activate():`` around work in any task or thread;
    each activation binds that same logical identity with a fresh ContextVar
    token. Call :meth:`RunHandle.finish` after every activation has exited.
    """
    validated_name, copied_tags = _validate_run_definition(
        name,
        tags,
        api_name="solwyn.create_run",
    )
    identity = _snapshot_run_identity(
        validated_name,
        copied_tags,
        inherit_tags=inherit_tags,
    )
    return RunHandle(None, identity.run_id, identity=identity)


def run_in_executor(
    executor: Executor,
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> Future[_T]:
    """Submit ``fn`` to an executor with the active ``solwyn.run(...)`` tag preserved.

    ``ThreadPoolExecutor`` does not copy ``ContextVar`` values into worker
    threads. This helper wraps ``fn`` in ``copy_context().run`` so the
    active run id propagates to the worker.

    Returns:
        The executor's ``concurrent.futures.Future[T]``, not an awaitable.
        Call ``.result()`` to wait for the value. In asyncio code, bridge it
        with ``asyncio.wrap_future(future)``.

    Args:
        executor: Any ``concurrent.futures.Executor``.
        fn: Callable to run in a worker.
        *args, **kwargs: Forwarded to ``fn`` through the copied context.

    Note:
        If ``executor.shutdown(cancel_futures=True)`` cancels work before it
        starts, ``future.result()`` raises ``CancelledError``.
    """
    ctx = copy_context()

    def run_with_context() -> _T:
        return ctx.run(fn, *args, **kwargs)

    return executor.submit(run_with_context)


def run(
    name: str,
    tags: Mapping[str, str] | None = None,
    *,
    inherit_tags: bool = True,
) -> _RunScope:
    """Open an agent-run scope.

    Cost events emitted inside the scope are tagged with a fresh stable
    ``agent_run_id``, the provided ``name``, and a copied snapshot of optional
    explicit customer ``tags``. Use as a sync or async context manager::

        with solwyn.run("nightly-batch", tags={"team": "research"}) as run_id:
            client.chat.completions.create(...)

        async with solwyn.run("ingest-job") as run_id:
            await aclient.chat.completions.create(...)

    Nested runs inherit outer tags additively by default. Inner values win
    only where they reuse an outer key. Pass ``inherit_tags=False`` to start
    a fresh tag scope. The inner run always gets its own id, and the outer
    context is restored after the inner ``__exit__``. Sequential scopes get
    distinct ids — the API aggregates by id, not name.

    Tasks created with ``asyncio.create_task(...)`` inside a scope capture
    that task-local context. Calls made by those tasks after the scope exits
    are still attributed to the captured run id. Use ``asyncio.TaskGroup`` or
    await spawned tasks before exiting when attribution must be bounded.

    ``solwyn.run(...)`` is not supported inside async generators. A scope
    opened before ``yield`` would remain active in the consumer's ``async for``
    body, so the SDK raises at scope entry instead.

    When an operator stops this run from the Solwyn dashboard, per-call traffic
    raises :class:`solwyn.RunStoppedError` on its next budget check. Leased
    traffic raises only after a lease renewal or re-grant learns the stop.
    Requests already in flight and streams already returned are not interrupted;
    control-plane connectivity failures retain the configured fail-open posture.
    """
    return _RunScope(name, tags, inherit_tags=inherit_tags)
