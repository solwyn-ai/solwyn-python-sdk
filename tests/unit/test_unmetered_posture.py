"""Runtime posture and guarded-resource behavior for untracked capabilities."""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY, foreground_records

from solwyn._base import _reset_unmetered_spend_warnings
from solwyn._surfaces import SurfaceSource, resolve_surface_rule
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ConfigurationError, UntrackedSpendSurfaceError


class _SafeURL:
    pass


class _ResponsesResource:
    def create(self) -> str:
        return "created"

    def retrieve(self) -> str:
        return "retrieved"

    def delete(self) -> str:
        return "deleted"


class _FutureResource:
    def launch(self) -> str:
        return "launched"

    def sibling(self) -> str:
        return "sibling"


class _ContextResource:
    def __enter__(self) -> _ContextResource:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _RawResponsesResource:
    def create(self) -> str:
        return "raw-created"

    def delete(self) -> str:
        return "raw-deleted"


class _RawResponseClientResource:
    @functools.cached_property
    def responses(self) -> _RawResponsesResource:
        return _RawResponsesResource()


class _CallsResource:
    def create(self) -> str:
        return "call-created"

    def reject(self) -> str:
        return "call-rejected"


class _RealtimeResource:
    def __init__(self, client: _OpenAIClient) -> None:
        self._client = client

    @functools.cached_property
    def calls(self) -> _CallsResource:
        self._client.calls_evaluations += 1
        return _CallsResource()


class _RawSpeechResponseResource:
    def create(self) -> str:
        return "raw-created"

    def future(self) -> str:
        return "future"


class _SpeechResource:
    @functools.cached_property
    def with_raw_response(self) -> _RawSpeechResponseResource:
        return _RawSpeechResponseResource()


class _AudioResource:
    @functools.cached_property
    def speech(self) -> _SpeechResource:
        return _SpeechResource()


_ResponsesResource.__module__ = "openai.resources.responses"
_FutureResource.__module__ = "openai.resources.future"
_ContextResource.__module__ = "openai.resources.future"
_RawResponsesResource.__module__ = "openai.resources.responses"
_RawResponseClientResource.__module__ = "openai.resources.with_raw_response"
_CallsResource.__module__ = "openai.resources.realtime.calls"
_RealtimeResource.__module__ = "openai.resources.realtime"
_RawSpeechResponseResource.__module__ = "openai.resources.audio.speech"
_SpeechResource.__module__ = "openai.resources.audio.speech"
_AudioResource.__module__ = "openai.resources.audio.audio"


class _OpenAIClient:
    def __init__(self) -> None:
        self.responses_evaluations = 0
        self.future_evaluations = 0
        self.base_url_evaluations = 0
        self.custom_auth_evaluations = 0
        self.calls_evaluations = 0
        self._safe_url = _SafeURL()

    @functools.cached_property
    def responses(self) -> _ResponsesResource:
        self.responses_evaluations += 1
        return _ResponsesResource()

    @functools.cached_property
    def future(self) -> _FutureResource:
        self.future_evaluations += 1
        return _FutureResource()

    @functools.cached_property
    def future_context(self) -> _ContextResource:
        return _ContextResource()

    @functools.cached_property
    def with_raw_response(self) -> _RawResponseClientResource:
        return _RawResponseClientResource()

    @functools.cached_property
    def realtime(self) -> _RealtimeResource:
        return _RealtimeResource(self)

    @functools.cached_property
    def audio(self) -> _AudioResource:
        return _AudioResource()

    @property
    def base_url(self) -> _SafeURL:
        self.base_url_evaluations += 1
        return self._safe_url

    @property
    def custom_auth(self) -> str:
        self.custom_auth_evaluations += 1
        return "custom-auth"

    def post(self) -> str:
        return "posted"

    def experimental_leaf(self) -> str:
        return "experimental"

    def close(self) -> None:
        return None


class _MissingResponsesClient:
    def __init__(self) -> None:
        self._safe_url = _SafeURL()

    @property
    def base_url(self) -> _SafeURL:
        return self._safe_url

    def close(self) -> None:
        return None


class _DynamicOpenAIClient(_MissingResponsesClient):
    def __init__(self) -> None:
        super().__init__()
        self.dynamic_evaluations = 0

    def __getattr__(self, name: str) -> Any:
        self.dynamic_evaluations += 1
        return lambda: name


class _DynamicGetattributeOpenAIClient(_MissingResponsesClient):
    def __init__(self) -> None:
        super().__init__()
        self.dynamic_evaluations = 0

    def __getattribute__(self, name: str) -> Any:
        if name == "future_operation":
            evaluations = object.__getattribute__(self, "dynamic_evaluations")
            object.__setattr__(self, "dynamic_evaluations", evaluations + 1)
            return lambda: name
        return super().__getattribute__(name)


class _DriftedBaseURLClient(_OpenAIClient):
    @property
    def base_url(self) -> Any:
        self.base_url_evaluations += 1
        return lambda: "dispatch"


class _DriftedPostClient(_OpenAIClient):
    @property
    def post(self) -> str:
        self.post_evaluations += 1
        return "drifted-post"


class _OpaqueClient(_OpenAIClient):
    def __init__(self) -> None:
        super().__init__()
        self.opaque = object()


class _DeepDynamicResource:
    def __getattr__(self, name: str) -> Any:
        if name == "call":
            return lambda: "called"
        return self


class _DeepOpenAIClient(_OpenAIClient):
    @functools.cached_property
    def deep(self) -> _DeepDynamicResource:
        return _DeepDynamicResource()


_DeepDynamicResource.__module__ = "openai.resources.deep"


for _client_type in (
    _OpenAIClient,
    _MissingResponsesClient,
    _DynamicOpenAIClient,
    _DynamicGetattributeOpenAIClient,
    _DriftedBaseURLClient,
    _DriftedPostClient,
    _OpaqueClient,
    _DeepOpenAIClient,
):
    _client_type.__module__ = "openai._client"


def _make_solwyn(client: object, **kwargs: object) -> Solwyn:
    with patch("solwyn.client.MetadataReporter", autospec=True):
        wrapper = Solwyn(client, api_key=VALID_API_KEY, **kwargs)
    return wrapper


def _close(wrapper: Solwyn) -> None:
    wrapper.close()


@pytest.fixture(autouse=True)
def _reset_warning_state() -> Iterator[None]:
    _reset_unmetered_spend_warnings()
    yield
    _reset_unmetered_spend_warnings()


@pytest.mark.unit
def test_default_warn_logs_once_and_returns_the_terminal_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrapper = _make_solwyn(_OpenAIClient())

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        first = wrapper.post
        second = wrapper.post

    assert first() == "posted"
    assert second() == "posted"
    records = foreground_records(caplog)
    assert len(records) == 1
    assert records[0].args == ("openai", "openai_sdk", "post", "arbitrary_endpoint")
    _close(wrapper)


@pytest.mark.unit
def test_reviewed_shape_drift_counts_after_the_matching_surface_already_warned(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="warn")
    reviewed_rule = resolve_surface_rule(
        context=wrapper._solwyn_surface_context,
        path="post",
        source=SurfaceSource.RAW,
    )
    assert reviewed_rule is not None

    # Act
    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        first = wrapper.post
        monkeypatch.setattr(_OpenAIClient, "post", property(lambda _client: "drifted"))
        second = wrapper.post

    # Assert
    assert first() == "posted"
    assert second == "drifted"
    records = foreground_records(caplog)
    assert len(records) == 1
    assert records[0].args == ("openai", "openai_sdk", "post", "arbitrary_endpoint")
    from solwyn import _base

    observation = _base._untracked_surface_observations[("openai", "openai_sdk", "sync", "post")]
    assert observation["occurrences"] == 2
    assert observation["rule_kind"] == "unknown"
    assert observation["capability_scope"] is None
    _close(wrapper)


@pytest.mark.unit
def test_untracked_observation_keys_include_mode_and_are_bounded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from solwyn import _base

    # Arrange
    _base._reset_unmetered_spend_warnings()
    sync_context = _base.SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )
    async_context = _base.SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="async",
    )

    try:
        # Act
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            for context in (sync_context, async_context):
                _base._warn_contextual_surface_once(
                    context=context,
                    surface="post",
                    rule_kind="unknown",
                    capability_scope=None,
                )

        # Assert
        assert len(_base._untracked_surface_observations) == 2

        # Act
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            for index in range(_base._UNTRACKED_SURFACE_LIMIT + 50):
                _base._warn_contextual_surface_once(
                    context=sync_context,
                    surface=f"surface_{index}",
                    rule_kind="unknown",
                    capability_scope=None,
                )

        # Assert
        assert len(_base._untracked_surface_observations) <= _base._UNTRACKED_SURFACE_LIMIT
        overflow_records = [
            record
            for record in caplog.records
            if "Untracked-surface warning limit" in record.getMessage()
        ]
        assert len(overflow_records) == 1
    finally:
        _base._reset_unmetered_spend_warnings()


@pytest.mark.unit
def test_warn_latch_overflow_logging_is_reentrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solwyn import _base

    class _SameThreadReentryGuard:
        entered = False

        def __enter__(self) -> None:
            if self.entered:
                raise RuntimeError("warning lock reentered")
            self.entered = True

        def __exit__(self, *args: object) -> None:
            self.entered = False

    # Arrange
    context = _base.SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )
    lock = _SameThreadReentryGuard()
    for index in range(_base._UNTRACKED_SURFACE_LIMIT):
        _base._record_untracked_surface_observation(
            context=context,
            surface=f"surface_{index}",
            rule_kind="unknown",
            capability_scope=None,
            posture="warn",
        )
    monkeypatch.setattr(_base, "_spend_surface_warn_lock", lock)
    warning_calls = 0

    def reenter_once(*args: object) -> None:
        nonlocal warning_calls
        warning_calls += 1
        if warning_calls == 1:
            _base._warn_contextual_surface_once(
                context=context,
                surface="recursive_surface",
                rule_kind="unknown",
                capability_scope=None,
            )

    monkeypatch.setattr(_base.logger, "warning", reenter_once)

    try:
        # Act
        _base._warn_contextual_surface_once(
            context=context,
            surface="overflow_surface",
            rule_kind="unknown",
            capability_scope=None,
        )

        # Assert
        assert warning_calls == 1
    finally:
        _base._reset_unmetered_spend_warnings()


@pytest.mark.unit
def test_observation_timestamps_cannot_regress_when_lock_acquisition_is_reordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solwyn import _base

    older = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    newer = datetime(2026, 8, 13, 12, 1, tzinfo=UTC)
    sampled_times = iter((older, newer))
    context = _base.SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )

    class _InterleavingClock:
        @classmethod
        def now(cls, timezone: object) -> datetime:
            assert timezone is UTC
            return next(sampled_times)

    class _LaterObservationEntersFirst:
        interleaved = False

        def __enter__(self) -> None:
            if self.interleaved:
                return
            self.interleaved = True
            _base._record_untracked_surface_observation(
                context=context,
                surface="post",
                rule_kind="unmetered_spend",
                capability_scope="arbitrary_endpoint",
                posture="warn",
            )

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(_base, "datetime", _InterleavingClock)
    monkeypatch.setattr(_base, "_spend_surface_warn_lock", _LaterObservationEntersFirst())

    _base._record_untracked_surface_observation(
        context=context,
        surface="post",
        rule_kind="unmetered_spend",
        capability_scope="arbitrary_endpoint",
        posture="allow",
    )

    observation = _base._untracked_surface_observations[("openai", "openai_sdk", "sync", "post")]
    assert observation["occurrences"] == 2
    assert observation["first_seen_at"] == older
    assert observation["last_seen_at"] == newer
    assert observation["posture"] == "allow"
    assert observation["warning_emitted"] is True


@pytest.mark.unit
def test_observation_interval_widens_monotonically_when_wall_clock_moves_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solwyn import _base

    first = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    backward = datetime(2026, 8, 13, 11, 59, tzinfo=UTC)
    sampled_times = iter((first, backward))
    context = _base.SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )

    class _BackwardClock:
        @classmethod
        def now(cls, timezone: object) -> datetime:
            assert timezone is UTC
            return next(sampled_times)

    monkeypatch.setattr(_base, "datetime", _BackwardClock)

    for posture in ("allow", "warn"):
        _base._record_untracked_surface_observation(
            context=context,
            surface="post",
            rule_kind="unmetered_spend",
            capability_scope="arbitrary_endpoint",
            posture=posture,
        )

    observation = _base._untracked_surface_observations[("openai", "openai_sdk", "sync", "post")]
    assert observation["occurrences"] == 2
    assert observation["first_seen_at"] == backward
    assert observation["last_seen_at"] == first
    assert observation["posture"] == "warn"
    assert observation["warning_emitted"] is True


@pytest.mark.unit
def test_fork_reset_replaces_module_warning_locks_with_unlocked_locks() -> None:
    from solwyn import _base

    # Arrange
    old_spend_lock = _base._spend_surface_warn_lock
    old_cost_lock = _base._cost_policy_warn_lock
    spend_sentinel = object()
    cost_sentinel = object()
    _base._spend_surface_warn_lock = spend_sentinel  # type: ignore[assignment]
    _base._cost_policy_warn_lock = cost_sentinel  # type: ignore[assignment]

    try:
        # Act
        _base._reset_warn_locks_after_fork_in_child()

        # Assert
        assert _base._spend_surface_warn_lock is not spend_sentinel
        assert _base._cost_policy_warn_lock is not cost_sentinel
        assert _base._spend_surface_warn_lock.acquire(blocking=False)
        _base._spend_surface_warn_lock.release()
        assert _base._cost_policy_warn_lock.acquire(blocking=False)
        _base._cost_policy_warn_lock.release()
    finally:
        _base._spend_surface_warn_lock = old_spend_lock
        _base._cost_policy_warn_lock = old_cost_lock


@pytest.mark.unit
def test_allow_is_silent_and_returns_the_terminal_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from solwyn import _base

    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="allow")

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        assert wrapper.post() == "posted"

    assert foreground_records(caplog) == []
    key = ("openai", "openai_sdk", "sync", "post")
    observation = _base._untracked_surface_observations[key]
    assert observation["occurrences"] == 1
    assert observation["rule_kind"] == "unmetered_spend"
    assert observation["capability_scope"] == "arbitrary_endpoint"
    assert observation["posture"] == "allow"
    _close(wrapper)


@pytest.mark.unit
def test_allow_then_warn_counts_both_and_warns_on_the_first_warn_hit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from solwyn import _base

    client = _OpenAIClient()
    allow_wrapper = _make_solwyn(client, on_unmetered="allow")
    warn_wrapper = _make_solwyn(client, on_unmetered="warn")

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        assert allow_wrapper.post() == "posted"
        assert warn_wrapper.post() == "posted"

    records = foreground_records(caplog)
    assert len(records) == 1
    assert records[0].args == ("openai", "openai_sdk", "post", "arbitrary_endpoint")
    observation = _base._untracked_surface_observations[("openai", "openai_sdk", "sync", "post")]
    assert observation["occurrences"] == 2
    assert observation["posture"] == "warn"
    _close(allow_wrapper)
    _close(warn_wrapper)


@pytest.mark.unit
def test_warn_logs_once_and_counts_every_observation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from solwyn import _base

    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="warn")

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        calls = [wrapper.post for _ in range(3)]

    assert [call() for call in calls] == ["posted", "posted", "posted"]
    assert len(foreground_records(caplog)) == 1
    key = ("openai", "openai_sdk", "sync", "post")
    observation = _base._untracked_surface_observations[key]
    assert observation["occurrences"] == 3
    assert observation["first_seen_at"] <= observation["last_seen_at"]
    assert observation["rule_kind"] == "unmetered_spend"
    assert observation["capability_scope"] == "arbitrary_endpoint"
    assert observation["posture"] == "warn"
    _close(wrapper)


@pytest.mark.unit
def test_raise_records_no_untracked_surface_observation() -> None:
    from solwyn import _base

    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.post

    assert _base._untracked_surface_observations == {}
    _close(wrapper)


@pytest.mark.unit
def test_untracked_surface_registry_contains_only_structural_data() -> None:
    from solwyn import _base

    class _PromptBearingOpenAIClient(_OpenAIClient):
        def post(self, prompt: str) -> str:
            return prompt

    _PromptBearingOpenAIClient.__module__ = "openai._client"
    prompt = "prompt-derived-secret"
    wrapper = _make_solwyn(_PromptBearingOpenAIClient(), on_unmetered="allow")

    assert wrapper.post(prompt) == prompt

    key = ("openai", "openai_sdk", "sync", "post")
    observation = _base._untracked_surface_observations[key]
    assert set(observation) == {
        "occurrences",
        "first_seen_at",
        "last_seen_at",
        "rule_kind",
        "capability_scope",
        "posture",
        "warning_emitted",
    }
    assert observation["warning_emitted"] is False
    assert prompt not in repr(_base._untracked_surface_observations)
    _close(wrapper)


@pytest.mark.unit
def test_raise_refuses_a_known_untracked_surface_with_its_scope() -> None:
    # Arrange
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="raise")

    # Act
    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.post

    # Assert
    assert exc_info.value.surface == "post"
    assert exc_info.value.token == "post"
    assert exc_info.value.capability_scope == "arbitrary_endpoint"
    assert exc_info.value.drifted_from_rule_id is None
    assert "acknowledge exact token 'post'" in str(exc_info.value)
    assert repr(exc_info.value) == (
        "UntrackedSpendSurfaceError(surface='post', provider='openai', kind='unmetered_spend')"
    )
    _close(wrapper)


@pytest.mark.unit
def test_metered_leaf_reaching_generic_resolver_raises_wiring_invariant() -> None:
    wrapper = _make_solwyn(_OpenAIClient())
    raw = _ResponsesResource()

    with pytest.raises(RuntimeError, match="metered surface reached generic resolver"):
        wrapper._resolve_public_attribute(
            raw,
            name="create",
            path="chat.completions.create",
            source=SurfaceSource.RAW,
        )

    _close(wrapper)


@pytest.mark.unit
def test_exact_acknowledgment_allows_only_the_terminal_leaf() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"responses.retrieve"},
    )

    responses = wrapper.responses

    assert responses is wrapper.responses
    assert responses is not client.responses
    assert responses.retrieve() == "retrieved"
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = responses.delete
    _close(wrapper)


@pytest.mark.unit
def test_raw_response_chain_stays_guarded_until_acknowledged_terminal() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"audio.speech.with_raw_response.create"},
    )

    guarded = wrapper.audio.speech.with_raw_response

    assert guarded is wrapper.audio.speech.with_raw_response
    assert guarded is not client.audio.speech.with_raw_response
    assert guarded.create() == "raw-created"
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = guarded.future
    _close(wrapper)


@pytest.mark.unit
def test_exact_scoped_raw_response_container_acknowledgment_returns_raw() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"audio.speech.with_raw_response"},
    )

    raw_response = wrapper.audio.speech.with_raw_response

    assert raw_response is client.audio.speech.with_raw_response
    _close(wrapper)


@pytest.mark.unit
def test_unknown_acknowledgment_keeps_prefix_guarded_and_future_sibling_refused() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"future.launch"},
    )

    future = wrapper.future

    assert future is wrapper.future
    assert future is not client.future
    assert future.launch() == "launched"
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = future.sibling
    _close(wrapper)


@pytest.mark.unit
def test_unmetered_resource_prefix_allows_only_an_exact_descendant() -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"with_raw_response.responses.create"},
    )

    # Act
    raw_response = wrapper.with_raw_response
    responses = raw_response.responses

    # Assert
    assert raw_response is wrapper.with_raw_response
    assert raw_response is not client.with_raw_response
    assert responses is raw_response.responses
    assert responses is not client.with_raw_response.responses
    assert responses.create() == "raw-created"
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = responses.delete
    _close(wrapper)


@pytest.mark.unit
def test_exact_scoped_raw_response_parent_acknowledgment_remains_raw() -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"with_raw_response"},
    )

    # Act
    raw_response = wrapper.with_raw_response

    # Assert
    assert raw_response is client.with_raw_response
    _close(wrapper)


@pytest.mark.unit
def test_exact_raw_response_acknowledgment_does_not_authorize_shape_drift() -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"with_raw_response"},
    )
    validated_value = client.with_raw_response
    client.__dict__["with_raw_response"] = lambda: "drifted"

    # Act
    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.with_raw_response

    # Assert
    assert isinstance(validated_value, _RawResponseClientResource)
    assert exc_info.value.kind == "unknown"
    _close(wrapper)


@pytest.mark.unit
@pytest.mark.parametrize("posture", ["warn", "allow"])
def test_permitted_raw_response_parent_without_exact_acknowledgment_is_guarded(
    posture: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered=posture)

    # Act
    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        raw_response = wrapper.with_raw_response
        repeated = wrapper.with_raw_response

    # Assert
    assert raw_response is repeated
    assert raw_response is not client.with_raw_response
    records = foreground_records(caplog)
    if posture == "warn":
        assert len(records) == 1
        assert records[0].args == (
            "openai",
            "openai_sdk",
            "with_raw_response",
            "raw_response",
        )
    else:
        assert records == []
    _close(wrapper)


@pytest.mark.unit
def test_exact_raw_response_acknowledgment_does_not_authorize_static_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"with_raw_response"},
    )
    evaluations = 0

    def drifted_raw_response(_client: _OpenAIClient) -> _RawResponseClientResource:
        nonlocal evaluations
        evaluations += 1
        return _RawResponseClientResource()

    monkeypatch.setattr(
        _OpenAIClient,
        "with_raw_response",
        property(drifted_raw_response),
    )

    # Act
    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.with_raw_response

    # Assert
    assert exc_info.value.kind == "unknown"
    assert evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_realtime_calls_namespace_returns_a_cached_guard_and_refuses_child() -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")

    # Act
    calls = wrapper.realtime.calls

    # Assert
    assert client.calls_evaluations == 1
    assert calls is wrapper.realtime.calls
    assert calls is not client.realtime.calls
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = calls.create
    _close(wrapper)


@pytest.mark.unit
def test_realtime_calls_namespace_parent_acknowledgment_is_rejected() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ConfigurationError, match="classified as namespace"):
        _make_solwyn(
            _OpenAIClient(),
            on_unmetered="raise",
            acknowledge_untracked={"realtime.calls"},
        )


@pytest.mark.unit
def test_realtime_calls_exact_descendant_allows_only_that_operation() -> None:
    # Arrange
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"realtime.calls.create"},
    )

    # Act
    calls = wrapper.realtime.calls
    created = calls.create()

    # Assert
    assert created == "call-created"
    assert calls is wrapper.realtime.calls
    assert calls is not client.realtime.calls
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = calls.reject
    _close(wrapper)


@pytest.mark.unit
def test_non_guardable_unevaluated_prefix_acknowledgment_is_rejected() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ConfigurationError, match="not a guardable provider resource"):
        _make_solwyn(
            _OpenAIClient(),
            on_unmetered="raise",
            acknowledge_untracked={"custom_auth.leaf"},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "token",
    [
        "responses",
        "responses.*",
        "missing",
        "messages.stream",
        "chat.completions.create",
        "future",
    ],
)
def test_invalid_acknowledgments_are_rejected_even_under_allow(token: str) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        _make_solwyn(
            _OpenAIClient(),
            on_unmetered="allow",
            acknowledge_untracked={token},
        )

    assert exc_info.value.field == "acknowledge_untracked"


@pytest.mark.unit
def test_unsupported_acknowledgment_is_rejected_before_live_graph_validation() -> None:
    with pytest.raises(ConfigurationError, match="classified as unsupported"):
        _make_solwyn(
            _OpenAIClient(),
            provider="groq",
            on_unmetered="allow",
            acknowledge_untracked={"videos.create"},
        )


@pytest.mark.unit
def test_blocked_acknowledgment_is_rejected_before_live_graph_validation() -> None:
    BedrockClient = type(
        "BedrockRuntime",
        (),
        {
            "__module__": "botocore.client",
            "invoke_model": lambda self: None,
        },
    )
    client = BedrockClient()
    client.meta = SimpleNamespace(
        service_model=SimpleNamespace(service_name="bedrock-runtime"),
        region_name="us-east-1",
    )

    with pytest.raises(ConfigurationError, match="classified as blocked"):
        _make_solwyn(
            client,
            on_unmetered="allow",
            acknowledge_untracked={"invoke_model"},
        )


@pytest.mark.unit
def test_exported_conditional_acknowledgment_is_valid_without_a_raw_path() -> None:
    wrapper = _make_solwyn(
        _OpenAIClient(),
        on_unmetered="raise",
        acknowledge_untracked={"audio.speech.create:gpt-4o-mini-tts"},
    )

    assert wrapper._solwyn_config.acknowledge_untracked == frozenset(
        {"audio.speech.create:gpt-4o-mini-tts"}
    )
    _close(wrapper)


@pytest.mark.unit
def test_strict_invisible_dynamic_unknown_refuses_before_descriptor_evaluation() -> None:
    client = _DynamicOpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.dynamic_evaluations = 0

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.future_operation

    assert client.dynamic_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
@pytest.mark.parametrize(
    "surface",
    [
        "not public",
        "foo-bar",
        "a..b",
    ],
    ids=["space", "hyphen", "empty-segment"],
)
def test_dynamic_access_rejects_a_non_structural_public_surface(
    surface: str,
) -> None:
    from solwyn import _base

    wrapper = _make_solwyn(_DynamicOpenAIClient(), on_unmetered="allow")

    with pytest.raises(RuntimeError, match="invalid public surface path"):
        getattr(wrapper, surface)

    assert _base._untracked_surface_observations == {}
    _close(wrapper)


@pytest.mark.unit
@pytest.mark.parametrize("posture", ["warn", "allow"])
@pytest.mark.parametrize(
    "surface",
    ["a" * 129, "café"],
    ids=["overlong", "non-ascii"],
)
def test_wire_ineligible_identifier_forwards_and_stays_local(
    surface: str,
    posture: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from solwyn import _base

    wrapper = _make_solwyn(_DynamicOpenAIClient(), on_unmetered=posture)
    notifier = wrapper._solwyn_untracked_observation_notifier
    assert isinstance(notifier, MagicMock)
    notifier.reset_mock()

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        first = getattr(wrapper, surface)
        second = getattr(wrapper, surface)

    assert first() == surface
    assert second() == surface
    observation = _base._untracked_surface_observations[("openai", "openai_sdk", "sync", surface)]
    assert observation["occurrences"] == 2
    notifier.assert_not_called()
    records = foreground_records(caplog)
    assert len(records) == (1 if posture == "warn" else 0)
    _close(wrapper)


@pytest.mark.unit
@pytest.mark.parametrize("posture", ["warn", "allow"])
def test_nine_segment_identifier_chain_forwards_without_an_advisory_report(
    posture: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from solwyn import _base

    wrapper = _make_solwyn(_DeepOpenAIClient(), on_unmetered=posture)
    resource = wrapper.deep
    for segment in ("one", "two", "three", "four", "five", "six", "seven"):
        resource = getattr(resource, segment)
    terminal_path = "deep.one.two.three.four.five.six.seven.call"
    notifier = wrapper._solwyn_untracked_observation_notifier
    assert isinstance(notifier, MagicMock)
    notifier.reset_mock()
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        assert hasattr(resource, "call") is True
        terminal = resource.call

    assert terminal() == "called"
    observation = _base._untracked_surface_observations[
        ("openai", "openai_sdk", "sync", terminal_path)
    ]
    assert observation["occurrences"] >= 1
    notifier.assert_not_called()
    records = foreground_records(caplog)
    assert len(records) == (1 if posture == "warn" else 0)
    _close(wrapper)


@pytest.mark.unit
def test_registry_boundary_rejects_an_invalid_wire_surface() -> None:
    from solwyn import _base

    context = _base.SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )

    with pytest.raises(RuntimeError, match="invalid public surface path"):
        _base._record_untracked_surface_observation(
            context=context,
            surface="prompt derived value",
            rule_kind="unknown",
            capability_scope=None,
            posture="allow",
        )

    assert _base._untracked_surface_observations == {}


@pytest.mark.unit
def test_strict_invisible_dynamic_known_path_refuses_before_dynamic_lookup() -> None:
    client = _DynamicOpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.dynamic_evaluations = 0

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.responses

    assert client.dynamic_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_strict_custom_getattribute_refuses_before_dynamic_lookup() -> None:
    # Arrange
    client = _DynamicGetattributeOpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.dynamic_evaluations = 0

    # Act
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.future_operation

    # Assert
    assert client.dynamic_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_unknown_resource_error_does_not_advertise_the_parent_token() -> None:
    # Arrange
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="raise")

    # Act
    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.future
    message = str(exc_info.value)
    _close(wrapper)
    with pytest.raises(ConfigurationError, match="names a resource container"):
        _make_solwyn(
            _OpenAIClient(),
            on_unmetered="raise",
            acknowledge_untracked={"future"},
        )

    # Assert
    assert "acknowledge exact token 'future'" not in message
    assert "review the provider graph and acknowledge an exact terminal capability token" in message


@pytest.mark.unit
def test_unknown_callable_error_allows_its_exact_terminal_token() -> None:
    # Arrange
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="raise")
    acknowledged = _make_solwyn(
        _OpenAIClient(),
        on_unmetered="raise",
        acknowledge_untracked={"experimental_leaf"},
    )

    # Act
    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.experimental_leaf
    acknowledged_result = acknowledged.experimental_leaf()

    # Assert
    assert "review the provider graph and acknowledge an exact terminal capability token" in str(
        exc_info.value
    )
    assert acknowledged_result == "experimental"
    _close(wrapper)
    _close(acknowledged)


@pytest.mark.unit
def test_strict_untracked_property_is_not_evaluated_before_refusal() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.custom_auth

    assert client.custom_auth_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_untracked_static_shape_drift_reenters_unknown_before_evaluation() -> None:
    client = _DriftedPostClient()
    client.post_evaluations = 0
    wrapper = _make_solwyn(client, on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.post

    assert exc_info.value.kind == "unknown"
    assert client.post_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_acknowledged_unevaluated_property_is_not_touched_during_validation() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"custom_auth"},
    )

    assert client.custom_auth_evaluations == 0
    assert wrapper.custom_auth == "custom-auth"
    assert client.custom_auth_evaluations == 1
    _close(wrapper)


@pytest.mark.unit
def test_known_missing_path_preserves_provider_attribute_error() -> None:
    wrapper = _make_solwyn(_MissingResponsesClient(), on_unmetered="raise")

    with pytest.raises(AttributeError) as exc_info:
        _ = wrapper.responses

    assert type(exc_info.value) is AttributeError
    _close(wrapper)


@pytest.mark.unit
def test_probing_a_nonexistent_unruled_name_is_a_plain_attribute_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    wrapper = _make_solwyn(_MissingResponsesClient(), on_unmetered="warn")

    # Act
    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        exists = hasattr(wrapper, "nonexistent_unruled")

    # Assert
    assert exists is False
    assert all("untracked surface" not in record.getMessage() for record in caplog.records)
    _close(wrapper)


@pytest.mark.unit
def test_probing_a_nonexistent_unruled_name_under_raise_is_a_plain_attribute_error() -> None:
    # Arrange
    wrapper = _make_solwyn(_MissingResponsesClient(), on_unmetered="raise")

    # Act
    with pytest.raises(AttributeError) as exc_info:
        _ = wrapper.nonexistent_unruled

    # Assert
    assert type(exc_info.value) is AttributeError
    _close(wrapper)


@pytest.mark.unit
def test_feature_probe_helpers_treat_strict_refusal_as_absent() -> None:
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="raise")
    sentinel = object()

    assert hasattr(wrapper, "post") is False
    assert getattr(wrapper, "post", sentinel) is sentinel
    _close(wrapper)


@pytest.mark.unit
def test_private_names_preserve_raw_passthrough() -> None:
    client = _OpenAIClient()
    client._provider_private = object()
    wrapper = _make_solwyn(client, on_unmetered="raise")

    assert wrapper._provider_private is client._provider_private
    _close(wrapper)


@pytest.mark.unit
def test_safe_metadata_preserves_identity_when_both_shapes_match() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.base_url_evaluations = 0

    assert wrapper.base_url is client._safe_url
    assert client.base_url_evaluations == 1
    _close(wrapper)


@pytest.mark.unit
def test_safe_metadata_return_shape_drift_reenters_strict_unknown_posture() -> None:
    # Arrange
    client = _DriftedBaseURLClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.base_url_evaluations = 0
    reviewed_rule = resolve_surface_rule(
        context=wrapper._solwyn_surface_context,
        path="base_url",
        source=SurfaceSource.RAW,
    )
    assert reviewed_rule is not None

    # Act
    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.base_url

    # Assert
    assert exc_info.value.kind == "unknown"
    assert exc_info.value.drifted_from_rule_id == reviewed_rule.rule_id
    assert f"reviewed rule {reviewed_rule.rule_id} no longer matches its shape" in str(
        exc_info.value
    )
    assert f"drifted_from_rule_id={reviewed_rule.rule_id!r}" in repr(exc_info.value)
    assert client.base_url_evaluations == 1
    _close(wrapper)


@pytest.mark.unit
def test_safe_metadata_return_shape_drift_warning_names_the_reviewed_rule(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    client = _DriftedBaseURLClient()
    wrapper = _make_solwyn(client, on_unmetered="warn")
    reviewed_rule = resolve_surface_rule(
        context=wrapper._solwyn_surface_context,
        path="base_url",
        source=SurfaceSource.RAW,
    )
    assert reviewed_rule is not None

    # Act
    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        _ = wrapper.base_url

    # Assert
    records = foreground_records(caplog)
    assert len(records) == 1
    assert (
        f"Reviewed rule {reviewed_rule.rule_id} no longer matches its shape."
        in records[0].getMessage()
    )
    _close(wrapper)


@pytest.mark.unit
def test_allow_guards_recognized_unknown_resources_but_returns_opaque_values() -> None:
    client = _OpaqueClient()
    wrapper = _make_solwyn(client, on_unmetered="allow")

    future = wrapper.future

    assert future is not client.future
    assert future.launch() == "launched"
    assert wrapper.opaque is client.opaque
    _close(wrapper)


@pytest.mark.unit
def test_guard_does_not_forward_context_manager_special_methods() -> None:
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="allow")

    guarded = wrapper.future_context

    with pytest.raises(TypeError), guarded:
        pass
    _close(wrapper)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_wrapper_uses_the_same_strict_resolver() -> None:
    wrapper = AsyncSolwyn(_OpenAIClient(), api_key=VALID_API_KEY, on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.post

    await wrapper.close()
