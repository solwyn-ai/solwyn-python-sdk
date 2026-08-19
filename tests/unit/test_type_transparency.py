"""Type-transparency contract for provider client wrappers.

Provider SDKs are test-only dependencies. Every real client constructed here
is pointed at an unreachable local endpoint or given explicit fake credentials,
so construction and type admission stay fully offline.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from conftest import VALID_API_KEY
from pydantic import ConfigDict, create_model

from solwyn import AsyncSolwyn, Solwyn
from solwyn._proxies import _AsyncChatProxy, _SyncChatProxy
from solwyn.exceptions import ConfigurationError


def _sync_wrapper(openai_mod: Any) -> tuple[Solwyn, Any]:
    inner = openai_mod.OpenAI(api_key="test-key", base_url="http://localhost:1")
    wrapper = Solwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    return wrapper, inner


def _async_wrapper(openai_mod: Any) -> tuple[AsyncSolwyn, Any]:
    inner = openai_mod.AsyncOpenAI(api_key="test-key", base_url="http://localhost:1")
    wrapper = AsyncSolwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    return wrapper, inner


@pytest.fixture(scope="module")
def openai_mod() -> Any:
    return pytest.importorskip("openai")


@pytest.fixture(scope="module")
def anthropic_mod() -> Any:
    return pytest.importorskip("anthropic")


@pytest.fixture(scope="module")
def google_genai_mod() -> Any:
    return pytest.importorskip("google.genai")


@pytest.fixture
def wrapped(openai_mod: Any) -> Iterator[tuple[Solwyn, Any]]:
    wrapper, inner = _sync_wrapper(openai_mod)
    try:
        yield wrapper, inner
    finally:
        wrapper._solwyn_reporter.close(timeout=0.0)
        wrapper._solwyn_budget.close()
        inner.close()


@pytest.fixture
async def async_wrapped(openai_mod: Any) -> AsyncIterator[tuple[AsyncSolwyn, Any]]:
    wrapper, inner = _async_wrapper(openai_mod)
    try:
        yield wrapper, inner
    finally:
        await wrapper._solwyn_reporter.close(timeout=0.0)
        await wrapper._solwyn_budget.close()
        await inner.close()


def test_sync_isinstance_reports_wrapped_and_wrapper_classes(
    wrapped: tuple[Solwyn, Any], openai_mod: Any
) -> None:
    wrapper, _ = wrapped

    assert isinstance(wrapper, openai_mod.OpenAI)
    assert isinstance(wrapper, Solwyn)
    assert wrapper.__class__ is openai_mod.OpenAI
    assert type(wrapper) is Solwyn


def test_sync_mock_framework_type_gate_admits_wrapper(
    wrapped: tuple[Solwyn, Any], openai_mod: Any
) -> None:
    def framework_entry(client: object) -> str:
        if not isinstance(client, openai_mod.OpenAI):
            raise TypeError("expected an openai.OpenAI client")
        return "admitted"

    wrapper, _ = wrapped

    assert framework_entry(wrapper) == "admitted"


def test_sync_pydantic_arbitrary_type_field_admits_wrapper(
    wrapped: tuple[Solwyn, Any], openai_mod: Any
) -> None:
    framework_config = create_model(
        "FrameworkConfig",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        client=(openai_mod.OpenAI, ...),
    )
    wrapper, _ = wrapped

    assert framework_config(client=wrapper).client is wrapper


def test_sync_pickle_fails_loud_with_guidance(wrapped: tuple[Solwyn, Any]) -> None:
    wrapper, _ = wrapped

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(wrapper)


def test_sync_copy_and_deepcopy_share_wrapper_identity(wrapped: tuple[Solwyn, Any]) -> None:
    wrapper, _ = wrapped

    assert copy.copy(wrapper) is wrapper
    assert copy.deepcopy(wrapper) is wrapper


def test_sync_public_attribute_set_forwards_to_wrapped_client(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped

    wrapper.timeout = 5.0

    assert inner.timeout == 5.0
    assert "timeout" not in vars(wrapper)


def test_sync_wrapper_defined_provider_attribute_set_still_forwards(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped
    replacement = object()

    wrapper.chat = replacement

    assert inner.chat is replacement
    assert "chat" not in vars(wrapper)
    assert isinstance(wrapper.chat, _SyncChatProxy)


def test_sync_public_attribute_delete_forwards_to_wrapped_client(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped
    inner.framework_marker = "present"

    del wrapper.framework_marker

    assert not hasattr(inner, "framework_marker")


def test_sync_wrapper_defined_provider_attribute_delete_still_forwards(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped
    inner.chat = object()

    del wrapper.chat

    assert "chat" not in vars(inner)
    assert "chat" not in vars(wrapper)


def test_sync_internal_attribute_set_and_delete_stay_local(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped

    wrapper._solwyn_probe = "local"

    assert vars(wrapper)["_solwyn_probe"] == "local"
    assert not hasattr(inner, "_solwyn_probe")

    del wrapper._solwyn_probe

    assert "_solwyn_probe" not in vars(wrapper)
    assert not hasattr(inner, "_solwyn_probe")


def test_sync_internal_prefix_lookup_never_forwards(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped
    inner._solwyn_provider_secret = "provider"

    with pytest.raises(AttributeError, match="_solwyn_provider_secret"):
        _ = wrapper._solwyn_provider_secret
    with pytest.raises(AttributeError, match="_solwyn_missing"):
        _ = wrapper._solwyn_missing


def test_sync_dir_unions_wrapper_and_wrapped_names(wrapped: tuple[Solwyn, Any]) -> None:
    wrapper, _ = wrapped

    listing = dir(wrapper)

    assert "chat" in listing
    assert "close" in listing
    assert "with_options" in listing


def test_sync_interception_property_wins_over_wrapped_attribute(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, _ = wrapped

    assert isinstance(wrapper.chat, _SyncChatProxy)


def test_sync_cached_property_dict_write_bypasses_setattr_forwarding(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped

    chat = wrapper.chat

    assert vars(wrapper)["chat"] is chat
    assert not isinstance(inner.chat, type(chat))


def test_sync_repr_names_wrapper_truth_and_includes_wrapped_repr(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped

    assert repr(wrapper) == f"Solwyn({inner!r})"


def test_sync_wrapping_a_wrapper_fails_before_adapter_detection(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, _ = wrapped

    with pytest.raises(ConfigurationError, match="already wrapped") as exc_info:
        Solwyn(wrapper, api_key=VALID_API_KEY)

    assert exc_info.value.field == "client"


def test_fallback_wrapper_fails_before_adapter_detection(
    wrapped: tuple[Solwyn, Any], openai_mod: Any
) -> None:
    existing_wrapper, _ = wrapped
    raw_primary = openai_mod.OpenAI(api_key="test-key", base_url="http://localhost:1")
    try:
        with pytest.raises(ConfigurationError, match="already wrapped") as exc_info:
            Solwyn(
                raw_primary,
                api_key=VALID_API_KEY,
                fallback=[(existing_wrapper, "gpt-test")],
            )
    finally:
        raw_primary.close()

    assert exc_info.value.field == "client"


def test_sync_close_forwards_once_after_solwyn_shutdown(openai_mod: Any) -> None:
    wrapper, inner = _sync_wrapper(openai_mod)
    order: list[str] = []
    reporter_close = wrapper._solwyn_reporter.close
    budget_close = wrapper._solwyn_budget.close
    provider_close = inner.close

    def close_reporter() -> None:
        order.append("reporter")
        reporter_close()

    def close_budget() -> None:
        order.append("budget")
        budget_close()

    def close_provider() -> None:
        order.append("provider")
        provider_close()

    with (
        patch.object(wrapper._solwyn_reporter, "close", side_effect=close_reporter) as reporter,
        patch.object(wrapper._solwyn_budget, "close", side_effect=close_budget) as budget,
        patch.object(inner, "close", side_effect=close_provider) as provider,
    ):
        wrapper.close()

    assert order == ["reporter", "budget", "provider"]
    reporter.assert_called_once_with()
    budget.assert_called_once_with()
    provider.assert_called_once_with()


def test_sync_close_without_provider_close_is_fail_soft(openai_mod: Any) -> None:
    wrapper, inner = _sync_wrapper(openai_mod)
    provider_close = inner.close
    inner.close = None
    try:
        wrapper.close()
    finally:
        inner.close = provider_close
        provider_close()


def test_sync_close_propagates_nested_provider_attribute_error(
    wrapped: tuple[Solwyn, Any],
) -> None:
    wrapper, inner = wrapped

    def broken_provider_close() -> None:
        _ = SimpleNamespace().missing_lifecycle_field

    with (
        patch.object(inner, "close", side_effect=broken_provider_close),
        pytest.raises(AttributeError, match="missing_lifecycle_field"),
    ):
        wrapper.close()


async def test_async_isinstance_reports_wrapped_and_wrapper_classes(
    async_wrapped: tuple[AsyncSolwyn, Any], openai_mod: Any
) -> None:
    wrapper, _ = async_wrapped

    assert isinstance(wrapper, openai_mod.AsyncOpenAI)
    assert isinstance(wrapper, AsyncSolwyn)
    assert wrapper.__class__ is openai_mod.AsyncOpenAI
    assert type(wrapper) is AsyncSolwyn


async def test_async_mock_framework_type_gate_admits_wrapper(
    async_wrapped: tuple[AsyncSolwyn, Any], openai_mod: Any
) -> None:
    def framework_entry(client: object) -> str:
        if not isinstance(client, openai_mod.AsyncOpenAI):
            raise TypeError("expected an openai.AsyncOpenAI client")
        return "admitted"

    wrapper, _ = async_wrapped

    assert framework_entry(wrapper) == "admitted"


async def test_async_pydantic_arbitrary_type_field_admits_wrapper(
    async_wrapped: tuple[AsyncSolwyn, Any], openai_mod: Any
) -> None:
    framework_config = create_model(
        "AsyncFrameworkConfig",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        client=(openai_mod.AsyncOpenAI, ...),
    )
    wrapper, _ = async_wrapped

    assert framework_config(client=wrapper).client is wrapper


async def test_async_pickle_fails_loud_with_guidance(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, _ = async_wrapped

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(wrapper)


async def test_async_copy_and_deepcopy_share_wrapper_identity(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, _ = async_wrapped

    assert copy.copy(wrapper) is wrapper
    assert copy.deepcopy(wrapper) is wrapper


async def test_async_public_attribute_set_forwards_to_wrapped_client(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped

    wrapper.timeout = 7.0

    assert inner.timeout == 7.0
    assert "timeout" not in vars(wrapper)


async def test_async_wrapper_defined_provider_attribute_set_still_forwards(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped
    replacement = object()

    wrapper.chat = replacement

    assert inner.chat is replacement
    assert "chat" not in vars(wrapper)
    assert isinstance(wrapper.chat, _AsyncChatProxy)


async def test_async_public_attribute_delete_forwards_to_wrapped_client(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped
    inner.framework_marker = "present"

    del wrapper.framework_marker

    assert not hasattr(inner, "framework_marker")


async def test_async_wrapper_defined_provider_attribute_delete_still_forwards(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped
    inner.chat = object()

    del wrapper.chat

    assert "chat" not in vars(inner)
    assert "chat" not in vars(wrapper)


async def test_async_internal_attribute_set_and_delete_stay_local(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped

    wrapper._solwyn_probe = "local"

    assert vars(wrapper)["_solwyn_probe"] == "local"
    assert not hasattr(inner, "_solwyn_probe")

    del wrapper._solwyn_probe

    assert "_solwyn_probe" not in vars(wrapper)
    assert not hasattr(inner, "_solwyn_probe")


async def test_async_internal_prefix_lookup_never_forwards(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped
    inner._solwyn_provider_secret = "provider"

    with pytest.raises(AttributeError, match="_solwyn_provider_secret"):
        _ = wrapper._solwyn_provider_secret
    with pytest.raises(AttributeError, match="_solwyn_missing"):
        _ = wrapper._solwyn_missing


async def test_async_dir_unions_wrapper_and_wrapped_names(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, _ = async_wrapped

    listing = dir(wrapper)

    assert "chat" in listing
    assert "close" in listing
    assert "with_options" in listing


async def test_async_interception_property_wins_over_wrapped_attribute(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, _ = async_wrapped

    assert isinstance(wrapper.chat, _AsyncChatProxy)


async def test_async_cached_property_dict_write_bypasses_setattr_forwarding(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped

    chat = wrapper.chat

    assert vars(wrapper)["chat"] is chat
    assert not isinstance(inner.chat, type(chat))


async def test_async_repr_names_wrapper_truth_and_includes_wrapped_repr(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped

    assert repr(wrapper) == f"AsyncSolwyn({inner!r})"


async def test_async_wrapping_a_wrapper_fails_before_adapter_detection(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, _ = async_wrapped

    with pytest.raises(ConfigurationError, match="already wrapped") as exc_info:
        AsyncSolwyn(wrapper, api_key=VALID_API_KEY)

    assert exc_info.value.field == "client"


async def test_async_close_forwards_once_after_solwyn_shutdown(openai_mod: Any) -> None:
    wrapper, inner = _async_wrapper(openai_mod)
    order: list[str] = []
    reporter_close = wrapper._solwyn_reporter.close
    budget_close = wrapper._solwyn_budget.close
    provider_close = inner.close

    async def close_reporter() -> None:
        order.append("reporter")
        await reporter_close()

    async def close_budget() -> None:
        order.append("budget")
        await budget_close()

    async def close_provider() -> None:
        order.append("provider")
        await provider_close()

    reporter = AsyncMock(side_effect=close_reporter)
    budget = AsyncMock(side_effect=close_budget)
    provider = AsyncMock(side_effect=close_provider)
    with (
        patch.object(wrapper._solwyn_reporter, "close", new=reporter),
        patch.object(wrapper._solwyn_budget, "close", new=budget),
        patch.object(inner, "close", new=provider),
    ):
        await wrapper.close()

    assert order == ["reporter", "budget", "provider"]
    reporter.assert_awaited_once_with()
    budget.assert_awaited_once_with()
    provider.assert_awaited_once_with()


async def test_async_google_context_manager_prefers_aclose_after_solwyn_shutdown(
    google_genai_mod: Any,
) -> None:
    root = google_genai_mod.Client(api_key="test-key")
    inner = root.aio
    wrapper = AsyncSolwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    order: list[str] = []
    reporter_close = wrapper._solwyn_reporter.close
    budget_close = wrapper._solwyn_budget.close
    provider_aclose = inner.aclose

    async def close_reporter() -> None:
        order.append("reporter")
        await reporter_close()

    async def close_budget() -> None:
        order.append("budget")
        await budget_close()

    async def close_provider() -> None:
        order.append("provider")
        await provider_aclose()

    reporter = AsyncMock(side_effect=close_reporter)
    budget = AsyncMock(side_effect=close_budget)
    provider = AsyncMock(side_effect=close_provider)
    try:
        with (
            patch.object(wrapper._solwyn_reporter, "close", new=reporter),
            patch.object(wrapper._solwyn_budget, "close", new=budget),
            patch.object(inner, "aclose", new=provider),
        ):
            async with wrapper:
                pass

        assert order == ["reporter", "budget", "provider"]
        reporter.assert_awaited_once_with()
        budget.assert_awaited_once_with()
        provider.assert_awaited_once_with()
    finally:
        if "provider" not in order:
            await provider_aclose()
        root.close()


async def test_async_aclose_propagates_nested_provider_attribute_error(
    google_genai_mod: Any,
) -> None:
    root = google_genai_mod.Client(api_key="test-key")
    inner = root.aio
    wrapper = AsyncSolwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    provider_aclose = inner.aclose

    async def broken_provider_aclose() -> None:
        _ = SimpleNamespace().missing_lifecycle_field

    try:
        with (
            patch.object(inner, "aclose", new=AsyncMock(side_effect=broken_provider_aclose)),
            pytest.raises(AttributeError, match="missing_lifecycle_field"),
        ):
            await wrapper.close()
    finally:
        await provider_aclose()
        root.close()


async def test_async_aclose_missing_provider_lifecycle_state_is_fail_soft(
    google_genai_mod: Any,
) -> None:
    root = google_genai_mod.Client(api_key="test-key")
    inner = root.aio
    wrapper = AsyncSolwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    provider_aclose = inner.aclose

    async def uninitialized_provider_aclose() -> None:
        _ = inner.missing_lifecycle_field

    try:
        with patch.object(
            inner,
            "aclose",
            new=AsyncMock(side_effect=uninitialized_provider_aclose),
        ):
            await wrapper.close()
    finally:
        await provider_aclose()
        root.close()


async def test_async_close_without_provider_close_is_fail_soft(openai_mod: Any) -> None:
    wrapper, inner = _async_wrapper(openai_mod)
    provider_close = inner.close
    inner.close = None
    try:
        await wrapper.close()
    finally:
        inner.close = provider_close
        await provider_close()


async def test_async_close_propagates_nested_provider_attribute_error(
    async_wrapped: tuple[AsyncSolwyn, Any],
) -> None:
    wrapper, inner = async_wrapped

    async def broken_provider_close() -> None:
        _ = SimpleNamespace().missing_lifecycle_field

    with (
        patch.object(inner, "close", new=AsyncMock(side_effect=broken_provider_close)),
        pytest.raises(AttributeError, match="missing_lifecycle_field"),
    ):
        await wrapper.close()


def test_anthropic_wrapper_is_admitted_as_real_client(anthropic_mod: Any) -> None:
    inner = anthropic_mod.Anthropic(api_key="test-key", base_url="http://localhost:1")
    wrapper = Solwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    try:
        assert isinstance(wrapper, anthropic_mod.Anthropic)
        assert type(wrapper) is Solwyn
    finally:
        wrapper._solwyn_reporter.close(timeout=0.0)
        wrapper._solwyn_budget.close()
        inner.close()


def test_modern_google_wrapper_is_admitted_as_real_client(google_genai_mod: Any) -> None:
    inner = google_genai_mod.Client(api_key="test-key")
    wrapper = Solwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    try:
        assert isinstance(wrapper, google_genai_mod.Client)
        assert type(wrapper) is Solwyn
    finally:
        wrapper._solwyn_reporter.close(timeout=0.0)
        wrapper._solwyn_budget.close()
        inner.close()


class _BedrockRuntime:
    """Minimal boto3-shaped client; core detection remains import-free."""

    __module__ = "botocore.client"

    def __init__(self) -> None:
        self.meta = SimpleNamespace(
            service_model=SimpleNamespace(service_name="bedrock-runtime"),
            region_name="us-east-1",
        )

    def converse(self, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def converse_stream(self, **_kwargs: Any) -> dict[str, Any]:
        return {"stream": iter(())}


def test_bedrock_duck_type_wrapper_is_admitted_as_client() -> None:
    inner = _BedrockRuntime()
    wrapper = Solwyn(
        inner,
        api_key=VALID_API_KEY,
        reporter_shutdown_deadline=0.0,
        report_untracked_surfaces=False,
    )
    try:
        assert isinstance(wrapper, _BedrockRuntime)
        assert type(wrapper) is Solwyn
    finally:
        wrapper._solwyn_reporter.close(timeout=0.0)
        wrapper._solwyn_budget.close()
