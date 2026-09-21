"""Webhook policy forwards exact acknowledgments without granting sibling access."""

from __future__ import annotations

import functools
import inspect
from unittest.mock import patch

import pytest
from conftest import VALID_API_KEY

from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import UntrackedSpendSurfaceError


class _WebhookResource:
    @functools.cached_property
    def event_types(self) -> _WebhookResource:
        return _WebhookResource()

    @functools.cached_property
    def with_raw_response(self) -> _WebhookResource:
        return _WebhookResource()

    @functools.cached_property
    def with_streaming_response(self) -> _WebhookResource:
        return _WebhookResource()

    def create(self, *, name: str) -> str:
        return name

    def list(self, *, name: str) -> str:
        return name

    def delete(self) -> None:
        raise AssertionError("unacknowledged sibling must not dispatch")

    def future_operation(self) -> None:
        raise AssertionError("unknown sibling must not dispatch")

    def unwrap(self) -> None:
        raise AssertionError("existing webhook helper must remain guarded")


class _OpenAIWebhooksClient:
    @functools.cached_property
    def webhooks(self) -> _WebhookResource:
        return _WebhookResource()

    @functools.cached_property
    def with_raw_response(self) -> _OpenAIWebhooksClient:
        return _OpenAIWebhooksClient()

    @functools.cached_property
    def with_streaming_response(self) -> _OpenAIWebhooksClient:
        return _OpenAIWebhooksClient()

    def close(self) -> None:
        return None


_OpenAIWebhooksClient.__module__ = "openai._client"
_WebhookResource.__module__ = "openai.resources.webhooks"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "azure_openai", "openai_compatible", "together"])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "path",
    [
        "webhooks.create",
        "webhooks.event_types.list",
        "webhooks.with_raw_response.create",
        "webhooks.with_streaming_response.create",
        "with_raw_response.webhooks.create",
        "with_streaming_response.webhooks.create",
        "webhooks.event_types.with_raw_response.list",
        "with_streaming_response.webhooks.event_types.list",
    ],
)
async def test_exact_webhook_leaf_acknowledgment_preserves_dispatch_and_guards_siblings(
    provider: str, mode: str, path: str
) -> None:
    # Arrange: all provider/control-plane boundaries are local doubles.
    raw = _OpenAIWebhooksClient()
    wrapper_type = AsyncSolwyn if mode == "async" else Solwyn
    reporter_name = "AsyncMetadataReporter" if mode == "async" else "MetadataReporter"
    budget_name = "AsyncBudgetEnforcer" if mode == "async" else "BudgetEnforcer"
    with (
        patch(f"solwyn.client.{reporter_name}", autospec=True) as reporter,
        patch(f"solwyn.client.{budget_name}", autospec=True) as budget,
    ):
        wrapper = wrapper_type(
            raw,
            provider=provider,
            api_key=VALID_API_KEY,
            on_unmetered="raise",
            acknowledge_untracked={path},
        )
        try:
            # Act: each prefix stays guarded, while the acknowledged callable is raw.
            guarded_resource: object = wrapper
            raw_resource: object = raw
            parts = path.split(".")
            for part in parts[:-1]:
                guarded_resource = getattr(guarded_resource, part)
                raw_resource = getattr(raw_resource, part)
                assert guarded_resource is not raw_resource
            method = getattr(guarded_resource, parts[-1])
            assert method == getattr(raw_resource, parts[-1])
            assert method(name="endpoint-name") == "endpoint-name"

            # Assert: neither reviewed siblings nor future additions inherit the token.
            for sibling, kind in (("delete", "unmetered_spend"), ("future_operation", "unknown")):
                with pytest.raises(UntrackedSpendSurfaceError) as error:
                    getattr(guarded_resource, sibling)
                assert error.value.surface == ".".join((*parts[:-1], sibling))
                if ".event_types" not in path or sibling == "future_operation":
                    assert error.value.kind == kind
            with pytest.raises(UntrackedSpendSurfaceError) as error:
                _ = wrapper.webhooks.unwrap
            assert error.value.kind == "unmetered_spend"
            budget.return_value.check_budget.assert_not_called()
            reporter.return_value.report_settlement.assert_not_called()
        finally:
            closed = wrapper.close()
            if inspect.isawaitable(closed):
                await closed
