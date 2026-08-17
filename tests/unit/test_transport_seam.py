"""Control-plane HTTP construction honors an injected transport everywhere."""

from __future__ import annotations

import ast
import gc
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID, call_uuid

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, ProviderName
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.client import Solwyn
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter

_API_URL = "http://control-plane.test"


class _Recorder:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/api/v1/budgets/check":
            return httpx.Response(200, json=ALLOW_BUDGET_RESPONSE)
        if request.url.path == "/api/v1/budgets/lease":
            return httpx.Response(
                200,
                json={
                    "eligible": True,
                    "allowed": True,
                    "lease_id": "lease_transport_seam",
                    "generation": 1,
                    "granted_tokens": 100_000,
                    "refresh_interval_s": 300.0,
                    "lease_length_s": 600.0,
                    "headroom_share_tokens": 50_000,
                    "posture": {"mode": "alert_only", "on_unreachable": "fail_open"},
                    "final_grant": False,
                    "project_id": VALID_PROJECT_ID,
                    "mode": "alert_only",
                    "budget_limit": 100.0,
                    "current_usage": 20.0,
                    "remaining_budget": 80.0,
                },
            )
        if request.url.path == "/api/v1/budgets/confirm":
            return httpx.Response(204)
        return httpx.Response(202, json={"ingested": 1, "rejected": []})


def _check(enforcer: BudgetEnforcer) -> None:
    result = enforcer.check_budget(
        estimated_input_tokens=10,
        model="gpt-5.5",
        provider="openai",
    )
    assert result.allowed is True


async def _acheck(enforcer: AsyncBudgetEnforcer) -> None:
    result = await enforcer.check_budget(
        estimated_input_tokens=10,
        model="gpt-5.5",
        provider="openai",
    )
    assert result.allowed is True


@pytest.mark.unit
def test_control_plane_component_transport_is_keyword_only() -> None:
    for component in (
        BudgetEnforcer,
        AsyncBudgetEnforcer,
        MetadataReporter,
        AsyncMetadataReporter,
    ):
        parameter = inspect.signature(component).parameters["transport"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    for component in (BudgetEnforcer, AsyncBudgetEnforcer):
        parameter = inspect.signature(component).parameters["control_plane_breaker"]
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.unit
def test_sync_enforcer_constructor_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = BudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    try:
        _check(enforcer)
    finally:
        enforcer.close()

    assert "/api/v1/budgets/check" in recorder.paths


@pytest.mark.unit
def test_sync_enforcer_fork_reset_reuses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = BudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    enforcer._reset_after_fork_in_child()
    try:
        _check(enforcer)
    finally:
        enforcer.close()

    assert "/api/v1/budgets/check" in recorder.paths


@pytest.mark.unit
def test_sync_reporter_exit_client_uses_injected_transport() -> None:
    recorder = _Recorder()
    reporter = MetadataReporter(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    client = reporter._new_exit_http_client()
    try:
        response = client.post(f"{_API_URL}/api/v1/budgets/confirm")
    finally:
        client.close()
        reporter.close()

    assert response.status_code == 204
    assert recorder.paths.count("/api/v1/budgets/confirm") == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_enforcer_constructor_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = AsyncBudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    try:
        await _acheck(enforcer)
    finally:
        await enforcer.close()

    assert "/api/v1/budgets/check" in recorder.paths


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_enforcer_fork_reset_reuses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = AsyncBudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    enforcer._reset_after_fork_in_child()
    try:
        await _acheck(enforcer)
    finally:
        await enforcer.close()

    assert "/api/v1/budgets/check" in recorder.paths


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_reporter_exit_client_uses_injected_transport() -> None:
    recorder = _Recorder()
    reporter = AsyncMetadataReporter(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    client = reporter._new_exit_http_client()
    try:
        response = client.post(f"{_API_URL}/api/v1/budgets/confirm")
    finally:
        client.close()
        await reporter.close()

    assert response.status_code == 204
    assert recorder.paths.count("/api/v1/budgets/confirm") == 1


@pytest.mark.unit
def test_async_reporter_gc_exit_flush_retains_transport_without_retaining_reporter() -> None:
    recorder = _Recorder()
    reporter = AsyncMetadataReporter(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )
    reporter.report_confirm(
        BudgetConfirmRequest(
            reservation_id="res_transport_seam",
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            call_id=call_uuid("transport-seam-finalizer"),
            token_details=TokenDetails(input_tokens=10, output_tokens=5),
        )
    )

    del reporter
    gc.collect()

    assert recorder.paths.count("/api/v1/budgets/confirm") == 1


@pytest.mark.unit
def test_sync_enforcer_surrender_drain_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = BudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        holder_id="holder_transport_seam",
        transport=httpx.MockTransport(recorder.handler),
    )

    result = enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=10,
        model="gpt-5.5",
        provider="openai",
        agent_run_id="run_transport_seam",
        call_id=call_uuid("transport-seam-surrender"),
    )
    assert result.lease_id == "lease_transport_seam"

    enforcer.close()

    assert "/api/v1/budgets/lease/surrender" in recorder.paths


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_enforcer_surrender_drain_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = AsyncBudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        holder_id="holder_transport_seam",
        transport=httpx.MockTransport(recorder.handler),
    )

    result = await enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=10,
        model="gpt-5.5",
        provider="openai",
        agent_run_id="run_transport_seam",
        call_id=call_uuid("async-transport-seam-surrender"),
    )
    assert result.lease_id == "lease_transport_seam"

    await enforcer.close()

    assert "/api/v1/budgets/lease/surrender" in recorder.paths


@pytest.mark.unit
def test_solwyn_threads_transport_to_budget_and_reporter() -> None:
    recorder = _Recorder()
    transport = httpx.MockTransport(recorder.handler)
    provider_client = MagicMock()
    provider_client.__class__.__module__ = "openai._client"
    provider_client.__class__.__name__ = "OpenAI"
    provider_client.with_options.return_value = provider_client
    provider_client.chat.completions.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    )
    solwyn = Solwyn(
        provider_client,
        api_key=VALID_API_KEY,
        api_url=_API_URL,
        control_plane_transport=transport,
        reporter_flush_interval=3600.0,
    )

    solwyn.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello"}],
    )
    solwyn.close()

    assert recorder.paths.count("/api/v1/budgets/check") == 1
    assert recorder.paths.count("/api/v1/budgets/confirm") == 1
    assert recorder.paths.count("/api/v1/metadata/ingest") == 1


@pytest.mark.unit
def test_http_clients_are_only_constructed_in_component_factories() -> None:
    source_root = Path(__file__).parents[2] / "src" / "solwyn"
    hits: list[str] = []
    violations: list[str] = []

    for path in source_root.rglob("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "httpx":
                continue
            if node.func.attr not in {"Client", "AsyncClient"}:
                continue
            owner: ast.AST | None = node
            while owner is not None and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents.get(owner)
            owner_name = (
                owner.name if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else ""
            )
            location = f"{path.relative_to(source_root)}:{node.lineno}:{owner_name}"
            hits.append(location)
            if path.name not in {"budget.py", "reporter.py"} or owner_name not in {
                "_new_http_client",
                "_new_async_http_client",
                "_new_exit_http_client",
            }:
                violations.append(location)

    assert not violations, f"direct httpx client construction bypasses factories: {violations}"
    assert len(hits) == 6, f"expected six component-factory construction sites, got {hits}"
