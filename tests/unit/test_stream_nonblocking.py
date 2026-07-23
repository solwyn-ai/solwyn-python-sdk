"""Stream completion must not block on Solwyn HTTP."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from conftest import call_uuid

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest
from solwyn.budget import BudgetEnforcer, _BudgetEnforcerBase

SDK_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "solwyn"


@pytest.mark.unit
def test_sync_on_complete_does_not_call_confirm_cost() -> None:
    """The sync on_complete closure in Solwyn._intercepted_call must NOT
    call budget.confirm_cost() -- it must use reporter.report_settlement()."""
    client_py = (SDK_SRC / "client.py").read_text()

    tree = ast.parse(client_py)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Solwyn":
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == "on_complete":
                    source_lines = client_py.splitlines()
                    fn_lines = source_lines[item.lineno - 1 : item.end_lineno]
                    fn_text = "\n".join(fn_lines)
                    assert "confirm_cost(" not in fn_text, (
                        "Solwyn._intercepted_call's on_complete closure must NOT "
                        "call budget.confirm_cost() — it blocks on httpx. "
                        "Use reporter.report_settlement() instead."
                    )
                    assert "report_settlement(" in fn_text, (
                        "Solwyn._intercepted_call's on_complete closure must "
                        "call reporter.report_settlement() for fire-and-forget."
                    )
                    return

    raise AssertionError("Could not find on_complete in Solwyn._intercepted_call")


@pytest.mark.unit
def test_async_on_complete_does_not_call_confirm_cost() -> None:
    """The async on_complete closure must also enqueue confirm fire-and-forget."""
    client_py = (SDK_SRC / "client.py").read_text()

    tree = ast.parse(client_py)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AsyncSolwyn":
            for item in ast.walk(node):
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "on_complete":
                    source_lines = client_py.splitlines()
                    fn_lines = source_lines[item.lineno - 1 : item.end_lineno]
                    fn_text = "\n".join(fn_lines)
                    assert "confirm_cost(" not in fn_text, (
                        "AsyncSolwyn._wrap_stream_async's on_complete closure must "
                        "not await budget.confirm_cost(); it blocks stream settlement. "
                        "Use reporter.report_settlement() instead."
                    )
                    assert "report_settlement(" in fn_text, (
                        "AsyncSolwyn._wrap_stream_async's on_complete closure must "
                        "call reporter.report_settlement() for fire-and-forget."
                    )
                    return

    raise AssertionError("Could not find on_complete in AsyncSolwyn._wrap_stream_async")


@pytest.mark.unit
def test_build_confirm_request_exists_on_base() -> None:
    """_BudgetEnforcerBase must expose build_confirm_request."""
    assert hasattr(_BudgetEnforcerBase, "build_confirm_request"), (
        "_BudgetEnforcerBase must have build_confirm_request method"
    )


@pytest.mark.unit
def test_build_confirm_request_returns_pydantic_model() -> None:
    """build_confirm_request must return a BudgetConfirmRequest, not a dict."""
    enforcer = BudgetEnforcer(
        api_url="http://localhost:8000",
        api_key="sk_test",
    )
    token_details = TokenDetails(
        input_tokens=10,
        output_tokens=20,
    )
    request = enforcer.build_confirm_request(
        reservation_id="r_test_123",
        model="gpt-5.5",
        token_details=token_details,
        provider="openai",
        call_id=call_uuid("call_stream_confirm"),
    )
    assert isinstance(request, BudgetConfirmRequest)
    assert request.reservation_id == "r_test_123"
    assert request.model == "gpt-5.5"
    assert request.provider == "openai"
    enforcer.close()


@pytest.mark.unit
def test_build_confirm_request_narrows_service_tier_to_wire_literals() -> None:
    """Adapters echo ARBITRARY tier strings; the confirm wire is a strict literal.

    A recognized tier rides the confirm verbatim; a novel echo narrows to None
    (settles at Standard rates) instead of 422-ing the strict Cloud-API model
    and stranding the reservation. The raw echo still reaches the API on the
    MetadataEvent, which is a bounded str.
    """
    enforcer = BudgetEnforcer(api_url="http://localhost:8000", api_key="sk_test")
    token_details = TokenDetails(input_tokens=10, output_tokens=20)

    def _build(tier: str | None) -> BudgetConfirmRequest:
        return enforcer.build_confirm_request(
            reservation_id="r_test_123",
            model="gpt-5.5",
            token_details=token_details,
            provider="openai",
            call_id=call_uuid("call_tier"),
            service_tier=tier,
        )

    assert _build("priority").service_tier == "priority"
    assert _build("flex").service_tier == "flex"
    assert _build(None).service_tier is None
    assert _build("hyperspeed").service_tier is None  # novel echo -> Standard
    enforcer.close()
