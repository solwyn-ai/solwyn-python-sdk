"""Structural guards against content capture or propagation through OTel."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SDK_SRC = Path(__file__).resolve().parents[2] / "src"


def _violations(source: str) -> list[int]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name] = name.name
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                aliases[name.asname or name.name] = f"{node.module}.{name.name}"

    def dotted(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{dotted(node.value)}.{node.attr}"
        return ""

    violations = []
    banned = {"record_exception", "content_mode", "max_content_bytes"}
    for node in ast.walk(tree):
        name = (
            node.id
            if isinstance(node, ast.Name)
            else node.attr
            if isinstance(node, ast.Attribute)
            else node.arg
            if isinstance(node, ast.keyword)
            else node.value
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            else None
        )
        if name is not None and any(part in banned for part in name.split(".")):
            violations.append(node.lineno)
        if not isinstance(node, ast.Call):
            continue
        target = dotted(node.func)
        if target.endswith("propagate.inject"):
            violations.append(node.lineno)
        if target.rsplit(".", 1)[-1] == "set_attribute":
            key = (
                node.args[0]
                if node.args
                else next(
                    (kw.value for kw in node.keywords if kw.arg == "key"),
                    None,
                )
            )
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.startswith("solwyn.")
            ):
                violations.append(node.lineno)
        if target.rsplit(".", 1)[-1] in {"set_status", "Status"}:
            description = (
                node.args[1]
                if len(node.args) > 1
                else next(
                    (kw.value for kw in node.keywords if kw.arg == "description"),
                    None,
                )
            )
            if description is not None and not isinstance(description, ast.Constant):
                violations.append(node.lineno)
    return violations


@pytest.mark.unit
def test_otel_privacy_guard_covers_all_source_modules() -> None:
    violations = {
        str(path.relative_to(SDK_SRC)): lines
        for path in SDK_SRC.rglob("*.py")
        if (lines := _violations(path.read_text()))
    }
    assert not violations


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        "span.record_exception(exc)",
        "options = {'content_mode': 'content'}",
        "options = {'max_content_bytes': 20}",
        "options.content_mode = 'content'",
        "from opentelemetry import propagate; propagate.inject(headers)",
        "from opentelemetry.propagate import inject as send; send(headers)",
        "import opentelemetry.propagate as p; p.inject(headers)",
        "span.set_attribute(key, value)",
        "span.set_attribute('gen_ai.prompt', value)",
        "span.set_attribute(key='customer.tags', value=tags)",
        "span.set_status(code, description=str(exc))",
        "span.set_status(Status(StatusCode.ERROR, str(exc)))",
        "span.set_status(Status(StatusCode.ERROR, description=str(exc)))",
        "from opentelemetry.trace import Status as S; span.set_status(S(code, str(exc)))",
        "set_attribute('customer.tags', tags)",
    ],
)
def test_otel_privacy_guard_rejects_forbidden_operations(source: str) -> None:
    assert _violations(source)


@pytest.mark.unit
def test_otel_privacy_guard_accepts_literal_solwyn_attribute_names() -> None:
    assert not _violations("span.set_attribute('solwyn.call_id', call_id)")
    assert not _violations("span.set_attribute(key='solwyn.provider', value=provider)")
    assert not _violations("span.set_status(Status(code, 'unavailable'))")


@pytest.mark.unit
def test_event_builders_and_call_helpers_require_explicit_trace_threading() -> None:
    for filename in ("client.py", "_base.py"):
        tree = ast.parse((SDK_SRC / "solwyn" / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
                names = {kw.arg for kw in node.keywords}
                if "agent_run" in names or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"_build_metadata_event", "_build_error_event"}
                ):
                    assert "trace" in names, node.lineno
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {arg.arg for arg in node.args.kwonlyargs}
                if "agent_run" in names:
                    assert "trace" in names, node.lineno


@pytest.mark.unit
def test_background_tasks_always_start_in_fresh_contexts() -> None:
    count = 0
    for filename in ("reporter.py", "budget.py"):
        tree = ast.parse((SDK_SRC / "solwyn" / filename).read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"create_task", "ensure_future"}
            ):
                continue
            count += 1
            context = next((kw.value for kw in node.keywords if kw.arg == "context"), None)
            assert isinstance(context, ast.Call), (filename, node.lineno)
            assert ast.unparse(context) == "contextvars.Context()", (filename, node.lineno)
    assert count == 9
