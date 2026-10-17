"""Structural guards against content capture or propagation through OTel."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from solwyn._otel import CALL_SPAN_ATTRIBUTES

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

    # Preserve method identity through simple local aliases, too.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Attribute)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = dotted(node.value)

    def literal_solwyn_name(node: ast.expr | None) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("solwyn.")
        )

    def literal_solwyn_key(node: ast.expr | None) -> bool:
        return (
            literal_solwyn_name(node)
            and isinstance(node, ast.Constant)
            and node.value in CALL_SPAN_ATTRIBUTES
        )

    def literal_solwyn_attributes(node: ast.expr) -> bool:
        # Dict expansion has a None key, so even a partial literal cannot hide
        # arbitrary attributes behind **attrs.
        return isinstance(node, ast.Dict) and all(literal_solwyn_key(key) for key in node.keys)

    violations = []
    banned = re.compile(r"\b(?:record_exception|content_mode|max_content_bytes)\b")
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
        if name is not None and banned.search(name):
            violations.append(node.lineno)
        if not isinstance(node, ast.Call):
            continue
        method = dotted(node.func).rsplit(".", 1)[-1]
        if method in {"inject", "record_exception", "add_event"}:
            violations.append(node.lineno)
        if method in {"start_span", "start_as_current_span", "update_name"}:
            span_name = (
                node.args[0]
                if node.args
                else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
            )
            if not literal_solwyn_name(span_name):
                violations.append(node.lineno)
        if method == "Link" and (
            len(node.args) > 1
            or any(isinstance(arg, ast.Starred) for arg in node.args)
            or any(kw.arg in {None, "attributes"} for kw in node.keywords)
        ):
            violations.append(node.lineno)
        attributes = [kw.value for kw in node.keywords if kw.arg == "attributes"]
        if method == "set_attributes":
            if node.args:
                attributes.append(node.args[0])
            elif not attributes:
                violations.append(node.lineno)
        if method in {"start_span", "start_as_current_span"} and len(node.args) > 3:
            attributes.append(node.args[3])
        if any(not literal_solwyn_attributes(value) for value in attributes):
            violations.append(node.lineno)
        if method in {
            "set_attribute",
            "set_attributes",
            "start_span",
            "start_as_current_span",
            "update_name",
        } and any(kw.arg is None for kw in node.keywords):
            violations.append(node.lineno)
        if method == "set_attribute":
            key = (
                node.args[0]
                if node.args
                else next(
                    (kw.value for kw in node.keywords if kw.arg == "key"),
                    None,
                )
            )
            if not literal_solwyn_key(key):
                violations.append(node.lineno)
        if method in {"set_status", "Status"}:
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
        "tracer.start_span(name)",
        "tracer.start_span(f'chat {prompt}')",
        "tracer.start_span('provider.call')",
        "tracer.start_as_current_span(name=prompt)",
        "span.update_name(prompt)",
        "span.update_name('provider.call')",
        "span.update_name(**kwargs)",
        "span.set_attribute('solwyn.unknown', prompt)",
        "span.set_attributes({'solwyn.unknown': prompt})",
        "tracer.start_span('solwyn.call', attributes={'solwyn.unknown': prompt})",
        "Link(ctx, {'gen_ai.prompt': prompt})",
        "Link(ctx, {'solwyn.call_id': call_id})",
        "Link(ctx, attrs)",
        "Link(ctx, attributes={})",
        "from opentelemetry.trace import Link as L; L(ctx, attrs)",
        "Link(*args)",
        "s = span.set_attributes; s(attrs)",
        "s = span.update_name; s(prompt)",
        "options = {'content_mode': 'content'}",
        "options = {'max_content_bytes': 20}",
        "options.content_mode = 'content'",
        "client.with_options(content_mode='content')",
        "options['content_mode'] = 'content'",
        "options.max_content_bytes = 1",
        "content_mode = 'content'",
        "getattr(options, 'max_content_bytes')",
        "options = {'traces.content_mode': 'content'}",
        "options = {'traces/content_mode': 'content'}",
        "from opentelemetry import propagate; propagate.inject(headers)",
        "from opentelemetry.propagate import inject as send; send(headers)",
        "import opentelemetry.propagate as p; p.inject(headers)",
        "propagate.get_global_textmap().inject(headers)",
        "TraceContextTextMapPropagator().inject(headers)",
        "W3CBaggagePropagator().inject(headers)",
        "custom_propagator.inject(headers)",
        "from custom_propagator import inject as send; send(headers)",
        "from opentelemetry.trace import record_exception as record; record(exc)",
        "span.set_attribute(key, value)",
        "span.set_attribute('gen_ai.prompt', value)",
        "span.set_attribute(key='customer.tags', value=tags)",
        "span.set_status(code, description=str(exc))",
        "span.set_status(Status(StatusCode.ERROR, str(exc)))",
        "span.set_status(Status(StatusCode.ERROR, description=str(exc)))",
        "from opentelemetry.trace import Status as S; span.set_status(S(code, str(exc)))",
        "set_attribute('customer.tags', tags)",
        "span.set_attributes({'gen_ai.prompt': prompt})",
        "span.set_attributes(attrs)",
        "span.set_attributes({key: value})",
        "span.set_attributes({'solwyn.call_id': call_id, **attrs})",
        "span.set_attributes(attributes=attrs)",
        "span.set_attributes(attributes={'customer.tags': tags})",
        "span.set_attributes(**kwargs)",
        "span.set_attributes([('solwyn.call_id', call_id)])",
        "tracer.start_span('x', attributes={'gen_ai.prompt': prompt})",
        "tracer.start_as_current_span('x', attributes={'gen_ai.prompt': prompt})",
        "tracer.start_span('x', attributes=attrs)",
        "tracer.start_as_current_span('x', attributes=attrs)",
        "tracer.start_span('x', attributes={key: value})",
        "tracer.start_span('x', attributes={'solwyn.call_id': call_id, **attrs})",
        "tracer.start_span('x', **kwargs)",
        "tracer.start_as_current_span('x', **kwargs)",
        "tracer.start_span('x', context, kind, {'gen_ai.prompt': prompt})",
        "tracer.start_as_current_span('x', context, kind, attrs)",
        "Span(attributes={'gen_ai.prompt': prompt})",
        "span.add_event('x', attributes={'gen_ai.prompt': prompt})",
        "span.add_event('x', attributes=attrs)",
        "span.add_event('solwyn.call', attributes={'solwyn.call_id': call_id})",
        "span.add_event('x')",
    ],
)
def test_otel_privacy_guard_rejects_forbidden_operations(source: str) -> None:
    assert _violations(source)


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        "span.set_attribute('solwyn.call_id', call_id)",
        "span.set_attribute(key='solwyn.provider.served', value=provider)",
        "span.set_attributes({'solwyn.call_id': call_id, 'solwyn.provider.served': provider})",
        "span.set_attributes(attributes={'solwyn.provider.served': provider})",
        "span.set_attributes({})",
        "tracer.start_span('solwyn.call', attributes={'solwyn.call_id': call_id})",
        "tracer.start_as_current_span('solwyn.call', "
        "attributes={'solwyn.provider.served': provider})",
        "tracer.start_span('solwyn.call', context, kind, {'solwyn.call_id': call_id})",
        "span.set_status(Status(code, 'unavailable'))",
    ],
)
def test_otel_privacy_guard_accepts_literal_solwyn_attribute_names(source: str) -> None:
    assert not _violations(source)


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
def test_span_context_scopes_disable_implicit_exception_capture() -> None:
    count = 0
    for path in SDK_SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "use_span"
            ):
                continue
            count += 1
            # API positional flags: end_on_exit, exception capture, error status.
            assert len(node.args) == 4, (path, node.lineno)
            assert all(
                isinstance(arg, ast.Constant) and arg.value is False for arg in node.args[1:]
            )
    assert count == 1


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
