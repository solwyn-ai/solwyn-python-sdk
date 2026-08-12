"""Mandatory classified drift canaries for supported real provider SDKs."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import warnings
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from solwyn._surface_graph import SurfaceCanaryError, audit_public_surface
from solwyn._surfaces import (
    AttributeShape,
    CapabilityScope,
    SurfaceContext,
    SurfaceKind,
    SurfaceRule,
    SurfaceSelector,
    SurfaceSource,
    UsageBasis,
    resolve_surface_rule,
)

ROOT = Path(__file__).parents[2]
CANARY_FAMILY_ENV = "SOLWYN_SURFACE_CANARY_FAMILY"
SUPPORTED_FAMILIES = {
    "anthropic",
    "bedrock",
    "google-genai",
    "google-generativeai",
    "openai",
    "together",
}
TEST_CONTEXT = SurfaceContext(
    provider="test",
    dialect="test",
    client_shape="test_sdk",
    mode="sync",
)


def _capture_module() -> ModuleType:
    path = ROOT / "scripts" / "capture_surface_inventory.py"
    spec = importlib.util.spec_from_file_location("capture_surface_inventory_canary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load inventory capture script at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rule(
    path: str,
    *,
    kind: SurfaceKind,
    descriptor_category: str,
    return_shape: str,
) -> SurfaceRule:
    kwargs: dict[str, object] = {}
    if kind is SurfaceKind.METERED:
        kwargs["usage_basis"] = UsageBasis.PROVIDER
    if kind is SurfaceKind.UNMETERED_SPEND:
        kwargs["acknowledgment_token"] = path
        kwargs["capability_scope"] = CapabilityScope.OPERATION
    if kind in {SurfaceKind.BLOCKED, SurfaceKind.UNSUPPORTED}:
        kwargs["reason"] = "test refusal"
    return SurfaceRule(
        rule_id=f"test.{path.replace('.', '-')}",
        surface=path,
        selectors=(SurfaceSelector.from_context(TEST_CONTEXT),),
        kind=kind,
        source=SurfaceSource.RAW,
        expected_shapes=(AttributeShape(descriptor_category, return_shape),),
        **kwargs,
    )


class _Resource:
    marker = "safe"

    def create(self) -> None:
        raise AssertionError("provider operations must never be invoked")


class _Client:
    def __init__(self) -> None:
        self.resources = _Resource()


@pytest.mark.unit
def test_canary_accepts_exact_rules_and_never_invokes_operations() -> None:
    # Arrange
    rules = (
        _rule(
            "resources",
            kind=SurfaceKind.NAMESPACE,
            descriptor_category="attribute",
            return_shape="resource",
        ),
        _rule(
            "resources.create",
            kind=SurfaceKind.UNMETERED_SPEND,
            descriptor_category="function",
            return_shape="callable",
        ),
        _rule(
            "resources.marker",
            kind=SurfaceKind.METADATA,
            descriptor_category="attribute",
            return_shape="scalar",
        ),
    )

    # Act
    rows = audit_public_surface(
        _Client(),
        context=TEST_CONTEXT,
        client_family="test",
        installed_version="1.2.3",
        rules=rules,
    )

    # Assert
    assert [row.path for row in rows] == [
        "resources",
        "resources.create",
        "resources.marker",
    ]
    assert [row.rule_id for row in rows] == [
        "test.resources",
        "test.resources-create",
        "test.resources-marker",
    ]


@pytest.mark.unit
def test_new_child_fails_with_family_version_and_exact_unknown_path() -> None:
    # Arrange
    rules = (
        _rule(
            "resources",
            kind=SurfaceKind.NAMESPACE,
            descriptor_category="attribute",
            return_shape="resource",
        ),
        _rule(
            "resources.marker",
            kind=SurfaceKind.METADATA,
            descriptor_category="attribute",
            return_shape="scalar",
        ),
    )

    # Act
    with pytest.raises(SurfaceCanaryError) as caught:
        audit_public_surface(
            _Client(),
            context=TEST_CONTEXT,
            client_family="test-family",
            installed_version="9.8.7",
            rules=rules,
        )

    # Assert
    assert caught.value.path == "resources.create"
    assert caught.value.stage == "unknown_classification"
    assert "test-family" in str(caught.value)
    assert "9.8.7" in str(caught.value)
    assert "resources.create" in str(caught.value)


@pytest.mark.unit
def test_declared_resource_without_known_children_is_still_traversed() -> None:
    # Arrange
    rules = (
        _rule(
            "resources",
            kind=SurfaceKind.NAMESPACE,
            descriptor_category="attribute",
            return_shape="resource",
        ),
    )

    # Act
    with pytest.raises(SurfaceCanaryError) as caught:
        audit_public_surface(
            _Client(),
            context=TEST_CONTEXT,
            client_family="test",
            installed_version="1",
            rules=rules,
        )

    # Assert
    assert caught.value.path == "resources.create"
    assert caught.value.stage == "unknown_classification"


@pytest.mark.unit
def test_ambiguous_rule_resolution_preserves_exact_canary_context() -> None:
    # Arrange
    rule = _rule(
        "resources",
        kind=SurfaceKind.NAMESPACE,
        descriptor_category="attribute",
        return_shape="resource",
    )
    client_family = "test-provider/test-client"
    installed_version = "9.8.7"

    # Act
    with pytest.raises(SurfaceCanaryError) as caught:
        audit_public_surface(
            _Client(),
            context=TEST_CONTEXT,
            client_family=client_family,
            installed_version=installed_version,
            rules=(rule, replace(rule, rule_id="test.resources.duplicate")),
        )

    # Assert
    assert caught.value.stage == "rule_resolution"
    assert caught.value.client_family == client_family
    assert caught.value.installed_version == installed_version
    assert caught.value.path == "resources"
    assert caught.value.cause_type == "RuntimeError"
    assert str(caught.value) == (
        "Surface canary failed for test-provider/test-client 9.8.7 "
        "at 'resources' during rule_resolution (RuntimeError)"
    )


@pytest.mark.unit
def test_floor_shape_variants_remain_scoped_to_anthropic() -> None:
    # Arrange
    eager_shape = AttributeShape("attribute", "resource")

    # Act
    anthropic_rule = resolve_surface_rule(
        context=SurfaceContext(
            provider="anthropic",
            dialect="anthropic",
            client_shape="anthropic_sdk",
            mode="sync",
        ),
        path="with_raw_response.completions",
        source=SurfaceSource.RAW,
    )
    openai_rule = resolve_surface_rule(
        context=SurfaceContext(
            provider="openai",
            dialect="openai",
            client_shape="openai_sdk",
            mode="sync",
        ),
        path="with_raw_response.completions",
        source=SurfaceSource.RAW,
    )

    # Assert
    assert anthropic_rule is not None
    assert openai_rule is not None
    assert anthropic_rule.accepts_shape(eager_shape)
    assert not openai_rule.accepts_shape(eager_shape)


@pytest.mark.unit
def test_shape_drift_fails_without_evaluating_a_terminal_descriptor() -> None:
    # Arrange
    class DriftClient:
        def __init__(self) -> None:
            self.evaluations = 0

        @property
        def operation(self) -> object:
            self.evaluations += 1
            raise AssertionError("terminal descriptor must remain unevaluated")

    client = DriftClient()
    rules = (
        _rule(
            "evaluations",
            kind=SurfaceKind.METADATA,
            descriptor_category="attribute",
            return_shape="scalar",
        ),
        _rule(
            "operation",
            kind=SurfaceKind.UNMETERED_SPEND,
            descriptor_category="function",
            return_shape="callable",
        ),
    )

    # Act
    with pytest.raises(SurfaceCanaryError) as caught:
        audit_public_surface(
            client,
            context=TEST_CONTEXT,
            client_family="test",
            installed_version="1",
            rules=rules,
        )

    # Assert
    assert caught.value.path == "operation"
    assert caught.value.stage == "shape_drift"
    assert client.evaluations == 0


@pytest.mark.unit
def test_declared_namespace_failure_is_exact_and_content_free() -> None:
    # Arrange
    class BrokenClient:
        @property
        def resources(self) -> object:
            raise RuntimeError("PRIVATE_DESCRIPTOR_CONTENT")

    rules = (
        _rule(
            "resources",
            kind=SurfaceKind.NAMESPACE,
            descriptor_category="property",
            return_shape="resource",
        ),
        _rule(
            "resources.create",
            kind=SurfaceKind.UNMETERED_SPEND,
            descriptor_category="function",
            return_shape="callable",
        ),
    )

    # Act
    with pytest.raises(SurfaceCanaryError) as caught:
        audit_public_surface(
            BrokenClient(),
            context=TEST_CONTEXT,
            client_family="test",
            installed_version="1",
            rules=rules,
        )

    # Assert
    assert caught.value.path == "resources"
    assert caught.value.stage == "namespace_evaluation"
    assert "PRIVATE_DESCRIPTOR_CONTENT" not in str(caught.value)


@pytest.mark.unit
def test_cycles_fail_deterministically_and_repeated_resources_remain_distinct() -> None:
    # Arrange
    cycle = _Resource()
    cycle.loop = cycle

    class CycleClient:
        def __init__(self) -> None:
            self.resources = cycle

    cycle_rules = (
        _rule(
            "resources",
            kind=SurfaceKind.NAMESPACE,
            descriptor_category="attribute",
            return_shape="resource",
        ),
        _rule(
            "resources.create",
            kind=SurfaceKind.UNMETERED_SPEND,
            descriptor_category="function",
            return_shape="callable",
        ),
        _rule(
            "resources.loop",
            kind=SurfaceKind.NAMESPACE,
            descriptor_category="attribute",
            return_shape="resource",
        ),
        _rule(
            "resources.loop.create",
            kind=SurfaceKind.UNMETERED_SPEND,
            descriptor_category="function",
            return_shape="callable",
        ),
    )
    # Act
    with pytest.raises(SurfaceCanaryError) as caught:
        audit_public_surface(
            CycleClient(),
            context=TEST_CONTEXT,
            client_family="test",
            installed_version="1",
            rules=cycle_rules,
        )
    # Assert
    assert caught.value.path == "resources.loop"
    assert caught.value.stage == "cycle"

    # Arrange
    shared = _Resource()

    class AliasClient:
        def __init__(self) -> None:
            self.left = shared
            self.right = shared

    alias_rules = tuple(
        _rule(
            path,
            kind=kind,
            descriptor_category=descriptor,
            return_shape=shape,
        )
        for path, kind, descriptor, shape in (
            ("left", SurfaceKind.NAMESPACE, "attribute", "resource"),
            ("left.create", SurfaceKind.UNMETERED_SPEND, "function", "callable"),
            ("left.marker", SurfaceKind.METADATA, "attribute", "scalar"),
            ("right", SurfaceKind.NAMESPACE, "attribute", "resource"),
            ("right.create", SurfaceKind.UNMETERED_SPEND, "function", "callable"),
            ("right.marker", SurfaceKind.METADATA, "attribute", "scalar"),
        )
    )
    # Act
    rows = audit_public_surface(
        AliasClient(),
        context=TEST_CONTEXT,
        client_family="test",
        installed_version="1",
        rules=alias_rules,
    )
    # Assert
    assert {row.path for row in rows} >= {"left.create", "right.create"}


@pytest.mark.unit
def test_bedrock_service_model_only_operation_is_not_invented_as_callable() -> None:
    # Arrange
    class BedrockLike:
        def converse(self) -> None:
            raise AssertionError("provider operations must never be invoked")

    rules = (
        _rule(
            "converse",
            kind=SurfaceKind.METERED,
            descriptor_category="function",
            return_shape="callable",
        ),
        _rule(
            "future_operation",
            kind=SurfaceKind.BLOCKED,
            descriptor_category="service_model_operation",
            return_shape="service_model_only",
        ),
    )
    # Act
    rows = audit_public_surface(
        BedrockLike(),
        context=TEST_CONTEXT,
        client_family="bedrock-test",
        installed_version="1",
        service_model_operations={"converse", "future_operation"},
        rules=rules,
    )
    by_path = {row.path: row for row in rows}

    # Assert
    assert by_path["converse"].observation_source == "public_attribute"
    assert by_path["converse"].return_shape == "callable"
    assert by_path["future_operation"].observation_source == "service_model_operation"
    assert by_path["future_operation"].return_shape == "service_model_only"


def _selected_shape_keys() -> tuple[str, ...]:
    capture = _capture_module()
    family = os.environ.get(CANARY_FAMILY_ENV)
    if family is None:
        return capture.shape_keys()
    if family not in SUPPORTED_FAMILIES:
        raise RuntimeError(f"unsupported canary family: {family}")
    return capture.shape_keys_for_families({family})


def _require_provider_sdk(shape_key: str) -> object:
    """Import each selected provider directly; absence is a hard failure."""

    if shape_key.startswith(("openai_", "azure_openai_")):
        import openai

        return openai
    if shape_key.startswith("anthropic_"):
        import anthropic

        return anthropic
    if shape_key.startswith("together_native_"):
        import together

        return together
    if shape_key.startswith("google_genai_"):
        from google import genai

        return genai
    if shape_key == "google_generativeai_sync":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as legacy_genai

        return legacy_genai
    if shape_key.startswith("bedrock_"):
        import aioboto3
        import boto3

        return boto3, aioboto3
    raise RuntimeError(f"unknown canary shape: {shape_key}")


def _context_for_spec(spec: Any) -> SurfaceContext:
    return SurfaceContext(
        provider=spec.provider,
        dialect={
            "anthropic": "anthropic",
            "azure_openai": "openai",
            "bedrock": "bedrock",
            "google": "google",
            "openai": "openai",
            "openai_compatible": "openai",
            "together": "openai",
        }[spec.provider],
        client_shape=spec.client_shape,
        mode=spec.mode,
    )


async def _audit_real_shape(shape_key: str) -> tuple[object, ...]:
    capture = _capture_module()
    spec = capture._SHAPES_BY_KEY[shape_key]
    with capture._deny_socket_access() as socket_counter:
        async with capture._client_for(spec) as client:
            service_operations = (
                capture._bedrock_service_model_operations(client)
                if spec.client_shape.startswith("bedrock_")
                else ()
            )
            rows = audit_public_surface(
                client,
                context=_context_for_spec(spec),
                client_family=shape_key,
                installed_version=version(spec.distribution),
                service_model_operations=service_operations,
            )
    assert socket_counter.attempts == 0
    return rows


@pytest.mark.unit
@pytest.mark.parametrize("shape_key", _selected_shape_keys())
def test_real_sdk_graph_has_no_unknown_ambiguous_or_shape_drifted_paths(
    shape_key: str,
) -> None:
    # Arrange
    _require_provider_sdk(shape_key)

    # Act
    first = asyncio.run(_audit_real_shape(shape_key))
    second = asyncio.run(_audit_real_shape(shape_key))

    # Assert
    assert first == second
    assert first
    assert [row.path for row in first] == sorted(row.path for row in first)
    assert len({row.path for row in first}) == len(first)


@pytest.mark.unit
def test_canary_is_required_by_local_and_structural_interval_gates() -> None:
    # Arrange
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"

    # Act
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    provider_steps = workflow["jobs"]["provider-surface-inventory"]["steps"]
    provider_commands = "\n".join(str(step.get("run", "")) for step in provider_steps)
    canary_steps = [
        step
        for step in provider_steps
        if "tests/unit/test_surface_canary.py" in str(step.get("run", ""))
    ]
    standard_test_steps = str(workflow["jobs"]["test"]["steps"])
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    source = Path(__file__).read_text(encoding="utf-8")

    # Assert
    assert "pytest tests/unit/test_surface_canary.py" in provider_commands
    assert "pyyaml>=6.0" in provider_commands
    assert len(canary_steps) == 1
    assert canary_steps[0]["if"] == "always()"
    assert canary_steps[0]["env"][CANARY_FAMILY_ENV] == "${{ matrix.family }}"
    assert workflow["permissions"] == {"contents": "read"}
    assert "aioboto3" not in standard_test_steps
    assert "--ignore=tests/unit/test_surface_canary.py" in standard_test_steps

    assert "test: test-unit check-surface-canary" in makefile
    assert "check-surface-canary:" in makefile
    assert "--ignore=tests/unit/test_surface_canary.py" in makefile
    assert "tests/unit/test_surface_canary.py" in makefile
    assert "aioboto3>=13.0" in makefile

    assert "importor" + "skip" not in source
