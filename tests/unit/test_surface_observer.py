"""Provider-agnostic tests for the offline public-surface observer."""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

import pytest


def _surface_graph() -> ModuleType:
    return importlib.import_module("solwyn._surface_graph")


class _RootResource:
    marker = "root"

    def create(self) -> None:
        raise AssertionError("provider operations must never be invoked")


class _FakeClient:
    api_version = "test"
    _private_transport = object()

    def __init__(self) -> None:
        self.namespace_evaluations = 0
        self.unsafe_property_evaluations = 0

    @functools.cached_property
    def resources(self) -> _RootResource:
        self.namespace_evaluations += 1
        return _RootResource()

    @property
    def unsafe_property(self) -> str:
        self.unsafe_property_evaluations += 1
        raise AssertionError("terminal descriptors must remain unevaluated")

    def create(self) -> None:
        raise AssertionError("provider operations must never be invoked")


class _BrokenStaticClient:
    def __dir__(self) -> list[str]:
        return ["broken"]


class _BrokenNamespaceClient:
    @property
    def resources(self) -> object:
        raise ValueError("SECRET_DESCRIPTOR_VALUE")


class _NestedResource:
    def __init__(self) -> None:
        self.nested = _RootResource()


class _NestedClient:
    def __init__(self) -> None:
        self.resources = _NestedResource()


class _CycleResource:
    def __init__(self) -> None:
        self.self_reference = self


class _CycleClient:
    def __init__(self) -> None:
        self.resources = _CycleResource()


@pytest.mark.unit
def test_observes_public_roots_deterministically_without_invoking_operations() -> None:
    graph = _surface_graph()
    client = _FakeClient()

    observations = graph.observe_public_surface(client, namespaces={"resources"})

    assert [item.path for item in observations] == sorted(item.path for item in observations)
    assert {item.path: (item.descriptor_category, item.return_shape) for item in observations} == {
        "api_version": ("attribute", "scalar"),
        "create": ("function", "callable"),
        "namespace_evaluations": ("attribute", "scalar"),
        "resources": ("cached_property", "resource"),
        "resources.create": ("function", "callable"),
        "resources.marker": ("attribute", "scalar"),
        "unsafe_property": ("property", "unevaluated_descriptor"),
        "unsafe_property_evaluations": ("attribute", "scalar"),
    }
    assert client.namespace_evaluations == 1
    assert client.unsafe_property_evaluations == 0


@pytest.mark.unit
def test_static_inspection_failure_identifies_the_exact_path() -> None:
    graph = _surface_graph()

    with pytest.raises(graph.SurfaceInspectionError) as caught:
        graph.observe_public_surface(_BrokenStaticClient())

    assert caught.value.path == "broken"
    assert caught.value.stage == "static_inspection"


@pytest.mark.unit
def test_namespace_evaluation_failure_is_path_exact_and_content_free() -> None:
    graph = _surface_graph()

    with pytest.raises(graph.SurfaceInspectionError) as caught:
        graph.observe_public_surface(_BrokenNamespaceClient(), namespaces={"resources"})

    assert caught.value.path == "resources"
    assert caught.value.stage == "namespace_evaluation"
    assert "SECRET_DESCRIPTOR_VALUE" not in str(caught.value)


@pytest.mark.unit
def test_depth_exhaustion_identifies_the_namespace_that_would_descend_too_far() -> None:
    graph = _surface_graph()

    with pytest.raises(graph.SurfaceInspectionError) as caught:
        graph.observe_public_surface(
            _NestedClient(),
            namespaces={"resources", "resources.nested"},
            max_depth=1,
        )

    assert caught.value.path == "resources.nested"
    assert caught.value.stage == "depth_exhaustion"


@pytest.mark.unit
def test_cycle_detection_identifies_the_reentering_namespace() -> None:
    graph = _surface_graph()

    with pytest.raises(graph.SurfaceInspectionError) as caught:
        graph.observe_public_surface(
            _CycleClient(),
            namespaces={"resources", "resources.self_reference"},
        )

    assert caught.value.path == "resources.self_reference"
    assert caught.value.stage == "cycle"


@pytest.mark.unit
def test_missing_declared_namespace_fails_with_the_exact_path() -> None:
    graph = _surface_graph()

    with pytest.raises(graph.SurfaceInspectionError) as caught:
        graph.observe_public_surface(_FakeClient(), namespaces={"missing"})

    assert caught.value.path == "missing"
    assert caught.value.stage == "missing_namespace"


@pytest.mark.unit
@pytest.mark.parametrize(
    "namespaces",
    [
        "resources",
        {""},
        {"_private"},
        {"resources..nested"},
        {"resources.nested"},
    ],
)
def test_invalid_namespace_declarations_are_rejected(namespaces: object) -> None:
    graph = _surface_graph()

    with pytest.raises(ValueError, match="namespace"):
        graph.observe_public_surface(_NestedClient(), namespaces=namespaces)


@pytest.mark.unit
@pytest.mark.parametrize("max_depth", [-1, True, 1.5])
def test_invalid_max_depth_is_rejected(max_depth: object) -> None:
    graph = _surface_graph()

    with pytest.raises(ValueError, match="max_depth"):
        graph.observe_public_surface(_FakeClient(), max_depth=max_depth)


@pytest.mark.unit
def test_repeated_resource_object_is_observed_under_each_public_path() -> None:
    graph = _surface_graph()
    shared = _RootResource()

    class AliasedClient:
        def __init__(self) -> None:
            self.left = shared
            self.right = shared

    observations = graph.observe_public_surface(AliasedClient(), namespaces={"left", "right"})

    assert {item.path for item in observations} >= {
        "left.create",
        "left.marker",
        "right.create",
        "right.marker",
    }


@pytest.mark.unit
def test_new_public_child_appears_without_policy_inference() -> None:
    graph = _surface_graph()

    class MutableResource:
        stable = "metadata"

    class MutableClient:
        def __init__(self) -> None:
            self.resources = MutableResource()

    client = MutableClient()
    before = graph.observe_public_surface(client, namespaces={"resources"})

    def future_operation() -> None:
        raise AssertionError("observer must not invoke a newly discovered operation")

    MutableResource.future_operation = future_operation
    after = graph.observe_public_surface(client, namespaces={"resources"})

    assert "resources.future_operation" not in {item.path for item in before}
    assert {item.path: item.return_shape for item in after}[
        "resources.future_operation"
    ] == "callable"


@pytest.mark.unit
def test_public_enumeration_failure_is_sanitized() -> None:
    graph = _surface_graph()

    class BrokenDirectory:
        def __dir__(self) -> list[str]:
            raise RuntimeError("SECRET_DIRECTORY_VALUE")

    with pytest.raises(graph.SurfaceInspectionError) as caught:
        graph.observe_public_surface(BrokenDirectory())

    assert caught.value.path == "<root>"
    assert caught.value.stage == "public_enumeration"
    assert "SECRET_DIRECTORY_VALUE" not in str(caught.value)


@pytest.mark.unit
def test_return_shape_detection_does_not_evaluate_special_attributes() -> None:
    graph = _surface_graph()

    class SpecialAttributeTrap:
        marker = "safe"

        def __getattribute__(self, name: str) -> object:
            if name in {
                "__aiter__",
                "__enter__",
                "__exit__",
                "__aenter__",
                "__aexit__",
            }:
                raise AssertionError("shape detection must remain static")
            return object.__getattribute__(self, name)

    class TrapClient:
        def __init__(self) -> None:
            self.resources = SpecialAttributeTrap()

    observations = graph.observe_public_surface(TrapClient(), namespaces={"resources"})

    assert {item.path: item.return_shape for item in observations}["resources"] == "resource"


@pytest.mark.unit
def test_reobservation_preserves_cached_property_descriptor_category() -> None:
    graph = _surface_graph()
    client = _FakeClient()

    first = graph.observe_public_surface(client, namespaces={"resources"})
    second = graph.observe_public_surface(client, namespaces={"resources"})

    assert second == first
    assert client.namespace_evaluations == 1


@pytest.mark.unit
def test_declared_namespace_must_return_a_resource() -> None:
    graph = _surface_graph()

    with pytest.raises(graph.SurfaceInspectionError) as caught:
        graph.observe_public_surface(_FakeClient(), namespaces={"create"})

    assert caught.value.path == "create"
    assert caught.value.stage == "invalid_namespace_shape"


@pytest.mark.unit
def test_repeated_resource_aliases_evaluate_shared_namespace_once() -> None:
    graph = _surface_graph()

    class SharedResource:
        def __init__(self) -> None:
            self.evaluations = 0

        @property
        def nested(self) -> _RootResource:
            self.evaluations += 1
            return _RootResource()

    shared = SharedResource()

    class AliasedClient:
        def __init__(self) -> None:
            self.left = shared
            self.right = shared

    observations = graph.observe_public_surface(
        AliasedClient(),
        namespaces={"left", "left.nested", "right", "right.nested"},
    )

    assert shared.evaluations == 1
    assert {item.path for item in observations} >= {
        "left.nested.create",
        "right.nested.create",
    }


@pytest.mark.unit
def test_return_shape_vocabulary_is_deterministic() -> None:
    graph = _surface_graph()

    class AwaitableValue:
        def __await__(self) -> object:
            return iter(())

    class AsyncIterableValue:
        def __aiter__(self) -> AsyncIterableValue:
            return self

    class ContextManagerValue:
        def __enter__(self) -> ContextManagerValue:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class AsyncContextManagerValue:
        async def __aenter__(self) -> AsyncContextManagerValue:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class ShapeClient:
        def __init__(self) -> None:
            self.async_context_manager = AsyncContextManagerValue()
            self.async_iterable = AsyncIterableValue()
            self.awaitable = AwaitableValue()
            self.callable = lambda: None
            self.context_manager = ContextManagerValue()
            self.mapping = {"key": "value"}
            self.none = None
            self.resource = _RootResource()
            self.scalar = "value"
            self.sequence = [1, 2]
            self.set = {1, 2}
            self.type = _RootResource

    observations = graph.observe_public_surface(ShapeClient())

    assert {item.path: item.return_shape for item in observations} == {
        "async_context_manager": "async_context_manager",
        "async_iterable": "async_iterable",
        "awaitable": "awaitable",
        "callable": "callable",
        "context_manager": "context_manager",
        "mapping": "mapping",
        "none": "none",
        "resource": "resource",
        "scalar": "scalar",
        "sequence": "sequence",
        "set": "set",
        "type": "type",
    }
