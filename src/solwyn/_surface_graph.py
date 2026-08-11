"""Offline observation of provider client public capability surfaces.

This module is deliberately provider-agnostic and sans-I/O. It records names
and structural types only; it never invokes provider operations or inspects
request/response content.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass

from solwyn._surfaces import (
    SURFACE_RULES,
    AttributeShape,
    SurfaceContext,
    SurfaceKind,
    SurfaceRule,
    SurfaceSource,
    resolve_surface_rule,
)

_MISSING = object()
_TRAVERSABLE_RETURN_SHAPES = frozenset({"resource", "context_manager", "async_context_manager"})


class SurfaceInspectionError(RuntimeError):
    """Raised when a public provider surface cannot be observed safely."""

    def __init__(self, path: str, stage: str, cause_type: str | None = None) -> None:
        self.path = path
        self.stage = stage
        self.cause_type = cause_type
        suffix = f" ({cause_type})" if cause_type is not None else ""
        super().__init__(f"Unable to observe provider surface '{path}' during {stage}{suffix}")


class SurfaceCanaryError(RuntimeError):
    """Raised when a real provider graph disagrees with the reviewed contract."""

    def __init__(
        self,
        *,
        client_family: str,
        installed_version: str,
        path: str,
        stage: str,
        cause_type: str | None = None,
    ) -> None:
        self.client_family = client_family
        self.installed_version = installed_version
        self.path = path
        self.stage = stage
        self.cause_type = cause_type
        suffix = f" ({cause_type})" if cause_type is not None else ""
        super().__init__(
            f"Surface canary failed for {client_family} {installed_version} "
            f"at '{path}' during {stage}{suffix}"
        )


@dataclass(frozen=True, order=True)
class SurfaceObservation:
    """One deterministic structural observation in a provider client graph."""

    path: str
    descriptor_category: str
    return_shape: str


@dataclass(frozen=True, order=True)
class SurfaceCanaryObservation:
    """One real observation joined to its exact contextual contract rule."""

    path: str
    descriptor_category: str
    return_shape: str
    observation_source: str
    rule_id: str


def observe_public_surface(
    root: object,
    *,
    namespaces: Collection[str] = (),
    max_depth: int = 8,
    require_all_namespaces: bool = True,
) -> tuple[SurfaceObservation, ...]:
    """Observe public roots and descend only through explicit namespaces."""

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    if not isinstance(require_all_namespaces, bool):
        raise ValueError("require_all_namespaces must be a boolean")
    declared_namespaces = _validate_namespaces(namespaces)
    observations: list[SurfaceObservation] = []
    observed_paths: set[str] = set()
    namespace_value_cache: dict[tuple[int, str], object] = {}

    def visit(value: object, prefix: str, depth: int, ancestor_ids: frozenset[int]) -> None:
        try:
            public_names = sorted({item for item in dir(value) if not item.startswith("_")})
        except Exception as exc:
            raise SurfaceInspectionError(
                prefix or "<root>", "public_enumeration", type(exc).__name__
            ) from None

        for name in public_names:
            if not name.isidentifier():
                invalid_path = f"{prefix}.{name}" if prefix else name or "<root>"
                raise SurfaceInspectionError(invalid_path, "invalid_public_name")
            path = f"{prefix}.{name}" if prefix else name
            try:
                static_value = _static_attribute(value, name)
            except Exception as exc:
                if _has_static_attribute(type(value), "__getattr__"):
                    if path in declared_namespaces:
                        raise SurfaceInspectionError(path, "dynamic_namespace") from None
                    observations.append(
                        SurfaceObservation(
                            path=path,
                            descriptor_category="dynamic_attribute",
                            return_shape="unevaluated_dynamic",
                        )
                    )
                    observed_paths.add(path)
                    continue
                raise SurfaceInspectionError(
                    path, "static_inspection", type(exc).__name__
                ) from None
            descriptor_category = _descriptor_category(static_value)

            if path in declared_namespaces:
                if depth >= max_depth:
                    raise SurfaceInspectionError(path, "depth_exhaustion")
                cache_key = (id(value), name)
                if cache_key in namespace_value_cache:
                    returned_value = namespace_value_cache[cache_key]
                else:
                    try:
                        returned_value = getattr(value, name)
                    except Exception as exc:
                        raise SurfaceInspectionError(
                            path, "namespace_evaluation", type(exc).__name__
                        ) from None
                    namespace_value_cache[cache_key] = returned_value
                return_shape = _return_shape(returned_value)
                has_enter = _has_static_attribute(returned_value, "__enter__")
                has_exit = _has_static_attribute(returned_value, "__exit__")
                has_async_enter = _has_static_attribute(returned_value, "__aenter__")
                has_async_exit = _has_static_attribute(returned_value, "__aexit__")
                has_sync_lifecycle = has_enter and has_exit
                has_async_lifecycle = has_async_enter and has_async_exit
                has_any_lifecycle_hook = any((has_enter, has_exit, has_async_enter, has_async_exit))
                if (
                    not has_sync_lifecycle
                    and not has_async_lifecycle
                    and (return_shape != "resource" or has_any_lifecycle_hook)
                ):
                    raise SurfaceInspectionError(path, "invalid_namespace_shape", return_shape)
            else:
                returned_value = None
                return_shape = _static_return_shape(static_value, descriptor_category)

            observations.append(
                SurfaceObservation(
                    path=path,
                    descriptor_category=descriptor_category,
                    return_shape=return_shape,
                )
            )
            observed_paths.add(path)
            if path in declared_namespaces:
                returned_id = id(returned_value)
                if returned_id in ancestor_ids:
                    raise SurfaceInspectionError(path, "cycle")
                visit(returned_value, path, depth + 1, ancestor_ids | {returned_id})

    visit(root, "", 0, frozenset({id(root)}))
    if require_all_namespaces:
        missing_namespaces = sorted(declared_namespaces - observed_paths)
        if missing_namespaces:
            raise SurfaceInspectionError(missing_namespaces[0], "missing_namespace")
    return tuple(sorted(observations))


def audit_public_surface(
    root: object,
    *,
    context: SurfaceContext,
    client_family: str,
    installed_version: str,
    service_model_operations: Collection[str] = (),
    rules: Iterable[SurfaceRule] | None = None,
    max_depth: int = 8,
) -> tuple[SurfaceCanaryObservation, ...]:
    """Join one real pre-call graph to the contextual contract or fail closed."""

    if not client_family or not installed_version:
        raise ValueError("client_family and installed_version must not be empty")
    selected_rules = tuple(SURFACE_RULES if rules is None else rules)
    resolution_rules = None if rules is None else selected_rules

    def resolve(path: str) -> SurfaceRule | None:
        try:
            return resolve_surface_rule(
                context=context,
                path=path,
                source=SurfaceSource.RAW,
                rules=resolution_rules,
            )
        except RuntimeError:
            raise SurfaceCanaryError(
                client_family=client_family,
                installed_version=installed_version,
                path=path,
                stage="rule_resolution",
                cause_type="RuntimeError",
            ) from None

    namespace_paths = _applicable_namespace_paths(selected_rules, resolve)
    try:
        observed = observe_public_surface(
            root,
            namespaces=namespace_paths,
            max_depth=max_depth,
            require_all_namespaces=False,
        )
    except SurfaceInspectionError as exc:
        raise SurfaceCanaryError(
            client_family=client_family,
            installed_version=installed_version,
            path=exc.path,
            stage=exc.stage,
            cause_type=exc.cause_type,
        ) from None

    joined: dict[str, tuple[SurfaceObservation, str]] = {
        item.path: (item, "public_attribute") for item in observed
    }
    for path in sorted(set(service_model_operations)):
        _validate_public_path(path)
        joined.setdefault(
            path,
            (
                SurfaceObservation(
                    path=path,
                    descriptor_category="service_model_operation",
                    return_shape="service_model_only",
                ),
                "service_model_operation",
            ),
        )

    rows: list[SurfaceCanaryObservation] = []
    for path, (observation, observation_source) in sorted(joined.items()):
        rule = resolve(path)
        if rule is None:
            raise SurfaceCanaryError(
                client_family=client_family,
                installed_version=installed_version,
                path=path,
                stage="unknown_classification",
            )
        shape = AttributeShape(
            descriptor_category=observation.descriptor_category,
            return_shape=observation.return_shape,
        )
        if not rule.accepts_shape(shape):
            raise SurfaceCanaryError(
                client_family=client_family,
                installed_version=installed_version,
                path=path,
                stage="shape_drift",
            )
        rows.append(
            SurfaceCanaryObservation(
                path=path,
                descriptor_category=observation.descriptor_category,
                return_shape=observation.return_shape,
                observation_source=observation_source,
                rule_id=rule.rule_id,
            )
        )
    return tuple(rows)


def _applicable_namespace_paths(
    rules: Collection[SurfaceRule],
    resolve: Callable[[str], SurfaceRule | None],
) -> frozenset[str]:
    applicable: dict[str, SurfaceRule] = {}
    for path in sorted({rule.surface for rule in rules}):
        resolved = resolve(path)
        if resolved is not None:
            applicable[path] = resolved

    parent_paths: set[str] = set()
    for path in applicable:
        parts = path.split(".")
        parent_paths.update(".".join(parts[:length]) for length in range(1, len(parts)))
    return frozenset(
        path
        for path, rule in applicable.items()
        if (rule.kind is SurfaceKind.NAMESPACE or path in parent_paths)
        and any(shape.return_shape in _TRAVERSABLE_RETURN_SHAPES for shape in rule.expected_shapes)
    )


def _validate_public_path(path: str) -> None:
    parts = path.split(".")
    if not path or any(
        not part or not part.isidentifier() or part.startswith("_") for part in parts
    ):
        raise ValueError(f"invalid public surface path: {path!r}")


def _validate_namespaces(namespaces: Collection[str]) -> frozenset[str]:
    if isinstance(namespaces, (str, bytes)):
        raise ValueError("namespaces must be a collection of dotted namespace paths")
    try:
        candidates = tuple(namespaces)
    except TypeError:
        raise ValueError("namespaces must be a collection of dotted namespace paths") from None

    for path in candidates:
        if not isinstance(path, str):
            raise ValueError("each namespace must be a dotted public attribute path")
        parts = path.split(".")
        if any(not part or not part.isidentifier() or part.startswith("_") for part in parts):
            raise ValueError(f"invalid public namespace path: {path!r}")

    declared_namespaces = frozenset(candidates)
    for path in sorted(declared_namespaces):
        parts = path.split(".")
        for length in range(1, len(parts)):
            parent = ".".join(parts[:length])
            if parent not in declared_namespaces:
                raise ValueError(
                    f"namespace {path!r} requires declared parent namespace {parent!r}"
                )
    return declared_namespaces


def _descriptor_category(value: object) -> str:
    if isinstance(value, functools.cached_property):
        return "cached_property"
    if isinstance(value, property):
        return "property"
    if isinstance(value, staticmethod):
        return "staticmethod"
    if isinstance(value, classmethod):
        return "classmethod"
    if inspect.isfunction(value):
        return "function"
    if inspect.isdatadescriptor(value):
        return "data_descriptor"
    if inspect.ismethoddescriptor(value):
        return "method_descriptor"
    if inspect.isroutine(value) or callable(value):
        return "callable"
    if _has_static_attribute(type(value), "__get__"):
        return "non_data_descriptor"
    return "attribute"


def _static_attribute(value: object, name: str) -> object:
    static_value = inspect.getattr_static(value, name)
    class_value = inspect.getattr_static(type(value), name, _MISSING)
    if isinstance(class_value, functools.cached_property):
        return class_value
    return static_value


def _static_return_shape(value: object, descriptor_category: str) -> str:
    if descriptor_category == "classmethod":
        return "callable"
    if descriptor_category in {
        "cached_property",
        "property",
        "data_descriptor",
        "non_data_descriptor",
    }:
        return "unevaluated_descriptor"
    return _return_shape(value)


def _return_shape(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, (str, bytes, bool, int, float, complex)):
        return "scalar"
    if isinstance(value, type):
        return "type"
    if callable(value):
        return "callable"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, Set):
        return "set"
    if isinstance(value, Sequence):
        return "sequence"
    if inspect.isawaitable(value):
        return "awaitable"
    if _has_static_attribute(value, "__aiter__"):
        return "async_iterable"
    if _has_static_attribute(value, "__enter__") and _has_static_attribute(value, "__exit__"):
        return "context_manager"
    if _has_static_attribute(value, "__aenter__") and _has_static_attribute(value, "__aexit__"):
        return "async_context_manager"
    return "resource"


def _has_static_attribute(value: object, name: str) -> bool:
    return inspect.getattr_static(value, name, _MISSING) is not _MISSING
