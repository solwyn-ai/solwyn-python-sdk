"""Offline observation of provider client public capability surfaces.

This module is deliberately provider-agnostic and sans-I/O. It records names
and structural types only; it never invokes provider operations or inspects
request/response content.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Collection, Mapping, Sequence, Set
from dataclasses import dataclass

_MISSING = object()


class SurfaceInspectionError(RuntimeError):
    """Raised when a public provider surface cannot be observed safely."""

    def __init__(self, path: str, stage: str, cause_type: str | None = None) -> None:
        self.path = path
        self.stage = stage
        self.cause_type = cause_type
        suffix = f" ({cause_type})" if cause_type is not None else ""
        super().__init__(f"Unable to observe provider surface '{path}' during {stage}{suffix}")


@dataclass(frozen=True, order=True)
class SurfaceObservation:
    """One deterministic structural observation in a provider client graph."""

    path: str
    descriptor_category: str
    return_shape: str


def observe_public_surface(
    root: object,
    *,
    namespaces: Collection[str] = (),
    max_depth: int = 8,
) -> tuple[SurfaceObservation, ...]:
    """Observe public roots and descend only through explicit namespaces."""

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
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
            path = f"{prefix}.{name}" if prefix else name
            try:
                static_value = _static_attribute(value, name)
            except Exception as exc:
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
                if return_shape != "resource":
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
    missing_namespaces = sorted(declared_namespaces - observed_paths)
    if missing_namespaces:
        raise SurfaceInspectionError(missing_namespaces[0], "missing_namespace")
    return tuple(sorted(observations))


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
    if inspect.ismethoddescriptor(value):
        return "method_descriptor"
    if inspect.isdatadescriptor(value):
        return "data_descriptor"
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
