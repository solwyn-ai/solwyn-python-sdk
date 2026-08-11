"""Local, deterministic coverage manifests for attached Solwyn clients.

This module observes structural provider-client metadata only. It performs no
network I/O, never invokes provider operations, and never reads request,
prompt, response, or credential content.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from solwyn._base import _client_shape, _SolwynBase
from solwyn._surface_graph import (
    SurfaceInspectionError,
    SurfaceObservation,
    _applicable_namespace_paths,
    observe_public_surface,
)
from solwyn._surfaces import (
    SURFACE_RULES,
    AttributeShape,
    SurfaceCondition,
    SurfaceContext,
    SurfaceKind,
    SurfaceRule,
    SurfaceSource,
    resolve_surface_rule,
)
from solwyn.exceptions import CoverageMismatchError

_GUARDABLE_RETURN_SHAPES = frozenset({"resource", "context_manager", "async_context_manager"})
_CHAT_SURFACES = frozenset(
    {
        "chat.completions.create",
        "messages.create",
        "models.generate_content",
        "models.generate_content_stream",
        "converse",
        "converse_stream",
        "generate_content",
    }
)
_AUDIT_CATEGORIES = (
    "guarded_namespaces",
    "tracked",
    "untracked",
    "unknown",
    "scoped_escapes",
    "blocked",
    "unsupported",
    "conditional",
    "safe",
)
_FIRST_CAP_RE = re.compile(r"(.)([A-Z][a-z]+)")
_ALL_CAP_RE = re.compile(r"([a-z0-9])([A-Z])")
_Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class CoverageRuntime(BaseModel):
    """One configured provider runtime in failover order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(...)
    dialect: str = Field(...)
    client_shape: str = Field(...)
    model: str = Field(...)


class _CoverageRow(BaseModel):
    """The complete audit identity and effective action for one capability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(...)
    surface: str = Field(...)
    token: str = Field(...)
    kind: str = Field(...)
    policy_action: str = Field(...)
    dispatch_action: str = Field(...)
    usage_basis: str | None = Field(...)
    source: str = Field(...)
    capability_scope: str | None = Field(...)
    condition: str | None = Field(...)
    reason: str | None = Field(...)
    expected_descriptor_category: str | None = Field(...)
    observed_descriptor_category: str = Field(...)
    expected_return_shape: str | None = Field(...)
    observed_return_shape: str = Field(...)


class CoverageEntry(_CoverageRow):
    """One effective manifest entry."""


class CoverageAuditEntry(_CoverageRow):
    """One literal row in a bidirectional coverage expectation."""


class CoverageExpectation(BaseModel):
    """Literal, exhaustive audit pin grouped by review category.

    Every category is required so an omitted category cannot silently weaken a
    pin. Entries are written independently of the report under test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    guarded_namespaces: tuple[CoverageAuditEntry, ...] = Field(...)
    tracked: tuple[CoverageAuditEntry, ...] = Field(...)
    untracked: tuple[CoverageAuditEntry, ...] = Field(...)
    unknown: tuple[CoverageAuditEntry, ...] = Field(...)
    scoped_escapes: tuple[CoverageAuditEntry, ...] = Field(...)
    blocked: tuple[CoverageAuditEntry, ...] = Field(...)
    unsupported: tuple[CoverageAuditEntry, ...] = Field(...)
    conditional: tuple[CoverageAuditEntry, ...] = Field(...)
    safe: tuple[CoverageAuditEntry, ...] = Field(...)


class CoverageFingerprint(BaseModel):
    """Compact literal pin over every audit field, grouped for useful drift."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    guarded_namespaces: _Digest = Field(...)
    tracked: _Digest = Field(...)
    untracked: _Digest = Field(...)
    unknown: _Digest = Field(...)
    scoped_escapes: _Digest = Field(...)
    blocked: _Digest = Field(...)
    unsupported: _Digest = Field(...)
    conditional: _Digest = Field(...)
    safe: _Digest = Field(...)


class CoverageReport(BaseModel):
    """Frozen local audit of one attached Solwyn client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(...)
    dialect: str = Field(...)
    client_shape: str = Field(...)
    posture: Literal["warn", "raise", "allow"] = Field(...)
    provider_chain: tuple[CoverageRuntime, ...] = Field(...)
    acknowledgments: tuple[str, ...] = Field(...)
    entries: tuple[CoverageEntry, ...] = Field(...)

    def fingerprint(self) -> CoverageFingerprint:
        """Return a compact pin suitable for review and later literal use."""

        return _fingerprint_from_expectation(_expectation_from_entries(self.entries))

    def expect(self, expected: CoverageExpectation | CoverageFingerprint) -> None:
        """Require an independently authored audit pin in both directions."""

        actual = _expectation_from_entries(self.entries)
        if isinstance(expected, CoverageExpectation):
            differences = _expectation_differences(expected, actual)
        elif isinstance(expected, CoverageFingerprint):
            differences = _fingerprint_differences(
                expected,
                _fingerprint_from_expectation(actual),
            )
        else:
            raise TypeError("expected must be a literal CoverageExpectation or CoverageFingerprint")
        if differences:
            raise CoverageMismatchError(differences=differences)


def coverage(client: object) -> CoverageReport:
    """Return the effective guarded capability graph for a Solwyn client."""

    if not isinstance(client, _SolwynBase):
        raise TypeError("coverage() requires a Solwyn or AsyncSolwyn client")

    context = client._surface_context
    raw_client = client._runtimes[0].sdk_client

    def resolve_raw(path: str) -> SurfaceRule | None:
        return resolve_surface_rule(
            context=context,
            path=path,
            source=SurfaceSource.RAW,
        )

    namespace_paths = _applicable_namespace_paths(SURFACE_RULES, resolve_raw)
    observed = list(
        observe_public_surface(
            raw_client,
            namespaces=namespace_paths,
            require_all_namespaces=False,
        )
    )
    observed.extend(_reachable_service_model_observations(raw_client, context, observed))

    entries: dict[str, CoverageEntry] = {}
    for observation in sorted(observed):
        raw_rule = resolve_raw(observation.path)
        wrapper_rule = _resolve_rule(
            context=context,
            path=observation.path,
            source=SurfaceSource.WRAPPER,
        )
        if wrapper_rule is not None and (
            wrapper_rule is not raw_rule
            or observation.descriptor_category == "service_model_operation"
        ):
            entry = _entry_for_rule(wrapper_rule, client=client)
        else:
            entry = _entry_for_observation(
                observation,
                rule=raw_rule,
                client=client,
            )
        entries[entry.rule_id] = entry

    for path in sorted({rule.surface for rule in SURFACE_RULES}):
        rule = _resolve_rule(
            context=context,
            path=path,
            source=SurfaceSource.WRAPPER,
        )
        if rule is not None and rule.rule_id not in entries:
            entries[rule.rule_id] = _entry_for_rule(rule, client=client)

    for candidate in SURFACE_RULES:
        if candidate.condition is None:
            continue
        rule = _resolve_rule(
            context=context,
            path=candidate.surface,
            source=candidate.source,
            condition=candidate.condition,
        )
        if rule is candidate and rule.rule_id not in entries:
            entries[rule.rule_id] = _entry_for_rule(rule, client=client)

    provider_chain = tuple(
        CoverageRuntime(
            provider=runtime.adapter.name,
            dialect=runtime.adapter.dialect,
            client_shape=(
                context.client_shape
                if index == 0
                else _client_shape(runtime.sdk_client, runtime.adapter.dialect)
            ),
            model=runtime.entry.model,
        )
        for index, runtime in enumerate(client._runtimes)
    )
    return CoverageReport(
        provider=context.provider,
        dialect=context.dialect,
        client_shape=context.client_shape,
        posture=client._config.on_unmetered,
        provider_chain=provider_chain,
        acknowledgments=tuple(sorted(client._config.acknowledge_untracked)),
        entries=tuple(sorted(entries.values(), key=lambda item: (item.surface, item.rule_id))),
    )


def _resolve_rule(
    *,
    context: SurfaceContext,
    path: str,
    source: SurfaceSource,
    condition: SurfaceCondition | None = None,
) -> SurfaceRule | None:
    return resolve_surface_rule(
        context=context,
        path=path,
        source=source,
        condition=condition,
    )


def _entry_for_observation(
    observation: SurfaceObservation,
    *,
    rule: SurfaceRule | None,
    client: _SolwynBase,
) -> CoverageEntry:
    observed_shape = AttributeShape(
        observation.descriptor_category,
        observation.return_shape,
    )
    expected_shape = _preferred_expected_shape(rule, observed_shape)
    if rule is not None and rule.accepts_shape(observed_shape):
        return _coverage_entry(
            rule=rule,
            client=client,
            expected_shape=expected_shape,
            observed_shape=observed_shape,
        )

    return _unknown_entry(
        surface=observation.path,
        client=client,
        expected_shape=expected_shape,
        observed_shape=observed_shape,
        capability_scope=(
            rule.capability_scope.value
            if rule is not None and rule.capability_scope is not None
            else None
        ),
    )


def _entry_for_rule(rule: SurfaceRule, *, client: _SolwynBase) -> CoverageEntry:
    shape = sorted(rule.expected_shapes)[0]
    return _coverage_entry(
        rule=rule,
        client=client,
        expected_shape=shape,
        observed_shape=shape,
    )


def _coverage_entry(
    *,
    rule: SurfaceRule,
    client: _SolwynBase,
    expected_shape: AttributeShape | None,
    observed_shape: AttributeShape,
) -> CoverageEntry:
    policy_action, dispatch_action = _effective_actions(
        kind=rule.kind,
        token=rule.token,
        surface=rule.surface,
        return_shape=observed_shape.return_shape,
        client=client,
    )
    return CoverageEntry(
        rule_id=rule.rule_id,
        surface=rule.surface,
        token=rule.token,
        kind=rule.kind.value,
        policy_action=policy_action,
        dispatch_action=dispatch_action,
        usage_basis=_effective_usage_basis(rule, client),
        source=rule.source.value,
        capability_scope=(
            rule.capability_scope.value if rule.capability_scope is not None else None
        ),
        condition=rule.condition.value if rule.condition is not None else None,
        reason=rule.reason,
        expected_descriptor_category=(
            expected_shape.descriptor_category if expected_shape is not None else None
        ),
        observed_descriptor_category=observed_shape.descriptor_category,
        expected_return_shape=(expected_shape.return_shape if expected_shape is not None else None),
        observed_return_shape=observed_shape.return_shape,
    )


def _unknown_entry(
    *,
    surface: str,
    client: _SolwynBase,
    expected_shape: AttributeShape | None,
    observed_shape: AttributeShape,
    capability_scope: str | None,
) -> CoverageEntry:
    context = client._surface_context
    policy_action, dispatch_action = _effective_actions(
        kind=SurfaceKind.UNKNOWN,
        token=surface,
        surface=surface,
        return_shape=observed_shape.return_shape,
        client=client,
    )
    return CoverageEntry(
        rule_id=(f"unknown:{context.client_shape}:{context.mode}:{context.provider}:{surface}"),
        surface=surface,
        token=surface,
        kind=SurfaceKind.UNKNOWN.value,
        policy_action=policy_action,
        dispatch_action=dispatch_action,
        usage_basis=None,
        source=SurfaceSource.RAW.value,
        capability_scope=capability_scope,
        condition=None,
        reason=None,
        expected_descriptor_category=(
            expected_shape.descriptor_category if expected_shape is not None else None
        ),
        observed_descriptor_category=observed_shape.descriptor_category,
        expected_return_shape=(expected_shape.return_shape if expected_shape is not None else None),
        observed_return_shape=observed_shape.return_shape,
    )


def _effective_actions(
    *,
    kind: SurfaceKind,
    token: str,
    surface: str,
    return_shape: str,
    client: _SolwynBase,
) -> tuple[str, str]:
    if kind is SurfaceKind.METERED:
        return "track", "intercept"
    if kind is SurfaceKind.NAMESPACE:
        return "pass", "guard"
    if kind in {SurfaceKind.METADATA, SurfaceKind.INFRASTRUCTURE}:
        return "pass", "return"
    if kind is SurfaceKind.BLOCKED:
        return "block", "refuse"
    if kind is SurfaceKind.UNSUPPORTED:
        return "unsupported", "refuse"

    acknowledgments = client._config.acknowledge_untracked
    has_acknowledged_descendant = any(
        acknowledged.startswith(f"{surface}.") for acknowledged in acknowledgments
    )
    if token in acknowledgments or has_acknowledged_descendant:
        policy_action = "acknowledged"
    else:
        policy_action = client._config.on_unmetered
    if policy_action == "raise":
        return policy_action, "refuse"
    if return_shape in _GUARDABLE_RETURN_SHAPES:
        return policy_action, "guard"
    return policy_action, "return"


def _preferred_expected_shape(
    rule: SurfaceRule | None,
    observed: AttributeShape,
) -> AttributeShape | None:
    if rule is None:
        return None
    if observed in rule.expected_shapes:
        return observed
    descriptor_matches = sorted(
        shape
        for shape in rule.expected_shapes
        if shape.descriptor_category == observed.descriptor_category
    )
    if descriptor_matches:
        return descriptor_matches[0]
    return sorted(rule.expected_shapes)[0]


def _effective_usage_basis(rule: SurfaceRule, client: _SolwynBase) -> str | None:
    if rule.usage_basis is None:
        return None
    if rule.surface not in _CHAT_SURFACES:
        return rule.usage_basis.value

    bases: list[str] = []
    for runtime in client._runtimes:
        context = SurfaceContext(
            provider=runtime.adapter.name,
            dialect=runtime.adapter.dialect,
            client_shape=_client_shape(runtime.sdk_client, runtime.adapter.dialect),
            mode=client._surface_context.mode,
        )
        reachable = _resolve_rule(
            context=context,
            path="chat.completions.create",
            source=SurfaceSource.WRAPPER,
        )
        if reachable is not None and reachable.kind is SurfaceKind.METERED:
            if reachable.usage_basis is None:
                raise RuntimeError(f"metered rule has no usage basis: {reachable.rule_id}")
            bases.append(reachable.usage_basis.value)
    if not bases:
        raise RuntimeError(f"metered surface has no reachable runtime: {rule.rule_id}")
    if "provider_or_estimate" in bases:
        return "provider_or_estimate"
    if len(set(bases)) == 1:
        return bases[0]
    order = {
        "provider": 0,
        "provider_and_request": 1,
        "request_derived": 2,
        "provider_or_estimate": 3,
    }
    return max(bases, key=order.__getitem__)


def _reachable_service_model_observations(
    raw_client: object,
    context: SurfaceContext,
    observed: Collection[SurfaceObservation],
) -> tuple[SurfaceObservation, ...]:
    if context.dialect != "bedrock":
        return ()
    observed_paths = {item.path for item in observed}
    rows: list[SurfaceObservation] = []
    for path in _bedrock_service_model_operations(raw_client):
        if path in observed_paths:
            continue
        raw_rule = _resolve_rule(context=context, path=path, source=SurfaceSource.RAW)
        wrapper_rule = _resolve_rule(context=context, path=path, source=SurfaceSource.WRAPPER)
        if raw_rule is None and wrapper_rule is None:
            continue
        rows.append(
            SurfaceObservation(
                path=path,
                descriptor_category="service_model_operation",
                return_shape="service_model_only",
            )
        )
    return tuple(rows)


def _bedrock_service_model_operations(client: object) -> tuple[str, ...]:
    meta = getattr(client, "meta", None)
    service_model = getattr(meta, "service_model", None)
    operation_names = getattr(service_model, "operation_names", ())
    if not operation_names:
        return ()
    if not isinstance(operation_names, Collection) or isinstance(operation_names, (str, bytes)):
        raise SurfaceInspectionError("<root>", "service_model_operations")
    normalized: set[str] = set()
    for name in operation_names:
        if not isinstance(name, str) or not name:
            raise SurfaceInspectionError("<root>", "service_model_operations")
        first_pass = _FIRST_CAP_RE.sub(r"\1_\2", name)
        normalized.add(_ALL_CAP_RE.sub(r"\1_\2", first_pass).lower())
    return tuple(sorted(normalized))


def _expectation_from_entries(entries: tuple[CoverageEntry, ...]) -> CoverageExpectation:
    grouped: dict[str, list[CoverageAuditEntry]] = {category: [] for category in _AUDIT_CATEGORIES}
    for entry in entries:
        grouped[_audit_category(entry)].append(
            CoverageAuditEntry.model_validate(entry.model_dump())
        )
    return CoverageExpectation(
        **{
            category: tuple(
                sorted(grouped[category], key=lambda item: (item.surface, item.rule_id))
            )
            for category in _AUDIT_CATEGORIES
        }
    )


def _audit_category(entry: CoverageEntry) -> str:
    if entry.condition is not None:
        return "conditional"
    if entry.kind == SurfaceKind.NAMESPACE.value:
        return "guarded_namespaces"
    if entry.kind == SurfaceKind.METERED.value:
        return "tracked"
    if entry.kind == SurfaceKind.UNKNOWN.value:
        return "unknown"
    if entry.kind == SurfaceKind.BLOCKED.value:
        return "blocked"
    if entry.kind == SurfaceKind.UNSUPPORTED.value:
        return "unsupported"
    if entry.kind in {SurfaceKind.METADATA.value, SurfaceKind.INFRASTRUCTURE.value}:
        return "safe"
    if entry.kind == SurfaceKind.UNMETERED_SPEND.value:
        if entry.capability_scope not in {None, "operation"}:
            return "scoped_escapes"
        return "untracked"
    raise RuntimeError(f"unsupported coverage kind: {entry.kind}")


def _expectation_differences(
    expected: CoverageExpectation,
    actual: CoverageExpectation,
) -> tuple[str, ...]:
    differences: list[str] = []
    for category in _AUDIT_CATEGORIES:
        expected_rows = {row.rule_id: row for row in getattr(expected, category)}
        actual_rows = {row.rule_id: row for row in getattr(actual, category)}
        for rule_id in sorted(expected_rows.keys() - actual_rows.keys()):
            differences.append(f"{category}: removed {rule_id}")
        for rule_id in sorted(actual_rows.keys() - expected_rows.keys()):
            differences.append(f"{category}: added {rule_id}")
        for rule_id in sorted(expected_rows.keys() & actual_rows.keys()):
            expected_data = expected_rows[rule_id].model_dump()
            actual_data = actual_rows[rule_id].model_dump()
            changed = sorted(
                field for field in expected_data if expected_data[field] != actual_data[field]
            )
            if changed:
                differences.append(f"{category}: changed {rule_id} ({', '.join(changed)})")
    return tuple(differences)


def _fingerprint_from_expectation(expected: CoverageExpectation) -> CoverageFingerprint:
    values: dict[str, str] = {}
    for category in _AUDIT_CATEGORIES:
        rows = [row.model_dump() for row in getattr(expected, category)]
        payload = json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        values[category] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return CoverageFingerprint(**values)


def _fingerprint_differences(
    expected: CoverageFingerprint,
    actual: CoverageFingerprint,
) -> tuple[str, ...]:
    return tuple(
        f"{category}: fingerprint changed"
        for category in _AUDIT_CATEGORIES
        if getattr(expected, category) != getattr(actual, category)
    )


__all__ = [
    "CoverageAuditEntry",
    "CoverageEntry",
    "CoverageExpectation",
    "CoverageFingerprint",
    "CoverageReport",
    "CoverageRuntime",
    "coverage",
]
