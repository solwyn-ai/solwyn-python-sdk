"""Tests for the contextual provider capability contract."""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from solwyn._surfaces import (
    SURFACE_RULES,
    AttributeShape,
    CapabilityScope,
    SurfaceCondition,
    SurfaceContext,
    SurfaceKind,
    SurfaceRule,
    SurfaceSelector,
    SurfaceSource,
    UsageBasis,
    resolve_surface_rule,
    surface_contract_data,
)

ROOT = Path(__file__).parents[2]
CONTRACT_PATH = ROOT / "build" / "surface_contract" / "surface-classification.json"
COMMITTED_CONTRACT_PATH = ROOT / "docs" / "contracts" / "surface-classification.json"


OPENAI_SYNC = SurfaceContext(
    provider="openai",
    dialect="openai",
    client_shape="openai_sdk",
    mode="sync",
)
OPENAI_ASYNC = SurfaceContext(
    provider="openai",
    dialect="openai",
    client_shape="openai_sdk",
    mode="async",
)
GROQ_SYNC = SurfaceContext(
    provider="groq",
    dialect="openai",
    client_shape="openai_sdk",
    mode="sync",
)


def _export_module() -> ModuleType:
    path = ROOT / "scripts" / "export_surface_contract.py"
    spec = importlib.util.spec_from_file_location("export_surface_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load surface contract exporter at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rule(
    rule_id: str,
    *,
    provider: str | None = "openai",
    condition: SurfaceCondition | None = None,
) -> SurfaceRule:
    return SurfaceRule(
        rule_id=rule_id,
        surface="responses.create",
        selectors=(
            SurfaceSelector(
                provider=provider,
                dialect="openai",
                client_shape="openai_sdk",
                mode="sync",
            ),
        ),
        kind=SurfaceKind.UNMETERED_SPEND,
        source=SurfaceSource.RAW,
        acknowledgment_token="responses.create",
        capability_scope=CapabilityScope.OPERATION,
        condition=condition,
        expected_shapes=(AttributeShape(descriptor_category="function", return_shape="callable"),),
    )


@pytest.mark.unit
def test_resolution_prefers_the_most_specific_exact_context() -> None:
    generic = _rule("surface.responses-create.generic", provider=None)
    native = _rule("surface.responses-create.native")

    resolved = resolve_surface_rule(
        context=OPENAI_SYNC,
        path="responses.create",
        source=SurfaceSource.RAW,
        rules=(generic, native),
    )

    assert resolved is native


@pytest.mark.unit
def test_resolution_rejects_an_ambiguous_context() -> None:
    first = _rule("surface.responses-create.first")
    second = _rule("surface.responses-create.second")

    with pytest.raises(RuntimeError, match="ambiguous surface rules"):
        resolve_surface_rule(
            context=OPENAI_SYNC,
            path="responses.create",
            source=SurfaceSource.RAW,
            rules=(first, second),
        )


@pytest.mark.unit
def test_conditional_rule_overrides_the_ordinary_tts_rule() -> None:
    ordinary = SurfaceRule(
        rule_id="surface.audio-speech-create.metered",
        surface="audio.speech.create",
        selectors=(SurfaceSelector.from_context(OPENAI_SYNC),),
        kind=SurfaceKind.METERED,
        source=SurfaceSource.BOTH,
        usage_basis=UsageBasis.REQUEST_DERIVED,
        expected_shapes=(AttributeShape(descriptor_category="function", return_shape="callable"),),
    )
    conditional = SurfaceRule(
        rule_id="surface.audio-speech-create.gpt-4o-mini-tts",
        surface="audio.speech.create",
        selectors=(SurfaceSelector.from_context(OPENAI_SYNC),),
        kind=SurfaceKind.UNMETERED_SPEND,
        source=SurfaceSource.SYNTHETIC_POLICY,
        condition=SurfaceCondition.OPENAI_UNTRACKED_TTS_MODEL,
        acknowledgment_token="audio.speech.create:gpt-4o-mini-tts",
        capability_scope=CapabilityScope.OPERATION,
        expected_shapes=(
            AttributeShape(
                descriptor_category="synthetic_policy",
                return_shape="conditional",
            ),
        ),
    )

    assert (
        resolve_surface_rule(
            context=OPENAI_SYNC,
            path="audio.speech.create",
            source=SurfaceSource.BOTH,
            rules=(ordinary, conditional),
        )
        is ordinary
    )
    assert (
        resolve_surface_rule(
            context=OPENAI_SYNC,
            path="audio.speech.create",
            source=SurfaceSource.SYNTHETIC_POLICY,
            condition=SurfaceCondition.OPENAI_UNTRACKED_TTS_MODEL,
            rules=(ordinary, conditional),
        )
        is conditional
    )


@pytest.mark.unit
def test_rule_invariants_reject_unsafe_or_ambiguous_contract_rows() -> None:
    with pytest.raises(RuntimeError, match="selector provider must not be empty"):
        SurfaceSelector(provider="")

    with pytest.raises(RuntimeError, match="namespace.*acknowledgment"):
        SurfaceRule(
            rule_id="surface.responses.namespace",
            surface="responses",
            selectors=(SurfaceSelector.from_context(OPENAI_SYNC),),
            kind=SurfaceKind.NAMESPACE,
            source=SurfaceSource.RAW,
            acknowledgment_token="responses",
            expected_shapes=(
                AttributeShape(
                    descriptor_category="cached_property",
                    return_shape="resource",
                ),
            ),
        )

    with pytest.raises(RuntimeError, match="metered.*usage basis"):
        SurfaceRule(
            rule_id="surface.chat-completions-create.metered",
            surface="chat.completions.create",
            selectors=(SurfaceSelector.from_context(OPENAI_SYNC),),
            kind=SurfaceKind.METERED,
            source=SurfaceSource.BOTH,
            expected_shapes=(
                AttributeShape(descriptor_category="function", return_shape="callable"),
            ),
        )

    with pytest.raises(RuntimeError, match="raw callable.*infrastructure"):
        SurfaceRule(
            rule_id="surface.post.infrastructure",
            surface="post",
            selectors=(SurfaceSelector.from_context(OPENAI_SYNC),),
            kind=SurfaceKind.INFRASTRUCTURE,
            source=SurfaceSource.RAW,
            expected_shapes=(
                AttributeShape(descriptor_category="function", return_shape="callable"),
            ),
        )

    with pytest.raises(RuntimeError, match="acknowledgment token must not be empty"):
        SurfaceRule(
            rule_id="surface.responses-create.empty-token",
            surface="responses.create",
            selectors=(SurfaceSelector.from_context(OPENAI_SYNC),),
            kind=SurfaceKind.UNMETERED_SPEND,
            source=SurfaceSource.RAW,
            acknowledgment_token="",
            capability_scope=CapabilityScope.OPERATION,
            expected_shapes=(
                AttributeShape(descriptor_category="function", return_shape="callable"),
            ),
        )

    with pytest.raises(RuntimeError, match="capability scope"):
        SurfaceRule(
            rule_id="surface.responses.scoped-namespace",
            surface="responses",
            selectors=(SurfaceSelector.from_context(OPENAI_SYNC),),
            kind=SurfaceKind.NAMESPACE,
            source=SurfaceSource.RAW,
            capability_scope=CapabilityScope.RESOURCE,
            expected_shapes=(
                AttributeShape(descriptor_category="cached_property", return_shape="resource"),
            ),
        )


@pytest.mark.unit
def test_generated_report_contract_check_covers_rules_and_shapes() -> None:
    exporter = _export_module()
    report = {
        "provider": "openai",
        "client_shape": "openai_sdk",
        "mode": "sync",
        "observations": [
            {
                "path": "responses.create",
                "descriptor_category": "function",
                "return_shape": "callable",
            }
        ],
    }

    assert exporter.compare_report_contract(report, label="report.json") == ()

    report["observations"][0]["path"] = "not.reviewed"
    assert (
        "no reviewed raw rule" in exporter.compare_report_contract(report, label="report.json")[0]
    )

    report["observations"][0] = {
        "path": "responses.create",
        "descriptor_category": "property",
        "return_shape": "scalar",
    }
    assert (
        "rejects observed shape" in exporter.compare_report_contract(report, label="report.json")[0]
    )


@pytest.mark.unit
def test_generated_report_directory_check_fails_closed_when_empty(tmp_path: Path) -> None:
    exporter = _export_module()

    assert exporter.compare_report_directory(tmp_path) == (
        f"no provider surface reports found in {tmp_path}",
    )


@pytest.mark.unit
def test_provider_matrix_checks_each_generated_report_against_the_contract() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow_data = yaml.safe_load(workflow)
    inventory_steps = workflow_data["jobs"]["provider-surface-inventory"]["steps"]
    verification_steps = [
        step
        for step in inventory_steps
        if step.get("name") == "Verify generated surfaces are classified"
    ]

    assert len(verification_steps) == 1
    assert verification_steps[0]["if"] == "always()"
    run_command = verification_steps[0]["run"]
    command_tokens = shlex.split(run_command)
    assert command_tokens == [
        "python",
        "scripts/export_surface_contract.py",
        "--check",
        "--output",
        "build/surface_contract/surface-classification.json",
        "--reports-dir",
        "build/provider_surface_inventory",
    ]

    upload_steps = [
        step for step in inventory_steps if step.get("name") == "Upload provider surface inventory"
    ]
    assert len(upload_steps) == 1
    assert set(upload_steps[0]["with"]["path"].splitlines()) == {
        "build/provider_surface_inventory",
        "build/surface_contract/surface-classification.json",
    }


@pytest.mark.unit
def test_native_openai_compatible_and_native_together_video_rules_differ() -> None:
    contexts = {
        "native": OPENAI_SYNC,
        "compatible": SurfaceContext(
            provider="openai_compatible",
            dialect="openai",
            client_shape="openai_sdk",
            mode="sync",
        ),
        "together": SurfaceContext(
            provider="together",
            dialect="openai",
            client_shape="native_together",
            mode="sync",
        ),
    }

    resolved = {
        name: resolve_surface_rule(
            context=context,
            path="videos.create",
            source=SurfaceSource.RAW,
        )
        for name, context in contexts.items()
    }

    assert resolved["native"] is not None
    assert resolved["native"].kind is SurfaceKind.METERED
    assert resolved["native"].usage_basis is UsageBasis.REQUEST_DERIVED
    assert resolved["compatible"] is not None
    assert resolved["compatible"].kind is SurfaceKind.UNSUPPORTED
    assert resolved["together"] is not None
    assert resolved["together"].kind is SurfaceKind.UNSUPPORTED


@pytest.mark.unit
def test_namespace_leaf_and_exact_infrastructure_depth_remain_distinct() -> None:
    responses = resolve_surface_rule(
        context=OPENAI_SYNC,
        path="responses",
        source=SurfaceSource.RAW,
    )
    create = resolve_surface_rule(
        context=OPENAI_SYNC,
        path="responses.create",
        source=SurfaceSource.RAW,
    )
    close = resolve_surface_rule(
        context=OPENAI_SYNC,
        path="close",
        source=SurfaceSource.WRAPPER,
    )

    assert responses is not None and responses.kind is SurfaceKind.NAMESPACE
    assert responses.acknowledgment_token is None
    assert create is not None and create.kind is SurfaceKind.UNMETERED_SPEND
    assert create.acknowledgment_token == "responses.create"
    assert close is not None and close.kind is SurfaceKind.INFRASTRUCTURE
    assert (
        resolve_surface_rule(
            context=OPENAI_SYNC,
            path="close.delete",
            source=SurfaceSource.WRAPPER,
        )
        is None
    )


def _assert_audio_translations_source_shapes(context: SurfaceContext) -> None:
    raw = resolve_surface_rule(
        context=context,
        path="audio.translations",
        source=SurfaceSource.RAW,
    )
    wrapper = resolve_surface_rule(
        context=context,
        path="audio.translations",
        source=SurfaceSource.WRAPPER,
    )

    assert raw is not None and raw.source is SurfaceSource.RAW
    assert wrapper is not None and wrapper.source is SurfaceSource.WRAPPER
    assert raw is not wrapper
    raw_shapes = {
        AttributeShape(descriptor_category="cached_property", return_shape="resource"),
        AttributeShape(descriptor_category="function", return_shape="callable"),
    }
    wrapper_shape = AttributeShape(descriptor_category="property", return_shape="resource")
    assert set(raw.expected_shapes) == raw_shapes
    assert all(raw.accepts_shape(shape) for shape in raw_shapes)
    assert wrapper.expected_shapes == (wrapper_shape,)
    assert wrapper.accepts_shape(wrapper_shape)


@pytest.mark.unit
def test_sync_audio_translations_resolves_source_specific_shapes() -> None:
    _assert_audio_translations_source_shapes(OPENAI_SYNC)


@pytest.mark.unit
def test_async_audio_translations_resolves_source_specific_shapes() -> None:
    _assert_audio_translations_source_shapes(OPENAI_ASYNC)


@pytest.mark.unit
def test_rule_ledger_preserves_usage_acknowledgment_and_escape_invariants() -> None:
    assert SURFACE_RULES
    for rule in SURFACE_RULES:
        if rule.kind is SurfaceKind.METERED:
            assert rule.usage_basis is not None
        else:
            assert rule.usage_basis is None
        if rule.kind is SurfaceKind.NAMESPACE:
            assert rule.acknowledgment_token is None
            assert rule.token == rule.surface
        if rule.kind is SurfaceKind.UNMETERED_SPEND:
            assert rule.acknowledgment_token is not None
            assert rule.capability_scope is not None
            assert rule.token == rule.acknowledgment_token
        if rule.kind is SurfaceKind.INFRASTRUCTURE and rule.source in {
            SurfaceSource.RAW,
            SurfaceSource.BOTH,
        }:
            assert all(shape.return_shape != "callable" for shape in rule.expected_shapes)

    tts_rules = [rule for rule in SURFACE_RULES if rule.surface == "audio.speech.create"]
    assert any(
        rule.kind is SurfaceKind.METERED and rule.usage_basis is UsageBasis.REQUEST_DERIVED
        for rule in tts_rules
    )
    assert any(
        rule.condition is SurfaceCondition.OPENAI_UNTRACKED_TTS_MODEL
        and rule.acknowledgment_token == "audio.speech.create:gpt-4o-mini-tts"
        for rule in tts_rules
    )


@pytest.mark.unit
def test_every_exported_selector_resolves_without_ambiguity() -> None:
    for rule in SURFACE_RULES:
        sources = (
            (SurfaceSource.RAW, SurfaceSource.WRAPPER)
            if rule.source is SurfaceSource.BOTH
            else (rule.source,)
        )
        for selector in rule.selectors:
            if selector.provider is None:
                assert selector.dialect == "openai"
                assert selector.client_shape == "openai_sdk"
            assert selector.dialect is not None
            assert selector.client_shape is not None
            assert selector.mode is not None
            context = SurfaceContext(
                provider=selector.provider or "representative_compatible",
                dialect=selector.dialect,
                client_shape=selector.client_shape,
                mode=selector.mode,
            )
            for source in sources:
                assert (
                    resolve_surface_rule(
                        context=context,
                        path=rule.surface,
                        source=source,
                        condition=rule.condition,
                    )
                    is rule
                )


@pytest.mark.unit
def test_named_openai_compatible_provider_uses_generic_shape_rules() -> None:
    chat = resolve_surface_rule(
        context=GROQ_SYNC,
        path="chat.completions.create",
        source=SurfaceSource.RAW,
    )
    responses = resolve_surface_rule(
        context=GROQ_SYNC,
        path="responses",
        source=SurfaceSource.RAW,
    )
    video = resolve_surface_rule(
        context=GROQ_SYNC,
        path="videos.create",
        source=SurfaceSource.RAW,
    )

    assert chat is not None and chat.usage_basis is UsageBasis.PROVIDER_OR_ESTIMATE
    assert responses is not None and responses.kind is SurfaceKind.NAMESPACE
    assert video is not None and video.kind is SurfaceKind.UNSUPPORTED


@pytest.mark.unit
def test_raw_escape_hatches_carry_exact_scopes() -> None:
    expectations = {
        "post": CapabilityScope.ARBITRARY_ENDPOINT,
        "with_options": CapabilityScope.CLIENT,
        "responses.with_raw_response": CapabilityScope.RAW_RESPONSE,
    }

    for path, scope in expectations.items():
        rule = resolve_surface_rule(
            context=OPENAI_SYNC,
            path=path,
            source=SurfaceSource.RAW,
        )
        assert rule is not None
        assert rule.kind is SurfaceKind.UNMETERED_SPEND
        assert rule.capability_scope is scope


@pytest.mark.unit
def test_exact_inert_provider_configuration_is_safe_but_raw_callables_are_not() -> None:
    base_url = resolve_surface_rule(
        context=OPENAI_SYNC,
        path="base_url",
        source=SurfaceSource.RAW,
    )
    timeout = resolve_surface_rule(
        context=OPENAI_SYNC,
        path="timeout",
        source=SurfaceSource.RAW,
    )
    is_closed = resolve_surface_rule(
        context=OPENAI_SYNC,
        path="is_closed",
        source=SurfaceSource.RAW,
    )

    assert base_url is not None and base_url.kind is SurfaceKind.METADATA
    assert timeout is not None and timeout.kind is SurfaceKind.INFRASTRUCTURE
    assert is_closed is not None and is_closed.kind is SurfaceKind.UNMETERED_SPEND


@pytest.mark.unit
def test_safe_descriptor_rows_pin_the_evaluated_attribute_shape() -> None:
    evaluated_shapes = {
        "auth_headers": "mapping",
        "base_url": "resource",
        "default_headers": "mapping",
        "default_query": "mapping",
        "model_name": "scalar",
        "qs": "resource",
        "user_agent": "scalar",
        "vertexai": "scalar",
    }
    safe_rules = [
        rule
        for rule in SURFACE_RULES
        if rule.kind in {SurfaceKind.METADATA, SurfaceKind.INFRASTRUCTURE}
    ]

    assert safe_rules
    for rule in safe_rules:
        assert any(
            shape.return_shape != "unevaluated_descriptor" for shape in rule.expected_shapes
        ), rule.rule_id
        if rule.surface in evaluated_shapes:
            assert (
                AttributeShape(
                    descriptor_category="property",
                    return_shape=evaluated_shapes[rule.surface],
                )
                in rule.expected_shapes
            )


@pytest.mark.unit
def test_generated_json_is_the_deterministic_python_export(tmp_path: Path) -> None:
    exporter = _export_module()
    output = tmp_path / "surface-classification.json"
    contract = surface_contract_data()
    expected = json.dumps(contract, indent=2, sort_keys=True) + "\n"

    exporter.write_contract(output)

    assert exporter.DEFAULT_OUTPUT == CONTRACT_PATH
    assert contract["contract_version"] == 1
    assert all(row["token"] for row in contract["rules"])
    assert output.read_text(encoding="utf-8") == expected
    assert not COMMITTED_CONTRACT_PATH.exists()


@pytest.mark.unit
def test_check_writes_contract_before_failing_closed_on_empty_reports(tmp_path: Path) -> None:
    exporter = _export_module()
    output = tmp_path / "nested" / "surface-classification.json"
    reports = tmp_path / "reports"
    reports.mkdir()

    mismatches = exporter.generate_and_check(output, reports_dir=reports)

    assert output.read_text(encoding="utf-8") == exporter.render_contract()
    assert mismatches == (f"no provider surface reports found in {reports}",)
