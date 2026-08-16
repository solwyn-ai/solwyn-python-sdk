"""Tests for the contextual provider capability contract."""

from __future__ import annotations

import base64
import importlib.util
import json
import shlex
import zlib
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from solwyn._surface_graph import declared_namespace_paths
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
    context_is_declared,
    payload_fingerprint,
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
AZURE_OPENAI_SYNC = SurfaceContext(
    provider="azure_openai",
    dialect="openai",
    client_shape="openai_sdk",
    mode="sync",
)
AZURE_OPENAI_ASYNC = SurfaceContext(
    provider="azure_openai",
    dialect="openai",
    client_shape="openai_sdk",
    mode="async",
)
ANTHROPIC_SYNC = SurfaceContext(
    provider="anthropic",
    dialect="anthropic",
    client_shape="anthropic_sdk",
    mode="sync",
)
ANTHROPIC_ASYNC = SurfaceContext(
    provider="anthropic",
    dialect="anthropic",
    client_shape="anthropic_sdk",
    mode="async",
)
GROQ_SYNC = SurfaceContext(
    provider="groq",
    dialect="openai",
    client_shape="openai_sdk",
    mode="sync",
)
GOOGLE_GENERATIVEAI_SYNC = SurfaceContext(
    provider="google",
    dialect="google",
    client_shape="google_generativeai",
    mode="sync",
)
GOOGLE_GENAI_SYNC = SurfaceContext(
    provider="google",
    dialect="google",
    client_shape="google_genai",
    mode="sync",
)

_DECLARED_CONTEXT_TUPLES = (
    ("openai", "openai", "openai_sdk", "sync"),
    ("openai", "openai", "openai_sdk", "async"),
    ("azure_openai", "openai", "openai_sdk", "sync"),
    ("azure_openai", "openai", "openai_sdk", "async"),
    ("openai_compatible", "openai", "openai_sdk", "sync"),
    ("openai_compatible", "openai", "openai_sdk", "async"),
    ("together", "openai", "openai_sdk", "sync"),
    ("together", "openai", "openai_sdk", "async"),
    ("together", "openai", "native_together", "sync"),
    ("together", "openai", "native_together", "async"),
    ("anthropic", "anthropic", "anthropic_sdk", "sync"),
    ("anthropic", "anthropic", "anthropic_sdk", "async"),
    ("google", "google", "google_genai", "sync"),
    ("google", "google", "google_genai", "async"),
    ("google", "google", "google_generativeai", "sync"),
    ("bedrock", "bedrock", "bedrock_boto3", "sync"),
    ("bedrock", "bedrock", "bedrock_aioboto3", "async"),
)

_UNDECLARED_CONTEXT_TUPLES = (
    ("bedrock", "bedrock", "bedrock_boto3", "async"),
    ("bedrock", "bedrock", "bedrock_aioboto3", "sync"),
    ("google", "google", "google_generativeai", "async"),
)


@pytest.mark.unit
def test_declared_namespace_paths_matches_the_contract_frontier() -> None:
    # Arrange
    context = SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )

    # Act
    paths = declared_namespace_paths(context)

    # Assert
    assert paths
    assert all(
        resolve_surface_rule(context=context, path=path, source=SurfaceSource.RAW) for path in paths
    )


@pytest.mark.unit
def test_every_constructible_context_is_declared_or_rejected() -> None:
    # Arrange / Act / Assert
    for provider, dialect, client_shape, mode in _DECLARED_CONTEXT_TUPLES:
        context = SurfaceContext(
            provider=provider,
            dialect=dialect,
            client_shape=client_shape,
            mode=mode,
        )
        assert context_is_declared(context), context

    for provider, dialect, client_shape, mode in _UNDECLARED_CONTEXT_TUPLES:
        context = SurfaceContext(
            provider=provider,
            dialect=dialect,
            client_shape=client_shape,
            mode=mode,
        )
        assert not context_is_declared(context), context

    named_compat = SurfaceContext(
        provider="groq",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )
    assert context_is_declared(named_compat)


@pytest.mark.unit
def test_embedded_surface_rule_rows_have_exactly_eleven_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solwyn import _surfaces

    # Arrange
    malformed = {
        "rules": [["too", "short"]],
        "schema_version": _surfaces.CONTRACT_VERSION,
    }
    encoded = base64.b85encode(zlib.compress(json.dumps(malformed).encode("utf-8"))).decode("ascii")
    monkeypatch.setattr(_surfaces, "_GENERATED_SURFACE_RULE_PAYLOAD", encoded)

    # Act / Assert
    with pytest.raises(RuntimeError, match="invalid embedded surface rule row"):
        _surfaces._build_surface_rules()


def _script_module(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load surface contract script at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_module() -> ModuleType:
    return _script_module("export_surface_contract")


@pytest.mark.unit
@pytest.mark.parametrize("script_name", ["export_surface_contract", "embed_surface_rules"])
def test_surface_contract_generators_refuse_another_installed_checkout_before_work(
    script_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import solwyn

    # Arrange
    module = _script_module(script_name)
    installed = tmp_path / "site-packages" / "solwyn" / "__init__.py"
    monkeypatch.setattr(solwyn, "__file__", str(installed))

    def fail_if_work_starts() -> None:
        raise AssertionError("argument parsing started before checkout identity validation")

    monkeypatch.setattr(module, "_parser", fail_if_work_starts)

    # Act
    result = module.main()

    # Assert
    expected = (ROOT / "src" / "solwyn" / "__init__.py").resolve()
    assert result == 1
    assert (
        f"solwyn resolves to {installed.resolve()}, not this checkout ({expected}); run via uv run"
        in capsys.readouterr().out
    )


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
def test_report_comparison_emits_dead_rule_advisories(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Arrange
    exporter = _export_module()
    report = {
        "provider": "anthropic",
        "client_shape": "anthropic_sdk",
        "mode": "sync",
        "observations": [],
    }
    (tmp_path / "anthropic_sync--latest.json").write_text(json.dumps(report), encoding="utf-8")

    # Act
    mismatches = exporter.compare_report_directory(tmp_path)

    # Assert
    captured = capsys.readouterr().out
    assert mismatches == ()
    assert "advisory:" in captured
    assert "unobserved" in captured


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
    assert create is not None and create.kind is SurfaceKind.METERED
    assert create.usage_basis is UsageBasis.PROVIDER
    assert create.acknowledgment_token is None
    assert close is not None and close.kind is SurfaceKind.INFRASTRUCTURE
    assert (
        resolve_surface_rule(
            context=OPENAI_SYNC,
            path="close.delete",
            source=SurfaceSource.WRAPPER,
        )
        is None
    )


@pytest.mark.unit
@pytest.mark.parametrize("context", [OPENAI_SYNC, OPENAI_ASYNC])
def test_native_openai_responses_parse_is_metered_on_raw_and_wrapper(
    context: SurfaceContext,
) -> None:
    # Arrange and act.
    for source in (SurfaceSource.RAW, SurfaceSource.WRAPPER):
        rule = resolve_surface_rule(
            context=context,
            path="responses.parse",
            source=source,
        )

        # Assert.
        assert rule is not None
        assert rule.kind is SurfaceKind.METERED
        assert rule.source is SurfaceSource.BOTH
        assert rule.usage_basis is UsageBasis.PROVIDER
        assert rule.acknowledgment_token is None
        assert rule.policy_action == "track"
        assert rule.dispatch_action == "intercept"
        assert any(shape.return_shape == "callable" for shape in rule.expected_shapes)


@pytest.mark.unit
@pytest.mark.parametrize("context", [AZURE_OPENAI_SYNC, AZURE_OPENAI_ASYNC])
def test_azure_responses_namespace_and_spend_leaves_are_available_from_both_sources(
    context: SurfaceContext,
) -> None:
    for source in (SurfaceSource.RAW, SurfaceSource.WRAPPER):
        namespace = resolve_surface_rule(
            context=context,
            path="responses",
            source=source,
        )
        assert namespace is not None
        assert namespace.kind is SurfaceKind.NAMESPACE
        assert namespace.source is SurfaceSource.BOTH
        assert namespace.acknowledgment_token is None

        for path in ("responses.create", "responses.parse", "responses.stream"):
            rule = resolve_surface_rule(context=context, path=path, source=source)
            assert rule is not None
            assert rule.kind is SurfaceKind.METERED
            assert rule.source is SurfaceSource.BOTH
            assert rule.usage_basis is UsageBasis.PROVIDER
            assert rule.acknowledgment_token is None
            assert rule.policy_action == "track"
            assert rule.dispatch_action == "intercept"


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    [
        GROQ_SYNC,
        SurfaceContext(
            provider="openai_compatible",
            dialect="openai",
            client_shape="openai_sdk",
            mode="sync",
        ),
        SurfaceContext(
            provider="together",
            dialect="openai",
            client_shape="openai_sdk",
            mode="sync",
        ),
    ],
)
def test_compatible_responses_parse_stays_raw_unmetered_without_wrapper_rule(
    context: SurfaceContext,
) -> None:
    # Arrange and act.
    raw = resolve_surface_rule(
        context=context,
        path="responses.parse",
        source=SurfaceSource.RAW,
    )
    wrapper = resolve_surface_rule(
        context=context,
        path="responses.parse",
        source=SurfaceSource.WRAPPER,
    )

    # Assert.
    assert raw is not None
    assert raw.kind is SurfaceKind.UNMETERED_SPEND
    assert raw.source is SurfaceSource.RAW
    assert raw.acknowledgment_token == "responses.parse"
    assert wrapper is None


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
def test_legacy_google_generation_is_exactly_wrapper_applicable() -> None:
    raw = resolve_surface_rule(
        context=GOOGLE_GENERATIVEAI_SYNC,
        path="generate_content",
        source=SurfaceSource.RAW,
    )
    wrapper = resolve_surface_rule(
        context=GOOGLE_GENERATIVEAI_SYNC,
        path="generate_content",
        source=SurfaceSource.WRAPPER,
    )

    assert raw is not None and raw.kind is SurfaceKind.METERED
    assert wrapper is raw
    assert (
        resolve_surface_rule(
            context=GOOGLE_GENAI_SYNC,
            path="generate_content",
            source=SurfaceSource.WRAPPER,
        )
        is None
    )


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
@pytest.mark.parametrize(
    ("context", "path"),
    [
        (OPENAI_SYNC, "realtime.calls"),
        (OPENAI_ASYNC, "realtime.calls"),
        (ANTHROPIC_SYNC, "beta.environments.work"),
        (ANTHROPIC_ASYNC, "beta.environments.work"),
    ],
)
def test_supported_operation_containers_are_guarded_namespaces(
    context: SurfaceContext,
    path: str,
) -> None:
    # Arrange / Act
    rule = resolve_surface_rule(
        context=context,
        path=path,
        source=SurfaceSource.RAW,
    )

    # Assert
    assert rule is not None
    assert rule.kind is SurfaceKind.NAMESPACE
    assert rule.acknowledgment_token is None
    assert rule.capability_scope is None
    assert rule.expected_shapes == (
        AttributeShape(descriptor_category="cached_property", return_shape="resource"),
    )


def _assert_unmetered_operation(context: SurfaceContext, path: str) -> None:
    rule = resolve_surface_rule(
        context=context,
        path=path,
        source=SurfaceSource.RAW,
    )

    assert rule is not None
    assert rule.kind is SurfaceKind.UNMETERED_SPEND
    assert rule.acknowledgment_token == path
    assert rule.capability_scope is CapabilityScope.OPERATION
    assert rule.expected_shapes == (
        AttributeShape(descriptor_category="function", return_shape="callable"),
    )


def _assert_raw_response_family(
    context: SurfaceContext,
    prefix: str,
    operations: set[str],
) -> None:
    for response_container in ("with_raw_response", "with_streaming_response"):
        container_path = f"{prefix}.{response_container}"
        container = resolve_surface_rule(
            context=context,
            path=container_path,
            source=SurfaceSource.RAW,
        )
        assert container is not None
        assert container.kind is SurfaceKind.UNMETERED_SPEND
        assert container.acknowledgment_token == container_path
        assert container.capability_scope is CapabilityScope.RAW_RESPONSE
        assert container.expected_shapes == (
            AttributeShape(descriptor_category="cached_property", return_shape="resource"),
        )
        for operation in operations:
            path = f"{container_path}.{operation}"
            leaf = resolve_surface_rule(
                context=context,
                path=path,
                source=SurfaceSource.RAW,
            )
            assert leaf is not None
            assert leaf.kind is SurfaceKind.UNMETERED_SPEND
            assert leaf.acknowledgment_token == path
            assert leaf.capability_scope is CapabilityScope.RAW_RESPONSE
            assert leaf.expected_shapes == (
                AttributeShape(descriptor_category="function", return_shape="callable"),
            )


@pytest.mark.unit
def test_realtime_calls_children_have_exact_operation_and_raw_response_scopes() -> None:
    # Arrange
    operations = {"accept", "create", "hangup", "refer", "reject"}

    # Act / Assert
    for context in (OPENAI_SYNC, OPENAI_ASYNC):
        for operation in operations:
            _assert_unmetered_operation(context, f"realtime.calls.{operation}")
        _assert_raw_response_family(context, "realtime.calls", operations)


@pytest.mark.unit
def test_anthropic_work_children_cover_sync_async_and_raw_response_shapes() -> None:
    # Arrange
    operations = {"ack", "heartbeat", "list", "poll", "retrieve", "stats", "stop", "update"}

    # Act / Assert
    for context in (ANTHROPIC_SYNC, ANTHROPIC_ASYNC):
        for operation in operations:
            _assert_unmetered_operation(context, f"beta.environments.work.{operation}")
        _assert_raw_response_family(context, "beta.environments.work", operations)
    for operation in ("poller", "worker"):
        _assert_unmetered_operation(ANTHROPIC_ASYNC, f"beta.environments.work.{operation}")
        assert (
            resolve_surface_rule(
                context=ANTHROPIC_SYNC,
                path=f"beta.environments.work.{operation}",
                source=SurfaceSource.RAW,
            )
            is None
        )


@pytest.mark.unit
def test_supported_container_children_preserve_reviewed_selectors_and_modes() -> None:
    # Arrange
    parents = {
        prefix: next(rule for rule in SURFACE_RULES if rule.surface == prefix)
        for prefix in ("realtime.calls", "beta.environments.work")
    }

    # Act / Assert
    for prefix, parent in parents.items():
        family = [rule for rule in SURFACE_RULES if rule.surface.startswith(f"{prefix}.")]
        assert family
        for rule in family:
            if rule.surface in {
                "beta.environments.work.poller",
                "beta.environments.work.worker",
            }:
                assert {selector.mode for selector in rule.selectors} == {"async"}
                assert set(rule.selectors) <= set(parent.selectors)
            else:
                assert rule.selectors == parent.selectors


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
    expected_contract = dict(contract)
    expected_contract["source_payload_fingerprint"] = payload_fingerprint()
    expected = json.dumps(expected_contract, indent=2, sort_keys=True) + "\n"

    exporter.write_contract(output)

    assert exporter.DEFAULT_OUTPUT == CONTRACT_PATH
    assert contract["contract_version"] == 1
    assert all(row["token"] for row in contract["rules"])
    assert output.read_text(encoding="utf-8") == expected
    assert not COMMITTED_CONTRACT_PATH.exists()


@pytest.mark.unit
def test_check_bootstraps_then_fails_closed_on_empty_reports(tmp_path: Path) -> None:
    exporter = _export_module()
    output = tmp_path / "nested" / "surface-classification.json"
    reports = tmp_path / "reports"
    reports.mkdir()

    mismatches = exporter.generate_and_check(output, reports_dir=reports)

    assert output.read_text(encoding="utf-8") == exporter.render_contract()
    assert mismatches == (f"no provider surface reports found in {reports}",)


@pytest.mark.unit
def test_check_preserves_and_rejects_a_differing_expanded_contract(tmp_path: Path) -> None:
    # Arrange
    exporter = _export_module()
    output = tmp_path / "surface-classification.json"
    edited = exporter.render_contract().replace("{", "{\n ", 1)
    output.write_text(edited, encoding="utf-8")

    # Act
    mismatches = exporter.generate_and_check(output)

    # Assert
    assert output.read_text(encoding="utf-8") == edited
    assert len(mismatches) == 1
    assert "differs from the embedded payload" in mismatches[0]


@pytest.mark.unit
def test_check_bootstraps_a_missing_expanded_contract(tmp_path: Path) -> None:
    # Arrange
    exporter = _export_module()
    output = tmp_path / "nested" / "surface-classification.json"

    # Act
    mismatches = exporter.generate_and_check(output)

    # Assert
    assert mismatches == ()
    assert output.read_text(encoding="utf-8") == exporter.render_contract()
