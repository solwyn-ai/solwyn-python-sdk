"""Deterministic, sans-I/O coverage manifest and literal audit pins."""

from __future__ import annotations

import functools
import inspect
from types import SimpleNamespace

import openai
import pytest

import solwyn
from solwyn import (
    CoverageAuditEntry,
    CoverageEntry,
    CoverageExpectation,
    CoverageFingerprint,
    CoverageMismatchError,
    CoverageReport,
    SurfaceInspectionError,
)
from solwyn._base import _SolwynBase
from solwyn._surfaces import SurfaceContext

OPENAI_STRICT_FINGERPRINT = CoverageFingerprint(
    guarded_namespaces="sha256:f8cd6c254dc5fb2076d33aeff6d9e96cd12b3d63cc89850ba86f0d7b5ed62818",
    tracked="sha256:ee52e554ddf531bea4560f69fdbef1ca0ac90e433fb9f93fba6e291d39e2aebc",
    untracked="sha256:989f425b3fd2f431f691fc83fda676879ef3cda89aafa123d1943af957ce7700",
    unknown="sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    scoped_escapes="sha256:3d8b7cfcf068bbf907c9e257e5c596844bec9a52b5414898835715f4a7dd406c",
    blocked="sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    unsupported="sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    conditional="sha256:ce837f71d1fc97849872c5d0f86b0b1f26e1bc4e46a29c3b1b8004bf4b9bcb77",
    safe="sha256:9029368e5fa0a7bf4260cc782560c8ec9a53c948fc280102b1c3633eee5234c5",
)


class _Completions:
    def create(self) -> None:
        raise AssertionError("coverage must never invoke provider operations")


class _Chat:
    @functools.cached_property
    def completions(self) -> _Completions:
        return _Completions()


class _Responses:
    def create(self) -> None:
        raise AssertionError("coverage must never invoke provider operations")


class _NovelResource:
    marker = "content-free"


class _OpenAIShape:
    @functools.cached_property
    def chat(self) -> _Chat:
        return _Chat()

    @functools.cached_property
    def responses(self) -> _Responses:
        return _Responses()

    @property
    def base_url(self) -> object:
        raise AssertionError("coverage must not evaluate terminal descriptors")

    def post(self) -> None:
        raise AssertionError("coverage must never invoke provider operations")

    def novel_operation(self) -> None:
        raise AssertionError("coverage must never invoke unknown operations")

    novel_scalar = "safe structural value"
    novel_resource = _NovelResource()


_OpenAIShape.__module__ = "openai._client"


def _wrapper(
    raw: object,
    *,
    provider: str = "openai",
    dialect: str = "openai",
    client_shape: str = "openai_sdk",
    mode: str = "sync",
    posture: str = "raise",
    acknowledgments: frozenset[str] = frozenset(),
    fallbacks: tuple[tuple[object, str, str, str], ...] = (),
) -> _SolwynBase:
    wrapper = object.__new__(_SolwynBase)
    runtimes = [
        SimpleNamespace(
            sdk_client=raw,
            adapter=SimpleNamespace(name=provider, dialect=dialect),
            entry=SimpleNamespace(model="primary-model"),
        )
    ]
    for fallback_raw, fallback_provider, fallback_dialect, fallback_model in fallbacks:
        runtimes.append(
            SimpleNamespace(
                sdk_client=fallback_raw,
                adapter=SimpleNamespace(name=fallback_provider, dialect=fallback_dialect),
                entry=SimpleNamespace(model=fallback_model),
            )
        )
    wrapper._runtimes = runtimes
    wrapper._surface_context = SurfaceContext(
        provider=provider,
        dialect=dialect,
        client_shape=client_shape,
        mode=mode,
    )
    wrapper._config = SimpleNamespace(
        on_unmetered=posture,
        acknowledge_untracked=acknowledgments,
    )
    return wrapper


def _by_identity(report: CoverageReport) -> dict[tuple[str, str | None], CoverageEntry]:
    return {(entry.surface, entry.condition): entry for entry in report.entries}


@pytest.mark.unit
def test_coverage_reports_effective_actions_and_unknown_shapes_without_io() -> None:
    report = solwyn.coverage(_wrapper(_OpenAIShape()))
    entries = _by_identity(report)

    assert report.provider == "openai"
    assert report.dialect == "openai"
    assert report.client_shape == "openai_sdk"
    assert report.posture == "raise"
    assert report.acknowledgments == ()
    assert report.provider_chain[0].provider == "openai"
    assert report.entries == tuple(
        sorted(report.entries, key=lambda entry: (entry.surface, entry.rule_id))
    )

    assert (entries[("chat", None)].policy_action, entries[("chat", None)].dispatch_action) == (
        "pass",
        "guard",
    )
    assert (
        entries[("chat.completions.create", None)].policy_action,
        entries[("chat.completions.create", None)].dispatch_action,
        entries[("chat.completions.create", None)].usage_basis,
    ) == ("track", "intercept", "provider")
    base_url = entries[("base_url", None)]
    assert (base_url.policy_action, base_url.dispatch_action) == (
        "pass",
        "return",
    )
    assert (entries[("close", None)].policy_action, entries[("close", None)].dispatch_action) == (
        "pass",
        "return",
    )
    assert (entries[("post", None)].policy_action, entries[("post", None)].dispatch_action) == (
        "raise",
        "refuse",
    )

    unknown = entries[("novel_operation", None)]
    assert unknown.kind == "unknown"
    assert unknown.rule_id == "unknown:openai_sdk:sync:openai:novel_operation"
    assert unknown.token == "novel_operation"
    assert unknown.source == "raw"
    assert unknown.expected_descriptor_category is None
    assert unknown.observed_descriptor_category == "function"
    assert unknown.expected_return_shape is None
    assert unknown.observed_return_shape == "callable"
    assert (unknown.policy_action, unknown.dispatch_action) == ("raise", "refuse")


@pytest.mark.unit
def test_allowed_unknown_shapes_return_or_guard_and_acknowledgments_are_sorted() -> None:
    report = solwyn.coverage(
        _wrapper(
            _OpenAIShape(),
            posture="allow",
            acknowledgments=frozenset({"responses.create", "post"}),
        )
    )
    entries = _by_identity(report)

    assert report.acknowledgments == ("post", "responses.create")
    assert entries[("novel_scalar", None)].policy_action == "allow"
    assert entries[("novel_scalar", None)].dispatch_action == "return"
    assert entries[("novel_resource", None)].policy_action == "allow"
    assert entries[("novel_resource", None)].dispatch_action == "guard"
    assert entries[("post", None)].policy_action == "acknowledged"
    assert entries[("post", None)].dispatch_action == "return"


@pytest.mark.unit
def test_conditional_and_wrapper_only_rows_remain_visible() -> None:
    compatible = _wrapper(_OpenAIShape(), provider="groq", posture="warn")
    entries = _by_identity(solwyn.coverage(compatible))

    tts = entries[("audio.speech.create", "openai_untracked_tts_model")]
    assert tts.token == "audio.speech.create:gpt-4o-mini-tts"
    assert tts.source == "synthetic_policy"
    assert (tts.policy_action, tts.dispatch_action) == ("warn", "return")

    video = entries[("videos.create", None)]
    assert video.kind == "unsupported"
    assert video.source == "both"
    assert video.policy_action == "unsupported"
    assert video.dispatch_action == "refuse"
    assert video.reason == "selected provider adapter does not support this wrapper surface"


@pytest.mark.unit
def test_usage_basis_aggregates_only_reachable_runtimes() -> None:
    report = solwyn.coverage(
        _wrapper(
            _OpenAIShape(),
            fallbacks=((_OpenAIShape(), "groq", "openai", "fallback-model"),),
        )
    )
    entries = _by_identity(report)

    assert [runtime.provider for runtime in report.provider_chain] == ["openai", "groq"]
    assert entries[("chat.completions.create", None)].usage_basis == "provider_or_estimate"
    assert entries[("embeddings.create", None)].usage_basis == "provider_or_estimate"
    assert entries[("audio.transcriptions.create", None)].usage_basis == "provider"
    assert entries[("videos.create", None)].usage_basis == "request_derived"


@pytest.mark.unit
def test_google_embedding_keeps_its_rule_owned_provider_or_estimate_basis() -> None:
    class Models:
        def embed_content(self) -> None:
            raise AssertionError("coverage must never invoke provider operations")

    class GoogleShape:
        @functools.cached_property
        def models(self) -> Models:
            return Models()

    GoogleShape.__module__ = "google.genai.client"
    report = solwyn.coverage(
        _wrapper(
            GoogleShape(),
            provider="google",
            dialect="google",
            client_shape="google_genai",
        )
    )

    assert _by_identity(report)[("models.embed_content", None)].usage_basis == (
        "provider_or_estimate"
    )


@pytest.mark.unit
def test_legacy_google_chat_aggregates_a_compatible_fallback_basis() -> None:
    class LegacyGoogle:
        def generate_content(self) -> None:
            raise AssertionError("coverage must never invoke provider operations")

    LegacyGoogle.__module__ = "google.generativeai.generative_models"
    report = solwyn.coverage(
        _wrapper(
            LegacyGoogle(),
            provider="google",
            dialect="google",
            client_shape="google_generativeai",
            fallbacks=((_OpenAIShape(), "groq", "openai", "fallback-model"),),
        )
    )

    assert _by_identity(report)[("generate_content", None)].usage_basis == ("provider_or_estimate")


@pytest.mark.unit
def test_native_together_uses_only_its_applicable_surface_rules() -> None:
    class Images:
        def generate(self) -> None:
            raise AssertionError("coverage must never invoke provider operations")

    class Videos:
        def create(self) -> None:
            raise AssertionError("coverage must never invoke provider operations")

    class Together:
        @functools.cached_property
        def chat(self) -> _Chat:
            return _Chat()

        @functools.cached_property
        def images(self) -> Images:
            return Images()

        @functools.cached_property
        def videos(self) -> Videos:
            return Videos()

    Together.__module__ = "together"
    report = solwyn.coverage(
        _wrapper(
            Together(),
            provider="together",
            client_shape="native_together",
        )
    )
    entries = _by_identity(report)

    assert entries[("chat.completions.create", None)].usage_basis == "provider_or_estimate"
    assert entries[("images.generate", None)].usage_basis == "provider_and_request"
    assert entries[("images.edit", None)].source == "wrapper"
    assert entries[("videos.create", None)].kind == "unsupported"


@pytest.mark.unit
def test_safe_shape_drift_reenters_unknown_posture_without_descriptor_evaluation() -> None:
    class Drifted:
        def __init__(self) -> None:
            self.evaluations = 0

        @property
        def max_retries(self) -> object:
            self.evaluations += 1
            raise AssertionError("coverage must not evaluate terminal descriptors")

    Drifted.__module__ = "openai._client"
    raw = Drifted()
    entry = _by_identity(solwyn.coverage(_wrapper(raw)))[("max_retries", None)]

    assert entry.kind == "unknown"
    assert entry.expected_descriptor_category == "attribute"
    assert entry.observed_descriptor_category == "property"
    assert entry.expected_return_shape == "scalar"
    assert entry.observed_return_shape == "unevaluated_descriptor"
    assert (entry.policy_action, entry.dispatch_action) == ("raise", "refuse")
    assert raw.evaluations == 0


@pytest.mark.unit
def test_inaccessible_declared_namespace_and_cycle_fail_with_exact_paths() -> None:
    class Broken:
        @functools.cached_property
        def chat(self) -> object:
            raise RuntimeError("PRIVATE_DESCRIPTOR_CONTENT")

    Broken.__module__ = "openai._client"
    with pytest.raises(SurfaceInspectionError) as inaccessible:
        solwyn.coverage(_wrapper(Broken()))
    assert inaccessible.value.path == "chat"
    assert inaccessible.value.stage == "namespace_evaluation"
    assert "PRIVATE_DESCRIPTOR_CONTENT" not in str(inaccessible.value)

    class Cycle:
        @functools.cached_property
        def chat(self) -> Cycle:
            return self

    Cycle.__module__ = "openai._client"
    with pytest.raises(SurfaceInspectionError) as cycle:
        solwyn.coverage(_wrapper(Cycle()))
    assert cycle.value.path == "chat"
    assert cycle.value.stage == "cycle"


@pytest.mark.unit
def test_bedrock_service_model_operation_is_visible_only_through_wrapper_reachability() -> None:
    class ServiceOnlyBedrock:
        meta = SimpleNamespace(
            service_model=SimpleNamespace(operation_names=("Converse", "FutureOperation"))
        )

    report = solwyn.coverage(
        _wrapper(
            ServiceOnlyBedrock(),
            provider="bedrock",
            dialect="bedrock",
            client_shape="bedrock_boto3",
        )
    )
    entries = _by_identity(report)

    assert entries[("converse", None)].kind == "metered"
    assert entries[("converse", None)].dispatch_action == "intercept"
    assert ("future_operation", None) not in entries


@pytest.mark.unit
def test_sync_and_async_reports_match_for_equivalent_known_shapes() -> None:
    class KnownOnly:
        @functools.cached_property
        def chat(self) -> _Chat:
            return _Chat()

    KnownOnly.__module__ = "openai._client"

    sync = solwyn.coverage(_wrapper(KnownOnly(), mode="sync"))
    async_report = solwyn.coverage(_wrapper(KnownOnly(), mode="async"))

    assert sync == async_report


def _audit_entry(**updates: object) -> CoverageAuditEntry:
    values: dict[str, object] = {
        "rule_id": "rule.operation",
        "surface": "resources.operation",
        "token": "resources.operation",
        "kind": "unmetered_spend",
        "policy_action": "warn",
        "dispatch_action": "return",
        "usage_basis": None,
        "source": "raw",
        "capability_scope": "operation",
        "condition": None,
        "reason": None,
        "expected_descriptor_category": "function",
        "observed_descriptor_category": "function",
        "expected_return_shape": "callable",
        "observed_return_shape": "callable",
    }
    values.update(updates)
    return CoverageAuditEntry(**values)


def _literal_expectation(entry: CoverageAuditEntry) -> CoverageExpectation:
    return CoverageExpectation(
        guarded_namespaces=(),
        tracked=(),
        untracked=(entry,),
        unknown=(),
        scoped_escapes=(),
        blocked=(),
        unsupported=(),
        conditional=(),
        safe=(),
    )


def _report_for_expectation(entry: CoverageAuditEntry) -> CoverageReport:
    return CoverageReport(
        provider="test",
        dialect="test",
        client_shape="test_sdk",
        posture="warn",
        provider_chain=(),
        acknowledgments=(),
        entries=(CoverageEntry(**entry.model_dump()),),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("surface", "resources.future_operation"),
        ("kind", "unknown"),
        ("policy_action", "allow"),
        ("dispatch_action", "guard"),
        ("expected_descriptor_category", "property"),
        ("observed_descriptor_category", "property"),
        ("expected_return_shape", "resource"),
        ("observed_return_shape", "resource"),
        ("capability_scope", "client"),
        ("usage_basis", "provider_or_estimate"),
    ],
)
def test_expect_compares_every_audit_dimension_in_both_directions(
    field: str,
    changed: object,
) -> None:
    literal = _audit_entry()
    report = _report_for_expectation(literal)
    report.expect(_literal_expectation(literal))

    changed_literal = literal.model_copy(update={field: changed})
    with pytest.raises(CoverageMismatchError) as caught:
        report.expect(_literal_expectation(changed_literal))

    assert "untracked" in str(caught.value)
    assert "rule.operation" in str(caught.value)


@pytest.mark.unit
def test_expect_reports_additions_and_removals_by_category_and_rule_id() -> None:
    entry = _audit_entry()
    report = _report_for_expectation(entry)
    empty = CoverageExpectation(
        guarded_namespaces=(),
        tracked=(),
        untracked=(),
        unknown=(),
        scoped_escapes=(),
        blocked=(),
        unsupported=(),
        conditional=(),
        safe=(),
    )

    with pytest.raises(CoverageMismatchError) as added:
        report.expect(empty)
    assert "added" in str(added.value)
    assert "rule.operation" in str(added.value)

    with pytest.raises(CoverageMismatchError) as removed:
        _report_for_expectation(_audit_entry(rule_id="rule.other")).expect(
            _literal_expectation(entry)
        )
    assert "removed" in str(removed.value)
    assert "rule.operation" in str(removed.value)


@pytest.mark.unit
def test_compact_fingerprint_is_frozen_literal_and_covers_every_category() -> None:
    report = _report_for_expectation(_audit_entry())
    fingerprint = report.fingerprint()

    assert isinstance(fingerprint, CoverageFingerprint)
    assert fingerprint.untracked.startswith("sha256:")
    report.expect(fingerprint)

    changed = fingerprint.model_copy(update={"untracked": "sha256:" + "0" * 64})
    with pytest.raises(CoverageMismatchError, match="untracked"):
        report.expect(changed)


@pytest.mark.unit
def test_real_openai_shape_matches_the_literal_exhaustive_audit_fingerprint() -> None:
    raw = openai.OpenAI(api_key="sk-test")
    try:
        report = solwyn.coverage(_wrapper(raw))
        report.expect(OPENAI_STRICT_FINGERPRINT)
    finally:
        raw.close()


@pytest.mark.unit
def test_coverage_rejects_non_solwyn_objects_without_importing_client_module() -> None:
    with pytest.raises(TypeError, match="Solwyn or AsyncSolwyn"):
        solwyn.coverage(object())

    import solwyn._coverage as coverage_module

    assert "solwyn.client" not in inspect.getsource(coverage_module)


@pytest.mark.unit
def test_public_coverage_models_are_frozen() -> None:
    report = solwyn.coverage(_wrapper(_OpenAIShape()))

    with pytest.raises(Exception, match="frozen"):
        report.provider = "changed"
