"""Deterministic, sans-I/O coverage manifest and literal audit pins."""

from __future__ import annotations

import ast
import functools
import inspect
import json
import re
from pathlib import Path
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

_PROJECT_ROOT = Path(__file__).parents[2]


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
def test_raw_response_acknowledgment_reports_exact_escape_but_guards_descendant() -> None:
    class RawResponseResource:
        pass

    class RawResponseClient:
        @functools.cached_property
        def with_raw_response(self) -> RawResponseResource:
            return RawResponseResource()

    RawResponseClient.__module__ = "openai._client"

    exact = _by_identity(
        solwyn.coverage(
            _wrapper(
                RawResponseClient(),
                acknowledgments=frozenset({"with_raw_response"}),
            )
        )
    )[("with_raw_response", None)]
    descendant = _by_identity(
        solwyn.coverage(
            _wrapper(
                RawResponseClient(),
                acknowledgments=frozenset({"with_raw_response.responses.create"}),
            )
        )
    )[("with_raw_response", None)]

    assert exact.capability_scope == "raw_response"
    assert (exact.policy_action, exact.dispatch_action) == ("acknowledged", "return")
    assert (descendant.policy_action, descendant.dispatch_action) == (
        "acknowledged",
        "guard",
    )


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
def test_coverage_expectation_rejects_duplicate_rule_ids_within_category() -> None:
    entry = _audit_entry()

    with pytest.raises(
        ValueError,
        match="duplicate rule_id 'rule.operation' in untracked",
    ):
        CoverageExpectation(
            guarded_namespaces=(),
            tracked=(),
            untracked=(entry, entry),
            unknown=(),
            scoped_escapes=(),
            blocked=(),
            unsupported=(),
            conditional=(),
            safe=(),
        )


@pytest.mark.unit
def test_coverage_expectation_rejects_duplicate_rule_ids_across_categories() -> None:
    entry = _audit_entry()

    with pytest.raises(
        ValueError,
        match="rule_id 'rule.operation' appears in multiple categories: tracked, untracked",
    ):
        CoverageExpectation(
            guarded_namespaces=(),
            tracked=(entry,),
            untracked=(entry,),
            unknown=(),
            scoped_escapes=(),
            blocked=(),
            unsupported=(),
            conditional=(),
            safe=(),
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
def test_fingerprint_mismatch_names_the_category_size_for_triage() -> None:
    # Arrange
    report = _report_for_expectation(_audit_entry())
    good = report.fingerprint()
    tampered = good.model_copy(update={"untracked": "sha256:" + "0" * 64})

    # Act
    with pytest.raises(CoverageMismatchError) as exc_info:
        report.expect(tampered)

    # Assert
    message = str(exc_info.value)
    assert "untracked: fingerprint changed" in message
    assert "entries" in message
    assert "CoverageExpectation" in message


@pytest.mark.unit
def test_effective_actions_rejects_an_unsupported_coverage_kind() -> None:
    # Arrange
    import solwyn._coverage as coverage_module

    client = _wrapper(_OpenAIShape())

    # Act / Assert
    with pytest.raises(RuntimeError, match="unsupported coverage kind"):
        coverage_module._effective_actions(
            kind=object(),
            token="future.operation",
            surface="future.operation",
            return_shape="callable",
            capability_scope=None,
            client=client,
        )


@pytest.mark.unit
def test_real_openai_shape_matches_the_literal_exhaustive_audit_fingerprint() -> None:
    raw = openai.OpenAI(api_key="sk-test")
    try:
        report = solwyn.coverage(_wrapper(raw))
        readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        report.expect(_documented_openai_fingerprint(readme))
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


def _python_fences(markdown: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL))


def _assigns_name(statement: ast.stmt, name: str) -> bool:
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    else:
        return False
    return any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
        for target in targets
        for node in ast.walk(target)
    )


def _expression_method_call(
    statement: ast.stmt,
    *,
    owner: str,
    method: str,
) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == owner
        and call.func.attr == method
    ):
        return call
    return None


def _documented_openai_fingerprint(readme: str) -> CoverageFingerprint:
    snippet = next(
        fence
        for fence in _python_fences(readme)
        if "OPENAI_STRICT_FINGERPRINT = CoverageFingerprint(" in fence
    )
    module = ast.parse(snippet)
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "fingerprint" for node in ast.walk(module)
    ), "documented strict coverage must not call .fingerprint(); must not access .fingerprint"
    assignments = tuple(
        (index, statement)
        for index, statement in enumerate(module.body)
        if _assigns_name(statement, "OPENAI_STRICT_FINGERPRINT")
    )
    assert len(assignments) == 1, (
        "documented strict coverage requires exactly one top-level assignment "
        "to OPENAI_STRICT_FINGERPRINT"
    )
    assignment_index, assignment = assignments[0]
    assert isinstance(assignment, ast.Assign)
    expect_calls = tuple(
        (index, call)
        for index, statement in enumerate(module.body)
        if (call := _expression_method_call(statement, owner="report", method="expect")) is not None
    )
    assert len(expect_calls) == 1, (
        "documented strict coverage report.expect must be a top-level expression called once"
    )
    expectation_index, expectation = expect_calls[0]
    assert assignment_index < expectation_index, (
        "documented literal assignment must precede report.expect"
    )
    assert (
        len(expectation.args) == 1
        and not expectation.keywords
        and isinstance(expectation.args[0], ast.Name)
        and expectation.args[0].id == "OPENAI_STRICT_FINGERPRINT"
    ), "report.expect must pass OPENAI_STRICT_FINGERPRINT directly"
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "CoverageFingerprint"
    assert not assignment.value.args
    values = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in assignment.value.keywords
        if keyword.arg is not None
    }
    return CoverageFingerprint(**values)


def _latest_openai_version_from_fingerprint_manifest() -> str:
    manifest = json.loads(
        (_PROJECT_ROOT / "tests" / "provider_surface_fingerprints.json").read_text(encoding="utf-8")
    )
    latest_rows = tuple(
        row
        for row in manifest["fingerprints"]
        if row["provider"] == "openai"
        and row["variant"] == "native"
        and row["structural_interval"] == "latest"
    )
    assert len(latest_rows) == 2
    assert {row["mode"] for row in latest_rows} == {"sync", "async"}
    versions: list[str] = []
    for row in latest_rows:
        openai_distributions = tuple(
            distribution
            for distribution in row["distributions"]
            if distribution["name"] == "openai"
        )
        assert len(openai_distributions) == 1
        version = openai_distributions[0]["version"]
        assert isinstance(version, str) and version
        versions.append(version)
    assert len(set(versions)) == 1, "latest OpenAI sync/async fingerprints disagree on version"
    return versions[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("executable_lines", "before_literal", "expected_message"),
    [
        (
            "actual = report.fingerprint()\n"
            "# report.expect(OPENAI_STRICT_FINGERPRINT)\n"
            "report.expect(actual)",
            False,
            "must not call .fingerprint()",
        ),
        (
            "actual = OPENAI_STRICT_FINGERPRINT\n"
            "# report.expect(OPENAI_STRICT_FINGERPRINT)\n"
            "report.expect(actual)",
            False,
            "must pass OPENAI_STRICT_FINGERPRINT",
        ),
        (
            "make_fp = report.fingerprint\n"
            "OPENAI_STRICT_FINGERPRINT = make_fp()\n"
            "report.expect(OPENAI_STRICT_FINGERPRINT)",
            False,
            "must not access .fingerprint",
        ),
        (
            "if False:\n    report.expect(OPENAI_STRICT_FINGERPRINT)",
            False,
            "must be a top-level expression",
        ),
        (
            "OPENAI_STRICT_FINGERPRINT = replacement\nreport.expect(OPENAI_STRICT_FINGERPRINT)",
            False,
            "exactly one top-level assignment",
        ),
        (
            "report.expect(OPENAI_STRICT_FINGERPRINT)",
            True,
            "literal assignment must precede report.expect",
        ),
    ],
)
def test_documented_fingerprint_rejects_alias_and_comment_bypasses(
    executable_lines: str,
    before_literal: bool,
    expected_message: str,
) -> None:
    # Arrange
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    reviewed_fingerprint = _documented_openai_fingerprint(readme)
    fields = "\n".join(
        f"    {name}={value!r}," for name, value in reviewed_fingerprint.model_dump().items()
    )
    literal_lines = f"OPENAI_STRICT_FINGERPRINT = CoverageFingerprint(\n{fields}\n)"
    fence_lines = (
        (executable_lines, literal_lines) if before_literal else (literal_lines, executable_lines)
    )
    fence = "\n".join(fence_lines)
    markdown = f"```python\n{fence}\n```"

    # Act
    with pytest.raises(AssertionError) as caught:
        _documented_openai_fingerprint(markdown)

    # Assert
    assert expected_message in str(caught.value)


@pytest.mark.unit
def test_readme_openai_fingerprint_uses_literal_sha256_digests() -> None:
    # Arrange
    readme = (_PROJECT_ROOT / "README.md").read_text()

    # Act
    documented_fingerprint = _documented_openai_fingerprint(readme)

    # Assert
    assert all(
        re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        for digest in documented_fingerprint.model_dump().values()
    )


@pytest.mark.unit
def test_readme_is_the_only_authored_openai_fingerprint_literal() -> None:
    # Arrange
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    # Act
    assignments = tuple(
        statement
        for statement in module.body
        if _assigns_name(statement, "OPENAI_STRICT_FINGERPRINT")
    )

    # Assert
    assert assignments == ()


@pytest.mark.unit
def test_readme_openai_provenance_matches_the_latest_fingerprint_manifest() -> None:
    # Arrange
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    # Act
    version = _latest_openai_version_from_fingerprint_manifest()

    # Assert
    assert f"openai=={version}" in readme


@pytest.mark.unit
def test_readme_states_the_strict_coverage_and_trust_boundary_contract() -> None:
    # Arrange
    readme = (_PROJECT_ROOT / "README.md").read_text()
    required_claims = (
        '`on_unmetered="warn"` logs once and permits the call',
        '`on_unmetered="raise"` refuses the call before provider I/O',
        '`on_unmetered="allow"` permits the call without warning',
        "`SOLWYN_ON_UNMETERED=raise`",
        'acknowledge_untracked={"responses.create"}',
        '`SOLWYN_ACKNOWLEDGE_UNTRACKED="responses.create,audio.speech.create:gpt-4o-mini-tts"`',
        "Namespace tokens such as `responses` are invalid",
        "`audio.speech.create:gpt-4o-mini-tts`",
        "Coverage is computed locally and transmits nothing.",
        "Strict mode is not a sandbox.",
        "retaining the raw provider client",
        "accessing private wrapper state",
        "acknowledging a scoped raw escape",
        "response, page, stream, job, or operation object",
        "Native OpenAI video is tracked",
        "video on an OpenAI-compatible provider is unsupported",
        "Tested SDK version intervals",
        "unknown` and follow `on_unmetered`",
    )

    # Act
    normalized_readme = " ".join(readme.split())

    # Assert
    for claim in required_claims:
        assert claim in normalized_readme


@pytest.mark.unit
def test_maintainer_docs_define_python_contract_and_ci_artifact() -> None:
    # Arrange
    maintainer_docs = (_PROJECT_ROOT / "src" / "solwyn" / "CLAUDE.md").read_text()

    # Act
    normalized_docs = " ".join(maintainer_docs.split())

    # Assert
    assert "build/surface_contract/surface-classification.json" in normalized_docs
    assert "short-lived CI artifact" in normalized_docs
    assert "not committed" in normalized_docs
    assert "cross-SDK consumption remains deferred" in normalized_docs


@pytest.mark.unit
def test_unreleased_changelog_keeps_posture_and_escape_controls_together() -> None:
    # Arrange
    changelog = (_PROJECT_ROOT / "CHANGELOG.md").read_text()

    # Act
    unreleased = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]

    # Assert
    for contract_name in (
        "on_unmetered",
        "SOLWYN_ON_UNMETERED",
        "acknowledge_untracked",
        "SOLWYN_ACKNOWLEDGE_UNTRACKED",
        "UntrackedSpendSurfaceError",
        "coverage(client)",
    ):
        assert contract_name in unreleased
