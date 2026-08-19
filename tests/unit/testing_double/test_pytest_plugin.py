"""Opt-in pytest fixtures and executable README recipes for the test double."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from solwyn.testing import FakeControlPlane

pytest_plugins = ["pytester"]

_README = Path(__file__).parents[3] / "README.md"
_CHANGELOG = Path(__file__).parents[3] / "CHANGELOG.md"
_README_SECTION = "## Testing your budget enforcement"
_SNIPPET_PATTERN = re.compile(
    r"<!-- test-double-snippet:(?P<name>[a-z-]+) -->\n"
    r"```python\n(?P<code>.*?)\n```",
    re.DOTALL,
)
_EXPECTED_SNIPPETS = {
    "deny-handler",
    "fail-open",
    "game-day",
    "operator-kill",
    "pytest-fixtures",
}


def _read_testing_section() -> str:
    readme = _README.read_text()
    _, separator, tail = readme.partition(_README_SECTION)
    assert separator, f"README is missing {_README_SECTION!r}"
    section, _, _ = tail.partition("\n## ")
    return section


def _read_testing_snippets() -> dict[str, str]:
    section = _read_testing_section()
    snippets = {
        match.group("name"): match.group("code") for match in _SNIPPET_PATTERN.finditer(section)
    }
    assert snippets.keys() == _EXPECTED_SNIPPETS
    assert section.count("```python") == len(snippets), (
        "every Python block in the testing guide must carry a test-double-snippet marker"
    )
    return snippets


@pytest.mark.unit
def test_plugin_is_inactive_without_explicit_registration(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_fixture_is_not_ambient(solwyn_control_plane):
            pass
        """
    )

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*fixture 'solwyn_control_plane' not found*"])


@pytest.mark.unit
def test_opt_in_exposes_fresh_shared_plane_and_denial_client(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        from solwyn import BudgetExceededError, Solwyn
        from solwyn.testing import FakeControlPlane

        pytest_plugins = ["solwyn.testing.pytest_plugin"]
        seen_planes = []


        def test_first_plane_is_recorded(solwyn_control_plane):
            assert isinstance(solwyn_control_plane, FakeControlPlane)
            solwyn_control_plane.deny_next()
            seen_planes.append(solwyn_control_plane)


        def test_second_test_gets_fresh_plane(solwyn_control_plane):
            assert solwyn_control_plane is not seen_planes[0]
            assert solwyn_control_plane.checks == []


        def test_client_uses_same_plane_and_denies_before_dispatch(
            solwyn_control_plane,
            solwyn_test_client,
        ):
            assert isinstance(solwyn_test_client, Solwyn)
            with pytest.raises(BudgetExceededError):
                solwyn_test_client.chat.completions.create(
                    model="solwyn-test/deny",
                    messages=[],
                )
            assert [check.model for check in solwyn_control_plane.checks] == [
                "solwyn-test/deny"
            ]
        """
    )

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(passed=3)
    result.stdout.no_fnmatch_line("*PytestCollectionWarning*FakeControlPlane*")


@pytest.mark.unit
def test_client_fixture_closes_after_a_failing_test(
    pytester: pytest.Pytester,
    tmp_path: Path,
) -> None:
    close_marker = tmp_path / "client-closed"
    pytester.makepyfile(
        f"""
        from pathlib import Path

        pytest_plugins = ["solwyn.testing.pytest_plugin"]


        def test_failure_still_runs_teardown(solwyn_test_client):
            # Wrapper attribute writes forward to the provider client, so the
            # close observation rides the provider-close seam the wrapper's own
            # close() forwards to at the end of its shutdown chain.
            inner = solwyn_test_client._solwyn_client

            def observed_close():
                Path({str(close_marker)!r}).write_text("closed")

            inner.close = observed_close
            raise RuntimeError("intentional failure")
        """
    )

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(failed=1)
    assert close_marker.read_text() == "closed"


@pytest.mark.unit
def test_client_fixture_closes_when_a_dependent_fixture_fails_during_setup(
    pytester: pytest.Pytester,
    tmp_path: Path,
) -> None:
    close_marker = tmp_path / "client-closed-after-setup-error"
    pytester.makepyfile(
        f"""
        from pathlib import Path

        import pytest

        pytest_plugins = ["solwyn.testing.pytest_plugin"]


        @pytest.fixture
        def failing_dependency(solwyn_test_client):
            # See test_failure_still_runs_teardown: the observation rides the
            # provider-close seam because wrapper writes forward to the provider.
            inner = solwyn_test_client._solwyn_client

            def observed_close():
                Path({str(close_marker)!r}).write_text("closed")

            inner.close = observed_close
            raise RuntimeError("intentional setup failure")


        def test_never_runs(failing_dependency):
            raise AssertionError("test body should not run")
        """
    )

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(errors=1)
    assert close_marker.read_text() == "closed"


@pytest.mark.unit
def test_unmatched_control_plane_request_fails_teardown(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import httpx

        pytest_plugins = ["solwyn.testing.pytest_plugin"]


        def test_hits_unknown_path(solwyn_control_plane):
            with httpx.Client(transport=solwyn_control_plane.transport) as client:
                client.post(
                    f"{solwyn_control_plane.api_url}/api/v1/not-a-real-endpoint",
                    json={},
                )
        """
    )

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*POST /api/v1/not-a-real-endpoint*"])


@pytest.mark.unit
def test_clearing_unmatched_requests_before_teardown_opts_out(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import httpx

        pytest_plugins = ["solwyn.testing.pytest_plugin"]


        def test_hits_unknown_path_and_clears(solwyn_control_plane):
            with httpx.Client(transport=solwyn_control_plane.transport) as client:
                client.post(
                    f"{solwyn_control_plane.api_url}/api/v1/not-a-real-endpoint",
                    json={},
                )
            solwyn_control_plane.unmatched_requests.clear()
        """
    )

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(passed=1)


@pytest.mark.unit
def test_readme_testing_guide_has_complete_boundary_and_magic_table() -> None:
    section = _read_testing_section()
    normalized = " ".join(section.split())

    assert (
        "The double never prices anything — the API owns pricing. "
        "Scripted denials test your handling, not your budget math."
    ) in normalized
    assert "forwards to the wrapped provider client's own close seam" in normalized
    assert "transport failure → endpoint refusal → verdict → allow" in normalized
    for magic_model in (
        "solwyn-test/deny",
        "solwyn-test/deny-alert",
        "solwyn-test/deny-tag",
        "solwyn-test/deny-stopped",
        "solwyn-test/runaway",
        "solwyn-test/kill",
        "solwyn-test/lease-ineligible",
    ):
        assert f"`{magic_model}`" in section


@pytest.mark.unit
def test_readme_magic_table_distinguishes_configured_and_forced_modes() -> None:
    rows = {
        cells[1].strip(" `"): cells[2].strip()
        for line in _read_testing_section().splitlines()
        if line.startswith("| `solwyn-test/")
        for cells in [line.split("|")]
    }

    for model in ("deny", "deny-tag"):
        description = rows[f"solwyn-test/{model}"].lower()
        assert "configured mode" in description
        assert "hard_deny" in description

    runaway_description = rows["solwyn-test/runaway"].lower()
    assert "configured mode" in runaway_description
    assert "hard_deny" in runaway_description

    for scripted in ("deny-stopped", "kill"):
        # An operator/dashboard stop is authoritative: it never softens to the
        # project's configured mode the way an ordinary budget verdict does.
        forced_description = rows[f"solwyn-test/{scripted}"].lower()
        assert "configured mode" not in forced_description
        assert "hard_deny" in forced_description

    for model in ("deny-stopped", "runaway", "kill"):
        assert "solwyn.run(" in rows[f"solwyn-test/{model}"].lower()

    assert "forces" in rows["solwyn-test/deny-alert"].lower()
    assert "alert_only" in rows["solwyn-test/deny-alert"].lower()


@pytest.mark.unit
def test_readme_provider_recipes_own_and_close_real_provider_clients() -> None:
    snippets = _read_testing_snippets()

    for name in ("deny-handler", "fail-open", "game-day"):
        recipe = snippets[name]
        assert "with OpenAI(" in recipe
        assert "as provider" in recipe
        assert "plane.wrap(provider" in recipe
        # The wrapper's close forwards to the provider's close seam; each
        # recipe pins that forwarding after the wrapper context exits.
        assert "forwards to the provider client's close seam" in recipe
        assert "assert provider.is_closed()" in recipe
        assert "assert not provider.is_closed()" not in recipe

    for name in ("fail-open", "game-day"):
        recipe = snippets[name]
        assert "import httpx2 as provider_httpx" in recipe
        assert "import httpx as provider_httpx" in recipe
        assert "provider_httpx.MockTransport" in recipe
        assert "http_client=provider_http_client" in recipe
        assert "respx" not in recipe


@pytest.mark.unit
def test_changelog_distinguishes_migrated_sdk_behavior_from_live_state() -> None:
    unreleased = _CHANGELOG.read_text().partition("## [Unreleased]")[2].partition("\n## [")[0]

    assert "SDK-behavior integration" in unreleased
    assert "zero-network CI lane" in unreleased
    assert "live integration retains" in unreleased
    assert "deployed server-state coverage" in unreleased


@pytest.mark.unit
@pytest.mark.parametrize("snippet_name", sorted(_EXPECTED_SNIPPETS))
def test_each_readme_recipe_runs_verbatim_in_an_isolated_test_module(
    pytester: pytest.Pytester,
    snippet_name: str,
) -> None:
    snippet = _read_testing_snippets()[snippet_name]
    pytester.makepyfile(snippet)

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(passed=1)


@pytest.mark.unit
def test_testing_package_docstring_states_the_three_boundaries() -> None:
    import solwyn.testing as testing

    sentences = [
        sentence.strip() for sentence in (testing.__doc__ or "").split(".") if sentence.strip()
    ]
    assert len(sentences) == 3
    assert "fakecontrolplane" in sentences[0].lower()
    assert "control-plane traffic" in sentences[0].lower()
    assert "without network i/o" in sentences[0].lower()
    assert "real wire shapes" in sentences[1].lower()
    assert "does not mock provider traffic" in sentences[1].lower()
    assert "no pricing" in sentences[2].lower()


@pytest.mark.unit
def test_all_public_fake_control_plane_methods_have_useful_docstrings() -> None:
    methods = {
        name: method
        for name, method in inspect.getmembers(FakeControlPlane, inspect.isfunction)
        if not name.startswith("_")
    }

    assert methods
    assert all(inspect.getdoc(method) for method in methods.values())
    assert "control_plane_failure_threshold" in inspect.getdoc(methods["outage"])
    assert "control_plane_recovery_timeout" in inspect.getdoc(methods["outage"])
    assert "budget_check_timeout" in inspect.getdoc(methods["slow"])
    assert "lease_enabled" in inspect.getdoc(methods["refuse_leases"])
