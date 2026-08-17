"""Opt-in pytest fixtures and executable README recipes for the test double."""

from __future__ import annotations

import inspect
import re
import textwrap
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
_EXPECTED_SNIPPETS = {"deny-handler", "fail-open", "game-day", "pytest-fixtures"}


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

        pytest_plugins = ["solwyn.testing.pytest_plugin"]
        seen_planes = []


        def test_first_plane_is_recorded(solwyn_control_plane):
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
            original_close = solwyn_test_client.close

            def observed_close():
                Path({str(close_marker)!r}).write_text("closed")
                original_close()

            solwyn_test_client.close = observed_close
            raise RuntimeError("intentional failure")
        """
    )

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(failed=1)
    assert close_marker.read_text() == "closed"


@pytest.mark.unit
def test_readme_testing_guide_has_complete_boundary_and_magic_table() -> None:
    section = _read_testing_section()

    assert (
        "The double never prices anything — the API owns pricing. "
        "Scripted denials test your handling, not your budget math."
    ) in section
    assert "transport failure → endpoint refusal → verdict → allow" in section
    for magic_model in (
        "solwyn-test/deny",
        "solwyn-test/deny-alert",
        "solwyn-test/deny-tag",
        "solwyn-test/deny-stopped",
        "solwyn-test/runaway",
        "solwyn-test/lease-ineligible",
    ):
        assert f"`{magic_model}`" in section


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
    provider_prelude = """
        import sys
        from types import ModuleType, SimpleNamespace

        import httpx


        class _Completions:
            def __init__(self, base_url):
                self.base_url = base_url

            def create(self, **kwargs):
                if self.base_url is None:
                    raise AssertionError("provider dispatch reached the denial-only client")
                response = httpx.post(f"{self.base_url}/chat/completions", json=kwargs)
                response.raise_for_status()
                payload = response.json()
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=payload["choices"][0]["message"]["content"]
                            )
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
                )


        class _Chat:
            def __init__(self, base_url):
                self.completions = _Completions(base_url)


        class OpenAI:
            def __init__(self, *, api_key, base_url=None):
                self.chat = _Chat(base_url)

            def with_options(self, **kwargs):
                return self


        OpenAI.__module__ = "openai._client"
        openai_module = ModuleType("openai")
        openai_module.OpenAI = OpenAI
        sys.modules["openai"] = openai_module
    """
    pytester.makepyfile(textwrap.dedent(provider_prelude) + "\n\n" + snippet)

    result = pytester.runpytest("-q", "-p", "no:asyncio")

    result.assert_outcomes(passed=1)


@pytest.mark.unit
def test_testing_package_docstring_states_the_three_boundaries() -> None:
    import solwyn.testing as testing

    sentences = [
        sentence.strip() for sentence in (testing.__doc__ or "").split(".") if sentence.strip()
    ]
    assert len(sentences) == 3
    assert "zero network" in sentences[0].lower()
    assert "real wire shapes" in sentences[1].lower()
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
