"""Tests for the API key and project selector format validators."""

from __future__ import annotations

import pytest

import solwyn._validation as validation
from solwyn._validation import (
    API_KEY_PATTERN,
    PROJECT_SELECTOR_PATTERN,
    validate_api_key_format,
    validate_project_selector,
)

_BAD_FORMAT = (
    "Invalid API key format: expected 'solwyn_sk_' followed by 64 lowercase hex characters. Got: "
)


@pytest.mark.unit
def test_validate_api_key_format_accepts_solwyn_sk_key() -> None:
    key = "solwyn_sk_" + "0123456789abcdef" * 4

    assert validate_api_key_format(key) == key


@pytest.mark.unit
def test_api_key_pattern_is_the_core_contract() -> None:
    # Vendored lock-step with core's solwyn_shared.validation.API_KEY_PATTERN.
    assert API_KEY_PATTERN.pattern == r"^solwyn_sk_[a-f0-9]{64}$"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "shown"),
    [
        ("solwyn_sk_" + "A" * 64, "solwyn_sk_AA"),
        ("solwyn_sk_" + "a" * 63, "solwyn_sk_aa"),
        ("solwyn_sk_" + "a" * 65, "solwyn_sk_aa"),
        ("solwyn_sk_" + "a" * 64 + "\n", "solwyn_sk_aa"),
        ("sk_solwyn_" + "a" * 64, "sk_solwyn_aa"),
        ("solwyn_sk_" + "é" * 64, "solwyn_sk_éé"),
        ("short", "<too short>"),
        ("", "<too short>"),
    ],
)
def test_validate_api_key_format_rejects_with_exact_message(key: str, shown: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_api_key_format(key)

    assert str(excinfo.value) == _BAD_FORMAT + shown


@pytest.mark.unit
def test_old_project_keys_are_refused_with_their_own_message() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_api_key_format("sk_proj_" + "a" * 64)

    assert str(excinfo.value) == (
        "This is an old Solwyn project key (sk_proj_). Create a new key in the "
        "dashboard or with solwyn keys create."
    )


@pytest.mark.unit
def test_old_project_key_names_are_gone() -> None:
    assert not hasattr(validation, "PROJECT_KEY_PATTERN")
    assert not hasattr(validation, "validate_project_key_format")


@pytest.mark.unit
def test_project_selector_pattern_is_the_core_contract() -> None:
    # Vendored lock-step with core's solwyn_shared.validation.PROJECT_SELECTOR_PATTERN.
    assert PROJECT_SELECTOR_PATTERN.pattern == (
        r"^(?:proj_[a-f0-9]{24}|[A-Za-z0-9._-](?:[A-Za-z0-9 ._-]{0,98}[A-Za-z0-9._-])?)$"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("selector", "trimmed"),
    [
        ("checkout-bot", "checkout-bot"),
        ("Checkout Bot 2", "Checkout Bot 2"),
        ("a", "a"),
        ("x" * 100, "x" * 100),
        ("proj_" + "0" * 24, "proj_" + "0" * 24),
        ("  checkout-bot\t", "checkout-bot"),
        ("v1.2_test", "v1.2_test"),
    ],
)
def test_validate_project_selector_accepts_names_and_ids(selector: str, trimmed: str) -> None:
    assert validate_project_selector(selector) == trimmed


@pytest.mark.unit
@pytest.mark.parametrize(
    "selector",
    [
        "",
        "   ",
        "x" * 101,
        "checkout/bot",
        "checkout bot!",
        "café",
        "a\nb",
        "line\n",
    ],
)
def test_validate_project_selector_rejects_with_exact_message(selector: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_project_selector(selector)

    assert str(excinfo.value) == (
        "Invalid project: use a project name (letters, digits, spaces, '.', '_' or '-', "
        "up to 100 characters) or a proj_ id."
    )
