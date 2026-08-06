"""Tests for public credential format validators."""

from __future__ import annotations

import pytest

import solwyn._validation as validation
from solwyn._validation import validate_project_key_format


@pytest.mark.unit
def test_validate_project_key_format_accepts_project_key() -> None:
    key = "sk_proj_" + "a" * 64

    assert validate_project_key_format(key) == key


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "sk_solwyn_" + "a" * 64,
        "sk_proj_" + "A" * 64,
        "sk_proj_" + "a" * 63,
        "sk_proj_" + "a" * 65,
    ],
)
def test_validate_project_key_format_rejects_non_project_key_formats(key: str) -> None:
    with pytest.raises(ValueError, match="sk_proj_"):
        validate_project_key_format(key)


@pytest.mark.unit
def test_legacy_validate_api_key_format_symbol_is_removed() -> None:
    assert not hasattr(validation, "validate_api_key_format")


@pytest.mark.unit
def test_dead_project_id_validator_is_absent_while_project_key_validator_remains() -> None:
    assert not hasattr(validation, "validate_project_id")
    assert callable(validation.validate_project_key_format)
