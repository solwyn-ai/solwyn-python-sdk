"""API key and project selector format validation.

The key validator NFC-normalizes before matching (no homographs) and the
pattern admits ASCII lowercase hex only, which also rules out path traversal.

Both patterns are vendored lock-step with core's ``solwyn_shared.validation``
(``API_KEY_PATTERN``, ``PROJECT_SELECTOR_PATTERN``); core's contract parity
test compares them.
"""

import re
import unicodedata
from typing import Final

API_KEY_PATTERN: Final = re.compile(r"^solwyn_sk_[a-f0-9]{64}$")

# A project selector is either a ``proj_`` id or a project name: 1 to 100
# ASCII characters from ``A-Z a-z 0-9 space . _ -`` with no leading or trailing
# space.
PROJECT_SELECTOR_PATTERN: Final = re.compile(
    r"^(?:proj_[a-f0-9]{24}|[A-Za-z0-9._-](?:[A-Za-z0-9 ._-]{0,98}[A-Za-z0-9._-])?)$"
)

# The retired project-key prefix. Rejected locally with a message that says
# what to do instead of a generic format error.
_OLD_PROJECT_KEY_PREFIX: Final = "sk_proj_"

OLD_PROJECT_KEY_MESSAGE: Final = (
    "This is an old Solwyn project key (sk_proj_). Create a new key in the "
    "dashboard or with solwyn keys create."
)

INVALID_PROJECT_MESSAGE: Final = (
    "Invalid project: use a project name (letters, digits, spaces, '.', '_' or '-', "
    "up to 100 characters) or a proj_ id."
)


def validate_api_key_format(api_key: str) -> str:
    """Validate and return a Solwyn API key (format only, not authentication).

    The key is NFC-normalized first, so a homograph cannot slip through; the
    pattern itself admits only ASCII lowercase hex after the prefix, which
    also excludes path traversal characters.
    """
    if api_key.startswith(_OLD_PROJECT_KEY_PREFIX):
        raise ValueError(OLD_PROJECT_KEY_MESSAGE)

    normalized = unicodedata.normalize("NFC", api_key)
    if not normalized.isascii() or not API_KEY_PATTERN.fullmatch(normalized):
        display = normalized[:12] if len(normalized) > 12 else "<too short>"
        raise ValueError(
            "Invalid API key format: expected 'solwyn_sk_' followed by 64 lowercase "
            f"hex characters. Got: {display}"
        )
    return normalized


def validate_project_selector(selector: str) -> str:
    """Trim and validate a project name or ``proj_`` id; return the trimmed value."""
    trimmed = selector.strip(" \t")
    if not PROJECT_SELECTOR_PATTERN.fullmatch(trimmed):
        raise ValueError(INVALID_PROJECT_MESSAGE)
    return trimmed
