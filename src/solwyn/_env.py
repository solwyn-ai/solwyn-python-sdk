"""Shared environment-value coercion without configuration import cycles."""


def _parse_bool(value: str) -> bool:
    """Interpret the SDK's supported truthy environment values."""
    return value.lower() in ("true", "1", "yes")
