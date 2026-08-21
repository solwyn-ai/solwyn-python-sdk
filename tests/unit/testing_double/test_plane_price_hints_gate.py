"""Price-hint opt-in behavior of the in-process control-plane double."""

from __future__ import annotations

import httpx
import pytest

from solwyn.testing import FakeControlPlane


@pytest.mark.unit
def test_price_hints_require_the_explicit_v1_opt_in() -> None:
    """A legacy check cannot receive hints intended for cost-aware routing."""
    plane = FakeControlPlane(price_hints={"openai": 2.0, "anthropic": 1.0})
    payload = {
        "estimated_input_tokens": 17,
        "model": "gpt-5.5",
        "provider": "openai",
        "fallback_providers": [],
        "fallback_models": [],
    }
    headers = {"Authorization": f"Bearer {plane.api_key}"}

    with httpx.Client(transport=plane.transport) as client:
        legacy = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=payload,
            headers=headers,
        )
        opted_in = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json={**payload, "price_hints_version": "1"},
            headers=headers,
        )

    assert legacy.status_code == 200
    assert legacy.json().get("price_hints") is None
    assert opted_in.status_code == 200
    assert opted_in.json()["price_hints"] == {"openai": 2.0, "anthropic": 1.0}
