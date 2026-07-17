"""E2E: base_url compat detection attributes calls to the right provider.

Detection is pure client-side inspection, but ATTRIBUTION is a wire contract:
the detected name rides check/confirm/metadata to the live API, which must
accept it. These tests pin both halves for the two local detection paths.
"""

from __future__ import annotations

import pytest
from conftest import WireRecorder, expected_compat_provider
from fake_provider import FakeProviderServer

MESSAGES = [{"role": "user", "content": "Say hello in five words."}]


@pytest.mark.integration
class TestCompatDetection:
    """Port-heuristic and generic-catch-all detection, end to end."""

    @pytest.mark.integration
    def test_generic_catch_all_detected_and_attributed(
        self, make_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client()
        assert client._adapter.name == "openai_compatible"

        recorder = WireRecorder().attach(client)
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        # The live API accepted a confirm attributed to the catch-all name.
        assert recorder.confirms[0]["provider"] == "openai_compatible"

    @pytest.mark.integration
    def test_conventional_port_detected_and_attributed(
        self, make_wrapped_client, fake_provider_known_port: FakeProviderServer
    ) -> None:
        expected = expected_compat_provider(fake_provider_known_port)
        client = make_wrapped_client(base_url=fake_provider_known_port.base_url)
        assert client._adapter.name == expected

        recorder = WireRecorder().attach(client)
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert fake_provider_known_port.request_count == 1
        assert recorder.confirms[0]["provider"] == expected
        assert recorder.events[0].provider.value == expected
