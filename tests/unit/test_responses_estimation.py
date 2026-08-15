"""Unit tests for Responses API request sizing."""

import pytest

from solwyn._base import _responses_output_bound
from solwyn._privacy import estimate_responses_content_length


@pytest.mark.unit
class TestResponsesContentLengthEstimation:
    def test_counts_string_input(self) -> None:
        assert estimate_responses_content_length({"input": "hello world"}) == 11

    def test_counts_instructions_and_string_input(self) -> None:
        kwargs = {"input": "hi", "instructions": "be brief"}

        assert estimate_responses_content_length(kwargs) == 10

    def test_counts_message_item_string_content(self) -> None:
        kwargs = {
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": "world"},
            ]
        }

        assert estimate_responses_content_length(kwargs) == 10

    def test_counts_only_text_strings_in_message_content_parts(self) -> None:
        kwargs = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe this"},
                        {"type": "input_image", "image_url": "data:image/png;base64,secret"},
                        {"type": "input_text", "text": 42},
                        "raw text is not a content part",
                    ],
                }
            ]
        }

        assert estimate_responses_content_length(kwargs) == 13

    def test_counts_function_call_output_string(self) -> None:
        kwargs = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "tool result",
                }
            ]
        }

        assert estimate_responses_content_length(kwargs) == 11

    def test_counts_only_text_in_function_call_output_parts(self) -> None:
        kwargs = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": [
                        {"type": "input_text", "text": "hello"},
                        {"type": "input_image", "image_url": "data:image/png;base64,secret"},
                        {"type": "input_file", "file_id": "file_123"},
                    ],
                }
            ]
        }

        assert estimate_responses_content_length(kwargs) == 5

    def test_unrecognized_content_returns_zero(self) -> None:
        assert estimate_responses_content_length({"metadata": {"tenant": "acme"}}) == 0

    @pytest.mark.parametrize(
        "input_value",
        [
            [42, None, "x"],
            {"type": "message", "content": "not in a list"},
        ],
    )
    def test_garbage_input_shapes_return_zero(self, input_value: object) -> None:
        assert estimate_responses_content_length({"input": input_value}) == 0


@pytest.mark.unit
class TestResponsesOutputBound:
    def test_positive_integer_max_output_tokens_wins(self) -> None:
        assert _responses_output_bound({"max_output_tokens": 321}, 1_000) == 321

    def test_missing_cap_uses_default(self) -> None:
        assert _responses_output_bound({}, 1_000) == 1_000

    @pytest.mark.parametrize("cap", [True, -1, "321"])
    def test_invalid_cap_uses_default(self, cap: object) -> None:
        assert _responses_output_bound({"max_output_tokens": cap}, 1_000) == 1_000
