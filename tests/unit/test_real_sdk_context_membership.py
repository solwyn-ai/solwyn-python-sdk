"""Real-SDK check: undeclared client/mode pairings are rejected at construction."""

from __future__ import annotations

import pytest
from conftest import VALID_API_KEY

from solwyn import AsyncSolwyn, Solwyn
from solwyn.exceptions import ConfigurationError

boto3 = pytest.importorskip("boto3")


def _bedrock_client() -> object:
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


@pytest.mark.unit
def test_async_wrapper_rejects_a_sync_boto3_bedrock_client(monkeypatch) -> None:
    # Arrange
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    client = _bedrock_client()

    # Act / Assert
    with pytest.raises(ConfigurationError) as exc_info:
        AsyncSolwyn(client, api_key=VALID_API_KEY)
    assert "aioboto3" in str(exc_info.value)
    assert exc_info.value.field == "client"

    # The matched pairing still constructs and releases owned resources.
    with Solwyn(_bedrock_client(), api_key=VALID_API_KEY) as wrapper:
        assert wrapper._surface_context.client_shape == "bedrock_boto3"
