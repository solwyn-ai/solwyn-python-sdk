"""Shared SDK constants."""

from __future__ import annotations

SERVICE_TIER_MAX_LENGTH = 32
AGENT_RUN_ID_MAX_LENGTH = 255
AGENT_RUN_NAME_MAX_LENGTH = 255
# Bedrock model identity can be a full ARN (e.g. an application inference
# profile). The AWS Converse API documents a 2048-char bound for modelId;
# match it so a legitimate ARN can never fail wire-model validation.
MODEL_NAME_MAX_LENGTH = 2048
PROVIDER_REGION_MAX_LENGTH = 32
