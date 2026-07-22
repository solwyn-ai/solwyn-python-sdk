"""Shared SDK constants."""

from __future__ import annotations

SERVICE_TIER_MAX_LENGTH = 32
AGENT_RUN_ID_MAX_LENGTH = 256
AGENT_RUN_NAME_MAX_LENGTH = 255
# Bedrock model identity can be a full ARN (e.g. an application inference
# profile). The AWS Converse API documents a 2048-char bound for modelId;
# match it so a legitimate ARN can never fail wire-model validation.
MODEL_NAME_MAX_LENGTH = 2048
PROVIDER_REGION_MAX_LENGTH = 32
# Lease holder identity on the wire is the SDK instance id (a uuid4, 36 chars).
HOLDER_ID_MAX_LENGTH = 64
# Server-issued lease id, echoed back on every renew/surrender. Bounded
# lock-step with core's lease schemas.
LEASE_ID_MAX_LENGTH = 64
# Canonical lowercase RFC 4122 text form — the exact shape str(uuid.uuid4())
# emits, which is the only shape the SDK has ever produced for a call_id. The
# API pins the same regex (shared CALL_ID_PATTERN) because call_id is durable
# spend identity: the cost-event ledger dedups on it. Mirrored here so a
# drifted id fails where it was built rather than as a 422 that loses the
# settlement it was carrying.
CALL_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
# Length of that canonical form; bounds the API's DB column and dedup key.
CALL_ID_MAX_LENGTH = 36
TAGS_MAX_KEYS = 10
TAG_KEY_MAX_LENGTH = 64
TAG_VALUE_MAX_LENGTH = 256
