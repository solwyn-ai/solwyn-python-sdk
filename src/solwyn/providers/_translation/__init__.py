"""Cross-provider request/response translation.

================================ PRIVACY-CRITICAL ==============================
This is one of EXACTLY TWO content-privileged modules (the other is
``solwyn/_privacy.py``). It is the only place in the SDK that reshapes customer
prompt CONTENT. The content-touching contract is non-negotiable:

  * In-memory transform ONLY. Content is never stored on a long-lived object;
    it lives only inside the call frame and the returned kwargs/response that go
    STRAIGHT to the destination provider SDK.
  * No I/O. This module imports neither the stdlib log module nor any HTTP
    client, and holds no log handle of any kind. It must never reach
    ``_solwyn_reporter``, ``_solwyn_budget``, or any client pointed at
    ``config.api_url``.
  * Fail loudly with ``UntranslatableRequestError(source, target, feature)``
    carrying STRUCTURAL labels only — NEVER the offending value or any prompt
    content. Re-raise across the boundary with ``... from None`` so a provider
    exception's text can never leak via ``__cause__``/``__context__``.
  * Pure. Same input -> same output; no global state.

If you change this file you must keep the firewall tests
(``tests/unit/test_privacy_firewall.py`` / ``test_translation.py``) green.
===============================================================================

Public surface (keyed by provider name ``openai`` | ``anthropic`` | ``google``):

  to_canonical(provider, kwargs)        -> CanonicalRequest
  from_canonical(provider, canonical, *, model) -> dict  (native target kwargs)
  normalize_response(*, served, requested, response) -> object (duck-typed)

The canonical form is intentionally tiny: ~10 request fields, ~5
response fields. Anything outside the subset RAISES on the first cross-provider
hop, BEFORE any network call. Usage extraction is NOT done here — the existing
extraction adapters own that.
"""

from __future__ import annotations

from ._api import (
    from_canonical as from_canonical,
)
from ._api import (
    normalize_response as normalize_response,
)
from ._api import (
    to_canonical as to_canonical,
)
from ._api import (
    translate_stream_chunk as translate_stream_chunk,
)
from ._common import (
    normalize_finish_reason as normalize_finish_reason,
)
from ._guardrails import (
    fail_cross_provider_tool_stream as fail_cross_provider_tool_stream,
)
from ._models import (
    CanonicalMessage as CanonicalMessage,
)
from ._models import (
    CanonicalRequest as CanonicalRequest,
)
from ._models import (
    CanonicalResponse as CanonicalResponse,
)
from ._models import (
    CanonicalTool as CanonicalTool,
)
from ._models import (
    TextPart as TextPart,
)
from ._models import (
    ToolChoice as ToolChoice,
)
from ._models import (
    ToolResultPart as ToolResultPart,
)
from ._models import (
    ToolUsePart as ToolUsePart,
)

__all__ = [
    "CanonicalMessage",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalTool",
    "ToolChoice",
    "fail_cross_provider_tool_stream",
    "from_canonical",
    "normalize_response",
    "normalize_finish_reason",
    "to_canonical",
    "translate_stream_chunk",
]
