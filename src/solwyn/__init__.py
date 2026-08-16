"""Solwyn -- AI Agent Control Plane SDK.

Drop-in wrapper for ``openai.OpenAI``, ``anthropic.Anthropic``, and
``google.genai`` clients -- plus any OpenAI-compatible endpoint (xAI,
DeepSeek, Mistral, Qwen, Groq, Together, Fireworks, Perplexity, Azure
OpenAI, OpenRouter, Ollama, vLLM, LM Studio, ...) via ``base_url``
detection -- adding hard spending caps, automatic provider failover, and
per-agent cost attribution without ever seeing customer prompts.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("solwyn")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from solwyn._constants import TAG_KEY_MAX_LENGTH, TAG_VALUE_MAX_LENGTH, TAGS_MAX_KEYS
from solwyn._coverage import (
    CoverageAuditEntry,
    CoverageEntry,
    CoverageExpectation,
    CoverageFingerprint,
    CoverageReport,
    CoverageRuntime,
    coverage,
)
from solwyn._routing import (
    CostPolicy,
    HealthBasedPolicy,
    LatencyPolicy,
    ProviderCandidate,
    RoutingRequest,
    SelectionPolicy,
)
from solwyn._run import (
    RunContext,
    RunHandle,
    create_run,
    current_run,
    current_run_context,
    run,
    run_in_executor,
    start_run,
)
from solwyn._surface_graph import SurfaceInspectionError
from solwyn._types import CircuitState, FailoverReason, ProviderEntry, ProviderName
from solwyn.circuit_breaker import CircuitBreakerState
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.config import SolwynConfig
from solwyn.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    CoverageMismatchError,
    ProviderUnavailableError,
    RunStoppedError,
    SolwynError,
    SolwynTagsClampedWarning,
    UnsupportedSurfaceError,
    UntrackedSpendSurfaceError,
    UntranslatableModelError,
    UntranslatableRequestError,
)

__all__ = [
    "__version__",
    "Solwyn",
    "AsyncSolwyn",
    "SolwynConfig",
    "coverage",
    "CoverageRuntime",
    "CoverageEntry",
    "CoverageAuditEntry",
    "CoverageExpectation",
    "CoverageFingerprint",
    "CoverageReport",
    "SurfaceInspectionError",
    "SolwynError",
    "SolwynTagsClampedWarning",
    "BudgetExceededError",
    "RunStoppedError",
    "ProviderUnavailableError",
    "ConfigurationError",
    "CoverageMismatchError",
    "UntranslatableRequestError",
    "UntranslatableModelError",
    "UnsupportedSurfaceError",
    "UntrackedSpendSurfaceError",
    "ProviderEntry",
    "ProviderName",
    "FailoverReason",
    "RoutingRequest",
    "ProviderCandidate",
    "CircuitState",
    "CircuitBreakerState",
    # Selection policies (constructor selection_policy= arg).
    "SelectionPolicy",
    "HealthBasedPolicy",
    "LatencyPolicy",
    "CostPolicy",
    "run",
    "create_run",
    "start_run",
    "run_in_executor",
    "current_run",
    "current_run_context",
    "RunContext",
    "RunHandle",
    "TAGS_MAX_KEYS",
    "TAG_KEY_MAX_LENGTH",
    "TAG_VALUE_MAX_LENGTH",
]
