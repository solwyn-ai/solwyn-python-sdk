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
from solwyn._routing import (
    CostPolicy,
    HealthBasedPolicy,
    LatencyPolicy,
    ProviderCandidate,
    RoutingRequest,
    SelectionPolicy,
)
from solwyn._run import RunContext, current_run, current_run_context, run, run_in_executor
from solwyn._types import CircuitState, FailoverReason, ProviderEntry, ProviderName
from solwyn.circuit_breaker import CircuitBreakerState
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.config import SolwynConfig
from solwyn.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    ProviderUnavailableError,
    SolwynError,
    SolwynTagsClampedWarning,
    UnsupportedSurfaceError,
    UntranslatableModelError,
    UntranslatableRequestError,
)

__all__ = [
    "__version__",
    "Solwyn",
    "AsyncSolwyn",
    "SolwynConfig",
    "SolwynError",
    "SolwynTagsClampedWarning",
    "BudgetExceededError",
    "ProviderUnavailableError",
    "ConfigurationError",
    "UntranslatableRequestError",
    "UntranslatableModelError",
    "UnsupportedSurfaceError",
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
    "run_in_executor",
    "current_run",
    "current_run_context",
    "RunContext",
    "TAGS_MAX_KEYS",
    "TAG_KEY_MAX_LENGTH",
    "TAG_VALUE_MAX_LENGTH",
]
