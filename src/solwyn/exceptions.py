"""Solwyn SDK exceptions.

BudgetExceededError -- raised when hard-deny mode blocks a request.
ProviderUnavailableError -- raised when all providers are circuit-broken.
ConfigurationError -- raised when configuration is invalid.
UntranslatableRequestError -- raised when a cross-provider hop cannot translate a request.
UntranslatableModelError -- raised when a model id is not configured for a target provider.
UnsupportedSurfaceError -- raised when an adapter does not serve a requested media surface.
"""

from __future__ import annotations


class SolwynError(Exception):
    """Base exception for all Solwyn SDK errors.

    Users can ``except solwyn.SolwynError:`` to catch any error
    produced by the SDK (budget, provider, configuration).
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.args!r})"


class BudgetExceededError(SolwynError):
    """Raised when a request would exceed the configured budget limit.

    Only raised in ``BudgetMode.HARD_DENY`` mode.  In ``ALERT_ONLY`` mode
    the SDK logs a warning instead.

    Attributes:
        project_id: Project identifier resolved by the API, if available.
        budget_limit: The configured spending cap (dollars).
        current_usage: Spending already consumed in the current period.
        estimated_cost: Estimated cost of the blocked request.
        budget_period: The budget window (daily / weekly / monthly).
        mode: The active budget mode when the error was raised.
    """

    def __init__(
        self,
        *,
        project_id: str | None,
        budget_limit: float,
        current_usage: float,
        estimated_cost: float,
        budget_period: str,
        mode: str,
    ) -> None:
        project_label = project_id if project_id is not None else "unknown project"
        message = f"Budget exceeded for project {project_label}"
        super().__init__(message)
        self.project_id = project_id
        self.budget_limit = budget_limit
        self.current_usage = current_usage
        self.estimated_cost = estimated_cost
        self.budget_period = budget_period
        self.mode = mode

    def __repr__(self) -> str:
        return (
            f"BudgetExceededError("
            f"budget_limit={self.budget_limit!r}, "
            f"current_usage={self.current_usage!r})"
        )


class ProviderUnavailableError(SolwynError):
    """Raised when a provider's circuit breaker is open or the chain is exhausted.

    Attributes:
        provider: Name of the unavailable provider (e.g. ``"openai"``), or
            ``None`` when the error describes an exhausted chain rather than a
            single provider.
        circuit_state: Current circuit breaker state string, or ``None``.
        attempted: Ordered list of providers the dispatcher tried before giving
            up, or ``None`` when not applicable. Never contains prompt content.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        circuit_state: str | None = None,
        attempted: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.circuit_state = circuit_state
        self.attempted = attempted


class UntranslatableRequestError(SolwynError):
    """Raised when a cross-provider failover hop cannot translate a request.

    Raised before any network call, aborting the whole candidate chain. Carries
    only STRUCTURAL labels describing *what* could not be translated -- never the
    offending value and never any prompt content.

    Attributes:
        source: Provider the request was authored against (e.g. ``"openai"``).
        target: Provider the request was being translated toward (e.g. ``"anthropic"``).
        feature: Structural token naming the untranslatable shape (e.g.
            ``"response_format"``, ``"temperature>1.0"``, ``"anthropic.computer_use"``,
            ``"dangling_tool_call"``). NEVER the offending value or prompt content.
    """

    def __init__(self, *, source: str, target: str, feature: str) -> None:
        message = f"cannot translate {feature} from {source} to {target}"
        super().__init__(message)
        self.source = source
        self.target = target
        self.feature = feature

    def __repr__(self) -> str:
        return (
            f"UntranslatableRequestError("
            f"source={self.source!r}, target={self.target!r}, feature={self.feature!r})"
        )


class UntranslatableModelError(SolwynError):
    """Raised when a model id is not configured for the target provider.

    A model id is a configuration value, not prompt content, so it is safe to
    surface in the message.

    Attributes:
        model: The model identifier that has no mapping (e.g. ``"gpt-4o"``).
        provider: The target provider the model was not configured for.
    """

    def __init__(self, *, model: str, provider: str) -> None:
        message = f"model {model} is not configured for provider {provider}"
        super().__init__(message)
        self.model = model
        self.provider = provider

    def __repr__(self) -> str:
        return f"UntranslatableModelError(model={self.model!r}, provider={self.provider!r})"


class UnsupportedSurfaceError(SolwynError):
    """Raised when a provider adapter does not serve a requested media surface.

    Non-chat media surfaces (embeddings, images, audio, video) are dispatched
    through ``prepare_media_call``; an adapter with no branch for the requested
    surface raises this. Both identifiers are configuration values — a surface
    name and a provider name — never prompt content.

    Attributes:
        surface: The media surface that was requested (e.g. ``"embeddings"``).
        provider: The provider whose adapter does not serve it.
    """

    def __init__(self, *, surface: str, provider: str) -> None:
        super().__init__(f"provider {provider} does not support the {surface} surface")
        self.surface = surface
        self.provider = provider

    def __repr__(self) -> str:
        return f"UnsupportedSurfaceError(surface={self.surface!r}, provider={self.provider!r})"


class ConfigurationError(SolwynError):
    """Raised when SDK configuration is invalid or incomplete.

    Attributes:
        field: The configuration field that failed validation (may be ``None``).
        message: Human-readable description of the problem.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.message = message
