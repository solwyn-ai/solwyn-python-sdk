"""Solwyn SDK exceptions.

BudgetExceededError -- raised when budget enforcement blocks a request.
RunStoppedError -- raised when a dashboard stop blocks an agent run.
ProviderUnavailableError -- raised when all providers are circuit-broken.
ConfigurationError -- raised when configuration is invalid.
UntranslatableRequestError -- raised when a cross-provider hop cannot translate a request.
UntranslatableModelError -- raised when a model id is not configured for a target provider.
UnsupportedSurfaceError -- raised when an adapter does not serve a requested media surface.
UntrackedSpendSurfaceError -- raised when strict posture refuses an untracked capability.
CoverageMismatchError -- raised when a literal coverage audit pin drifts.
SolwynTagsClampedWarning -- emitted when merged tags exceed the event cap.
"""

from __future__ import annotations

from functools import partial


class SolwynError(Exception):
    """Base exception for all Solwyn SDK errors.

    Users can ``except solwyn.SolwynError:`` to catch any error
    produced by the SDK (budget, provider, configuration).
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.args!r})"


class SolwynTagsClampedWarning(UserWarning):
    """Warns that lower-priority tags were dropped from an event snapshot."""


class CoverageMismatchError(SolwynError):
    """Raised when an attached client's coverage differs from a literal pin."""

    def __init__(self, *, differences: tuple[str, ...]) -> None:
        if not differences:
            raise RuntimeError("coverage mismatch requires at least one difference")
        super().__init__("coverage expectation mismatch: " + "; ".join(differences))
        self.differences = differences


class BudgetExceededError(SolwynError):
    """Raised when Solwyn budget enforcement blocks a request.

    A Cloud denial raises in ``BudgetMode.HARD_DENY`` mode; in ``ALERT_ONLY``
    mode it logs a warning instead. When Cloud is unreachable and
    ``fail_open=False``, local fail-closed enforcement can also raise while
    retaining the configured mode, including ``ALERT_ONLY``.

    Attributes:
        project_id: Project identifier resolved by the API, if available.
        budget_limit: The configured spending cap (dollars).
        current_usage: Spending already consumed in the current period.
        estimated_cost: Estimated cost of the blocked request.
        budget_period: The Cloud denial label when supplied (for example,
            daily, weekly, monthly, agent_run, tag, or run_stopped); unknown
            for local enforcement or a Cloud response without a label.
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
            f"{type(self).__name__}("
            f"budget_limit={self.budget_limit!r}, "
            f"current_usage={self.current_usage!r})"
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...], dict[str, object]]:
        """Reconstruct keyword-only exception state across process boundaries."""
        return (
            partial(
                type(self),
                project_id=self.project_id,
                budget_limit=self.budget_limit,
                current_usage=self.current_usage,
                estimated_cost=self.estimated_cost,
                budget_period=self.budget_period,
                mode=self.mode,
            ),
            (),
            dict(self.__dict__),
        )


class RunStoppedError(BudgetExceededError):
    """Raised when an operator stopped an agent run from the dashboard.

    This is a ``BudgetExceededError`` subclass so existing hard-deny handlers
    keep catching it. It carries the same budget snapshot fields plus the
    stopped run id; no prompt or response content is retained.
    """

    def __init__(
        self,
        *,
        agent_run_id: str | None,
        project_id: str | None,
        budget_limit: float,
        current_usage: float,
        estimated_cost: float,
        mode: str,
    ) -> None:
        super().__init__(
            project_id=project_id,
            budget_limit=budget_limit,
            current_usage=current_usage,
            estimated_cost=estimated_cost,
            budget_period="run_stopped",
            mode=mode,
        )
        self.agent_run_id = agent_run_id
        self.args = (f"Run {agent_run_id} was stopped from the Solwyn dashboard",)

    def __reduce__(self) -> tuple[object, tuple[object, ...], dict[str, object]]:
        """Preserve the stopped run id when copying or pickling."""
        return (
            partial(
                type(self),
                agent_run_id=self.agent_run_id,
                project_id=self.project_id,
                budget_limit=self.budget_limit,
                current_usage=self.current_usage,
                estimated_cost=self.estimated_cost,
                mode=self.mode,
            ),
            (),
            dict(self.__dict__),
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
        model: The model identifier that has no mapping (e.g. ``"gpt-5.5"``).
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


class UntrackedSpendSurfaceError(SolwynError, AttributeError):
    """Raised before strict mode exposes an untracked provider capability.

    The ``AttributeError`` base preserves Python feature-probe behavior for
    ``hasattr`` and ``getattr(..., default)``. All attached fields are
    structural capability labels; the exception never carries request,
    prompt, response, or credential content.
    """

    def __init__(
        self,
        *,
        surface: str,
        token: str,
        provider: str,
        client_shape: str,
        kind: str,
        capability_scope: str | None,
        drifted_from_rule_id: str | None = None,
    ) -> None:
        scope = f" (scope: {capability_scope})" if capability_scope is not None else ""
        if kind == "unmetered_spend":
            guidance = f"acknowledge exact token '{token}'"
        else:
            guidance = (
                "review the provider graph and acknowledge an exact terminal capability token"
            )
        drift = (
            f" (reviewed rule {drifted_from_rule_id} no longer matches its shape)"
            if drifted_from_rule_id is not None
            else ""
        )
        super().__init__(
            f"Solwyn refused untracked surface '{surface}' for provider "
            f"{provider}{scope}; {guidance} or choose "
            f"on_unmetered='warn'/'allow'{drift}"
        )
        self.surface = surface
        self.token = token
        self.provider = provider
        self.client_shape = client_shape
        self.kind = kind
        self.capability_scope = capability_scope
        self.drifted_from_rule_id = drifted_from_rule_id

    def __repr__(self) -> str:
        drift = (
            f", drifted_from_rule_id={self.drifted_from_rule_id!r}"
            if self.drifted_from_rule_id is not None
            else ""
        )
        return (
            "UntrackedSpendSurfaceError("
            f"surface={self.surface!r}, provider={self.provider!r}, "
            f"kind={self.kind!r}{drift})"
        )


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
