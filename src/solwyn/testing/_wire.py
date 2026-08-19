"""Pydantic-backed parsing and serialization for the fake control plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

_ModelT = TypeVar("_ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class PlaneResponse:
    """Sans-I/O response emitted by the fake plane."""

    status_code: int
    body: BaseModel | dict[str, Any] | list[Any] | None = None
    headers: dict[str, str] | None = None
    exclude_none: bool = True
    model_only: bool = False


@dataclass(frozen=True, slots=True)
class PreparedPlaneRequest:
    """Frozen request effects captured before transport delay or failure."""

    delay_seconds: float
    outage: bool
    response: PlaneResponse | None


def parse_model(model: type[_ModelT], body: object) -> _ModelT | PlaneResponse:
    """Parse a request with the vendored wire model or return a 422 response."""
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        return PlaneResponse(
            422,
            {
                "detail": exc.errors(
                    include_url=False,
                    include_context=False,
                    include_input=False,
                )
            },
        )


def parse_model_list(
    model: type[_ModelT],
    body: object,
) -> list[_ModelT] | PlaneResponse:
    """Parse a bare JSON array atomically through one vendored wire model."""
    if not isinstance(body, list):
        return PlaneResponse(
            422,
            {
                "detail": [
                    {
                        "type": "list_type",
                        "loc": ["body"],
                        "msg": "Input should be a valid list",
                    }
                ]
            },
        )
    parsed: list[_ModelT] = []
    for index, item in enumerate(body):
        try:
            parsed.append(model.model_validate(item))
        except ValidationError as exc:
            errors = exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
            for error in errors:
                error["loc"] = (index, *error["loc"])
            return PlaneResponse(422, {"detail": errors})
    return parsed


def serialize_response(response: PlaneResponse) -> object | None:
    """Serialize model responses while failing loudly on handler drift."""
    if response.body is None:
        return None
    if response.model_only:
        if not isinstance(response.body, BaseModel):
            raise RuntimeError("solwyn.testing handler emitted a non-model response")
        return response.body.model_dump(mode="json", exclude_none=response.exclude_none)
    if isinstance(response.body, BaseModel):
        return response.body.model_dump(mode="json", exclude_none=response.exclude_none)
    return response.body
