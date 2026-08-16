"""Reusable offline HTTP harness for real-framework smoke tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import respx
from openai import AsyncOpenAI, OpenAI

CONTROL_PLANE_URL = "https://control-plane.framework-smoke.test"
OPENAI_BASE_URL = "https://openai.framework-smoke.test/v1"
MODEL = "gpt-4.1-mini"
SOLWYN_API_KEY = "sk_proj_" + "a" * 64

_ALLOW_BUDGET_RESPONSE = {
    "allowed": True,
    "remaining_budget": 80.0,
    "reservation_id": "res_framework_smoke",
    "mode": "alert_only",
    "budget_limit": 100.0,
    "current_usage": 20.0,
    "project_id": "proj_" + "a" * 24,
}

_DENY_BUDGET_RESPONSE = {
    **_ALLOW_BUDGET_RESPONSE,
    "allowed": False,
    "remaining_budget": 0.0,
    "reservation_id": None,
    "mode": "hard_deny",
    "current_usage": 100.0,
    "denied_by_period": "monthly",
}


def _chat_completion(
    *, content: str = "This is a fake response.", response_id: str = "chatcmpl-framework-smoke"
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "total_tokens": 18,
        },
    }


def _tool_call_chat_completion(*, tool_name: str, response_id: str, call_id: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "total_tokens": 18,
        },
    }


def _chat_completion_stream(*, content: str = "This is a fake response.") -> str:
    def chunk(
        delta: dict[str, Any] | None,
        finish_reason: str | None,
        usage: dict[str, int] | None,
    ) -> dict[str, Any]:
        return {
            "id": "chatcmpl-framework-smoke",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MODEL,
            "choices": (
                [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                if delta is not None
                else []
            ),
            "usage": usage,
        }

    events = [
        chunk({"role": "assistant", "content": content}, None, None),
        chunk({}, "stop", None),
        chunk(None, None, {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}),
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


def _tool_call_chat_completion_stream(*, tool_name: str, response_id: str, call_id: str) -> str:
    events = [
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": None,
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        },
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


class FrameworkSmokeHarness:
    """Stub provider and Solwyn control-plane HTTP while retaining wire payloads.

    Tasks 6-7 can reuse this harness and add framework-specific routes without
    importing provider SDKs or reaching an external service.
    """

    def __init__(
        self,
        router: respx.MockRouter,
        *,
        streaming: bool = False,
        allow_budget: bool = True,
        handoff: bool = False,
        function_tool: bool = False,
        transient_failure: bool = False,
        model_calls: int = 1,
    ) -> None:
        if handoff and function_tool:
            raise ValueError("framework smoke supports one canned tool path at a time")
        if model_calls < 1:
            raise ValueError("framework smoke requires at least one model call")
        self.budget_checks: list[dict[str, Any]] = []
        self.confirms: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._allow_budget = allow_budget
        self._transient_failure_pending = transient_failure

        self.model_route: respx.Route | None = None
        if allow_budget:
            final_content = "The specialist resolved this request."
            if function_tool:
                final_content = "The worker completed both turns."
            elif not handoff:
                final_content = "This is a fake response."
            tool_name = "transfer_to_specialist" if handoff else "lookup_case"
            tool_response_id = (
                "chatcmpl-framework-handoff" if handoff else "chatcmpl-framework-function-tool"
            )
            tool_call_id = "call-framework-handoff" if handoff else "call-framework-function-tool"
            if streaming:
                provider_responses: list[httpx.Response] = []
                if handoff or function_tool:
                    provider_responses.append(
                        httpx.Response(
                            200,
                            text=_tool_call_chat_completion_stream(
                                tool_name=tool_name,
                                response_id=tool_response_id,
                                call_id=tool_call_id,
                            ),
                            headers={"content-type": "text/event-stream"},
                        )
                    )
                provider_responses.extend(
                    [
                        httpx.Response(
                            200,
                            text=_chat_completion_stream(content=final_content),
                            headers={"content-type": "text/event-stream"},
                        )
                        for _ in range(model_calls)
                    ]
                )
            else:
                provider_responses = []
                if handoff or function_tool:
                    provider_responses.append(
                        httpx.Response(
                            200,
                            json=_tool_call_chat_completion(
                                tool_name=tool_name,
                                response_id=tool_response_id,
                                call_id=tool_call_id,
                            ),
                        )
                    )
                provider_responses.extend(
                    [
                        httpx.Response(
                            200,
                            json=_chat_completion(
                                content=final_content,
                                response_id=(
                                    "chatcmpl-framework-specialist"
                                    if handoff
                                    else "chatcmpl-framework-smoke"
                                ),
                            ),
                        )
                        for _ in range(model_calls)
                    ]
                )
            response_iterator = iter(provider_responses)
            self.model_route = router.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
                side_effect=lambda _request: self._next_provider_response(response_iterator)
            )
        router.post(f"{CONTROL_PLANE_URL}/api/v1/budgets/check").mock(
            side_effect=self._check_budget
        )
        if allow_budget:
            router.post(f"{CONTROL_PLANE_URL}/api/v1/budgets/confirm").mock(
                side_effect=self._confirm
            )
        router.post(f"{CONTROL_PLANE_URL}/api/v1/metadata/ingest").mock(side_effect=self._ingest)

    @property
    def model_call_count(self) -> int:
        return self.model_route.call_count if self.model_route is not None else 0

    def _next_provider_response(
        self,
        responses: Iterator[httpx.Response],
    ) -> httpx.Response:
        if self._transient_failure_pending:
            self._transient_failure_pending = False
            raise ConnectionError("offline framework smoke transient provider failure")
        response = next(responses)
        if not isinstance(response, httpx.Response):
            raise TypeError("framework smoke provider response must be an httpx.Response")
        return response

    @staticmethod
    def _body(request: httpx.Request) -> Any:
        return json.loads(request.content)

    def _check_budget(self, request: httpx.Request) -> httpx.Response:
        body = self._body(request)
        if not isinstance(body, dict):
            raise TypeError("framework smoke budget check must be a JSON object")
        self.budget_checks.append(body)
        response = _ALLOW_BUDGET_RESPONSE if self._allow_budget else _DENY_BUDGET_RESPONSE
        return httpx.Response(200, json=response)

    def _confirm(self, request: httpx.Request) -> httpx.Response:
        body = self._body(request)
        if not isinstance(body, dict):
            raise TypeError("framework smoke confirm must be a JSON object")
        self.confirms.append(body)
        return httpx.Response(200, json={"status": "confirmed"})

    def _ingest(self, request: httpx.Request) -> httpx.Response:
        body = self._body(request)
        if not isinstance(body, list):
            raise TypeError("framework smoke ingest must be a JSON array")
        self.events.extend(body)
        return httpx.Response(202, json={"rejected": []})


def make_offline_openai_client(router: respx.MockRouter) -> AsyncOpenAI:
    """Build a real AsyncOpenAI client whose transport terminates at respx."""
    common = {
        "base_url": OPENAI_BASE_URL,
        "api_key": "sk-provider-test",
        # Runner-managed retry tests must not be hidden by provider-SDK retries.
        "max_retries": 0,
    }
    if _openai_major() >= 3:
        import httpx2

        return AsyncOpenAI(
            **common,
            http_client=httpx2.AsyncClient(transport=_respx_httpx2_transport(router)),
        )
    return AsyncOpenAI(
        **common,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(router.async_handler)),
    )


def make_offline_openai_sync_client(router: respx.MockRouter) -> OpenAI:
    """Build a real OpenAI client whose transport terminates at respx."""
    common = {
        "base_url": OPENAI_BASE_URL,
        "api_key": "sk-provider-test",
        "max_retries": 0,
    }
    if _openai_major() >= 3:
        import httpx2

        return OpenAI(
            **common,
            http_client=httpx2.Client(transport=_respx_httpx2_sync_transport(router)),
        )
    return OpenAI(
        **common,
        http_client=httpx.Client(transport=httpx.MockTransport(router.handler)),
    )


def _respx_httpx2_sync_transport(router: respx.MockRouter) -> Any:
    """Bridge OpenAI 3's sync httpx2 transport into the respx router."""
    import httpx2

    class RespxHttpx2Transport(httpx2.BaseTransport):
        def handle_request(self, request: httpx2.Request) -> httpx2.Response:
            content = request.read()
            routed_request = httpx.Request(
                request.method,
                str(request.url),
                headers=list(request.headers.multi_items()),
                content=content,
            )
            routed_response = router.handler(routed_request)
            routed_content = routed_response.read()
            return httpx2.Response(
                routed_response.status_code,
                headers=list(routed_response.headers.multi_items()),
                content=routed_content,
                request=request,
            )

    return RespxHttpx2Transport()


def _respx_httpx2_transport(router: respx.MockRouter) -> Any:
    """Bridge OpenAI 3's httpx2 transport into the shared respx router.

    OpenAI Agents 0.21 moved its provider dependency to OpenAI 3, which uses
    the separately named ``httpx2`` package. Solwyn's control plane still uses
    ``httpx``. Translating only at this test transport keeps both sides on the
    same respx route registry and prevents either client from reaching a real
    network. The import stays lazy so ordinary tests need no framework-only
    transitive dependencies.
    """
    import httpx2

    class RespxHttpx2Transport(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
            content = await request.aread()
            routed_request = httpx.Request(
                request.method,
                str(request.url),
                headers=list(request.headers.multi_items()),
                content=content,
            )
            routed_response = await router.async_handler(routed_request)
            routed_content = await routed_response.aread()
            return httpx2.Response(
                routed_response.status_code,
                headers=list(routed_response.headers.multi_items()),
                content=routed_content,
                request=request,
            )

    return RespxHttpx2Transport()


def _openai_major() -> int:
    from openai import __version__

    return int(__version__.split(".", 1)[0])
