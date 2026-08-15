"""A minimal local OpenAI-compatible provider server for E2E tests.

Test infrastructure ONLY — never imported by src/. Serves POST
``/v1/chat/completions`` and ``/v1/responses`` in both JSON and SSE-streaming
form, always returning a valid usage block (configurable token counts), and
records every request so tests can assert the provider was (or was NOT) called
and with which payload.

Why this doubles as detection coverage: Solwyn detects OpenAI-compatible
providers from the client's base_url — conventional local ports (1234/11434/
8000) map to lmstudio/ollama/vllm, any other local port lands in the
``openai_compatible`` generic catch-all. Binding this server to an ephemeral
vs. conventional port therefore exercises both detection paths with zero
external dependencies.

Extension seams for the failover session:
- ``fail_next(status, count=N, retry_after=...)`` queues N error responses
  before recovery, optionally carrying a ``Retry-After`` header.
- Instantiate TWO servers and wire them as primary + fallback clients.
- ``set_omit_usage(True)`` strips ``usage`` from JSON completions and the
  terminal usage chunk from SSE streams, for exercising the length-based
  estimation fallback.
- ``drop_next_stream(after_chunks=N)`` truncates the NEXT stream after N SSE
  events, closing the connection without the chunked terminator — see
  ``_send_stream`` for why this requires HTTP/1.1.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Conventional local ports Solwyn's compat detection maps to named providers
# (see src/solwyn/providers/openai_compatible.py COMPAT_PROFILES).
PORT_PROFILES: dict[int, str] = {1234: "lmstudio", 11434: "ollama", 8000: "vllm"}

RESPONSE_CONTENT = "This is a fake response."
STRUCTURED_RESPONSE_CONTENT = {"answer": "structured fake response", "score": 7}


@dataclass
class RecordedRequest:
    """One captured provider request (test-authored content only)."""

    path: str
    body: dict[str, Any]


@dataclass
class _ServerState:
    """Mutable state shared between the server object and its handler threads."""

    prompt_tokens: int
    completion_tokens: int
    requests: list[RecordedRequest] = field(default_factory=list)
    fail_statuses: list[tuple[int, str | None]] = field(default_factory=list)
    omit_usage: bool = False
    drop_stream_after: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class _BaseHandler(BaseHTTPRequestHandler):
    """Shared plumbing for both dialect handlers: body recording, JSON replies,
    and queued-failure injection. Dialect-specific routing/payload shaping
    (``do_POST``, completion/usage shaping, streaming) lives on subclasses."""

    state: _ServerState  # injected by FakeProviderServer via subclassing
    protocol_version = "HTTP/1.1"  # chunked framing + keep-alive for SSE truncation detection

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request stderr noise."""

    def _read_and_record(self) -> dict[str, Any]:
        """Read + parse the request body, record it, and return the parsed body."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        with self.state.lock:
            self.state.requests.append(RecordedRequest(path=self.path, body=body))
        return body

    def _maybe_send_queued_failure(self) -> bool:
        """Pop and send the next queued failure, if any. Returns True if sent."""
        with self.state.lock:
            entry = self.state.fail_statuses.pop(0) if self.state.fail_statuses else None
        if entry is None:
            return False
        status, retry_after = entry
        headers = {"Retry-After": retry_after} if retry_after is not None else None
        self._send_json(
            status,
            {"error": {"message": "injected failure", "type": "fake_error"}},
            extra_headers=headers,
        )
        return True

    def _send_json(
        self, status: int, payload: dict[str, Any], *, extra_headers: dict[str, str] | None = None
    ) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)


class _Handler(_BaseHandler):
    """OpenAI chat-completions and Responses dialect."""

    # do_POST is the BaseHTTPRequestHandler dispatch contract (name is fixed).
    def do_POST(self) -> None:
        body = self._read_and_record()

        if self.path.endswith("/responses"):
            if self._maybe_send_queued_failure():
                return
            if body.get("stream"):
                self._send_responses_stream(body)
            else:
                self._send_json(200, self._response(body))
            return
        if self.path.endswith("/chat/completions"):
            if self._maybe_send_queued_failure():
                return
            if body.get("stream"):
                self._send_stream(body)
            else:
                self._send_json(200, self._completion(body))
            return
        # Unrelated routes must not silently consume a fail_next() entry.
        self._send_json(404, {"error": {"message": f"no route: {self.path}"}})

    def _usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.state.prompt_tokens,
            "completion_tokens": self.state.completion_tokens,
            "total_tokens": self.state.prompt_tokens + self.state.completion_tokens,
        }

    def _completion(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "id": "chatcmpl-fake-0001",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "fake-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": RESPONSE_CONTENT},
                    "finish_reason": "stop",
                }
            ],
            "usage": self._usage(),
        }
        if self.state.omit_usage:
            del payload["usage"]
        return payload

    def _responses_usage(self) -> dict[str, object]:
        return {
            "input_tokens": self.state.prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": self.state.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": self.state.prompt_tokens + self.state.completion_tokens,
        }

    def _response_output_text(self, body: dict[str, Any]) -> str:
        text = body.get("text")
        response_format = text.get("format") if isinstance(text, dict) else None
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            return json.dumps(STRUCTURED_RESPONSE_CONTENT, separators=(",", ":"))
        return RESPONSE_CONTENT

    def _response(
        self,
        body: dict[str, Any],
        *,
        completed: bool = True,
        output_text: str | None = None,
    ) -> dict[str, Any]:
        if completed and output_text is None:
            output_text = self._response_output_text(body)
        response: dict[str, Any] = {
            "id": "resp_fake_0001",
            "object": "response",
            "created_at": 1700000000.0,
            "status": "completed" if completed else "in_progress",
            "model": body.get("model", "fake-model"),
            "output": (
                [
                    {
                        "id": "msg_fake_0001",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": output_text,
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    }
                ]
                if completed
                else []
            ),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "service_tier": body.get("service_tier", "default"),
            "usage": self._responses_usage() if completed else None,
        }
        if completed and self.state.omit_usage:
            del response["usage"]
        return response

    def _send_responses_stream(self, body: dict[str, Any]) -> None:
        """SSE stream ending in a full ``response.completed`` usage event."""
        with self.state.lock:
            drop_after = self.state.drop_stream_after
            self.state.drop_stream_after = None
        output_text = self._response_output_text(body)
        message_item = {
            "id": "msg_fake_0001",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        empty_text_part = {
            "type": "output_text",
            "text": "",
            "annotations": [],
            "logprobs": [],
        }

        events = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": self._response(body, completed=False),
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": message_item,
            },
            {
                "type": "response.content_part.added",
                "sequence_number": 2,
                "item_id": "msg_fake_0001",
                "output_index": 0,
                "content_index": 0,
                "part": empty_text_part,
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 3,
                "item_id": "msg_fake_0001",
                "output_index": 0,
                "content_index": 0,
                "delta": output_text,
                "logprobs": [],
            },
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": self._response(body, output_text=output_text),
            },
        ]
        self._send_sse(events, drop_after=drop_after)

    def _send_sse(self, events: list[dict[str, Any]], *, drop_after: int | None) -> None:
        """Send events with explicit chunked framing so truncation is detectable."""

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_chunk(data: bytes) -> None:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

        for sent, event in enumerate(events):
            if drop_after is not None and sent >= drop_after:
                # Close without the chunked terminator: the client's HTTP layer
                # sees an incomplete body and raises RemoteProtocolError.
                self.close_connection = True
                return
            write_chunk(f"data: {json.dumps(event)}\n\n".encode())
        write_chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")

    def _send_stream(self, body: dict[str, Any]) -> None:
        """SSE stream: two content chunks, a stop chunk, a usage-only final chunk.

        The final usage-bearing chunk mirrors real OpenAI-dialect behavior with
        ``stream_options.include_usage`` AND the always-on final-chunk usage many
        compat providers emit — so both the include_usage-injected (lmstudio)
        and never-inject (generic catch-all) adapter policies extract real usage
        (tier 1), never the estimation fallback. ``set_omit_usage(True)`` drops
        that final chunk entirely, forcing the length-based estimation fallback.

        Shared chunked framing makes a ``drop_next_stream()`` truncation
        distinguishable from normal end-of-stream.
        """
        model = body.get("model", "fake-model")
        with self.state.lock:
            drop_after = self.state.drop_stream_after
            self.state.drop_stream_after = None  # one-shot
            omit_usage = self.state.omit_usage

        def chunk(delta: dict[str, Any] | None, finish: str | None, usage: dict[str, int] | None):
            return {
                "id": "chatcmpl-fake-0001",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
                if delta is not None
                else [],
                "usage": usage,
            }

        events = [
            chunk({"role": "assistant", "content": "This is "}, None, None),
            chunk({"content": "a fake response."}, None, None),
            chunk({}, "stop", None),
        ]
        if not omit_usage:
            events.append(chunk(None, None, self._usage()))
        self._send_sse(events, drop_after=drop_after)


class _AnthropicHandler(_BaseHandler):
    """Anthropic Messages dialect — JSON only, enough for ONE translated hop.

    Streaming deliberately returns 501: the cross-dialect E2E scope is a single
    non-streaming hop (building an SSE event-typed Messages stream would be a
    second protocol emulator).
    """

    def do_POST(self) -> None:
        body = self._read_and_record()
        if not self.path.endswith("/v1/messages"):
            self._send_json(404, {"error": {"type": "not_found_error", "message": self.path}})
            return
        if self._maybe_send_queued_failure():
            return
        if body.get("stream"):
            self._send_json(
                501, {"error": {"type": "api_error", "message": "fake server: no streaming"}}
            )
            return
        self._send_json(
            200,
            {
                "id": "msg_fake_0001",
                "type": "message",
                "role": "assistant",
                "model": body.get("model", "fake-model"),
                "content": [{"type": "text", "text": RESPONSE_CONTENT}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.state.prompt_tokens,
                    "output_tokens": self.state.completion_tokens,
                },
            },
        )


class FakeProviderServer:
    """A fake OpenAI-compatible provider on 127.0.0.1.

    ``port=0`` binds an ephemeral port (detected as the ``openai_compatible``
    generic catch-all). A conventional port from ``PORT_PROFILES`` is detected
    as that named provider. Thread-per-request; state access is locked.
    """

    handler_base: type[_BaseHandler] = _Handler

    def __init__(self, port: int = 0, *, prompt_tokens: int = 120, completion_tokens: int = 45):
        self._state = _ServerState(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        handler = type("_BoundHandler", (self.handler_base,), {"state": self._state})
        self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def prompt_tokens(self) -> int:
        return self._state.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._state.completion_tokens

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._state.lock:
            return list(self._state.requests)

    @property
    def request_count(self) -> int:
        with self._state.lock:
            return len(self._state.requests)

    def fail_next(self, status: int, count: int = 1, *, retry_after: str | None = None) -> None:
        """Queue *count* error responses, optionally carrying a Retry-After header."""
        with self._state.lock:
            self._state.fail_statuses.extend([(status, retry_after)] * count)

    def set_omit_usage(self, omit: bool = True) -> None:
        """Omit usage from JSON and SSE terminal payloads on both API routes."""
        with self._state.lock:
            self._state.omit_usage = omit

    def drop_next_stream(self, *, after_chunks: int) -> None:
        """The NEXT streaming response sends *after_chunks* SSE events then
        closes the connection without the chunked terminator. One-shot."""
        with self._state.lock:
            self._state.drop_stream_after = after_chunks

    def reset(self) -> None:
        """Clear recorded requests, queued failures, omit flag, and pending drop."""
        with self._state.lock:
            self._state.requests.clear()
            self._state.fail_statuses.clear()
            self._state.omit_usage = False
            self._state.drop_stream_after = None

    def start(self) -> FakeProviderServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the server. Safe to call twice, and before start().

        ``shutdown()`` blocks on an event that only ``serve_forever()`` sets, so
        calling it on a never-started server would hang forever — guard on the
        thread. ``server_close()`` always runs so the port is released even when
        start() was never reached (e.g. a fixture's defensive finally block).
        """
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> FakeProviderServer:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()


class FakeAnthropicServer(FakeProviderServer):
    """Anthropic-dialect sibling. base_url has NO /v1 (the SDK appends /v1/messages)."""

    handler_base = _AnthropicHandler

    def __init__(self, port: int = 0, *, prompt_tokens: int = 88, completion_tokens: int = 44):
        super().__init__(port, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def start_on_conventional_port(**kwargs: Any) -> FakeProviderServer | None:
    """Start a server on the first free conventional local port, else None.

    Tries lmstudio's 1234 first (least likely to be occupied), then ollama's
    11434, then vllm's 8000. None means every port is taken by a real service
    — callers should skip rather than fight a live local server.
    """
    for port in (1234, 11434, 8000):
        try:
            return FakeProviderServer(port, **kwargs).start()
        except OSError:
            continue
    return None
