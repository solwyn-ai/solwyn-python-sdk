"""A minimal local OpenAI-compatible chat-completions server for E2E tests.

Test infrastructure ONLY — never imported by src/. Serves POST
``/v1/chat/completions`` in both JSON and SSE-streaming form, always returning
a valid usage block (configurable token counts), and records every request so
tests can assert the provider was (or was NOT) called and with which payload.

Why this doubles as detection coverage: Solwyn detects OpenAI-compatible
providers from the client's base_url — conventional local ports (1234/11434/
8000) map to lmstudio/ollama/vllm, any other local port lands in the
``openai_compatible`` generic catch-all. Binding this server to an ephemeral
vs. conventional port therefore exercises both detection paths with zero
external dependencies.

Extension seams for the failover session:
- ``fail_next(status, count=N)`` queues N error responses before recovery.
- Instantiate TWO servers and wire them as primary + fallback clients.
- Streaming responses always end with a usage-bearing final chunk; add
  mid-stream abort support here if a test needs it.
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
    fail_statuses: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Handler(BaseHTTPRequestHandler):
    state: _ServerState  # injected by FakeProviderServer via subclassing

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence per-request stderr noise."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        with self.state.lock:
            self.state.requests.append(RecordedRequest(path=self.path, body=body))
            fail_status = self.state.fail_statuses.pop(0) if self.state.fail_statuses else None

        if not self.path.endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"no route: {self.path}"}})
            return
        if fail_status is not None:
            self._send_json(
                fail_status, {"error": {"message": "injected failure", "type": "fake_error"}}
            )
            return
        if body.get("stream"):
            self._send_stream(body)
        else:
            self._send_json(200, self._completion(body))

    def _usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.state.prompt_tokens,
            "completion_tokens": self.state.completion_tokens,
            "total_tokens": self.state.prompt_tokens + self.state.completion_tokens,
        }

    def _completion(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
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

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_stream(self, body: dict[str, Any]) -> None:
        """SSE stream: two content chunks, a stop chunk, a usage-only final chunk.

        The final usage-bearing chunk mirrors real OpenAI-dialect behavior with
        ``stream_options.include_usage`` AND the always-on final-chunk usage many
        compat providers emit — so both the include_usage-injected (lmstudio)
        and never-inject (generic catch-all) adapter policies extract real usage
        (tier 1), never the estimation fallback.
        """
        model = body.get("model", "fake-model")

        def chunk(delta: dict[str, Any], finish: str | None, usage: dict[str, int] | None):
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
            chunk(None, None, self._usage()),
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


class FakeProviderServer:
    """A fake OpenAI-compatible provider on 127.0.0.1.

    ``port=0`` binds an ephemeral port (detected as the ``openai_compatible``
    generic catch-all). A conventional port from ``PORT_PROFILES`` is detected
    as that named provider. Thread-per-request; state access is locked.
    """

    def __init__(self, port: int = 0, *, prompt_tokens: int = 120, completion_tokens: int = 45):
        self._state = _ServerState(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        handler = type("_BoundHandler", (_Handler,), {"state": self._state})
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

    def fail_next(self, status: int, count: int = 1) -> None:
        """Queue *count* error responses (failover-session extension seam)."""
        with self._state.lock:
            self._state.fail_statuses.extend([status] * count)

    def reset(self) -> None:
        """Clear recorded requests and queued failures (between tests)."""
        with self._state.lock:
            self._state.requests.clear()
            self._state.fail_statuses.clear()

    def start(self) -> FakeProviderServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> FakeProviderServer:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()


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
