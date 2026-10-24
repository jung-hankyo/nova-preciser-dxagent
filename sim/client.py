"""Clients for the simulated server.

MirrorClient follows the SDK client's public interface, written for this project: the same methods and keyword
arguments, local checks before anything is sent (ValueError: 1-30 NFC characters for SAY/EXAM/TEST, the exact S/O/A/P
fields, a non-empty diagnosis), NovaError(code, status) for an error response and NovaError('API_UNAVAILABLE') for a
lost connection.

real_sdk_client() wraps the real nova_sdk.Client when nova_sdk.py is importable (it is git-ignored: the SDK file
copied to the repository root): only its socket layer (_open) is replaced, so its own request
building, error mapping and checks run unchanged against the simulated server.

Both apply the scenario's network faults: a request lost before the server sees it, or a response lost after the
server executed it (the case that makes the same-request_id resend necessary).
"""

from __future__ import annotations

import io
import json
import unicodedata
import urllib.error
import urllib.parse
import uuid
from typing import Any, Optional

from sim.server import FakeServer, record_ok


class NovaError(RuntimeError):
    def __init__(self, code: str, status: Optional[int] = None):
        self.code = code
        self.status = status
        super().__init__(code)


class Net:
    """Lost requests and responses, drawn from the server's seeded generator."""

    def __init__(self, server: FakeServer):
        self.server = server
        self.lost: list[tuple[str, str]] = []

    def call(self, method: str, path: str, body: Optional[dict]) -> tuple[int, dict]:
        f, rng = self.server.f, self.server.rng
        if f.net_before and path != "/_nova/ready" and rng.random() < f.net_before:
            self.server.clock.advance(5.0)
            self.lost.append(("before", path))
            raise ConnectionError("simulated: request lost before the server")
        status, payload = self.server.handle(method, path, body)
        if f.net_after and path != "/_nova/ready" and rng.random() < f.net_after:
            self.server.clock.advance(5.0)
            self.lost.append(("after", path))
            raise TimeoutError("simulated: response lost after the server executed the request")
        return status, payload


class MirrorClient:
    def __init__(self, server: FakeServer, timeout: float = 240):
        self.server = server
        self.net = Net(server)
        self.session_id = server.session_id  # the harness sets NOVA_SESSION_ID; the SDK sends it with chat()
        self.timeout = timeout
        self.sample = False

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        try:
            status, payload = self.net.call(method, path, body)
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise NovaError("API_UNAVAILABLE") from exc
        if not 200 <= status < 300:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            raise NovaError(error.get("code", "API_ERROR"), status)
        return payload

    def ready(self) -> dict:
        return self._request("POST", "/_nova/ready", {})

    def session(self) -> dict:
        return self._request("GET", "/v1/session")

    def action(self, action: str, *, request_id: Optional[str] = None, **fields: Any) -> dict:
        body = {"request_id": request_id or str(uuid.uuid4()), "action": action, **fields}
        if action in ("SAY", "EXAM", "TEST"):
            text = unicodedata.normalize("NFC", str(body.get("content", "")))
            if not 0 < len(text) <= 30:
                raise ValueError(f"{action}: content must be 1 to 30 characters after NFC")
            body["content"] = text
        return self._request("POST", "/v1/session/actions", body)

    def say(self, content: str, *, request_id: Optional[str] = None) -> dict:
        return self.action("SAY", content=content, request_id=request_id)

    def exam(self, content: str, *, request_id: Optional[str] = None) -> dict:
        return self.action("EXAM", content=content, request_id=request_id)

    def test(self, content: str, *, request_id: Optional[str] = None) -> dict:
        return self.action("TEST", content=content, request_id=request_id)

    def diagnose(self, *, note: dict, diagnosis: str, request_id: Optional[str] = None) -> dict:
        if not record_ok(note):
            raise ValueError("DIAGNOSE: the note needs exactly the S/O/A/P fields, each a string")
        if not (isinstance(diagnosis, str) and diagnosis.strip()):
            raise ValueError("DIAGNOSE: a primary diagnosis is required")
        return self.action("DIAGNOSE", note=note, diagnosis=diagnosis, request_id=request_id)

    def chat(self, messages: list, *, max_tokens: int = 2048, reasoning_effort: Optional[str] = None,
             temperature: Optional[float] = None, model: Optional[str] = None, request_id: Optional[str] = None) -> dict:
        body: dict[str, Any] = {"request_id": request_id or str(uuid.uuid4()), "messages": messages,
                                "max_tokens": max_tokens}
        if model is not None:
            body["model"] = model
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            body["temperature"] = temperature
        if self.session_id:
            body["session_id"] = self.session_id
        return self._request("POST", "/v1/chat/completions", body)


class _Response(io.BytesIO):
    def __init__(self, status: int, payload: dict):
        super().__init__(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.status = status


def real_sdk_available() -> bool:
    try:
        import nova_sdk  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def real_sdk_client(server: FakeServer) -> Any:
    """The real nova_sdk.Client with its socket layer replaced by the simulated server."""
    import nova_sdk  # type: ignore

    net = Net(server)

    class SimulatedClient(nova_sdk.Client):  # type: ignore[misc, valid-type]
        def _open(self, request):  # noqa: D102  (the only override: urllib's network call)
            path = urllib.parse.urlsplit(request.full_url).path
            body = json.loads(request.data.decode("utf-8")) if request.data else None
            try:
                status, payload = net.call(request.get_method(), path, body)
            except ConnectionError as exc:
                raise urllib.error.URLError(str(exc)) from exc
            return _Response(status, payload)

    client = SimulatedClient(api_base="http://sim.invalid", token="simulated-run-token", session_id=server.session_id)
    client.net = net
    return client
