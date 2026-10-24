"""Adapter for the official SDK (nova_sdk 1.0, starter package) and its harness.

`nova_sdk.run_agent(prepare, consult)`: state = prepare(); client.ready(); consult(client, state). The case is reached
only through the client:

    client.session()                    {"session_id", "status", "turns_remaining", "deadline" (ISO 8601),
                                         "initial": {"opening", "vital_signs"}, "available_actions": [...]}
    client.say(text, request_id=...)    {"status": "OK", "content": <patient>, "turn", "turns_remaining"}
    client.exam(text, request_id=...)   {"status": "OK", "result": <finding>, "turn", "turns_remaining"}
    client.test(text, request_id=...)   final round only (ACTION_NOT_AVAILABLE in the preliminary round)
    client.diagnose(note=, diagnosis=)  note = empty_record() filled: S{cc,hpi,pmh,fh,sh,ros} O{exam,tests}
                                        A{evidence,differential} P{tests,treatment,disposition,follow_up,education};
                                        diagnosis: one line, <= 500 bytes. Ends the case.
    client.chat(messages, max_tokens=, reasoning_effort=, temperature=, request_id=)
                                        the Doctor model (`model` may be omitted), OpenAI-style
                                        response (SDK 1.0.1; 1.0.0 took only max_tokens); temperature is ignored

Errors are `NovaError(code, status)`; the SDK's own checks raise ValueError before anything is sent (text of 0 or
more than 30 NFC characters, a note without exactly the official fields, an empty diagnosis) and cost nothing. A
wrongly sent SAY/EXAM refused by the server (protocol.TURN_USING_CODES) costs one turn. protocol.py lists the codes.
Same request_id = the same request (safe to resend after a network error); a new request_id = a new request.

The readers below also accept other shapes (plain strings, objects, other key names), so the agent keeps working if a
later SDK version changes them; tests/test_nova.py checks dict, object and string replies.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from dxagent.nova.llm import HttpError
from dxagent.nova.protocol import (MAX_DIAGNOSIS_BYTES, MODEL_LIMIT_CODES, REFUSAL_CODES, RETRY_NEW_CODES,
                                   RETRY_SAME_CODES, STOP_CODES, TURN_USING_CODES, Action, Observation, Opening,
                                   error_code)

log = logging.getLogger(__name__)

_TEXT_KEYS = ("content", "result", "response", "reply", "answer", "text", "message", "observation", "output", "patient")
_OPENING_KEYS = ("opening", "first_utterance", "first_words", "patient_opening", "utterance", "chief_complaint",
                 "complaint", "intro", "introduction", "presentation", "message", "patient", "text")
_VITAL_KEYS = ("vital_signs", "vitals", "vitalSigns", "vital", "vs", "v_s")


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    for k in _TEXT_KEYS:
        v = _get(resp, k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, (dict, list)):
            t = _text(v) if not isinstance(v, list) else " ".join(_text(x) for x in v)
            if t.strip():
                return t
    for k in ("data", "event"):
        v = _get(resp, k)
        if v is not None and v is not resp:
            t = _text(v)
            if t:
                return t
    return "" if isinstance(resp, (dict, list)) or hasattr(resp, "__dict__") else str(resp)


def _int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _turn(resp: Any, max_turns: int) -> Optional[int]:
    for k in ("turn", "turns_used", "used_turns", "turn_count", "turn_index"):
        n = _int(_get(resp, k))
        if n is not None:
            return n
    for k in ("turns_remaining", "turns_left", "remaining_turns", "remaining"):
        n = _int(_get(resp, k))
        if n is not None:
            return max_turns - n
    return None


def _done(resp: Any) -> bool:
    for k in ("done", "finished", "ended", "is_done", "terminal"):
        v = _get(resp, k)
        if isinstance(v, bool):
            return v
    status = _get(resp, "status")
    if isinstance(status, str) and status.lower() in ("completed", "closed", "expired"):
        return True
    for k in ("turns_remaining", "turns_left", "remaining_turns"):
        n = _int(_get(resp, k))
        if n is not None and n <= 0:
            return True
    return False


def seconds_left(session: Any) -> Optional[float]:
    """Seconds to the session deadline (ISO 8601), or None."""
    d = _get(session, "deadline") if session is not None else None
    if not isinstance(d, str) or not d:
        return None
    try:
        t = datetime.fromisoformat(d.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (t - datetime.now(timezone.utc)).total_seconds()


def opening_from_state(state: Any, client: Any = None) -> Opening:
    """The start information: `client.session()["initial"]` (nova_sdk 1.0), else `state` if it carries it."""
    from dxagent.nova.agent import opening_from_dict

    sources: list[Any] = []
    if client is not None:
        sources.append(_Lazy(getattr(client, "session", None)))
        for name in ("get_session", "get_state"):
            sources.append(_Lazy(getattr(client, name, None)))
    sources.append(state)
    for src in sources:
        if isinstance(src, _Lazy):
            src = src()
        if src is None or isinstance(src, (int, float, bool)):
            continue
        if isinstance(src, dict) and src.get("_dxagent_prepared"):
            continue
        if isinstance(src, str):
            if src.strip():
                return Opening(text=src.strip())
            continue
        d = src if isinstance(src, dict) else {k: getattr(src, k) for k in dir(src)
                                                if not k.startswith("_") and not callable(getattr(src, k, None))}
        for nested in ("initial", "start_information", "start", "case", "intake", "data"):
            if isinstance(d.get(nested), dict):
                d = {**d[nested], **{k: v for k, v in d.items() if k != nested}}
        text = next((d[k] for k in _OPENING_KEYS if isinstance(d.get(k), str) and d[k].strip()), "")
        vitals = next((d[k] for k in _VITAL_KEYS if d.get(k)), None)
        if text or vitals:
            return opening_from_dict({**d, "opening": text, "vitals": vitals})
    return Opening(text="")


class _Lazy:
    def __init__(self, fn):
        self.fn = fn if callable(fn) else None

    def __call__(self) -> Any:
        if self.fn is None:
            return None
        try:
            return self.fn()
        except Exception as exc:
            log.info("client session lookup failed: %s", exc)
            return None


def _call(fn, text: str, request_id: str):
    try:
        return fn(text, request_id=request_id)
    except TypeError as exc:
        if "request_id" not in str(exc):
            raise
        return fn(text)


class SdkEnv:
    """Pull-style environment over the SDK client: start / act / submit (see NovaAgent.run)."""

    def __init__(self, client: Any, state: Any = None, max_turns: int = 50, retries: int = 4,
                 sleep=time.sleep, before_submit=None):
        self.client = client
        self.state = state
        self.max_turns = max_turns
        self.retries = retries
        self.sleep = sleep
        self.before_submit = before_submit  # e.g. make sure the case has its required client.chat() call
        self.turns = 0
        self.session: Any = None
        self.seconds_left: Optional[float] = None
        self.available: Optional[list[str]] = None
        self.closed = False
        self.log: list[dict[str, Any]] = []

    def start(self) -> Opening:
        fn = getattr(self.client, "session", None)
        if callable(fn):
            for attempt in range(self.retries):
                try:
                    self.session = fn()
                    break
                except Exception as exc:
                    code = error_code(exc)
                    self.log.append({"action": "session", "error": str(exc)[:200], "code": code})
                    if code in RETRY_NEW_CODES + RETRY_SAME_CODES and attempt + 1 < self.retries:
                        self.sleep(1.0 + attempt)
                        continue
                    break
        if self.session is not None:
            self.seconds_left = seconds_left(self.session)
            left = _int(_get(self.session, "turns_remaining"))
            if left is not None:
                self.turns = max(0, self.max_turns - left)
            acts = _get(self.session, "available_actions")
            self.available = [str(a).upper() for a in acts] if isinstance(acts, (list, tuple)) else None
            op = opening_from_state(self.session)
            if op.text or op.vitals:
                return op
        return opening_from_state(self.state, None if self.session is not None else self.client)

    def act(self, action: Action) -> Observation:
        if self.closed:
            return Observation(accepted=False, done=True, code="SESSION_CLOSED")
        if self.available is not None and action.kind not in self.available:
            return Observation(accepted=False, code="ACTION_NOT_AVAILABLE")
        fn = getattr(self.client, {"SAY": "say", "EXAM": "exam", "TEST": "test"}[action.kind], None)
        if fn is None:
            return Observation(accepted=False, code="ACTION_NOT_AVAILABLE")
        rid = str(uuid.uuid4())
        for attempt in range(self.retries):
            try:
                resp = _call(fn, action.text, rid)
            except Exception as exc:
                code = error_code(exc)
                self.log.append({"action": action.kind, "text": action.text, "error": str(exc)[:200], "code": code})
                if code in STOP_CODES:
                    self.closed = True
                    return Observation(accepted=False, done=True, code=code)
                if code in REFUSAL_CODES:
                    if code in TURN_USING_CODES and not isinstance(exc, ValueError):  # the server counted a turn
                        self.turns += 1
                        return Observation(accepted=False, code=code, turn=self.turns, info={"turn_used": True},
                                           done=self.turns >= self.max_turns)
                    return Observation(accepted=False, code=code)
                if code in RETRY_SAME_CODES + RETRY_NEW_CODES and attempt + 1 < self.retries:
                    if code in RETRY_NEW_CODES:
                        rid = str(uuid.uuid4())
                    self.sleep(2.0 * (attempt + 1))
                    continue
                return Observation(accepted=False, code=code or "ERROR", info={"error": str(exc)[:300]})
            turn = _turn(resp, self.max_turns)
            self.turns = turn if turn is not None and turn > self.turns else self.turns + 1
            obs = Observation(text=_text(resp), turn=self.turns, accepted=True,
                              done=_done(resp) or self.turns >= self.max_turns)
            self.log.append({"action": action.kind, "text": action.text, "reply": obs.text, "turn": obs.turn})
            return obs
        return Observation(accepted=False, code="PROVIDER_UNAVAILABLE")

    def submit(self, action: Action) -> Any:
        if self.before_submit is not None:
            try:
                self.before_submit()
            except Exception as exc:
                self.log.append({"action": "before_submit", "error": str(exc)[:200]})
        note = action.note or {}
        dx = one_line_diagnosis(action.diagnosis)
        rid = str(uuid.uuid4())
        fixed: set[str] = set()
        last: Optional[BaseException] = None
        for attempt in range(self.retries + 3):
            try:
                try:
                    return self.client.diagnose(note=note, diagnosis=dx, request_id=rid)
                except TypeError as exc:
                    if "request_id" not in str(exc):
                        raise
                    return self.client.diagnose(note=note, diagnosis=dx)
            except Exception as exc:
                last = exc
                code = error_code(exc)
                self.log.append({"action": "DIAGNOSE", "error": str(exc)[:300], "code": code})
                if code in STOP_CODES:
                    break
                if code in RETRY_SAME_CODES + RETRY_NEW_CODES and attempt < self.retries:
                    if code in RETRY_NEW_CODES:
                        rid = str(uuid.uuid4())
                    self.sleep(2.0 * (attempt + 1))
                    continue
                if code not in fixed and code in ("INVALID_RECORD_FIELDS", "RECORD_TOO_LARGE", "FINAL_RECORD_REQUIRED",
                                                  "SINGLE_DIAGNOSIS_REQUIRED"):
                    fixed.add(code)
                    note, dx = repair_record(note, dx, code)
                    rid = str(uuid.uuid4())
                    continue
                break
        if last is not None:
            raise last
        return None


def one_line_diagnosis(dx: str) -> str:
    d = " ".join((dx or "").split()) or "진단 미정"
    return d.encode("utf-8")[:MAX_DIAGNOSIS_BYTES].decode("utf-8", "ignore").strip()


def record_fields() -> dict[str, tuple[str, ...]]:
    """The official field set: nova_sdk.RECORD_FIELDS if importable, else soap.FIELDS (the same in SDK 1.0)."""
    try:
        import nova_sdk  # type: ignore

        f = getattr(nova_sdk, "RECORD_FIELDS", None)
        if isinstance(f, dict) and f:
            return {s: tuple(v) for s, v in f.items()}
    except Exception:
        pass
    from dxagent.nova.soap import FIELDS

    return dict(FIELDS)


def repair_record(note: dict, dx: str, code: str) -> tuple[dict, str]:
    """One fix per refusal: exact fields (copy what fits), shrink, fill an empty record, one-line diagnosis."""
    fields = record_fields()
    out = {s: {k: str(((note.get(s) or {}) if isinstance(note.get(s), dict) else {}).get(k, "") or "") for k in ks}
           for s, ks in fields.items()}
    if code == "RECORD_TOO_LARGE":
        for _ in range(12):
            if len(str(out).encode("utf-8")) <= 80_000:
                break
            s, k = max(((s, k) for s in out for k in out[s]), key=lambda x: len(out[x[0]][x[1]]))
            out[s][k] = out[s][k][: len(out[s][k]) // 2]
    if code == "FINAL_RECORD_REQUIRED" and not any(v.strip() for f in out.values() for v in f.values()):
        first = next(iter(out))
        out[first][next(iter(out[first]))] = "기록 작성 중 오류: 문진 내용 확인 필요"
    return out, one_line_diagnosis(dx)


class SdkTransport:
    """NovaLLM transport over `client.chat(messages, max_tokens=, reasoning_effort=, temperature=, request_id=)`: the
    contest's metered route to the Doctor model, with per-case call and token limits.
    `model` is never passed: the server uses the Doctor model.

    Only the parameters the client takes are passed (SDK 1.0.0: max_tokens and request_id; 1.0.1 adds reasoning_effort
    and temperature). PROVIDER_UNAVAILABLE (no answer started in time) and REQUEST_PENDING (this or
    another Doctor call still running) are resent here with the SAME request_id, which returns the finished answer
    instead of starting a second call. Other errors become HttpError so NovaLLM's retry logic applies:
    PROVIDER_RESPONSE_INVALID (an empty answer, usually reasoning that used the whole token budget; counted as a call
    but not charged) -> 503 "empty", which NovaLLM answers with a larger budget; other service errors -> 503;
    a spent budget (MODEL_CALL_LIMIT, MODEL_TOKEN_BUDGET) -> 402, and every later call fails at once."""

    PENDING_WAITS = (5.0, 10.0, 20.0, 30.0)  # seconds before each resend of a pending or unstarted request

    def __init__(self, client: Any):
        self.client = client
        self.dropped: set[str] = set()
        self.calls = 0  # client.chat() invocations (the preliminary round requires at least one per case)
        self.exhausted = ""
        try:
            sig = inspect.signature(client.chat)
            self.accepts_any = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            self.params = set(sig.parameters)
        except (TypeError, ValueError):
            self.accepts_any, self.params = True, set()

    @property
    def supports_reasoning_effort(self) -> bool:
        return (self.accepts_any or "reasoning_effort" in self.params) and "reasoning_effort" not in self.dropped

    def chat(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self.exhausted:
            raise HttpError(402, self.exhausted)
        kwargs = {k: v for k, v in payload.items() if k not in ("model", "messages") and k not in self.dropped}
        if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        if not self.accepts_any and self.params:
            kwargs = {k: v for k, v in kwargs.items() if k in self.params}
        if "request_id" in self.params or self.accepts_any:
            kwargs["request_id"] = str(uuid.uuid4())
        messages = [{"role": m["role"], "content": m["content"]} for m in payload["messages"]
                    if isinstance(m.get("content"), str) and m["content"].strip()]
        waited, pending = 0.0, 0
        for attempt in range(12):
            self.calls += 1
            try:
                return _normalize(self.client.chat(messages, **kwargs))
            except TypeError as exc:
                m = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if m and m.group(1) in kwargs:
                    self.dropped.add(m.group(1))
                    kwargs.pop(m.group(1))
                    continue
                raise HttpError(400, str(exc)) from None
            except Exception as exc:
                code = error_code(exc)
                if code in ("PROVIDER_UNAVAILABLE", "REQUEST_PENDING") and pending < len(self.PENDING_WAITS) \
                        and waited + self.PENDING_WAITS[pending] <= max(timeout, 30.0):
                    time.sleep(self.PENDING_WAITS[pending])  # the same request_id: the server finishes the one call
                    waited += self.PENDING_WAITS[pending]
                    pending += 1
                    continue
                if code in RETRY_SAME_CODES and code not in ("PROVIDER_UNAVAILABLE", "REQUEST_PENDING") and attempt < 2:
                    time.sleep(1.0 + attempt)  # network: the same request again (no double charge)
                    continue
                if code in MODEL_LIMIT_CODES:
                    self.exhausted = code
                    raise HttpError(402, code) from None
                if code == "INVALID_COMPLETION_FIELDS":  # the server takes messages, max_tokens, effort, temperature
                    opt = next((o for o in ("temperature", "reasoning_effort") if o in kwargs), None)
                    if opt is None:
                        raise HttpError(400, code) from None
                    self.dropped.add(opt)  # without it from now on (the effort then rides in the system message)
                    kwargs.pop(opt)
                    if "request_id" in kwargs:  # a changed request is a new request
                        kwargs["request_id"] = str(uuid.uuid4())
                    continue
                if code == "PROVIDER_RESPONSE_INVALID":
                    raise HttpError(503, "empty answer (PROVIDER_RESPONSE_INVALID)") from None
                if code in RETRY_NEW_CODES + RETRY_SAME_CODES:
                    raise HttpError(503, code) from None
                if code in STOP_CODES:
                    raise HttpError(410, code) from None
                if code == "INVALID_MESSAGES" or isinstance(exc, ValueError):
                    raise HttpError(400, f"{code or 'ValueError'}: {exc}") from None
                raise
        raise HttpError(400, "too many unsupported arguments")


def ensure_chat_call(transport: Optional["SdkTransport"]) -> bool:
    """The preliminary round scores a case 0 if client.chat() was never called. If every model call of the case was
    skipped (budget, breaker, offline mode), make one small call now. Returns True if a call was attempted here."""
    if transport is None or transport.calls > 0:
        return False
    for _ in range(2):
        try:
            # room for a reasoning model's thinking: an answer cut at the limit comes back empty
            transport.chat({"messages": [{"role": "system", "content": "Answer with one word."},
                                         {"role": "user", "content": "Ready?"}],
                            "max_tokens": 2048, "reasoning_effort": "low"}, timeout=60)
            break
        except Exception as exc:
            log.warning("required client.chat() call failed: %s", exc)
            if transport.exhausted:
                break
    return True


def _normalize(resp: Any) -> dict[str, Any]:
    if isinstance(resp, dict) and "choices" in resp:
        return resp
    if hasattr(resp, "model_dump"):
        try:
            d = resp.model_dump()
            if isinstance(d, dict) and "choices" in d:
                return d
        except Exception:
            pass
    choices = _get(resp, "choices")
    if choices:
        c0 = choices[0]
        msg = _get(c0, "message")
        content = _get(msg, "content") if msg is not None else _get(c0, "text")
        usage = _get(resp, "usage")
        u = {k: _get(usage, k) for k in ("prompt_tokens", "completion_tokens")} if usage is not None else {}
        return {"choices": [{"message": {"content": content or ""}, "finish_reason": _get(c0, "finish_reason") or "stop"}],
                "usage": u}
    text = resp if isinstance(resp, str) else _text(resp)
    usage = _get(resp, "usage") if not isinstance(resp, str) else None
    u = {k: _get(usage, k) for k in ("prompt_tokens", "completion_tokens")} if usage is not None else {}
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": u}
