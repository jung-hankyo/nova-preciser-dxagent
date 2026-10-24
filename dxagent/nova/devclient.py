"""LocalClient: a stand-in for `nova_sdk.Client` (SDK 1.0.1), backed by LocalSimEnv.

It follows the SDK's public interface and response shapes, so `run.consult(client, state)` runs end to end in tests through exactly the harness's code path:

    session()                      {"session_id", "status", "turns_remaining", "initial": {"opening", "vital_signs"},
                                    "available_actions", "deadline"}
    say / exam / test(content, *, request_id=None)
                                   {"status": "OK", "content" | "result", "turn", "turns_remaining"};
                                   refusals raise NovaError(code) (EXAM_NOT_SPECIFIC uses a turn); 0 or > 30 NFC
                                   characters raise ValueError before anything is sent
    diagnose(*, note, diagnosis, request_id=None)   exact RECORD_FIELDS or ValueError, then NovaError codes
    chat(messages, *, max_tokens=2048, reasoning_effort=None, temperature=None, model=None, request_id=None)
                                   OpenAI-style completion; a model other than the Doctor raises MODEL_NOT_ALLOWED

The competition's own development kit remains the reference. `reply_style="object"` or `"str"` returns other shapes to test the adapter's tolerance.
"""

from __future__ import annotations

import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from dxagent.nova.config import NovaConfig
from dxagent.nova.env import LocalSimEnv, NovaError
from dxagent.nova.protocol import Action
from dxagent.nova.soap import FIELDS, note_shape_ok



class LocalClient:
    def __init__(self, case: dict[str, Any], cfg: Optional[NovaConfig] = None,
                 chat: Optional[Callable[..., Any]] = None, reply_style: str = "sdk", patient_llm: Any = None,
                 minutes: float = 30.0):
        self.env = LocalSimEnv(case, cfg, patient_llm=patient_llm)
        self.cfg = self.env.cfg
        self._chat = chat
        self.reply_style = reply_style  # sdk | object | str
        self.deadline = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self.session_id = uuid.uuid4().hex
        self.opening = self.env.start()
        self.chat_calls = 0
        self.chat_options: list[dict[str, Any]] = []  # what each chat() call asked for (tests)
        self.request_ids: list[str] = []
        self.diagnosed: Optional[dict[str, Any]] = None

    # -- session ------------------------------------------------------------------------------------------------

    def ready(self) -> dict[str, bool]:
        return {"ready": True}

    def session(self) -> dict[str, Any]:
        op = self.opening
        vitals = op.vitals if isinstance(op.vitals, str) else ", ".join(f"{k}: {v}" for k, v in (op.vitals or {}).items())
        acts = ["SAY", "EXAM", "DIAGNOSE"] + (["TEST"] if self.env.round == "final" else [])
        return {"session_id": self.session_id, "status": "completed" if self.diagnosed else "active",
                "turns_remaining": self.cfg.max_turns - self.env.turns,
                "initial": {"opening": op.text, "vital_signs": vitals}, "available_actions": acts,
                "deadline": self.deadline.isoformat(), "policy_version": "dxagent-local"}

    # -- actions ------------------------------------------------------------------------------------------------

    def _act(self, kind: str, content: str, request_id: Optional[str]) -> Any:
        text = unicodedata.normalize("NFC", str(content))
        if not 0 < len(text) <= 30:
            raise ValueError(f"{kind}: content must be 1 to 30 characters after NFC")
        self.request_ids.append(request_id or str(uuid.uuid4()))
        if self.diagnosed:
            raise NovaError("SESSION_CLOSED")
        obs = self.env.act(Action(kind=kind, text=text))
        if not obs.accepted:
            raise NovaError(obs.code or "API_ERROR")
        left = self.cfg.max_turns - (obs.turn or 0)
        key = "content" if kind == "SAY" else "result"
        if self.reply_style == "str":
            return obs.text
        if self.reply_style == "object":
            return _Obj(status="OK", **{key: obs.text}, turn=obs.turn, turns_remaining=left)
        return {"status": "OK", key: obs.text, "turn": obs.turn, "turns_remaining": left}

    def say(self, content: str, *, request_id: Optional[str] = None) -> Any:
        return self._act("SAY", content, request_id)

    def exam(self, content: str, *, request_id: Optional[str] = None) -> Any:
        return self._act("EXAM", content, request_id)

    def test(self, content: str, *, request_id: Optional[str] = None) -> Any:
        return self._act("TEST", content, request_id)

    def diagnose(self, *, note: dict, diagnosis: str, request_id: Optional[str] = None) -> dict[str, Any]:
        if not note_shape_ok(note, FIELDS):
            raise ValueError("DIAGNOSE: the note needs exactly the S/O/A/P fields, each a string")
        if not (isinstance(diagnosis, str) and diagnosis.strip()):
            raise ValueError("DIAGNOSE: a primary diagnosis is required")
        if "\n" in diagnosis:
            raise NovaError("SINGLE_DIAGNOSIS_REQUIRED")
        res = self.env.submit(Action(kind="DIAGNOSE", note=note, diagnosis=diagnosis))
        self.diagnosed = {"note": note, "diagnosis": diagnosis}
        return {"status": "completed", "score_status": "not_scored", "turn": res.get("turns")}

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 2048, reasoning_effort: Optional[str] = None,
             temperature: Optional[float] = None, model: Optional[str] = None, request_id: Optional[str] = None) -> Any:
        if not isinstance(messages, list) or any(set(m) != {"role", "content"} or not str(m["content"]).strip()
                                                 for m in messages):
            raise NovaError("INVALID_MESSAGES")
        if model not in (None, "openai/gpt-6-luna", "openai/gpt-oss-20b"):  # both IDs reach the Doctor
            raise NovaError("MODEL_NOT_ALLOWED")
        if reasoning_effort not in (None, "low", "medium", "high") or (
                temperature is not None and not 0 <= float(temperature) <= 2):
            raise NovaError("INVALID_COMPLETION_FIELDS")
        self.chat_calls += 1
        self.chat_options.append({"reasoning_effort": reasoning_effort, "temperature": temperature})
        if self._chat is None:
            return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
        return self._chat(messages=messages, max_tokens=max_tokens)


class _Obj:
    def __init__(self, **kw: Any):
        self.__dict__.update(kw)
