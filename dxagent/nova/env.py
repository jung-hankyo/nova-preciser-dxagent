"""Environments the agent can play against.

SdkEnv        (sdk.py) the official harness: consult(client, state) with the competition SDK's client.
LocalSimEnv   a development stand-in that enforces the published rules with the official error codes:
              SAY/EXAM/TEST <= 30 code points after NFC (CONTENT_TOO_LONG), one maneuver per EXAM (EXAM_NOT_SPECIFIC),
              no TEST in the preliminary round (ACTION_NOT_AVAILABLE); CONTENT_TOO_LONG and EXAM_NOT_SPECIFIC cost a
              turn, the other refusals do not; a bundled question
              gets "한 번에 하나씩 물어봐 주세요."; 50 turns, then only DIAGNOSE; the note must have exactly the
              official fields (INVALID_RECORD_FIELDS) within 100,000 / 500 bytes. It answers as a Korean simulated
              patient: an LLM role-play (patient_llm) or, offline, the case file's keyword answers. The competition's
              development kit is the reference simulator; this one exists for tests without it.
StdioEnv      JSON lines on stdin/stdout, for driving the agent from another process during development.

Case file (JSON), see dxagent/nova/cases/:
    {"case_id", "round", "opening": {"text", "name", "age", "sex"}, "vitals": {...},
     "patient": {"facts": "..."}, "qa": [{"keys": [...], "answer": "..."}],
     "exam": [{"keys": [...], "result": "..."}], "tests": [{"keys": [...], "result": "..."}],
     "diagnosis": "...", "key_history": [...], "key_exam": [...], "notes": "...",
     "bundle_compound": false}   # true: "A이나 B" questions get the bundled-question reply (tests the fallback)
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable, Optional, TextIO

from dxagent.nova.config import NovaConfig
from dxagent.nova.protocol import (MAX_DIAGNOSIS_BYTES, MAX_NOTE_BYTES, TURN_USING_CODES, Action, Observation, Opening,
                                   char_len, check_exam, check_test, question_count)

BUNDLED = "한 번에 하나씩 물어봐 주세요."


class NovaError(RuntimeError):
    """An error response from the (simulated) server, with its code."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code

PATIENT_SYSTEM_KO = """당신은 진료실에 온 환자를 연기합니다. 아래 사례 정보 안에서만 대답하세요.
- 의사가 물어본 것만, 일상적인 말로 1~2문장으로 답합니다. 의학 용어나 진단명은 말하지 않습니다.
- 사례 정보에 없는 것을 물으면 그런 문제가 없는 사람처럼 짧게 답합니다(보통 "아니요, 그런 건 없어요").
- 검사 결과를 먼저 말하지 않습니다. 의사가 설명하면 짧게 반응합니다."""


class LocalSimEnv:
    def __init__(self, case: dict[str, Any], cfg: Optional[NovaConfig] = None, patient_llm: Any = None,
                 clock: Callable[[], float] = time.monotonic, round_name: Optional[str] = None):
        self.case = case
        self.cfg = cfg or NovaConfig()
        self.round = round_name or self.cfg.round  # the case file's "round" is informational
        self.patient_llm = patient_llm
        self.clock = clock
        self.t0 = clock()
        self.turns = 0
        self.done = False
        self.history: list[dict[str, str]] = []
        self.violations: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.submission: Optional[dict[str, Any]] = None

    def start(self) -> Opening:
        self.t0 = self.clock()
        op = self.case.get("opening", {})
        if isinstance(op, str):
            op = {"text": op}
        sex = op.get("sex")
        sex = {"M": "male", "F": "female", "남": "male", "여": "female"}.get(sex, sex)
        return Opening(text=op.get("text", ""), vitals=self.case.get("vitals"), age=op.get("age"), sex=sex,
                       name=op.get("name", ""), case_id=self.case.get("case_id", ""), raw=self.case)

    def _refuse(self, action: Action, code: str, why: str = "") -> Observation:
        used = code in TURN_USING_CODES and action.kind != "TEST"
        if used:  # a wrongly sent SAY/EXAM: no answer, one turn
            self.turns += 1
            self.done = self.turns >= self.cfg.max_turns
        self.violations.append({"turn": self.turns, "action": action.kind, "text": action.text, "code": code, "why": why})
        self.actions.append({"kind": action.kind, "text": action.text, "accepted": False, "why": f"{code} {why}".strip(),
                             **({"turn": self.turns} if used else {})})
        return Observation(text="", turn=self.turns, accepted=False, done=self.done, code=code,
                           info={"turn_used": True} if used else {})

    def _bundled(self, text: str) -> bool:
        if question_count(text) > 1:
            return True
        return bool(self.case.get("bundle_compound")) and any(w in text for w in ("이나 ", "거나 "))

    def act(self, action: Action) -> Observation:
        if self.done:
            return self._refuse(action, "TURN_LIMIT_REACHED", "after the turn or time limit")
        if char_len(action.text) > self.cfg.say_max_chars:
            return self._refuse(action, "CONTENT_TOO_LONG", f"{char_len(action.text)} code points")
        if action.kind == "TEST" and self.round != "final":
            return self._refuse(action, "ACTION_NOT_AVAILABLE", "TEST in the preliminary round")
        if action.kind in ("EXAM", "TEST"):
            ok, why = (check_exam if action.kind == "EXAM" else check_test)(action.text, self.cfg.say_max_chars)
            if not ok:
                code = ("TEST_NOT_SPECIFIC" if action.kind == "TEST" else
                        "EXAM_NOT_RECOGNIZED" if "not a physical examination" in why else "EXAM_NOT_SPECIFIC")
                return self._refuse(action, code, why)
        self.turns += 1
        if action.kind == "SAY":
            reply = BUNDLED if self._bundled(action.text) else self._patient(action.text)
        else:
            reply = self._record(action.kind, action.text)
        self.history.append({"role": "의사", "text": action.text})
        self.history.append({"role": "환자" if action.kind == "SAY" else "기록", "text": reply})
        elapsed = self.clock() - self.t0
        self.done = self.turns >= self.cfg.max_turns or elapsed >= self.cfg.time_limit_s
        self.actions.append({"kind": action.kind, "text": action.text, "accepted": True, "turn": self.turns, "reply": reply})
        return Observation(text=reply, turn=self.turns, accepted=True, done=self.done)

    def submit(self, action: Action) -> dict[str, Any]:
        from dxagent.nova.soap import FIELDS, note_shape_ok

        note = action.note
        if not note_shape_ok(note, FIELDS):
            self.violations.append({"turn": self.turns, "action": "DIAGNOSE", "code": "INVALID_RECORD_FIELDS"})
            raise NovaError("INVALID_RECORD_FIELDS", "the note needs exactly the S/O/A/P fields, each a string")
        if len(json.dumps(note, ensure_ascii=False).encode("utf-8")) > MAX_NOTE_BYTES or \
                len((action.diagnosis or "").encode("utf-8")) > MAX_DIAGNOSIS_BYTES:
            raise NovaError("CONTENT_TOO_LONG", "note or diagnosis over its byte limit")
        if not (action.diagnosis or "").strip():
            raise NovaError("INVALID_RECORD_FIELDS", "empty diagnosis")
        self.submission = action.to_json()
        return {"ok": True, "turns": self.turns, "seconds": round(self.clock() - self.t0, 1)}

    # -- voices ---------------------------------------------------------------------------------------------------

    def _patient(self, text: str) -> str:
        if self.patient_llm is not None:
            facts = self.case.get("patient", {}).get("facts", "")
            qa = "\n".join(f"- {', '.join(i['keys'])}: {i['answer']}" for i in self.case.get("qa", []))
            convo = "\n".join(f"{h['role']}: {h['text']}" for h in self.history[-10:])
            user = (f"사례 정보:\n{facts}\n\n자주 받는 질문과 답:\n{qa}\n\n지금까지의 대화:\n{convo}\n\n의사: {text}\n환자:")
            try:
                out = self.patient_llm.text(PATIENT_SYSTEM_KO, user, role="patient")
                if out.strip():
                    return out.strip()
            except Exception:
                pass
        best, score = None, 0
        for item in self.case.get("qa", []):
            s = sum(1 for k in item.get("keys", []) if k and k in text)
            if s > score:
                best, score = item, s
        if best is not None:
            return best["answer"]
        if "?" not in text:
            return "네, 알겠습니다."
        return self.case.get("default_answer", "아니요, 그런 건 없어요.")

    def _record(self, kind: str, text: str) -> str:
        table = self.case.get("exam" if kind == "EXAM" else "tests", [])
        best, score = None, 0
        for item in table:
            s = sum(1 for k in item.get("keys", []) if k and k in text)
            if s > score:
                best, score = item, s
        if best is not None:
            return best["result"]
        return "특이 소견 없습니다." if kind == "EXAM" else "정상 범위입니다."


class StdioEnv:
    """JSON lines. In:  {"type":"start","case_id":...,"opening":"...","vitals":{...},"age":..,"sex":..}
                         {"type":"observation","text":"...","turn":3,"accepted":true,"done":false}
                   Out: {"action":"SAY"|"EXAM"|"TEST","content":"..."}
                         {"action":"DIAGNOSE","note":{"S":{...},"O":{...},"A":{...},"P":{...}},"diagnosis":"..."}
    Logs go to stderr only."""

    def __init__(self, inp: TextIO = sys.stdin, out: TextIO = sys.stdout):
        self.inp, self.out = inp, out
        self.eof = False

    def _read(self) -> Optional[dict[str, Any]]:
        while True:
            line = self.inp.readline()
            if not line:
                self.eof = True
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return {"type": "observation", "text": line}

    def _write(self, obj: dict[str, Any]) -> None:
        self.out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.out.flush()

    def start(self) -> Opening:
        from dxagent.nova.agent import opening_from_dict

        while True:  # skip anything that is not the start of a case (late observations, acknowledgements)
            d = self._read()
            if d is None:
                raise EOFError("no case")
            if d.get("type") in ("start", "case", "reset") or (d.get("type") is None and
                                                                 any(k in d for k in ("opening", "vitals", "case_id"))):
                return opening_from_dict(d)

    def act(self, action: Action) -> Observation:
        from dxagent.nova.agent import observation_from_dict

        self._write(action.to_json())
        d = self._read()
        if d is None:
            return Observation(text="", accepted=False, done=True)
        return observation_from_dict(d)

    def submit(self, action: Action) -> None:
        self._write(action.to_json())
