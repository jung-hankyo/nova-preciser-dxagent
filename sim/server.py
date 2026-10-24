"""A simulated contest server: one case, its session, actions, the Doctor model route and the faults of a scenario.

It reproduces the server behaviour the agent depends on, as far as the SDK shows it, and is written independently of
the agent's own checks: the session and its deadline; SAY / EXAM / TEST / DIAGNOSE with request_id (the same
request_id returns the stored response, never a second turn); refusals and the turns they cost; the turn limit; the
Doctor model route with its field check and per-case limits, a hold before PROVIDER_UNAVAILABLE and the answer
collected later with the same request_id; and the readiness gate.

Every request is logged with the virtual time, so a run can be checked afterwards (sim/check.py).
"""

from __future__ import annotations

import json
import random
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sim.actors import Examiner, ScriptedPatient
from sim.doctor import Answer, Doctor, estimate_tokens, role_of
from sim.vclock import VirtualClock

RECORD_FIELDS = {"S": ("cc", "hpi", "pmh", "fh", "sh", "ros"), "O": ("exam", "tests"),
                 "A": ("evidence", "differential"), "P": ("tests", "treatment", "disposition", "follow_up", "education")}
DOCTOR_IDS = (None, "openai/gpt-6-luna", "openai/gpt-oss-20b")
CHAT_FIELDS = {"request_id", "messages", "max_tokens", "reasoning_effort", "temperature", "model", "session_id"}
HOLD_S = 180.0


def record_ok(note: Any) -> bool:
    """The note has the record's sections and fields exactly, and every value is a string."""
    if not isinstance(note, dict) or sorted(note) != sorted(RECORD_FIELDS):
        return False
    return all(isinstance(note[sec], dict) and sorted(note[sec]) == sorted(names)
               and all(isinstance(v, str) for v in note[sec].values()) for sec, names in RECORD_FIELDS.items())


@dataclass
class Faults:
    net_before: float = 0.0  # share of requests lost before they reach the server (API_UNAVAILABLE)
    net_after: float = 0.0  # share of responses lost after the server executed the request
    patient_invalid: float = 0.0  # share of SAYs answered PATIENT_RESPONSE_INVALID (no turn)
    examiner_invalid: float = 0.0  # share of EXAMs answered EXAMINER_RESPONSE_INVALID (no turn)
    patient_latency: float = 3.0  # seconds per answer
    examiner_latency: float = 2.0
    reject_effort: bool = False  # an older server: INVALID_COMPLETION_FIELDS when reasoning_effort is sent
    expire_at_turn: int = 0  # the session expires once this many turns are used (0: never)
    diagnose_refuse: str = ""  # the first DIAGNOSE gets this code (INVALID_RECORD_FIELDS, RECORD_TOO_LARGE, ...)
    minutes: float = 30.0  # consult() limit
    turns_used_at_start: int = 0
    no_opening: bool = False  # the start information has no opening statement
    no_vitals: bool = False
    seed: int = 0


@dataclass
class ChatJob:
    rid: str
    role: str
    start: float
    finish_at: float
    answer: Answer
    index: int
    delivered: bool = False


def _err(code: str) -> dict[str, Any]:
    return {"error": {"code": code}}


class FakeServer:
    def __init__(self, case: dict[str, Any], clock: VirtualClock, doctor: Doctor, *, faults: Optional[Faults] = None,
                 patient_behaviour: str = "", round_name: str = "prelim", max_turns: int = 50):
        self.case = case
        self.clock = clock
        self.doctor = doctor
        self.f = faults or Faults()
        self.rng = random.Random(self.f.seed)
        self.round = round_name
        self.max_turns = max_turns
        self.patient = ScriptedPatient(case.get("script", ""), patient_behaviour)
        self.examiner = Examiner(case.get("exams") or {})
        self.session_id = str(uuid.UUID(int=self.rng.getrandbits(128)))
        self.ready = False
        self.turns = self.f.turns_used_at_start
        self.t0 = clock.monotonic()
        self.deadline = self.t0 + self.f.minutes * 60.0
        self.status = "active"
        self.actions: dict[str, tuple[int, dict]] = {}
        self.jobs: dict[str, ChatJob] = {}
        self.chat_calls = 0
        self.chat_refused = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.diagnose_attempts = 0
        self.submission: Optional[dict[str, Any]] = None
        self.log: list[dict[str, Any]] = []

    # -- helpers ------------------------------------------------------------------------------------------------

    def now(self) -> float:
        return self.clock.monotonic()

    def _log(self, **kw: Any) -> None:
        self.log.append({"t": round(self.now() - self.t0, 2), **kw})

    def _expired(self) -> bool:
        if self.status == "active" and (self.now() >= self.deadline or
                                        (self.f.expire_at_turn and self.turns >= self.f.expire_at_turn)):
            self.status = "expired"
        return self.status == "expired"

    def deadline_iso(self) -> str:
        wall = self.clock.epoch + (self.deadline - self.clock.start)
        return datetime.fromtimestamp(wall, timezone.utc).isoformat().replace("+00:00", "Z")

    # -- routing ------------------------------------------------------------------------------------------------

    def handle(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
        body = json.loads(json.dumps(body)) if body is not None else None  # what crosses the wire
        if path == "/_nova/ready":
            self.ready = True
            return 200, {"ready": True}
        if not self.ready:
            return 409, _err("MODEL_NOT_READY")
        if path == "/v1/session" and method == "GET":
            return 200, self.session()
        if path == "/v1/session/actions" and method == "POST":
            return self.action(body or {})
        if path == "/v1/chat/completions" and method == "POST":
            return self.chat(body or {})
        return 404, _err("NOT_FOUND")

    def session(self) -> dict[str, Any]:
        self._expired()
        case = self.case
        opening = "" if self.f.no_opening else case.get("opening", "")
        vitals = "" if self.f.no_vitals else case.get("vital_signs", "")
        acts = ["SAY", "EXAM", "DIAGNOSE"] + (["TEST"] if self.round == "final" else [])
        self._log(kind="SESSION")
        return {"session_id": self.session_id, "status": self.status,
                "turns_remaining": max(0, self.max_turns - self.turns), "deadline": self.deadline_iso(),
                "initial": {"opening": opening, "vital_signs": vitals}, "available_actions": acts}

    # -- actions ------------------------------------------------------------------------------------------------

    def action(self, body: dict) -> tuple[int, dict]:
        rid = str(body.get("request_id") or "")
        if rid and rid in self.actions:
            status, payload = self.actions[rid]
            self._log(kind=str(body.get("action")), rid=rid, duplicate=True, code=payload.get("error", {}).get("code", ""))
            return status, payload
        status, payload = self._action(body)
        if rid:
            self.actions[rid] = (status, payload)
        return status, payload

    def _ok(self, kind: str, key: str, text: str, content: str, rid: str, row: Optional[str] = None) -> tuple[int, dict]:
        self._log(kind=kind, rid=rid, content=content, reply=text, turn=self.turns, row=row)
        return 200, {"status": "OK", key: text, "turn": self.turns, "turns_remaining": max(0, self.max_turns - self.turns)}

    def _refuse(self, kind: str, code: str, content: str, rid: str, turn_used: bool, status: int = 422) -> tuple[int, dict]:
        if turn_used:
            self.turns += 1
        self._log(kind=kind, rid=rid, content=content, code=code, turn=self.turns, turn_used=turn_used)
        return status, _err(code)

    def _action(self, body: dict) -> tuple[int, dict]:
        kind = str(body.get("action") or "")
        rid = str(body.get("request_id") or "")
        if self.status == "completed":
            return self._refuse(kind, "SESSION_CLOSED", str(body.get("content", "")), rid, False, 409)
        if self._expired():
            return self._refuse(kind, "SESSION_EXPIRED", str(body.get("content", "")), rid, False, 409)
        if kind == "DIAGNOSE":
            return self._diagnose(body, rid)
        if kind not in ("SAY", "EXAM", "TEST"):
            return self._refuse(kind, "INVALID_ACTION_FIELDS", "", rid, False, 400)
        if set(body) - {"request_id", "action", "content"}:
            return self._refuse(kind, "INVALID_ACTION_FIELDS", "", rid, False, 400)
        content = unicodedata.normalize("NFC", str(body.get("content") or ""))
        if self.turns >= self.max_turns:
            return self._refuse(kind, "TURN_LIMIT_REACHED", content, rid, False, 409)
        if kind == "TEST" and self.round != "final":
            return self._refuse(kind, "ACTION_NOT_AVAILABLE", content, rid, False, 400)
        if not content.strip():
            return self._refuse(kind, "EMPTY_CONTENT", content, rid, True)
        if len(content) > 30:
            return self._refuse(kind, "CONTENT_TOO_LONG", content, rid, True)
        if kind == "SAY":
            if self.f.patient_invalid and self.rng.random() < self.f.patient_invalid:
                self.clock.advance(self.f.patient_latency)
                return self._refuse(kind, "PATIENT_RESPONSE_INVALID", content, rid, False, 503)
            self.clock.advance(self.f.patient_latency)
            self.turns += 1
            return self._ok(kind, "content", self.patient.reply(content), content, rid)
        if kind == "EXAM":
            if self.f.examiner_invalid and self.rng.random() < self.f.examiner_invalid:
                self.clock.advance(self.f.examiner_latency)
                return self._refuse(kind, "EXAMINER_RESPONSE_INVALID", content, rid, False, 503)
            self.clock.advance(self.f.examiner_latency)
            code, finding, row = self.examiner.examine(content)
            if code:
                return self._refuse(kind, code, content, rid, True)
            self.turns += 1
            return self._ok(kind, "result", finding, content, rid, row)
        self.turns += 1  # TEST, final round
        return self._ok(kind, "result", "정상 범위", content, rid)

    def _diagnose(self, body: dict, rid: str) -> tuple[int, dict]:
        self.diagnose_attempts += 1
        note, dx = body.get("note"), body.get("diagnosis")
        if set(body) - {"request_id", "action", "note", "diagnosis"}:
            return self._refuse("DIAGNOSE", "INVALID_ACTION_FIELDS", "", rid, False, 400)
        if self.f.diagnose_refuse and self.diagnose_attempts == 1:
            return self._refuse("DIAGNOSE", self.f.diagnose_refuse, "", rid, False, 400)
        if not record_ok(note):
            return self._refuse("DIAGNOSE", "INVALID_RECORD_FIELDS", "", rid, False, 400)
        if len(json.dumps(note, ensure_ascii=False).encode("utf-8")) > 100_000:
            return self._refuse("DIAGNOSE", "RECORD_TOO_LARGE", "", rid, False, 413)
        if not isinstance(dx, str) or not dx.strip():
            return self._refuse("DIAGNOSE", "FINAL_RECORD_REQUIRED", "", rid, False, 400)
        if "\n" in dx or len(dx.encode("utf-8")) > 500:
            return self._refuse("DIAGNOSE", "SINGLE_DIAGNOSIS_REQUIRED", "", rid, False, 400)
        self.status = "completed"
        self.submission = {"note": note, "diagnosis": dx, "t": round(self.now() - self.t0, 2), "turns": self.turns,
                           "chat_calls": self.chat_calls}
        self._log(kind="DIAGNOSE", rid=rid, content=dx, turn=self.turns)
        return 200, {"status": "completed", "score_status": "not_scored", "turn": self.turns}

    # -- the Doctor model -----------------------------------------------------------------------------------------

    def chat(self, body: dict) -> tuple[int, dict]:
        rid = str(body.get("request_id") or "")
        msgs = body.get("messages")
        if set(body) - CHAT_FIELDS or (self.f.reject_effort and "reasoning_effort" in body):
            self.chat_refused += 1
            self._log(kind="CHAT", rid=rid, code="INVALID_COMPLETION_FIELDS")
            return 400, _err("INVALID_COMPLETION_FIELDS")
        if (not isinstance(msgs, list) or not msgs or any(not isinstance(m, dict) or set(m) != {"role", "content"} or
                                                         not str(m.get("content", "")).strip() for m in msgs)):
            self.chat_refused += 1
            self._log(kind="CHAT", rid=rid, code="INVALID_MESSAGES")
            return 400, _err("INVALID_MESSAGES")
        if body.get("model") not in DOCTOR_IDS:
            return 403, _err("MODEL_NOT_ALLOWED")
        if body.get("reasoning_effort") not in (None, "low", "medium", "high"):
            return 400, _err("INVALID_COMPLETION_FIELDS")
        now = self.now()
        job = self.jobs.get(rid)
        if job is None:
            running = [j for j in self.jobs.values() if j.finish_at > now]
            if running:
                self.clock.advance(0.2)
                self._log(kind="CHAT", rid=rid, code="REQUEST_PENDING", other=running[0].rid)
                return 409, _err("REQUEST_PENDING")
            if self.chat_calls >= 200:
                return 429, _err("MODEL_CALL_LIMIT")
            if self.in_tokens >= 500_000 or self.out_tokens >= 100_000:
                return 429, _err("MODEL_TOKEN_BUDGET")
            self.chat_calls += 1
            index = self.chat_calls
            role = role_of(msgs)
            ans = self.doctor.answer(msgs, int(body.get("max_tokens") or 2048), body.get("reasoning_effort"), index)
            job = ChatJob(rid, role, now, now + max(0.1, ans.latency), ans, index)
            self.jobs[rid] = job
            hold = min(job.finish_at, now + HOLD_S) - now
            self.clock.advance(hold)
            if self.now() < job.finish_at:
                self._log(kind="CHAT", rid=rid, role=role, index=index, code="PROVIDER_UNAVAILABLE")
                return 503, _err("PROVIDER_UNAVAILABLE")
        elif self.now() < job.finish_at:
            self.clock.advance(0.2)
            self._log(kind="CHAT", rid=rid, role=job.role, index=job.index, code="REQUEST_PENDING", resend=True)
            return 409, _err("REQUEST_PENDING")
        ans = job.answer
        if ans.error:
            self._log(kind="CHAT", rid=rid, role=job.role, index=job.index, code=ans.error)
            return 429 if ans.error.startswith("MODEL_") else 503, _err(ans.error)
        if not (ans.content or "").strip():
            self._log(kind="CHAT", rid=rid, role=job.role, index=job.index, code="PROVIDER_RESPONSE_INVALID",
                      finish=ans.finish)
            return 503, _err("PROVIDER_RESPONSE_INVALID")
        tin = sum(estimate_tokens(m["content"]) for m in msgs)
        tout = max(ans.completion_tokens, estimate_tokens(ans.content))
        tout = min(tout, int(body.get("max_tokens") or 2048))
        if not job.delivered:
            self.in_tokens += tin
            self.out_tokens += tout
            job.delivered = True
        self._log(kind="CHAT", rid=rid, role=job.role, index=job.index, in_tokens=tin, out_tokens=tout,
                  max_tokens=body.get("max_tokens"), effort=body.get("reasoning_effort"),
                  seconds=round(job.finish_at - job.start, 1))
        return 200, {"id": f"chatcmpl-{job.index}", "object": "chat.completion", "model": "openai/gpt-6-luna",
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": ans.content},
                                  "finish_reason": ans.finish}],
                     "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}}
