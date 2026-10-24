"""Local scoring for development (the contest's own rubric and prompts are not public).

proxy_score   deterministic, free: coverage of the case file's key history / exam / test items (asked AND recorded),
              diagnosis keyword match, explanation turns, turns and LLM calls. Use it for regressions.
llm_judge     a six-area rubric (100 points per case) applied by an LLM; rank versions with one judge locally.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

AXES = {"information_gathering": 25, "diagnostic_reasoning": 15, "tests_followup": 10, "safety": 15,
        "communication": 15, "efficiency": 20}


class AxisScore(BaseModel):
    score: float = Field(description="Points awarded on this axis.")
    comment: str = ""


class RubricScore(BaseModel):
    information_gathering: AxisScore = Field(description="0-25: necessary history and exam actually done AND recorded")
    diagnostic_reasoning: AxisScore = Field(description="0-15: primary diagnosis correct; rationale and differential valid")
    tests_followup: AxisScore = Field(description="0-10: needed tests and follow-up planned")
    safety: AxisScore = Field(description="0-15: management decisions appropriate and not harmful")
    communication: AxisScore = Field(description="0-15: communicated well and gave needed explanations during the visit")
    efficiency: AxisScore = Field(description="0-20: avoided unnecessary actions and model usage")


JUDGE_SYSTEM = """You grade a doctor agent's encounter with a simulated patient for a medical AI competition.
Score each axis with the maximum shown (100 points in total):
- information_gathering (25): were the necessary history questions and examinations actually performed AND recorded
  in the note? Items written in the note but never asked or examined earn nothing.
- diagnostic_reasoning (15): is the primary diagnosis correct, and are the rationale and differential sound?
- tests_followup (10): are the needed tests and the follow-up planned?
- safety (15): are the management decisions appropriate and not harmful (allergies, disposition, dangerous
  alternatives addressed)?
- communication (15): did the doctor communicate well and give the needed explanations DURING the visit (an
  education plan written in the note is not the same as explaining)?
- efficiency (20): were unnecessary actions and model usage avoided?
Be strict and consistent. Give a one-sentence comment per axis."""


def judge_user(case: dict[str, Any], transcript: str, soap_text: str, diagnosis: str, usage: dict[str, Any]) -> str:
    ref = {k: case.get(k) for k in ("diagnosis", "key_history", "key_exam", "key_tests", "notes")}
    return (f"Reference (hidden from the agent):\n{json.dumps(ref, ensure_ascii=False)}\n\n"
            f"Encounter transcript:\n{transcript}\n\nSubmitted primary diagnosis: {diagnosis}\n\n"
            f"Submitted SOAP note:\n{soap_text}\n\nUsage: {json.dumps(usage, ensure_ascii=False)}")


def llm_judge(llm: Any, case: dict[str, Any], transcript: str, soap_text: str, diagnosis: str,
              usage: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        out: RubricScore = llm.structured(RubricScore, JUDGE_SYSTEM, judge_user(case, transcript, soap_text, diagnosis, usage))
    except Exception as exc:
        return {"error": str(exc)[:200]}
    scores = {k: max(0.0, min(float(getattr(out, k).score), mx)) for k, mx in AXES.items()}
    return {**scores, "total": round(sum(scores.values()), 1),
            "comments": {k: getattr(out, k).comment for k in AXES}}


def _covered(item: dict[str, Any], doctor_text: str) -> bool:
    return any(k and k in doctor_text for k in item.get("keys", []))


def proxy_score(case: dict[str, Any], rec: dict[str, Any], violations: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic stand-in for the rubric. Coverage counts an item only if it was asked and the note records it."""
    led = [e for e in rec.get("ledger", []) if e.get("accepted")]
    asked_say = " ".join(e["text"] for e in led if e["kind"] == "SAY")
    asked_exam = " ".join(e["text"] for e in led if e["kind"] in ("EXAM", "TEST"))
    soap = (rec.get("soap") or {})
    plan = soap.get("P", "")
    kh, ke, kt = case.get("key_history", []), case.get("key_exam", []), case.get("key_tests", [])
    hist = sum(_covered(i, asked_say) for i in kh) / len(kh) if kh else 1.0
    exam = sum(_covered(i, asked_exam) for i in ke) / len(ke) if ke else 1.0
    tests = sum(_covered(i, plan + " " + asked_exam) for i in kt) / len(kt) if kt else 1.0
    dx_keys = [k.lower() for k in case.get("diagnosis_keys", [])]
    dx = (rec.get("diagnosis") or "").lower()
    in_diff = any(k in soap.get("A", "").lower() for k in dx_keys)
    dx_score = 15.0 if any(k in dx for k in dx_keys) else (5.0 if in_diff else 0.0)
    explained = sum(1 for e in led if e.get("section") == "education")  # explanation turns said to the patient
    empathy = any("불편하셨" in e["text"] or "힘드셨" in e["text"] for e in led)
    comm = 15.0 * min(1.0, 0.375 * min(explained, 2) + (0.25 if empathy else 0.0))
    turns = rec.get("turns", 0)
    calls = rec.get("budget", {}).get("calls", 0)
    eff = max(0.0, 20.0 - max(0, turns - 20) * 0.5 - max(0, calls - 25) * 0.3 - 2.0 * len(violations))
    safety = 15.0 - 5.0 * len([v for v in violations if "TEST" in v.get("why", "")])
    out = {"information_gathering": round(25 * (0.7 * hist + 0.3 * exam), 1), "diagnostic_reasoning": dx_score,
           "tests_followup": round(10 * tests, 1), "safety": max(0.0, safety), "communication": round(min(comm, 15.0), 1),
           "efficiency": round(eff, 1)}
    out["total"] = round(sum(out.values()), 1)
    out["coverage"] = {"history": round(hist, 2), "exam": round(exam, 2), "tests_in_plan": round(tests, 2)}
    return out
