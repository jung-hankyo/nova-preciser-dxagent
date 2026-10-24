"""Baseline: one LLM plays the doctor with no knowledge layer (plan section 7, baseline 1).

AgentClinic-style protocol: each turn the model emits QUESTION / EXAM / TEST / DIAGNOSIS READY.
It runs against the same simulated encounter and judge as the knowledge-driven agent.
"""

from __future__ import annotations

import re
from typing import Any

from dxagent.config import DEFAULT_ACTION_COST
from dxagent.env.agentclinic import SimulatedEncounter
from dxagent.llm import LLMLike
from dxagent.prompts import BASELINE_DOCTOR_SYSTEM

PREFIX = re.compile(r"^\s*(QUESTION|EXAM|TEST|DIAGNOSIS READY)\s*:\s*(.*)$", re.I | re.S)


def parse_turn(text: str) -> tuple[str, str]:
    m = PREFIX.match(text.strip())
    if not m:
        return "QUESTION", text.strip()
    return m.group(1).upper(), m.group(2).strip()


def parse_diagnosis(body: str) -> tuple[str, list[str]]:
    parts = re.split(r"\|\s*DIFFERENTIAL\s*:", body, flags=re.I)
    final = parts[0].strip().rstrip(".")
    diff = [final]
    if len(parts) > 1:
        diff += [d.strip().rstrip(".") for d in parts[1].split(";") if d.strip()]
    return final, diff


def run_baseline_case(doctor: LLMLike, enc: SimulatedEncounter, max_turns: int) -> dict[str, Any]:
    transcript: list[dict[str, Any]] = [{"role": "referral", "text": enc.presentation(), "turn": 0}]
    cost = 0.0
    final, diff = "", []
    for turn in range(1, max_turns + 1):
        convo = "\n".join(f"{t['role']}: {t['text']}" for t in transcript)
        if turn == max_turns:
            convo += "\n\nThis is your final turn. You must reply with DIAGNOSIS READY."
        out = doctor.text(BASELINE_DOCTOR_SYSTEM.format(max_turns=max_turns, turn=turn - 1), convo)
        kind, body = parse_turn(out)
        transcript.append({"role": "doctor", "text": out, "turn": turn})
        if kind == "DIAGNOSIS READY":
            final, diff = parse_diagnosis(body)
            return {"final_diagnosis": final, "differential": diff, "turns": turn, "cost_spent": cost, "transcript": transcript, "commit_reason": "model chose to commit"}
        action_type = {"QUESTION": "history", "EXAM": "exam", "TEST": "lab"}[kind]
        cost += DEFAULT_ACTION_COST[action_type]
        reply = enc.respond(body, action_type)
        transcript.append({"role": "patient" if action_type == "history" else "clinical_team", "text": reply, "turn": turn})
    return {"final_diagnosis": final or "Undetermined", "differential": diff, "turns": max_turns, "cost_spent": cost, "transcript": transcript, "commit_reason": "turn budget"}
