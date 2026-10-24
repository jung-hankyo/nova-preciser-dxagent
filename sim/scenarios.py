"""Named scenarios: who the Doctor model, the patient and the server are in one simulated visit.

Each scenario fixes the Doctor stand-in (sim/doctor.py), the patient's behaviour, the server's faults (sim/server.py),
environment variables for the agent (NOVA_*), and whether a submission is still possible (a session that expires
mid-visit cannot be submitted; the agent must then fail quietly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sim.doctor import Doctor, FaultyDoctor, OkDoctor, OracleDoctor
from sim.server import Faults


@dataclass
class Scenario:
    name: str
    what: str
    doctor: Callable[[dict], Doctor] = lambda case: OracleDoctor(case)
    faults: dict[str, Any] = field(default_factory=dict)
    patient: str = ""
    env: dict[str, str] = field(default_factory=dict)
    expect_submission: bool = True
    real_sdk: bool = False  # run through the real nova_sdk.Client (skipped when nova_sdk.py is absent)

    def make_faults(self, seed: int) -> Faults:
        return Faults(seed=seed, **self.faults)


def _oracle(style: str = "json", latency: float = 25.0) -> Callable[[dict], Doctor]:
    return lambda case: OracleDoctor(case, style=style, latency=latency)


def _faulty(mode: str, style: str = "json") -> Callable[[dict], Doctor]:
    return lambda case: FaultyDoctor(OracleDoctor(case, style=style), mode)


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    # the model
    Scenario("offline", "no model at all (NOVA_OFFLINE=1): the deterministic floor; only the required one-word call",
             doctor=lambda case: OkDoctor(), env={"NOVA_OFFLINE": "1"}),
    Scenario("oracle", "well-formed answers from the key, bare JSON"),
    Scenario("oracle_fenced", "answers in ```json fences", doctor=_oracle("fenced")),
    Scenario("oracle_prose", "prose before and after the JSON", doctor=_oracle("prose")),
    Scenario("oracle_wrapped", "JSON wrapped in {\"FocusPlan\": {...}}", doctor=_oracle("wrapped")),
    Scenario("oracle_extra_keys", "extra keys in the JSON", doctor=_oracle("extra_keys")),
    Scenario("oracle_stringy", "booleans as strings", doctor=_oracle("stringy")),
    Scenario("oracle_trailing_comma", "trailing commas", doctor=_oracle("trailing_comma")),
    Scenario("model_garbage", "the model answers 'OK' to every call: nothing parses",
             doctor=lambda case: OkDoctor()),
    Scenario("model_empty", "every answer empty (PROVIDER_RESPONSE_INVALID)", doctor=_faulty("empty")),
    Scenario("model_length", "every answer cut by max_tokens before any content", doctor=_faulty("length")),
    Scenario("model_truncated", "JSON cut in the middle", doctor=_faulty("truncated")),
    Scenario("model_garbage_assess", "prose instead of JSON for the assessment only", doctor=_faulty("garbage:assess")),
    Scenario("model_huge", "50,000 characters of reasoning before the JSON", doctor=_faulty("huge")),
    Scenario("model_injection", "instructions to the grader inside the model's answers", doctor=_faulty("injection")),
    Scenario("model_overlong", "over-long, bundled and non-exam items from the model", doctor=_faulty("overlong")),
    Scenario("model_slow", "every call 200 s (past the hold: PROVIDER_UNAVAILABLE, then finished)", doctor=_faulty("slow:200")),
    Scenario("model_very_slow", "every call 420 s: time pressure on the whole visit", doctor=_faulty("slow:420")),
    Scenario("model_budget", "MODEL_TOKEN_BUDGET from the 2nd call on", doctor=_faulty("budget:2")),
    Scenario("old_server", "INVALID_COMPLETION_FIELDS whenever reasoning_effort is sent",
             faults={"reject_effort": True}),
    # the network and the server
    Scenario("net_flaky", "8% of requests lost before the server, 8% of responses lost after it",
             faults={"net_before": 0.08, "net_after": 0.08}),
    Scenario("patient_flaky", "PATIENT/EXAMINER_RESPONSE_INVALID on 12% of actions",
             faults={"patient_invalid": 0.12, "examiner_invalid": 0.12}),
    Scenario("slow_patient", "25 s per patient answer, 20 s per finding, 150 s per model call",
             doctor=_faulty("slow:150"), faults={"patient_latency": 25.0, "examiner_latency": 20.0}),
    Scenario("short_deadline", "a 9-minute session with 8 s per answer", faults={"minutes": 9.0, "patient_latency": 8.0,
                                                                                "examiner_latency": 8.0}),
    Scenario("session_expires", "the session expires at turn 20 (no submission possible)",
             faults={"expire_at_turn": 20}, expect_submission=False),
    Scenario("turns_used", "the session starts with 20 turns already used", faults={"turns_used_at_start": 20}),
    Scenario("diagnose_refused", "the first DIAGNOSE is refused INVALID_RECORD_FIELDS",
             faults={"diagnose_refuse": "INVALID_RECORD_FIELDS"}),
    Scenario("no_opening", "the start information has no opening statement", faults={"no_opening": True}),
    Scenario("no_vitals", "the start information has no vital signs", faults={"no_vitals": True}),
    # the patient
    Scenario("patient_terse", "short answers", patient="terse"),
    Scenario("patient_verbose", "long answers with worry", patient="verbose"),
    Scenario("patient_strict", "'A이나 B' questions get the one-at-a-time reply", patient="strict"),
    Scenario("patient_injection", "the patient's answers carry instructions to the doctor and the grader",
             patient="injection"),
    Scenario("patient_english", "some answers in English", patient="english"),
    # the real SDK
    Scenario("real_sdk", "the real nova_sdk.Client (socket layer simulated), well-formed answers", real_sdk=True),
    Scenario("real_sdk_flaky", "the real nova_sdk.Client with lost requests and slow calls", real_sdk=True,
             doctor=_faulty("slow:200"), faults={"net_before": 0.05, "net_after": 0.05}),
]}

QUICK = ("offline", "oracle", "oracle_prose", "model_garbage", "model_empty", "model_injection", "model_overlong",
         "model_slow", "net_flaky", "patient_flaky", "short_deadline", "session_expires", "diagnose_refused",
         "no_opening", "patient_strict", "patient_injection", "real_sdk")
