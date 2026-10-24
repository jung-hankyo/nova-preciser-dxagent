"""AgentClinic benchmark adapter (MedQA-derived OSCE cases, MIT-licensed).

The simulated encounter is one conversational environment with two voices, routed by
the doctor's action type (plan section 5, "Action space"):

    history                      -> patient voice (LLM, lay language, only what is asked)
    exam / poc / lab / imaging   -> clinical records voice (LLM that reads ONLY the case's fixed
                                    exam and test tables; unlisted items come back "normal / unremarkable",
                                    matching AgentClinic's convention)

The doctor sees only the case objective at the start, exactly as in AgentClinic.
Data: https://github.com/SamuelSchmidgall/AgentClinic
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from dxagent.llm import LLMLike
from dxagent.prompts import MEASUREMENT_SYSTEM, PATIENT_SYSTEM, measurement_user, patient_user

BASE_URL = "https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/main/"
DATASETS = {
    "medqa": "agentclinic_medqa.jsonl",
    "medqa_extended": "agentclinic_medqa_extended.jsonl",
}


def download(dest: Path, names: list[str] | None = None) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in names or list(DATASETS):
        fname = DATASETS[name]
        path = dest / fname
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL + fname, path)
        out.append(path)
    return out


@dataclass
class Case:
    index: int
    objective: str
    patient: dict[str, Any]
    exam: dict[str, Any]
    tests: dict[str, Any]
    diagnosis: str

    @property
    def record(self) -> dict[str, Any]:
        return {"Physical_Examination_Findings": self.exam, "Test_Results": self.tests}


def load_cases(path: Path) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            osce = json.loads(line)["OSCE_Examination"]
            cases.append(
                Case(
                    index=i,
                    objective=osce.get("Objective_for_Doctor", ""),
                    patient=osce.get("Patient_Actor", {}),
                    exam=osce.get("Physical_Examination_Findings", {}),
                    tests=osce.get("Test_Results", {}),
                    diagnosis=osce.get("Correct_Diagnosis", ""),
                )
            )
    return cases


class SimulatedEncounter:
    """Answers the doctor's turns for one case. Keeps its own transcript for the patient voice."""

    def __init__(self, case: Case, patient_llm: LLMLike, measurement_llm: LLMLike, terms_llm: Optional[LLMLike] = None):
        self.case = case
        self.patient_llm = patient_llm
        self.measurement_llm = measurement_llm
        self.history: list[dict[str, str]] = []
        self.leak_flags = 0
        self.leaks: dict[str, list[str]] = {"patient": [], "records": []}  # v0.3 (G2): matched term per leak
        self.leak_terms = self._terms(terms_llm)

    def _terms(self, llm: Optional[LLMLike]) -> list[str]:
        """G2: the diagnosis plus the synonyms, abbreviations, and eponyms a simulator could leak it by."""
        terms = [self.case.diagnosis]
        if llm is not None:
            try:
                from dxagent.prompts import LEAK_TERMS_SYSTEM
                from dxagent.schemas import LeakTermsOutput

                terms += llm.structured(LeakTermsOutput, LEAK_TERMS_SYSTEM, f"Diagnosis: {self.case.diagnosis}").terms[:12]
            except Exception:
                pass
        out = []
        for t in terms:
            n = re.sub(r"[^a-z0-9 ]", " ", t.lower()).strip()
            n = re.sub(r"\s+", " ", n)
            if n and (len(n) >= 4 or n.isupper() or len(n.split()) > 1) and n not in out:
                out.append(n)
        return out

    def presentation(self) -> str:
        return self.case.objective

    def respond(self, doctor_message: str, action_type: str) -> str:
        if action_type == "history":
            reply = self.patient_llm.text(PATIENT_SYSTEM, patient_user(self.case.patient, self.history, doctor_message))
            role = "Patient"
        else:
            reply = self.measurement_llm.text(MEASUREMENT_SYSTEM, measurement_user(self.case.record, doctor_message))
            role = "Records"
        hit = self._leaks(reply)
        if hit:
            channel = "patient" if action_type == "history" else "records"
            self.leaks[channel].append(hit)
            if channel == "patient":
                self.leak_flags += 1
        self.history.append({"role": "Doctor", "text": doctor_message})
        self.history.append({"role": role, "text": reply})
        return reply

    def _leaks(self, text: str) -> str:
        """The first diagnosis term the text contains (word-bounded), or "". Patient leaks count as leak_flags;
        records mentions are reported separately (a requested confirmatory test may legitimately name it)."""
        t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())) + " "
        for term in self.leak_terms:
            if f" {term} " in t:
                return term
        return ""
