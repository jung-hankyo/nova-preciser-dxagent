"""Offline stand-ins for every LLM role, for plumbing tests and demos without an API key.

These are keyword heuristics, NOT medical reasoning. They let you verify that the
graph, the deterministic core, the terminal chat, and the benchmark runner all work
end to end (`dxagent chat --offline`, `pytest`). Real runs use LLM-backed models.
"""

from __future__ import annotations

import json
import re

from dxagent.knowledge.cards import CardStore, norm_key
from dxagent.llm import ScriptedLLM
from dxagent.schemas import (
    ClaimExtraction,
    ComplaintMatchOutput,
    ContrastDraft,
    CriticOutput,
    ExtractedComplaint,
    ExtractedFinding,
    ExtractedTest,
    ExtractionOutput,
    FreeQuestionOutput,
    HypothesisOutput,
    InterviewerOutput,
    JudgeOutput,
    PriorAssessment,
    PriorItem,
    ProposedHypothesis,
    ReportOutput,
)

NEGATIONS = ("no ", "not ", "denies", "deny", "never", "normal", "negative", "without", "none", "unremarkable", "absent")
STOP = {"with", "from", "that", "this", "above", "below", "brought", "when", "flat", "one", "side", "the", "and", "or", "on", "of", "to"}


def _block(text: str, start: str, end: str | None = None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    return text[i + len(start) : j if j >= 0 else None]


def _vocab_from_prompt(user: str) -> list[tuple[str, str]]:
    out = []
    for m in re.finditer(r"^- (?!test )([a-z0-9_]+): (.+?)(?: \([a-z_]+\))?$", user, flags=re.M):
        out.append((m.group(1), m.group(2)))
    return out


def _keywords(label: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", label.lower()) if len(w) > 3 and w not in STOP]


def _tests_from_prompt(user: str) -> list[tuple[str, str, list[str]]]:
    out = []
    for m in re.finditer(r"^- test ([a-z0-9_]+): (.+?) \| outcomes: (.+)$", user, flags=re.M):
        out.append((m.group(1), m.group(2).lower(), [o.strip() for o in m.group(3).split(",")]))
    return out


def offline_extract(system: str, user: str) -> ExtractionOutput:
    reply = _block(user, 'Reply received:\n"""\n', '\n"""').lower()
    findings = []
    tests = []
    for line in re.split(r"[;\n]|\.\s", reply):
        if ":" not in line:
            continue
        head, tail = line.split(":", 1)
        for tid, name, outcomes in _tests_from_prompt(user):
            if name in head.strip() or tid in head:
                best = max(outcomes, key=lambda o: (sum(w in tail for w in o.split("_")) / len(o.split("_")), -len(o)))
                if all(w in tail for w in best.split("_")):
                    tests.append(ExtractedTest(test_id=tid, outcome=best, confidence=0.95, evidence=line.strip()[:80]))
    reply_no_tests = "\n".join(l for l in re.split(r"[;\n]|\.\s", reply) if ":" not in l)
    opening = "opening statement" in user
    for sentence in re.split(r"[.;\n]", reply_no_tests):
        s = sentence.strip()
        if not s:
            continue
        negated = any(n in f" {s} " for n in NEGATIONS)
        for code, label in _vocab_from_prompt(user):
            kws = _keywords(label) or [code.split("_")[0]]
            hits = sum(1 for k in kws if k in s)
            if hits and hits / len(kws) >= 0.6:
                findings.append(
                    ExtractedFinding(code=code, label=label, present=not negated, confidence=0.8,
                                     source="history", evidence=s[:80],
                                     primary_complaint=opening and not negated and not any(f.primary_complaint for f in findings))
                )
    age = re.search(r"(\d{1,3})[- ]year", reply)
    sex = "female" if re.search(r"\b(woman|female|she)\b", reply) else "male" if re.search(r"\b(man|male|he)\b", reply) else None
    complaints = [ExtractedComplaint(label=f.label, quote=f.evidence, stated_reason=f.primary_complaint, finding_code=f.code)
                  for f in findings if opening and f.present]
    return ExtractionOutput(findings=findings, test_results=tests, age=int(age.group(1)) if age else None, sex=sex,
                            problem_representation=reply[:160], complaints=complaints)


def make_offline_llm(store: CardStore) -> ScriptedLLM:
    def hypotheses(system: str, user: str) -> HypothesisOutput:
        hyps = []
        for c in store.all_cards():
            bucket = "common" if c.harm.untreated_course != "lethal" else "uncommon"
            hyps.append(ProposedHypothesis(name=c.name, disease_id=c.id, prior_bucket=bucket, cant_miss=c.cant_miss, rationale="offline"))
        return HypothesisOutput(hypotheses=hyps[:10])

    def interviewer(system: str, user: str) -> InterviewerOutput:
        plan = _block(user, "Planned action(s):\n")
        items = re.findall(r"- \[([a-z_]+)\] (.+?) \(to learn", plan)
        if not items:
            return InterviewerOutput(message="Could you tell me more?")
        kind = items[0][0]
        names = [n for _, n in items]
        if kind == "history":
            return InterviewerOutput(message="Have you had any of the following: " + "; ".join(names) + "?")
        if kind == "exam":
            return InterviewerOutput(message="I'll examine you now. Please report: " + "; ".join(names) + ".")
        return InterviewerOutput(message="I'm ordering: " + "; ".join(names) + ". Please share the results.")

    def report(system: str, user: str) -> ReportOutput:
        m = re.search(r"^1\. (.+?) p=", user, flags=re.M)
        return ReportOutput(final_diagnosis=m.group(1) if m else "Undetermined", rationale="Offline report: top of the computed differential.")

    def judge(system: str, user: str) -> JudgeOutput:
        ref = norm_key(_block(user, "Reference diagnosis: ", "\n"))
        final = norm_key(_block(user, "Doctor's final diagnosis: ", "\n"))
        diff = [norm_key(x) for x in re.findall(r"^\d+\. (.+)$", user, flags=re.M)]
        rank = next((i + 1 for i, d in enumerate(diff) if d and (ref in d or d in ref)), None)
        return JudgeOutput(top1_correct=bool(ref) and (ref in final or final in ref), match_rank=rank, rationale="offline string match")

    def priors(system: str, user: str) -> PriorAssessment:
        items = []
        for did in re.findall(r"^- ([a-z0-9_]+): ", user, flags=re.M):
            c = store.get(did)
            items.append(PriorItem(disease_id=did, prior_bucket="uncommon" if c is None or c.cant_miss else "common"))
        return PriorAssessment(items=items)

    def no_draft(system: str, user: str):
        raise RuntimeError("offline mode does not elicit or extract knowledge")

    return ScriptedLLM(
        {
            ExtractionOutput: offline_extract,
            HypothesisOutput: hypotheses,
            CriticOutput: lambda s, u: CriticOutput(),
            InterviewerOutput: interviewer,
            FreeQuestionOutput: lambda s, u: FreeQuestionOutput(question="Is there anything else you have noticed?", target="open question"),
            ReportOutput: report,
            JudgeOutput: judge,
            ContrastDraft: no_draft,
            ClaimExtraction: no_draft,
            ComplaintMatchOutput: lambda s, u: ComplaintMatchOutput(same_as="", acuity=3),
            PriorAssessment: priors,
        },
        name="offline",
    )


def make_offline_simulator() -> ScriptedLLM:
    """Offline patient and records voices: echo the case file's history or the full record."""

    def text(system: str, user: str) -> str:
        record = _block(user, "Record:\n", "\n\nDoctor's request:")
        if record:
            data = json.loads(record)
            flat = []

            def walk(prefix: str, node: object) -> None:
                if isinstance(node, dict):
                    for k, v in node.items():
                        walk(f"{prefix}{k.replace('_', ' ')}: " if not prefix else f"{prefix}{k.replace('_', ' ')} ", v)
                else:
                    flat.append(f"{prefix}{node}")

            walk("", data)
            return "\n".join(flat)
        case = _block(user, "Your case file:\n", "\n\nConversation so far:")
        try:
            return json.loads(case).get("History", "I'm not sure.")
        except Exception:
            return "I'm not sure."

    return ScriptedLLM({"text": text}, name="offline-sim")
