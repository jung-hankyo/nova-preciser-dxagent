"""Offline stand-ins for every role of the competition runtime: tests and demos with no model endpoint.

Keyword heuristics, not medical reasoning. Korean answers are read with the registry's Korean terms and the yes/no
helper; every call is charged to the case budget so budget logic is exercised exactly as in a live run.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from pydantic import BaseModel

from dxagent.knowledge.cards import CardStore
from dxagent.nova.budget import Budget
from dxagent.nova.korean import parse_demographics, parse_vitals, yes_no
from dxagent.nova.llm import ROLE_BY_SCHEMA
from dxagent.nova.prompts import (Assessment, BankItem, ChartLine, ChartOutput, DxItem, FocusPlan, LeadDx, PhraseOutput,
                                  PlanOut, QuestionBank, ReviewPlan, TopicPlan)
from dxagent.offline import _block, make_offline_llm
from dxagent.schemas import CriticOutput, ExtractedFinding, ExtractionOutput, LabValue

_NEG = re.compile(r"(없|아니|않|안\s|아뇨|전혀)")


def _vocab(user: str) -> list[str]:
    return re.findall(r"^- (?!test )([a-z0-9_]+): ", user, flags=re.M)


def korean_extract(store: CardStore, user: str) -> ExtractionOutput:
    reply = _block(user, 'Reply received:\n"""\n', '\n"""')
    priority = re.findall(r"^- ([a-z0-9_]+): ", _block(user, "Codes the last action was meant to reveal", "\n\nTests"), flags=re.M)
    codes = list(dict.fromkeys(priority + _vocab(user)))
    pairs = re.findall(r"(?:Q|Exam): (.*?)\n(?:A|Result): (.*?)(?=\n(?:Q|Exam): |\Z)", reply, flags=re.S)
    if not pairs:
        pairs = [("", reply)]
    opening = "opening statement" in user
    findings: list[ExtractedFinding] = []
    seen: set[str] = set()
    for q, a in pairs:
        for code in codes:
            fd = store.finding_def(code)
            if fd is None or code in seen:
                continue
            terms = [t.replace(" ", "") for t in fd.ko]
            qa, aa = q.replace(" ", ""), a.replace(" ", "")
            in_a = any(t and t in aa for t in terms)
            in_q = any(t and t in qa for t in terms)
            targeted = code in priority and len(pairs) == 1
            if not (in_a or in_q or targeted):
                continue
            yn = yes_no(a)
            if yn is not None and (in_q or targeted):
                present, conf = yn
            elif in_a:
                clause = next((c for c in re.split(r"[.,!?]", a) if any(t in c.replace(" ", "") for t in terms)), a)
                present, conf = not bool(_NEG.search(clause)), 0.8
            else:
                continue
            seen.add(code)
            findings.append(ExtractedFinding(code=code, label=fd.label, present=present, confidence=conf,
                                             source=fd.source, evidence=(a or q)[:60],
                                             primary_complaint=opening and present and not any(f.primary_complaint for f in findings)))
    labs = [LabValue(analyte=x["analyte"], value=x["value"], unit=x["unit"]) for x in parse_vitals(reply)[0]] if opening else []
    age, sex = parse_demographics(reply) if opening else (None, None)
    return ExtractionOutput(findings=findings, lab_values=labs, age=age, sex=sex,
                            problem_representation=re.sub(r"\s+", " ", reply)[:160])


def offline_assessment(user: str) -> Assessment:
    """The first of the leading diagnoses given in the prompt (the core's or the focus call's), else "Undetermined"."""
    lead = _block(user, "Leading diagnoses so far (a hint):\n", "\n\n")
    names = [re.split(r"(?:: [0-9.]+| \(|: )", ln[2:], maxsplit=1)[0].strip()
             for ln in lead.splitlines() if ln.startswith("- ") and ln.strip() != "- (none)"]
    name = names[0] if names else "Undetermined"
    fixed = re.search(r"it is fixed: (.+?)\. Write", user)
    name = fixed.group(1) if fixed else name
    tests = [t.strip() for t in _block(user, "Computed test ranking (most informative first): ", "\n").split(",") if t.strip() and t.strip() != "none"]
    others = [n for n in names if n != name][:2]
    return Assessment(
        primary_diagnosis_en=name, primary_diagnosis_ko=name, evidence_ko=[],  # the note lists the visit's facts
        differentials=[DxItem(name_ko=o, name_en=o) for o in others],  # reasons and plan: the playbook's (soap.py)
        plan=PlanOut(tests=tests[:3]),
        closing_lines=["검사로 원인을 확인할게요.", "결과에 맞춰 치료할게요.", "증상이 심해지면 바로 오세요."],
        chief_complaint_ko="")


def offline_chart(user: str) -> ChartOutput:
    """Chart S/O fields from the rows of chart_user: plain 'question → answer' text, exam lines as reported.
    (build_soap uses this only when NOVA_OFFLINE_CHART=1; by default an offline run writes the deterministic note.)"""
    fields: dict[str, list[tuple[str, str, str]]] = {}
    current = ""
    for line in user.splitlines():
        h = re.match(r"^\[(S\.[a-z +.]+|O\.[a-z]+)\]$", line.strip())
        if h:
            current = h.group(1)
            continue
        m = re.match(r"^\[(e\d+)\] \(턴 \d+\) Q: (.*?) / A: (.*)$", line)
        if m and current:
            fields.setdefault(current, []).append((m.group(1), m.group(2), m.group(3)))

    def join(key: str) -> str:
        return "  ".join(f"{q} → {a}" for _, q, a in fields.get(key, []))

    hpi = join("S.cc + S.hpi")
    return ChartOutput(cc="", hpi=hpi, pmh=join("S.pmh"), fh=join("S.fh"), sh=join("S.sh"),
                       ros=join("S.ros"),
                       exam=[ChartLine(id=i, text=f"{q.rstrip('.')} : {a}") for i, q, a in fields.get("O.exam", [])],
                       tests=[ChartLine(id=i, text=f"{q.rstrip('.')} : {a}") for i, q, a in fields.get("O.tests", [])])


def offline_focus(user: str) -> FocusPlan:
    """No skips, no additions, no new items; the leading diagnoses are the presentation's usual causes (the
    deterministic visit: the whole plan up to the soft cap)."""
    usual = _block(user, "Usual causes: ", ". Can't-miss")
    names = [n.strip() for n in usual.split(",") if n.strip() and n.strip() != "none listed"][:3]
    return FocusPlan(leading=[LeadDx(name_en=n) for n in names])


def offline_review(user: str) -> ReviewPlan:
    """The time-out without a model: the hint's leading diagnoses, no new items, the planned order kept."""
    hint = _block(user, "Leading diagnoses at the focus call (a hint): ", "\n")
    names = [n.strip() for n in hint.split(";") if n.strip() and n.strip() != "none"][:3]
    ids = re.findall(r"^\[(m\d+)\]", user, flags=re.M)
    return ReviewPlan(leading=[LeadDx(name_en=n) for n in names], exam_order=ids)


def offline_bank(user: str) -> QuestionBank:
    items = [BankItem(kind="question", text_ko="다른 불편한 곳은 없으세요?", target="other symptoms", section="ros"),
             BankItem(kind="question", text_ko="최근에 체중이 줄었나요?", target="weight loss", section="ros"),
             BankItem(kind="exam", text_ko="복부 촉진을 하겠습니다.", target="abdominal tenderness"),
             BankItem(kind="exam", text_ko="피부 상태를 보겠습니다.", target="skin findings")]
    return QuestionBank(items=items)


def offline_phrase(user: str) -> PhraseOutput:
    kind = _block(user, "Kind: ", "\n").strip()
    return PhraseOutput(text_ko="해당 부위 진찰을 하겠습니다." if kind == "exam" else "그런 증상이 있으세요?")


class OfflineNovaLLM:
    def __init__(self, store: CardStore, budget: Optional[Budget] = None, fail_roles: tuple[str, ...] = ()):
        self.name = "offline"
        self.store = store
        self.budget = budget
        self.fail_roles = fail_roles  # tests: roles that raise, to exercise the fallbacks
        self.trace: list[dict[str, Any]] = []
        core = make_offline_llm(store)
        self.handlers = dict(core.handlers)
        self.handlers.update({
            ExtractionOutput: lambda s, u: korean_extract(store, u),
            CriticOutput: lambda s, u: CriticOutput(),
            Assessment: lambda s, u: offline_assessment(u),
            ChartOutput: lambda s, u: offline_chart(u) if os.getenv("NOVA_OFFLINE_CHART") == "1" else ChartOutput(),
            QuestionBank: lambda s, u: offline_bank(u),
            TopicPlan: lambda s, u: TopicPlan(),  # offline: the deterministic checklist only
            FocusPlan: lambda s, u: offline_focus(u),
            ReviewPlan: lambda s, u: offline_review(u),
            PhraseOutput: lambda s, u: offline_phrase(u),
        })

    def structured(self, schema: type[BaseModel], system: str, user: str, role: Optional[str] = None):
        role = role or ROLE_BY_SCHEMA.get(schema.__name__, "other")
        if self.budget is not None:
            self.budget.check(role)
        ok = role not in self.fail_roles
        if self.budget is not None:
            self.budget.charge(role, (len(system) + len(user)) // 3, 200, ok, 0.0)
        self.trace.append({"role": role, "ok": ok})
        if not ok:
            raise RuntimeError(f"offline: role {role} set to fail")
        return self.handlers[schema](system, user)

    def text(self, system: str, user: str, role: Optional[str] = None) -> str:
        return ""
