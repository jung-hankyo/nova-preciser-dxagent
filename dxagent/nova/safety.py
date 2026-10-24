"""Deterministic checks on the plan before it is submitted (the contest's Safety axis, 15 points).

  allergy      a treatment from a drug class the patient reported an allergy to is removed (and noted);
               a contrast allergy annotates contrast studies; penicillin allergy flags cephalosporins
  can't-miss   every dangerous diagnosis still >= 3% that is not the primary one gets an excluding test in the plan
  disposition  unstable vital signs, or a primary diagnosis that is lethal within hours untreated, require admission or
               emergency observation
  pregnancy    a woman of reproductive age gets a pregnancy check before imaging with radiation
"""

from __future__ import annotations

import re
from typing import Any

DRUG_CLASSES: dict[str, dict[str, list[str]]] = {
    "페니실린계": {"allergen": ["페니실린", "penicillin", "아목시실린", "amoxicillin", "오구멘틴", "augmentin"],
                "drugs": ["페니실린", "penicillin", "아목시실린", "amoxicillin", "암피실린", "ampicillin", "피페라실린",
                          "piperacillin", "오구멘틴", "augmentin", "아목시클라브", "타조신", "tazocin", "나프실린", "nafcillin"]},
    "세팔로스포린계": {"allergen": ["세팔로스포린", "cephalosporin", "세프"],
                   "drugs": ["세프", "cef", "세팔로", "cephalo"]},
    "설폰아마이드계": {"allergen": ["설파", "sulfa", "박트림", "bactrim"],
                   "drugs": ["설파", "sulfa", "박트림", "bactrim", "트리메토프림", "trimethoprim", "sulfamethoxazole"]},
    "NSAID·아스피린": {"allergen": ["소염진통제", "nsaid", "이부프로펜", "ibuprofen", "부루펜", "나프록센", "아스피린", "aspirin"],
                    "drugs": ["nsaid", "소염진통제", "이부프로펜", "ibuprofen", "부루펜", "나프록센", "naproxen", "케토롤락",
                              "ketorolac", "디클로페낙", "diclofenac", "아스피린", "aspirin", "셀레콕시브", "celecoxib"]},
    "퀴놀론계": {"allergen": ["퀴놀론", "레보플록사신", "시프로플록사신", "levofloxacin", "ciprofloxacin"],
              "drugs": ["퀴놀론", "플록사신", "floxacin", "quinolone"]},
    "마크로라이드계": {"allergen": ["마크로라이드", "아지스로마이신", "클래리스로마이신", "azithromycin", "clarithromycin"],
                  "drugs": ["마크로라이드", "아지스로마이신", "azithromycin", "클래리스로마이신", "clarithromycin",
                            "에리스로마이신", "erythromycin", "macrolide"]},
    "조영제": {"allergen": ["조영제", "contrast"], "drugs": []},
}
_NO_ALLERGY = re.compile(r"(없어|없습니다|없었|없음|없다|없고|없네|없는데|아니요|아뇨|아니에요|모르겠|딱히|별로|전혀)")
_RADIATION = re.compile(r"(CT|X선|x-ray|엑스레이|조영|촬영|씨티)", re.I)
_ADMIT = re.compile(r"(입원|응급|중환자|ICU|전원|집중\s*관찰|모니터링)", re.I)


def allergy_classes(allergy_text: str) -> list[tuple[str, str]]:
    """[(class, allergen term)] the patient reported. A bare denial ("없어요") reports none."""
    t = (allergy_text or "").lower()
    if _NO_ALLERGY.search(t) and not re.search(r"있(어요|습니다|었|고|는데|어서)", t):
        return []  # "아니요, 없어요" / "페니실린 알레르기는 없어요"
    found = []
    for cls, d in DRUG_CLASSES.items():
        hit = next((a for a in d["allergen"] if a.lower() in t), None)
        if hit:
            found.append((cls, hit))
    if not found and ("알레르기" in t or "두드러기" in t) and not _NO_ALLERGY.search(t):
        found.append(("미상", "약물(종류 미상)"))
    return found


def _mentions(text: str, terms: list[str]) -> bool:
    low = (text or "").lower()
    return any(term.lower() in low for term in terms)


def lint_plan(plan: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """ctx: allergy_text, sex, age, unstable (reason or ""), lethal_primary (bool), confirm (test name or None),
    dangerous: [{"name": str, "tests": [str]}]."""
    p = {k: (list(v) if isinstance(v, list) else v) for k, v in plan.items()}
    for k in ("tests", "treatment", "follow_up", "education"):
        p[k] = [str(x).strip() for x in p.get(k) or [] if str(x).strip()]
    p["disposition"] = str(p.get("disposition") or "").strip()
    notes: list[str] = []

    # 1. allergies
    for cls, term in allergy_classes(ctx.get("allergy_text", "")):
        if cls == "조영제":
            p["tests"] = [t + " (조영제 알레르기: 전처치 또는 비조영 검사 고려)" if "조영" in t and "알레르기" not in t else t
                          for t in p["tests"]]
            continue
        if cls == "미상":
            notes.append("약물 알레르기 보고(종류 미상): 투약 전 확인")
            continue
        drugs = DRUG_CLASSES[cls]["drugs"]
        kept = [t for t in p["treatment"] if not _mentions(t, drugs)]
        if len(kept) < len(p["treatment"]):
            notes.append(f"{term} 알레르기: {cls} 약물 제외")
        p["treatment"] = kept
        if cls == "페니실린계" and any(_mentions(t, DRUG_CLASSES["세팔로스포린계"]["drugs"]) for t in p["treatment"]):
            notes.append("페니실린 알레르기: 세팔로스포린 투여 시 교차반응 주의")
    if notes and not any("알레르기" in t for t in p["treatment"]):
        p["treatment"].append("기록된 약물 알레르기를 고려해 대체 약제 선택")

    # 2. the confirmatory test of the primary diagnosis, then an excluding test for each can't-miss alternative
    tests_text = " ".join(p["tests"])
    confirm = ctx.get("confirm")
    if confirm and confirm not in tests_text:
        p["tests"].insert(0, f"{confirm} : 주진단 확인")
        tests_text += " " + confirm
    for d in ctx.get("dangerous", []):
        for t in d.get("tests", [])[:1]:
            if t and t not in tests_text:
                p["tests"].append(f"{t} : {d['name']} 배제")
                tests_text += " " + t

    # 3. disposition
    reason = ctx.get("unstable") or ("치료하지 않으면 수 시간 내 위험한 질환" if ctx.get("lethal_primary") else "")
    if reason and not _ADMIT.search(p["disposition"]):
        p["disposition"] = f"입원 또는 응급실 집중 관찰 ({reason})"
        notes.append(f"처치 결정 상향: {reason}")
    if not p["disposition"]:
        p["disposition"] = "외래 추적 관찰 (악화 시 즉시 재내원)"

    # 4. pregnancy before radiation
    age, sex = ctx.get("age"), ctx.get("sex")
    if sex == "female" and (age is None or 12 <= int(age) <= 55) and any(_RADIATION.search(t) for t in p["tests"]):
        if not any("임신" in t for t in p["tests"]):
            p["tests"].insert(0, "소변 임신 반응 검사 : 방사선 영상 검사 전 확인")

    for k in ("tests", "treatment", "follow_up", "education"):
        p[k] = list(dict.fromkeys(p[k]))
    return p, notes


def lint_lines(lines: list[str], allergy_text: str) -> list[str]:
    """Drop closing lines that name a drug from a class the patient is allergic to."""
    bad = [d for cls, _ in allergy_classes(allergy_text) if cls in DRUG_CLASSES for d in DRUG_CLASSES[cls]["drugs"]]
    return [ln for ln in lines if not _mentions(ln, bad)]
