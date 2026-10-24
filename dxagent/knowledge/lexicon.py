"""Deterministic lexical matching between clinical text and finding codes (v0.3, A2 and A3).

Two uses:

    retrieval   reply text -> the registry codes it most likely mentions, so the extraction LLM is
                offered codes chosen by what the patient SAID, not by which diseases are on the list.
    matching    new code (+ label) -> existing codes that mean the same thing, so synonyms such as
                "fever", "pyrexia", and "fever_present" collapse onto one canonical code.

Text is reduced to CONCEPT tokens: multiword lay phrases first ("short of breath" -> dyspnea), then
single-word synonyms ("winded" -> dyspnea), then a light stemmer. No embeddings and no network.
Negation is deliberately ignored: polarity is the extraction LLM's job, retrieval only proposes codes.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

# Multiword lay phrases, matched on normalized text before tokenization. Longest first.
PHRASES: dict[str, str] = {
    "short of breath": "dyspnea", "shortness of breath": "dyspnea", "out of breath": "dyspnea",
    "hard to breathe": "dyspnea", "trouble breathing": "dyspnea", "difficulty breathing": "dyspnea",
    "cant catch my breath": "dyspnea", "can t catch my breath": "dyspnea", "catch my breath": "dyspnea",
    "lie down": "lying", "lying down": "lying", "lie flat": "lying", "lying flat": "lying",
    "passed out": "syncope", "blacked out": "syncope", "black out": "syncope",
    "throwing up": "vomiting", "threw up": "vomiting", "coughing up": "cough",
    "breathe in": "inspiration", "breathing in": "inspiration", "deep breath": "inspiration",
    "take a breath": "inspiration", "heart racing": "palpitation", "heart pounding": "palpitation",
    "skipped beat": "palpitation", "lost weight": "weightloss", "weight loss": "weightloss",
    "losing weight": "weightloss", "night sweats": "nightsweat", "sweat at night": "nightsweat",
    "wake up at night": "nocturnal", "woke up at night": "nocturnal", "waking up at night": "nocturnal",
    "at night": "nocturnal", "leaning forward": "forward", "sitting up": "upright",
    "heart attack": "myocardial", "blood clot": "thrombus", "long flight": "immobility",
    "long trip": "immobility", "bed rest": "immobility", "birth control": "estrogen",
    "blood pressure": "bloodpressure", "heart rate": "heartrate", "pulse rate": "heartrate",
    "breath sounds": "breathsound", "tearing pain": "tearing", "ripping pain": "tearing",
    "calf swelling": "calf edema", "leg swelling": "leg edema", "swollen leg": "leg edema",
    "swollen ankles": "ankle edema", "swollen feet": "leg edema",
    "high blood pressure": "hypertension", "neck veins": "jugular", "neck vein": "jugular",
    "sore throat": "viral", "runny nose": "viral", "common cold": "viral", "flu like": "viral",
    "one sided": "unilateral", "one side": "unilateral", "gasping for air": "dyspnea", "air hunger": "dyspnea",
    "heart sound": "heartsound", "heart sounds": "heartsound", "cold sweat": "diaphoresis",
    "chest wall": "chestwall", "coughed up": "cough", "brought up": "sputum",
}

# Single words -> concept. Lay terms and medical terms that mean the same thing share a concept.
SYNONYMS: dict[str, str] = {
    # breathing
    "breathless": "dyspnea", "breathlessness": "dyspnea", "winded": "dyspnea", "sob": "dyspnea",
    "dyspnoea": "dyspnea", "dyspneic": "dyspnea", "breathing": "dyspnea",
    "orthopnoea": "orthopnea", "pnd": "nocturnal", "paroxysmal": "nocturnal",
    "wheeze": "wheezing", "wheezy": "wheezing", "whistling": "wheezing", "wheezing": "wheezing", "wheezes": "wheezing",
    "tachypnea": "tachypnea", "tachypnoea": "tachypnea",
    "pleuritic": "inspiration", "inhale": "inspiration", "inhaling": "inspiration", "inspiratory": "inspiration",
    "crackle": "crackles", "rales": "crackles", "crepitations": "crackles", "crepitation": "crackles",
    # cough and sputum
    "coughing": "cough", "coughs": "cough", "phlegm": "sputum", "mucus": "sputum", "productive": "sputum",
    "hemoptysis": "blood", "haemoptysis": "blood", "bloody": "blood", "bleeding": "blood",
    # fever
    "fever": "fever", "feverish": "fever", "fevers": "fever", "pyrexia": "fever", "febrile": "fever",
    "temperature": "fever", "chills": "chills", "rigors": "chills", "shivering": "chills", "shaking": "chills",
    # pain
    "ache": "pain", "aching": "pain", "hurt": "pain", "hurts": "pain", "hurting": "pain", "sore": "pain",
    "discomfort": "pain", "painful": "pain", "tender": "tenderness", "tenderness": "tenderness",
    "pressure": "pressure", "tight": "pressure", "tightness": "pressure", "squeezing": "pressure",
    "crushing": "pressure", "heaviness": "pressure", "heavy": "pressure",
    "sharp": "sharp", "stabbing": "sharp", "knife": "sharp",
    "tearing": "tearing", "ripping": "tearing",
    "radiates": "radiation", "radiating": "radiation", "spreads": "radiation", "spreading": "radiation",
    "arm": "arm", "jaw": "jaw", "back": "back", "shoulder": "shoulder",
    # cardiovascular
    "palpitations": "palpitation", "racing": "palpitation", "pounding": "palpitation", "fluttering": "palpitation",
    "tachycardia": "heartrate", "tachycardic": "heartrate",
    "hypotension": "hypotension", "hypotensive": "hypotension", "shock": "hypotension",
    "hypertension": "hypertension", "hypertensive": "hypertension",
    "murmur": "murmur", "rub": "rub", "friction": "rub",
    "edema": "edema", "oedema": "edema", "swelling": "edema", "swollen": "edema", "puffy": "edema",
    "jvd": "jugular", "jvp": "jugular", "distension": "distension", "distended": "distension",
    "syncope": "syncope", "faint": "syncope", "fainted": "syncope", "fainting": "syncope", "collapse": "syncope",
    "dizzy": "dizziness", "dizziness": "dizziness", "lightheaded": "dizziness", "vertigo": "dizziness",
    "diaphoresis": "diaphoresis", "diaphoretic": "diaphoresis", "sweating": "diaphoresis", "sweaty": "diaphoresis",
    "sweats": "diaphoresis", "clammy": "diaphoresis",
    "pulse": "pulse heartrate", "pulses": "pulse", "deficit": "deficit", "asymmetric": "asymmetry", "unequal": "asymmetry",
    # gi and general
    "vomit": "vomiting", "vomited": "vomiting", "emesis": "vomiting",
    "nausea": "nausea", "nauseous": "nausea", "nauseated": "nausea", "queasy": "nausea",
    "tired": "fatigue", "fatigue": "fatigue", "exhausted": "fatigue", "weary": "fatigue", "lethargic": "fatigue",
    "lethargy": "fatigue", "weakness": "weakness", "weak": "weakness",
    # risk factors and context
    "smoker": "smoking", "smokes": "smoking", "cigarettes": "smoking", "tobacco": "smoking", "pack": "smoking",
    "surgery": "surgery", "operation": "surgery", "postoperative": "surgery",
    "immobilization": "immobility", "immobile": "immobility", "bedridden": "immobility",
    "dvt": "thrombus", "thrombosis": "thrombus", "clot": "thrombus", "embolism": "thrombus",
    "exertion": "exertion", "exertional": "exertion", "exercise": "exertion", "walking": "exertion",
    "stairs": "exertion", "effort": "exertion",
    "sudden": "sudden", "suddenly": "sudden", "abrupt": "sudden", "abruptly": "sudden",
    "gradual": "gradual", "gradually": "gradual", "slowly": "gradual",
    "worse": "worse", "worsens": "worse", "worsening": "worse", "aggravated": "worse",
    "better": "relief", "relieved": "relief", "improves": "relief", "eases": "relief",
    "recumbent": "lying", "supine": "lying", "flat": "lying",
    "leaning": "forward", "lean": "forward", "sit": "upright", "sitting": "upright", "propped": "lying",
    "pillows": "lying", "pillow": "lying",
    # added in v0.3 tuning: common lay and chart terms
    "walk": "exertion", "walked": "exertion", "uphill": "exertion", "climb": "exertion", "climbing": "exertion",
    "exert": "exertion", "stab": "sharp", "stabs": "sharp", "sweat": "diaphoresis", "drenched": "diaphoresis",
    "gasping": "dyspnea", "breathe": "dyspnea", "breath": "dyspnea",
    "emphysema": "copd", "copd": "copd", "bronchitis": "copd",
    "previous": "prior", "prior": "prior", "before": "prior", "past": "prior",
    "flight": "immobility", "travel": "immobility", "operated": "surgery",
    "cold": "viral", "flu": "viral", "influenza": "viral", "viral": "viral", "virus": "viral", "uri": "viral",
    "smoke": "smoking", "smoked": "smoking", "ex": "prior",
    "pitting": "edema", "shin": "leg", "shins": "leg", "ankle": "leg", "ankles": "leg", "feet": "leg", "foot": "leg",
    "legs": "leg", "calves": "calf", "both": "bilateral", "bilaterally": "bilateral", "bilateral": "bilateral",
    "unilateral": "unilateral", "scratchy": "rub", "gallop": "s3", "third": "s3",
    "bases": "lung", "base": "lung", "lungs": "lung",
    "decreased": "decreased", "diminished": "decreased", "reduced": "decreased", "absent": "decreased",
    "hyperresonance": "hyperresonance", "hyperresonant": "hyperresonance",
    "dull": "dullness", "dullness": "dullness",
}

# v0.3.1 (step 5): lay words for risk factors and exposures, so context facts are offered their codes.
SYNONYMS.update({'cigarette': 'smok', 'bedbound': 'immobility'})

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "without", "my", "i", "me",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "have", "has", "had", "do", "does",
    "did", "no", "not", "any", "some", "very", "really", "quite", "bit", "little", "lot", "when", "if", "but",
    "so", "just", "also", "get", "got", "gets", "feel", "feels", "felt", "feeling", "been", "since", "about",
    "from", "by", "as", "than", "then", "there", "they", "he", "she", "you", "your", "we", "our", "can", "could",
    "would", "should", "will", "into", "over", "under", "up", "down", "out", "off", "more", "most", "other",
    "present", "positive", "negative", "finding", "sign", "symptom", "history", "patient", "reports", "denies",
    "yes", "doctor", "day", "days", "week", "weeks", "ago", "now", "today", "time", "times",
}


def _stem(w: str) -> str:
    for suf, rep in (("ies", "y"), ("ing", ""), ("ness", ""), ("edly", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 4 and not w.endswith("ss"):
            return w[: -len(suf)] + rep
    return w


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower().replace("_", " "))).strip()


HANGUL = re.compile(r"[\uac00-\ud7a3]")


def _compact(text: str) -> str:
    """Lowercase, whitespace removed: the form Korean terms are matched in."""
    return re.sub(r"\s+", "", text.lower())


def concepts(text: str) -> list[str]:
    """Concept tokens of a text, in order (duplicates kept for the caller to set() if needed)."""
    t = f" {normalize(text)} "
    out: list[str] = []
    for ph in sorted(PHRASES, key=len, reverse=True):
        key = f" {ph} "
        while key in t:
            out.extend(PHRASES[ph].split())
            t = t.replace(key, " ", 1)
    for w in t.split():
        if w in STOPWORDS or len(w) < 2:
            continue
        c = SYNONYMS.get(w) or SYNONYMS.get(_stem(w)) or _stem(w)
        out.extend(c.split())
    return out


class Lexicon:
    """Concept index over the finding registry: code tokens + label (+ aliases) per code."""

    def __init__(self, entries: Iterable[tuple[str, str, Iterable[str]]]):
        self.codes: dict[str, set[str]] = {}
        self.ko: dict[str, list[str]] = {}  # v0.5: Korean terms, matched as substrings (Korean is agglutinative)
        for code, label, aliases in entries:
            cs = set(concepts(code)) | set(concepts(label))
            for a in aliases:
                if HANGUL.search(a):
                    self.ko.setdefault(code, []).append(_compact(a))
                    continue
                cs |= set(concepts(a))
            if cs:
                self.codes[code] = cs
        df: dict[str, int] = {}
        for cs in self.codes.values():
            for c in cs:
                df[c] = df.get(c, 0) + 1
        n = max(len(self.codes), 1)
        self.idf = {c: math.log(1 + n / k) for c, k in df.items()}

    def _w(self, cs: set[str]) -> float:
        return sum(self.idf.get(c, math.log(1 + len(self.codes))) for c in cs)

    def retrieve(self, text: str, k: int = 60, min_cover: float = 0.2) -> list[tuple[str, float]]:
        """Codes whose concepts are mostly covered by the text, best first (IDF-weighted coverage)."""
        tc = set(concepts(text))
        best: dict[str, float] = {}
        for code, cs in self.codes.items() if tc else ():
            hit = cs & tc
            if not hit:
                continue
            cover = self._w(hit) / self._w(cs)
            if cover >= min_cover:
                best[code] = cover
        if self.ko and HANGUL.search(text):  # v0.5: a Korean term found anywhere in the reply offers its code
            flat = _compact(text)
            for code, terms in self.ko.items():
                if any(t and t in flat for t in terms):
                    best[code] = 1.0
        scored = sorted(best.items(), key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def similar(self, code: str, label: str, k: int = 3, exclude: Iterable[str] = ()) -> list[tuple[str, float]]:
        """Existing codes most similar to a (code, label) pair: IDF-weighted Dice over concept sets."""
        q = set(concepts(code)) | set(concepts(label))
        if not q:
            return []
        ex = set(exclude)
        wq = self._w(q)
        scored = []
        for other, cs in self.codes.items():
            if other in ex:
                continue
            inter = q & cs
            if not inter:
                continue
            scored.append((other, 2 * self._w(inter) / (wq + self._w(cs))))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
