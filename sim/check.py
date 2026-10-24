"""What a simulated visit must satisfy (invariants), and what it achieved (metrics).

Invariants are the submission's safety net, checked on the server's own log, never on the agent's account of itself:
the case was submitted (DIAGNOSE accepted, once) before the deadline, the Doctor model was called at least once,
no action was refused for being empty or too long, nothing was sent twice, the turn limit held, the note is clean
(no instruction to a grader, no raw model output such as JSON or English reasoning, no placeholder), the diagnosis is
one line and not "undetermined", consult() did not raise.

Metrics: turns, refused actions by code, model calls and tokens, the virtual duration, the history items and exam rows
of the case's key that the visit covered, and whether the submitted diagnosis names the key's (a word-overlap proxy;
the model-in-the-loop runs are judged properly).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

GENERIC = {
    "onset": r"(언제부터|언제 시작|얼마나 됐|얼마나 되었|언제였)",
    "character": r"(어떻게 아프|어떤 느낌|양상|어떤 식으로|어떻게 불편|어떻게 아파)",
    "location": r"(어디가|어느 부위|어디를|어디에|위치|어느 쪽)",
    "radiation": r"(퍼지|번지|뻗|다른 곳)",
    "severity": r"(얼마나 심|몇 점|점수|정도가|일상생활)",
    "timing": r"(계속|자주|얼마나 오래|지속|몇 번|하루에|간헐|얼마 동안)",
    "aggravating": r"(심해지|악화|더 아프|더 심|더 불편)",
    "relieving": r"(좋아지|나아지|완화|덜 아프|편해지|덜하)",
    "prior": r"(전에도|이전에|예전에)",
    "pmh": r"(앓는 병|앓고 있|지병|진단받|고혈압|당뇨)",
    "meds": r"(드시는 약|먹는 약|복용|약이 있|약을)",
    "surgery": r"(수술|입원)",
    "fh": r"가족",
    "smoking": r"(담배|흡연)",
    "alcohol": r"(술|음주)",
    "lmp": r"(마지막 생리|마지막 월경|생리는 언제)",
    "pregnancy": r"(임신|성관계)",
    "job": r"(직업|무슨 일|일을 하|하시는 일)",
}

INJECTION = re.compile(r"(채점|만점|점수를\s*(주|매)|평가자|프롬프트|시스템\s*(메시지|프롬프트)|지시(를|사항을)?\s*무시|"
                       r"이전\s*지시|ignore\s+(all|any|previous|prior)|system\s+prompt|grader|full\s+marks|jailbreak)", re.I)
ARTIFACT = re.compile(r"(```|\{\s*\"|\"\s*:\s*[\[{\"]|primary_diagnosis|text_ko|name_en|closing_lines|"
                      r"Let me think|I hope this helps|As an AI|step by step|\bnull\b|\bNone\b|TODO|placeholder|N/A)")
UNDETERMINED = re.compile(r"(진단\s*미정|undetermined|unknown|미상$)", re.I)


def key_items(key: dict) -> dict[str, str]:
    items: dict[str, str] = {}
    for h in key.get("history") or []:
        if isinstance(h, str):
            if h in GENERIC:
                items[h] = GENERIC[h]
        elif isinstance(h, dict):
            items.update({str(k): str(v) for k, v in h.items()})
    return items


def _words(text: str) -> set[str]:
    t = re.sub(r"[()]", " ", (text or "").lower())  # "급성 충수염 (Acute appendicitis)": the English in parentheses counts
    return {w for w in re.findall(r"[a-z]{3,}", t)} - {"acute", "chronic", "with", "due", "the", "and", "disease",
                                                         "disorder", "syndrome", "type", "left", "right", "primary"}


def dx_match(submitted: str, key_dx: str) -> Optional[bool]:
    """Word overlap between the key's English diagnosis and the submitted one (None without a key)."""
    want = _words(key_dx)
    if not want:
        return None
    got = _words(submitted)
    return len(want & got) >= max(1, (len(want) + 1) // 2)


def evaluate(case: dict, server: Any, outcome: dict, expect_submission: bool = True) -> dict[str, Any]:
    log = server.log
    sent = [e for e in log if e["kind"] in ("SAY", "EXAM", "TEST") and not e.get("duplicate")]
    accepted = [e for e in sent if not e.get("code")]
    refused = [e for e in sent if e.get("code")]
    codes: dict[str, int] = {}
    for e in refused:
        codes[e["code"]] = codes.get(e["code"], 0) + 1
    texts = [e["content"] for e in sent if e.get("code") not in ("PATIENT_RESPONSE_INVALID", "EXAMINER_RESPONSE_INVALID",
                                                                   "SESSION_EXPIRED", "TURN_LIMIT_REACHED", "SESSION_CLOSED")]
    dups = sorted({t for t in texts if texts.count(t) > 1})
    sub = server.submission
    note = (sub or {}).get("note") or {}
    note_text = "\n".join(v for s in note.values() for v in s.values())
    dx = (sub or {}).get("diagnosis", "")
    says = [e["content"] for e in accepted if e["kind"] == "SAY"]
    inv: dict[str, Any] = {
        "no_crash": outcome.get("exception") is None,
        "submitted": (sub is not None) if expect_submission else True,
        "before_deadline": (sub is None and not expect_submission) or (sub is not None and sub["t"] < server.f.minutes * 60),
        "chat_called": server.chat_calls >= 1 if expect_submission else True,
        "no_length_refusal": not any(c in codes for c in ("EMPTY_CONTENT", "CONTENT_TOO_LONG")),
        "no_duplicates": not dups,
        "turn_limit": server.turns <= server.max_turns and codes.get("TURN_LIMIT_REACHED", 0) <= 1,
        "note_clean": not INJECTION.search(note_text) and not ARTIFACT.search(note_text),
        "says_clean": not any(INJECTION.search(s) for s in says),
        "dx_one_line": (not sub) or ("\n" not in dx and 0 < len(dx.encode("utf-8")) <= 500),
        "dx_determined": (not sub) or not UNDETERMINED.search(dx.strip()),
        "note_has_core": (not sub) or all(note.get(s, {}).get(k, "").strip() for s, k in
                                          (("S", "cc"), ("S", "hpi"), ("A", "evidence"), ("P", "tests"))),
    }
    key = case.get("key") or {}
    items = key_items(key)
    asked = sorted(i for i, rx in items.items() if any(re.search(rx, s) for s in says))
    rows = {e.get("row") for e in accepted if e["kind"] == "EXAM" and e.get("row")}
    want_rows = set(key.get("exams") or [])
    chats = [e for e in log if e["kind"] == "CHAT"]
    metrics = {
        "turns": server.turns, "says": len(says), "exams": sum(1 for e in accepted if e["kind"] == "EXAM"),
        "refused": codes, "turn_refusals": sum(n for c, n in codes.items() if c in ("EXAM_NOT_SPECIFIC",
                                                                                    "EXAM_NOT_RECOGNIZED")),
        "duplicates": dups[:5], "chat_calls": server.chat_calls, "chat_refused": server.chat_refused,
        "chat_errors": sorted({e["code"] for e in chats if e.get("code")}),
        "in_tokens": server.in_tokens, "out_tokens": server.out_tokens,
        "seconds": (sub or {}).get("t"), "virtual_end": round(server.now() - server.t0, 1),
        "history": round(len(asked) / len(items), 3) if items else None,
        "missed": sorted(set(items) - set(asked)),
        "key_exams": round(len(want_rows & rows) / len(want_rows), 3) if want_rows else None,
        "missed_exams": sorted(want_rows - rows),
        "diagnosis": dx, "key_diagnosis": key.get("diagnosis", ""), "dx_match": dx_match(dx, key.get("diagnosis", "")),
        "note_bytes": len(json.dumps(note, ensure_ascii=False).encode("utf-8")) if note else 0,
        "closing": sum(1 for e in accepted if e["kind"] == "SAY" and "?" not in e["content"]),
    }
    failed = [k for k, v in inv.items() if not v]
    return {"case": case["case_id"], "ok": not failed, "failed": failed, "invariants": inv, "metrics": metrics,
            "exception": outcome.get("exception"), "result": outcome.get("result")}
