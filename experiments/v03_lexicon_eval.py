"""v0.3 A2/A3 check: does lexical retrieval offer the right code for patient- and chart-style phrasing?

Dev set (paraphrases_seed.jsonl, 87 phrases) was used to tune the synonym table.
Test set (paraphrases_seed_test.jsonl, 58 phrases) was written after tuning was frozen and is reported once.

A2 relevance: under v0.2 a finding outside the current hypotheses' cards is never offered to the extraction
LLM (0%). Under v0.3 it is offered whenever retrieval finds it: the recall below.
A3 relevance: code-to-code similarity for synonym codes, auto-merge threshold 0.85.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxagent.config import Settings  # noqa: E402
from dxagent.knowledge.cards import CardStore  # noqa: E402

D = Path(__file__).resolve().parent / "data"


def run(name: str, k: int = 60) -> dict:
    lex = CardStore(Settings().registry_dir, None, card_dirs=[Settings().fixture_dir]).lexicon()
    rows = [json.loads(l) for l in open(D / name)]
    hit, top1, sizes, miss = 0, 0, [], []
    for r in rows:
        codes = [c for c, _ in lex.retrieve(r["text"], k=k)]
        sizes.append(len(codes))
        if codes and codes[0] == r["code"]:
            top1 += 1
        if r["code"] in codes:
            hit += 1
        else:
            miss.append({"code": r["code"], "text": r["text"], "got": codes[:3]})
    n = len(rows)
    return {"set": name, "n": n, "recall": round(hit / n, 3), "top1": round(top1 / n, 3),
            "mean_offered": round(sum(sizes) / n, 2), "misses": miss}


def synonym_codes() -> dict:
    """A3: synonym code pairs a drafter might plausibly produce, and unrelated pairs that must NOT merge."""
    lex = CardStore(Settings().registry_dir, None, card_dirs=[Settings().fixture_dir]).lexicon()
    same = [("pyrexia", "pyrexia", "fever"), ("fever_present", "fever present", "fever"),
            ("dyspnea_lying_flat", "dyspnoea lying flat", "orthopnea"),
            ("coughing_blood", "coughing blood", "hemoptysis"), ("sweating", "sweating", "diaphoresis"),
            ("raised_jvp", "raised JVP", "jvd"), ("calf_swelling_unilateral", "unilateral calf swelling", "unilateral_leg_swelling"),
            ("pleuritic_pain", "pleuritic chest pain", "chest_pain_pleuritic"), ("wheezing", "wheezing", "wheeze"),
            ("smoker", "current or former smoker", "smoking_history")]
    diff = [("chest_pain_at_rest", "chest pain at rest", "chest_pain_exertional"),
            ("bilateral_crackles", "bilateral crackles", "bilateral_leg_edema"),
            ("hypotension", "low blood pressure", "hypertension_history"),
            ("bradycardia", "heart rate below 60", "tachycardia"),
            ("dry_cough", "dry cough", "productive_cough")]

    def best(code, label):
        s = lex.similar(code, label, k=1)
        return s[0] if s else ("", 0.0)

    out = {"same": [], "different": []}
    for code, label, target in same:
        b = best(code, label)
        out["same"].append({"new": code, "target": target, "best": b[0], "score": round(b[1], 3)})
    for code, label, other in diff:
        b = best(code, label)
        out["different"].append({"new": code, "must_not_merge_into": other, "best": b[0], "score": round(b[1], 3)})
    return out


if __name__ == "__main__":
    res = {"dev": run("paraphrases_seed.jsonl"), "test": run("paraphrases_seed_test.jsonl"), "a3": synonym_codes()}
    out = Path(__file__).resolve().parent / "results" / "v03_lexicon.json"
    out.write_text(json.dumps(res, indent=2))
    for k in ("dev", "test"):
        r = res[k]
        print(f"{k:4s} n={r['n']:3d} recall={r['recall']:.2f} top1={r['top1']:.2f} mean_offered={r['mean_offered']}")
    for m in res["test"]["misses"]:
        print("  test miss:", m)
    print("A3 synonyms (auto-merge >= 0.85, confirm >= 0.60):")
    for r in res["a3"]["same"]:
        print(f"  same  {r['new']:26s} -> {r['best']:30s} {r['score']:.2f} {'OK' if r['best']==r['target'] else 'WRONG TARGET'}")
    for r in res["a3"]["different"]:
        print(f"  diff  {r['new']:26s} -> {r['best']:30s} {r['score']:.2f}")
