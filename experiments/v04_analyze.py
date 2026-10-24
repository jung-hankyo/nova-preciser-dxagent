"""Paired comparison of v0.4 arms against the v0.3 baselines on the same held-out cases (v04_sim.py, v03_sim.py).

    python experiments/v04_analyze.py BASE_DIR V04_DIR OUT.json

BASE_DIR holds v03_sim.py output run against the v0.3.1 code ("v0.3" = v0.3.0, "v0.3+P1" = v0.3.1); V04_DIR holds
v04_sim.py output. Rows are paired by (world seed, case index): both harnesses draw the same patient and answers for
that key. Differences are B - A with 95% paired-bootstrap CIs; top-1 also gets an exact McNemar p. Each comparison
is also split by whether the true disease is in the pre-built / covered set (the `known` flag of the A arm).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import random  # noqa: E402

from v03_sim import mcnemar, metrics  # noqa: E402


def paired(a: list[dict], b: list[dict], key, reps: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    """Mean of key(b) - key(a) over paired cases with a 95% bootstrap CI (same estimator as v03_sim.paired,
    resampling via random.choices for speed)."""
    diffs = [key(y) - key(x) for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(reps))
    return sum(diffs) / n, boots[int(0.025 * reps)], boots[int(0.975 * reps)]

KEYS = {
    "top1": lambda r: float(r["correct"]),
    "in_list": lambda r: float(r["in_list"]),
    "brier": lambda r: (r["top_prob"] - r["correct"]) ** 2,
    "logloss": lambda r: -math.log(max(r["p_truth"], 1e-3)),
    "turns": lambda r: r["turns"],
    "cost": lambda r: r["cost"],
    "llm_calls": lambda r: sum(r["calls"].values()),
}

PAIRS = [
    ("v0.3__default", "v0.3+P1__default"),
    ("v0.3__default", "v0.4__default"),
    ("v0.3+P1__default", "v0.4__default"),
    ("v0.3+P1__default", "v0.4+fallback__default"),
    ("v0.3+P1__default", "v0.4+cov100__default"),
    ("v0.3+P1__default", "v0.4+v3know__default"),
    ("v0.3+P1__KNOWN=1.0", "v0.4+v3know__KNOWN=1.0"),
    ("v0.3+P1__KNOWN=1.0", "v0.4+cov100__F_STATED=1.0_T_STATED=1.0"),
    ("v0.4__default", "v0.4+fallback__default"),
    ("v0.4+v3know__default", "v0.4__default"),
    ("v0.4+cov100__default", "v0.4+cov100__F_STATED=1.0_T_STATED=1.0"),
    ("v0.3+P1__default", "v0.4+cov100__F_STATED=0.85"),
    ("v0.3+P1__default", "v0.4+cov100__F_STATED=0.5"),
    ("v0.3+P1__default", "v0.4__F_STATED=0.85"),
]


def load(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {(r["seed"], r["case"]): r for r in rows}


def compare(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    xa, xb = [a[k] for k in keys], [b[k] for k in keys]
    out = {"n": len(keys), "A": metrics(xa), "B": metrics(xb), "diff": {}}
    for name, f in KEYS.items():
        m, lo, hi = paired(xa, xb, f)
        out["diff"][name] = [round(m, 4), round(lo, 4), round(hi, 4)]
    out["mcnemar_top1"] = mcnemar(xa, xb)
    for grp, sel in (("truth_covered", True), ("truth_uncovered", False)):
        ks = [k for k in keys if a[k].get("known") == sel]
        if ks:
            out[grp] = {"n": len(ks), "A_top1": round(sum(a[k]["correct"] for k in ks) / len(ks), 4),
                        "B_top1": round(sum(b[k]["correct"] for k in ks) / len(ks), 4)}
    return out


def main():
    base, v04, dest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    files = {p.stem: p for d in (base, v04) for p in d.glob("*.jsonl")}
    res = {}
    for a, b in PAIRS:
        if a in files and b in files:
            ra, rb = load(files[a]), load(files[b])
            if len(ra) == len(rb) == 5000:
                res[f"{a} -> {b}"] = compare(ra, rb)
    dest.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        d = v["diff"]
        cov = v.get("truth_covered", {}); unc = v.get("truth_uncovered", {})
        print(f"{k:66s} top1 {v['A']['top1']:.3f}->{v['B']['top1']:.3f} ({d['top1'][0]*100:+.1f} pp "
              f"[{d['top1'][1]*100:+.1f},{d['top1'][2]*100:+.1f}], p={v['mcnemar_top1']:.2g}) brier {d['brier'][0]:+.3f} "
              f"ll {d['logloss'][0]:+.3f} turns {d['turns'][0]:+.2f} cost {d['cost'][0]:+.0f} calls {d['llm_calls'][0]:+.1f} "
              f"| cov {cov.get('A_top1')}->{cov.get('B_top1')} unc {unc.get('A_top1')}->{unc.get('B_top1')}")


if __name__ == "__main__":
    main()
