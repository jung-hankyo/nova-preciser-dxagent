"""Paired analysis for v03_sim.py results.

Two comparisons per fix:
    total        v0.3 (all fixes) minus v0.2 (none)
    marginal     v0.3 minus v0.3-without-this-fix: what the fix adds given all the others

Acceptance rule (pre-registered, same as earlier decision records): a fix is kept if it improves accuracy or
calibration with no metric significantly worse (95% paired bootstrap CI excluding zero in the bad direction).
Metrics: top-1 (McNemar p), truth in list at commit, Brier, ECE, log loss on the true disease, turns, cost,
LLM calls per case.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v03_sim import ece, mcnemar, metrics, paired  # noqa: E402

KEYS = [  # name, per-case function, higher is better
    ("top1", lambda r: float(r["correct"]), True),
    ("in_list", lambda r: float(r["in_list"]), True),
    ("brier", lambda r: (r["top_prob"] - r["correct"]) ** 2, False),
    ("logloss", lambda r: -__import__("math").log(max(r["p_truth"], 1e-3)), False),
    ("turns", lambda r: r["turns"], False),
    ("cost", lambda r: r["cost"], False),
    ("llm", lambda r: sum(r["calls"].values()), False),
]


def load(d: Path, arm: str, tag: str) -> list[dict]:
    p = d / f"{arm}__{tag}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def ece_diff(a, b, reps=1000, seed=3):
    import random
    rng = random.Random(seed)
    n = len(a)
    base = ece(b) - ece(a)
    boots = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(ece([b[i] for i in idx]) - ece([a[i] for i in idx]))
    boots.sort()
    return base, boots[int(0.025 * reps)], boots[int(0.975 * reps)]


def compare(a: list[dict], b: list[dict]) -> dict:
    assert [(x["seed"], x["case"]) for x in a] == [(y["seed"], y["case"]) for y in b], "unpaired"
    out = {}
    for name, f, hib in KEYS:
        m, lo, hi = paired(a, b, f)
        worse = (hi < 0) if hib else (lo > 0)
        better = (lo > 0) if hib else (hi < 0)
        out[name] = {"diff": m, "lo": lo, "hi": hi, "better": better, "worse": worse}
    m, lo, hi = ece_diff(a, b)
    out["ece"] = {"diff": m, "lo": lo, "hi": hi, "better": hi < 0, "worse": lo > 0}
    out["mcnemar_p"] = mcnemar(a, b)
    idx = [i for i, r in enumerate(a) if r["cant_miss"]]
    if idx:  # reported, not part of the acceptance rule (as in earlier records)
        m, lo, hi = paired([a[i] for i in idx], [b[i] for i in idx], lambda r: float(not r["correct"]))
        out["cm_missed"] = {"diff": m, "lo": lo, "hi": hi, "better": hi < 0, "worse": lo > 0, "n": len(idx)}
    acc_or_cal = out["top1"]["better"] or out["brier"]["better"] or out["logloss"]["better"] or out["ece"]["better"]
    bad = [k for k in ("top1", "in_list", "brier", "ece", "logloss", "turns", "cost") if out[k]["worse"]]
    out["verdict"] = "accept" if acc_or_cal and not bad else ("fail: " + ", ".join(bad) if bad else "no measurable gain")
    return out


def fmt(c: dict, k: str, scale: float = 1.0, digits: int = 3) -> str:
    x = c[k]
    star = "*" if (x["better"] or x["worse"]) else ""
    return f"{x['diff'] * scale:+.{digits}f}{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/results/v03_sim")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--fixes", nargs="+", default=["A1", "A2", "A3", "A4", "A5", "B2", "B3", "C2", "C3", "C3a", "C3b"])
    ap.add_argument("--pairs", nargs="*", default=[], help="extra comparisons BASE:ARM (ARM minus BASE)")
    ap.add_argument("--variant", default="", help="configuration suffix, e.g. final: compares v0.3+final and no-X+final")
    a = ap.parse_args()
    d = Path(a.dir)
    sfx = f"+{a.variant}" if a.variant else ""
    v02, v03 = load(d, "v0.2", a.tag), load(d, f"v0.3{sfx}", a.tag)
    res = {"marginal": {}}
    if v02:
        res.update({"v0.2": metrics(v02), "total": compare(v02, v03)})
    res["v0.3"] = metrics(v03)
    print(f"n = {len(v03)} paired cases, tag {a.tag}")
    for arm in ("v0.2", "v0.3"):
        if arm not in res:
            continue
        m = res[arm]
        print(f"{arm:5s} top1 {m['top1']:.3f} in_list {m['in_list']:.3f} brier {m['brier']:.4f} ece {m['ece']:.4f} "
              f"logloss {m['logloss']:.3f} turns {m['turns']:.2f} cost {m['cost']:.0f} llm {m['llm_calls']:.1f}")
    hdr = f"{'':26s} {'top1 pp':>9s} {'p':>7s} {'in_list pp':>10s} {'brier':>8s} {'ece':>8s} {'logloss':>8s} {'turns':>7s} {'cost':>7s} {'llm':>6s} {'cm pp':>7s}  verdict"
    print(hdr)

    def row(name, c):
        print(f"{name[:26]:26s} {fmt(c, 'top1', 100, 1):>9s} {c['mcnemar_p']:7.3f} {fmt(c, 'in_list', 100, 1):>10s} "
              f"{fmt(c, 'brier', 1, 4):>8s} {fmt(c, 'ece', 1, 4):>8s} {fmt(c, 'logloss', 1, 3):>8s} {fmt(c, 'turns', 1, 2):>7s} "
              f"{fmt(c, 'cost', 1, 0):>7s} {fmt(c, 'llm', 1, 1):>6s} {fmt(c, 'cm_missed', 100, 1) if 'cm_missed' in c else '':>7s}  {c['verdict']}")

    if "total" in res:
        row("v0.3-v0.2", res["total"])
    for fx in a.fixes:
        wo = load(d, f"no-{fx}{sfx}", a.tag)
        if not wo:
            continue
        c = compare(wo, v03)
        res["marginal"][fx] = c
        row(f"+{fx}", c)
    for pr in a.pairs:
        base, arm = pr.split(":")
        x, y = load(d, base, a.tag), load(d, arm, a.tag)
        if x and y:
            c = compare(x, y)
            res["marginal"][pr] = c
            row(f"{arm} vs {base}"[:40], c)
    (d / f"analysis__{a.tag}{sfx}.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
