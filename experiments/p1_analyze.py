"""Paired analysis for the v0.3.1 prior-context change (P1) and the leakage scenarios.

    python experiments/p1_analyze.py DIR --base v0.3 --arms v0.3+P1 v0.3+P1+prior_samples=1 --tags LEAK=0.0 LEAK=0.5 LEAK=1.0

For each tag (a --set scenario of v03_sim.py) and each arm: arm minus base on identical cases, with 95% paired
bootstrap CIs, McNemar p for top-1, and a bootstrap CI for the ECE difference. Also reports the prior-sensitivity
diagnostic (share of cases whose leader changes under flat priors) and, for P1 arms, the mean shift between the
proposer's bucket and the S-only assessor's bucket (mean absolute difference, in buckets): a leak detector.
Acceptance rule as for v0.3: better accuracy or calibration, no metric significantly worse.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v03_analyze import KEYS, ece_diff, load  # noqa: E402
from v03_sim import ece, mcnemar, paired  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--base", default="v0.3")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", default=["default"])
    a = ap.parse_args()
    d = Path(a.dir)
    for tag in a.tags:
        base = load(d, a.base, tag)
        if not base:
            continue
        n = len(base)
        sens = sum(bool(r["prior_sens"].get("changed_flat")) for r in base) / n
        print(f"\n== {tag}  (n = {n})  {a.base}: top1 {sum(r['correct'] for r in base) / n:.3f}  ECE {ece(base):.3f}  "
              f"leader changes under flat priors {sens:.1%}")
        for arm in a.arms:
            rs = load(d, arm, tag)
            if len(rs) != n:
                continue
            assert all(x["case"] == y["case"] and x["seed"] == y["seed"] for x, y in zip(base, rs))
            parts = []
            for name, fn, _ in KEYS:
                m, lo, hi = paired(base, rs, fn)
                parts.append(f"{name} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            e, elo, ehi = ece_diff(base, rs)
            shifts = [s for r in rs for s in r.get("prior_shift", [])]
            sh = (f"  proposer-vs-assessor |shift| {sum(shifts) / len(shifts):.2f} buckets, "
                  f">=2 in {sum(s >= 2 for s in shifts) / len(shifts):.1%}") if shifts else ""
            sens = sum(bool(r["prior_sens"].get("changed_flat")) for r in rs) / n
            print(f"  {arm}: top1 {sum(r['correct'] for r in rs) / n:.3f} (McNemar p {mcnemar(base, rs):.3g})  "
                  f"ECE {e:+.3f} [{elo:+.3f}, {ehi:+.3f}]  flat-prior leader change {sens:.1%}{sh}")
            print("    " + " | ".join(parts))


if __name__ == "__main__":
    main()
