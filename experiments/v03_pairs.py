"""Paired comparisons BASE:ARM@TAG (ARM minus BASE) for any result directory, with the acceptance rule of
v03_analyze.py. Also prints veto bookkeeping when the records carry it.

    python experiments/v03_pairs.py DIR OUTNAME "v0.2:v0.3@default" "no-A1:v0.3@default" ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v03_analyze import compare, fmt, load  # noqa: E402
from v03_sim import metrics  # noqa: E402


def main():
    d, outname, specs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    res = {}
    print(f"{'comparison':58s} {'top1 pp':>8s} {'p':>6s} {'inlist':>7s} {'brier':>8s} {'ece':>8s} {'logloss':>8s} "
          f"{'turns':>6s} {'cost':>5s} {'llm':>5s} {'cm pp':>6s}  verdict")
    for sp in specs:
        pair, _, tag = sp.partition("@")
        base, arm = pair.split(":")
        a, b = load(d, base, tag or "default"), load(d, arm, tag or "default")
        if not a or not b:
            print(f"{sp}: missing")
            continue
        c = compare(a, b)
        c["base_metrics"], c["arm_metrics"] = metrics(a), metrics(b)
        for side, rs in (("base", a), ("arm", b)):
            if "vetoes" in rs[0]:
                n = len(rs)
                c[f"{side}_veto"] = {"vetoes_per_case": sum(r["vetoes"] for r in rs) / n,
                                     "honored_per_case": sum(r["vetoes_honored"] for r in rs) / n,
                                     "cases_with_veto": sum(1 for r in rs if r["vetoes"]) / n}
        res[sp] = c
        print(f"{sp[:58]:58s} {fmt(c, 'top1', 100, 1):>8s} {c['mcnemar_p']:6.3f} {fmt(c, 'in_list', 100, 1):>7s} "
              f"{fmt(c, 'brier', 1, 4):>8s} {fmt(c, 'ece', 1, 4):>8s} {fmt(c, 'logloss', 1, 3):>8s} {fmt(c, 'turns', 1, 2):>6s} "
              f"{fmt(c, 'cost', 1, 0):>5s} {fmt(c, 'llm', 1, 1):>5s} {fmt(c, 'cm_missed', 100, 1):>6s}  {c['verdict']}")
    (d / outname).write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
