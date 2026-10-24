"""Run a list of (arm, assumption overrides) jobs for v03_sim.py, resumable, within a wall-clock budget.

    python experiments/v03_batch.py JOBFILE --seeds 11 12 13 --cases 500 --out DIR --budget 270

JOBFILE: one job per line, "ARM [KEY=VALUE ...]". Finished jobs (output file present) are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v03_sim as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=270)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = dict(V.P)
    todo = [l.split() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    done = left = 0
    for job in todo:
        arm, sets = job[0], job[1:]
        tag = "_".join(sets) or "default"
        path = out / f"{arm}__{tag}.jsonl"
        if path.exists():
            done += 1
            continue
        if time.time() - t0 > a.budget * 0.55 and done + left > 0:  # an arm may take ~45% of the budget
            left += 1
            continue
        V.P.clear()
        V.P.update(base)
        for kv in sets:
            k, v = kv.split("=")
            V.P[k] = float(v)
        rs = V.run_arm(arm, a.seeds, a.cases, log=lambda *_: None)
        path.write_text("\n".join(json.dumps(r) for r in rs))
        m = V.metrics(rs)
        print(f"{arm:32s} {tag:28s} top1 {m['top1']:.3f} cm {m['cm_missed']:.3f} turns {m['turns']:.2f} cost {m['cost']:.0f} "
              f"llm {m['llm_calls']:.1f} ({time.time() - t0:.0f}s)", flush=True)
        done += 1
    print(f"done {done}/{len(todo)}, remaining {len(todo) - done}")


if __name__ == "__main__":
    main()
