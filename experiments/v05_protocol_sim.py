"""v0.5: what the N.O.V.A action rules cost the unchanged core (paired, same held-out worlds as v04_sim.py).

Arms (knowledge fixed per --know: "cov100" = every disease has a compiled card, "known" = 75% coverage):
  shipped    v0.4.2 defaults: up to 3 history / exam / lab actions per turn, tests allowed, 60-turn cap
  single     one action per turn (contest SAY / EXAM / TEST rule), 50-turn cap
  prelim     single + no tests (the preliminary round has no TEST action): Settings.allowed_action_types

Scripted LLM roles; no API. Usage:
    python experiments/v05_protocol_sim.py --seeds 21 22 23 24 25 26 27 28 29 30 --cases 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v03_sim as V3  # noqa: E402
import v04_sim as V4  # noqa: E402


def settings_for(proto: str):
    s = V4.arm_settings("v0.4")
    if proto in ("single", "prelim"):
        s.max_batch_history = 1
        s.max_batch_tests = 1
        s.max_turns = 50
    if proto == "prelim":
        s.allowed_action_types = ("history", "exam")
    return s


def paired(a: list[dict], b: list[dict]) -> tuple[float, float, float]:
    A = {(r["seed"], r["case"]): r for r in a}
    d = [int(r["correct"]) - int(A[(r["seed"], r["case"])]["correct"]) for r in b]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    return round(m, 4), round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--know", nargs="+", default=["cov100", "known"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--out", default="experiments/results/v05_protocol")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for know in a.know:
        runs = {}
        for proto in ("shipped", "single", "prelim"):
            rs, t0 = [], time.time()
            for seed in a.seeds:
                W = V3.build_world(seed, int(V3.P["N_FAM"]))
                store = V4.make_store_v04(W, "all" if know == "cov100" else "known")
                s = settings_for(proto)
                for i in range(a.cases):
                    rs.append({**V4.run_case(W, store, V3.sample_case(W, i), "v0.4", s), "seed": seed})
            runs[proto] = rs
            m = V3.metrics(rs)
            low = [r for r in rs if r["top_prob"] < 0.8]
            row = {k: round(m[k], 4) for k in ("top1", "brier", "turns", "cost", "llm_calls", "cm_missed")}
            row.update({"n": len(rs), "in_list": round(sum(r["in_list"] for r in rs) / len(rs), 4),
                        "share_leader_below_0.8": round(len(low) / len(rs), 4),
                        "top1_when_leader_below_0.8": round(sum(r["correct"] for r in low) / max(len(low), 1), 4)})
            if proto != "shipped":
                row["paired_dtop1_vs_shipped"] = paired(runs["shipped"], rs)
            summary[f"{know}/{proto}"] = row
            print(f"{know:7s} {proto:8s} {row} ({time.time() - t0:.0f}s)", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
