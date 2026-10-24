"""Go / no-go before uploading the submission ZIP: every check the repository can run without the contest server.

    python -m sim.presubmit            (about two minutes)
    python -m sim.presubmit --fast     unit tests, playbooks, cases, the quick matrix, the ZIP check

1. the runtime and simulator tests (pytest; the core's own tests are not needed for the submission);
2. the 58 playbooks (schema, budgets, routing words);
3. the simulated cases (experiments/case_check.py);
4. the scenario matrix: every case under every scenario (--fast: the quick scenarios), every invariant held;
5. the ZIP: built, unpacked into an empty folder, run in an isolated interpreter against the simulated contest,
   with no file write and no network connection (sim/zipcheck.py).

Prints one line per check and GO only when all pass. Live runs on the contest server remain the final word on what
the simulator cannot know (the server's patient and examiner models, the Doctor model's own answers).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_nova.py", "tests/test_playbooks.py", "tests/test_visit.py", "tests/test_review.py",
         "tests/test_robust.py", "tests/test_sim.py", "tests/test_friendly.py"]


def _run(name: str, cmd: list[str], timeout: int = 3600) -> tuple[bool, str, float]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timed out", time.time() - t0
    lines = [x for x in (p.stdout + "\n" + p.stderr).splitlines() if x.strip()]
    return p.returncode == 0, (lines[-1] if lines else "")[:160], time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--workers", default="4")
    a = ap.parse_args(argv)
    py = sys.executable
    cases = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "experiments" / "data" / "nova_cpx").glob("*.yaml"))
    checks = [
        ("tests", [py, "-m", "pytest", "-q", *TESTS]),
        ("playbooks", [py, "-m", "dxagent.nova.playbooks"]),
        ("cases", [py, "experiments/case_check.py", *cases]),
        ("matrix", [py, "-m", "sim", "--scenarios", "quick" if a.fast else "all", "--workers", a.workers,
                    "--out", "runs/sim/presubmit"]),
        ("zip", [py, "-m", "sim.zipcheck"]),
    ]
    ok_all = True
    for name, cmd in checks:
        ok, last, secs = _run(name, cmd)
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<10} {secs:6.0f} s  {last}")
    print("\nGO: the ZIP is dist/submission.zip" if ok_all else "\nNO-GO: fix the failing checks above")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
