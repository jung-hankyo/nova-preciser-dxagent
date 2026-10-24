"""The submission ZIP as the server will run it: unpacked into an empty folder, imported from there in an isolated
interpreter (python -I: no environment variables, no user site, not the repository's own dxagent), played against
the simulated contest, with every file write and network connection outside the run recorded.

    python -m sim.zipcheck                       build dist/submission.zip, unpack, check
    python -m sim.zipcheck --zip path/to.zip     check a ZIP built elsewhere (e.g. the CI artifact)

Checks: the ZIP's own rules (dxagent.nova.package.verify), run.py and requirements.txt at the root, prepare() returns
within its 10 minutes and never raises, every simulated visit holds its invariants, and the agent writes nothing and
connects nowhere (the server's file system is read-only and offline).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = "L01,L05,L11,L14,L17,M03,M08,M16,M18,S01,S03,S06"
DEFAULT_SCENARIOS = "offline,oracle,oracle_prose,model_garbage,model_slow,net_flaky,short_deadline,diagnose_refused"

CHILD = r'''
import json, os, sys, time
unpacked, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
argv = json.loads(sys.argv[4])
events = []
allowed = (os.path.realpath(out), "/dev/null")
def hook(event, args):
    if event == "open" and args and isinstance(args[0], str):
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if (any(c in mode for c in "wax+") or flags & 0o3) and not os.path.realpath(args[0]).startswith(allowed):
            events.append(["write", args[0]])
    elif event == "socket.connect":
        events.append(["connect", str(args[1])[:80]])
sys.addaudithook(hook)
sys.path[:0] = [unpacked, repo]
import dxagent
assert os.path.realpath(dxagent.__file__).startswith(os.path.realpath(unpacked)), dxagent.__file__
t0 = time.monotonic()
spec = __import__("importlib.util").util.spec_from_file_location("run_from_zip", os.path.join(unpacked, "run.py"))
run = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(run)
state = run.prepare()
prep = round(time.monotonic() - t0, 2)
from sim.runner import main
rc = main(argv + ["--agent-root", unpacked, "--workers", "1", "--out", out])
json.dump({"rc": rc, "prepare_s": prep, "prepared": bool(state.get("_dxagent_prepared")), "events": events[:50],
           "dxagent": dxagent.__file__}, open(os.path.join(out, "zipcheck.json"), "w"))
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="", help="a built ZIP (default: build dist/submission.zip now)")
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from dxagent.nova.package import build, verify

    zpath = Path(a.zip) if a.zip else ROOT / "dist" / "submission.zip"
    info = verify(zpath) if a.zip else build(zpath)
    print(f"{zpath}: {info['files']} files, {info['size_mb']} MB ({'OK' if info['ok'] else 'NOT OK'})")
    for p in info["problems"] + info.get("warnings", []):
        print(f"  - {p}")
    work = Path(tempfile.mkdtemp(prefix="nova_zipcheck_"))
    unpacked, out = work / "app", work / "out"
    out.mkdir()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(unpacked)
    missing = [n for n in ("run.py", "requirements.txt") if not (unpacked / n).exists()]
    if missing:
        print(f"NOT OK: missing at the ZIP root: {missing}")
        return 1
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work), "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    sim_argv = ["--scenarios", a.scenarios, "--ids", a.ids]
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(unpacked), str(ROOT), str(out),
                           json.dumps(sim_argv)], cwd=str(work), env=env, capture_output=True, text=True,
                          timeout=3600)
    tail = "\n".join(proc.stdout.splitlines()[-30:])
    print(tail)
    res = json.loads((out / "zipcheck.json").read_text()) if (out / "zipcheck.json").exists() else {}
    if not res:
        print(proc.stderr[-3000:])
        print("NOT OK: the check did not complete")
        return 1
    ok = info["ok"] and res["rc"] == 0 and not res["events"] and res["prepared"] and res["prepare_s"] < 600
    print(f"\nprepare(): {res['prepare_s']} s, prepared={res['prepared']}; dxagent imported from {res['dxagent']}")
    print(f"writes or connections: {res['events'] or 'none'}")
    print(f"ZIP check: {'OK' if ok else 'NOT OK'} (results in {out})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
