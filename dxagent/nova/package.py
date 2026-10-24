"""Build the contest submission ZIP.

    python -m dxagent.nova.package --out dist/submission.zip [--sdk path/to/nova_sdk.py] [--with-kb] [--no-fixtures]

Root: run.py, requirements.txt (the repository's pinned file: PyPI name==version, prebuilt wheels only), nova_sdk.py
(from the starter package; taken from --sdk or the repository root, and the build is marked NOT OK without it), the
dxagent package (no tests, experiments, caches), NOTICE.md, and kb/ with --with-kb.

Checks before upload, as the server rejects: ZIP > 50,000,000 bytes, > 500,000,000 bytes unpacked, > 5,000 files,
symbolic links, absolute or parent-relative paths, duplicate paths; requirements lines that are not name==version.
The guideline library is NOT packed: the server mounts it at NOVA_GUIDELINES_DIR, and its licences forbid
redistributing changed versions.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

from dxagent.config import PROJECT_DIR

MAX_ZIP_BYTES = 50_000_000
MAX_UNZIPPED_BYTES = 500_000_000
MAX_FILES = 5_000
_REQ_LINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+!_-]+$")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "runs", "cache", "sources", ".extract_cache", ".passage_cache"}
NOTICE = """# NOTICE

dxagent, an interviewing-doctor agent for the N.O.V.A diagnostic agent competition.

Bundled knowledge:
- dxagent/knowledge/registry: curated finding and test registries written for this project.
- dxagent/knowledge/fixtures/chest_pain_demo: 8 illustrative disease cards; frequencies approximated from textbook
  and JAMA Rational Clinical Examination data, not individually sourced.
- kb/ (if included): compiled from the sources listed in kb/build.json (MONDO CC BY 4.0, HPO and its annotations under
  the HPO licence, and the documents named there with their licences).
External models: none bundled. The preliminary round calls only the contest's hosted model through the
contest SDK (client.chat).
Guideline library: read at run time from NOVA_GUIDELINES_DIR (contest copy, competition use only); no document or
derived text is included in this ZIP.
"""


def check_requirements(text: str) -> list[str]:
    """Lines the server would not install: anything but name==version (comments and blank lines are fine)."""
    bad = []
    for line in text.splitlines():
        t = line.split("#", 1)[0].strip()
        if t and not _REQ_LINE.match(t):
            bad.append(t)
    return bad


def _files(root: Path, include_kb: bool, include_fixtures: bool, sdk: Optional[Path] = None) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for name in ("run.py",):
        if (root / name).exists():
            out.append((root / name, name))
    sdk = sdk or (root / "nova_sdk.py")
    if sdk.exists():
        out.append((sdk, "nova_sdk.py"))
    pkg = root / "dxagent"
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(root)
        if p.is_symlink() or p.is_dir() or any(part in SKIP_DIRS for part in rel.parts) or p.suffix in (".pyc", ".pyo"):
            continue
        if not include_fixtures and "fixtures" in rel.parts:
            continue
        out.append((p, rel.as_posix()))
    if include_kb and (root / "kb").exists():
        for p in sorted((root / "kb").rglob("*")):
            rel = p.relative_to(root)
            if p.is_file() and not p.is_symlink() and not any(part in SKIP_DIRS for part in rel.parts):
                out.append((p, rel.as_posix()))
    return out


def verify(path: Path) -> dict:
    """The server's ZIP checks, applied to a built file."""
    problems: list[str] = []
    size = path.stat().st_size
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        unpacked = sum(i.file_size for i in infos)
        for i in infos:
            n = i.filename
            if n.startswith("/") or ".." in Path(n).parts or "\\" in n:
                problems.append(f"unsafe path {n}")
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                problems.append(f"symbolic link {n}")
            if i.flag_bits & 0x1:
                problems.append(f"encrypted {n}")
        req = z.read("requirements.txt").decode("utf-8") if "requirements.txt" in names else ""
    if size > MAX_ZIP_BYTES:
        problems.append(f"ZIP {size:,} bytes > {MAX_ZIP_BYTES:,}")
    if unpacked > MAX_UNZIPPED_BYTES:
        problems.append(f"unpacked {unpacked:,} bytes > {MAX_UNZIPPED_BYTES:,}")
    if len(names) > MAX_FILES:
        problems.append(f"{len(names)} files > {MAX_FILES}")
    if len(set(names)) != len(names):
        problems.append("duplicate paths")
    for need in ("run.py", "requirements.txt", "nova_sdk.py"):
        if need not in names:
            problems.append(f"missing {need} at the root" + (" (copy it from the starter package)" if need == "nova_sdk.py" else ""))
    problems += [f"requirements line not name==version: {b}" for b in check_requirements(req)]
    if "nova_sdk.py" in names:
        with zipfile.ZipFile(path) as z:
            v = sdk_version(z.read("nova_sdk.py").decode("utf-8", "replace"))
        if v < MIN_SDK:
            problems.append(f"nova_sdk.py {'.'.join(map(str, v)) or 'without __version__'} < "
                            f"{'.'.join(map(str, MIN_SDK))} (the server refuses older ones; copy the current "
                            "starter package's)")
    return {"path": str(path), "files": len(names), "bytes": size, "unpacked_bytes": unpacked,
            "size_mb": round(size / 1e6, 2), "problems": problems, "ok": not problems}


MIN_SDK = (1, 0, 1)  # the oldest SDK the server accepts


def sdk_version(text: str) -> tuple[int, ...]:
    m = re.search(r"^__version__\s*=\s*['\"](\d+(?:\.\d+)*)['\"]", text, re.M)
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def build(out: Path, include_kb: bool = False, include_fixtures: bool = True, root: Path = PROJECT_DIR,
          sdk: Optional[Path] = None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = _files(root, include_kb, include_fixtures, sdk)
    req = (root / "requirements.txt").read_text(encoding="utf-8")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in files:
            z.write(src, arc)
        z.writestr("requirements.txt", req)
        z.writestr("NOTICE.md", NOTICE)
    return verify(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist/submission.zip")
    ap.add_argument("--sdk", default="", help="nova_sdk.py from the starter package (default: repository root)")
    ap.add_argument("--with-kb", action="store_true", help="include kb/ (compiled knowledge)")
    ap.add_argument("--no-fixtures", action="store_true", help="leave out the 8 illustrative fixture cards")
    a = ap.parse_args(argv)
    info = build(Path(a.out), include_kb=a.with_kb, include_fixtures=not a.no_fixtures, sdk=Path(a.sdk) if a.sdk else None)
    print(f"{info['path']}: {info['files']} files, {info['size_mb']} MB, unpacked {info['unpacked_bytes']:,} bytes "
          f"({'OK' if info['ok'] else 'NOT OK'})")
    for p in info["problems"]:
        print(f"  - {p}")
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
