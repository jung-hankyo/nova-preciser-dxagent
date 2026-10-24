"""`dxagent kb fetch`: download the ontology releases and build the indexes (v0.4).

Files land in `<sources>/ontology/` (gitignored) and are recorded in the manifest with their release version and
SHA-256. The HPO annotation file is also registered as a curated_db source (`hpo_annotations`, tier 2). Indexes are
written to `<kb>/ontology/`. Licences: MONDO CC BY 4.0; HPO and its annotations under the HPO licence (free use
with attribution); Orphanet-derived annotations CC BY 4.0.
"""

from __future__ import annotations

import datetime as dt
import shutil
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from dxagent.knowledge.compiler.hpo import read_hpoa
from dxagent.knowledge.compiler.sources import Manifest, OntologySource, SourceDoc, sha256_file
from dxagent.knowledge.ontology import build_index

RELEASES = {
    "mondo": "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo",
    "hpo": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo",
    "hpo_annotations": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa",
}
FILES = {"mondo": "mondo.obo", "hpo": "hp.obo", "hpo_annotations": "phenotype.hpoa"}
ROOTS = {"mondo": ("MONDO", "MONDO:0000001", "mondo_index.json.gz"), "hpo": ("HP", "HP:0000118", "hpo_index.json.gz")}


def _download(url: str, dest: Path, timeout: float = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dxagent/0.4"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    tmp.replace(dest)


def _obo_version(path: Path) -> str:
    with open(path, encoding="utf-8") as fh:
        for _ in range(30):
            line = fh.readline()
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
    return ""


def fetch(sources_dir: Path, kb_dir: Path, which: tuple[str, ...] = ("mondo", "hpo", "hpo_annotations"),
          download: Callable[[str, Path], None] = _download, local: Optional[dict[str, Path]] = None,
          progress=print) -> Manifest:
    """Download (or copy from `local`) each ontology file, record it, and build indexes. Returns the manifest."""
    sources_dir, kb_dir = Path(sources_dir), Path(kb_dir)
    mpath = sources_dir / "manifest.yaml"
    manifest = Manifest.load(mpath)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for key in which:
        dest = sources_dir / "ontology" / FILES[key]
        if local and key in local:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local[key], dest)
            progress(f"copied {local[key]} -> {dest}")
        else:
            progress(f"downloading {RELEASES[key]}")
            download(RELEASES[key], dest)
        version = read_hpoa(dest)[0] if key == "hpo_annotations" else _obo_version(dest)
        manifest.upsert_ontology(OntologySource(id=key, path=str(dest.relative_to(sources_dir)), url=RELEASES[key],
                                                version=version, sha256=sha256_file(dest), fetched_at=now))
        if key == "hpo_annotations":
            src = manifest.source("hpo_annotations") or SourceDoc(
                id="hpo_annotations", kind="curated_db", title="HPO disease annotations (OMIM, Orphanet)",
                publisher="Human Phenotype Ontology / Orphanet", licence="HPO licence (free use with attribution); Orphanet CC BY 4.0",
                access="open", path=str(dest.relative_to(sources_dir)), url=RELEASES[key])
            src.edition, src.sha256 = version, sha256_file(dest)
            manifest.sources = [s for s in manifest.sources if s.id != "hpo_annotations"] + [src]
        if key in ROOTS:
            prefix, root, out = ROOTS[key]
            stats = build_index(dest, kb_dir / "ontology" / out, prefix, root)
            progress(f"indexed {key}: {stats['terms']} terms ({stats['version']})")
    manifest.save(mpath)
    progress(f"manifest -> {mpath}")
    return manifest
