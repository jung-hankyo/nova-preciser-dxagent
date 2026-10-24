"""Ontology spine (v0.4): MONDO for diseases, HPO for findings. Deterministic, offline, no LLM.

Both ontologies arrive as OBO files downloaded by `dxagent kb fetch` (GitHub releases of
monarch-initiative/mondo and obophenotype/human-phenotype-ontology). `build_index` parses one OBO file
into a compact JSON index (labels, synonyms, parents, selected cross-references); `OntologyIndex`
loads that index and answers:

    resolve(name)    free-text name -> (term id, method, score); exact label/synonym first, then a
                     token-overlap match over an inverted index. Never an LLM, so the same name always
                     resolves the same way.
    ancestors(id)    {ancestor id: distance}, breadth-first over is_a.
    xref_map(pfx)    cross-reference -> term ids (e.g. "OMIM:154700" -> MONDO ids), used to join HPO
                     annotations onto MONDO diseases.

Why an ontology: disease identity was an LLM name slug, so the same disease could appear under two ids,
and a parent and its subtype competed as rivals. MONDO gives one id per disease, a hierarchy (used by the
belief update and the planner: see reasoning/belief.py `tree_structure`), and cross-references to ICD-10,
ICD-11, SNOMED CT, Orphanet and OMIM. HPO gives one code per phenotype with synonyms, replacing similarity
merging for the findings it covers.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Optional

_SYN = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"\s+(EXACT|RELATED|NARROW|BROAD)')
_REF = re.compile(r"^([A-Za-z_]+:\S+)")

# Cross-reference prefixes worth keeping (the rest of MONDO's xrefs are not needed at run time).
KEEP_XREF = {"OMIM", "Orphanet", "ICD10CM", "ICD10WHO", "icd11.foundation", "SCTID", "UMLS", "MESH", "DOID", "NCIT"}

_STOP = {"of", "the", "and", "or", "with", "without", "due", "to", "in", "by", "a", "an", "type", "nos", "unspecified",
         "disease", "disorder"}


def norm_name(text: str) -> str:
    """Lowercase, unify punctuation and spacing. 'Myasthenia-Gravis ' -> 'myasthenia gravis'."""
    t = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    t = re.sub(r"[^a-z0-9']+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    out = []
    for w in norm_name(text).replace("'s", "").split():
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# OBO parsing
# ---------------------------------------------------------------------------


def parse_obo(path: Path, prefix: str) -> tuple[dict[str, dict], dict[str, str]]:
    """Return ({id: term}, header). Only [Term] stanzas whose id starts with `prefix`; obsolete terms are kept
    (flagged) so replaced_by can redirect old ids."""
    terms: dict[str, dict] = {}
    header: dict[str, str] = {}
    cur: Optional[dict] = None
    in_term = False
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("["):
                if cur is not None and cur["id"].startswith(prefix + ":"):
                    terms[cur["id"]] = cur
                cur = None
                in_term = line == "[Term]"
                if in_term:
                    cur = {"id": "", "name": "", "exact": [], "other": [], "parents": [], "xrefs": {}, "obsolete": False,
                           "replaced_by": "", "alt_ids": []}
                continue
            if cur is None:
                if not in_term and ":" in line and not terms:
                    k, v = line.split(":", 1)
                    header.setdefault(k.strip(), v.strip())
                continue
            if not line or line.startswith("!"):
                continue
            key, _, val = line.partition(": ")
            if key == "id":
                cur["id"] = val.strip()
            elif key == "name":
                cur["name"] = val.strip()
            elif key == "synonym":
                m = _SYN.match(line)
                if m:
                    text = m.group(1).replace('\\"', '"')
                    (cur["exact"] if m.group(2) == "EXACT" else cur["other"]).append(text)
            elif key == "is_a":
                pid = val.split()[0]
                if pid.startswith(prefix + ":"):
                    cur["parents"].append(pid)
            elif key == "xref":
                m = _REF.match(val.strip())
                if m:
                    ref = m.group(1)
                    pfx = ref.split(":", 1)[0]
                    if pfx in KEEP_XREF:
                        cur["xrefs"].setdefault(pfx, []).append(ref)
            elif key == "is_obsolete" and val.strip() == "true":
                cur["obsolete"] = True
            elif key == "replaced_by":
                cur["replaced_by"] = val.strip()
            elif key == "alt_id":
                cur["alt_ids"].append(val.strip())
    if cur is not None and cur["id"].startswith(prefix + ":"):
        terms[cur["id"]] = cur
    return terms, header


def build_index(obo_path: Path, out_path: Path, prefix: str, root: str = "") -> dict:
    """Parse an OBO file and write a compact gzipped JSON index. Returns summary stats.

    `root` keeps only the root and its descendants (e.g. HP:0000118 'Phenotypic abnormality' for HPO findings,
    MONDO:0000001 'disease' for MONDO), plus obsolete terms that redirect into that set.
    """
    terms, header = parse_obo(Path(obo_path), prefix)
    keep = set(terms)
    if root:
        children: dict[str, list[str]] = defaultdict(list)
        for t in terms.values():
            for p in t["parents"]:
                children[p].append(t["id"])
        keep, q = {root}, deque([root])
        while q:
            for c in children.get(q.popleft(), []):
                if c not in keep:
                    keep.add(c)
                    q.append(c)
        keep |= {t["id"] for t in terms.values() if t["obsolete"] and t["replaced_by"] in keep}
    rows = {}
    for tid in sorted(keep):
        t = terms.get(tid)
        if t is None:
            continue
        rows[tid] = [t["name"], t["exact"], t["other"], [p for p in t["parents"] if p in keep], t["xrefs"],
                     t["replaced_by"] if t["obsolete"] else "", t["alt_ids"]]
    data = {"prefix": prefix, "root": root, "version": header.get("data-version", ""), "source": str(Path(obo_path).name),
            "terms": rows}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return {"terms": len(rows), "version": data["version"]}


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class OntologyIndex:
    """Loaded index. Terms are tuples: (label, exact synonyms, other synonyms, parents, xrefs, replaced_by, alt_ids)."""

    def __init__(self, data: dict):
        self.prefix: str = data.get("prefix", "")
        self.root: str = data.get("root", "")
        self.version: str = data.get("version", "")
        self.source: str = data.get("source", "")
        self.terms: dict[str, list] = data["terms"]
        self._exact: dict[str, list[str]] = defaultdict(list)  # normalized label/exact synonym -> ids
        self._other: dict[str, list[str]] = defaultdict(list)
        self._tok_index: Optional[dict[str, set[str]]] = None  # token -> ids, built on first token-level lookup
        self._children: dict[str, list[str]] = defaultdict(list)
        self._alt: dict[str, str] = {}
        self._xref: dict[str, dict[str, list[str]]] = {}
        for tid, (label, exact, other, parents, _x, replaced, alts) in self.terms.items():
            for a in alts:
                self._alt[a] = tid
            if replaced:
                continue
            for p in parents:
                self._children[p].append(tid)
            for name in [label, *exact]:
                key = norm_name(name)
                if key:
                    self._exact[key].append(tid)
            for name in other:
                key = norm_name(name)
                if key:
                    self._other[key].append(tid)
        self._anc_cache: dict[str, dict[str, int]] = {}

    @property
    def _tok(self) -> dict[str, set[str]]:
        if self._tok_index is None:
            idx: dict[str, set[str]] = defaultdict(set)
            for tid, row in self.terms.items():
                if row[5]:
                    continue
                for name in [row[0], *row[1]]:
                    for tok in tokens(name):
                        idx[tok].add(tid)
            self._tok_index = idx
        return self._tok_index

    @classmethod
    def load(cls, path: Path) -> "OntologyIndex":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def from_terms(cls, prefix: str, terms: dict[str, dict], root: str = "", version: str = "test") -> "OntologyIndex":
        """Small in-memory index (tests, fixtures). `terms`: {id: {name, exact, other, parents, xrefs}}."""
        rows = {tid: [t.get("name", tid), t.get("exact", []), t.get("other", []), t.get("parents", []),
                      t.get("xrefs", {}), t.get("replaced_by", ""), t.get("alt_ids", [])] for tid, t in terms.items()}
        return cls({"prefix": prefix, "root": root, "version": version, "source": "inline", "terms": rows})

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, tid: str) -> bool:
        return self.canonical_id(tid) in self.terms

    def canonical_id(self, tid: str) -> str:
        tid = self._alt.get(tid, tid)
        row = self.terms.get(tid)
        seen = set()
        while row and row[5] and tid not in seen:  # obsolete -> replaced_by
            seen.add(tid)
            tid = row[5]
            row = self.terms.get(tid)
        return tid

    def label(self, tid: str) -> str:
        row = self.terms.get(self.canonical_id(tid))
        return row[0] if row else ""

    def synonyms(self, tid: str, exact_only: bool = True) -> list[str]:
        row = self.terms.get(self.canonical_id(tid))
        if not row:
            return []
        return list(row[1]) if exact_only else list(row[1]) + list(row[2])

    def parents(self, tid: str) -> list[str]:
        row = self.terms.get(self.canonical_id(tid))
        return list(row[3]) if row else []

    def children(self, tid: str) -> list[str]:
        return list(self._children.get(self.canonical_id(tid), []))

    def xrefs(self, tid: str) -> dict[str, list[str]]:
        row = self.terms.get(self.canonical_id(tid))
        return dict(row[4]) if row else {}

    def ancestors(self, tid: str) -> dict[str, int]:
        """{ancestor id: shortest is_a distance}; the term itself is not included."""
        tid = self.canonical_id(tid)
        hit = self._anc_cache.get(tid)
        if hit is not None:
            return hit
        out: dict[str, int] = {}
        q = deque((p, 1) for p in self.parents(tid))
        while q:
            node, d = q.popleft()
            if node in out and out[node] <= d:
                continue
            out[node] = d
            for p in self.parents(node):
                q.append((p, d + 1))
        self._anc_cache[tid] = out
        return out

    def depth(self, tid: str) -> int:
        """Distance from the index root (0 at the root); large if unknown."""
        if not self.root:
            return 99
        tid = self.canonical_id(tid)
        return 0 if tid == self.root else self.ancestors(tid).get(self.root, 99)

    def is_ancestor(self, anc: str, tid: str) -> bool:
        return self.canonical_id(anc) in self.ancestors(tid)

    def xref_map(self, prefix: str) -> dict[str, list[str]]:
        """Cross-reference (e.g. 'Orphanet:558') -> live term ids."""
        if prefix not in self._xref:
            m: dict[str, list[str]] = defaultdict(list)
            for tid, row in self.terms.items():
                if row[5]:
                    continue
                for ref in row[4].get(prefix, []):
                    m[ref].append(tid)
            self._xref[prefix] = dict(m)
        return self._xref[prefix]

    def resolve(self, name: str, min_score: float = 0.85) -> tuple[Optional[str], str, float]:
        """Free text -> (id, method, score). Methods: id, exact, synonym, token, ambiguous, none.

        Exact label or exact-synonym matches win; RELATED/NARROW/BROAD synonyms are used only when unique; otherwise
        the best token-Dice match at or above `min_score`, provided it is unique at that score.
        """
        raw = (name or "").strip()
        if not raw:
            return None, "none", 0.0
        if re.fullmatch(rf"{self.prefix}[:_]\d+", raw, flags=re.I):
            tid = self.canonical_id(raw.upper().replace("_", ":"))
            return (tid, "id", 1.0) if tid in self.terms else (None, "none", 0.0)
        key = norm_name(raw)
        ids = [self.canonical_id(i) for i in self._exact.get(key, [])]
        ids = list(dict.fromkeys(i for i in ids if i in self.terms))
        if len(ids) == 1:
            return ids[0], "exact", 1.0
        if len(ids) > 1:
            by_label = [i for i in ids if norm_name(self.label(i)) == key]
            if len(by_label) == 1:
                return by_label[0], "exact", 1.0
            return sorted(ids)[0], "ambiguous", 0.9
        other = list(dict.fromkeys(self.canonical_id(i) for i in self._other.get(key, [])))
        if len(other) == 1:
            return other[0], "synonym", 0.9
        toks = set(tokens(raw))
        if not toks:
            return None, "none", 0.0
        cands: dict[str, int] = defaultdict(int)
        for t in toks:
            for tid in self._tok.get(t, ()):
                cands[tid] += 1
        best: list[tuple[float, str]] = []
        for tid, hits in cands.items():
            if hits * 2 < len(toks):  # needs at least half the query tokens
                continue
            row = self.terms[tid]
            score = max(_dice(toks, set(tokens(n))) for n in [row[0], *row[1]])
            best.append((score, tid))
        if not best:
            return None, "none", 0.0
        best.sort(key=lambda x: (-x[0], x[1]))
        top, tid = best[0]
        if top < min_score:
            return None, "none", top
        if len(best) > 1 and best[1][0] == top:
            return tid, "ambiguous", top
        return tid, "token", top


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


class OntologySet:
    """The indexes the runtime uses: diseases (MONDO) and, optionally, phenotypes (HPO)."""

    def __init__(self, disease: Optional[OntologyIndex] = None, phenotype: Optional[OntologyIndex] = None):
        self.disease = disease
        self.phenotype = phenotype

    @classmethod
    def load_dir(cls, d: Path) -> "OntologySet":
        d = Path(d)
        dis = d / "mondo_index.json.gz"
        phe = d / "hpo_index.json.gz"
        return cls(OntologyIndex.load(dis) if dis.exists() else None, OntologyIndex.load(phe) if phe.exists() else None)

    def __bool__(self) -> bool:
        return self.disease is not None


def curie_to_card_id(curie: str) -> str:
    """'MONDO:0005249' -> 'mondo_0005249' (snake_case ids, as everywhere else in the state)."""
    return curie.replace(":", "_").lower()


def card_id_to_curie(card_id: str) -> str:
    m = re.fullmatch(r"([a-z]+)_(\d+)", card_id or "")
    return f"{m.group(1).upper()}:{m.group(2)}" if m else ""


def iter_names(index: OntologyIndex, tid: str, max_n: int = 12) -> Iterable[str]:
    seen = set()
    for n in [index.label(tid), *index.synonyms(tid)]:
        k = norm_name(n)
        if k and k not in seen:
            seen.add(k)
            yield n
            if len(seen) >= max_n:
                return
