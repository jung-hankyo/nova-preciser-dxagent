"""Knowledge compiler (v0.4): cited sources -> verified claims -> reconciled, versioned disease cards.

    fetch.py      download MONDO, HPO and HPO annotations; build ontology indexes
    sources.py    source manifest, evidence tiers, confidence table
    hpo.py        HPO annotations -> claims (structured, no LLM)
    documents.py  local PDF / HTML / text -> located passages
    extract.py    LLM extraction of verbatim claims from passages (cached), then deterministic verification
    verify.py     span, number and subject checks; value derivation
    assemble.py   reconcile claims (higher tier wins, conflicts logged, never averaged), map findings onto codes,
                  build cards, derive missing background rates
    compiler.py   orchestration; writes kb/ (cards.jsonl, findings.jsonl, build.json, compile_report.json)
    online.py     live fallback (off by default): PubMed abstracts through the same extract-verify-assemble path
"""
