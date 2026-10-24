# Provisional cards (runtime)

YAML disease cards placed here are loaded by the competition runtime (`dxagent.nova`) next to the compiled `kb/`, and
the 8 illustrative chest-pain/dyspnea fixture cards when `NOVA_USE_FIXTURES=1` (default 0: off). Use the same format as
`dxagent/knowledge/fixtures/chest_pain_demo/*.yaml`, with `provenance.status: provisional` and a source line. More
directories: `NOVA_CARDS_DIRS=dir1:dir2`. The cards matter only with the Bayesian core (`NOVA_CORE=1`, `docs/CORE.md`).
