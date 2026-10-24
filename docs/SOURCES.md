# Source documents to download (v0.4.1)

For the knowledge compiler of the research core (`docs/CORE.md`); the contest runtime uses none of it. Put files under
`sources/docs/` (gitignored) and list each in `sources/manifest.yaml` (template:
`docs/sources_manifest.template.yaml`). Ontologies and HPO annotations come from `dxagent kb fetch`. Licences: keep
subscription material out of the repository; do not bulk-crawl sites (NCBI Bookshelf allows automated download only
through the NLM LitArch FTP for its open-access subset).

## Must

| # | Source | Kind / tier | What it supplies | Where |
|---|---|---|---|---|
| 1 | The Rational Clinical Examination (book) and later JAMA RCE articles | rce / 1 | Likelihood ratios for history and exam findings, about 50+ presentations | JAMAevidence: https://jamaevidence.mhmedical.com ; JAMA: https://jamanetwork.com (institutional access) |
| 2 | McGee, Evidence-Based Physical Diagnosis | reference / 2 | LR tables for physical-exam findings | ClinicalKey: https://www.clinicalkey.com (institutional access) |
| 3 | StatPearls chapters for the target diseases | reference / 2 | Broad disease coverage: history, exam, labs, differential | https://www.ncbi.nlm.nih.gov/books/NBK430685/ (save chapter pages; bulk only via ftp://ftp.ncbi.nlm.nih.gov/pub/litarch/ if listed in file_list.csv) |
| 4 | Core guidelines, first domain (chest pain / dyspnea): ESC acute PE 2019; ESC ACS 2023; AHA/ACC chest pain 2021; ESC heart failure 2021 + 2023 update; ATS/IDSA community-acquired pneumonia 2019; GOLD report (latest); ESC aortic diseases (latest); ESC pericardial diseases (latest) | guideline / 1 | Diagnostic criteria, test accuracy, harm | https://www.escardio.org/Guidelines ; https://www.acc.org/guidelines ; https://www.atsjournals.org ; https://goldcopd.org |

## Recommended

| # | Source | Kind / tier | What it supplies | Where |
|---|---|---|---|---|
| 5 | Cochrane diagnostic test accuracy reviews | systematic_review / 1 | Pooled sensitivity and specificity of tests | https://www.cochranelibrary.com (library access) |
| 6 | Open-access diagnostic-accuracy reviews (BMJ, Annals, CMAJ, etc.) | systematic_review / 1 | Test and finding accuracy | https://pmc.ncbi.nlm.nih.gov (PMC open-access subset) |
| 7 | ACR Appropriateness Criteria | guideline / 1 | Imaging choice per presentation | https://acsearch.acr.org/list |
| 8 | NICE guidelines | guideline / 1 | Diagnostic criteria and pathways, broad scope | https://www.nice.org.uk/guidance |
| 9 | Korean guidelines (KoMGI portal and society sites) | guideline / 1 | Korean epidemiology and practice (e.g. tuberculosis, scrub typhus) | https://www.guideline.or.kr |
| 10 | Other internal-medicine guidelines as coverage grows: IDSA, KDIGO, ADA Standards of Care, GINA, AASLD, ACG, ACR (rheumatology), AAN | guideline / 1 | Criteria and tests per specialty | https://www.idsociety.org/practice-guideline/ ; https://kdigo.org/guidelines/ ; https://diabetesjournals.org/care ; https://ginasthma.org/reports/ |
| 11 | NCI PDQ, health-professional versions | reference / 2 | Cancer presentation and workup | https://www.cancer.gov/publications/pdq |
| 12 | WHO and CDC clinical guidance | guideline / 1 | Infectious and tropical diseases | https://iris.who.int ; https://www.cdc.gov |
| 13 | GeneReviews | reference / 2 | Genetic disorders (complements HPO) | https://www.ncbi.nlm.nih.gov/books/NBK1116/ |

## Not recommended for compiling

UpToDate, DynaMed, and full textbooks such as Harrison's: subscription terms generally prohibit automated extraction,
and textbooks state few frequencies. Wikipedia and unreviewed web pages: not tier-graded sources.
