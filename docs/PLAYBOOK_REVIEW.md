# Playbook review log

The 58 playbooks (`dxagent/nova/playbooks/*.yaml`) are `status: draft`. They were reviewed twice by AI and never by a
clinician: in 0.6.0 six independent reviewers, one per organ-system group, worked read-only from a shared brief on the
first 52 (clinical errors first, then omissions, then question quality; 320 findings, 33 rated errors, most of them
misroutes); in 0.8.0 four physician-level passes applied an evidence review's per-playbook findings to all 53 then
existing and wrote five new ones. Every finding was checked by hand and adopted, adapted or declined, and every routing
claim was re-tested. **This log replaces neither a clinician's review nor the `status: reviewed` mark.** Each
playbook's own `notes` field records its changes with their sources; `python -m dxagent.nova.playbooks --sheet
docs/PLAYBOOKS.md` writes a readable sheet (generated, not committed).

## Engine fixes found by the review

| Finding | Fix |
|---|---|
| Excluded words were plain substrings, so "ankle swollen" was excluded from edema by the body-part word inside it | Excluded words follow the same word-start rule as match words |
| Adverbs were removed from the opening but not from match words, so "urinating often" could never match | Each match word is tried on the opening as written and with adverbs removed; frequency and volume complaints stay distinct |
| A duration between subject and verb hid the match ("periods stopped for 3 months") | Durations are removed along with the adverbs |
| A bare "yes" to a negatively phrased question was recorded as present; in Korean it means "no" | A bare yes/no after a negative question is left to the model; every playbook question is positively phrased |
| Can't-miss diagnoses below the 6th differential entry never reached the topic plan | The topic plan receives every can't-miss cause, then the usual causes |
| A 9th or 10th maneuver was never requested | The validator rejects more than 8 |
| Adult-only items were asked of guardians in "any"-population playbooks | New `only: adult/child`; three playbooks became adult-only and a pediatric rash playbook was added |
| The 8 chest-pain fixture cards added irrelevant maneuvers to a short-stature visit | Cards are off by default |
| 18 files had differential names cut short by an unquoted comma in YAML | Quoted; the validator rejects such an entry |

## Maneuver names as single actions (0.7.2, decision 92)

The examiner refuses a request that bundles actions, covers a whole region or names a symptom without its method, at
the cost of a turn. Eighteen of our maneuver names were therefore renamed to one specific action across 30 playbooks;
standard vocabulary names were kept. Examples: "cranial nerve exam" became facial strength, extraocular movement or
tongue movement depending on the complaint; "tuning-fork test" became the Weber and Rinne tests; "pelvic exam" became
bimanual pelvic exam; "sensation exam" became light touch (or perineal sensation); "ascites check" became shifting
dullness; "jugular vein check" became jugular venous pressure. **For the clinician to confirm:** each new maneuver is
the most informative single action for its complaint.

## The evidence review (0.8.0)

**In every playbook:** tests as `{name, for, why}` with each differential entry linked to at least one test, plus
pre-treatment safety tests (hCG, baseline blood count and liver tests, ECG) where the usual treatment needs them; a
`plan` block (first-line treatment, disposition, follow-up, 2-4 named warning signs); follow-ups (`then`) where a quantity
decides the cause; priorities 1-2 within 40 and priority 1 within 30 turns per patient type.

Main corrections by group (each recorded with its source in the playbook's `notes`):

- **Cardiopulmonary, constitutional:** the chest-pain emergency-call line (20 minutes defines prolonged angina, not a safe
  wait); PERC/Wells and HEART items; syncope high-risk symptoms; the TB question at priority 1 in cough, hemoptysis,
  dyspnea, fever, weight loss, neck mass and fatigue; fever travel and exposure items; aldosterone/renin screening and
  national thresholds in hypertension; new **dyslipidemia**.
- **GI, urology:** the abdominal-pain budget fixed and McBurney point tenderness in place of the Rovsing sign; hepatitis B
  question in distension, jaundice and upper GI bleeding; anal inspection in hematochezia; stool occult blood removed
  from acute diarrhea; current guidance on male UTI, colorectal screening age and Rome criteria; cauda equina items in
  voiding difficulty; new **oliguria**.
- **Neurology, mental health, MSK, ENT:** Ottawa SAH and POUND items in headache; supine roll test in dizziness; tongue
  inspection and arrhythmic-syncope flags in seizure; a suicide-risk preparatory-behavior item with an access-to-means
  follow-up in depressed mood; steroid and stenosis items in back pain; an informant at priority 1 in memory loss;
  hypnotic follow-ups in insomnia; new **neck_pain** and **nasal_symptoms**.
- **Women's health, children, skin:** pelvic and speculum exams `only: sexually_active` with `gate: sexual`; a bleeding-
  disorder screen in abnormal uterine bleeding; adolescent PCOS rules; drug start date and eschar in rash; hernia,
  testis and ketoacidosis in pediatric vomiting; the Kawasaki path in pediatric fever; paroxysm and apnea in pediatric
  cough; new **easy_bruising**.

## Doubts for the clinician (not verified from a primary source)

| Playbook | Claim or choice to confirm |
|---|---|
| high_bp | The national home-BP threshold (135/85); the high-risk list for a target below 130/80 |
| dyslipidemia | LDL-C targets (a 2022 coronary target below 55 and a 50% fall; a 2026 society guideline known from press reports; primary-prevention 130/160 from 2018); pregnancy test before a statin as a safety test |
| syncope | The 2018 ESC listing of GI bleeding on rectal examination as high-risk |
| polyuria | Wording of AVP deficiency/resistance; a "high blood sugar" opening now routes here |
| fatigue | Both the TB/hepatitis item and the direct hepatitis B question are asked (carriers often deny hepatitis) |
| hematochezia, constipation | The 2025 revision of national colorectal screening (45-74 years) from abstracts only; Rome frequency from secondary sources |
| oliguria | HFRS peak season; "hardly any urine" routes to oliguria although it also screens retention |
| abdominal_pain | Abdominal percussion dropped for the bimanual exam (priority 2 for sexually active women; the focus call must trim it) |
| hematuria | The title "red urine" also changes the fallback chief complaint |
| dysuria | Cystitis antibiotics written generically (drug availability in Korea) |
| weakness_numbness | The 4.5-hour thrombolysis window; Bell palsy steroids within 72 hours |
| sore_throat | Ten days of penicillin; the 2025 IDSA scoring approach |
| nasal_symptoms | The 3-day limit for decongestant sprays; adult-only (a child's runny nose goes to the topic plan) |
| neck_pain | "Pain to the shoulder" as a match phrase; stiff neck with fever routes here (fever and meningitis items included) |
| joint_pain | No maneuver at priority 1: examining the named joint depends on the focus call; the 2020 ACR urate target below 6 mg/dL |
| insomnia | The name and query route of the national controlled-drug prescription history |
| developmental_delay | A developmental screening questionnaire is kept as an exam (the examiner may refuse it); the item `id: concern` replaces the standard worry question |
| AUB, vaginal_discharge | The recto-abdominal exam convention for women who never had intercourse (no Korean source found) |
| all | Examiner acceptance of the 0.8.0 maneuver names (McBurney point tenderness, gallbladder palpation, anal inspection, cough stress test, attention test, tongue inspection, leg pitting-edema check, eschar check, ...): confirm in the first server runs |

Maneuvers the reviewers wanted but the vocabulary lacks (each can be added to `EXTRA_EXAMS` once the examiner is known to
accept it): test of skew, Hoffmann sign, 128 Hz vibration sense, spiral drawing and finger tapping, crossed straight-leg
raise, water-swallow test, dry axilla, pulsatile abdominal mass, single-site foot pulse, Achilles tendon palpation.
Not covered (decision 103): counselling and visit-type stations.

## Changing a playbook

```bash
$EDITOR dxagent/nova/playbooks/chest_pain.yaml   # edit; to sign off set  status: reviewed  and  reviewer: "<name>"
python -m dxagent.nova.playbooks                   # must print "58 playbooks valid"
python -m dxagent.nova.playbooks --budget          # priority 1 within 30 and priorities 1+2 within 40 keeps a visit in 49 turns
pytest -q tests/test_playbooks.py tests/test_visit.py tests/test_review.py
```

Priority: 1 always asked; 2 asked unless the focus call skips it or the plan cap cuts it (then asked if turns remain after
the time-out); 3 only when the focus call adds it. Quote any YAML value that contains a comma or colon
(`en: "A (B, C)"`): the validator rejects an entry split by one.
