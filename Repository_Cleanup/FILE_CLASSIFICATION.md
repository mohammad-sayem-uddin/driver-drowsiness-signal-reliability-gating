# FILE CLASSIFICATION

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** Assign every project file a single freeze-time class so the next
researcher instantly knows what is authoritative, what is working material, what
is history, what must never be cited, and what can be removed.

> **Rule applied throughout:** prefer ARCHIVE over DELETE when in any doubt. A
> file is only in DELETE if it carries zero information (OS/interpreter cruft,
> empty dirs) or is foreign to the project. Every DELETE and SUPERSEDED entry
> carries an explicit justification. No file required for reproducibility is
> ever placed in DELETE.

---

## 1. Classification legend

| Class | Meaning | Cite it? | Keep it? |
|---|---|---|---|
| **FINAL** | Authoritative, frozen, citable source of truth | Yes | Yes, verbatim |
| **ACTIVE** | Living working material (code, indexes, logs) still in use | As code, not as result | Yes |
| **ARCHIVE** | Historical/process record; correct in its time, now superseded by newer docs | No | Yes, moved aside |
| **SUPERSEDED** | Explicitly replaced by a newer authoritative file; content now contradicts the record | **Never** | Yes, with banner |
| **DELETE** | Zero information value or foreign to the project | No | No (with justification) |

---

## 2. FINAL — authoritative, frozen, citable

### 2.1 Governing documents

| File | Justification |
|---|---|
| `PROJECT_CONTEXT.md` | Declared single source of truth for design/scope/status |
| `README.md` | Project front door + read-order (status banner needs the EXP-005 fix — see `CONTRADICTION_REPORT.md`) |
| `HANDOVER.md` | Supervisor/examiner handover (same EXP-005 status fix) |
| `EXPERIMENT_REGISTRY.md` | The experiment ledger; nothing is citable without a row here |
| `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md` | The frozen engineering contract |

### 2.2 Final experiment reports (one authoritative report per experiment)

| File | Experiment | Justification |
|---|---|---|
| `reports/EXP-002_REPORT.md` | EXP-002 | Authoritative training result (+ its two audit companions) |
| `reports/EXP-002_DATASET_VERIFICATION.md` | EXP-002 | Dataset-integrity evidence for EXP-002 |
| `reports/EXP-002_PARAMETER_AUDIT.md` | EXP-002 | Param-count evidence (19,745) |
| `reports/EXP-003_REPORT.md` | EXP-003 | Authoritative quantization result |
| `reports/EXP-004_REPORT.md` | EXP-004 | Authoritative LOSO-ablation (negative) result |
| `reports/EXP-004_AUDIT/EXP-004_SCIENTIFIC_AUDIT_REPORT.md` | EXP-004 | Independent re-audit of EXP-004 |
| `reports/EXP-005_REPORT.md` | EXP-005 | Authoritative event-level alarm result |
| `reports/EXP-005_AUDIT.md` | EXP-005 | Independent audit, verdict ACCEPT |

### 2.3 Final measured artifacts (the evidence the reports cite)

| Artifact | Justification |
|---|---|
| `results/measured_results.json` | Canonical measured-results ledger; sole figure source |
| `results/*.png` (2 canonical figures) | Regenerate only from measured JSON |
| `experiments/EXP-002_*`, `EXP-003_*`, `EXP-004_*`, `EXP-005_*` | Raw committed evidence per experiment — **never deletable** (reproducibility) |
| `models/*.tflite` (3) | Quantization outputs cited by EXP-003 |
| `checkpoints/*.keras` (4) | Training checkpoints backing EXP-002 |
| `tensorboard/EXP-002_*` | Training-curve evidence for EXP-002 |

### 2.4 Final code (the pipeline + evaluation that produced the results)

| File group | Justification |
|---|---|
| `src/` — 17 real modules (excl. the orphan; see §3) | The single per-frame pipeline that produced every result |
| `evaluation/` — all 8 `.py` | LOSO/event harness + integrity verifier; reproduce every EXP number |
| `tools/build_subject_disjoint_splits.py`, `train_cnn.py`, `train_exp002_microeyenet.py`, `export_and_evaluate_quantization.py`, `dataset_validator.py`, `verify_integrity.py` | Canonical data-prep + experiment executors |
| `tests/test_suite.py`, `tests/smoke_test.py`, `tests/test_event_metrics.py` | The verification gates |
| `requirements.txt`, config/CI files | Environment contract |

---

## 3. ACTIVE — living working material, still in use

| File | Justification |
|---|---|
| `AGENT_MEMORY.md` | Fast-start brief; regularly updated, not a frozen result |
| `IMPLEMENTATION_LOG.md` | Running log of why the repo looks as it does |
| `CNN_IMPLEMENTATION_SPECIFICATION.md` | Live spec for the ablation-only CNN arm |
| `reports/README.md` | Reports-folder index; must be kept current |
| `docs/archive/README.md`, `reports/archive/README.md` | Archive indexes; navigational, kept current |
| `Data/mrl_eye/readme.md`, `Data/mrl_eye/splits_subject_disjoint/SPLIT_MANIFEST.md` | Dataset provenance/manifest, still referenced by the split tooling |
| `logs/*` | Run logs; gitignored working output |
| `results/measured_results.json.pre_exp004.bak` | Meaningful pre-EXP-004 snapshot; keep as a backup, do not cite |
| `tools/data_fetcher.py`, `generate_metadata.py`, `preprocess_data.py`, `quality_checker.py`, `split_dataset.py`, `collect_eye_data.py` | Data-prep utilities; used ad hoc, retained |

> **Note on `src/camera_base.py`:** imported by nobody today, but it is a small
> base class in the live-camera layer, not stale scientific material. Classified
> **ACTIVE (orphan)** and flagged in `CONTRADICTION_REPORT.md` for the engineer
> to either wire in or drop — **not** an archive/delete candidate at freeze time.

---

## 4. ARCHIVE — historical/process record, correct in its time

These are already correctly located under `reports/archive/` and
`docs/archive/`. They stay for provenance; **no current doc should cite them**.

| Group | Count | Justification |
|---|---|---|
| `reports/archive/audit_v3.1/` | 13 | Superseded v3.1 audit cycle |
| `reports/archive/phase01/` | 9 | Phase-01 process/status snapshots |
| `reports/archive/phase02/` | 11 | Phase-02 dataset docs + manifest |
| `reports/archive/phase02_5/` | 2 | Phase-02.5 reliability-estimation notes |
| `reports/archive/planning/` | 6 | Early planning/strategy docs |
| `reports/archive/reviews/` | 5 | Early review docs (pre-publication-review layer) |
| `reports/archive/verification/` | 10 | Early verification docs |
| `docs/archive/TECHNICAL_AUDIT_REPORT.md` | 1 | Superseded early technical audit |
| `docs/archive/old_setup_guide_README.md` | 1 | Superseded setup guide (README covers this now) |
| `docs/archive/research_notes.md` | 1 | Early scratch notes |

### 4.1 Publication-review layer — ARCHIVE-on-freeze

These three are genuine outputs, but they are an **input→output→plan chain** that
the recovery plan already consumes. They are currently invisible to every nav
path (see `CONTRADICTION_REPORT.md`). Keep them, but as a dated review bundle,
not as top-level current reports.

| File | Class | Justification |
|---|---|---|
| `reports/INDEPENDENT_SCIENTIFIC_REVIEW.md` | ARCHIVE (review bundle) | Review output R1 (2026-07-30); consumed by the recovery plan |
| `reports/PUBLICATION_READINESS_ASSESSMENT.md` | ARCHIVE (review bundle) | Review output R2 (2026-07-30); consumed by the recovery plan |
| `reports/PUBLICATION_RECOVERY_PLAN.md` | **FINAL (action doc)** | The newest doc in the chain; the authoritative C1–C13 to-do list. Keep top-level. |

> The two Prompt files are their read-only *inputs* — see §6 (SUPERSEDED/INPUT).

---

## 5. SUPERSEDED — explicitly replaced; never cite (keep, banner)

| File | Superseded by | Justification |
|---|---|---|
| `reports/EXP005_ROOT_CAUSE_ANALYSIS.md` | `reports/EXP-005_REPORT.md` + `reports/EXP-005_AUDIT.md` | Describes an **old run** (10,800 frames, recall 0.0, 16/16 GT missed) that contradicts the committed final EXP-005 (66,521 frames, recall 0.122). The `PUBLICATION_RECOVERY_PLAN.md` tear-off checklist (C7) explicitly directs marking it SUPERSEDED — the project's own newest doc authorizes this. **Keep** as the root-cause narrative that led to the fixed run; band with a banner. |
| `Data/FRESH_DATASET_AUDIT_REPORT.md` | `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md` | Already carries a SUPERSEDED banner; kept for provenance |
| `Data/PROJECT_PREPARATION_AND_REAL_DATA_CONNECTIVITY_REPORT.md` | `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md` | Already carries a SUPERSEDED banner; kept for provenance |

> `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md` is the **FINAL** canonical
> dataset-integrity report (the doc that supersedes the two above).

### 5.1 INPUT prompts (read-only, not project outputs)

| File | Justification |
|---|---|
| `Prompt_1_Independent_Scientific_Research_Review.md` | Read-only task prompt that *produced* R1; not a finding. Retain as the review's input of record, out of the top-level current set. |
| `Prompt_2_Independent_Publication_Readiness_Assessment.md` | Read-only task prompt that *produced* R2; same treatment. |

---

## 6. DELETE — zero information value or foreign (justified)

| Item | Justification for deletion |
|---|---|
| `.DS_Store` (5) | macOS Finder metadata; no project value; should be gitignored |
| `__pycache__/` (5 dirs) | Python bytecode cache; regenerated on run |
| `benchmark/` (empty dir) | 0 B, referenced by nothing; the latency harness lives in `evaluation/` |
| `gitlab-vscode-extension-main/` (8.9 MB) | Third-party VS Code extension download; foreign to the pipeline, gitignored, referenced by nothing |

**Regenerable, not deleted at freeze (kept, gitignored):**
`.venv/` (1.7 GB) is regenerable from `requirements.txt` and is not "junk" — it
is left in place for the working environment and simply excluded from the
archive bundle. It is **not** a DELETE recommendation.

### 6.1 Judgment-call scripts kept OUT of DELETE

| Item | Decision | Justification |
|---|---|---|
| `test_pipeline.py`, `test_pose.py`, `test_variance.py`, `test_webcam.py` (root scratch) | **ARCHIVE**, not DELETE | Ad-hoc probes, not the test framework; `IMPLEMENTATION_LOG.md` records them as "left untouched, low risk." Moving them to an archive/scratch folder de-clutters root without destroying history. `test_variance.py` holds a dead-duplicate `EMASmoother` — its removal is safe but "prefer archiving" governs. |

---

## 7. Classification summary

| Class | Approx. count | Handling at freeze |
|---|---|---|
| FINAL | ~18 docs + all measured artifacts + pipeline/eval code | Freeze verbatim (2 status fixes) |
| ACTIVE | ~15 files | Keep, keep current |
| ARCHIVE | 57 archived reports + 3 docs/archive + review bundle (2) + 4 root scratch | Keep, moved aside |
| SUPERSEDED | 5 (1 report + 2 data + 2 prompts) | Keep, banner, never cite |
| DELETE | 5 `.DS_Store` + 5 `__pycache__` + `benchmark/` + `gitlab-vscode-extension-main/` | Remove (justified) |

**One authoritative file per topic** is resolved in `CONTRADICTION_REPORT.md`;
the physical moves are staged in `ARCHIVE_PLAN.md`; nothing here is executed —
these are recommendations pending your approval.
