# REPOSITORY INVENTORY

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Inventory date:** 2026-07-30
**Purpose:** Complete, factual census of the repository at freeze time. Counts are facts on disk, not judgments. Classification, resolution, and action plans live in the sibling documents.
**Scope note:** `.venv/` (1.7 GB Python 3.11 virtualenv) and `gitlab-vscode-extension-main/` (8.9 MB vendored VS Code extension) are excluded from all "project file" counts below and enumerated separately in §7.

---

## 1. Top-level size map

| Path | Size | What it holds |
|---|---|---|
| `Data/` | 7.9 GB | NTHU-DDD, MRL Eye, YawDD, quarantined `drowsiness_detection` + 5 data-status docs |
| `.venv/` | 1.7 GB | Python 3.11 virtualenv — **regenerable from `requirements.txt`** |
| `experiments/` | 62 MB | Per-experiment artifacts (EXP-002/003/004/005) |
| `results/` | 11 MB | `measured_results.json` + 2 canonical figures + 1 `.bak` |
| `gitlab-vscode-extension-main/` | 8.9 MB | Foreign VS Code extension download (gitignored) |
| `reports/` | 1.3 MB | 14 current docs + 57 archived docs |
| `checkpoints/` | 1.1 MB | 4 MicroEyeNet Keras training checkpoints |
| `src/` | 792 KB | Detection pipeline (19 `.py`, incl. `utils/`) |
| `evaluation/` | 352 KB | LOSO/event harness, benchmarks, integrity verifier (8 `.py`) |
| `docs/` | 304 KB | `archive/` only (4 docs) |
| `tools/` | 172 KB | Data-prep + CNN training/quantization (12 `.py`) |
| `tests/` | 148 KB | 3 `.py` test suites |
| `models/` | 104 KB | 3 `.tflite` + `.gitkeep` |
| `logs/` | 24 KB | 3 training/run logs (gitignored) |
| `tensorboard/` | 20 KB | EXP-002 TensorBoard event files |
| `paper/` | 12 KB | `main.tex` manuscript |
| `benchmark/` | 0 B | **Empty directory** |

## 2. Markdown documentation census

**89 project Markdown files** (excludes 9 `.venv/` license/readme files). Grouped by location.

### 2.1 Root level (9)

| File | One-line role |
|---|---|
| `README.md` | Project front door + read-order + status banner |
| `PROJECT_CONTEXT.md` | Declared "single source of truth" |
| `HANDOVER.md` | Supervisor/thesis handover |
| `AGENT_MEMORY.md` | Fast-start brief for a new engineer/agent |
| `EXPERIMENT_REGISTRY.md` | Experiment ledger (EXP-000…004 + roadmap §4) |
| `IMPLEMENTATION_LOG.md` | Why the repo looks the way it does |
| `CNN_IMPLEMENTATION_SPECIFICATION.md` | MicroEyeNet spec (ablation arm) |
| `Prompt_1_Independent_Scientific_Research_Review.md` | Read-only INPUT task prompt |
| `Prompt_2_Independent_Publication_Readiness_Assessment.md` | Read-only INPUT task prompt |

### 2.2 `Data/` (5)

| File | One-line role |
|---|---|
| `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md` | CANONICAL dataset-integrity report |
| `Data/FRESH_DATASET_AUDIT_REPORT.md` | Carries a SUPERSEDED banner |
| `Data/PROJECT_PREPARATION_AND_REAL_DATA_CONNECTIVITY_REPORT.md` | Carries a SUPERSEDED banner |
| `Data/mrl_eye/readme.md` | Upstream MRL dataset readme |
| `Data/mrl_eye/splits_subject_disjoint/SPLIT_MANIFEST.md` | Subject-disjoint split manifest (seed 42) |

### 2.3 `docs/archive/` (4)

| File | One-line role |
|---|---|
| `docs/archive/README.md` | Archive-folder index |
| `docs/archive/TECHNICAL_AUDIT_REPORT.md` | Superseded early audit |
| `docs/archive/old_setup_guide_README.md` | Superseded setup guide |
| `docs/archive/research_notes.md` | Early scratch notes |

### 2.4 `reports/` — current (14)

| File | One-line role |
|---|---|
| `reports/README.md` | Reports-folder index + current/archive split |
| `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md` | Frozen engineering contract |
| `reports/EXP-002_REPORT.md` | MicroEyeNet training result |
| `reports/EXP-002_DATASET_VERIFICATION.md` | EXP-002 dataset integrity |
| `reports/EXP-002_PARAMETER_AUDIT.md` | EXP-002 param-count audit (19,745) |
| `reports/EXP-003_REPORT.md` | INT8/FP16 quantization result |
| `reports/EXP-004_REPORT.md` | LOSO V0–V4 ablation (negative) |
| `reports/EXP-004_AUDIT/EXP-004_SCIENTIFIC_AUDIT_REPORT.md` | Independent EXP-004 re-audit |
| `reports/EXP-005_REPORT.md` | Event-level alarm evaluation result |
| `reports/EXP-005_AUDIT.md` | Independent EXP-005 audit (ACCEPT) |
| `reports/EXP005_ROOT_CAUSE_ANALYSIS.md` | **Stale** root-cause doc (old 10,800-frame run) |
| `reports/INDEPENDENT_SCIENTIFIC_REVIEW.md` | Publication-review output R1 |
| `reports/PUBLICATION_READINESS_ASSESSMENT.md` | Publication-review output R2 |
| `reports/PUBLICATION_RECOVERY_PLAN.md` | Supervisor second-reader plan (C1–C13) |

### 2.5 `reports/archive/` (57)

| Sub-folder | Count | Holds |
|---|---|---|
| `reports/archive/` (root) | 1 | `README.md` archive index |
| `reports/archive/audit_v3.1/` | 13 | v3.1-era audit docs |
| `reports/archive/phase01/` | 9 | Phase-01 process/status docs |
| `reports/archive/phase02/` | 11 | Phase-02 dataset docs + manifest |
| `reports/archive/phase02_5/` | 2 | Phase-02.5 reliability-estimation docs |
| `reports/archive/planning/` | 6 | Early planning/strategy docs |
| `reports/archive/reviews/` | 5 | Early review docs |
| `reports/archive/verification/` | 10 | Early verification docs |

---

## 3. Source-code inventory

### 3.1 `src/` — detection pipeline (19 `.py`)

| File | Role | Note |
|---|---|---|
| `src/config.py` | Central dataclass config (all thresholds/weights) | — |
| `src/detector.py` | EAR + MAR, **both 2D image-plane** | Contradicts "3D EAR" doc claims |
| `src/pose_estimator.py` | Head pose via `solvePnP` | — |
| `src/robustness.py` | 3-component gate, geometric mean `(0.45,0.30,0.25)` | — |
| `src/temporal_analyzer.py` | Speech filter (mean\|ΔMAR\|), PERCLOS, EMA | Filter is mean-abs-delta, not σ² |
| `src/fatigue_fusion.py` | `raw_score *= reliability` (:196-197) | Unconditional multiply before severity |
| `src/state_manager.py` | 5-state hysteresis, SEVERE guard (:361-388) | — |
| `src/frame_processor.py` | Headless per-frame core, `enable_cnn=False` | Shared by app + harness |
| `src/cnn_validator.py` | MicroEyeNet TFLite eye validator | Ablation-only |
| `src/alarm_controller.py` | Alarm decision/debounce | — |
| `src/camera_async.py` | Threaded camera capture | — |
| `src/camera_base.py` | Camera base class | **ORPHAN — imported by nobody** |
| `src/data_loaders.py` | Dataset loaders (incl. quarantine raise) | — |
| `src/dataset_manager.py` | Dataset path/registry management | — |
| `src/main.py` | Live webcam app | Constructs `CNNValidator` at :96 |
| `src/utils/audio_alert.py` | Audio alert helper | — |
| `src/utils/landmark_indices.py` | FaceMesh landmark index constants | — |
| `src/__init__.py`, `src/utils/__init__.py` | Package markers | — |

### 3.2 `evaluation/` — harnesses + verifier (8 `.py`)

| File | Role |
|---|---|
| `evaluation/nthu_ground_truth.py` | NTHU filename → label mapping |
| `evaluation/loso_harness.py` | V0–V4 LOSO, ROC/AUC, fixed-operating-point |
| `evaluation/benchmark_nthan_yawdd.py` | Latency benchmark harness (EXP-001) |
| `evaluation/exp004_report.py` | EXP-004 report generator |
| `evaluation/event_metrics.py` | Event/episode metric math (EXP-005) |
| `evaluation/exp005_event_report.py` | EXP-005 report generator |
| `evaluation/plot_paper_figures.py` | Figure generator (measured-JSON only) |
| `evaluation/verify_integrity.py` | I1–I6 integrity CI gate |

### 3.3 `tools/` — data prep + CNN (12 `.py`)

| File | Role |
|---|---|
| `tools/data_fetcher.py` | Dataset download helper |
| `tools/generate_metadata.py` | Dataset metadata generation |
| `tools/preprocess_data.py` | Preprocessing pipeline |
| `tools/quality_checker.py` | Data quality checks |
| `tools/split_dataset.py` | Generic split helper |
| `tools/dataset_validator.py` | Pre-training dataset validator |
| `tools/build_subject_disjoint_splits.py` | Canonical subject-disjoint splits (seed 42) |
| `tools/train_cnn.py` | Canonical CNN trainer (`load_mrl_dataset_tensors`) |
| `tools/train_exp002_microeyenet.py` | EXP-002 executor |
| `tools/export_and_evaluate_quantization.py` | EXP-003 executor |
| `tools/collect_eye_data.py` | One-off eye-data collector |
| `tools/verify_integrity.py` | Dataset SHA-256 dedup audit (**distinct** from `evaluation/` version) |

### 3.4 `tests/` — test suites (3 `.py`)

| File | Role |
|---|---|
| `tests/test_suite.py` | 17 unit tests |
| `tests/smoke_test.py` | 3 smoke tests (incl. `test_reliability_gate_is_three_component`) |
| `tests/test_event_metrics.py` | ~65 event-metric tests (not reflected in "17+3" doc claims) |

### 3.5 Root-level scratch scripts (4)

| File | Role | Note |
|---|---|---|
| `test_pipeline.py` | Ad-hoc pipeline probe | Not part of the test framework |
| `test_pose.py` | Ad-hoc pose probe | Not part of the test framework |
| `test_variance.py` | Ad-hoc variance probe | Dead-duplicate local `EMASmoother` |
| `test_webcam.py` | 269-line webcam diagnostic | Not part of the test framework |

---

## 4. Experiment-artifact inventory

| Path | Size | Contents |
|---|---|---|
| `experiments/EXP-002_*` | 180 KB | 8 files: training metrics/history/params JSON |
| `experiments/EXP-003_*` | 104 KB | 6 files incl. `eye_state_model_fp32.tflite` |
| `experiments/EXP-004_*` | 17 MB | `exp004_metrics.json`, `per_subject/`, `per_variant/` CSV, `plots/`, `scores/` |
| `experiments/EXP-005_*` | 44 MB | `episodes/`, `event_streams/`, `exp005_event_metrics.json`, `exp005_run.log`, `per_subject/`, `per_variant/` CSV, `plots/` |
| `results/measured_results.json` | 10.7 MB | Canonical measured-results ledger (figure source) |
| `results/*.png` (×2) | — | 2 canonical paper figures |
| `results/measured_results.json.pre_exp004.bak` | 481 B | Pre-EXP-004 snapshot (backup) |
| `models/*.tflite` (×3) | 104 KB | 26488 / 44500 / 26160 B TFLite models + `.gitkeep` |
| `checkpoints/*.keras` (×4) | 1.1 MB | epoch01/02/03/08 (best val_loss 0.1793) |
| `tensorboard/EXP-002_*` | 20 KB | train + validation event files |
| `logs/*` (×3) | 24 KB | Training/run logs (gitignored) |

---

## 5. Dataset inventory (facts on disk)

| Dataset | Count | Subjects | Role | Status |
|---|---|---|---|---|
| NTHU-DDD | 66,521 JPG | 4 (001/002/005/006); 36,030 drowsy / 30,491 notdrowsy | Primary temporal eval | Active |
| MRL Eye | 84,898 PNG | 37 (subject-disjoint split) | CNN-ablation training only | Active |
| YawDD | 348 AVI | — | Listed for yawn eval | **Never evaluated** |
| drowsiness_detection | — | — | 100% byte-duplicate of MRL Eye | **QUARANTINED** (loader raises) |

---

## 6. Junk, empty dirs, and git-tracking status

### 6.1 Removable cruft (OS/interpreter, no information value)

| Item | Count | Note |
|---|---|---|
| `.DS_Store` | 5 | macOS Finder metadata |
| `__pycache__/` | 5 dirs | Python bytecode cache |
| `benchmark/` | 1 dir | Empty directory (0 B) |

### 6.2 Regenerable / foreign (large, not project source)

| Item | Size | Note |
|---|---|---|
| `.venv/` | 1.7 GB | Regenerable from `requirements.txt` |
| `gitlab-vscode-extension-main/` | 8.9 MB | Foreign VS Code extension, gitignored, unreferenced |

### 6.3 Git-tracking status (reproducibility-critical)

Git `HEAD` is a **stale pre-reorg snapshot**: 101 tracked files, but
`evaluation/`, `experiments/`, `results/`, and `checkpoints/` each have **0
tracked files**, and `README.md`, `PROJECT_CONTEXT.md`, `HANDOVER.md`, the
EXP-005 reports, `loso_harness.py`, and `frame_processor.py` are all
**untracked**. The reorg (69 files moved into `reports/archive/` +
`docs/archive/`) is entirely uncommitted. A git-only handover today loses every
measured result, the evaluation code, and the current docs. Detailed in
`REPRODUCIBILITY_CHECK.md`.

---

## 7. Explicitly excluded from project-file counts

| Item | Size | Reason for exclusion |
|---|---|---|
| `.venv/` | 1.7 GB | Regenerable virtualenv; not project source (9 vendored license/readme `.md` files also excluded from the 89-doc census) |
| `gitlab-vscode-extension-main/` | 8.9 MB | Third-party download, gitignored, referenced by nothing in the pipeline |

---

*End of inventory. Classifications and actions are in the sibling
`Repository_Cleanup/` documents — this file records only what exists.*
