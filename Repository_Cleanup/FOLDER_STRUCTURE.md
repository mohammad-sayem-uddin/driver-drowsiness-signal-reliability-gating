# FOLDER STRUCTURE

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** Show the repository's folder organization — as it is now, and as it
should be after the (approved) archive plan — so the next researcher can
navigate by folder alone and always know where a thing lives and whether it is
authoritative.

> The tree below is **organizational**. It moves superseded prose and scratch
> aside; it never relocates code, datasets, measured artifacts, or figures. The
> physical moves are staged in `ARCHIVE_PLAN.md`; nothing here is executed.

---

## 1. Target repository tree (after approved cleanup)

```
Driver Drowsiness/
├── README.md                      # FINAL — front door + read-order (EXP-005 status fix)
├── PROJECT_CONTEXT.md             # FINAL — declared single source of truth
├── HANDOVER.md                    # FINAL — supervisor/examiner handover
├── EXPERIMENT_REGISTRY.md         # FINAL — experiment ledger (no number citable w/o a row)
├── IMPLEMENTATION_LOG.md          # ACTIVE — why the repo looks as it does
├── AGENT_MEMORY.md                # ACTIVE — fast-start brief
├── CNN_IMPLEMENTATION_SPECIFICATION.md  # ACTIVE — ablation-arm CNN spec
├── requirements.txt               # FINAL — environment contract
├── .gitignore                     # (updated: .DS_Store, __pycache__, .venv, foreign dl)
│
├── src/                           # FINAL — the single per-frame pipeline (17 real modules)
├── evaluation/                    # FINAL — LOSO/event harness + integrity gate (8 .py)
├── tools/                         # FINAL/ACTIVE — data-prep + experiment executors (12 .py)
├── tests/                         # FINAL — the 3 real verification suites
│
├── experiments/                   # FINAL EVIDENCE — EXP-002/003/004/005 raw artifacts (62 MB)
├── results/                       # FINAL EVIDENCE — measured_results.json + 2 figures + .bak
├── models/                        # FINAL EVIDENCE — 3 .tflite quantization outputs
├── checkpoints/                   # FINAL EVIDENCE — 4 .keras training checkpoints
├── tensorboard/                   # FINAL EVIDENCE — EXP-002 training curves
├── logs/                          # ACTIVE — gitignored run logs
│
├── Data/                          # DATASETS (7.9 GB) + canonical integrity report
├── paper/                         # manuscript (main.tex — needs C3/C4 prose fixes)
│
├── reports/                       # CURRENT reports (11) + archive/ (see §3)
├── docs/                          # docs/archive/ only
├── scratch/                       # NEW — non-authoritative dev probes (see §4)
│
└── Repository_Cleanup/            # THIS cleanup dossier (9 planning docs)
```

---

## 2. Per-folder navigation guide (what lives here / what is authoritative)

The single question this table answers: *"I need X — which folder, and which
file in it do I trust?"*

| Folder | What lives here | Authoritative entry point |
|---|---|---|
| *(root)* | 7 governing docs + `requirements.txt` | `PROJECT_CONTEXT.md` (design/scope/status) |
| `src/` | The detection pipeline — one per-frame path shared by live app and offline harness | `src/frame_processor.py` (the shared core) |
| `evaluation/` | LOSO + event harness, latency benchmark, integrity gate | `evaluation/loso_harness.py`; gate `evaluation/verify_integrity.py` (I1–I6) |
| `tools/` | Data-prep + the four experiment executors | `tools/build_subject_disjoint_splits.py` (splits, seed 42) |
| `tests/` | The three real suites (unit / smoke / event-metric) | `tests/test_suite.py` |
| `experiments/` | Raw committed evidence per experiment | the `EXP-00N_*` dir named in the matching report |
| `results/` | Canonical numbers + the two paper figures | `results/measured_results.json` |
| `models/`, `checkpoints/`, `tensorboard/` | EXP-002/003 model + training evidence | cited by their reports; never edited by hand |
| `Data/` | The four datasets + integrity report | `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md` |
| `paper/` | The manuscript | `paper/main.tex` (pending C3/C4 prose fixes) |
| `reports/` | The 11 current reports (see §3) | `reports/README.md` index |
| `docs/` | Nothing current — archive only | `docs/archive/README.md` |
| `scratch/` | Ad-hoc dev probes, non-authoritative | `scratch/README.md` (marks it non-authoritative) |
| `Repository_Cleanup/` | This freeze dossier | `FINAL_REPOSITORY_STATUS.md` |

---

## 3. `reports/` — current vs archive (the folder most likely to confuse)

After the plan, `reports/` holds **11 current docs** (down from 14). Everything
else lives under `reports/archive/` and must not be cited.

```
reports/
├── README.md                              # ACTIVE — index; current/archive split
├── IMPLEMENTATION_SPECIFICATION_FROZEN.md # FINAL — frozen engineering contract
├── EXP-002_REPORT.md                      # FINAL — MicroEyeNet training
├── EXP-002_DATASET_VERIFICATION.md        # FINAL — EXP-002 dataset integrity
├── EXP-002_PARAMETER_AUDIT.md             # FINAL — 19,745-param audit
├── EXP-003_REPORT.md                      # FINAL — INT8/FP16 quantization
├── EXP-004_REPORT.md                      # FINAL — LOSO V0–V4 (negative)
├── EXP-004_AUDIT/                         # FINAL — independent EXP-004 re-audit
├── EXP-005_REPORT.md                      # FINAL — event-level alarm eval
├── EXP-005_AUDIT.md                       # FINAL — EXP-005 audit (ACCEPT)
├── PUBLICATION_RECOVERY_PLAN.md           # FINAL — the live C1–C13 to-do list
└── archive/
    ├── README.md
    ├── audit_v3.1/ phase01/ phase02/ phase02_5/
    ├── planning/ reviews/ verification/
    ├── publication_review_2026-07-30/     # NEW — R1 + R2 + their 2 input prompts + README
    └── superseded/
        └── EXP005_ROOT_CAUSE_ANALYSIS.md  # SUPERSEDED — banner → EXP-005_REPORT.md
```

**What changed vs before the plan (all documentation-tidy, no evidence moved):**
- `EXP005_ROOT_CAUSE_ANALYSIS.md` → `archive/superseded/` (stale 10,800-frame run; C7).
- `INDEPENDENT_SCIENTIFIC_REVIEW.md` (R1) + `PUBLICATION_READINESS_ASSESSMENT.md`
  (R2) → `archive/publication_review_2026-07-30/`, joined by the two root Prompt
  files that produced them.
- `PUBLICATION_RECOVERY_PLAN.md` **stays** top-level — it is the newest, live doc.

---

## 4. `scratch/` — the new home for non-authoritative dev probes

Four ad-hoc test scripts currently sit in the repository **root**, where they
read as if they were part of the project's real test suite. They are not: the
authoritative suites live in `tests/`. Moving them into a clearly-named,
clearly-non-authoritative folder is the single change that removes the most
"which of these do I trust?" ambiguity for the next reader.

```
scratch/
├── README.md          # NEW — one line: "Ad-hoc dev probes. Non-authoritative.
│                      #        The real suites are in tests/. Nothing here is cited."
└── archive/
    ├── test_*.py      # the 4 root-level probe scripts, moved verbatim
    └── ...            # (see ARCHIVE_PLAN.md §5 for the exact filenames + the
                       #  pytest-discovery caution when they move)
```

**Why a folder and not a delete:** the probes are historical dev context, not
evidence, and the "prefer archiving over deleting" rule applies. Isolating them
under `scratch/archive/` keeps them for reference while ensuring `pytest`'s
default discovery no longer sweeps them in alongside the real suites.

---

## 5. What is *not* in the target tree (removed, with justification)

These appear in the current repo but are absent above — each is either junk or a
foreign import, and each removal is justified in `ARCHIVE_PLAN.md` §6:

| Path | Why it is gone |
|---|---|
| `benchmark/` (empty) | Empty directory, no tracked contents — nothing to preserve. |
| `gitlab-vscode-extension-main/` (8.9 MB) | Foreign download, unrelated to this project; not evidence. |
| `**/.DS_Store` (5) | macOS Finder metadata; now `.gitignore`d. |
| `**/__pycache__/` (5) | Python bytecode caches; regenerated on run; now `.gitignore`d. |

`.venv/` (1.7 GB) is **kept on disk** but `.gitignore`d — it is a rebuildable
environment, not evidence, and deleting it would break the working setup for no
archival benefit (`ARCHIVE_PLAN.md` §7).

---

> **Recommendation only.** This document describes the *target* organization.
> No folder is created, moved, or removed by this file. The physical operations
> — and their exact, reversible commands — are staged in `ARCHIVE_PLAN.md` and
> ordered in `CLEANUP_CHECKLIST.md`, to be run only after your approval.




