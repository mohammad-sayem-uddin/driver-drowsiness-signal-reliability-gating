# FINAL REPOSITORY STATUS

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** The one-page answer to "what is this repository, what is
authoritative in it, and what must I never cite?" — written for the professor
who opens it cold, after the cleanup plan has been approved and executed.

> **Recommendation only (until the checklist runs).** This document describes
> the repository's status *as it will stand once `CLEANUP_CHECKLIST.md` is
> approved and executed*. Nothing here changes a scientific conclusion.

---

## 1. What this repository is

A real-time driver-drowsiness detection system whose research contribution is a
**signal-reliability gate** — not a new classifier. The gate multiplicatively
attenuates fatigue evidence before temporal accumulation, using a decomposed
reliability score (landmark-stability, brightness-quality, cue-consistency;
weighted geometric mean, weights 0.45/0.30/0.25). A second, smaller contribution
is a speech-jitter MAR filter (mean |ΔMAR|, threshold 0.05).

The pipeline is one fixed per-frame path (`src/frame_processor.py`), shared by
the live app and the offline evaluation harness.

---

## 2. The single source of truth

**`PROJECT_CONTEXT.md`** is the declared single source of truth for design,
scope, and status. When any other document disagrees with it or with the code,
the code and `PROJECT_CONTEXT.md` win.

---

## 3. Read order for a new reader

1. `PROJECT_CONTEXT.md` — what the project is and where it stands.
2. `README.md` — front door and orientation.
3. `HANDOVER.md` — supervisor/examiner handover.
4. `EXPERIMENT_REGISTRY.md` — the ledger; no number is citable without a row.
5. `reports/PUBLICATION_RECOVERY_PLAN.md` — the live to-do list (C1–C13).
6. The `reports/EXP-00N_*` reports, in number order, for the evidence.

---

## 4. What is authoritative, by topic

| Topic | Authoritative source |
|---|---|
| Design / scope / status | `PROJECT_CONTEXT.md` |
| The pipeline | `src/frame_processor.py` and the modules it calls |
| Evaluation method | `evaluation/loso_harness.py` + `verify_integrity.py` (I1–I6) |
| Which results exist | `EXPERIMENT_REGISTRY.md` |
| Latency | EXP-001 → `results/measured_results.json` |
| CNN training | EXP-002 → `reports/EXP-002_REPORT.md` |
| Quantization | EXP-003 → `reports/EXP-003_REPORT.md` |
| Gate/filter ablation (negative) | EXP-004 → `reports/EXP-004_REPORT.md` |
| Event-level alarm eval | EXP-005 → `reports/EXP-005_REPORT.md` + `EXP-005_AUDIT.md` |
| Publication standing | `reports/PUBLICATION_RECOVERY_PLAN.md` (newest review) |
| Datasets integrity | `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md` |

---

## 5. What must never be cited

| Do not cite | Because |
|---|---|
| `reports/archive/superseded/EXP005_ROOT_CAUSE_ANALYSIS.md` | stale 10,800-frame run; superseded by EXP-005 (recall 0.122, not 0.0) |
| anything under `reports/archive/` or `docs/archive/` | archived by design; kept for history, not for citation |
| `reports/archive/publication_review_2026-07-30/` (R1, R2, prompts) | folded into `PUBLICATION_RECOVERY_PLAN.md`, which supersedes them |
| `scratch/` | ad-hoc dev probes; the real suites are in `tests/` |
| `drowsiness_detection/` dataset | quarantined byte-dup of MRL; loader raises by design |
| YawDD results | none exist — YawDD was never evaluated |

---

## 6. Honest standing of the science (unchanged, not re-judged)

This cleanup does **not** re-review the research. For the record, and as already
established in the frozen reports:

- EXP-004 is an **honest negative**: the reliability gate does not improve
  FPR@matched-TPR on NTHU-DDD (V2 0.6244 vs V0 0.6241); V4 ≡ V3 byte-identical.
- EXP-005 shows low recall (0.122) and all three observability gates failing.
- The newest publication review (`PUBLICATION_RECOVERY_PLAN.md`) records the
  standing and its C1–C13 recovery items.

These conclusions are preserved verbatim. The cleanup's job was to make them
easy to find and impossible to confuse with superseded versions.

---

## 7. Status line

**After the approved checklist runs:** single source of truth established
(`PROJECT_CONTEXT.md`); every result has code + config + dataset + report +
figures + raw outputs, all committed; superseded and scratch material archived
and banner-marked; documentation matches the code; the working tree equals the
tracked repository. The one open dependency is executing
`CLEANUP_CHECKLIST.md` after your approval — until then, this status is the
*target*, not the *current* state.

