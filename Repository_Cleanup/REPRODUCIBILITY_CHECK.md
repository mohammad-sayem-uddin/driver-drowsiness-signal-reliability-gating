# REPRODUCIBILITY CHECK

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** Verify that every cited result can be reproduced from what is in the
repository — that each experiment has its **code, config, dataset, report,
figures, and raw outputs** present and linked. Flag only genuine gaps.

> **Recommendation only.** This is an audit. It changes no file and reruns no
> experiment. Where an artifact is present it is recorded as present; where one
> is missing the gap is named so it can be closed before archival.

---

## 0. What "reproducible" means here

A result is reproducible in this repo when all six are present and cross-linked:

1. **Code** — the exact module(s) that produced it.
2. **Config** — seed, split definition, thresholds, operating point.
3. **Dataset** — the specific frames/subjects, identifiable and present.
4. **Report** — an `EXP-###` doc stating the numbers.
5. **Figures** — any plots the report or paper cites.
6. **Raw outputs** — the committed JSON/CSV the numbers were read from.

The integrity regime (`evaluation/verify_integrity.py`, invariants I1–I6)
already enforces that **no number is citable without an `EXP-###` row and a
committed artifact** — this check confirms that regime holds file-by-file.

---

## 1. Per-experiment coverage matrix

Legend: ✅ present & linked · ⚠️ present but needs a doc pointer · ❌ missing.

### EXP-001 — Host latency benchmark

| Artifact | Status | Location / note |
|---|---|---|
| Code | ✅ | `evaluation/` latency benchmark |
| Config | ✅ | seed 42, 300 NTHU frames, Darwin-arm64 host (explicitly **not** Pi 4) |
| Dataset | ✅ | 300 NTHU-DDD frames |
| Report | ✅ | latency numbers recorded (3.205 ms/frame mean, p95 3.316, ~312 FPS) |
| Figures | n/a | no figure cited |
| Raw outputs | ✅ | `results/measured_results.json` |

**Caveat to preserve, not fix:** the 29.27 ms max is a cold-start outlier; the
host-vs-Pi distinction must stay explicit in any doc citing this number.

### EXP-002 — MicroEyeNet training (subject-disjoint MRL)

| Artifact | Status | Location / note |
|---|---|---|
| Code | ✅ | training executor in `tools/` |
| Config | ✅ | subject-disjoint MRL split, early-stop epoch 13 (best 8) |
| Dataset | ✅ | MRL Eye (84,898 PNG, 37 subjects) |
| Report | ✅ | `reports/EXP-002_REPORT.md` + `EXP-002_DATASET_VERIFICATION.md` + `EXP-002_PARAMETER_AUDIT.md` |
| Figures | ✅ | `tensorboard/` training curves |
| Raw outputs | ✅ | `checkpoints/` (4 .keras), metrics in report (VAL acc 0.9402/F1 0.9262; TEST acc 0.9362/F1 0.9623) |

### EXP-003 — INT8/FP16 quantization

| Artifact | Status | Location / note |
|---|---|---|
| Code | ✅ | quantization executor in `tools/` |
| Config | ✅ | no retraining; INT8 + FP16 from EXP-002 checkpoint |
| Dataset | ✅ | inherits EXP-002 test split |
| Report | ✅ | `reports/EXP-003_REPORT.md` (INT8 25.55 KB, 3.18×, −0.026% F1; FP16 43.46 KB, 0% loss) |
| Figures | n/a | table-only |
| Raw outputs | ✅ | `models/` (3 .tflite) |

### EXP-004 — LOSO V0–V4 gate/filter ablation (honest negative)

| Artifact | Status | Location / note |
|---|---|---|
| Code | ✅ | `evaluation/loso_harness.py` (`_fpr_at_tpr`, `_fix_operating_point`) |
| Config | ✅ | LOSO/GroupKFold, seed 42, target_tpr 0.80, realized 0.80/0.7989/0.8006 |
| Dataset | ✅ | NTHU-DDD (66,521 frames, 4 subjects) |
| Report | ✅ | `reports/EXP-004_REPORT.md` + `reports/EXP-004_AUDIT/` |
| Figures | ✅ | ROC/operating-point figures in `results/` |
| Raw outputs | ⚠️ | per-variant metrics present; **per-subject CSV must be confirmed present** to close C9 (subject-006 AUC 0.372733 vs 0.304675) |

**Two facts to preserve verbatim:** V4 ≡ V3 byte-identical
(md5 `f8c298c8a7f521011ad9317da0b9c9b5`), and the result is a **negative** — the
gate does not improve FPR (V2 0.6244 vs V0 0.6241). Neither is a defect to fix.

**Gap G-1 (C9):** confirm the EXP-004 per-subject AUC CSV is committed. If
present, C9 is a doc reconciliation only (see `DOCUMENTATION_CLEANUP.md` §4). If
absent, this is a real reproducibility gap — the disagreeing subject-006 number
cannot be adjudicated without it.

### EXP-005 — Event-level alarm evaluation (complete, audited ACCEPT)

| Artifact | Status | Location / note |
|---|---|---|
| Code | ✅ | event harness in `evaluation/` |
| Config | ✅ | 66,521 frames, 4 subjects, wall-clock 46.5 min (2790.88 s) |
| Dataset | ✅ | NTHU-DDD (same 66,521 frames) |
| Report | ✅ | `reports/EXP-005_REPORT.md` + `reports/EXP-005_AUDIT.md` (ACCEPT) |
| Figures | ✅ | event-level plots in `results/` |
| Raw outputs | ✅ | committed event/episode outputs (recall 0.122, FA/hr 6.5–9.7, G1/G2/G3 FAIL, 2 of 4 subjects fire) |

**Superseded artifact:** the old 10,800-frame run
(`EXP005_ROOT_CAUSE_ANALYSIS.md`, recall 0.0, 16/16 GT missed) is **not** the
result of record — it must carry the SUPERSEDED banner and move to
`reports/archive/superseded/` (`ARCHIVE_PLAN.md` §3). Keeping it unbannered in
place is itself a reproducibility hazard (two "EXP-005" numbers, one stale).

---

## 2. The one blocking reproducibility finding: git HEAD is stale

This is the single most important item in this document. **The working tree is
complete and reproducible; the committed history is not.** A handover done by
`git clone` alone would lose most of the evidence above.

Observed git state at audit time:

- HEAD tracks **101 files**, but `evaluation/`, `experiments/`, `results/`, and
  `checkpoints/` each have **0 tracked files** — the code and evidence that
  produce and hold every EXP result are **untracked on disk**.
- Untracked (present on disk, not in git): `README.md`, `PROJECT_CONTEXT.md`,
  `HANDOVER.md`, the EXP-005 reports, `evaluation/loso_harness.py`,
  `src/frame_processor.py`.
- 69 files git reports as "deleted" were **moved** into `reports/archive/` and
  `docs/archive/` during the reorg — they are **not lost**, but the entire
  reorg is **uncommitted**.

**Impact:** every ✅ in §1 is a *working-tree* ✅. On a fresh clone they become
❌. This is the gap that most threatens the professor handover.

**Gap G-2 (BLOCKER for archival):** the reorganized working tree must be
committed so that the tracked repository equals the reproducible repository.
This is the one place this audit recommends an action beyond documentation —
because a missing commit *is* a missing artifact under the six-part definition
in §0. Per git-safety rules, this commit is **staged for your approval**, made
on a branch, and is listed as the first item in `CLEANUP_CHECKLIST.md`; it is
not performed by this document.

---

## 3. Non-experiment reproducibility items

| Item | Status | Note |
|---|---|---|
| Environment | ✅ | `requirements.txt` pins protobuf>=4.25.3,<5 and tensorflow==2.17.1; jax must NOT be installed |
| Splits | ✅ | `tools/build_subject_disjoint_splits.py`, seed 42 — regenerable |
| Integrity gate | ✅ | `evaluation/verify_integrity.py` enforces I1–I6 |
| Quarantined data | ✅ | `drowsiness_detection/` is a 100% byte-dup of MRL; loader raises RuntimeError by design — a *safeguard*, not a gap |
| YawDD | ✅ (n/a) | present, never evaluated; not part of any result, so no reproducibility obligation |

---

## 4. Summary

| Finding | Severity | Resolution |
|---|---|---|
| EXP-001/002/003/005 fully reproducible in working tree | — | none needed |
| EXP-004 per-subject AUC CSV (G-1, C9) | verify | confirm CSV committed; else real gap |
| Stale `EXP005_ROOT_CAUSE_ANALYSIS.md` unbannered | MAJOR | banner + archive (`ARCHIVE_PLAN.md` §3) |
| **Git HEAD stale — evidence untracked (G-2)** | **BLOCKER** | commit the reorganized tree on a branch (`CLEANUP_CHECKLIST.md` step 1) |

With G-2 committed and G-1 confirmed, every cited result has code + config +
dataset + report + figures + raw outputs, all tracked. **No experiment needs to
be re-run** — the artifacts already exist; they only need to be committed and,
in one case, banner-corrected. Nothing here is executed until you approve.



