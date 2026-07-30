# EXP-002 — Dataset Verification Report

**Experiment:** EXP-002 (MicroEyeNet Training)
**Date:** 2026-07-28
**Scope:** MRL Eye dataset, subject-disjoint split
**Source:** `Data/mrl_eye/splits_subject_disjoint/{train,val,test}.csv`
**Method:** Every property below was **measured** by reading the manifests and
decoding the referenced image files (see commands in Phase 3). Nothing is
estimated.

---

## 1. Split source

- Loader accessor: `MRLEyeDataLoader.get_subject_disjoint_files(split)`
  (added in EXP-002 Phase 2, leak-free).
- Manifests: `Data/mrl_eye/splits_subject_disjoint/{split}.csv`
- Manifest columns: `filepath, label, subject_id`
- Generator: `tools/build_subject_disjoint_splits.py` (seed 42,
  TRAIN_FRAC=0.70, VAL_FRAC=0.15).

## 2. Subject-disjointness (MEASURED)

| Split | Subjects | Images |
|-------|----------|--------|
| train | 26       | 70,551 |
| val   | 6        | 4,970  |
| test  | 5        | 9,377  |
| **total** | **37** | **84,898** |

Pairwise subject intersections (MEASURED):

- train ∩ val  = ∅
- train ∩ test = ∅
- val ∩ test   = ∅

**Zero subject overlap confirmed.**

## 3. Labels (MEASURED)

- `class_map = {"awake": 0, "sleepy": 1}` → **awake = 0, sleepy = 1** (matches frozen contract).
- Observed label domain across all splits: `{0, 1}` (no other values).

## 4. Class counts (MEASURED)

| Split | class 0 (awake) | class 1 (sleepy) | total |
|-------|-----------------|------------------|-------|
| train | 38,658          | 31,893           | 70,551 |
| val   | 2,945           | 2,025            | 4,970  |
| test  | 1,349           | 8,028            | 9,377  |

Note: the test split is class-imbalanced toward sleepy (86% class 1). This is a
property of the frozen subject-disjoint assignment (5 fixed subjects) and is
**not** modified — balanced accuracy / F1 / PR-AUC are reported at evaluation
time to account for it.

## 5. Image format (MEASURED, sampled ≈200 per split)

- Channel count: **1 (grayscale)** for every sampled image.
- Raw dtype: **uint8**.
- Raw spatial size: variable (130 distinct raw shapes observed, e.g. 56×56 … 81×81+);
  handled by the resize step below.

## 6. Preprocessing pipeline (MEASURED)

Applied identically at training and inference:

1. Read grayscale (`cv2.IMREAD_GRAYSCALE`).
2. Resize to **24×24** via `cv2.INTER_AREA`.
3. Normalize `/255.0` → **float32**.
4. Add channel axis → final tensor shape **(24, 24, 1)**.

Verified on a decoded sample:
- final shape = `(24, 24, 1)`
- dtype = `float32`
- value range ⊆ **[0, 1]** (min 0.1608, max 0.2941 on the sample; formula bounds guarantee [0,1]).

## 7. File integrity (MEASURED)

- Missing files across all three splits: **0** (all 84,898 referenced paths exist).

---

## Verdict

All frozen dataset requirements are satisfied by measurement:
subject-disjoint (zero overlap), correct label polarity (awake=0/sleepy=1),
grayscale, 24×24 INTER_AREA, /255 float32 normalization, shape (24,24,1),
class/subject counts matching `SPLIT_MANIFEST.md`. **Dataset is cleared for
EXP-002 training.**
