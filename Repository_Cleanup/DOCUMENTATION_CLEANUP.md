# DOCUMENTATION CLEANUP

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** List every prose edit needed so the documentation matches the code
and the measured results — nothing more. Each entry gives an exact file:line
target, the current (wrong) text, and the corrected text.

> **Recommendation only.** This document specifies edits; it does not apply
> them. No source code is changed by any edit below — every fix corrects
> *prose about* the code so the docs stop contradicting it. Scientific
> conclusions are untouched.

---

## 0. Two kinds of edit, and one rule

Every edit here falls into one of two buckets, and both obey one rule:

- **Status-flip edits (K1):** six navigation docs still describe EXP-005 as
  *planned / next / not-yet-done*. EXP-005 is **complete and audited ACCEPT**
  (`reports/EXP-005_REPORT.md`, `reports/EXP-005_AUDIT.md`). These docs must
  move EXP-005 from future tense to done.
- **Description-accuracy edits (C1–C5, T1):** prose that describes the code
  incorrectly (3D vs 2D, variance vs mean, "never suppressed", test counts).

**The rule:** when doc and code disagree, **the code is authoritative and the
doc is corrected** — never the reverse. (Cross-referenced in
`CONTRADICTION_REPORT.md`.)

---

## 1. Status-flip edits — EXP-005 is done, not planned (K1)

EXP-005 (event-level alarm evaluation) is COMPLETE: 66,521 frames, recall 0.122,
FA/hr 6.5–9.7, all three observability gates G1/G2/G3 FAIL, audited **ACCEPT**.
Six docs still say it is upcoming. Correct each to past tense and point to the
report.

### 1.1 `README.md`

| Line | Current (wrong) | Corrected |
|---|---|---|
| 20–22 | "…event-level alarm evaluation is the **next step**." | "…event-level alarm evaluation is **complete** — see `reports/EXP-005_REPORT.md` (audited ACCEPT). Result: recall 0.122, FA/hr 6.5–9.7, all three observability gates fail." |

### 1.2 `HANDOVER.md`

| Line | Current (wrong) | Corrected |
|---|---|---|
| 175–185 (§5) | "EXP-005 … **Not yet done**." | "EXP-005 … **Done and audited (ACCEPT)**. Event-level alarm eval over 66,521 frames; recall 0.122, FA/hr 6.5–9.7; only 2 of 4 subjects fire an alarm; G1/G2/G3 all FAIL. See `reports/EXP-005_REPORT.md` and `reports/EXP-005_AUDIT.md`." |

### 1.3 `paper/main.tex`

| Line | Current (wrong) | Corrected |
|---|---|---|
| 116, 135–137 | EXP-005 described in **future tense** | Rewrite as completed work reporting the measured event-level numbers, citing the EXP-005 report. (Author's prose — this file also carries C3/C4 below.) |

### 1.4 `EXPERIMENT_REGISTRY.md`, `AGENT_MEMORY.md`, `IMPLEMENTATION_LOG.md`

| Doc | Fix |
|---|---|
| `EXPERIMENT_REGISTRY.md` | Ensure the EXP-005 row status reads **DONE / AUDITED ACCEPT** with the artifact path, not PLANNED. |
| `AGENT_MEMORY.md` | Update the "next experiment" note that names EXP-005 to "EXP-005 complete; see report." |
| `IMPLEMENTATION_LOG.md` | Append a dated line recording EXP-005 completion + audit outcome. |

---

## 2. Description-accuracy edits — prose must match the code (C1–C5)

Each row cites the code line that is authoritative. The doc is wrong; the code
stays as-is.

### C1 — EAR/MAR are 2D, not 3D

Code: `src/detector.py:55-123` computes **pure 2D** geometric EAR and MAR.

| Doc:line | Current (wrong) | Corrected |
|---|---|---|
| `paper/main.tex:39,46` | "3D EAR" / 3D landmark distances | "2D EAR/MAR from image-plane landmark coordinates" |
| `README.md:36-38` | implies 3D | "EAR and MAR are 2D geometric ratios" |

### C2 — Speech-jitter filter uses **mean |ΔMAR|**, not variance

Code: `src/temporal_analyzer.py:294-298` — mean absolute frame-to-frame ΔMAR,
threshold **0.05**. It is not a variance / σ²(MAR).

| Doc:line | Current (wrong) | Corrected |
|---|---|---|
| `README.md:36-38` | "variance-based σ²(MAR)" | "mean absolute per-frame change in MAR (mean \|ΔMAR\|), threshold 0.05" |
| `HANDOVER.md:37-39` (§1) | "variance filter" | "mean \|ΔMAR\| filter, threshold 0.05" |

### C3 — SEVERE state: guard is state-level, not "never suppressed"

Code: `src/state_manager.py:361-388` applies a SEVERE guard at the **state**
level; the reliability gate still multiplies the **score** upstream
(`src/fatigue_fusion.py:196-197`, `raw_score *= reliability`, unconditional).
So "SEVERE is never suppressed" is inaccurate as written.

| Doc:line | Current (wrong) | Corrected |
|---|---|---|
| `README.md:34` | "SEVERE … never suppressed" | "the reliability gate attenuates the fatigue score for all states; a separate state-level guard governs exit from SEVERE (see `state_manager.py`)" |
| `HANDOVER.md:37-39` (§1) | "SEVERE never suppressed" | same correction as above |
| `paper/main.tex` | if it repeats the claim | align to the state-level-guard wording |

### C4 — YawDD was never evaluated

Datasets: YawDD (348 AVI) is present but **never evaluated** (no EXP row).

| Doc:line | Current (wrong) | Corrected |
|---|---|---|
| `paper/main.tex:22` | YawDD listed among **evaluated** datasets | Move YawDD to "available but not evaluated"; evaluated sets are NTHU-DDD (EXP-004/005) and MRL Eye (EXP-002 CNN only). |

### C5 — Operating point is per-variant nearest TPR, not a single held TPR

Code: `evaluation/loso_harness.py:_fpr_at_tpr (155-162)` picks the **nearest
achievable TPR per variant**; realized matched TPRs are 0.80 / 0.7989 / 0.8006,
not a single constant.

| Doc:line | Current (wrong) | Corrected |
|---|---|---|
| `README.md:94-97` | "TPR held constant across variants" | "FPR is read at each variant's nearest achievable TPR to the 0.80 target; realized TPRs are 0.80/0.7989/0.8006 (`loso_harness.py`)" |
| `HANDOVER.md` (§4) | same "held constant" language | same correction |

---

## 3. Test-count edits — the suite is larger than stated (T1)

Docs say "17 unit + 3 smoke" tests. That predates the event-metric suite
(~65 event-level tests in `tests/`). Understating coverage misleads a reader
about how much is verified.

| Doc:line | Current (wrong) | Corrected |
|---|---|---|
| `README.md:63` | "17 unit + 3 smoke" | "17 unit + 3 smoke + the event-metric suite (~65 event-level tests) — see `tests/`" |
| `HANDOVER.md:156` | "17/17 unit + 3/3 smoke" | add the event-metric suite line with its current pass count |

> Confirm the exact event-test count against `tests/` at edit time and write the
> real number — do not copy "~65" verbatim if the suite has changed.

---

## 4. One numeric reconciliation, not a rewrite (C9)

`CONTRADICTION_REPORT.md` C9 flags subject-006 ROC-AUC reported as **0.372733**
in one place and **0.304675** in another. This is a **reconciliation**, not a
prose rewrite:

1. Read the per-subject AUC from the committed EXP-004 per-subject CSV
   (the raw output under `experiments/EXP-004_*`).
2. Whichever value the CSV holds is authoritative; correct the other doc to it.
3. Do **not** re-run or recompute — the raw artifact already decides it.

If the CSV is missing or ambiguous, stop and flag it as a reproducibility gap in
`REPRODUCIBILITY_CHECK.md` rather than guessing.

---

## 5. Edit order (so nothing is done twice)

1. Apply the **status-flip** edits (§1) — highest reader-impact, lowest risk.
2. Apply the **description-accuracy** edits (§2) doc-by-doc; `paper/main.tex`
   carries C1/C3/C4 together, so edit it once.
3. Apply the **test-count** edits (§3) after confirming the live count.
4. Resolve the **C9 number** (§4) from the CSV.
5. Re-read each edited doc top-to-bottom once, checking no other sentence still
   repeats a just-corrected claim.

---

## 6. Summary

| Class | Edits | Files touched |
|---|---|---|
| Status-flip (K1) | EXP-005 planned → done | README, HANDOVER, paper, EXPERIMENT_REGISTRY, AGENT_MEMORY, IMPLEMENTATION_LOG |
| Accuracy (C1–C5) | 2D not 3D; mean\|ΔMAR\| not variance; state-level SEVERE guard; YawDD not evaluated; per-variant TPR | README, HANDOVER, paper |
| Test count (T1) | add event-metric suite | README, HANDOVER |
| Reconcile (C9) | subject-006 AUC from CSV | one of the two disagreeing docs |

Every edit corrects a document to match code or a committed artifact. **No code,
no result, and no scientific conclusion is altered.** Nothing here is executed
until you approve.



