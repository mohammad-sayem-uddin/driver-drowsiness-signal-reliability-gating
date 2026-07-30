# EXP-004 — Leave-One-Subject-Out (LOSO) Ablation Evaluation on NTHU-DDD

**Experiment ID:** EXP-004
**Date:** 2026-07-28
**Status:** Completed (measured, logged artifacts). Process exit code 0.
**Author role context:** executed under the frozen research design
(`reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md`); no frozen algorithm,
threshold, weight, split, or configuration was modified.

> **Truth policy.** Every number in this report was produced by executing the
> frozen pipeline over real NTHU-DDD frames. Nothing is estimated, interpolated,
> smoothed, or invented. Where an effect is absent or negative, it is reported
> as such.

---

## 1. Objective

Run the frozen Leave-One-Subject-Out ablation (variants **V0–V4**) over the full
NTHU-DDD dataset and record the first honestly-measured **accuracy / ROC**
results for the system, in order to test the project's primary research
hypothesis:

> *Does the decomposed signal-reliability gate (and the σ²(MAR) speech-jitter
> filter) reduce false-positive rate (FPR) at a matched true-positive rate (TPR)
> versus a plain weighted-fusion baseline?*

The primary metric is **FPR @ matched TPR = 0.80**, with the operating point
fixed on the **V0 baseline** ROC and held constant across all variants.

---

## 2. Method (exactly as frozen — nothing changed)

- **Dataset:** NTHU-DDD, `Data/nthu_ddd/`. 66,521 labelled JPG frames, 4
  subjects (001, 002, 005, 006). Labels parsed from filenames by
  `evaluation/nthu_ground_truth.py` (`drowsy→1`, `notdrowsy→0`); no pixels are
  used for labelling and nothing is fabricated.
- **Split:** subject-disjoint LOSO, seed 42, deterministic frame order
  (frames sorted by `(subject, condition, frame_index)`).
- **Timing:** offline **video clock** `ts = frame_index / 30.0` fps, per the
  frozen protocol (not wall-clock).
- **Sweep variable:** `FrameResult.fatigue_score ∈ [0,1]` (the continuous ROC
  threshold variable).
- **Variants** (`(speech_filter, reliability_gate, cnn)` toggles):
  - **V0** `V0_baseline` `(F, F, F)`
  - **V1** `V1_speech_filter` `(T, F, F)`
  - **V2** `V2_reliability_gate` `(F, T, F)`
  - **V3** `V3_full` `(T, T, F)`
  - **V4** `V4_full_cnn` `(T, T, T)`
- **ROC / AUC / operating point:** computed by the **unmodified** frozen helpers
  in `evaluation/loso_harness.py` (`_roc_curve`, `_auc` [trapezoid],
  `_fix_operating_point(target_tpr=0.80)`, `_fpr_at_tpr`). The canonical `roc`
  block written to `results/measured_results.json` is produced by the frozen
  `LOSOHarness` writer and is byte-equivalent to `loso_harness.py --write`.
- **Extended descriptive metrics:** an additive layer
  (`evaluation/exp004_report.py`) imports the frozen functions unmodified and
  derives Accuracy, Precision, Recall/TPR, Specificity, F1, FPR, PR-AUC,
  balanced accuracy, and confusion matrices from the *same* `(score, label)`
  pairs, at the fixed V0 operating threshold. This is permitted by
  PROJECT_CONTEXT §7 ("MAY extend: add new results artifacts") and changes no
  frozen logic.
- **`sklearn` is absent** from `.venv`; all metrics are hand-rolled numpy with
  standard definitions (same methodology as EXP-002/EXP-003).
- **Wall clock:** 24.8 min for the complete 5-variant × 4-subject sweep
  (20 subject-variant evaluations) on this Darwin-arm64 host.

**Frame accounting (matches ground truth exactly — no decode drops):**

| Subject | Frames | Drowsy | NotDrowsy |
|:--|--:|--:|--:|
| 001 | 19,016 | 9,584 | 9,432 |
| 002 | 18,833 | 10,596 | 8,237 |
| 005 | 21,933 | 13,087 | 8,846 |
| 006 | 6,739 | 2,763 | 3,976 |
| **Total** | **66,521** | **36,030** | **30,491** |

---

## 3. Operating point (fixed on V0, held constant)

| Field | Value |
|:--|--:|
| Definition | V0 baseline score threshold at TPR closest to target=0.80 |
| Target TPR | 0.80 |
| Fixed score threshold | 2.588e-08 (≈0; see §7 for interpretation) |
| V0 TPR at threshold | 0.800000 |
| V0 FPR at threshold | 0.624119 |

The threshold sits at essentially zero because the fatigue score distribution is
strongly zero-inflated: **22.6 %** of all V0 frames (and **30.3 %** of the
not-drowsy frames) have `fatigue_score == 0`. To reach TPR = 0.80 the operating
point must admit almost every non-zero score, which necessarily also admits a
large fraction of non-drowsy frames — hence the high FPR. This is a measured
property of the frozen scorer on frame-level NTHU labels, reported here without
adjustment.

---

## 4. Overall results — all variants (N = 66,521 frames, 4 subjects)

Metrics below are at the **fixed V0 operating threshold** (confusion-matrix
metrics), plus threshold-free **ROC-AUC / PR-AUC** and the primary
**FPR @ matched TPR = 0.80**.

| V | Name | ROC-AUC | PR-AUC | Acc | Prec | Recall/TPR | Spec | F1 | **FPR@mTPR** |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| V0 | V0_baseline | 0.628786 | 0.637354 | 0.605598 | 0.602332 | 0.800000 | 0.375881 | 0.687235 | **0.624119** |
| V1 | V1_speech_filter | 0.617005 | 0.628353 | 0.597450 | 0.600500 | 0.767166 | 0.396904 | 0.673678 | **0.669411** |
| V2 | V2_reliability_gate | 0.624539 | 0.621659 | 0.606560 | 0.603477 | 0.797835 | 0.380539 | 0.687177 | **0.624447** |
| V3 | V3_full | 0.613299 | 0.614859 | 0.598322 | 0.601678 | 0.764530 | 0.401922 | 0.673398 | **0.669017** |
| V4 | V4_full_cnn | 0.613299 | 0.614859 | 0.598322 | 0.601678 | 0.764530 | 0.401922 | 0.673398 | **0.669017** |

**Confusion matrices at the fixed operating threshold** (TP / FP / TN / FN):

| V | TP | FP | TN | FN |
|:--|--:|--:|--:|--:|
| V0 | 28,824 | 19,030 | 11,461 | 7,206 |
| V1 | 27,641 | 18,389 | 12,102 | 8,389 |
| V2 | 28,746 | 18,888 | 11,603 | 7,284 |
| V3 | 27,546 | 18,236 | 12,255 | 8,484 |
| V4 | 27,546 | 18,236 | 12,255 | 8,484 |

---

## 5. Per-subject results (fixed V0 threshold)

Full table in `experiments/EXP-004_loso/per_subject_metrics.csv`. Selected
columns (Accuracy / Recall / Specificity / ROC-AUC):

| V | Subj | N | Acc | Recall | Spec | ROC-AUC |
|:--|:--|--:|--:|--:|--:|--:|
| V0 | 001 | 19,016 | 0.607488 | 0.729341 | 0.483673 | 0.625401 |
| V0 | 002 | 18,833 | 0.596294 | 0.832390 | 0.292582 | 0.620676 |
| V0 | 005 | 21,933 | 0.692290 | 0.916176 | 0.361067 | 0.718970 |
| V0 | 006 | 6,739 | 0.344116 | 0.370612 | 0.325704 | 0.372733 |
| V2 | 001 | 19,016 | 0.608277 | 0.724750 | 0.489928 | 0.623841 |
| V2 | 002 | 18,833 | 0.596772 | 0.830974 | 0.295496 | 0.609187 |
| V2 | 005 | 21,933 | 0.694114 | 0.915030 | 0.367285 | 0.717901 |
| V2 | 006 | 6,739 | 0.344116 | 0.369164 | 0.326710 | 0.373615 |
| V3 | 001 | 19,016 | 0.608961 | 0.696473 | 0.520038 | 0.639794 |
| V3 | 002 | 18,833 | 0.591515 | 0.820404 | 0.297074 | 0.587598 |
| V3 | 005 | 21,933 | 0.676105 | 0.858256 | 0.406624 | 0.687144 |
| V3 | 006 | 6,739 | 0.334174 | 0.342381 | 0.328471 | 0.359070 |

**Subject 006 is a clear outlier** (per-subject ROC-AUC ≈ 0.36–0.37, below
chance) across every variant. Its label balance is inverted relative to the
others (more not-drowsy than drowsy) and the frozen scorer generalises poorly to
it under LOSO — a genuine subject-independence weakness, reported, not hidden.

---

## 6. Variant comparison — does the hypothesis hold?

**Primary metric — FPR @ matched TPR = 0.80 (lower is better):**

| Comparison | FPR@mTPR | Δ vs V0 | Verdict |
|:--|--:|--:|:--|
| V0 baseline | 0.624119 | — | reference |
| V2 reliability gate (isolated) | 0.624447 | **+0.000328** | no improvement (flat) |
| V1 speech filter (isolated) | 0.669411 | +0.045292 | worse |
| V3 full (speech + gate) | 0.669017 | +0.044898 | worse |
| V4 full + CNN | 0.669017 | +0.044898 | worse |

**Measured conclusion: the primary hypothesis is NOT supported on frame-level
NTHU-DDD at this operating point.** The reliability gate in isolation (V2) leaves
FPR essentially unchanged (+0.0003, i.e. flat within noise), and every variant
that includes the speech-jitter filter (V1, V3, V4) *increases* FPR at matched
TPR. ROC-AUC likewise does not improve over baseline (V0 = 0.6288 is the highest;
V3/V4 = 0.6133 the lowest). This is a **negative / null result** for the stated
claim under these evaluation conditions, and is reported as such.

**Why V4 ≡ V3 exactly (byte-identical scores).** This was verified directly
(the two per-variant score CSVs are byte-for-byte identical; md5
`f8c298c8a7f521011ad9317da0b9c9b5`). The root cause is structural in the frozen
code, not an evaluation error:

- The ROC sweep variable is `fusion.fatigue_score`.
- `FatigueFusionEngine.update(ts, reliability)` (`src/fatigue_fusion.py:154`)
  takes **no CNN input** — the fused score cannot depend on the CNN.
- The CNN verdict enters only in `StateManager.update`
  (`src/state_manager.py:374–388`), where it can flip the boolean
  `should_alarm` (alarm suppression), but it **never modifies** `fatigue_score`
  (returned unchanged at `src/state_manager.py:425`).
- Therefore, on a **frame-level ROC over `fatigue_score`**, enabling the CNN arm
  cannot change any score, so V4 is identical to V3 by construction.

The CNN *is* active and loaded: instrumented on subject 005 it fired 89 times
over 6,000 frames (EAR entered the uncertainty zone [0.17, 0.27] on 2,825
frames; rate-limited to 89 actual invocations). Its effect is confined to
alarm-suppression logic, which this frame-level fatigue-score ROC does not
measure. Measuring the CNN's contribution would require an alarm-event-level
metric, which is out of the frozen EXP-004 scope and is flagged as a limitation
(§9) and a next experiment (§10).

---

## 7. Interpretation notes (measured facts, not spin)

1. **Low absolute AUCs (0.61–0.63).** The frozen heuristic fatigue score is a
   weak *frame-level* discriminator on NTHU-DDD. NTHU labels are
   clip-condition-derived, so a single frame inside a "drowsy" clip may show
   open eyes; frame-level FPR is therefore conservative/pessimistic (documented
   known limitation, PROJECT_CONTEXT §4). This depresses all variants equally
   and does not by itself invalidate the comparison.
2. **Zero-inflation drives the operating point to ≈0.** 22.6 % of frames score
   exactly 0. Reaching TPR = 0.80 forces the threshold below the smallest
   positive score, which is why the fixed threshold is 2.6e-08 and FPR is high.
   This is a property of the temporal accumulator on short frame contexts, not a
   measurement artifact.
3. **The gate does what it is designed to do, just not enough to move this
   metric.** V2 slightly raises specificity per subject (e.g. subj 001:
   0.4837 → 0.4899) but the aggregate FPR@matched-TPR is flat. The claimed
   FPR reduction is not observed at frame level here.

---

## 8. Generated artifacts

All under `experiments/EXP-004_loso/` unless noted:

| Path | Contents |
|:--|:--|
| `exp004_metrics.json` (10.8 MB) | full frozen `roc` block (per-variant FPR/TPR curve arrays + AUC), extended per-variant and per-subject metrics, protocol, operating point |
| `per_variant_metrics.csv` | 5 variants × overall metrics + confusion matrix |
| `per_subject_metrics.csv` | 20 rows (5 variants × 4 subjects) |
| `scores/V{0..4}_*.csv` | per-frame `(subject, score, label)` — the raw measured evidence |
| `plots/roc_overlay.png` | ROC curves, all 5 variants |
| `plots/auc_prauc_bars.png` | ROC-AUC vs PR-AUC bars |
| `plots/fpr_at_matched_tpr.png` | primary-metric bars |
| `plots/confusion_matrices.png` | confusion matrices at fixed op point |
| `results/measured_results.json` | canonical `roc` block merged by the **frozen** writer (schema-valid, I5 PASS) |
| `results/fig_roc_curves.png` | canonical paper ROC figure (regenerated from measured JSON) |
| `results/fig_latency_breakdown.png` | regenerated latency figure (EXP-001 data) |

Backup: `results/measured_results.json.pre_exp004.bak` (pre-run state, EXP-001
latency only) retained.

---

## 9. Known limitations

1. **Frame-level labels.** NTHU labels are clip-condition-derived; frame-level
   FPR is conservative. An event-/episode-level evaluation would be a fairer
   test of the alarm behaviour (and the only way to observe the CNN's effect).
2. **CNN not observable in this metric.** V4 ≡ V3 at the fatigue-score ROC by
   construction (§6). The CNN affects alarm suppression only.
3. **4 subjects only.** NTHU-DDD provides 4 subjects here; LOSO folds are
   correspondingly coarse, and subject 006 dominates the negative variance.
4. **Host, not Pi 4.** No on-device (Raspberry Pi 4) latency/memory/thermal
   number exists; EXP-001 is a Darwin-arm64 host figure. None is claimed here.
5. **Negative result.** The primary hypothesis is not supported under these
   conditions. This is a measured outcome, honestly reported; it constrains the
   claims the paper may make.

---

## 10. Next recommended experiment (EXP-005)

**Event-level / episode-level ablation.** Replace the frame-level fatigue-score
ROC with an alarm-event metric (e.g. FPR per hour and detection latency of
genuine drowsy episodes), so that (a) the CNN arm's alarm-suppression effect
becomes observable and V4 can differ from V3, and (b) the reliability gate is
evaluated on the quantity it was designed to protect (spurious alarms), not on
raw per-frame scores. Keep LOSO, seed 42, and the fixed-operating-point
discipline. Log as EXP-005 before citing.

Secondary: an on-device Raspberry Pi 4 latency/memory/thermal profile remains
the outstanding feasibility claim for the paper.

---

## 11. Verification (all green, post-run)

- `evaluation/verify_integrity.py` → **6/6 PASS** (I1–I6), exit 0.
- `python -m unittest tests.test_suite` → **17/17 OK**.
- `tests/smoke_test.py` → **3/3 OK** (incl. 3-component reliability-gate assertion).
- Determinism: every subject's scored-frame and drowsy counts match the ground
  truth exactly (§2 table), confirming no decode drops and a reproducible order.

---

## 12. Reproduction

```bash
# full sweep (writes canonical roc block + extended artifacts)
.venv/bin/python evaluation/exp004_report.py --write

# canonical figures (measured JSON only)
.venv/bin/python evaluation/plot_paper_figures.py

# verification
.venv/bin/python evaluation/verify_integrity.py
.venv/bin/python -m unittest tests.test_suite
.venv/bin/python tests/smoke_test.py
```

Seed 42, video clock 30 fps, deterministic frame order. Frozen helpers
unmodified.
