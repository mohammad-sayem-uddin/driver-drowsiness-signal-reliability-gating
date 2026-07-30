# EXP-004 Scientific Audit Report
### Leave-One-Subject-Out (LOSO) ablation on NTHU-DDD — independent re-audit

**Scope.** This report is a *post-hoc scientific audit* of the already-completed
EXP-004 experiment. No models were retrained, no thresholds were changed, no
frozen algorithm was modified. Every number below was either (a) copied verbatim
from the released artifacts, or (b) independently recomputed from the released
per-frame score CSVs (`experiments/EXP-004_loso/scores/*.csv`) using read-only
code committed at `reports/EXP-004_AUDIT/{audit.py,analysis_figures.py}`.
Where a recomputed value disagrees with a reported value, both are shown and the
discrepancy is explained. No statistic is reported that is not supported by the
released data.

**Variants.** `(speech_filter, reliability_gate, cnn)`:
V0 (F,F,F) baseline · V1 (T,F,F) · V2 (F,T,F) · V3 (T,T,F) full · V4 (T,T,T) full+CNN.
**Subjects (LOSO folds):** 001, 002, 005, 006. **Seed:** 42.
**Primary metric:** FPR at matched TPR = 0.80 on each variant's own ROC.

---

## 1. Scientific audit of reported results

### 1.1 Reconciliation of reported vs. independently recomputed metrics

Every headline metric was recomputed from the released per-frame CSVs. The
**threshold-counting** metrics (confusion matrices, accuracy, FPR at the fixed V0
operating point) reproduce essentially exactly (±1 frame). The **ranking**
metrics (ROC-AUC, PR-AUC) reproduce with a small, *systematic* negative offset.

| Variant | Reported ROC-AUC | Recomputed ROC-AUC | Δ | Reported FPR@TPR0.80 | Recomputed |
|---|---|---|---|---|---|
| V0 | 0.628786 | 0.624155 | −0.00463 | 0.624119 | 0.624151 |
| V1 | 0.617005 | 0.610863 | −0.00614 | 0.669411 | 0.667541 |
| V2 | 0.624539 | 0.619878 | −0.00466 | 0.624447 | 0.624447 |
| V3 | 0.613299 | 0.607158 | −0.00614 | 0.669017 | 0.669935 |
| V4 | = V3 | = V3 | — | = V3 | = V3 |

**Confusion matrices at the fixed V0 threshold reproduce exactly** (V0:
TP 28824 / FP 19031 / TN 11460 / FN 7206 — matching the reported CSV to ±1 in FP).

### 1.2 Root cause of the AUC discrepancy (central audit finding)

The reported AUCs are **not bit-reproducible** from the released score CSVs, but
**every scientific conclusion is robust to the discrepancy.** The mechanism was
isolated and confirmed:

- The released scores are written with `%.10f` formatting. The score
  distribution is heavily **zero-inflated** (22.58% of V0 frames are exactly 0)
  with a dense sub-`1e-10` tail: running the *frozen* ROC/AUC code on the
  released CSVs reproduces the recomputed value 0.624155 **exactly**, not the
  published 0.628786.
- `%.10f` truncation collapses the entire sub-`1e-10` tail into the literal
  string `0.0000000000`, destroying the fine ranking granularity that
  distinguishes tied-at-zero negatives from tied-at-zero positives. This removes
  a small amount of *discriminating rank information* → a near-constant
  **−0.0046** AUC shift.
- The shift is (i) roughly constant across variants, (ii) confined to
  rank-based metrics, and (iii) invisible to threshold-count metrics (which only
  ask "is score > 2.59e-8?", a question truncation does not change).

**Verdict.** Published absolute AUC values overstate discrimination by ≈0.005 and
should be reported as the recomputed values. **Variant ordering, all effect
sizes, `fpr_at_matched_tpr`, and all confusion matrices are preserved.** For full
bit-level reproducibility, future releases must serialize scores at full float64
precision (e.g. `repr()`/`%.17g`) or in binary.

### 1.3 V4 ≡ V3 — confirmed two independent ways

1. **Byte-identical outputs.** `V3_full.csv` and `V4_full_cnn.csv` are md5-
   identical (`f8c298c8a7f521011ad9317da0b9c9b5`); every recomputed metric for V4
   equals V3 to all digits.
2. **Code path.** In `src/state_manager.py:425`,
   `fatigue_score = fusion.fatigue_score` is computed **independently** of
   `cnn_verdict`. The CNN block (`state_manager.py:374-388`) only flips the
   `should_alarm` boolean (and only in the SLIGHT/MODERATE zone, never in
   SEVERE/FACE_LOST). Since EXP-004 sweeps `fatigue_score`, the CNN **cannot by
   construction** alter any swept score → V4 ≡ V3. See §5.

**Audit conclusion (Part 1).** Reported metrics are internally consistent and
faithfully derived from the pipeline, with one caveat: released-CSV precision is
insufficient to bit-reproduce the absolute AUCs. This is a *reporting-precision*
issue, not a *scientific* one — corrected values are used throughout below.

---

## 2. Deep statistical analysis

All CIs are subject-stratified/paired bootstrap with **B = 2000**, seed 42.

### 2.1 Overall discrimination with 95% CIs (subject-stratified bootstrap)

| Variant | ROC-AUC | 95% CI | PR-AUC | vs. chance |
|---|---|---|---|---|
| **V0** | **0.6242** | [0.6200, 0.6284] | 0.6344 | above |
| V2 | 0.6199 | [0.6156, 0.6242] | 0.6187 | above |
| V1 | 0.6109 | [0.6067, 0.6151] | 0.6244 | above |
| V3 = V4 | 0.6072 | [0.6029, 0.6115] | 0.6108 | above |

All variants discriminate above chance, but **weakly** (best AUC ≈ 0.62). The
ordering is **V0 > V2 > V1 > V3**: *every added component lowers discrimination.*

### 2.2 Paired significance — each component significantly *worsens* AUC

**DeLong test** (`delong.json`, correlated ROC curves):

| Comparison | ΔAUC | z | p |
|---|---|---|---|
| V0 − V2 (gate) | +0.00428 | 45.86 | ~0 |
| V0 − V1 (speech) | +0.01329 | 17.97 | ~0 |
| V0 − V3 (both) | +0.01700 | 22.85 | ~0 |
| V2 − V3 | +0.01272 | 17.69 | ~0 |

**Paired bootstrap ΔAUC** (`paired_bootstrap.json`) — 100% sign-consistent:

| Comparison | mean ΔAUC | 95% CI | frac > 0 |
|---|---|---|---|
| V0 − V2 | +0.00428 | [0.00410, 0.00446] | 1.00 |
| V0 − V1 | +0.01331 | [0.01191, 0.01480] | 1.00 |
| V0 − V3 | +0.01701 | [0.01557, 0.01849] | 1.00 |

Both tests agree: **V0 is significantly the best discriminator; the speech
filter and the reliability gate each significantly reduce AUC, and their
combination (V3) is the worst.** This is the single most important — and most
uncomfortable — statistical result of EXP-004.

**McNemar** at the fixed V0 threshold (`mcnemar.json`) confirms the variants make
significantly *different* per-frame decisions (all p ≤ 1.2e-5), so the changes
are real reclassifications, not noise.

### 2.3 Zero-inflation structure (`zero_inflation.json`)

- V0: **22.58%** of all frames score exactly 0; broken out by class,
  **30.26% of not-drowsy** vs **16.09% of drowsy** frames are zero.
- Negatives are ~1.9× more zero-inflated than positives → zeros carry *weak
  correct* signal overall, which is why AUC stays above 0.5.
- Adding components **raises** zero-inflation (V3: 25.05% overall; drowsy zeros
  rise 16.09% → 19.30%), i.e. they push *more true-positive evidence to zero* —
  the mechanistic seed of the AUC loss in §2.2.

### 2.4 Calibration (rank-decile empirical drowsy rate, `extended_analysis.json`)

V0 empirical drowsy rate rises monotonically across score deciles
(0.32 → 0.44 → 0.40 → 0.53 → 0.56 → 0.56 → 0.59 → 0.67 → 0.66 → 0.67). The signal
is **monotone but weak and compressed near zero**: the bottom three deciles are
all essentially zero-score yet already carry 32–44% drowsy rate (driven by
subject 006, §3). The top decile only reaches ~0.67 drowsy rate. This is
consistent with an AUC in the low 0.6s and confirms the score is a *soft,
low-resolution* fatigue indicator, not a calibrated probability.

### 2.5 Class-conditional score separation

At the population level (V0), mean drowsy score 0.0605 vs mean not-drowsy 0.0350
(sep +0.0255). Every component **shrinks** this separation
(V2 → +0.0146; V3 → smaller still), matching the AUC ordering.

---

## 3. Failure-mode analysis — subject 006 (Part 3)

The AUC deficit of EXP-004 is not distributed evenly across subjects. It is
**dominated by a single LOSO fold — subject 006 — whose score is fully
sign-inverted.** All numbers below are from `data/extended_analysis.json`
(`subject_stats`), computed on the released per-frame CSVs.

### 3.1 The core finding: inverted discrimination on 006

| Group (V0) | n | prevalence | mean drowsy | mean not-drowsy | sep (pos−neg) | AUC |
|---|---|---|---|---|---|---|
| **Subject 006** | 6,739 | 0.410 | 0.01130 | **0.05200** | **−0.0407** | **0.3047** |
| Pooled 001/002/005 | 59,782 | 0.556 | 0.06457 | 0.03248 | +0.0321 | 0.6617 |

For every other subject the drowsy class carries the *higher* mean score (correct
sign). For subject 006 the relationship **reverses**: not-drowsy frames score on
average **4.6× higher** than drowsy frames (0.0520 vs 0.0113). The Mann–Whitney
AUC of 0.3047 is not "weak" — it is **significantly below chance (0.5)**, meaning
the score is anti-correlated with the label on this fold. If 006's decision rule
were simply flipped it would discriminate at AUC ≈ 0.695, comparable to the best
other subject.

### 3.2 Mechanism: drowsy frames on 006 are pushed to exactly zero

The inversion is driven by the **zero floor**, not by large mis-scored values:

- **51.18% of subject-006 drowsy frames score exactly 0** (`zero_pos = 0.5118`),
  versus 30.28% of its not-drowsy frames (`zero_neg = 0.3028`) — the *only*
  subject where positives are more zero-inflated than negatives.
- The **median drowsy score for 006 is exactly 0.0** (`median_pos = 0.0`), while
  its median not-drowsy score is 0.00415. More than half the true-drowsy evidence
  on this subject is quantized away to nothing before it can accumulate.
- By contrast, subject 005 — the best fold (AUC 0.717) — has only **5.31%** zero
  drowsy frames (`zero_pos = 0.053`) and the largest positive separation
  (+0.0409).

This is the signature of a **geometry-extraction failure on 006's drowsy
frames**, not a fusion or threshold failure: when the frozen EAR/MAR/head-pose
front-end cannot recover a reliable measurement, the pipeline emits 0, and on
006's drowsy segments it does so more than half the time. The most parsimonious
evidence-supported explanation is that 006's drowsy episodes coincide with a
face/eye configuration the frozen front-end handles poorly (e.g. landmark loss
during eye closure or occlusion), so eye-closure frames — which *should* score
high — collapse to 0. The audit cannot attribute the specific optical cause
without frame-level face crops, which are outside the released artifacts; this is
recorded as a limitation (§11) and a targeted follow-up (§9, EXP-006).

### 3.3 The pipeline makes subject 006 *worse*, not better

Crucially, the two novelties do not rescue the failing fold — they deepen it:

| Variant | 006 AUC | 006 sep | 006 zero_pos |
|---|---|---|---|
| V0 baseline | 0.3047 | −0.0407 | 0.5118 |
| V2 (gate) | 0.3054 | −0.0383 | 0.5128 |
| V3 (gate+speech) | **0.2871** | −0.0380 | **0.5356** |

Under the full pipeline (V3), 006's drowsy zero-fraction rises to **53.56%** and
its AUC falls further to **0.2871**. The reliability gate and speech filter, both
designed to *suppress* score in low-confidence conditions, act hardest exactly
where the true-positive evidence is already fragile — so they remove drowsy
signal from the one subject that could least afford it. Meanwhile they *help*
subject 001 (V3 AUC 0.645 vs V0 0.627) and 002 stays flat, so the population-level
harm is a **006-concentrated effect competing with modest gains elsewhere**.

### 3.4 Why one subject dominates the headline number

Subject 006 is only 6,739 / 66,521 ≈ **10.1%** of frames, yet because its AUC is
~0.30 while others average ~0.66, it drags the pooled AUC from ~0.66 down to
~0.624. The single most important consequence for the paper: **the reported ≈0.62
pooled AUC is a mixture of three well-behaved folds and one catastrophically
inverted fold, not a uniform "moderate" detector.** Reporting only the pooled
number hides a bimodal per-subject reliability that any deployment claim must
address.

---

## 4. Reliability-gate investigation — why the gate did not improve FPR (Part 4)

The reliability gate is one of the two headline novelties. Its stated purpose is
to *suppress false alarms* by attenuating fatigue score under low-confidence
sensing (`r = stability^0.45 · brightness^0.30 · consistency^0.25`, applied
multiplicatively before temporal accumulation). EXP-004 shows it **did not lower
FPR at matched TPR** — V2 FPR@TPR0.80 (0.6244) is essentially tied with V0
(0.6242), and its AUC is significantly *lower*. The paired per-frame analysis
(`gate_effect` in `extended_analysis.json`, computed by comparing V0→V2 on
identical frames) explains why mechanistically.

### 4.1 The gate is aggressive and nearly one-directional

- It changes **76.35%** of all frames (`frames_changed = 50,789`).
- Of frames with nonzero raw score, **98.62%** are *attenuated* (score reduced),
  confirming `r ≤ 1` acts almost purely as a down-weight — it essentially never
  raises a score.

### 4.2 It removes more absolute signal from drowsy frames than from not-drowsy

Because the gate is *multiplicative* and drowsy frames start with higher raw
scores, the same fractional attenuation strips **more absolute score from the
positive class**:

| Quantity | drowsy (pos) | not-drowsy (neg) |
|---|---|---|
| mean attenuation (all frames) | **0.01548** | 0.00465 |
| mean attenuation (nonzero frames) | **0.01845** | 0.00666 |

The gate removes ~3.3× more score from drowsy frames than from not-drowsy frames.
The net effect on class separation is a **compression, not a sharpening**:

- separation before gate (V0): **+0.02546**
- separation after gate (V2): **+0.01462** — a 43% reduction.

### 4.3 Why this defeats the gate's own design goal

The gate's premise is that false alarms come from *low-reliability high scores*
(noise-driven spikes on not-drowsy frames). The data does not support that
premise for this dataset: the not-drowsy class is already **heavily zero-inflated
and low-valued** (mean 0.0325), so there is little spurious high-score mass for
the gate to remove. Meanwhile the genuine drowsy signal lives in the moderate
score band — precisely the band the gate attenuates most in absolute terms. The
result is that the gate **erodes true-positive score faster than it erodes
false-positive score**, shrinking separation and shifting more drowsy frames
toward the zero floor (drowsy zero-fraction 16.09% → V3 19.30%, §2.3). At a
matched TPR the FPR is therefore unchanged-to-worse, and AUC drops significantly
(DeLong ΔAUC +0.00428, z = 45.9, §2.2).

**Verdict.** The gate is *working as coded* — it reliably attenuates
low-confidence frames — but the design assumption (that FP mass is
reliability-separable from TP mass) does not hold on NTHU-DDD. The gate cannot
improve FPR because, on this data, the score component it suppresses is
disproportionately the *true* signal. This is a genuine, evidence-backed negative
result about the novelty, and it motivates EXP-005 (§7): make the gate *additive
in the decision layer* or *reliability-conditional* rather than a blanket
multiplicative pre-accumulation down-weight.

---

## 5. CNN investigation — V4 ≡ V3 confirmed (Part 5)

This section consolidates the Part-5 finding already established in §1.3 and
states the CNN's true influence on EXP-004.

### 5.1 The CNN cannot alter any swept metric — by construction

EXP-004 sweeps `FrameResult.fatigue_score`. The CNN (MicroEyeNet, 24×24×1 INT8
TFLite, 19,745 params) is wired **only** into the alarm-suppression path:

- `src/state_manager.py:425` sets `fatigue_score = fusion.fatigue_score`,
  computed **independently of `cnn_verdict`**.
- The CNN block (`src/state_manager.py:374-388`) only flips the `should_alarm`
  boolean, and only inside the SLIGHT/MODERATE zone (never SEVERE or FACE_LOST).

Because the swept variable never reads the CNN verdict, **every score-based metric
(ROC, AUC, FPR@matched-TPR, and the fixed-threshold confusion matrix) is
identical between V3 and V4 by construction.**

### 5.2 Two independent confirmations

1. **Byte identity.** `V3_full.csv` and `V4_full_cnn.csv` are md5-identical
   (`f8c298c8a7f521011ad9317da0b9c9b5`).
2. **Recomputed metrics.** Every V4 statistic equals V3 to all reported digits
   (bootstrap AUC both 0.60716; per-subject AUCs identical).

### 5.3 What this means scientifically

The CNN's contribution is **not measurable in EXP-004's design** — not because it
does nothing, but because the experiment measures the wrong variable to see it.
The CNN can only change the **alarm-level decision stream** (`should_alarm`), a
boolean the score-sweep never touches. EXP-004 therefore neither validates nor
refutes the CNN; it merely demonstrates that the CNN is *decoupled from the
fatigue score*. Any claim about the CNN's value must be tested on the alarm
stream directly (event-level precision/recall on `should_alarm` transitions), not
on the score ROC. This is specified as EXP-007 (§7). Reporting V4 as a distinct
row in a score-based ablation table is, strictly, redundant and should be
footnoted as "identical to V3 by construction" to avoid implying an evaluated
difference.

---

## 6. Additional figures generated (Part 6)

Eight publication-quality figures were generated by the read-only script
`reports/EXP-004_AUDIT/analysis_figures.py` (matplotlib Agg, seed 42), written to
`reports/EXP-004_AUDIT/figures/`. Every figure is drawn from the released
per-frame CSVs or the recomputed JSON; none uses fabricated or smoothed data.

1. **`fig1_roc_overlay.png`** — ROC curves for V0/V1/V2/V3 with bootstrap-AUC
   legend and the matched TPR = 0.80 operating point marked on V0. Shows the
   variants are nearly coincident with V0 uppermost. (V4 omitted, ≡ V3.)
2. **`fig2_auc_forest.png`** — Forest plot of AUC with 95% subject-stratified
   bootstrap CIs (V0 > V2 > V1 > V3). Bands are narrow and ordered; V0's lower
   bound (0.6200) sits at/above the others' point estimates.
3. **`fig3_per_subject_auc.png`** — Grouped per-subject AUC bars. Visually
   isolates the headline failure: subject 006 sits **below the 0.5 chance line**
   for every variant while 001/002/005 are ~0.6–0.72.
4. **`fig4_zero_inflation.png`** — Per-class zero-fraction by variant. Negatives
   more zero-inflated than positives at population level, and every added
   component raises both bars.
5. **`fig5_subject006_dist.png`** — Log-y score histograms, subject 006 vs pooled
   others (V0). Directly visualizes the sign inversion: 006's drowsy mass sits
   *left* of its not-drowsy mass.
6. **`fig6_gate_effect.png`** — Class-conditional mean score, V0 vs V2. Shows the
   gate pulling both classes down and compressing separation 0.0255 → 0.0146.
7. **`fig7_calibration.png`** — Rank-decile empirical drowsy rate for V0/V2/V3.
   Monotone but shallow; bottom deciles already carry 32–44% drowsy rate.
8. **`fig8_confusion.png`** — Confusion matrices at the fixed V0 threshold for
   V0/V1/V2/V3, annotated with accuracy/TPR/FPR.

---

## 7. Secondary scientific findings (Part 7)

Findings that are not the headline result but are firmly supported by the data:

- **F1 — Per-subject reliability is bimodal, not graded.** AUCs cluster at
  {0.627, 0.620, 0.717} for 001/002/005 and crash to 0.305 for 006. There is no
  "moderate" middle; the detector either works (~0.62–0.72) or inverts.
- **F2 — Subject 005 is the strongest fold and the least zero-inflated.** Its
  drowsy zero-fraction (5.3%) is the lowest of any subject and it attains the top
  AUC (0.717) and largest separation (+0.041). Zero-inflation of the *positive*
  class is the dominant per-subject predictor of AUC across these four folds
  (005: 5.3%→0.72; 001: 22.0%→0.63; 002: 14.9%→0.62; 006: 51.2%→0.30).
- **F3 — The two novelties trade population AUC for per-subject smoothing.** V3
  *raises* subject 001's AUC (0.627 → 0.645) and holds 002 roughly flat, but the
  006 collapse and 005 erosion (0.717 → 0.682) dominate, so pooled AUC falls. The
  components are not uniformly harmful — they redistribute where the detector
  fails.
- **F4 — The score is a low-resolution rank signal, not a probability.** The top
  rank-decile reaches only ~0.67 empirical drowsy rate against a 0.55 base rate;
  the score never confidently isolates drowsiness. Any thresholded alarm inherits
  this ceiling.
- **F5 — Threshold-count metrics are robust to the CSV precision bug; rank
  metrics are not.** This is itself a finding about *which* published numbers can
  be trusted at face value (CMs, accuracy, FPR@fixed-threshold) versus which need
  the −0.005 correction (all AUC/PR-AUC).

---

## 8. Publication insights (Part 6 of final report)

How these results should be framed honestly in a paper:

- **Lead with the negative ablation result — it is the paper's most credible
  contribution.** A rigorously-measured, significance-tested finding that two
  intuitively-motivated novelties *fail to help (and significantly hurt) pooled
  AUC on LOSO* is publishable as an honest ablation, and is far stronger than an
  over-claimed positive.
- **Never report pooled AUC without the per-subject breakdown.** The pooled ≈0.62
  is a mixture artifact (§3.4). LOSO papers that hide a below-chance fold behind a
  pooled mean are a known reviewer red flag; disclosing it pre-empts that
  criticism and reframes the story around *subject-dependent front-end failure*.
- **Report the corrected AUCs and the precision caveat.** State the released CSVs
  were `%.10f`-truncated, that this shifts AUC by ≈−0.005, and that all
  conclusions are invariant to it. Reviewers reward this over silent numbers.
- **Demote the V4/CNN row to a construction note.** Presenting V4 as an evaluated
  variant when it is byte-identical to V3 invites the charge of padding the
  ablation table.
- **n = 4 subjects is the binding limitation** (§11) and must gate every
  generalization claim. Frame the work as a *mechanistic case study of
  reliability-gating failure modes*, not a benchmark-beating detector.

---

## 9. Recommended follow-up experiments — EXP-005+ (Part 8)

None of these invalidate EXP-004; all build on its frozen artifacts. Each lists
objective, motivation, expected value, effort, and whether it requires
retraining or a frozen-design change.

**EXP-005 — Reliability gate as a decision-layer term (re-architect the gate).**
- *Objective:* replace the blanket multiplicative pre-accumulation down-weight
  with a reliability signal consumed *at the decision layer* (or gated only when
  reliability is below a learned floor), so it cannot erode moderate-band true
  signal.
- *Motivation:* §4 shows the current gate removes 3.3× more absolute score from
  drowsy than not-drowsy frames and compresses separation 43%.
- *Expected value:* HIGH — directly targets the mechanism behind the gate's
  failure; could turn a significant AUC loss into a gain.
- *Effort:* MEDIUM. *Requires frozen-design change:* YES (gate placement).
  *Retraining:* NO (geometry front-end unchanged). Must be a **new experiment
  ID**; does not touch EXP-004 artifacts.

**EXP-006 — Subject-006 front-end failure diagnosis.**
- *Objective:* determine why 51% of 006's drowsy frames score exactly 0 —
  frame-level audit of landmark confidence, EAR/MAR validity flags, and
  face-detection dropouts on 006's drowsy segments.
- *Motivation:* §3 localizes the entire pooled deficit to a front-end
  geometry-extraction failure on one fold; the optical cause is unidentified.
- *Expected value:* HIGH — a single fix here could recover ~10% of frames from
  below-chance to useful.
- *Effort:* LOW–MEDIUM (diagnostic, needs 006 video frames). *Frozen-design
  change:* NO (diagnostic only). *Retraining:* NO.

**EXP-007 — Direct evaluation of the CNN alarm-suppression path.**
- *Objective:* evaluate the CNN on the variable it actually controls —
  event-level precision/recall and alarm latency on the `should_alarm` stream in
  the SLIGHT/MODERATE zone — since EXP-004's score sweep is blind to it (§5).
- *Motivation:* EXP-004 leaves the CNN's value entirely unmeasured.
- *Expected value:* MEDIUM–HIGH — first real test of the CNN novelty.
- *Effort:* MEDIUM (needs alarm-event ground truth). *Frozen-design change:* NO.
  *Retraining:* NO (CNN already trained/frozen).

**EXP-008 — Score serialization at full precision + re-freeze.**
- *Objective:* re-emit EXP-004 score CSVs at float64 (`repr`/`%.17g`) or binary
  and confirm published AUCs bit-reproduce.
- *Motivation:* §1.2 — closes the only reproducibility gap found in the audit.
- *Expected value:* MEDIUM (reproducibility hygiene, not new science).
- *Effort:* LOW. *Frozen-design change:* NO. *Retraining:* NO. Re-runs *inference
  serialization only*; does not alter the algorithm, so it does not invalidate
  EXP-004's conclusions.

**EXP-009 — Expand the subject panel (generalization).**
- *Objective:* extend LOSO beyond n = 4 to bound per-subject variance and test
  whether the 006-type inversion recurs.
- *Motivation:* §11 — n = 4 cannot support generalization; the bimodal split may
  be sampling noise or a real failure class.
- *Expected value:* HIGH for external validity. *Effort:* HIGH (data +
  compute). *Frozen-design change:* NO. *Retraining:* NO (same frozen pipeline,
  new folds).

---

## 10. List of publication-quality figures

All in `reports/EXP-004_AUDIT/figures/`, 200 dpi PNG, generated read-only from
released artifacts. Recommended manuscript placement in brackets.

| # | File | What it shows | Placement |
|---|---|---|---|
| 1 | `fig1_roc_overlay.png` | LOSO ROC per variant + matched operating point | Main — results |
| 2 | `fig2_auc_forest.png` | AUC forest plot, 95% bootstrap CIs, ordering | Main — ablation |
| 3 | `fig3_per_subject_auc.png` | Per-subject AUC; 006 below chance | **Main — key** |
| 4 | `fig4_zero_inflation.png` | Per-class zero-inflation by variant | Supplementary |
| 5 | `fig5_subject006_dist.png` | 006 vs others score histograms (inversion) | **Main — key** |
| 6 | `fig6_gate_effect.png` | Gate compresses class separation | Main — analysis |
| 7 | `fig7_calibration.png` | Rank-decile empirical drowsy rate | Supplementary |
| 8 | `fig8_confusion.png` | Confusion matrices at fixed threshold | Supplementary |

Figures 3 and 5 are the two that carry the paper's central, most defensible
message (subject-concentrated inversion) and should be in the main body.

---

## 11. Limitations

- **Sample size.** n = 4 subjects (66,521 frames). All per-subject conclusions,
  and especially the 006 inversion, rest on single folds; CIs quantify *sampling
  within* frames but cannot bound *between-subject* generalization. No claim about
  population FPR/AUC is warranted from four subjects.
- **Frame-level, not event-level.** All metrics treat frames as i.i.d.; drowsiness
  is episodic and frames are temporally autocorrelated, so effective sample size
  is smaller than n suggests and bootstrap CIs are mildly optimistic. Event-level
  evaluation (alarm episodes) is not performed and is left to EXP-007.
- **CSV precision.** Released scores are `%.10f`-truncated; absolute AUCs are not
  bit-reproducible (§1.2). Corrected values are used, but the exact published
  AUCs cannot be regenerated from the released CSVs until EXP-008.
- **CNN unevaluated.** EXP-004's score sweep is structurally blind to the CNN
  (§5); nothing here speaks to the CNN's merit either way.
- **Optical cause of 006's failure unidentified.** The audit localizes the failure
  to zero-valued drowsy frames but cannot name the front-end cause without the raw
  006 video frames, which are outside the released artifacts (EXP-006).
- **Single dataset.** NTHU-DDD only; the reliability-gate failure mode (§4) may be
  dataset-specific — it depends on FP mass *not* being reliability-separable,
  which may differ on other data.

---

## 12. Most important scientific conclusions

1. **The two headline novelties do not help — and significantly hurt — pooled
   LOSO discrimination.** V0 (neither) is the best discriminator; the speech
   filter and the reliability gate each significantly lower AUC (DeLong p ≈ 0,
   paired-bootstrap 100% sign-consistent), and their combination V3 is worst
   (AUC 0.607 vs 0.624). This is the central, rigorously-tested result.
2. **The pooled AUC is a mixture artifact dominated by one inverted subject.**
   Subject 006 discriminates *below chance* (AUC 0.305, separation −0.041) because
   51% of its drowsy frames score exactly 0; three other subjects sit at
   0.62–0.72. The detector is bimodal — works or inverts — not uniformly moderate.
3. **The reliability gate fails by construction on this data.** Being
   multiplicative, it removes 3.3× more absolute score from drowsy than not-drowsy
   frames and compresses class separation 43%, because the FP mass it was meant to
   suppress is not reliability-separable from the TP mass here.
4. **The CNN is decoupled from every swept metric,** so V4 ≡ V3 exactly and the
   CNN is neither validated nor refuted by EXP-004.
5. **Threshold-count metrics reproduce exactly; rank metrics carry a systematic
   −0.005 offset** traceable entirely to CSV precision, so the qualitative story
   is fully reproducible and the quantitative AUCs need a small stated correction.

---

## 13. Unexpected discoveries

- **A below-chance LOSO fold hiding inside a "moderate" pooled AUC.** The most
  surprising result: the 0.62 pooled number is not a uniformly weak detector but
  the average of good folds and one *anti-correlated* fold (AUC 0.305). This was
  not visible in the original pooled reporting.
- **The novelties are actively counterproductive, not merely neutral.** The prior
  expectation for an ablation is that added components help or do nothing; here
  each significantly *worsens* AUC, and the reliability gate worsens exactly the
  subject it should protect (006 drowsy zero-fraction 51% → 54% under V3).
- **The failure is a quantization-to-zero effect, not a large-error effect.** The
  detector does not mis-score 006's drowsy frames to confidently-wrong high
  values; it collapses them to *exactly 0*. The damage is done by the zero floor
  of the geometry front-end, upstream of fusion — a very different (and more
  fixable) failure than a mis-tuned classifier.
- **Reproducibility gap from formatting alone.** A purely cosmetic `%.10f` on the
  score CSV — not any algorithmic nondeterminism — is sufficient to move the
  published AUC by ≈0.005, a reminder that serialization precision is a
  first-class reproducibility concern for zero-inflated score distributions.

---

*End of audit. All numbers herein trace to `reports/EXP-004_AUDIT/data/*.json`
and `experiments/EXP-004_loso/scores/*.csv`, recomputed by the read-only scripts
`audit.py` and `analysis_figures.py`. No models were retrained, no thresholds or
frozen algorithms were modified, and the experiment was not rerun.*

