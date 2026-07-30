# Independent Scientific Research Review

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Review type:** Internal scientific-quality review (pre-paper-writing gate)
**Reviewer stance:** Independent senior research scientist. Not the author. Skeptical by default; the research is assumed incorrect until the project's own evidence proves otherwise.
**Constraints honored:** No web search. No comparison to published literature. No judgment of publication potential or novelty against prior art. This review evaluates **only the internal scientific quality** of the project as evidenced by its own committed documents, code, and artifacts.
**Review date:** 2026-07-30

---

## 0. How this review was conducted

I read the project's own materials and cross-checked every headline number against the artifact that is supposed to back it. Specifically, I consulted:

- The frozen engineering contract: `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md`.
- EXP-005 (event-level): `reports/EXP-005_REPORT.md`, its audit `reports/EXP-005_AUDIT.md`, and the root-cause note `reports/EXP005_ROOT_CAUSE_ANALYSIS.md`.
- EXP-004 (frame-level LOSO ablation) artifacts, its independent audit `reports/EXP-004_AUDIT/`, and their JSON/CSV.
- EXP-002 (CNN training) and EXP-003 (quantization) metrics and reports.
- The pipeline source: `src/detector.py`, `src/robustness.py`, `src/temporal_analyzer.py`, `src/fatigue_fusion.py`, `src/state_manager.py`, `src/frame_processor.py`, `src/config.py`, and the evaluation harnesses `evaluation/loso_harness.py`, `evaluation/exp005_event_report.py`, `evaluation/event_metrics.py`, `evaluation/nthu_ground_truth.py`.
- The registry `EXPERIMENT_REGISTRY.md`.

Two independent verification passes were performed to ground the findings below: a **source-code audit** keyed to the nine implementation claims the documents make, and a **numeric-consistency audit** re-deriving every reported metric from the committed confusion matrices, CSVs, and JSON.

**Named input that does not exist.** The task named `recursive-churning-lecun.md` as a required read. It **does not exist anywhere on disk** (independently confirmed here and already flagged by `reports/EXP-005_AUDIT.md` §1, finding M-1). No conclusion in this review depends on it, but its absence is noted because the project's own review scaffolding references it.

**A word on what "assume incorrect until proven" produced here.** Applying that standard, most of the project's *reported numbers* survived scrutiny — the arithmetic is real and reproducible. What did **not** survive is a set of claims about *what the code does* and *what the system achieves*: several stated invariants and mechanisms are not implemented as written, one superseded document flatly contradicts the committed results, and the system's headline capability (event-level detection) is essentially null. The distinction matters: this is an honestly-measured project with a partly-misdescribed implementation and an unproven core claim, not a fabricated one.

---

## 1. Research Problem

**Is the problem clearly defined?** Yes. The frozen spec §1 states a single, testable research question: *does a decomposed signal-reliability gate combined with a speech-jitter MAR filter reduce false-positive rate (FPR) at a matched true-positive rate (TPR), relative to a weighted-fusion baseline, while remaining real-time on a Raspberry Pi 4 CPU?* This is a well-formed, falsifiable question with a named baseline, a named primary metric, and a named deployment constraint.

**Is the motivation convincing?** Internally, yes. False alarms are the accepted failure mode that erodes trust in driver-monitoring systems, and the project frames its contributions (reliability gate, speech-jitter filter) as false-alarm-reduction mechanisms. The motivation is coherent with the metric chosen (FPR@matched-TPR).

**Is the scope appropriate?** Mostly. The scope is a CPU-only, geometry-first pipeline with an optional CNN arm evaluated by ablation. That is a reasonable, bounded scope. However, two scope commitments in the frozen spec are **not met by the evidence**: (a) §3 names *NTHU-DDD + YawDD* as primary evaluation datasets, but no YawDD evaluation artifact exists — all reported results are NTHU-DDD only; (b) §1 and §3 commit to *Raspberry Pi 4* real-time feasibility, but the only latency artifact is Darwin/arm64 (Apple M1), not the Pi. The problem is well-scoped on paper; the executed scope is narrower than the frozen contract claims.

**Is it worth solving?** Within the project's own framing, yes — but see §5: the primary experiment (EXP-004) returns a **negative result** on the central claim, and the follow-up (EXP-005) shows the system essentially does not fire alarms on the evaluation corpus. The problem is worth solving; the current evidence does not yet show this approach solves it.

---

## 2. Methodology

**Research design.** The design is a variant ablation V0→V4 toggling (speech_filter, reliability_gate, cnn), evaluated under leave-one-subject-out (LOSO) on NTHU-DDD. This is an appropriate design for isolating component contributions, and the subject-disjoint constraint is genuinely enforced in code (`nthu_ground_truth.py` derives labels from directory/filename grammar only, and the LOSO harness constructs a fresh processor per subject). Subject-disjointness — the single most important guard against leakage in this setting — is real.

**Evaluation methodology — two levels, appropriately motivated.** EXP-004 evaluates at the **frame level** (ROC over the continuous `fatigue_score`). EXP-005 correctly identifies that a frame-level ROC is *structurally blind* to two components (the CNN arm and the reliability gate act on the `should_alarm` decision, not the continuous score) and therefore re-evaluates at the **event/alarm level**. This is sound scientific reasoning: the second experiment exists because the first could not observe part of the system. The two-regime design in EXP-005 (PRIMARY per-recording reset vs. SECONDARY per-subject concatenation as a contamination cross-check) is also sound, and the report correctly designates PRIMARY as the headline regime.

**Statistical methodology.** For EXP-005 the project is *appropriately modest*: n = 4 subjects, no significance tests, and the bootstrap band is explicitly labeled a dispersion estimate rather than a hypothesis test (EXP-005 §4.6). For EXP-004 the audit adds DeLong, paired bootstrap (B=2000, seed 42), and McNemar tests. This is a genuinely rigorous statistical layer for the frame-level experiment.

**Where the methodology is unsound or misdescribed:**

- **The primary metric is not implemented as specified.** The frozen spec §3 defines the primary metric as *"FPR at matched TPR (operating point fixed on the baseline ROC, then held constant across variants)."* The source-code audit found that `loso_harness._fix_operating_point` returns a **TPR value** (`v0_tpr[idx]`), not a score threshold, and the downstream `_fpr_at_tpr` computes each variant's FPR at its *own nearest-achieved TPR* (`argmin|tpr − target|` per variant). This means FPR is compared at the nearest-achieved-TPR **per variant**, not at a single held-constant score threshold. The "operating point held constant across variants" framing — the crux of a fair FPR comparison — is not what the code does. This is a methodological defect, not merely a wording issue, because the fairness of the headline FPR comparison depends on it.

- **"Seed 42" determinism is asserted but not implemented in the LOSO harness.** The spec and reports repeatedly cite seed 42. The source-code audit found no RNG seeding in `loso_harness.py` (only a docstring mention); determinism there comes from sorted enumeration, not a seed. The seeded RNG is real only in the EXP-005 bootstrap (`default_rng(42)`). The determinism claim is therefore *true by construction* in the harness but *not via the mechanism the documents state*.

**Verdict on methodology:** The high-level design (ablation, LOSO, two-level evaluation, honest statistics) is sound and well-reasoned. But the *operationalization* of the primary metric contradicts the frozen specification, and that discrepancy is not disclosed in any report. This is a Major issue.

---

## 3. Implementation

The source-code audit checked nine specific implementation claims. Five verified cleanly; four are contradicted or only partially true. The pattern is important: **the numerical machinery is largely correct, but the code's own comments and the design documents overstate or mislabel what several components do.**

**Verified (implementation matches claim):**

- **Reliability gate is exactly 3 components, geometric mean, exponents 0.45/0.30/0.25** (`robustness.py:86-88`, `:278-282`). Matches the frozen invariant. (Caveat: the gate output is EMA-smoothed, `alpha=0.2`, not the raw instantaneous product the prose implies.)
- **CNN routes only to `should_alarm`, never to `fatigue_score`** (`fatigue_fusion.update` takes no CNN input; CNN acts only in `state_manager.py:374-388`). This correctly explains why V4 ≡ V3 at the frame level.
- **MAR is 2D with landmark indices [78,13,308,14]** (`frame_processor.py:44`, `detector.py:99-123`).
- **`frame_processor.py` is a single shared path**, and the ablation bypass reads `snap.system_reliability` else 1.0 (`frame_processor.py:193-196`, `:210-212`).
- **NTHU ground truth is derived from filenames only, no pixel decoding** (`nthu_ground_truth.py:50-111`) — no label leakage from image content.

**Contradicted or misdescribed (implementation does not match the stated design):**

1. **"SEVERE is never suppressed" is NOT implemented in the fusion engine.** This is the most serious implementation finding. The frozen spec lists as a core contribution and a §5 integrity invariant that reliability attenuation is *safety-asymmetric*: **SEVERE states are never suppressed.** In `fatigue_fusion.py:197`, `raw_score *= reliability` is applied **unconditionally**, *before* severity is computed at `:214`. There is no SEVERE bypass in the fusion engine. Sustained low reliability can therefore mathematically prevent the score from ever reaching the severe threshold. The only SEVERE protection that exists is a **separate downstream boolean** in `state_manager.py:337-341/362` (`if alert_suppressed and status != SEVERE_FATIGUE`), which guards the *alarm boolean* — not the continuous `fatigue_score`/severity that the fusion engine produces. The stated invariant and the implemented behavior do not match, and the mismatch is safety-relevant. (Note: EXP-005 §6.3 reports `any_alert_suppressed = 0` across all alarm events, so on this corpus the gap was never exercised — but the invariant is nonetheless not implemented as written.)

2. **EAR is computed in 2D, not 3D.** The frozen spec §2 module table states "EAR (3D)" and `frame_processor.py:9` carries a comment to that effect. The code computes EAR in 2D via x/y only (`detector.py:55-88`, `:51-53`); the design deliberately standardizes both EAR and MAR to 2D (`detector.py:29-34`). The "EAR 3D" claim is a stale comment contradicting the code.

3. **The "variance-based" speech-jitter filter does not compute variance.** The frozen spec names the contribution "temporal σ²(MAR) gating," and comments label the statistic σ² (`config.py:574`, `temporal_analyzer.py:262/342`). The implemented statistic is the **mean absolute first difference** `mean|ΔMAR|` (`temporal_analyzer.py:296-298`), not variance. The threshold (0.05) is correct; the *named statistic* is wrong. This mislabels one of the two claimed core contributions.

4. **A parallel live-path reimplementation ignores the ablation gate.** `main.py:249-283` reimplements the frame flow and does not honor the ablation gate (`:278`), diverging from the shared `frame_processor.py` core that A3 of the frozen spec says exists precisely to prevent divergence.

**Engineering quality.** Error handling is solid; the event-metric core is pure and deterministic; magic numbers are mostly centralized (with some leakage: robustness ramp 30/240, temporal weights, `MIN_DWELL_TIME=2.0`). There is dead/aspirational code (`LearnedReliabilityEstimator`, ensemble/learned-logistic modes, `CalibrationManager`) that is not on any evaluated path.

**Verdict on implementation:** The compute paths that produce the reported numbers are correct and reproducible. But **the single biggest implementation liability is that comments and design documents overstate/mislabel the implementation** — including a safety-relevant invariant (SEVERE), a named contribution (σ²), and a documented feature (EAR 3D). A reader trusting the documents would misunderstand the system. These are correctness-of-*description* defects with one genuine correctness-of-*behavior* gap (SEVERE).

---

## 4. Experimental Design

**Controls and baselines.** V0 (all-off weighted fusion) is a clean baseline, and the ablation cleanly isolates each toggle. The design is capable of attributing effects to components — and, to the project's credit, it did: EXP-005 §8 correctly attributes the false-alarm reduction to the speech-filter path and shows the gate alone (V2) does not help.

**Ablations.** Complete (V0–V4), one variant per registry row, with V3/V4 byte-identity (md5 `f8c298...`) confirmed at the artifact level — a genuine, verifiable check that the CNN arm does not touch the swept score.

**Evaluation protocol and metrics.** Frame-level ROC/AUC (EXP-004) plus event-level FA/h, recall, precision, latency (EXP-005). The metric set is appropriate. Two design weaknesses:

- **The evaluation corpus is very small and unbalanced for the event-level claim.** n = 4 subjects; in EXP-005 PRIMARY only 2 subjects (002, 005) ever fire an alarm, and **all false alarms originate from a single subject (005)**. The entire V2-vs-V3 false-alarm difference rests on one subject. The report discloses this (EXP-005 §9), which is the correct scientific behavior, but it means the event-level design cannot support component-level conclusions with any confidence.

- **The primary experiment answers the research question in the negative.** EXP-004 at matched TPR=0.80 shows the reliability gate does **not** reduce FPR and the speech-jitter filter **raises** it (frozen spec A4 records this honestly). So the experimental design is sound enough to have *falsified the main hypothesis at the frame level* — which is a sign of a working experimental apparatus, but also means the experiments as designed do not support the project's motivating claim.

**Verdict on experimental design:** Well-structured and genuinely diagnostic, but under-powered for the event-level conclusions (single-subject FP dependence) and, at the frame level, it refutes rather than supports the central claim.

---

## 5. Results

**Are the conclusions supported?** For EXP-005, yes — and notably, the conclusions are *modest and correctly scoped*. The report concludes V3 has the best false-alarm profile "at the event level, on this corpus," explicitly declines to claim significance or generalization, and foregrounds the single-subject dependence. Every headline EXP-005 number triangulates across JSON/CSV/log/plots (confirmed by `reports/EXP-005_AUDIT.md` and re-confirmed here).

**Are any conclusions overstated?** Two places:

- The **frozen spec's abstract-level framing** (NTHU-DDD + YawDD; Raspberry Pi 4 feasibility) overstates what was executed: no YawDD evaluation artifact and no on-Pi latency artifact exist. If this framing propagates into the paper, it will be an overclaim.
- The **σ² / EAR-3D / SEVERE-exempt** descriptions overstate/mislabel implemented behavior (§3).

**Are negative results honestly reported?** **Yes — this is a genuine strength.** EXP-004's negative result (gate does not reduce FPR; speech filter raises it) is stated plainly in the frozen spec A4 and the EXP-004 report. EXP-005's near-zero event-level recall (~0.12–0.15; 40+ of 41 GT episodes missed) is reported without spin. The EXP-002 TEST-F1 > VAL-F1 oddity is disclosed and correctly explained as a class-prevalence artifact (TEST positive prevalence 85.6% vs VAL 40.7%; balanced accuracy correctly *lower* on TEST). Honesty in reporting negative and awkward results is consistently high.

**Are limitations acknowledged?** Yes, thoroughly, in EXP-005 §9 (small n, single-subject FP, clip-level labels, unstable latency IQR, contaminated secondary regime, failing observability gates).

**Verdict on results:** The results that are reported are supported by the artifacts and interpreted with appropriate caution. The overstatements live in the *design/spec framing and code comments*, not in the results sections themselves.

---

## 6. Internal Consistency

This is where the most consequential findings surface. I report every inconsistency found, ranked by materiality.

**6.1 — MAJOR: A committed root-cause document contradicts the committed EXP-005 results and is never marked superseded.**

`reports/EXP005_ROOT_CAUSE_ANALYSIS.md` describes an EXP-005 run in which the PRIMARY regime produced **essentially zero alarms**: "recall 0.0, F1 0.0, 16/16 positive GT events missed for every variant," over **10,800 frames** (27 recordings × 400-frame cap), with **all five variants byte-identical** (single MD5 `0b7c087a586ec5b6480805010c935ac7`), max fatigue_score 0.2156, and **GT = 16 events**.

The committed final EXP-005 artifacts (`reports/EXP-005_REPORT.md`, JSON, CSVs) describe a *different run*: PRIMARY TP = 5–6, **41 GT episodes**, **66,521 frames**, and **distinct per-variant MD5s** (V0 `d694f72b`, V1 `00d9e52b`, V2 `574bdaec`, V3=V4 `f958e65d`).

These two documents cannot both describe the same experiment: 16 vs 41 GT episodes, 10,800 vs 66,521 frames, single vs distinct MD5s, recall 0.0 vs 0.12–0.15. The root-cause note is evidently a **stale analysis of an earlier capped run** that was superseded by the full run — but it is committed alongside the final artifacts, is never marked "superseded/obsolete," and is **not reconciled** by either the EXP-005 report or the EXP-005 audit (the audit does not mention it at all). A professor reading the repository would find two contradictory accounts of the same experiment with no pointer indicating which is current. This is the single most important internal-consistency defect.

**6.2 — MODERATE: Cross-document AUC discrepancy for subject 006, undisclosed in the audit's reconciliation.**

The EXP-004 audit (`reports/EXP-004_AUDIT/`) states that its recomputed AUCs differ from EXP-004's reported AUCs by a "roughly constant" ≈ −0.005 offset attributed to `%.10f` CSV truncation. The numeric-consistency audit confirms this holds for the overall AUC and for subjects 001/002/005 (offsets +0.0015 to −0.0054). It does **not** hold for **subject 006**:

| Variant | EXP-004 `exp004_metrics.json` (006 AUC) | Audit `recomputed_metrics.json` (006 AUC) | Δ |
|---|---|---|---|
| V0 | 0.372733 | 0.304675 | −0.068 |
| V1 | 0.358282 | ≈0.286 | −0.072 |
| V2 | 0.373615 | ≈0.305 | −0.068 |
| V3 | 0.359070 | ≈0.287 | −0.072 |

Subject 006's offset (~−0.07) is 10–15× the "constant" the audit describes, and subject 006 in fact **dominates** the aggregate offset that the audit narrates as uniform. The audit's §3 headline that subject 006 is "0.3047, below chance, sign-inverted" rests on a recomputed value that conflicts with EXP-004's own committed 0.372733 (also below chance, but materially less extreme). The per-subject-006 conflict is **never disclosed** in the audit's §1.2 reconciliation. Files in conflict: `experiments/EXP-004_loso/exp004_metrics.json` vs `reports/EXP-004_AUDIT/data/recomputed_metrics.json`, as narrated in `EXP-004_SCIENTIFIC_AUDIT_REPORT.md`.

**6.3 — MODERATE: Code/documentation mismatches (see §3).** EAR "3D" (2D in code); σ²(MAR) (mean|ΔMAR| in code); "SEVERE never suppressed" invariant (not implemented in fusion); "operating point held constant across variants" (per-variant nearest-TPR in code); "seed 42" (not seeded in LOSO harness). Each is an inconsistency between a stated design/comment and the committed code.

**6.4 — MINOR (already disclosed by the audit).** ±1-frame confusion-matrix differences between the audit recomputation and EXP-004; the `k_sensitivity_recall.png` caption regime mismatch (C-2); the `k` parameter mislabeled "minimum-duration floor" when it is a minimum-overlap tolerance (C-1); the `streams_primary` variable bound to secondary streams (I-1); the seed not echoed in the EXP-005 run log (R-1). These are genuinely minor and, to the project's credit, mostly caught by its own audit.

**What IS consistent (verified):** EXP-002, EXP-003, and EXP-004 are **fully internally consistent** — 100% of checked metrics recompute exactly from their confusion matrices and agree across report/JSON/CSV/registry, and the V3/V4 byte-identity is confirmed by md5. EXP-005's headline numbers triangulate exactly across five artifact types. The arithmetic backbone of the project is sound.

---

## 7. Reproducibility

**Strengths.** The event-metric core is pure and deterministic (no wall-clock, no RNG in the metric path); the bootstrap is seeded; environment is recorded (Apple M1, TFLite XNNPACK CPU, model input [1,24,24,1]); artifacts are mutually consistent and the run log's summary reproduces the report tables. Hyperparameters and splits are frozen and committed. A researcher with the NTHU-DDD corpus could very likely reproduce the frame-level and event-level numbers.

**Gaps.**

- **The "seed 42" reproducibility guarantee is partly illusory** in the LOSO harness (no RNG seeded there; §2). Determinism holds by sorted enumeration, but not by the documented mechanism.
- **The stale root-cause document (§6.1) actively harms reproducibility**: a reproducer cannot tell from the repository which PRIMARY result (0 alarms vs 5–6 TP) is the intended one without external knowledge.
- **The datasets required to reproduce are not bundled** (expected), and NTHU-DDD access is assumed.
- **The Pi 4 feasibility claim is not reproducible** because it was never measured on-device; only Darwin/arm64 latency exists.

**Verdict:** Reproducible in its numerical core; undermined by the unreconciled stale document and the mismatch between documented and actual determinism mechanisms.

---

## 8. Failure Analysis

**Strengths.** The project does not hide its failures. EXP-004's negative result is carried forward honestly and even used to motivate EXP-005. EXP-005's near-zero recall is investigated in `EXP005_ROOT_CAUSE_ANALYSIS.md` with a structured 10-hypothesis table (H1–H10), instrumentation of the fatigue trajectory, and a defensible mechanistic explanation (single active cue EAR caps raw_score ≤ 0.45 < 0.50 MODERATE gate; cold-start EMA cannot climb within a short window; SECONDARY fires only via cross-recording state carryover). As *root-cause reasoning*, this is scientifically convincing and appropriately concludes "no implementation bug, a real system-on-benchmark limitation."

**Critical caveat.** That root-cause analysis was performed on the **superseded 10,800-frame / 16-GT / recall-0.0 run**, not on the committed 66,521-frame / 41-GT / recall-0.12–0.15 run (§6.1). Its central quantitative anchor (recall exactly 0.0, "16/16 missed," max score 0.2156, single MD5) does **not** describe the final experiment. So the *reasoning* is good but its *object* is a stale run. The final EXP-005 report, which shows non-zero (if tiny) recall, does **not** contain an equivalent root-cause analysis of why recall is ~0.12 rather than 0. The failure analysis, as it stands, explains a version of the result that the committed artifacts no longer show.

**Verdict:** Strong failure-analysis *methodology* attached to the *wrong (stale) run*; the current results lack a matching root-cause treatment.

---

## 9. Statistical Quality

**Strengths.** Metrics are computed correctly and recompute exactly from confusion matrices (verified across EXP-002/003/004). The EXP-004 audit layer (DeLong, paired bootstrap B=2000, McNemar, subject-stratified CIs) is genuinely rigorous for frame-level analysis. EXP-005 is appropriately descriptive-only and refuses to run significance tests it cannot support. Event matching (one-to-one, deterministic tie-break, shared-frame overlap) is correctly specified and has no phantom event-level TN/specificity.

**Weaknesses.**

- **Event-level statistics rest on tiny counts** (TP = 5–6; latency median over n = 5, IQR 35.23). Every event-level rate is one event away from swinging. The report says so, but the conclusions ("V3 best FA profile") are drawn from differences of 1–2 false alarms on a single subject.
- **Single-subject dependence** (all FP from subject 005) means the event-level comparison is effectively n = 1 for the quantity of interest.
- **The primary-metric operationalization (§2) weakens the frame-level FPR comparison**: FPR@nearest-achieved-TPR-per-variant is not the same estimator as FPR@held-constant-threshold, and the difference is not quantified anywhere.
- **The subject-006 AUC discrepancy (§6.2)** means one of the two committed AUC values for subject 006 is wrong, and the audit's uniform-offset explanation does not cover it.

**Verdict:** The statistical *machinery* is strong; the statistical *power* for the event-level claims is very low, and two operationalization/consistency issues (metric definition, subject-006 AUC) are unresolved.

---

## 10. Overall Scientific Quality

- **Scientific rigor:** Mixed-to-good. Sound design and honest reporting, undercut by an unproven core claim and unreconciled contradictions.
- **Engineering rigor:** Good compute core; poor discipline in keeping comments/specs synchronized with code (SEVERE, σ², EAR-3D, metric definition).
- **Reproducibility:** Good numerical core; harmed by the stale document and the documented-vs-actual determinism gap.
- **Transparency:** High. Limitations, negative results, and small samples are disclosed prominently.
- **Honesty in reporting:** High for measured results; the failures are the *descriptions* (overstated invariants/contributions) and the *process* (a stale contradictory document left in place, an experiment not registered as required).

The project reads as a **conscientiously-measured but not-yet-validated** research effort: the apparatus works and reports honestly, but it has (a) refuted its own central hypothesis at the frame level, (b) shown near-null capability at the event level, and (c) left a set of implementation/description mismatches and one internal contradiction unresolved.

---

## Weakness Analysis

### Critical
- **None that rise to fabrication or invalidating error.** The numbers are real and reproducible. (I reserve "Critical" for defects that invalidate results or indicate misconduct; none is present. The most severe issues are Major.)

### Major
1. **Central hypothesis is unsupported by the project's own primary experiment.** EXP-004 shows the reliability gate does not reduce FPR@matched-TPR and the speech filter raises it; EXP-005 shows near-zero event-level recall. The motivating claim is, on current evidence, refuted or unproven. *(§4, §5)*
2. **The "SEVERE is never suppressed" safety invariant is not implemented in the fusion engine** — reliability multiplication at `fatigue_fusion.py:197` is unconditional and precedes severity computation. *(§3)*
3. **Stale root-cause document contradicts the committed EXP-005 results and is not marked superseded** (10,800 frames / 16 GT / recall 0.0 / single MD5 vs 66,521 / 41 GT / recall 0.12–0.15 / distinct MD5s). *(§6.1)*
4. **Primary metric is operationalized differently than specified**: FPR at per-variant nearest-achieved-TPR, not at a single held-constant threshold; discrepancy undisclosed. *(§2, §9)*

### Moderate
5. **Subject-006 AUC cross-document discrepancy** (~−0.07) uncovered by the audit's uniform-offset explanation and undisclosed. *(§6.2)*
6. **Code/doc mismatches on named contributions/features:** EAR "3D" (2D in code); "σ²(MAR)" (mean|ΔMAR| in code); "seed 42" not seeded in LOSO harness. *(§3, §6.3)*
7. **Executed scope narrower than frozen spec:** no YawDD evaluation artifact; no on-Pi latency (only Darwin/arm64). *(§1, §5)*
8. **Event-level conclusions rest on a single subject** (all FP from 005; TP = 5–6). *(§4, §9)*
9. **Current EXP-005 result lacks a matching root-cause analysis** (the one on file describes the stale run). *(§8)*

### Minor
10. `k` mislabeled "minimum-duration floor" (is minimum-overlap tolerance); `k_sensitivity_recall.png` caption regime mismatch; `streams_primary` variable bound to secondary streams; seed not echoed in run log; ±1-frame CM rounding. *(all self-disclosed by the EXP-005/EXP-004 audits)*
11. Dead/aspirational code paths (`LearnedReliabilityEstimator`, ensemble/learned-logistic, `CalibrationManager`) and a divergent `main.py` live path.
12. `recursive-churning-lecun.md` referenced but absent from disk.

---

## Reviewer #2 Questions

1. **"Your primary experiment (EXP-004) shows the reliability gate does not reduce FPR and the speech filter increases it. What, then, is the empirical basis for presenting them as contributions?"**
   *Why it matters:* it goes to whether the paper's central claim is supported at all. *Evidence:* EXP-004 report + frozen A4 concede the negative result; EXP-005 shows only a 1–2 false-alarm difference on one subject. *Missing:* any experiment on a corpus where the mechanisms measurably help.

2. **"The frozen spec says SEVERE is never suppressed. Where in the fusion engine is that guaranteed?"**
   *Why it matters:* it is a safety-asymmetry claim. *Evidence:* `fatigue_fusion.py:197` multiplies unconditionally before severity; the only SEVERE guard is a downstream boolean on `should_alarm`, not on the score. *Missing:* a fusion-level SEVERE bypass, or a corrected claim.

3. **"Which EXP-005 PRIMARY result is authoritative — the 0-alarm 10,800-frame run in the root-cause note, or the 5–6-TP 66,521-frame run in the report?"**
   *Why it matters:* the repository asserts both. *Evidence:* both documents are committed; neither cross-references the other. *Missing:* an explicit supersession marker and reconciliation.

4. **"Your FPR@matched-TPR compares variants at each variant's nearest achieved TPR, not at a single fixed threshold. How does that affect the FPR ranking?"**
   *Why it matters:* the fairness of the headline comparison. *Evidence:* `loso_harness._fix_operating_point`/`_fpr_at_tpr`. *Missing:* a sensitivity analysis at a truly fixed score threshold.

5. **"Every false alarm in EXP-005 comes from subject 005. Is the V3-vs-V2 improvement anything more than one subject's idiosyncrasy?"**
   *Why it matters:* generalization. *Evidence:* per-subject CSV; §9 discloses it. *Missing:* additional subjects that fire alarms.

6. **"The spec commits to YawDD and Raspberry Pi 4. Where are those artifacts?"**
   *Why it matters:* claims must match evidence. *Evidence:* only NTHU-DDD results and Darwin/arm64 latency exist. *Missing:* YawDD evaluation and on-Pi profiling.

7. **"The audit calls the AUC offset a near-constant −0.005, but subject 006 differs by −0.07. Which subject-006 AUC is correct, and why does the outlier go unmentioned?"**
   *Why it matters:* one committed value is wrong. *Evidence:* `exp004_metrics.json` vs audit `recomputed_metrics.json`. *Missing:* a reconciliation of subject 006.

8. **"You name a 'variance-based' speech-jitter filter, but the code computes a mean absolute difference. Which is the contribution?"**
   *Why it matters:* a named contribution is mislabeled. *Evidence:* `temporal_analyzer.py:296-298`. *Missing:* corrected description or an actual variance implementation.

9. **"Event-level recall is ~0.12 — the system misses ~88% of drowsy episodes. In what sense is this a working detector?"**
   *Why it matters:* fitness for purpose. *Evidence:* EXP-005 §6.1/§10. *Missing:* an alarm path capable of firing on this corpus, or a reframing as a limitation study.

---

## Missing Evidence (genuinely missing only)

- **A root-cause analysis of the *current* EXP-005 result** (why recall ≈ 0.12, not the stale 0.0).
- **A supersession/reconciliation note** resolving the stale root-cause document against the committed artifacts.
- **A fixed-threshold FPR sensitivity check** to validate the primary-metric comparison.
- **Subject-006 AUC reconciliation** between EXP-004 and its audit.
- **YawDD evaluation** and **on-Pi (Raspberry Pi 4) latency/thermal/memory** artifacts, both promised by the frozen spec.
- **Corrected documentation** for the SEVERE invariant, the σ² statistic, EAR dimensionality, and the LOSO seed.

I do **not** consider the following "missing," because the project reasonably scopes them out: significance testing at n = 4 (correctly declined), a production alarm-controller replay (correctly disclosed as reimplemented), or additional datasets beyond the frozen protocol.

---

## Scientific Score (0–10)

| Category | Score | Justification |
|---|---|---|
| **Research Problem** | 7 | Clear, falsifiable RQ with named baseline and metric. Docked for executed scope (no YawDD/Pi) being narrower than the stated problem. |
| **Methodology** | 5 | Sound design (ablation, LOSO, two-level evaluation, honest stats), but the primary metric is operationalized differently than specified and the seed mechanism is misdocumented. |
| **Experimental Design** | 5 | Complete, diagnostic ablation with real controls; undermined by single-subject FP dependence and by primarily refuting the central claim. |
| **Implementation** | 5 | Correct compute core, but a safety-relevant invariant (SEVERE) is unimplemented and multiple named features/contributions are mislabeled in code and docs. |
| **Statistical Quality** | 5 | Strong machinery (DeLong/bootstrap/McNemar) and correct recomputation; very low power for event-level claims; unresolved subject-006 AUC and metric-definition issues. |
| **Reproducibility** | 5 | Deterministic numerical core and frozen configs, but the stale contradictory document and documented-vs-actual determinism gap materially hurt reproducibility. |
| **Scientific Rigor** | 6 | Honest negative results, structured hypothesis testing, appropriate caution — offset by unreconciled contradictions and an unproven core claim. |
| **Documentation** | 5 | Extensive and mostly transparent, but contains a stale contradictory report, overstated invariants, and mislabeled contributions; a reader trusting the docs would be misled on several points. |
| **Overall Quality** | 5 | An honest, well-instrumented project whose central hypothesis is currently unsupported and whose documentation/implementation are out of sync on several material points. |

---

## Final Verdict

**Scientifically Weak.**

Justification: The project is not *unsound* — its arithmetic is real and reproducible, its statistics are competently executed, and it reports negative results and limitations with unusual honesty (EXP-004 §, EXP-005 §9, EXP-002 F1 disclosure). That honesty and the verified numerical consistency of EXP-002/003/004 keep it above "Unsound." But it does not reach "Acceptable," because: (1) its central hypothesis is currently **refuted at the frame level and unproven at the event level**; (2) a stated **safety invariant is not implemented**; (3) a **committed document contradicts the committed results** with no reconciliation; (4) the **primary metric is operationalized differently than specified**; and (5) **two claimed contributions are mislabeled** in the code. These are Major issues that must be resolved before the work can be called scientifically acceptable.

---

## Final Recommendation

1. **Is this research scientifically sound?**
   Partially. The measurement and reporting are sound; the *claims* are not yet supported by the *evidence*, and several stated mechanisms are not implemented as described. As a body of scientific work it is currently **weak, not sound**.

2. **Is there any evidence of incorrect methodology?**
   Yes. The primary metric (FPR at a held-constant operating point) is not what the code computes (`loso_harness._fix_operating_point` returns a per-variant TPR, and FPR is taken at each variant's nearest-achieved TPR). The "seed 42" determinism is not implemented in the LOSO harness. Neither discrepancy is disclosed.

3. **Is there any evidence of implementation errors?**
   Yes. The "SEVERE is never suppressed" invariant is not implemented in the fusion engine (`fatigue_fusion.py:197`, unconditional reliability multiply before severity). Additionally, EAR is 2D despite "3D" documentation, and the "variance-based" speech filter computes a mean absolute difference, not variance. The compute paths producing the reported numbers are otherwise correct.

4. **Are the experiments sufficient to support the conclusions?**
   The EXP-005 conclusions *as narrowly stated* ("on this corpus, descriptively") are supported. But they are insufficient to support the project's motivating claims: event-level recall ≈ 0.12 with all false alarms from one subject, and a frame-level result that refutes the gate's benefit. The experiments are not sufficient to conclude the approach works.

5. **Is any additional experiment scientifically required, or would further experiments only strengthen the work?**
   Additional experiments are **required**, not merely strengthening. At minimum: (a) reconcile/rerun EXP-005 so a single authoritative PRIMARY result exists with a matching root-cause analysis; (b) a fixed-threshold FPR sensitivity check; (c) an alarm path (or corpus) on which the components can actually fire, so the central claim can be tested rather than refuted-by-silence; (d) the promised YawDD and Raspberry-Pi artifacts if those claims are to remain. Reconciling subject 006's AUC is also required.

6. **If you were my PhD supervisor, would you approve moving forward to paper writing based on the current evidence?**
   **No — not yet.** I would not approve paper writing on the current evidence. The blocking reasons are concrete and fixable: (1) resolve the contradictory committed documents so the repository states one authoritative EXP-005 result; (2) correct or implement the SEVERE invariant and the σ²/EAR-3D/seed/metric descriptions so the paper's method section can be truthful; (3) provide at least one experiment in which the claimed contributions demonstrably help, or reframe the paper honestly as a *negative/limitation study* (which the evidence would genuinely support and which the project's honest reporting is well-suited to). Once (1)–(3) are addressed, a limitation-focused paper is defensible; a positive-claim paper is not, on current evidence. I would meet again after the reconciliation and the fixed-threshold check before authorizing writing.

---

*End of independent scientific review. Every conclusion above is tied to a named project artifact or source location; where evidence is absent (YawDD, on-Pi latency, a current-run root-cause), this review states the absence rather than assuming a result.*
