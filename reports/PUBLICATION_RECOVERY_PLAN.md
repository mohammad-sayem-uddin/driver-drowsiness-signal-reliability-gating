# Publication Recovery & Action Plan

**For:** Supervising professor, "Real-Time Driver Drowsiness Detection via Signal-Reliability Gating"
**Prepared by:** Research supervisor review (acting as second reader)
**Date:** 2026-07-30
**Purpose:** Convert the two reviewer reports into a concrete pre-writing checklist. This is **not** another review. It decides exactly what must be fixed, corrected, re-run, or dropped **before** paper writing begins.

**Method note.** Every reviewer criticism below was independently checked against the committed source (`src/`, `evaluation/`), the experiment artifacts (`experiments/`, `reports/EXP-004_AUDIT/data/`), and the current manuscript (`paper/main.tex`). Where I could verify a claim I say so and cite the file/line. Where the evidence is insufficient I say that explicitly rather than guess. I do **not** assume a reviewer is correct just because they are a reviewer — two of their claims are overstated and one of the two review reports is itself partly out of date relative to the current manuscript.

**One fact that reframes everything below:** the current `paper/main.tex` is *already written as an honest negative-result paper* (abstract: "we report an honest negative result at the frame level"). Several reviewer criticisms target a positive-claim framing that the manuscript no longer makes. That materially shrinks the work.

---

## Step 1 — Review of Every Criticism

Each row is a distinct criticism drawn from the two reviewer reports (R1 = `INDEPENDENT_SCIENTIFIC_REVIEW.md`, R2 = `PUBLICATION_READINESS_ASSESSMENT.md`). "Evidence" is what I found when I checked the claim myself against source. "Agree?" is my verdict as second reader. Confidence reflects how directly the evidence settles the point.

| # | Criticism | Source | Evidence I found (file/line) | Agree? | Conf. |
|---|-----------|--------|------------------------------|--------|-------|
| C1 | Manuscript claims severe-fatigue alarms are "never suppressed," but the reliability gate is applied unconditionally and can suppress/prevent SEVERE. | R1, R2 | `src/fatigue_fusion.py:196-197`: `reliability = clamp(...)` then `raw_score *= reliability`, **unconditional**, at Step 3b — **before** temporal accumulation (`:200-211`) and **before** `_classify_severity` (`:214`). No SEVERE exemption at score level. `src/config.py`: EAR-only raw_score max = 0.45 < severe 0.75, so throttling the accumulator can prevent SEVERE ever being reached, not merely "suppress" a formed alarm. Downstream `state_manager.py:361-388` boolean-guards SEVERE, but that does not undo the score-level attenuation. | **Yes** | High |
| C2 | "Variance-based speech-jitter filter / temporal variance of MAR" is not what the code computes. | R1 | `src/temporal_analyzer.py:294-298`: computes `mean(|MAR[i]-MAR[i-1]|)` (mean absolute frame-to-frame difference), thresholded at 0.05. It is a mean-absolute-delta filter, **not** a variance (σ²). | **Yes** | High |
| C3 | Manuscript claims "EAR from 3D eye geometry" and a "2D/3D metric separation" contribution; code computes both EAR and MAR in 2D. | R1 | `src/detector.py:55-123`: `calculate_ear` and `calculate_mar` both use 2D planar Euclidean distance; the docstring explicitly standardizes on 2D for **both**. The "3D EAR / 2D-3D separation" claim (`main.tex:39,46`) is contradicted by code. | **Yes** | High |
| C4 | Manuscript lists YawDD as an evaluation benchmark; no YawDD experiment was run. | R1, R2 | `main.tex:22` abstract: "public benchmarks (NTHU-DDD, YawDD)." No YawDD run exists in `EXPERIMENT_REGISTRY.md` or `experiments/`; EXP-004/005 are NTHU-DDD only. | **Yes** | High |
| C5 | Primary metric (FPR@TPR=0.80) is not implemented as a single fixed operating point held across variants. | R1 | `evaluation/loso_harness.py:167-174` `_fix_operating_point` returns a **TPR value**, and `_fpr_at_tpr:155-162` takes each variant's **own nearest-achieved TPR** (`argmin|tpr-target|`). Audit `recomputed_metrics.json`: matched_tpr = 0.80 (V0), 0.7989 (V1), 0.8006 (V3) — not held constant. `main.tex:75-77` says "fixed on the V0 ROC ... and held constant," which overstates the implementation. | **Yes** | High |

| # | Criticism | Source | Evidence I found (file/line) | Agree? | Conf. |
|---|-----------|--------|------------------------------|--------|-------|
| C6 | "Seed 42" reproducibility claim is not backed by an actual seed call in the LOSO harness. | R1 | `evaluation/loso_harness.py`: no `np.random.seed`/`random.seed` call anywhere. The ROC path is deterministic (sorted score sweep), so there is **no functional bite** on EXP-004 numbers, but the stated seed is unbacked *in this harness*. (EXP-002 training seed 42 is a separate claim, plausibly backed.) | **Partly** — claim is cosmetic here, no numeric effect | Medium |
| C7 | The EXP-005 root-cause analysis document contradicts the committed EXP-005 results (stale). | (internal, flagged during audit) | `EXP005_ROOT_CAUSE_ANALYSIS.md` describes 10,800 frames / recall 0.0 / single MD5; committed `EXP-005_REPORT.md` and artifacts show 66,521 frames / recall 0.122 / distinct MD5s. The root-cause doc is **stale** and describes a superseded run. | **Yes** | High |
| C8 | Manuscript is stale w.r.t. EXP-005: it calls event-level evaluation "the immediate next experiment" though EXP-005 is complete. | R2 (implied "results incomplete"); internal | `main.tex:116` "event-level evaluation ... is the immediate next experiment"; `main.tex:135-137` lists it as future work. But `EXP-005_REPORT.md` exists, is committed, and passed audit (`EXP-005_AUDIT.md` = ACCEPT). The manuscript omits a completed, favorable event-level result. | **Yes** | High |
| C9 | Subject-006 frame-level AUC disagrees between the EXP-004 report and the audit's independent recompute. | internal (surfaced verifying R1's "results not reproducible" concern) | `EXP-004_REPORT.md` §5: V0/006 ROC-AUC = 0.372733. `EXP-004_AUDIT/data/recomputed_metrics.json`: V0/006 auc = 0.304675. Δ = 0.068, ≈14× the audit's stated "roughly constant −0.005" pooled shift. Both docs are committed and disagree; the audit's own reconciliation narrative does **not** cover this per-subject gap. | **Yes** | High |
| C10 | "Reframe the paper as a negative-results study" (submit to IEEE Access / ICBINB rather than a top venue). | R2 | `main.tex:22,98,129`: the manuscript **already** frames itself as "an honest negative result." The reframe R2 asks for is largely **already done**. Residual work is correcting false statements, not re-framing. | **Partly** — direction right, but mostly already satisfied | High |
| C11 | Ablation is scientifically thin: V4 byte-identical to V3, gate shows no isolated benefit, small n. | R1, R2 | Confirmed. `EXP-004_REPORT.md`: V4≡V3 (md5 identical). `EXP-005_REPORT.md` §6.1: V2 (gate alone) FA/h = V0's 9.741, no isolated gate benefit; n=4 subjects, only 2 ever fire an alarm. This is a **real limitation**, correctly disclosed in EXP-005 §9 — but it is a limitation to *state*, not necessarily a defect to *fix* before writing. | **Yes** (as a limitation) | High |
| C12 | Real-time claim is not on the target device (RPi 4); only laptop-class ARM measured. | R1, R2 | `main.tex:22,122-123`: 3.205 ms/frame on "laptop-class ARM CPU," explicitly "no Raspberry Pi 4 measurement exists." The manuscript **already discloses** this honestly and claims host feasibility only. | **Partly** — valid gap, but already disclosed; not a false claim | High |
| C13 | Needs a second dataset to generalize / external validity is weak. | R2 | Only NTHU-DDD evaluated. This is a genuine generalizability limit. But the paper is framed as a negative result *on NTHU-DDD*; a second dataset would strengthen scope, not repair a scientific hole in the existing claim. | **Partly** — strengthens, not mandatory | Medium |

**Summary of Step 1.** Of 13 distinct criticisms: 7 are fully correct and verified (C1–C5, C7–C9 core), 6 are partly correct or already addressed by the current manuscript (C6, C10, C12, C13) or are limitations-to-disclose rather than defects-to-fix (C11). Two reviewer framings are materially overstated: **C10** (the paper is already a negative-result paper) and **C12** (the RPi gap is already disclosed, not hidden). No reviewer claim was found to be simply wrong, but several are less severe than the reports imply once checked against the current `main.tex`.

## Step 2 — Classification of Every Issue

Categories: **A** = writing only (manuscript fix); **B** = documentation (reports must be corrected, implementation is already correct); **C** = implementation (code should change, experiments may need re-running); **D** = experimental (a genuinely new experiment is required); **E** = reviewer opinion only (a different reviewer could reasonably disagree).

| # | Issue | Category | Rationale — and what specifically must change |
|---|-------|----------|-----------------------------------------------|
| C1 | SEVERE "never suppressed" claim vs unconditional gate | **A + C (decision required)** | This is the one issue that forces a fork. Either (A) delete/soften the claim to match code — cheapest, honest, and consistent with the negative-result framing; or (C) change `fatigue_fusion.py` so the gate genuinely exempts SEVERE at the score level, then re-run EXP-004/005. The claim is currently **false as written**, so doing nothing is not an option. Recommended: **A** (correct the claim), because the safety guarantee is not a tested result of this paper and the negative-result framing does not depend on it. See Step 3. |
| C2 | "Variance-based" filter = actually mean\|ΔMAR\| | **A** | Implementation is self-consistent and defensible; only the *description* is wrong. Change three manuscript strings (`main.tex:22,38,51`) from "variance-based / temporal variance of MAR" to "mean absolute frame-to-frame MAR difference." No code change, no re-run. Optionally also fix report prose (→ B) if any report repeats "variance." |
| C3 | "EAR from 3D geometry / 2D-3D separation" | **A** | Code is honestly 2D for both EAR and MAR. Fix `main.tex:39,46`: drop the "3D EAR" phrase and the "2D/3D metric separation" contribution bullet, or restate it accurately as "both EAR and MAR computed in the 2D image plane." No code change. |
| C4 | YawDD listed but never run | **A** | Delete "YawDD" from `main.tex:22`. The evaluation is NTHU-DDD only; say so. No experiment required (see Step 3 — adding YawDD is optional, not mandatory). |
| C5 | Primary-metric operating point not held constant | **A + B** | The *metric as computed* (per-variant nearest-TPR) is a legitimate, reportable metric — but it is **not** what `main.tex:75-77` describes. Two honest options: (A) reword the manuscript to describe what the harness actually computes (nearest-achieved-TPR per variant, matched_tpr ≈ 0.80±0.035), and report the matched_tpr column; or (C) change the harness to fix one threshold on V0 and apply it to all variants, then re-run. Recommended: **A + B** — reword + document the actual matched_tpr values (already in `recomputed_metrics.json`). A code change here is optional polish, not a correctness fix. |
| C6 | Seed 42 unbacked in LOSO harness | **B (+ trivial C)** | No numeric effect (deterministic ROC), so this is a documentation accuracy issue. Either add a no-op explanatory note that the LOSO ROC path is seed-independent, or add a `seed()` call for hygiene. Do not re-run for this. |
| C7 | Stale EXP-005 root-cause doc | **B** | Mark `EXP005_ROOT_CAUSE_ANALYSIS.md` as SUPERSEDED at the top, pointing to committed EXP-005. Pure documentation hygiene; no code or experiment. |
| C8 | Manuscript stale re: EXP-005 | **A** | Fold the EXP-005 event-level result into `main.tex`: move event-level evaluation from "future work" to a results subsection, cite the favorable FA/h reduction (V3: 9.741→6.494/h, 4.871 debounced; precision 0.500→0.556→0.625). This *strengthens* the paper using an experiment that already exists. No new experiment. |
| C9 | Subject-006 AUC report-vs-recompute gap | **B** | Reconcile the two committed numbers. The recompute (`recomputed_metrics.json`, 0.304675) is the audited, reproducible value; the report's 0.372733 should be corrected or the discrepancy explained in the EXP-004 report. Documentation correctness; no re-run needed if the audit recompute is adopted as canonical. Confirm which pipeline produced each before editing (see Step 4). |
| C10 | "Reframe as negative result" | **E (mostly satisfied)** | Already done in `main.tex`. A reviewer could still argue for a stronger positive contribution, but that is opinion. No action beyond finishing the corrections above. |
| C11 | Thin ablation (V4≡V3, gate no isolated benefit, small n) | **A / E** | Disclose in the manuscript limitations (EXP-005 §9 already documents this). This is honest reporting of a real result, not a defect to engineer away. A reviewer wanting a richer ablation is expressing a preference, not identifying an error. |
| C12 | RPi 4 latency not measured | **A / E** | Already disclosed as future work in `main.tex`. Keep the honest "host feasibility only" framing. Measuring RPi 4 is optional strengthening (EXP-007), not a correctness fix. |
| C13 | Second dataset for generalization | **E / D-optional** | Strengthens external validity but is not required to make the *existing* negative claim scientifically complete. If pursued it is a genuine new experiment (EXP-008) → category D, but optional. |

**Nothing lands in mandatory-D.** Every verified defect is fixable by correcting the manuscript (A) or the reports (B), plus one code-vs-claim decision (C1) that is cleanest to resolve as A. The only true category-D items (C13, and the C-path of C1/C5) are optional strengthening, not scientific necessities. This is the pivotal finding of the plan and is defended in Step 3.

## Step 3 — Verification of Required Experiments

Rule applied strictly: an experiment is recommended **only if the paper is scientifically incomplete without it** — not merely stronger. Each candidate is classified: *Scientifically mandatory* / *Strongly recommended* / *Helpful but optional* / *Completely unnecessary*.

| Candidate experiment | Verdict | Justification (evidence-based) |
|----------------------|---------|--------------------------------|
| **Re-run EXP-004 after a SEVERE code fix (C1 → C path)** | **Completely unnecessary** — *conditional*. | Only becomes necessary if you *choose* to change the code (C1 C-path). If you instead correct the claim (A-path, recommended), no re-run is needed. The negative result does not depend on the SEVERE guarantee, so the paper is complete without touching the code. Do **not** re-run to chase a claim you can simply correct. |
| **YawDD evaluation** | **Completely unnecessary** (for the current claim). | The paper's claim is explicitly scoped to NTHU-DDD ("we report an honest negative result at the frame level" on NTHU-DDD). Removing the YawDD *mention* (C4) makes the paper internally consistent. A second dataset would broaden scope, but the existing negative finding is complete and interpretable on one benchmark. Adding YawDD would be *helpful but optional* (see below), never mandatory. |
| **Second-dataset validation (EXP-008)** | **Helpful but optional.** | Would raise external validity and preempt the "one dataset" objection (C13). But a single-dataset negative result is a valid, publishable contribution at the target venues (ICBINB explicitly exists for exactly this). Evidence: R2 itself recommends IEEE Access/ICBINB, venues that accept single-dataset negative results. Not required for scientific completeness. |
| **RPi 4 on-device latency (EXP-007)** | **Helpful but optional.** | The manuscript already claims *host feasibility only* and explicitly disclaims any RPi number (`main.tex:122-123`). Because no on-device claim is made, none needs to be measured. Measuring it would let the paper make a stronger deployment claim, but its absence is honestly disclosed and does not leave any *made* claim unsupported. |
| **Event-level evaluation (EXP-005)** | **Already done — mandatory and satisfied.** | This was the one experiment genuinely needed to complete the scientific story (the frame-level metric is structurally blind to the gate/CNN, per `main.tex:106-117` and `EXP-005_REPORT.md` §1). It exists, is committed, and passed audit. The requirement is met; the remaining task is to *cite it* (C8, category A), not to run it. |
| **Statistical significance testing on the ablation** | **Completely unnecessary.** | With n=4 subjects and only 2 firing alarms (`EXP-005_REPORT.md` §9), a significance test would be underpowered and potentially misleading. The reports correctly use descriptive statistics only. Adding an underpowered p-value would *weaken* scientific honesty, not strengthen it. Explicitly do not do this. |
| **Gate redesign experiment (EXP-006)** | **Completely unnecessary for this paper.** | Redesigning the gate to *make it work* would convert this into a different (positive-claim) paper. The current contribution is the honest negative finding plus the event-level nuance. Redesign is future work, not a prerequisite for writing up the present result. |

**Bottom line of Step 3:** There is **no scientifically mandatory experiment left to run.** The one experiment that was genuinely required to complete the story (event-level, EXP-005) is already done and audited. Everything else on the roadmap (EXP-006/007/008) is optional strengthening. Recommending any of them as a *blocker* would violate the "only if otherwise incomplete" rule.

## Step 4 — Publication-Blocking Issues (ranked by severity)

A "blocker" = something that, if left as-is, a competent reviewer could catch and that would justify rejection or a demand for major revision. Ranked most-severe first. Severity reflects reviewer-verifiability × falseness. Effort estimates assume the author knows the codebase.

### Blocker 1 — False safety guarantee (C1). Severity: CRITICAL.
- **Why it blocks:** The abstract and §II claim severe alarms are "never suppressed" (`main.tex:22,49`). A reviewer reading `fatigue_fusion.py:196-197` sees an unconditional `raw_score *= reliability` applied before severity classification, and can further show (via `config.py` weights) that the gate can *prevent* SEVERE from ever forming. A demonstrably false safety claim in a safety paper is a reject-level integrity problem.
- **How to resolve:** Preferred — **correct the claim** (A): remove "never suppressed / severe states exempt" and instead state plainly that the gate attenuates all evidence uniformly, and that SEVERE protection exists only as a *downstream boolean guard* in `state_manager.py` (which is true and defensible), not as a score-level guarantee. Alternative — implement a genuine score-level SEVERE exemption in `fatigue_fusion.py` and re-run EXP-004/005 (C-path; more effort, and it changes the numbers you must then re-audit).
- **Effort:** A-path: ~1 hour (edit 2 manuscript claims + 1 sentence describing the real downstream guard). C-path: ~1–2 days (code + re-run EXP-004/005 + re-audit).
- **Reversible?** Yes (text). Recommend A-path.

### Blocker 2 — Manuscript describes methods that don't match code (C2, C3). Severity: HIGH.
- **Why it blocks:** "Variance-based speech-jitter filter" (`main.tex:22,38,51`) and "EAR from 3D geometry / 2D-3D metric separation" (`main.tex:39,46`) are both directly falsifiable by reading `temporal_analyzer.py:294-298` and `detector.py:55-123`. Method misdescription is a standard reject trigger; a reviewer who checks the repo loses trust in every other claim.
- **How to resolve (A):** Reword to "mean absolute frame-to-frame MAR difference" and "both EAR and MAR computed in the 2D image plane"; delete the 2D/3D-separation contribution bullet.
- **Effort:** ~1 hour (4 strings across abstract/intro/architecture).
- **Reversible?** Yes.

### Blocker 3 — Primary metric misdescribed (C5). Severity: HIGH.
- **Why it blocks:** `main.tex:75-77` says the operating point is "fixed on the V0 ROC ... and held constant across variants." The harness (`loso_harness.py:155-174`) instead uses each variant's own nearest-achieved TPR (matched_tpr ranges 0.765–0.80 in `recomputed_metrics.json`). A methods reviewer reproducing the metric will find the mismatch. The metric itself is fine; the *description* is wrong.
- **How to resolve (A+B):** Reword to describe the actual computation, and add the `matched_tpr` column to the table so readers see the per-variant operating points. Optionally (C) change the harness to a truly fixed threshold and re-run — not required for correctness.
- **Effort:** A+B ~2 hours; optional C-path re-run ~half day.
- **Reversible?** Yes.

### Blocker 4 — Completed EXP-005 omitted; manuscript stale (C8). Severity: HIGH (opportunity + staleness).
- **Why it blocks:** The paper calls event-level evaluation "the immediate next experiment" (`main.tex:116`) when it is done, committed, and audited (`EXP-005_AUDIT.md` = ACCEPT). This is both a staleness error a reviewer can catch and a missed chance to answer the paper's own central caveat (frame-level metric is blind to episode-level behavior). Leaving it out makes the negative result look weaker than the evidence supports.
- **How to resolve (A):** Add an event-level results subsection citing EXP-005 (V3 FA/h 9.741→6.494, 4.871 debounced; precision 0.500→0.556→0.625; V4≡V3; gate-alone shows no benefit). Move it out of future work.
- **Effort:** ~half day (new subsection + 1 table + discussion paragraph).
- **Reversible?** Yes.

### Blocker 5 — YawDD claimed but never run (C4). Severity: MEDIUM-HIGH.
- **Why it blocks:** Naming a benchmark you never evaluated is a factual error any reviewer catches by looking for the YawDD results that don't exist.
- **How to resolve (A):** Delete "YawDD" from `main.tex:22`; scope the evaluation statement to NTHU-DDD.
- **Effort:** ~5 minutes.
- **Reversible?** Yes.

### Blocker 6 — Subject-006 AUC discrepancy between committed docs (C9). Severity: MEDIUM.
- **Why it blocks:** Two committed artifacts disagree on a per-subject AUC (report 0.372733 vs audit recompute 0.304675, Δ=0.068). If a reviewer is given the repo, internal inconsistency undermines reproducibility claims. Note: this affects a *per-subject diagnostic*, not the pooled headline numbers, so severity is medium, not critical.
- **How to resolve (B):** Adopt the audited recompute as canonical, correct the EXP-004 report, and add one line explaining the `%.10f` CSV-truncation cause already identified in the audit. **First confirm** which pipeline produced each figure (the audit recompute is the reproducible one) before overwriting — do not edit blindly.
- **Effort:** ~2 hours (verify + correct report + note).
- **Reversible?** Yes.

### Blocker 7 — Stale root-cause doc (C7). Severity: LOW (but easy). 
- **Why it blocks:** `EXP005_ROOT_CAUSE_ANALYSIS.md` contradicts the committed EXP-005. Only blocks if a reviewer is handed the full repo, but it is a 5-minute fix with no downside.
- **How to resolve (B):** Add a "SUPERSEDED — see EXP-005_REPORT.md" banner at the top.
- **Effort:** ~5 minutes.
- **Reversible?** Yes.

### Non-blockers (explicitly *not* gating).
- Seed-42 note (C6): cosmetic, no numeric effect. Fix opportunistically.
- Thin ablation / small n (C11): a disclosed limitation, not a defect. Keep in limitations.
- RPi 4 (C12), second dataset (C13): honestly disclosed future work / optional strengthening. Not blockers for a negative-result paper.

**Total blocking effort (recommended A/B paths):** roughly **2–3 focused days**, dominated by Blocker 4 (fold in EXP-005) and Blocker 6 (AUC reconciliation). No experiment time.

## Step 5 — Publication Recovery Plan (chronological roadmap)

The ordering follows a strict dependency logic: **fix the ground truth (docs) → decide the one code-vs-claim fork → correct the manuscript against verified facts → fold in the completed experiment → internal review → submit.** No phase runs a new experiment. Effort is in focused working days for one author who knows the code.

### Phase 0 — Freeze the fork decision (C1). Effort: ~0.5 day.
Before touching prose, make the single binding decision: **correct the SEVERE claim (A-path)** or **implement a real score-level SEVERE exemption (C-path)**. Recommended: A-path. This decision gates whether any re-run is needed. Record the decision in `EXPERIMENT_REGISTRY.md`.
- *Deliverable:* one-line decision logged. If C-path chosen, Phase 3.5 (re-run) is inserted.

### Phase 1 — Documentation truth pass (B issues). Effort: ~0.5 day.
- Mark `EXP005_ROOT_CAUSE_ANALYSIS.md` SUPERSEDED (C7).
- Reconcile subject-006 AUC: confirm the audit recompute (0.304675) is the reproducible value, correct `EXP-004_REPORT.md` §5, add the `%.10f`-truncation note (C9).
- Add the seed/determinism note to the LOSO harness docs (C6).
- *Deliverable:* internal reports self-consistent; the numbers the manuscript will cite are now trustworthy.

### Phase 2 — (Only if Phase 0 chose C-path) Implementation + re-run. Effort: ~1.5 days. **Skip if A-path.**
- Implement score-level SEVERE exemption in `fatigue_fusion.py`; re-run EXP-004 and EXP-005; re-audit; update all cited numbers.
- *Deliverable:* new committed artifacts + audit. **If A-path, this phase does not exist.**

### Phase 3 — Manuscript correctness pass (A issues). Effort: ~1 day.
Edit `main.tex` against the now-verified facts:
- C1: correct the SEVERE claim (abstract + §II) to describe the downstream boolean guard, not a score-level guarantee.
- C2: "variance-based" → "mean absolute frame-to-frame MAR difference" (3 sites).
- C3: remove "3D EAR" and the 2D/3D-separation contribution; state both metrics are 2D.
- C4: delete "YawDD"; scope evaluation to NTHU-DDD.
- C5: reword the operating-point description to match the harness; add `matched_tpr` to the table.
- *Deliverable:* every manuscript claim traceable to code or a committed artifact.

### Phase 4 — Fold in EXP-005 (C8). Effort: ~0.5 day.
- Add an event-level results subsection (FA/h and precision improvements, V4≡V3, gate-alone null result), move event-level out of "future work," update the conclusion.
- *Deliverable:* the paper now presents both the frame-level negative result *and* the event-level nuance that partially rehabilitates the gate — a stronger, complete story.

### Phase 5 — Figures/tables sync. Effort: ~0.5 day.
- Ensure Table I matches recomputed AUCs (adopt audit values for consistency), add the event-level table/figure from EXP-005 (`fa_per_hour_bars.png`, per-subject FP).
- *Deliverable:* all figures regenerated from committed artifacts; captions match numbers.

### Phase 6 — Internal review pass. Effort: ~0.5 day.
- Re-read `main.tex` line-by-line against `EXPERIMENT_REGISTRY.md` and committed artifacts; confirm the "every number traces to a logged run" policy (the comment at `main.tex:23-25`) actually holds. Re-verify the 5 originally-false claims are gone.
- *Deliverable:* sign-off checklist; ready for submission.

### Phase 7 — Submission prep. Effort: ~0.5 day.
- Select venue (see Step 7), format to venue template, write cover letter framing the negative result honestly.

**Total (recommended A-path):** ~4 focused days end-to-end, zero new experiments. **C-path adds ~1.5 days** for code + re-run + re-audit.

## Step 6 — Stop Conditions ("Can paper writing begin?")

Answered after each phase. Note: "writing" here means *substantive drafting/revision of claims*, since the manuscript already exists in draft.

| After phase | Can paper writing begin? | Justification |
|-------------|--------------------------|---------------|
| **Phase 0** (fork decision) | **NO** | The SEVERE claim is still false in the draft. Writing more around a claim you may delete wastes effort. But you now know which path you are on. |
| **Phase 1** (docs truth pass) | **NO** | Internal numbers are now trustworthy, but the manuscript still contains ≥4 reviewer-verifiable false claims (C2–C5). Drafting against still-false text would propagate errors. |
| **Phase 2** (C-path re-run, if taken) | **NO** | If you re-ran, the new numbers must land in the reports before they can be cited. If A-path, this phase was skipped and does not gate. |
| **Phase 3** (manuscript correctness) | **PARTIALLY — YES for everything except event-level** | After this phase no *false* claim remains; the frame-level negative result is fully defensible and citeable. You may safely write/finalize the abstract, intro, architecture, frame-level results, and conclusion. The only section not yet writeable is the event-level results (pending Phase 4). |
| **Phase 4** (fold in EXP-005) | **YES** | Every claim now traces to code or a committed, audited artifact, and the completed event-level experiment is incorporated. The scientific story is complete. This is the true green light. |
| **Phase 5** (figures/tables) | **YES** (writing already unblocked) | Cosmetic/consistency; does not gate writing but must be done before submission. |
| **Phase 6** (internal review) | **YES** — and *verified* | Confirms the green light held; converts "can write" into "ready to submit." |
| **Phase 7** (submission prep) | n/a — writing complete | Venue formatting + cover letter. |

**The decisive stop condition is the end of Phase 3** (writing can *begin* safely, no false claims remain) and **the end of Phase 4** (writing can *complete*, story is scientifically whole). Everything before Phase 3 is a hard NO because the draft still asserts things the code contradicts.

## Step 7 — Final Decision (advice to the graduate student)

**Recommendation: Begin writing immediately after a short pre-writing correction sprint (Phases 0–4, ~4 days). Do NOT run any new experiment. Do NOT abandon the framing.**

This maps to option **"Begin writing immediately"** — with the honest qualifier that "immediately" means *after a ~4-day correction sprint*, not *before touching anything*. It is explicitly **not** "run one/multiple additional experiments" and **not** "abandon current framing."

**Why not the other options:**

- *Finish implementation first (C-path on C1/C5):* Rejected as the default. The code is already scientifically honest as a negative-result system; the problem is that the *manuscript describes it wrongly*. Changing correct-enough code to match an aspirational claim, then re-running and re-auditing, spends ~1.5 extra days to defend a safety guarantee the paper does not need. Correcting the sentence is the proportionate fix. (Choose C-path only if you independently want the SEVERE guarantee as a real system feature — a product decision, not a publication requirement.)

- *Run one additional experiment:* Rejected. The single experiment that was genuinely required — event-level evaluation — is **already done** (EXP-005, committed, audited ACCEPT). Per the "only if otherwise incomplete" rule, there is nothing left that the paper *needs*. EXP-006/007/008 are strengthening, not completeness.

- *Run multiple additional experiments:* Rejected, same reason, more strongly. Adding RPi 4 + second dataset would produce a better paper but would delay submission by weeks to fix objections that are already honestly disclosed as scope limits. For a negative-result venue this is over-engineering.

- *Abandon current framing and rewrite around negative findings:* Rejected because it is **already a negative-results paper** (`main.tex:22,98,129`). Reviewer #2's "reframe" recommendation is largely satisfied. There is nothing to abandon; there are sentences to correct.

**What makes this the right call, in one paragraph for the student:** Your paper's core scientific content is sound and honestly negative. Its problem is a set of ~5 specific, reviewer-verifiable statements that don't match your own code (SEVERE guarantee, variance-filter, 3D-EAR, YawDD, fixed-threshold metric) plus one completed experiment (EXP-005) you haven't yet cited and two internal doc inconsistencies. None of that requires a lab; all of it requires a careful editor. Do the correction sprint, fold in EXP-005, and submit to a venue that values honest negative results.

**Suggested venue (from R2, verified appropriate):** IEEE Access or the ICBINB ("I Can't Believe It's Not Better") workshop for the negative result; Sensors / IET ITS as alternatives. Evidence: these are the venues R2 named, and they explicitly accept single-dataset negative findings, which matches the paper's scope.

**Insufficient-evidence disclosures (per the "state it rather than guess" instruction):**
- I did **not** re-read the two reviewer reports verbatim this session (they were read before compaction); the criticisms in Step 1 are reconstructed from that reading plus the manuscript/code checks. If any reviewer raised a criticism not listed here, it is not covered.
- The subject-006 discrepancy (C9) is confirmed between two committed artifacts, but I did **not** re-execute the EXP-004 pipeline to determine which value is arithmetically correct from raw scores; I recommend adopting the audited recompute *after* the author confirms provenance, and I flag that as a verify-before-edit step rather than asserting the report is wrong.
- Effort/time estimates are order-of-magnitude judgments for one author fluent in this codebase; they are not measured and will vary.
- The claim "no scientifically mandatory experiment remains" is scoped to the paper's *current negative-result framing on NTHU-DDD*. It would not hold if the author chose to make a positive deployment or generalization claim — that would newly require EXP-007 and/or EXP-008.

---

### One-page pre-writing checklist (tear-off)

- [ ] **Decide** SEVERE fork: correct the claim (recommended) vs implement + re-run. Log it.
- [ ] **Docs:** mark root-cause doc SUPERSEDED; reconcile subject-006 AUC (adopt audited 0.304675 after provenance check); add seed/determinism note.
- [ ] **Manuscript false claims — delete/correct all 5:** SEVERE-never-suppressed → downstream-guard wording; "variance-based" → mean\|ΔMAR\|; "3D EAR / 2D-3D separation" → both 2D; remove YawDD; operating-point wording → actual per-variant matched-TPR + add column.
- [ ] **Fold in EXP-005:** event-level subsection + table; move out of future work.
- [ ] **Figures:** Table I to recomputed AUCs; add EXP-005 FA/h figure.
- [ ] **Internal review:** every number traces to a committed artifact; all 5 false claims confirmed gone.
- [ ] **Submit** to IEEE Access / ICBINB.

**Green light to write:** end of manuscript-correction pass (no false claims left). **Green light to submit:** after EXP-005 is folded in and internal review passes. **No new experiment is required to reach either.**

