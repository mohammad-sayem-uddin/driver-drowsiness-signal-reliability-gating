# Independent Publication Readiness Assessment

**Role of this document.** This is an independent, adversarial peer review of the
driver-drowsiness-detection project, written as by an external senior reviewer for a
mid-tier peer-reviewed venue in Computer Vision / Machine Learning / Intelligent
Transportation Systems. I am not an author. The default posture is **rejection unless the
evidence compels otherwise**. Every conclusion below is tied either to a project artifact I
verified, or to published literature; where evidence is insufficient I say so rather than
guess.

**Headline verdict (stated up front, defended throughout).** The project is a *rigorous,
unusually honest negative result*: on NTHU-DDD (LOSO, n = 4 subjects) the central claimed
mechanism — a decomposed multiplicative signal-reliability gate that should cut the
false-positive rate — **does not reduce false positives at the frame level and provides no
benefit in isolation at the event level**, the optional CNN arm is **inert**, and the target
deployment platform (Raspberry Pi 4) is **never measured**. The engineering discipline and
the candor of the reporting are real strengths. But as a scientific contribution for a
mid-tier conference the work is, in its current form, **not yet publishable**: the positive
claim is refuted, the negative claim is under-powered (single corpus, n = 4, all false
alarms from one subject), and one of the two headline novelties is **not actually implemented**
as described.

**Final decision: Weak Reject** (a clean *Reject* if submitted as a positive-result system
paper; potentially *Borderline* only after the mandatory additional experiments in §9). Full
reasoning in §12–§13.

---

## 1. Inputs Read (and what could not be found)

Per the task, I read and independently re-verified the following:

| Input named in the prompt | Located? | What I did with it |
|---|---|---|
| Independent Scientific Research Review | `reports/INDEPENDENT_SCIENTIFIC_REVIEW.md` | Treated as an evidence *summary*; independently re-verified its major conclusions against JSON/CSV artifacts (see §5, §8). |
| EXP-005 Report | `reports/EXP-005_REPORT.md` | Re-verified headline numbers against `exp005_event_metrics.json` and the per-variant/per-subject CSVs. |
| EXP-005 Audit | `reports/EXP-005_AUDIT.md` | Read in full; adopted its cross-artifact triangulation, re-checked its two flagged prose defects. |
| Root Cause Analysis | `reports/EXP005_ROOT_CAUSE_ANALYSIS.md` | Read; **found it stale** — it describes a superseded run (primary recall 0.0, 16/16 GT missed, 10,800 frames) that **contradicts** the committed EXP-005 (66,521 frames, recall 0.12). Flagged in §8. |
| Research Evidence Package | **Not found as a discrete document.** | No file by this name exists on disk. I treated the union of `PROJECT_CONTEXT.md`, the experiment JSON/CSV artifacts, and `paper/main.tex` as the de-facto evidence package, and note the absence explicitly. |
| Supporting reports / artifacts | `experiments/EXP-004_loso/exp004_metrics.json`, `experiments/EXP-005_events/*`, `paper/main.tex`, `PROJECT_CONTEXT.md` | Verified frame-level ROC/AUC, event-level metrics, and the manuscript draft. |

**Also named but absent:** `recursive-churning-lecun.md` (named as a read-only input in both
prompt files) **does not exist anywhere in the workspace**. The EXP-005 audit records the same
absence (its finding M-1). No conclusion here depends on it, but a submission package that
references inputs which are not in the repository is itself a reproducibility red flag.

## 2. Literature Review (recent work, ~last five years)

Per the revised scope, this section **summarizes** the relevant recent literature grouped by
the eight subtopics in the prompt. It is a map of the field the project sits in, not an
exhaustive project-vs-paper matrix. Every citation below was surfaced by literature search
during this review; items I could not fully verify are marked **[PARTIAL]** or
**[UNVERIFIED]** and should be checked against the primary PDF before being cited in a
manuscript.

### 2.1 Driver drowsiness detection & CV driver monitoring
- Classical facial-landmark pipelines (EAR/PERCLOS, MAR, head pose) remain the dominant
  low-cost approach; NTHU-DDD is the standard public LOSO benchmark, with reported accuracies
  clustering **85–95%+** for supervised models.
- Regulation is now a first-class driver of this field: **EU GSR (Reg. 2019/2144)** makes
  Driver Drowsiness & Attention Warning (DDAW) mandatory (new vehicle *types* from 6 Jul 2022;
  *all* new vehicles from 7 Jul 2024), detailed by **Reg. 2021/1341**. Type-approval defines a
  concrete episode-level bar: system **sensitivity average > 40%** with **lower-90%-CI > 20%**,
  auto-activation above ~70 km/h, warning at **KSS ≥ 8** (optionally at 7). Euro NCAP ties
  5-star ratings to direct camera-based monitoring.
- Field/on-road evidence: **Ahlström et al. 2025** (*J. Sleep Research*, doi:10.1111/jsr.14259,
  24 truck drivers) supports KSS ≥ 8 as the operational drowsiness threshold; **Wörle et al.
  2023** and the scoping review **PMC11344370** (accuracy 27–100% across systems) show that
  relaxing thresholds to raise recall inflates false alarms, leading to driver "disuse."

### 2.2 Multi-cue fatigue detection & fusion
- **M3ER** (Mittal et al., AAAI 2020, arXiv:1911.05659) — multiplicative modality-reliability
  gating for multimodal emotion recognition; the closest published ancestor of a
  multiplicative reliability weight.
- **TMU-Net** (2025, PMC12431429) — uncertainty-weighted gating that down-weights unreliable
  modalities before the decision.
- **Weighted-multiplication fusion** (npj Flexible Electronics 2026, s41528-026-00543-7):
  weighted multiplication (95.0%) modestly beat weighted averaging (94.1%) — evidence that
  multiplicative fusion is an established, not novel, design.
- **Face image quality** as a per-frame reliability signal: **SER-FIQ** (CVPR 2020,
  arXiv:2003.09373) and **SDD-FIQA** (CVPR 2021, arXiv:2103.05977).
- **Dempster–Shafer evidential fusion** — the classic ancestor of confidence-weighted evidence
  combination. All-weather/illumination-robust fatigue work: Sci. Reports 2024
  (s41598-024-67131-5; PMC11236172).

### 2.3 Temporal fatigue modeling
- **Confidence-driven adaptive temporal window** (Frontiers Neurorobotics 2026, art. 1857548) —
  varies the temporal integration window by confidence; conceptually adjacent to gating
  evidence before temporal accumulation, but not the same mechanism.
- LSTM-over-landmark-features temporal models (see §2.7) are the mainstream temporal approach
  on NTHU-DDD.

### 2.4 Geometry-based fatigue detection & yawn-vs-talking
- **Alioua et al. 2014** (*Int. J. Vehicular Technology*, Art. 678786, doi:10.1155/2014/678786) —
  wide-open-mouth geometry to reject talking/laughing/singing.
- **YawDD** (Abtahi et al., ACM MMSys 2014, doi:10.1145/2557642.2563678) — the three-state
  (normal / talking-singing / yawning) mouth benchmark that defines the talking-vs-yawn problem.
- Discrimination cues in the literature: **duration** (yawn mean ~4 s; neuromorphic-yawn
  arXiv:2305.02888) and **periodicity/frequency** (talking ~2–8 Hz vs. a single sub-1-Hz yawn
  pulse); variance-based Haar+Kalman mouth-state tracking. The project's "variance-based
  speech-jitter MAR filter" is a specific instance of this well-established goal.

### 2.5 Event-level evaluation
- **Fujiwara et al. 2019** (*IEEE TBME*, doc 8520803) — 12/13 pre-N1 episodes detected at
  ~1.7 FP/hour; a canonical event-level (episode-level) evaluation.
- Event-level metrics (FA/hour, per-episode recall, latency) are the **exception, not the norm**
  in the CV-drowsiness literature, which is dominated by frame-level accuracy/AUC. This makes
  the project's event-level protocol a legitimate methodological differentiator, and it aligns
  the work with the regulatory episode-level framing in §2.1.

### 2.6 Hybrid rule-based + deep-learning systems
- Landmark-rule front-ends combined with a small learned classifier are common on NTHU-DDD
  (e.g., LSTM heads over EAR/MAR/head-pose, §2.7). The project's design — geometric rules plus
  an *optional, off-by-default* micro-CNN — is a recognized hybrid pattern; the novelty is not
  the hybridization itself.

### 2.7 Edge AI driver monitoring (target: Raspberry Pi 4)
- **Chen et al. 2025** (*Sensors* 25(3):920) — LSTM on EAR/MAR/head-pose, trained on NTHU-DDD,
  **RPi 4 ≈ 10 FPS, 95.23%** — the most directly comparable edge baseline.
- **Florez et al. 2024** (*Sensors* 24(19):6261) — RPi 4, **96.3 ms/frame ≈ 10.4 FPS, 94.7%**
  **[PARTIAL — verify attribution in PDF]**.
- **Kim & Koo 2022** (arXiv:2209.15148) — Jetson Nano 94.27 ms/frame.
- **FastKAN-DDD** (*PLOS ONE*, doi:10.1371/journal.pone.0332577) — 99.94%, 0.04 ms, ~35 KB.
- **DDD TinyML** (*Sensors* 23(12):5696); **DrowSAFE** (Pi 5 + MediaPipe ≈ 30 FPS).
- **Pattern:** RPi 4 deep-CNN pipelines run ≈ 10 FPS; landmark-only pipelines on Pi 5 reach
  ≈ 30 FPS; TF-Lite is slow on Jetson without TensorRT. This is the reference set against which
  the project's **unmeasured** RPi-4 claim must eventually be judged.

### 2.8 Negative results — publishability & venues
- Negative/■falsification results have real homes: the **ICBINB** workshop series (NeurIPS
  2020–2023; ICLR 2025 "Challenges in Applied Deep Learning"), **ReScience C** (accepts failed
  replications), **IEEE Access** (scope explicitly welcomes "negative results"), and position
  pieces such as **"Embracing Negative Results in ML"** (arXiv:2406.03980, ICML 2024) and
  arXiv:2104.08878. No existing **NTHU-DDD LOSO negative-result** paper was found — which helps
  novelty but means reviewers lack a template, so framing carries unusual weight.

## 3. Independent Comparison — how novel is this?

**Classification: Incremental, bordering on Reproduction of established ideas** — with one
genuinely uncommon *methodological* element (the event-level LOSO negative-result protocol).

- The **multiplicative reliability gate** is a hand-crafted special case of an established
  family — multiplicative modality-reliability gating (M3ER, §2.2), uncertainty-weighted gating
  (TMU-Net), weighted-multiplication fusion, and FIQA-style per-frame quality. Forming
  `r = stability^0.45 · brightness^0.30 · consistency^0.25` and applying it multiplicatively
  before temporal accumulation is a *specific parameterization* nobody has published in exactly
  this form, but the underlying principle is prior art. **Narrow novelty at best.**
- The **variance-based speech-jitter MAR filter** targets the well-defined YawDD talking-vs-yawn
  problem (§2.4) using variance rather than the more common duration/periodicity cues. The
  *goal* and *insight* are established; the specific statistic is a minor variation.
  **Incremental.**
- The **optional micro-CNN (MicroEyeNet)** is a small INT8 eye-state classifier; small edge CNNs
  on NTHU-DDD are routine (§2.7). Since it is inert on this corpus, it contributes no novelty.
- The **event-level LOSO evaluation reporting an honest negative result** is the least common
  element (§2.5, §2.8). It is not a new *method*, but as a *contribution to the evidence base* it
  is genuinely uncommon and defensible.

Net: the *methods* are incremental; the *finding* (a rigorously reported negative result on a
standard benchmark) is where any real novelty lives. That reframing is essential to how this
work could ever be published, and it is not how the current manuscript is framed.

---

## 4. Scientific Contribution

**Genuinely new (supported):**
- A specific decomposed reliability index and its multiplicative-attenuation placement, together
  with **direct evidence that it does not achieve its stated purpose** on NTHU-DDD (frame-level
  FPR@matched-TPR not reduced; AUC never beats baseline; event-level gate-alone V2 gives no FP
  reduction). A well-documented "this mechanism did not work here" is a real, if modest,
  contribution.
- An event-level LOSO protocol (FA/hour, per-episode recall, latency) with two regimes
  (per-recording reset vs. per-subject concatenation) and observability gates — methodologically
  careful and uncommon in this subfield.

**Already done elsewhere (not new):**
- Multiplicative/uncertainty-weighted reliability fusion (§2.2); talking-vs-yawn discrimination
  (§2.4); landmark-rule + small-CNN hybrids and RPi-class edge deployment (§2.6–§2.7).

**Claimed but unsupported by the current evidence:**
- **"Reduces false positives at matched TPR"** — *refuted* at the frame level; not supported in
  isolation at the event level (`fatigue_fusion.py`; EXP-004; EXP-005 V2).
- **"Safety-asymmetric: SEVERE is never suppressed"** — this invariant is **not implemented**
  (`fatigue_fusion.py:197`); the gate attenuates fused evidence unconditionally. A safety claim
  that the code does not enforce cannot be made in a paper.
- **Suitability for Raspberry Pi 4** — **unmeasured.** Only Apple-M1 host latency (3.205 ms/frame)
  exists; there is no RPi-4 measurement, so no deployment claim is currently supported (§2.7 shows
  edge behavior differs sharply from host).

## 5. Publication Readiness (against mid-tier IEEE/ACM standards)

Judged against realistic **mid-tier** venue expectations (not top-tier), the work is **not yet
ready** as currently framed. A mid-tier paper can carry a negative result and a small dataset,
but it must (a) not make claims its own evidence refutes, (b) not describe an unimplemented
invariant as a feature, and (c) either measure its target platform or drop the deployment claim.
This manuscript currently fails all three.

| Criterion | Mid-tier bar | This work | Verdict |
|---|---|---|---|
| Problem significance | Real, regulated problem | DDAW is EU-mandated (§2.1) | **Meets** |
| Novelty | Some genuine increment | Incremental methods; uncommon negative-result protocol (§3) | **Marginal** |
| Methodology soundness | Protocol matches claims | Event-level protocol sound; but claims exceed evidence (§4) | **Fails as framed** |
| Datasets | Recognized benchmark | NTHU-DDD, but only **n = 4** LOSO subjects | **Weak** |
| Statistical quality | Honest, appropriately modest | Descriptive-only, discloses n=4 & single-subject FP | **Meets (for a negative result)** |
| Reproducibility | Runnable, artifacts present | Deterministic artifacts; but references missing inputs; RPi-4 unmeasured | **Partial** |
| Practical contribution | Supported deployment story | RPi-4 claim unmeasured | **Fails as framed** |

**Bottom line:** publishable *only* if reframed as an explicit negative-results / falsification
paper (venues in §11), after fixing the refuted/unimplemented claims and either measuring the
target platform or removing the deployment claim.

---

## 6. Strengths

1. **Candor.** The reporting is unusually honest: the negative result is stated plainly, and the
   EXP-005 report's own §9 pre-empts the key caveats (n=4, single-subject FP, clip-level labels,
   observability-gate failures). This is rarer than it should be and is the work's best quality.
2. **Reproducible, internally consistent artifacts.** The EXP-005 audit triangulated every
   headline number across five independent artifacts with no numerical error; the frame-level
   EXP-004 headline reproduces exactly. Determinism is by construction (seeded bootstrap, no RNG
   in the metric path).
3. **Event-level protocol.** FA/hour + per-episode recall + latency, two regimes, observability
   gates — genuinely more rigorous than the frame-level-accuracy norm (§2.5), and aligned with
   the regulatory episode-level framing.
4. **Appropriately modest statistics.** The bootstrap is explicitly a dispersion band, not a
   significance test; with n=4 this is the correct, non-overreaching choice.
5. **Engineering discipline.** Clear variant ablation (V0→V4), frozen configs, MD5-hashed event
   streams, a standing audit document.

## 7. Weaknesses (by severity)

For each: does it **prevent publication**, is it **fixable by writing**, or does it **require more
experiments**?

### Critical
- **C1 — The headline positive claim is refuted by the project's own data.** The gate does not
  reduce FPR@matched-TPR (frame level) and gives no FP reduction in isolation (event-level V2).
  A paper cannot assert a benefit its own experiments contradict. *Prevents publication as a
  positive-result paper.* **Fixable by writing** — reframe as a negative result — **not** by more
  experiments.
- **C2 — A headline safety claim is not implemented.** "SEVERE is never suppressed" is described
  as a property but is absent from the code (`fatigue_fusion.py:197`). Claiming a safety
  invariant the software does not enforce is a correctness/integrity issue. *Prevents
  publication.* **Fixable by writing** (retract the claim) *or* by code+experiment (implement and
  re-test).

### Major
- **M1 — Target platform never measured.** RPi-4 suitability is claimed; only M1 host latency
  exists. *Prevents any deployment claim.* **Requires an experiment** (measure on RPi 4) **or**
  drop the claim in writing.
- **M2 — Under-powered evidence base.** n = 4 LOSO subjects; only 2 fire any alarm; **all** false
  alarms come from **subject 005**. The central event-level contrast rests on one subject. *Weakens
  every quantitative conclusion.* **Requires more experiments** (more subjects / a second dataset)
  to strengthen; a negative result can still be reported honestly at n=4 if scoped as such.
- **M3 — Very low recall.** Per-episode recall ≈ 0.12 (≈ 86–88% of GT episodes never alarmed).
  Even as a negative result this needs explicit framing so it is not mistaken for a working
  detector. **Fixable by writing** (scope/framing); **strengthened by experiments**.

### Moderate
- **Mo1 — Inert CNN arm.** MicroEyeNet never changes a decision (`any_cnn_override = 0`);
  reporting it as a contribution overstates it. **Fixable by writing** (present as an
  ablation-only null result).
- **Mo2 — Stale, contradictory Root-Cause doc.** `EXP005_ROOT_CAUSE_ANALYSIS.md` describes a
  superseded run (recall 0.0, 16/16 GT missed, 10,800 frames) that contradicts the committed
  EXP-005; it is not marked superseded. **Fixable by writing** (mark superseded or delete).
- **Mo3 — Missing referenced inputs.** `recursive-churning-lecun.md` and a discrete "Research
  Evidence Package" are named as inputs but do not exist in the repo — a reproducibility red flag
  for a submission package. **Fixable by writing** (remove/replace the references).

### Minor
- **Mi1 — Subject-006 AUC magnitude discrepancy.** JSON gives V0 006 AUC = 0.3727 while the audit
  cites ≈ 0.3047; both are below chance, but the numbers disagree. **Fixable by writing**
  (reconcile the figure).
- **Mi2 — EXP-005 audit prose defects** (C-1 k mislabeled "duration floor" vs. minimum-overlap
  tolerance; C-2 figure-caption regime mismatch). Already logged; **fixable by writing**.
- **Mi3 — Seed not echoed in the run log** (R-1). **Fixable** by logging the seed.

## 8. Additional Experiments (only those whose absence makes the paper scientifically incomplete)

I recommend experiments **only** where their absence leaves a claim scientifically unsupported.
Pure "would-be-nice" additions are omitted.

**Mandatory (a submission is scientifically incomplete without these):**

1. **Broaden the evidence base beyond n = 4 / one FP subject.** *Why:* every event-level
   conclusion currently rests on 2 firing subjects with all false alarms from subject 005 (M2).
   *What:* add subjects (ideally a full NTHU-DDD LOSO, and/or a second dataset such as
   YawDD/DROZY) so the negative result is not a single-subject artifact. *Mandatory?* Yes for any
   generalization claim. *Submission reasonable without it?* Only if the paper explicitly and
   narrowly scopes itself to "n=4, exploratory" — which most venues will find thin.

2. **Resolve the refuted/unimplemented core claims (C1, C2) — analysis + code, not just prose.**
   *Why:* the paper cannot claim FP reduction (refuted) or a "SEVERE-never-suppressed" invariant
   (not in code). *What:* either (a) drop both claims and reframe as a negative result, or (b)
   implement the invariant and re-run EXP-004/005 to test it honestly. *Mandatory?* Yes — this is
   integrity, not polish. *Reasonable without?* No, in its current claim form.

**Optional (strengthen, not required):**

3. **RPi-4 latency/throughput measurement.** *Why:* to make *any* edge-deployment claim (M1);
   comparable baselines run ≈ 10 FPS on RPi 4 (§2.7). *Mandatory?* Only if the paper keeps a
   deployment claim — otherwise drop the claim and this becomes optional. *Reasonable without?*
   Yes, if the deployment framing is removed.

4. **Gate redesign / sensitivity analysis** (the exponents 0.45/0.30/0.25, threshold choices).
   *Why:* to show whether *any* gate parameterization helps, strengthening the negative result
   from "this one didn't" toward "this family doesn't here." *Optional.* *Reasonable without?* Yes.

---

## 9. Conference Suitability

The realistic path is a **negative-results / applied venue**, not a flagship. Prospects assume
the C1/C2 fixes and honest reframing are done first.

| Venue | Fit | Strengths for this work | Weaknesses / risk | Prospects |
|---|---|---|---|---|
| **IEEE Access** (journal, OA) | **Best fit** | Scope explicitly welcomes negative results; fast OA; tolerates applied + modest-n | Reviewer variance; must be framed as a contribution, not a failure | Moderate **after** reframing + broader n |
| **ICBINB** workshop (NeurIPS/ICLR) | Strong conceptual fit | Purpose-built for "it didn't work and here's why"; §2.8 | Workshop (non-archival at some editions); smaller audience | Moderate–good if framed as falsification |
| **Sensors / IEEE Sensors Journal** (MDPI/IEEE) | Good | EAR/landmark + edge-friendly community | Expects a working system or measured edge results (needs §8.3) | Moderate with RPi-4 data |
| **IET Intelligent Transport Systems** (OA, Q3) | Reasonable | Applied ITS, DDAW-relevant | Q3; still expects a supported contribution | Moderate |
| **IEEE ITSC 2026** (Naples) / **IEEE IV 2026** (Detroit) | Possible | Relevant tracks (ADAS, human factors) | Competitive; a primarily negative result is a hard sell at conference review | Low–moderate |
| **IEEE T-ITS** (flagship journal) | **Poor** | — | Flagship expects a clear positive advance | **Not recommended** now |

**Recommendation:** target **IEEE Access** or **ICBINB**, explicitly framed as a rigorous
negative result on NTHU-DDD LOSO, after §8.1–§8.2.

## 10. Reviewer #2 Simulation

**Summary.** The paper proposes a decomposed multiplicative signal-reliability gate and a
variance-based speech-jitter MAR filter for facial-geometry driver-drowsiness detection, with an
optional micro-CNN eye-state arm, evaluated on NTHU-DDD under LOSO with both frame-level
(ROC/AUC) and event-level (FA/hour, per-episode recall, latency) protocols. The central finding
is a **negative result**: the reliability gate does not reduce the false-positive rate at matched
TPR, and in isolation gives no event-level benefit.

**Strengths.** (1) Unusual honesty; the negative result and its limitations are reported
plainly. (2) A careful, reproducible event-level protocol that exceeds the frame-level-accuracy
norm of the subfield. (3) Deterministic, internally consistent artifacts verified by a standing
audit. (4) Appropriately modest, non-overreaching statistics.

**Weaknesses.** (1) The abstract/claims assert a false-positive-reduction benefit that the
paper's own data refute. (2) A "SEVERE is never suppressed" safety invariant is described but not
implemented in the released code. (3) The target platform (RPi 4) is never measured — the
deployment claim is unsupported. (4) n = 4 subjects, only 2 firing, all false alarms from a
single subject; per-episode recall ≈ 0.12. (5) The optional CNN is inert (zero decision changes)
yet presented as a contribution. (6) Referenced inputs (`recursive-churning-lecun.md`, a
"Research Evidence Package") are absent from the repository, and a stale root-cause document
contradicts the committed results.

**Questions.**
1. Given EXP-004/005, on what evidence does the abstract claim the gate reduces false positives?
2. Where in the code is "SEVERE is never suppressed" enforced? Please point to the line, or
   retract the claim.
3. Can you provide any RPi-4 measurement (latency, FPS, thermals) to support the deployment
   claim? If not, will you remove it?
4. With all false alarms from subject 005, how do you argue the event-level results generalize?
5. Why include MicroEyeNet as a contribution if `any_cnn_override = 0` on the entire corpus?
6. Which run is authoritative — the committed EXP-005 (recall 0.12) or the root-cause doc
   (recall 0.0)? Please reconcile or mark one superseded.

**Recommendation:** **Reject** as submitted (positive-result framing). Would consider a
**major-revision resubmission** *only* if reframed as an honest negative result, with C1/C2 fixed
and the evidence base broadened (§8).

---

## 11. Final Decision

**Weak Reject.**

Reasoning: the work is not scientifically empty — it is a careful, honest, reproducible negative
result with a genuinely uncommon event-level protocol, which is why this is *Weak* Reject and not
a flat Reject. But in its **current form and framing** it cannot be accepted: the headline
positive claim is refuted by its own data (C1), a headline safety invariant is unimplemented
(C2), the target-platform claim is unmeasured (M1), and the evidence base is under-powered with
all false alarms from one subject (M2). As a positive-result system paper it is a clean
**Reject**. Reframed as a negative-results paper with C1/C2 fixed and a broader evidence base, it
could reach **Borderline** at a suitable venue (§9) — but that reframing and those fixes have not
yet been done, so the honest decision today is **Weak Reject**.

---

## 12. Final Recommendation

1. **Is this work publishable in a mid-tier peer-reviewed conference?** Not as currently written
   and framed. It is publishable *only* after reframing as a negative-results paper and fixing
   the refuted/unimplemented claims (and preferably broadening the evidence base).

2. **If not, why not?** It claims a benefit its own experiments refute (gate does not reduce
   FP@matched-TPR), describes a safety invariant that is not implemented, and claims RPi-4
   suitability that is never measured — on top of an under-powered evidence base (n=4, all FP from
   subject 005) and very low recall (≈0.12).

3. **Biggest scientific weakness?** The **contradiction between the central claim and the
   evidence** (C1): the paper's headline mechanism is presented as reducing false positives, but
   the project's own frame-level and event-level results show it does not. Everything else is
   secondary to this.

4. **Are additional experiments required?** Yes — two mandatory (§8.1 broaden beyond n=4/one FP
   subject; §8.2 resolve C1/C2 in code+analysis, not just prose) and two optional (§8.3 RPi-4
   measurement if any deployment claim is kept; §8.4 gate-parameter sensitivity).

5. **If yes, which and why?** §8.1 (more subjects / second dataset) so the negative result is not
   a single-subject artifact; §8.2 (implement-and-test or retract the gate's FP-reduction and
   "SEVERE-never-suppressed" claims) because a paper cannot assert what its code and data do not
   support. §8.3 is required *only* if a deployment claim is retained.

6. **If no, why sufficient?** Not applicable — additional evidence *is* required for any
   generalization or deployment claim. (A strictly scoped "n=4 exploratory negative result" could
   be written on current data alone, but it would be thin and still needs C1/C2 fixed.)

7. **As your PhD advisor, would I recommend Submit now / Revise manuscript only / Perform
   additional experiments before submission?** **Perform additional experiments before
   submission.** Concretely: first fix C1/C2 (retract or implement-and-test), then broaden the
   evidence base (§8.1) and — if you want a deployment claim — measure on RPi 4 (§8.3). With the
   claims corrected and the evidence broadened, reframe the manuscript as a negative-results
   contribution and target IEEE Access or ICBINB (§9). Do **not** submit the current
   positive-result manuscript.

---

*End of assessment. Deliverable: `reports/PUBLICATION_READINESS_ASSESSMENT.md`. Every conclusion
above is tied to a verified project artifact or to the literature summarized in §2; items that
could not be fully verified are marked [PARTIAL]/[UNVERIFIED], and absent inputs are flagged
rather than assumed.*






