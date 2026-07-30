# EXP-005 Root Cause Analysis — Event-Level Alarm Evaluation

**Scope:** Root-cause investigation only. No implementation was modified, no
threshold or parameter was tuned, no methodology was redesigned. The code was
treated as frozen. Temporary, logging-only instrumentation was used to capture
the numeric fatigue trajectory (reading `FrameResult.fatigue_score`, which is
already exposed at [frame_processor.py:232](../src/frame_processor.py#L232));
that instrumentation lives in a throwaway script outside the repository and
changed nothing in the pipeline.

**Observed behavior under investigation:** In EXP-005, the PRIMARY regime
(fresh per-recording temporal/state reset) produced essentially **zero** alarm
events (recall 0.0, F1 0.0, 16/16 positive GT events missed for every variant).
The SECONDARY regime (one processor per subject, no reset at recording
boundaries) produced alarm events *and* false positives.

---

## 1. Executive Summary — Exact Root Cause

The PRIMARY regime produces no alarm events because, in NTHU-DDD, **eye closure
(EAR) is effectively the only active fatigue cue**, and a single active cue can
never drive the fused fatigue score above the `moderate_threshold = 0.50`
required to raise even the lowest alarming severity (MODERATE). The fusion
engine computes `raw_score = ear_conf·0.45 + mar_conf·0.25 + pose_conf·0.30`;
with only EAR active the raw score is capped at `0.45 × ear_conf ≈ 0.36–0.45`,
and the multi-cue agreement multiplier stays at `1.0×` because fewer than two
cues are active. That ceiling (≤ 0.45) is *structurally below* 0.50. On top of
that ceiling, the fused score is passed through a slow asymmetric EMA
(`acc_rate = 0.08` rising, `decay_rate = 0.04` falling) that, starting from the
PRIMARY per-recording reset of `_accumulated_score = 0.0`, cannot even reach its
own instantaneous ceiling within a single 400-frame (~13.3 s) recording.
Instrumentation confirms this directly: on the most drowsy recording tested
(006/glasses/sleepyCombination, 272 drowsy GT frames) the accumulated
fatigue_score **peaked at 0.2156** and never once reached 0.50. The SECONDARY
regime fires alarms only because its per-subject processor **carries the fusion
accumulator (and the dwell-gate transition clock) across recording boundaries**,
so the accumulator is warm-started from earlier recordings — which is also
exactly why SECONDARY generates *false* alarms on non-drowsy clips. This is an
expected consequence of the algorithm design + configuration + the (correct)
per-recording reset interacting with NTHU-DDD's eye-closure-only paradigm — it
is **not an implementation bug**.

---

## 2. Evidence

### 2.1 Aggregate stream counts (artifact: `event_streams/V3_full.csv`, 21,600 rows)

| Regime | Frames | Σ should_alarm | Σ debounced_alarm | Σ alarm_level>0 |
|---|---|---|---|---|
| PRIMARY   | 10,800 | **0**   | **0**   | **0** |
| SECONDARY | 10,800 | 104 | 331 | 104 |

Identical recordings, identical labels, identical frame inputs — the *only*
difference between the two regimes is processor-state carryover across recording
boundaries. PRIMARY = 0 alarms; SECONDARY = 104. This isolates state carryover
as the sole causal variable.

### 2.2 Numeric fatigue trajectory (temporary instrumentation, PRIMARY protocol)

Fresh `FrameProcessor` per recording, `ts = local_idx / 30`, first 400 frames,
reading `result.fatigue_score` per frame. `moderate_threshold = 0.50`.

| Recording (PRIMARY) | drowsy GT frames | max score | mean | final | frames ≥ 0.50 | should_alarm |
|---|---|---|---|---|---|---|
| 006/glasses/sleepyCombination | 272 | **0.2156** | 0.035 | 0.000 | 0 | 0 |
| 002/glasses/sleepyCombination | 86  | **0.0000** | 0.000 | 0.000 | 0 | 0 |
| 001/noglasses/sleepyCombination | 70 | **0.1048** | 0.010 | 0.000 | 0 | 0 |

Per-frame checkpoints for 006/glasses/sleepyCombination: frame 50 = 0.0,
100 = 0.0, 200 = 0.2123 (the peak region), 300 = 0.0187, 399 = 0.0003. The
score rises slowly toward ~0.21 mid-recording, then **decays back to zero** —
it is nowhere near 0.50, and does not even hold the SLIGHT band (≥0.25). The
002 recording reads a flat 0.0 because MediaPipe returns no usable face on that
prefix (no active cue at all), which independently guarantees no alarm.

### 2.3 State/alarm transitions and timestamps (SECONDARY, artifact CSV)

SECONDARY alarms concentrate in subject 002 and fire *early* inside non-drowsy
clips — impossible for a cold-started EMA:

| Recording (SECONDARY) | first alarm local idx | ≈ time | label @ first alarm | n_alarm frames |
|---|---|---|---|---|
| 002/glasses/nonsleepyCombination   | 45  | ~1.5 s | 0 (not drowsy) | 37 |
| 002/noglasses/nonsleepyCombination | 47  | ~1.6 s | 0 (not drowsy) | 32 |
| 002/glasses/slowBlinkWithNodding   | 325 | ~10.8 s | — | 35 |

Subject 001's eight recordings are processed *before* 002's in the frozen sort
order; the fusion accumulator and `state_manager._last_transition_time` built up
during 001 persist into 002's recordings because the SECONDARY processor is
never reset. An alarm at local frame 45 (~1.5 s) on a **nonsleepy** clip is the
signature of a warm-started accumulator, not of evidence within that clip.

### 2.4 Suppression / accumulated-fatigue / event-construction mechanics

- **Accumulated fatigue:** PRIMARY starts every recording at
  `_accumulated_score = 0.0` (fusion `reset()`), confirmed by the 0.0 readings
  at frame 50 across all three recordings above.
- **Severity gate:** MODERATE (the lowest severity with `should_alarm = True`,
  `alarm_level = 1`) requires `score ≥ 0.50 and active_cues ≥ 1`. SLIGHT
  (`score ≥ 0.25`) yields `should_alarm = False`. The measured PRIMARY ceiling
  (0.2156) sits below *both* thresholds.
- **Event construction:** an alarm event is a maximal contiguous run of
  `should_alarm == True` within a recording, re-debounced on the video clock
  (`min_alarm_duration_s = 3.0`, `cooldown_period_s = 5.0`). With zero
  `should_alarm` frames in PRIMARY, zero events are constructed → tp = 0,
  fn = 16, recall = 0.0 for every variant (`per_variant_event_metrics.csv`).
- **Dwell gate (SECONDARY only):** `ts` resets to `local_idx/fps = 0` at each
  recording boundary, but `StateManager._last_transition_time` carries over
  (~13 s from the prior recording). `time_since_transition = 0 − 13.3 < 0 <
  MIN_DWELL_TIME (2.0)`, so the dwell gate *pins* the prior severity into the
  new recording — a second carryover channel stacked on the fusion-accumulator
  carryover.

### 2.5 Ablation non-observability (artifact: `exp005_event_metrics.json`)

All five variant streams (V0–V4) are byte-identical
(MD5 `0b7c087a586ec5b6480805010c935ac7`). Observability gates G1 (CNN),
G2 (reliability gate), G3 (speech filter) all **FAIL** with `n_diff_frames = 0`
over 10,800 compared frames. The ablation arms cannot move a single frame
because EAR is the sole active cue: MAR is suppressed by the speech filter and
pose confidence is ~0, so the toggled components never touch the decision path.
This is corroborating evidence for the single-active-cue root cause.

---

## 3. Hypotheses Tested

| # | Hypothesis | Investigated? | Evidence | Verdict |
|---|---|---|---|---|
| H1 | Frames were never loaded / recordings empty (I/O bug) | Yes | 27 recordings × 400 frames = 10,800 PRIMARY rows present; instrumentation loaded and processed real pixels (max_score 0.2156, non-zero mid-clip) | **Rejected** |
| H2 | The 400-frame cap head-slices only alert frames | Yes | Cap is applied per-recording prefix `v[:400]`; instrumented recordings contain 70–272 drowsy GT frames in their 400-frame prefix | **Rejected** |
| H3 | Ground-truth labels are wrong / all zero | Yes | `nthu_ground_truth.py` derives labels from the official NTHU directory + filename grammar; drowsy recordings show hundreds of label-1 frames | **Rejected** |
| H4 | Alarm exposure fields are mis-wired in FrameResult | Yes | [frame_processor.py:236-241](../src/frame_processor.py#L236-L241) copies `should_alarm`/`alarm_level` verbatim from the StateManager `state`; SECONDARY proves the path can emit alarms | **Rejected** |
| H5 | Debounce (3 s / 5 s) erased short PRIMARY events | Yes | `should_alarm` sum is 0 *before* debounce; there is nothing to debounce away | **Rejected** |
| H6 | Reliability gate / speech filter / CNN suppressed the alarms | Yes | Ablation V0–V4 byte-identical; G1/G2/G3 all FAIL (`n_diff_frames = 0`); toggling those components changes nothing | **Rejected** |
| H7 | Single active cue (EAR) caps raw_score below moderate_threshold | Yes | `raw_score ≤ 0.45·ear_conf ≈ 0.45 < 0.50`; multiplier stays 1.0× with one cue; instrumented max = 0.2156 | **Accepted (primary mechanism)** |
| H8 | Slow cold-start EMA cannot reach the ceiling within 400 frames | Yes | PRIMARY resets `_accumulated_score = 0.0` per recording; `acc_rate = 0.08`; instrumented score peaks ~0.21 mid-clip then decays to ~0 | **Accepted (compounding mechanism)** |
| H9 | SECONDARY alarms come from cross-recording state carryover | Yes | SECONDARY fires at local frame 45 (~1.5 s) on a *nonsleepy* clip after subject 001's recordings warm the accumulator; PRIMARY (reset) never fires | **Accepted (explains PRIMARY vs SECONDARY)** |
| H10 | Dwell gate pins prior severity across the boundary (SECONDARY) | Yes | `ts` resets to 0 while `_last_transition_time` carries (~13 s) → negative `time_since_transition` < `MIN_DWELL_TIME` | **Accepted (secondary carryover channel)** |

The accepted set (H7–H10) is mutually consistent and jointly sufficient: H7+H8
explain why PRIMARY is silent; H9+H10 explain why SECONDARY is not.

---

## 4. PRIMARY vs SECONDARY

The two regimes run the **same recordings, labels, and geometry** through the
**same pipeline code**. The single difference is state lifetime:

- **PRIMARY** ([exp005_event_report.py:206-220](../evaluation/exp005_event_report.py#L206-L220)):
  a **fresh** `FrameProcessor` is constructed **per recording**
  `(subject, glasses, condition)`, so `_accumulated_score`, the temporal
  history, and `_last_transition_time` all reset to their initial values at
  every boundary. This matches the FrameProcessor contract
  ([frame_processor.py:83-87](../src/frame_processor.py#L83-L87)) and is the
  scientifically clean regime for event-level evaluation.
- **SECONDARY** ([exp005_event_report.py:222-239](../evaluation/exp005_event_report.py#L222-L239)):
  **one** `FrameProcessor` per subject, explicitly **not reset** at boundaries
  (line 226 comment: "the processor is NOT reset at boundaries — EXP-004
  regime"). Only the per-recording *frame counter* (`local_counters`) resets;
  the fusion accumulator and state clock persist.

**Proof that carryover is the cause (not a difference in inputs):**

1. Within a subject, SECONDARY inherits a non-zero accumulator from the previous
   recording. The frozen sort processes subject 001's eight recordings before
   002's, so by the time 002's nonsleepy clip starts, the accumulator is already
   elevated.
2. SECONDARY's first alarm on `002/glasses/nonsleepyCombination` is at local
   frame **45 (~1.5 s)**, with GT label **0**. A fresh EMA at `acc_rate = 0.08`
   from 0.0 cannot reach 0.50 in 45 frames even at the instantaneous ceiling —
   so the score at frame 45 *must* have been inherited. PRIMARY, which resets,
   emits nothing on the identical frames.
3. The dwell-gate carryover (H10) compounds this: with `ts` reset to 0 and
   `_last_transition_time` still ~13 s, `time_since_transition` is negative, so
   the gate holds whatever severity was active at the end of the prior
   recording.

Therefore the PRIMARY silence and the SECONDARY alarms are **the same
phenomenon viewed two ways**: the algorithm only ever alarms when the
accumulator is warm, and only SECONDARY keeps it warm — at the cost of leaking
one recording's evidence into the next (hence SECONDARY's false positives on
nonsleepy clips). SECONDARY's alarms are an artifact of state bleed, not
evidence that PRIMARY is broken.

---

## 5. Dataset Effects — Attribution

The behavior is attributable to a combination of **dataset characteristics**,
**algorithm design**, **configuration**, and the **temporal-reset regime**
interacting — and explicitly **not** to an implementation bug.

- **NTHU-DDD characteristics (primary contributor):** the drowsiness signal in
  these clips is overwhelmingly *eye closure*. MAR-based yawning is suppressed
  by the speech-jitter filter, and head-pose confidence is ~0 on this footage.
  So the fusion engine sees a single active cue almost everywhere. NTHU's
  clip-condition labelling (an open-eyed frame in a drowsy clip is still
  labelled 1) further weakens per-frame EAR evidence. The dataset simply does
  not supply the multi-cue agreement the fusion design rewards.
- **Algorithm design (contributor, working as specified):** the fused-score
  formula weights EAR at 0.45 and grants the score-boosting agreement
  multiplier only when ≥ 2 cues are active. A lone EAR cue is capped at ≤ 0.45,
  below the 0.50 MODERATE gate — by design, to demand corroboration before
  alarming. The slow asymmetric EMA is likewise a deliberate stability choice.
- **Configuration (contributor):** `moderate_threshold = 0.50`,
  `acc_rate = 0.08`, `max_frames_per_subject = 400` (~13.3 s). A 400-frame
  window is too short for a cold-start EMA to climb toward its own ceiling even
  if that ceiling were reachable.
- **Temporal-reset regime (contributor, and the PRIMARY/SECONDARY switch):** the
  PRIMARY per-recording reset is *correct* for isolating event-level behavior;
  it is precisely what exposes that a single 400-frame recording carries
  insufficient EAR-only evidence to alarm. SECONDARY's non-reset masks this by
  carrying state.
- **Implementation bug:** ruled out. Every wiring hypothesis (H1, H4, H5) was
  tested and rejected; SECONDARY demonstrates the alarm path is fully
  functional on the same code and data.
- **Other:** none identified.

---

## 6. Scientific Impact

- **Expected behavior — yes.** Given the frozen fusion design (single-cue cap
  below the MODERATE gate), the slow cold-start EMA, the 400-frame window, and
  NTHU's eye-closure-only paradigm, a PRIMARY event-level recall of ~0 is the
  *predicted* outcome, now confirmed numerically (max score 0.2156 ≪ 0.50).
- **Known limitation — yes.** EXP-005's PRIMARY event-level recall is
  structurally zero on this dataset. This is a genuine, reportable limitation of
  the *system-on-this-benchmark*, not a coding defect: the multi-cue fusion
  cannot activate on a single-cue corpus within short windows. It should be
  stated plainly in the paper alongside the existing frame-level results.
- **Implementation bug — no.** No fix is warranted or (per the frozen-code
  constraint) permitted.
- **Does it invalidate EXP-005? — no, with a caveat.** EXP-005 remains a valid,
  honestly-reported *negative/limitation* result under PRIMARY. What it does
  invalidate is any attempt to use the **SECONDARY** stream as a performance
  result: SECONDARY's alarms are contaminated by cross-recording state carryover
  (false positives on nonsleepy clips at ~1.5 s), so SECONDARY must be treated
  as a diagnostic regime only, never as the reported metric.

---

## 7. Recommendation

**Proceed to Step 6 unchanged.**

The investigation found **no implementation bug**. The PRIMARY regime behaves
exactly as the frozen algorithm + configuration dictate on NTHU-DDD: a single
active cue (EAR) is capped below the MODERATE alarm gate, and a cold-start slow
EMA cannot climb within a 400-frame recording. The correct action is to carry
this forward as a **reported limitation** of event-level alarming on this
benchmark — with PRIMARY as the scientifically valid regime and SECONDARY
flagged as a contaminated diagnostic that must not be used as a performance
number. No code change, threshold tune, or redesign is justified by the
evidence, and none is made.

*(Should the research goal later require non-zero event-level recall on
NTHU-DDD, that is a methodology decision for a future experiment — e.g. an
EAR-primary alarm path, a longer evaluation window, or a multi-dataset corpus
with genuine yawning/pose signal — not a bug fix to EXP-005. It is out of scope
for this frozen investigation and is noted only for planning.)*

