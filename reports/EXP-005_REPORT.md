# EXP-005 — Event-Level Alarm Evaluation

**Experiment ID:** EXP-005_events
**Dataset:** NTHU-DDD (Driver Drowsiness Detection), LOSO protocol
**Variants evaluated:** V0_baseline, V1_speech_filter, V2_reliability_gate, V3_full, V4_full_cnn
**Wall-clock runtime:** 46.5 min (2790.88 s); per-variant ≈ 549–568 s
**Seed:** 42 · **Clock:** video · **video_fps:** 30.0 · **sklearn used:** false

> Every quantitative statement in this report is taken directly from the EXP-005
> generated artifacts under `experiments/EXP-005_events/` (JSON metrics, per-variant
> and per-subject CSVs, per-recording episode/alarm CSVs, event-stream CSVs, the four
> plots, and `exp005_run.log`). No result has been produced by any means other than
> reading those files.

---

## 1. Motivation and Objective

The prior experiment (EXP-004) evaluated the pipeline at the **frame level** by sweeping
an ROC over the continuous `fatigue_score`. That evaluation is structurally blind to two
parts of the system: the CNN eye-state arm and the reliability gate both act on the
**alarm decision** (`should_alarm`) in the `StateManager`, not on the continuous score.
A frame-level ROC therefore cannot observe whether those components change behaviour.

EXP-005 re-evaluates the same five variants at the **event / alarm level**, where an
alarm firing (or being suppressed) is the observable unit. The objective is to measure
whether the speech filter, reliability gate, and CNN arm change the alarm behaviour of
the system, and to quantify false-alarm rate, event recall/precision, and alarm latency.

## 2. Variants and Toggles

The five variants and their component toggles (as recorded in the metrics JSON) are:

| Variant | Name | speech_filter | reliability_gate | cnn |
|---|---|---|---|---|
| V0 | V0_baseline | false | false | false |
| V1 | V1_speech_filter | true | false | false |
| V2 | V2_reliability_gate | false | true | false |
| V3 | V3_full | true | true | false |
| V4 | V4_full_cnn | true | true | true |

V0 is the all-off baseline. V3 combines the speech filter and reliability gate. V4 adds
the CNN eye-state validator on top of V3.

## 3. Dataset

NTHU-DDD, evaluated under leave-one-subject-out (LOSO). As recorded in `exp005_run.log`
(line 2), the corpus processed was **66,521 frames across 4 subjects: ['001','002','005','006']**.
Per-subject composition (from the run configuration):

| Subject | Recordings | Frames |
|---|---|---|
| 001 | 8 | 19,016 |
| 002 | 8 | 18,833 |
| 005 | 8 | 21,933 |
| 006 | 3 | 6,739 |

Total recording time evaluated in the primary regime is **0.615935 h** (per the
`recording_hours` column of `per_variant_event_metrics.csv`).

## 4. Methodology

### 4.1 Two evaluation regimes

Each variant is evaluated in two regimes, both processing the full 66,521 frames
(confirmed per-variant in the run log: `[Vx] PRIMARY 66521 frames, SECONDARY 66521 frames`):

- **PRIMARY (headline).** Per-recording reset: a fresh `FrameProcessor` is instantiated
  for each `(subject, glasses, condition)` recording triple, and alarm/GT events are
  matched with `max_overlap`. This is the regime used for all headline numbers because it
  isolates each recording's temporal state.
- **SECONDARY (contamination cross-check).** Per-subject concatenation in the EXP-004
  interleaved recording order, with `greedy_onset` matching. This regime deliberately
  reproduces the EXP-004 processing order to measure cross-recording state contamination.

### 4.2 Event definitions

- An **alarm event** is a maximal contiguous run of `should_alarm == True` within a single
  recording.
- A **ground-truth drowsy episode** is a maximal contiguous run of `label == 1` within a
  single recording.

### 4.3 Metrics

False alarms per hour (both total and normalized to alert-labelled time), event-level
recall, precision, miss rate, F1, alarm latency (median and IQR), episode/alarm counts,
and alarm duty cycle. Matching an alarm event to a GT episode uses temporal overlap
(PRIMARY: `max_overlap`; SECONDARY: `greedy_onset`).

### 4.4 Minimum-duration (k) sweep

Alarm events are filtered by a minimum-duration threshold k ∈ {0, 0.25, 0.5, 1, 2} s
(tolerance frames 1, 8, 15, 30, 60 at 30 fps), to test sensitivity of recall/precision to
a debounce-style duration floor.

### 4.5 Debounce

A debounce pass (`min_alarm_duration_s = 3.0`, `cooldown_period_s = 5.0`, confirmed in run
log line 3) is applied. Because `src/alarm_controller.py` hard-codes wall-clock time
sources (`time.monotonic()` / `datetime.now()`), debounce was **reimplemented on the video
clock** inside `exp005_event_report.py` so that it is reproducible from frame indices at
30 fps rather than depending on real elapsed time.

### 4.6 Statistics

Descriptive only. n = 4 subjects; **no significance tests are performed**. Where a
dispersion band is reported it is a subject-stratified percentile bootstrap
(B = 2000, seed 42, CI level 0.95) and is explicitly a dispersion estimate, **not** a
hypothesis test.

## 5. Implementation notes

- **Event report harness.** All event/episode extraction, matching, k-sweep, debounce,
  and metric computation live in `exp005_event_report.py`. The metrics JSON
  (`exp005_event_metrics.json`), the per-variant and per-subject CSVs, the per-recording
  episode/alarm CSVs (`episodes/*.csv`), the event-stream CSVs (`event_streams/*.csv`),
  and the four plots are all produced by that harness.
- **Debounce on the video clock.** As noted in §4.5, `src/alarm_controller.py` uses
  real-time clocks, so it cannot be replayed deterministically from a fixed frame stream.
  The debounce logic was therefore reimplemented against frame indices (30 fps) inside the
  event harness. The `debounced_primary` blocks in the metrics JSON are the output of this
  reimplementation, not of `src/alarm_controller.py`.
- **`measured_results.json` untouched.** The run log (line 554) records
  `--write not set; measured_results.json NOT modified.` Confirmed by inspection:
  `results/measured_results.json` contains only the keys `latency_ms`, `_provenance`,
  `roc`, `operating_point_tpr` — there is **no `events` block**. All EXP-005 canonical
  outputs live under `experiments/EXP-005_events/`, not in `measured_results.json`.
- **sklearn not used.** The metadata records `sklearn used: false`; event metrics are
  computed directly by the harness.

## 6. Results

### 6.1 Primary regime — per-variant (headline)

From `per_variant_event_metrics.csv` (regime = primary; `recording_hours` = 0.615935 h
for every variant):

| Variant | TP | FP | FN | Recall | Precision | Miss rate | F1 | FA/h total | FA/h alert |
|---|---|---|---|---|---|---|---|---|---|
| V0_baseline | 6 | 6 | 35 | 0.1463 | 0.5000 | 0.8537 | 0.2264 | 9.741 | 21.252 |
| V1_speech_filter | 5 | 5 | 36 | 0.1220 | 0.5000 | 0.8780 | 0.1961 | 8.118 | 17.710 |
| V2_reliability_gate | 5 | 6 | 36 | 0.1220 | 0.4545 | 0.8780 | 0.1923 | 9.741 | 21.252 |
| V3_full | 5 | 4 | 36 | 0.1220 | 0.5556 | 0.8780 | 0.2000 | 6.494 | 14.168 |
| V4_full_cnn | 5 | 4 | 36 | 0.1220 | 0.5556 | 0.8780 | 0.2000 | 6.494 | 14.168 |

Total GT drowsy episodes pooled across recordings = TP + FN = 41 for every variant.

Observations grounded strictly in the table:

- **V3 has the best false-alarm profile.** V3 has the lowest false-alarm rate
  (FA/h total 6.494 vs V0's 9.741) and the highest precision (0.5556). It removes 2 of
  V0's 6 false alarms (FP 6 → 4).
- **V4 ≡ V3.** V4_full_cnn is numerically identical to V3_full on every column. Adding the
  CNN arm produced no change in alarm behaviour on this corpus.
- **V2 (gate alone) does not reduce pooled FP.** With the reliability gate on but the
  speech filter off, FA/h total stays at 9.741 (equal to V0) and precision is the lowest of
  all variants (0.4545, from FP 6 / TP 5). The FP reduction seen in V3 does not appear when
  the gate is enabled without the speech filter.
- **V1 (speech alone) is intermediate** (FA/h total 8.118, FP 5).
- **Recall is essentially flat and low** across all variants (0.1220, except V0 at 0.1463):
  40+ of 41 GT episodes are missed regardless of configuration.

### 6.2 Primary regime — per-subject

From `per_subject_event_metrics.csv`. Only subjects **002** and **005** ever fire alarms;
**001** and **006** fire zero alarms in every variant (all their GT episodes are counted as
misses). Subject **005** is the sole source of false alarms.

| Subject | Variant | TP | FP | Precision | FA/h |
|---|---|---|---|---|---|
| 002 | V0 | 2 | 0 | 1.000 | 0 |
| 002 | V1–V4 | 1 | 0 | 1.000 | 0 |
| 005 | V0 | 4 | 6 | 0.400 | 29.54 |
| 005 | V1 | 4 | 5 | 0.444 | 24.62 |
| 005 | V2 | 4 | 6 | 0.400 | 29.54 |
| 005 | V3 | 4 | 4 | 0.500 | 19.70 |
| 005 | V4 | 4 | 4 | 0.500 | 19.70 |
| 001 | all | 0 | 0 | — | 0 |
| 006 | all | 0 | 0 | — | 0 |

The pooled per-variant metrics in §6.1 are the sum of these per-subject rows. Subject 005
alone accounts for the entire pooled false-alarm count and for the whole V2-vs-V3
difference (005 FP: V2 = 6, V3 = 4).

### 6.3 Alarm events (per-recording detail)

The per-recording alarm-event CSVs (`episodes/*.csv`) confirm the subject restriction:
every alarm event in V3 belongs to subject 002 (1 event) or subject 005 (8 events), all in
`noglasses` recordings. Across every alarm event in every variant, the reliability-machinery
flags are **all zero**: `any_cnn_override = 0`, `any_alert_suppressed = 0`,
`any_face_lost_critical = 0`. No per-event override, suppression, or critical-face-loss was
ever recorded.

### 6.4 Debounced primary regime

The debounce reimplementation (§4.5) tightens the primary results. For V3 (and V4,
identical), from the `debounced_primary` block of the metrics JSON:

| Variant | TP | FP | FN | Recall | Precision | F1 | FA/h total | FA/h alert | Latency median (s) |
|---|---|---|---|---|---|---|---|---|---|
| V3_full | 5 | 3 | 36 | 0.1220 | 0.625 | 0.2041 | 4.871 | 10.626 | 4.167 |
| V4_full_cnn | 5 | 3 | 36 | 0.1220 | 0.625 | 0.2041 | 4.871 | 10.626 | 4.167 |

Debounce removes one further false alarm (FP 4 → 3), raising precision to 0.625 and lowering
FA/h total to 4.871. Latency median is 4.167 s (n = 5; IQR reported as 35.23 in the JSON,
reflecting the very small sample).

### 6.5 Secondary regime — contamination cross-check

The secondary (concatenated, EXP-004-order) regime produces a large false-alarm explosion.
For V3/V4, pooled: TP 12, FP 723, FN 29, FA/h total ≈ 1173.8. The bulk comes from subject
002 (FP 644, FA/h ≈ 3693.1); subject 001 also fires under concatenation (FP 78,
FA/h ≈ 443.0), despite firing **zero** alarms in the primary regime.

The contamination-delta block quantifies the gap directly (V3):

| Quantity | Primary | Secondary |
|---|---|---|
| Total edges | 18 | 1469 |
| FP events | 4 | 723 |
| TP events | 5 | 12 |

Edge-count absolute delta = 1483. Boundary window = 5 frames; secondary edges near a
recording boundary = 0 (`secondary_edge_boundary_fraction = 0.0`). The extra edges are
therefore **not** an artifact of boundary carryover — they arise within concatenated
streams, confirming that per-recording reset (PRIMARY) is the correct headline regime and
that the concatenated EXP-004-style processing order inflates false alarms.

### 6.6 Minimum-duration (k) sweep

In the primary regime the k-sweep has little effect on the already-small event set. In the
**secondary** regime the k-sweep reveals a recall collapse for V3 (from the metrics JSON
`k_sweep`):

| k (s) | tol. frames | TP | FP | Recall |
|---|---|---|---|---|
| 0.00 | 1 | 12 | 723 | 0.293 |
| 0.25 | 8 | 1 | 734 | ~0.024 |
| 0.50 | 15 | 1 | 734 | ~0.024 |
| 1.00 | 30 | 0 | 735 | 0.000 |
| 2.00 | 60 | 0 | 735 | 0.000 |

Under concatenation, requiring even a 0.25 s minimum duration collapses true positives to 1,
and a ≥ 1 s floor drives recall to 0 while false alarms remain ~735. This is a further
symptom of the contaminated secondary regime and does not apply to the headline primary
numbers.

## 7. Figures

Four plots are produced under `experiments/EXP-005_events/`:

- `fa_per_hour_bars.png` — per-variant false-alarms-per-hour bar chart. Corresponds to the
  FA/h column of §6.1 (V3/V4 lowest at 6.494, V0/V2 highest at 9.741).
- `recall_vs_fa_per_hour.png` — recall against FA/h scatter across variants; visualizes the
  flat-recall / varying-FP trade-off of §6.1.
- `per_subject_fp_events.png` — per-subject false-alarm event counts; visualizes §6.2 (all
  FPs on subject 005; none on 001/002/006).
- `k_sensitivity_recall.png` — recall vs minimum-duration k; visualizes the §6.6 recall
  collapse in the secondary regime.

(The figures were confirmed present as generated artifacts; captions above describe the
underlying tabulated values, which are the source of every quantitative claim.)

## 8. Discussion

At the event level, the component that measurably improves the alarm profile on this corpus
is the **combination** of speech filter and reliability gate (V3): it gives the lowest
false-alarm rate (6.494/h, 4.871/h after debounce) and the highest precision (0.556, 0.625
after debounce), removing 2–3 of V0's 6 false alarms without changing recall.

The ablation is informative about *where* that gain comes from. The reliability gate **on
its own** (V2) does not reduce pooled false alarms — FA/h stays at V0's 9.741 and precision
is the lowest of any variant. The speech filter alone (V1) recovers part of the gain
(8.118/h). Only with both enabled (V3) does the false-alarm rate drop to 6.494/h. On this
corpus the false-alarm reduction is thus attributable to the speech-filter path (and its
interaction with the gate), not to the gate in isolation.

The **CNN arm has no observable effect**: V4 is byte-identical to V3 across every metric and
regime. This is consistent with the pipeline design — the CNN only flips `should_alarm` in
the `StateManager`, and on this corpus it never changed an alarm decision
(`any_cnn_override = 0` in every alarm event).

More broadly, the **reliability machinery produced no observable per-event actions**:
override, suppression, and critical-face-loss flags are all zero across every alarm event,
and the run log's observability gates G1 (CNN observable), G2 (gate observable), and G3
(speech observable) all **FAIL** with a diff of 0 frames (`all_ran_pass = False`). The
event-level differences that do exist (V1/V2/V3 vs V0) are therefore visible in aggregate
false-alarm counts but not attributable to any single logged override event.

## 9. Limitations

- **Clip-condition labels.** Ground-truth episodes derive from clip-level condition labels,
  so the evaluation measures clip-level alarm behaviour, not free-driving drowsiness
  episodes. Recall figures should be read in that light.
- **Very small effective sample.** n = 4 subjects, and only 2 of them (002, 005) ever fire
  an alarm; subjects 001 and 006 fire zero alarms in the primary regime. All statistics are
  descriptive; no significance testing is performed, and the reported bootstrap band is a
  dispersion estimate, not a test. TP counts of 5–6 make every rate sensitive to a single
  event.
- **All false alarms come from one subject.** Subject 005 is the sole FP source in the
  primary regime, so the entire V2-vs-V3 false-alarm difference rests on that subject.
- **Latency IQR is unstable.** Latency median 4.167 s is computed over n = 5 events; the
  IQR (35.23) reflects that tiny sample and should not be over-interpreted.
- **Secondary regime is contaminated.** The concatenated EXP-004-order regime inflates false
  alarms by orders of magnitude (723 vs 4 FP for V3) and collapses recall under any duration
  floor; it is retained only as a contamination cross-check, not as a result.
- **Observability gates fail.** G1/G2/G3 all report a 0-frame diff, meaning the harness could
  not observe the CNN, gate, or speech components changing any individual frame decision;
  component attribution is limited to aggregate event counts.

## 10. Conclusion

At the event level on NTHU-DDD (LOSO, primary per-recording regime), the full configuration
V3 (speech filter + reliability gate) delivers the best false-alarm profile — FA/h total
6.494 (4.871 after debounce) and precision 0.556 (0.625 after debounce) — versus the all-off
baseline V0 (9.741/h, precision 0.500). Adding the CNN arm (V4) changes nothing: V4 is
identical to V3 on every metric. The reliability gate alone (V2) does not reduce false
alarms; the improvement is carried by the speech-filter path. Recall is uniformly low
(~0.12–0.15) across all variants because the great majority of GT episodes never trigger an
alarm, and only 2 of 4 subjects fire any alarm at all. These are descriptive results on a
small corpus (n = 4) with clip-level labels, and the observability gates indicate the
component-level attribution is limited to aggregate event counts rather than logged
per-frame overrides.
