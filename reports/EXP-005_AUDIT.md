# EXP-005 — Independent Scientific Audit

**Audit target:** `reports/EXP-005_REPORT.md` (Event-Level Alarm Evaluation) and its
supporting artifacts under `experiments/EXP-005_events/`.
**Audit date:** 2026-07-29.
**Scope requested:** implementation correctness, statistics, conclusions, reproducibility,
consistency.
**Constraint honored:** the implementation (`evaluation/exp005_event_report.py`,
`evaluation/event_metrics.py`) was **read only and not modified**. This audit is a new
document and changes no experiment code or data.

---

## 0. Verdict

**The report is a faithful, reproducible, and unusually candid account of the underlying
artifacts.** Every quantitative claim I checked in §6 (per-variant, per-subject, debounced,
secondary, and k-sweep results), the discussion in §8, and the conclusions in §10
triangulate **exactly** across five independent sources: the report prose, the metrics JSON,
the per-variant/per-subject CSVs, the per-recording episode/stream CSVs, the run log, and the
four plots. The implementation is deterministic and internally consistent, and the report's
own §9 Limitations section pre-empts the most important scientific caveats (n = 4,
single-subject false-alarm dependence, clip-level labels, observability-gate failures).

I found **no numerical error and no implementation defect.** I found **two low-severity
documentation inconsistencies** in the report prose (a mislabeled sweep parameter and a
mis-captioned figure), and I record **two reproducibility caveats** (the run log does not
echo the seed; the debounce path is a reimplementation, as the report itself discloses). None
of these change any result or conclusion.

Overall assessment: **ACCEPT.** The headline finding — that at the event level on this corpus
V3 (speech filter + reliability gate) gives the best false-alarm profile, that the CNN arm
(V4) has zero observable effect, and that the gate alone (V2) does not reduce false alarms —
is supported by the artifacts, and the report states its own limitations accurately.

---

## 1. Materials audited

| Source | Path | Role |
|---|---|---|
| Report (read-only) | `reports/EXP-005_REPORT.md` (324 lines) | under audit |
| Orchestrator (read-only, **not modified**) | `evaluation/exp005_event_report.py` (1188 lines) | implementation |
| Metric core (read-only, **not modified**) | `evaluation/event_metrics.py` (557 lines) | implementation |
| Metrics JSON | `experiments/EXP-005_events/exp005_event_metrics.json` (117,620 B) | canonical results |
| Per-variant CSV | `experiments/EXP-005_events/per_variant_event_metrics.csv` | tabulated results |
| Per-subject CSV | `experiments/EXP-005_events/per_subject_event_metrics.csv` | tabulated results |
| Episode CSVs | `experiments/EXP-005_events/episodes/*.csv` (10 files) | raw events |
| Event-stream CSVs | `experiments/EXP-005_events/event_streams/*.csv` (5 files) | raw per-frame streams |
| Run log | `experiments/EXP-005_events/exp005_run.log` (79,635 B) | execution record |
| Plots | `experiments/EXP-005_events/plots/*.png` (4 files) | figures |

**Not located:** `recursive-churning-lecun.md` was named in the task as a read-only input but
**does not exist anywhere on disk** (searched the workspace). It could not be consulted. No
claim in this audit depends on it; I flag its absence for completeness.

---

## 2. Method

I treated the metrics JSON, CSVs, and run log as the ground truth produced by the harness,
re-derived the reported numbers from them independently, and then read the implementation
source to confirm that the harness computes what the report says it computes. Where the
report makes a claim, I sought the same value in **at least two** independent artifacts before
accepting it. I did not re-run the pipeline (that would require the NTHU-DDD data and a
~46-minute run); instead I verified that the persisted artifacts are mutually consistent and
that the code path producing them is deterministic.

---

## 3. Findings by dimension

### 3.1 Implementation correctness — **PASS**

- **Metric core (`event_metrics.py`).** Pure, deterministic functions over frame-indexed
  sequences. GT episodes and alarm events are maximal runs (`build_*_events`, L260–320);
  overlap is shared-frame-index count (`_overlap_frames`, L358–361); matching is one-to-one
  with a deterministic tie-break (`match_events`, L363–442). No wall-clock, RNG, or global
  state in the metric path. Event recall/precision/F1/FA-rate derive from tp/fp/fn only
  (`event_metrics_from_counts`, L448). Verified: there is no event-level TN, hence no
  specificity is reported — correct for event matching.
- **Orchestrator (`exp005_event_report.py`).** Builds both regimes, runs the k-sweep, the
  debounce reimplementation, the bootstrap, and the observability gates, then writes JSON +
  CSVs + plots. I traced each reported number to its producing call.
- **Debounce is a documented reimplementation.** `src/alarm_controller.py` hard-codes
  `time.monotonic()`/`datetime.now()`, so it cannot be replayed from a frozen frame stream.
  The harness reimplements debounce on the video clock (`min_alarm_duration = 3.0 s`,
  `cooldown_period = 5.0 s`, read from the frozen `cfg.alarm`; run log line 3 and JSON agree).
  The report discloses this explicitly (§4.5, §5). **Correctly handled and correctly
  disclosed**; the `debounced_primary` numbers are the reimplementation's output, not the
  production controller's.
- **`measured_results.json` untouched.** Run log: `--write not set; measured_results.json NOT
  modified.` Consistent with the report (§5).

**Resolved apparent contradiction (no bug).** The distinct `event_stream_md5` values across
variants (V0 `d694f72b`, V1 `00d9e52b`, V2 `574bdaec`, V3 = V4 `f958e65d`) coexist with
observability gates that report a **0-frame** difference. These are reconciled fully by the
implementation:
- `write_event_stream_csv` (L317–335) writes **both** regimes (PRIMARY + SECONDARY) into one
  CSV and hashes the **combined** file. I confirmed each stream CSV has 133,042 rows =
  66,521 × 2 regimes. The md5 differences across variants are driven by the **PRIMARY** regime,
  where the toggles genuinely change per-frame `should_alarm` (directly observed: V3 has
  965 alarm-frames in PRIMARY vs 1137 in SECONDARY).
- The observability gates (`check_observability_gates`, L607–651) run `frames_differ`
  **exclusively on the SECONDARY streams**, where all five variants are byte-identical
  (every variant: tp = 12, fp = 723, fn = 29 in secondary). Hence a 0-frame diff there is
  the *expected* result, not a contradiction.

**Cosmetic caveat (not a bug).** At L642–646 the three gate calls pass an argument whose local
name is `streams_primary`, but the value bound to it is the **secondary** stream set. The name
is misleading; the behavior is correct and matches the report's description that the gates are
computed on the secondary regime. No effect on any result. *(Read-only observation; per the
task constraint I did not modify the code. If ever revised, renaming this parameter would
remove a genuine readability trap.)*

### 3.2 Statistics — **PASS (appropriately modest)**

- The bootstrap (subject-stratified percentile, B = 2000, seed 42, CI 0.95,
  `np.random.default_rng`) is presented **as a dispersion band, explicitly not a significance
  test** (§4.6). This is the correct framing: with n = 4 subjects and only 2 firing any alarm,
  no inferential test is warranted, and the report does not attempt one.
- Rates rest on tiny counts (TP = 5–6; latency median over n = 5 events). The report says so
  (§9) and warns the latency IQR (35.23) is unstable. I concur; these numbers should be read
  as descriptive, not estimative.
- **Single-subject dependence (material, and disclosed).** Every false alarm in the primary
  regime originates from **subject 005** (verified in `per_subject_event_metrics.csv`, the
  episode CSVs, and the `per_subject_fp_events.png` plot: 005 = 6/5/6/4/4 for V0–V4; 001, 002,
  006 = 0). The entire V2-vs-V3 false-alarm difference therefore rests on one subject. This is
  the single most important statistical limitation, and §9 states it plainly.

Statistical conclusion: the analysis does not over-reach. The audit's only emphasis is that
readers must treat the headline FA/h numbers as *one subject's behavior*, which the report
already instructs.

### 3.3 Conclusions — **PASS**

Each §8/§10 conclusion is supported by the artifacts:

| Conclusion | Evidence (verified) |
|---|---|
| V3 best FA profile | Primary FA/h: V0 9.741, V1 8.118, V2 9.741, V3 = V4 6.494; debounced V3 4.871, precision 0.625 (JSON + CSV + log + `fa_per_hour_bars.png`) |
| CNN arm (V4) has no observable effect | V4 byte-identical to V3 on every metric/regime; `event_stream_md5` V3 = V4; `any_cnn_override = 0` in all alarm events; gate G1 FAIL (0 frames) |
| Reliability gate alone (V2) does not reduce FP | V2 FA/h = V0's 9.741; paired delta V0→V2 shows only `d_tp = -1` (hurts recall), no FP reduction; gate G2 FAIL |
| Improvement carried by speech-filter path | paired delta V0→V1 on subject 005: `d_fp = -1`, `d_fa = -4.924` — the sole FP reducer |
| Recall uniformly low (~0.12–0.15) | V0 recall 0.146 (6/41), V1–V4 0.122 (5/41); GT = 41 episodes, only 5–6 ever matched |
| Observability limited to aggregate counts | G1/G2/G3 all FAIL, `n_diff_frames = 0` over 66,521, `all_ran_pass = false` |

The conclusions are correctly scoped to "at the event level, on this corpus," and the report
does not claim significance or generalization.

### 3.4 Reproducibility — **PASS with caveats**

- **Deterministic by construction.** The metric path has no RNG; the only randomness (the
  bootstrap) is seeded (`default_rng(42)`). Same inputs ⇒ same JSON/CSVs.
- **Artifacts are mutually consistent** and the run log's final SUMMARY table reproduces §6.1
  and §6.4 exactly (V0 6/6/35, V1 5/5/36, V2 5/6/36, V3 5/4/36, V4 ≡ V3; debounced FP
  5/4/5/3/3). Wall-clock 46.5 min matches `wall_clock_seconds = 2790.88`.
- **Caveat 1 — seed not echoed in the log.** The report claims seed 42 (§4.6) and the code
  uses it, but `exp005_run.log` does **not** print the seed value anywhere. Reproducibility of
  the bootstrap band therefore rests on the report + code, not on an execution-time echo. A
  future run would benefit from logging the seed. *(This is the only reproducibility gap; it
  does not affect the deterministic point estimates, which carry no RNG.)*
- **Caveat 2 — debounce path is a reimplementation** (see §3.1). Fully disclosed; the
  production `alarm_controller.py` timing is not exercised, so the debounced numbers are the
  harness's video-clock model, not the shipped controller. This is unavoidable given the
  wall-clock coupling and is correctly flagged.
- Environment recorded: Apple M1, TFLite XNNPACK CPU delegate, `sklearn_used = false`, model
  `eye_state_model.tflite` input `[1 24 24 1]`. 66,521 frames, subjects [001, 002, 005, 006].

### 3.5 Consistency — **PASS, with two minor prose defects**

Cross-artifact numerical consistency is **exact** everywhere I checked (§6.1/§6.2/§6.4/§6.5/§6.6
↔ JSON ↔ CSVs ↔ log ↔ plots). Episode CSVs confirm GT = 41 (per variant: 42 lines − header)
and alarm-event counts V0/V1/V2/V3/V4 = 12/10/11/9/9, matching tp+fp in §6.1. The k-sweep JSON
matches §6.6's table digit-for-digit in both regimes.

Two **documentation** inconsistencies (report prose vs. the artifacts/code they describe;
neither affects a number):

- **C-1 — k-parameter mislabeled as a "minimum-duration floor."** §4.4/§6.6/§7 describe the
  sweep parameter k ∈ {0, 0.25, 0.5, 1, 2} s as a "minimum-duration threshold" / "debounce-style
  duration floor." In the implementation it is the **minimum-overlap-in-frames** required for an
  (episode, alarm) pair to match (`match_events(tolerance=…)`, `if ov >= tolerance`,
  `event_metrics.py` L363–396; fed by `_seconds_to_tolerance_frames`, L389). The code and the
  plot call it "matching tolerance" (`exp005_event_report.py` L99, L875, L877). Semantically it
  is neither a pure event-duration floor nor a near-miss tolerance — it is a *minimum required
  overlap*, and the report's own §6.6 secondary data proves it: as k rises, **FP increases**
  723 → 735 while TP falls 12 → 0. A genuine duration floor would *remove* short events and make
  FP *fall*; a minimum-overlap requirement pushes unmatched alarms into FP, so FP *rises* — which
  is what the table shows. The numbers are correct; only the label ("duration floor") is a
  misnomer. **Severity: low (wording).**

- **C-2 — figure caption for `k_sensitivity_recall.png` misattributed.** §7 says this plot
  "visualizes the §6.6 recall collapse in the secondary regime." The plot is in fact the
  **PRIMARY** regime (title and code both say so; L871 reads
  `results["per_variant"][n]["primary"]["k_sweep"]`), and it shows **flat** recall (V0 0.146,
  V1–V4 0.122) with **no** collapse — correctly visualizing §6.6's first sentence ("little
  effect" in primary). The secondary-regime collapse described by §6.6's table is **not plotted
  at all**. The plot itself is labeled correctly; the §7 caption describes the wrong regime and
  the wrong phenomenon. **Severity: low (caption).**

Note: the PRIMARY flatness and the SECONDARY collapse are *both* correct and both explained by
the min-overlap semantics — in PRIMARY, true alarms overlap their (long, tens-of-seconds) GT
episodes by far more than 60 frames, so a rising floor does not drop them; in the contaminated
SECONDARY concatenation, overlaps are tiny and collapse under any floor. The implementation
behaves correctly in both; only the report's §7 sentence mislabels the figure.

---

## 4. Findings register (severity-ranked)

| # | Severity | Type | Location | Summary | Result impact |
|---|---|---|---|---|---|
| C-1 | Low | Report wording | §4.4, §6.6, §7 | k called "minimum-duration floor"; it is a minimum-overlap-frames match threshold (code/plot call it "matching tolerance"); §6.6's rising FP confirms overlap semantics | None |
| C-2 | Low | Report caption | §7 | `k_sensitivity_recall.png` captioned as secondary-regime collapse; plot is PRIMARY and flat | None |
| R-1 | Low | Reproducibility | run log | Seed (42) not echoed at runtime; rests on report + code | None (deterministic point estimates carry no RNG) |
| R-2 | Info | Disclosed | §4.5, §5 | Debounce reimplemented on video clock; production `alarm_controller.py` timing not exercised | Scope caveat only; disclosed |
| I-1 | Info | Code readability | `exp005_event_report.py` L642–646 | Gate arg locally named `streams_primary` is bound to the **secondary** streams; behavior correct | None |
| M-1 | Info | Missing input | task input | `recursive-churning-lecun.md` not found on disk; could not be consulted | None on verifiable claims |

No high- or medium-severity findings. No implementation defect. No numerical discrepancy.

---

## 5. What this audit did **not** do

- Did not modify `evaluation/exp005_event_report.py` or `evaluation/event_metrics.py` (task
  constraint). All code observations are read-only.
- Did not re-execute the pipeline end-to-end (would require the NTHU-DDD corpus and ~46 min);
  instead verified mutual consistency of persisted artifacts and the determinism of the
  producing code path.
- Could not consult `recursive-churning-lecun.md` (absent from disk).

---

## 6. Recommendations (non-blocking, optional)

1. **C-1/C-2 (wording):** In a future report revision, rename the k-sweep description to
   "minimum-overlap (matching) tolerance" to match the code and the plot, and correct the §7
   caption to state that `k_sensitivity_recall.png` shows the **primary** (flat) sweep, with the
   secondary-regime collapse living in the §6.6 table. No code change required.
2. **R-1 (reproducibility):** Echo the RNG seed into the run log at start so bootstrap
   reproducibility is evidenced at execution time, not only asserted.
3. **I-1 (readability):** If the harness is ever touched for other reasons, rename the
   `streams_primary` parameter at L642–646 to reflect that the gates consume the secondary
   streams. (Explicitly out of scope for this audit per the no-modification constraint.)
4. **Scientific framing (already largely present):** Keep foregrounding that the entire
   false-alarm signal is one subject (005) and that recall ≈ 0.12–0.15 means ~86–88% of GT
   episodes are never alarmed — the report's §9/§10 already do this; no change needed, only
   emphasis if the numbers are cited elsewhere.

---

*End of audit. Deliverable: `reports/EXP-005_AUDIT.md`. Implementation untouched.*
