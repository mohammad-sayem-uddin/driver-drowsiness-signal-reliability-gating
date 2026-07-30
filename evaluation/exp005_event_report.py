"""
EXP-005 — Event-Level Alarm Evaluation: additive orchestrator
=============================================================
This is an ADDITIVE reporting/orchestration layer, mirroring the
``evaluation/exp004_report.py`` pattern. It changes NO frozen algorithm,
threshold, weight, split, or the swept ``fatigue_score``. It:

  1. Reuses the frozen machinery UNMODIFIED — ``VARIANTS``, ``VariantConfig``,
     the ground-truth enumeration (``enumerate_labelled_frames`` /
     ``group_by_subject`` / ``NTHUFrame`` / ``LABEL_*``), ``SystemConfig`` and
     its ablation toggles, and the ``FrameProcessor`` pipeline. The variant
     toggles are applied EXACTLY as ``loso_harness._run_subject`` applies them.

  2. Generates, per variant, BOTH alarm streams (plan decision #4):
       * PRIMARY   — a fresh ``FrameProcessor`` per ``(subject, glasses,
                     condition)`` recording, so each recording's temporal
                     accumulator and state machine start clean; local video
                     clock ``ts = local_frame_index / fps`` (monotonic within
                     the recording). This is the headline regime for event
                     definition — no alarm edge is contaminated by a prior
                     recording's dwell-pin or fusion carryover.
       * SECONDARY — one ``FrameProcessor`` per subject over the frozen
                     interleaved sorted order (byte-identical to how EXP-004
                     executed). Reported only to QUANTIFY the boundary
                     contamination the PRIMARY regime removes.

  3. Delegates ALL event math to the pure, unit-tested core
     ``evaluation/event_metrics.py`` — segmentation, GT-episode / alarm-event
     construction, matching (max-overlap PRIMARY, greedy-by-onset SECONDARY),
     event metrics, latency, FA/hour, and the descriptive subject-stratified
     bootstrap. No event logic is re-implemented here.

  4. Produces the deployment-realistic **debounced view** via a video-clock
     debounce **reimplemented in this module** (3s-min-duration / 5s-cooldown
     read from the frozen ``cfg.alarm`` block — NO new magic numbers). It is
     NOT a call into ``src/alarm_controller.py``, whose ``update()`` hard-codes
     ``time.monotonic()`` (:108) and ``_log_event`` uses ``datetime.now()``
     (:298) and so cannot be driven by the offline video clock. The
     reimplementation reproduces the documented AlarmController semantics and is
     unit-tested against them (``tests/test_event_metrics.py`` covers the core;
     the debounce transform here is a thin, deterministic edge filter).

  5. Writes artifacts under ``experiments/EXP-005_events/`` (raw per-frame event
     streams, episodes/alarm events, JSON metrics, CSVs, plots), computes the
     PRIMARY↔SECONDARY contamination delta, checks the pre-registered
     observability gates G1–G3, and — only under ``--write`` (gated exactly like
     EXP-004) — merges an ADDITIVE ``events`` block into
     ``results/measured_results.json`` (never touching the frozen ``roc`` block).

Determinism: fixed frame order (frozen enumerate/sort), fixed video clock
(frame_index/fps), fixed seed regime, and a fixed bootstrap seed — re-running
yields byte-identical event-stream CSVs and metric JSON.
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Frozen machinery (imported UNMODIFIED — reuse, never re-implement) ─────────
from src.config import SystemConfig
from src.frame_processor import FrameProcessor
from evaluation.loso_harness import VARIANTS, VariantConfig, RESULTS_PATH
from evaluation.nthu_ground_truth import (
    enumerate_labelled_frames, group_by_subject, NTHUFrame,
    LABEL_DROWSY, LABEL_ALERT,
)

# ── Pure event-metric core (imported UNMODIFIED — all event math lives there) ──
from evaluation import event_metrics as em
from evaluation.event_metrics import (
    FrameRecord, Segment, GTEpisode, AlarmEvent, MatchResult,
    build_segments, gt_episodes_for_segment, alarm_events_for_segment,
    match_events, event_metrics_from_counts, event_metrics_from_matches,
    match_latencies, latency_stats, hours_from_frames, fa_per_hour, bootstrap_ci,
)

EXP_ID = "EXP-005"
EXP_DIR = os.path.join("experiments", "EXP-005_events")
STREAMS_DIR = os.path.join(EXP_DIR, "event_streams")
EPISODES_DIR = os.path.join(EXP_DIR, "episodes")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
METRICS_JSON = os.path.join(EXP_DIR, "exp005_event_metrics.json")
PER_VARIANT_CSV = os.path.join(EXP_DIR, "per_variant_event_metrics.csv")
PER_SUBJECT_CSV = os.path.join(EXP_DIR, "per_subject_event_metrics.csv")

VARIANT_ORDER = ["V0", "V1", "V2", "V3", "V4"]

# Matching-tolerance sweep (SECONDS). This is a matching-tolerance sensitivity
# curve, NOT tuning of any frozen algorithm threshold (the alarm stream is
# fixed). k=0 -> any nonzero (1-frame) overlap (the PRIMARY criterion).
K_SWEEP_SECONDS = [0.0, 0.25, 0.5, 1.0, 2.0]

# Boundary window (frames) for the contamination-edge fraction (plan §Validation).
BOUNDARY_WINDOW_FRAMES = 5


# ══════════════════════════════════════════════════════════════════════════════
#  Per-frame alarm-stream generation (PRIMARY + SECONDARY)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_variant_toggles(cfg: SystemConfig, variant: VariantConfig) -> None:
    """Apply the variant ablation switches EXACTLY as loso_harness._run_subject
    does (frozen protocol §3). Mutates ``cfg`` in place; no other config touched."""
    cfg.ablation.speech_filter_enabled = variant.enable_speech_filter
    cfg.ablation.reliability_gate_enabled = variant.enable_reliability_gate


def _frame_record_from_result(fr: NTHUFrame, local_idx: int, ts: float,
                              result) -> FrameRecord:
    """Build the pure-core ``FrameRecord`` from a frozen ``FrameResult``.

    The ``face_lost_critical`` channel is derived from the exposed alarm level:
    level 3 is the face-lost-critical escalation (``state_manager`` alarm_level
    docstring), which is the only place the event core flags it. No decision is
    made here — every field is read verbatim from the pipeline output.
    """
    return FrameRecord(
        subject=fr.subject,
        glasses=fr.glasses,
        condition=fr.condition,
        frame_index=local_idx,
        ts=ts,
        label=fr.label,
        should_alarm=bool(result.should_alarm),
        alarm_level=int(result.alarm_level),
        alert_suppressed=bool(result.alarm_suppressed_actual),
        cnn_override=bool(result.cnn_override_active),
        face_lost_critical=(int(result.alarm_level) >= 3),
    )


def _segment_recordings(frames: List[NTHUFrame]) -> "OrderedDict[Tuple[str, str, str], List[NTHUFrame]]":
    """Split one subject's frames into ``(subject, glasses, condition)`` recordings.

    The true recording unit is the full triple (the frozen ground-truth sort key
    omits ``glasses`` — plan §Methodology). Recording order and intra-recording
    frame order follow the already-sorted input, so the PRIMARY local clock is
    monotonic within each recording. Deterministic.
    """
    rec: "OrderedDict[Tuple[str, str, str], List[NTHUFrame]]" = OrderedDict()
    for fr in frames:
        key = (fr.subject, fr.glasses, fr.condition)
        rec.setdefault(key, []).append(fr)
    return rec


def generate_streams(by_subject: "OrderedDict[str, List[NTHUFrame]]",
                     cfg: SystemConfig, variant: VariantConfig,
                     fps: float,
                     max_frames_per_subject: Optional[int]) -> Tuple[List[FrameRecord], List[FrameRecord]]:
    """Generate both per-frame alarm streams for one variant (plan decision #4).

    Returns ``(primary_records, secondary_records)``:

      * PRIMARY   — a FRESH ``FrameProcessor`` per ``(subject, glasses,
                    condition)`` recording; ``ts = local_frame_index / fps``
                    resets to 0 at each recording (clean state, monotonic clock).
      * SECONDARY — ONE ``FrameProcessor`` per subject over the frozen
                    interleaved sorted order (reproduces the EXP-004 regime); the
                    local per-recording ``frame_index`` and ``ts`` are recorded so
                    that segmentation and event construction remain per-recording,
                    but the processor state carries across recording boundaries.

    Frame decode drops (``cv2.imread`` -> ``None``) are skipped identically in
    both regimes, matching ``loso_harness._run_subject``.
    """
    import cv2

    _apply_variant_toggles(cfg, variant)

    primary: List[FrameRecord] = []
    secondary: List[FrameRecord] = []

    for subj, subj_frames in by_subject.items():
        recordings = _segment_recordings(subj_frames)
        if max_frames_per_subject:
            # Smoke-run cap only (never used in the full run). Cap PER RECORDING,
            # not with a subj_frames[:N] head-slice: the frozen sort key is
            # (subject, condition, frame_index), so nonsleepyCombination (all
            # label==0, ~6k frames/subject) sorts first and a head-slice would
            # sample ONLY alert frames from ONE condition — never drowsy content
            # and never the alarm/CNN/gate paths, defeating the smoke run. A
            # per-recording prefix keeps every (glasses, condition) recording
            # represented. The SECONDARY per-subject stream is rebuilt from the
            # capped set, preserving the frozen interleaved order (a stable
            # filter of subj_frames, keyed on the within-subject-unique
            # (glasses, condition, frame_index) triple).
            recordings = OrderedDict(
                (k, v[:max_frames_per_subject]) for k, v in recordings.items())
            keep = {(fr.glasses, fr.condition, fr.frame_index)
                    for v in recordings.values() for fr in v}
            subj_frames = [fr for fr in subj_frames
                           if (fr.glasses, fr.condition, fr.frame_index) in keep]

        # ── PRIMARY: fresh processor per recording ────────────────────────────
        for key, rec_frames in recordings.items():
            fp = FrameProcessor(cfg, enable_cnn=variant.enable_cnn)
            try:
                local_idx = 0
                for fr in rec_frames:
                    img = cv2.imread(fr.path)
                    if img is None:
                        continue
                    ts = local_idx / fps
                    result = fp.process(img, timestamp=ts)
                    primary.append(_frame_record_from_result(fr, local_idx, ts, result))
                    local_idx += 1
            finally:
                fp.close()

        # ── SECONDARY: one processor per subject over the interleaved order ───
        fp = FrameProcessor(cfg, enable_cnn=variant.enable_cnn)
        try:
            # Local per-recording frame counter so events stay per-recording even
            # though the processor is NOT reset at boundaries (EXP-004 regime).
            local_counters: Dict[Tuple[str, str, str], int] = {}
            for fr in subj_frames:
                img = cv2.imread(fr.path)
                if img is None:
                    continue
                key = (fr.subject, fr.glasses, fr.condition)
                local_idx = local_counters.get(key, 0)
                ts = local_idx / fps
                result = fp.process(img, timestamp=ts)
                secondary.append(_frame_record_from_result(fr, local_idx, ts, result))
                local_counters[key] = local_idx + 1
        finally:
            fp.close()

    return primary, secondary


# ══════════════════════════════════════════════════════════════════════════════
#  Video-clock debounce reimplementation (decision #2 — NOT AlarmController)
# ══════════════════════════════════════════════════════════════════════════════

def debounce_should_alarm(records: Sequence[FrameRecord],
                          min_alarm_duration: float,
                          cooldown_period: float) -> List[bool]:
    """Reimplement the AlarmController debounce on the OFFLINE video clock.

    Reproduces the documented AlarmController state machine
    (``src/alarm_controller.py`` ``_handle_alarm_request`` / ``_handle_alarm_release``)
    but driven by each record's video-clock ``ts`` instead of ``time.monotonic()``
    (which the real controller hard-codes at :108, making it undriveable offline):

      * Rising request while INACTIVE: start alarming UNLESS still within
        ``cooldown_period`` seconds of the previous stop.
      * Falling request while ACTIVE: keep alarming until ``min_alarm_duration``
        seconds have elapsed since start, then stop (records stop time).

    Runs per recording (state never crosses a ``(subject, glasses, condition)``
    boundary), so the debounced stream matches how the alarm would behave on each
    recording played in isolation. Returns a per-frame ``bool`` "alarm sounding"
    stream aligned 1:1 with ``records``. Pure and deterministic (no wall-clock).

    NOTE: parameters are the frozen ``cfg.alarm.min_alarm_duration`` (3.0s) and
    ``cfg.alarm.cooldown_period`` (5.0s); no thresholds are introduced here.
    """
    out = [False] * len(records)
    # Group indices by recording, preserving order.
    groups: "OrderedDict[Tuple[str, str, str], List[int]]" = OrderedDict()
    for i, r in enumerate(records):
        groups.setdefault((r.subject, r.glasses, r.condition), []).append(i)

    for _, idxs in groups.items():
        is_active = False
        alarm_start_ts = 0.0
        alarm_stop_ts = None  # None => no prior stop (no cooldown yet)
        for i in idxs:
            r = records[i]
            now = r.ts
            if r.should_alarm:
                # _handle_alarm_request
                if is_active:
                    pass  # already sounding (escalation is level-only, no edge)
                else:
                    in_cooldown = (alarm_stop_ts is not None
                                   and (now - alarm_stop_ts) < cooldown_period)
                    if not in_cooldown:
                        is_active = True
                        alarm_start_ts = now
            else:
                # _handle_alarm_release
                if is_active:
                    elapsed = now - alarm_start_ts
                    if elapsed >= min_alarm_duration:
                        is_active = False
                        alarm_stop_ts = now
                    # else: still within min duration -> keep sounding
            out[i] = is_active
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Raw per-frame event-stream persistence (auditable evidence)
# ══════════════════════════════════════════════════════════════════════════════

_STREAM_HEADER = [
    "regime", "subject", "glasses", "condition", "local_frame_index", "local_ts",
    "label", "should_alarm", "debounced_alarm", "alarm_level", "alert_suppressed",
    "cnn_override", "face_lost_critical",
]


def write_event_stream_csv(path: str, primary: Sequence[FrameRecord],
                           secondary: Sequence[FrameRecord],
                           primary_debounced: Sequence[bool],
                           secondary_debounced: Sequence[bool]) -> str:
    """Persist both raw per-frame streams (PRIMARY + SECONDARY) with the debounced
    channel. Returns the md5 of the written file (for the determinism check)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_STREAM_HEADER)
        for regime, recs, deb in (("primary", primary, primary_debounced),
                                  ("secondary", secondary, secondary_debounced)):
            for r, d in zip(recs, deb):
                w.writerow([
                    regime, r.subject, r.glasses, r.condition, r.frame_index,
                    f"{r.ts:.6f}", r.label, int(r.should_alarm), int(bool(d)),
                    r.alarm_level, int(r.alert_suppressed), int(r.cnn_override),
                    int(r.face_lost_critical),
                ])
    return _md5(path)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  Event construction + matching per regime (delegates to event_metrics core)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentEvents:
    """Per-recording episodes, alarm events, and the primary match."""
    segment: Segment
    episodes: List[GTEpisode]
    alarms: List[AlarmEvent]


def _apply_debounce_to_records(records: Sequence[FrameRecord],
                               debounced: Sequence[bool]) -> List[FrameRecord]:
    """Return copies of ``records`` with ``should_alarm`` replaced by the debounced
    stream (all other channels preserved) — used to build the debounced alarm
    events through the same core path as the raw stream."""
    out: List[FrameRecord] = []
    for r, d in zip(records, debounced):
        out.append(FrameRecord(
            subject=r.subject, glasses=r.glasses, condition=r.condition,
            frame_index=r.frame_index, ts=r.ts, label=r.label,
            should_alarm=bool(d), alarm_level=r.alarm_level,
            alert_suppressed=r.alert_suppressed, cnn_override=r.cnn_override,
            face_lost_critical=r.face_lost_critical,
        ))
    return out


def build_events(records: Sequence[FrameRecord]) -> List[SegmentEvents]:
    """Segment records and build GT episodes + alarm events per recording
    (delegates to the pure core). Returns one entry per segment."""
    segs = build_segments(records)
    out: List[SegmentEvents] = []
    for seg in segs:
        out.append(SegmentEvents(
            segment=seg,
            episodes=gt_episodes_for_segment(seg),
            alarms=alarm_events_for_segment(seg),
        ))
    return out


def _seconds_to_tolerance_frames(k_seconds: float, fps: float) -> int:
    """Convert a seconds tolerance to a min-overlap in frames (>=1). k=0 -> 1
    frame (any nonzero overlap)."""
    return max(1, int(round(k_seconds * fps)))


def match_all(seg_events: Sequence[SegmentEvents], tolerance_frames: int,
              strategy: str) -> List[MatchResult]:
    """Match every segment with the given strategy/tolerance (core call)."""
    return [match_events(se.episodes, se.alarms,
                         tolerance=tolerance_frames, strategy=strategy)
            for se in seg_events]


# ══════════════════════════════════════════════════════════════════════════════
#  Metric aggregation (per subject, pooled-micro, macro) — delegates to core
# ══════════════════════════════════════════════════════════════════════════════

def _group_seg_events_by_subject(seg_events: Sequence[SegmentEvents]) -> "OrderedDict[str, List[SegmentEvents]]":
    g: "OrderedDict[str, List[SegmentEvents]]" = OrderedDict()
    for se in seg_events:
        g.setdefault(se.segment.subject, []).append(se)
    return g


def _subject_totals(seg_events: Sequence[SegmentEvents], fps: float,
                    tolerance_frames: int, strategy: str) -> Dict:
    """Per-subject event counts, metrics, FA/hour, latency, and episode stats."""
    matches = match_all(seg_events, tolerance_frames, strategy)
    metrics = event_metrics_from_matches(matches)

    n_frames = sum(len(se.segment) for se in seg_events)
    hours = hours_from_frames(n_frames, fps)
    alert_frames = sum(1 for se in seg_events for fr in se.segment.frames
                       if fr.label == LABEL_ALERT)
    alert_hours = hours_from_frames(alert_frames, fps)

    latencies: List[float] = []
    for se, m in zip(seg_events, matches):
        latencies.extend(match_latencies(se.episodes, se.alarms, m))
    lat = latency_stats(latencies)

    n_episodes = sum(len(se.episodes) for se in seg_events)
    n_alarms = sum(len(se.alarms) for se in seg_events)
    alarm_frames = sum(a.n_frames for se in seg_events for a in se.alarms)
    duty_cycle = (alarm_frames / n_frames) if n_frames else 0.0

    return {
        "n_recordings": len(seg_events),
        "n_frames": n_frames,
        "recording_hours": hours,
        "alert_hours": alert_hours,
        "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn,
        "recall": metrics.recall, "precision": metrics.precision,
        "miss_rate": metrics.miss_rate, "f1": metrics.f1,
        "fa_per_hour_total": fa_per_hour(metrics.fp, hours),
        "fa_per_hour_alert": fa_per_hour(metrics.fp, alert_hours),
        "n_gt_episodes": n_episodes,
        "n_alarm_events": n_alarms,
        "alarm_duty_cycle": duty_cycle,
        "latency": _latency_to_dict(lat),
        "_fp_hours_unit": (metrics.fp, hours),      # for the bootstrap
        "_tp_fn_unit": (metrics.tp, metrics.fn),    # for the bootstrap
    }


def _latency_to_dict(lat: em.LatencyStats) -> Dict:
    return {"n": lat.n, "median": lat.median, "q1": lat.q1, "q3": lat.q3,
            "iqr": lat.iqr, "min": lat.min, "max": lat.max, "mean": lat.mean}


def _macro_mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaNs
    return float(np.mean(vals)) if vals else float("nan")


def compute_variant_regime_metrics(seg_events: Sequence[SegmentEvents], fps: float,
                                   strategy: str,
                                   k_sweep_seconds: Sequence[float]) -> Dict:
    """Full per-variant/per-regime metric block: per-subject, pooled-micro, macro,
    the k-sweep, and the descriptive bootstrap band. All math via the core."""
    by_subj = _group_seg_events_by_subject(seg_events)

    # Primary tolerance = 1 frame (any nonzero overlap).
    per_subject: "OrderedDict[str, Dict]" = OrderedDict()
    for subj, ses in by_subj.items():
        per_subject[subj] = _subject_totals(ses, fps, tolerance_frames=1,
                                             strategy=strategy)

    # Pooled (micro): counts summed across all recordings/subjects.
    pooled_matches = match_all(seg_events, tolerance_frames=1, strategy=strategy)
    pooled = event_metrics_from_matches(pooled_matches)
    total_frames = sum(len(se.segment) for se in seg_events)
    total_hours = hours_from_frames(total_frames, fps)
    total_alert_frames = sum(1 for se in seg_events for fr in se.segment.frames
                             if fr.label == LABEL_ALERT)
    total_alert_hours = hours_from_frames(total_alert_frames, fps)

    pooled_latencies: List[float] = []
    for se, m in zip(seg_events, pooled_matches):
        pooled_latencies.extend(match_latencies(se.episodes, se.alarms, m))

    pooled_block = {
        "tp": pooled.tp, "fp": pooled.fp, "fn": pooled.fn,
        "recall": pooled.recall, "precision": pooled.precision,
        "miss_rate": pooled.miss_rate, "f1": pooled.f1,
        "recording_hours": total_hours, "alert_hours": total_alert_hours,
        "fa_per_hour_total": fa_per_hour(pooled.fp, total_hours),
        "fa_per_hour_alert": fa_per_hour(pooled.fp, total_alert_hours),
        "latency": _latency_to_dict(latency_stats(pooled_latencies)),
    }

    # Macro-over-subjects (mean of per-subject ratios; NaN-safe).
    macro_block = {
        "recall": _macro_mean([d["recall"] for d in per_subject.values()]),
        "precision": _macro_mean([d["precision"] for d in per_subject.values()]),
        "miss_rate": _macro_mean([d["miss_rate"] for d in per_subject.values()]),
        "f1": _macro_mean([d["f1"] for d in per_subject.values()]),
        "fa_per_hour_total": _macro_mean(
            [d["fa_per_hour_total"] for d in per_subject.values()]),
        "fa_per_hour_alert": _macro_mean(
            [d["fa_per_hour_alert"] for d in per_subject.values()]),
    }

    # Matching-tolerance k-sweep (pooled counts at each k; sensitivity curve).
    k_sweep: List[Dict] = []
    for k in k_sweep_seconds:
        tol = _seconds_to_tolerance_frames(k, fps)
        ms = match_all(seg_events, tolerance_frames=tol, strategy=strategy)
        mm = event_metrics_from_matches(ms)
        k_sweep.append({
            "k_seconds": k, "tolerance_frames": tol,
            "tp": mm.tp, "fp": mm.fp, "fn": mm.fn,
            "recall": mm.recall, "precision": mm.precision, "f1": mm.f1,
            "fa_per_hour_total": fa_per_hour(mm.fp, total_hours),
        })

    # Descriptive subject-stratified bootstrap (NOT a hypothesis test).
    fp_hours_units = [d["_fp_hours_unit"] for d in per_subject.values()]
    tp_fn_units = [d["_tp_fn_unit"] for d in per_subject.values()]
    fa_boot = bootstrap_ci(
        fp_hours_units,
        combine=lambda s: (sum(fp for fp, _ in s)
                           / max(1e-12, sum(h for _, h in s))),
        n_boot=2000, seed=42, ci_level=0.95)
    recall_boot = bootstrap_ci(
        tp_fn_units,
        combine=lambda s: (sum(tp for tp, _ in s)
                           / max(1, sum(tp + fn for tp, fn in s))),
        n_boot=2000, seed=42, ci_level=0.95)

    # Strip the private bootstrap-unit helpers from the serialized per-subject.
    per_subject_clean = OrderedDict()
    for subj, d in per_subject.items():
        per_subject_clean[subj] = {k: v for k, v in d.items()
                                   if not k.startswith("_")}

    return {
        "strategy": strategy,
        "per_subject": per_subject_clean,
        "pooled_micro": pooled_block,
        "macro_over_subjects": macro_block,
        "k_sweep": k_sweep,
        "bootstrap_descriptive": {
            "note": ("Subject-stratified percentile bootstrap dispersion band "
                     "(B=2000, seed 42). NOT a significance test: with n=4 "
                     "subjects it bounds within-sample dispersion only "
                     "(plan §Statistics)."),
            "fa_per_hour_total": _bootstrap_to_dict(fa_boot),
            "recall": _bootstrap_to_dict(recall_boot),
        },
    }


def _bootstrap_to_dict(b: em.BootstrapCI) -> Dict:
    return {"point": b.point, "lo": b.lo, "hi": b.hi, "mean": b.mean,
            "std": b.std, "n_boot": b.n_boot, "ci_level": b.ci_level,
            "seed": b.seed}


# ══════════════════════════════════════════════════════════════════════════════
#  Observability gates G1–G3 + contamination delta (plan §Validation)
# ══════════════════════════════════════════════════════════════════════════════

def _alarm_key_stream(records: Sequence[FrameRecord]) -> "OrderedDict[Tuple, List[bool]]":
    """Per-recording should_alarm streams keyed by (subject, glasses, condition),
    frames in order. Used for cross-variant frame-level comparison (gates)."""
    out: "OrderedDict[Tuple, List[bool]]" = OrderedDict()
    for r in records:
        out.setdefault((r.subject, r.glasses, r.condition), []).append(
            bool(r.should_alarm))
    return out


def frames_differ(records_a: Sequence[FrameRecord],
                  records_b: Sequence[FrameRecord]) -> Dict:
    """Count frames where ``should_alarm`` differs between two aligned streams
    (same regime, two variants). Alignment is by recording key + position; only
    positions present in both are compared. Used for gates G1–G3."""
    sa = _alarm_key_stream(records_a)
    sb = _alarm_key_stream(records_b)
    n_diff = 0
    n_compared = 0
    subjects_with_diff = set()
    for key in sa:
        if key not in sb:
            continue
        a, b = sa[key], sb[key]
        m = min(len(a), len(b))
        for i in range(m):
            n_compared += 1
            if a[i] != b[i]:
                n_diff += 1
                subjects_with_diff.add(key[0])
    return {"n_diff": n_diff, "n_compared": n_compared,
            "subjects_with_diff": sorted(subjects_with_diff)}


def check_observability_gates(streams_primary: Dict[str, List[FrameRecord]],
                              streams_secondary: Dict[str, List[FrameRecord]],
                              variant_keys: Sequence[str]) -> Dict:
    """Pre-registered observability gates (plan §Validation). Declared before the
    run; PASS licenses reporting the event metrics. The gates test only that the
    harness is NOT blind — the SIGN of any event effect is whatever it measures.

      * G1 (CNN observable):  V3 vs V4 should_alarm differ on >=1 frame on the
        SECONDARY (EXP-004-identical) regime (subject 005 known to fire the CNN).
      * G2 (gate observable): V0 vs V2 should_alarm differ on >=1 frame.
      * G3 (speech observable): V0 vs V1 should_alarm differ on >=1 frame.

    A gate whose variants are not both in ``variant_keys`` is reported as
    ``skipped`` (not run), never as a pass.
    """
    gates: "OrderedDict[str, Dict]" = OrderedDict()

    def _gate(name: str, va: str, vb: str, regime_streams: Dict,
              regime_name: str, extra: str = "") -> None:
        if va not in variant_keys or vb not in variant_keys:
            gates[name] = {"status": "skipped",
                           "reason": f"{va} and/or {vb} not in run",
                           "regime": regime_name}
            return
        diff = frames_differ(regime_streams[va], regime_streams[vb])
        gates[name] = {
            "status": "PASS" if diff["n_diff"] >= 1 else "FAIL",
            "comparison": f"{va} vs {vb}",
            "regime": regime_name,
            "n_diff_frames": diff["n_diff"],
            "n_compared_frames": diff["n_compared"],
            "subjects_with_diff": diff["subjects_with_diff"],
            "note": extra,
        }

    _gate("G1_cnn_observable", "V3", "V4", streams_secondary, "secondary",
          "CNN arm must be observable at the event level (EXP-004 was blind).")
    _gate("G2_gate_observable", "V0", "V2", streams_secondary, "secondary",
          "Reliability-gate suppression path must be observable.")
    _gate("G3_speech_observable", "V0", "V1", streams_secondary, "secondary",
          "Speech-filter path must be observable.")

    ran = [g for g in gates.values() if g["status"] in ("PASS", "FAIL")]
    all_pass = bool(ran) and all(g["status"] == "PASS" for g in ran)
    return {"gates": gates, "all_ran_pass": all_pass}


def _edges(stream: Sequence[bool]) -> List[int]:
    """Indices where ``should_alarm`` transitions (rising or falling) within a
    per-recording stream. Position i is an edge if stream[i] != stream[i-1]."""
    out: List[int] = []
    for i in range(1, len(stream)):
        if stream[i] != stream[i - 1]:
            out.append(i)
    return out


def contamination_delta(primary: Sequence[FrameRecord],
                        secondary: Sequence[FrameRecord],
                        primary_events: Sequence[SegmentEvents],
                        secondary_events: Sequence[SegmentEvents],
                        fps: float) -> Dict:
    """PRIMARY↔SECONDARY contamination delta (plan §Validation). REPORTED, not
    pass/fail: (a) count of should_alarm edges that flip between regimes,
    (b) fraction of SECONDARY edges within N frames of a recording start,
    (c) net change in FP/TP event counts PRIMARY->SECONDARY."""
    p_streams = _alarm_key_stream(primary)
    s_streams = _alarm_key_stream(secondary)

    # (a) edge-count delta per recording (aligned by key + position).
    total_edge_flips = 0
    p_edge_total = 0
    s_edge_total = 0
    for key in p_streams:
        p = p_streams[key]
        s = s_streams.get(key, [])
        pe, se = _edges(p), _edges(s)
        p_edge_total += len(pe)
        s_edge_total += len(se)
        total_edge_flips += abs(len(pe) - len(se))

    # (b) fraction of SECONDARY edges within N frames of the recording start.
    near_boundary = 0
    s_edge_all = 0
    for key, s in s_streams.items():
        for e in _edges(s):
            s_edge_all += 1
            if e < BOUNDARY_WINDOW_FRAMES:  # near the recording start boundary
                near_boundary += 1
    boundary_frac = (near_boundary / s_edge_all) if s_edge_all else 0.0

    # (c) net FP/TP-event change PRIMARY -> SECONDARY (max-overlap, tol=1).
    p_m = event_metrics_from_matches(
        match_all(primary_events, tolerance_frames=1, strategy="max_overlap"))
    s_m = event_metrics_from_matches(
        match_all(secondary_events, tolerance_frames=1, strategy="max_overlap"))

    return {
        "primary_total_edges": p_edge_total,
        "secondary_total_edges": s_edge_total,
        "edge_count_abs_delta": total_edge_flips,
        "boundary_window_frames": BOUNDARY_WINDOW_FRAMES,
        "secondary_edges_near_boundary": near_boundary,
        "secondary_edges_total": s_edge_all,
        "secondary_edge_boundary_fraction": boundary_frac,
        "fp_events_primary": p_m.fp, "fp_events_secondary": s_m.fp,
        "fp_events_delta": s_m.fp - p_m.fp,
        "tp_events_primary": p_m.tp, "tp_events_secondary": s_m.tp,
        "tp_events_delta": s_m.tp - p_m.tp,
        "note": ("Reported finding, not pass/fail. Near-zero delta => boundary "
                 "carryover negligible for this corpus; large delta justifies "
                 "PRIMARY (per-recording reset) as the headline regime."),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Episode / alarm-event artifact persistence
# ══════════════════════════════════════════════════════════════════════════════

def write_episode_artifacts(variant_name: str,
                            primary_events: Sequence[SegmentEvents]) -> None:
    """Persist GT episodes and alarm events (PRIMARY regime) per recording."""
    os.makedirs(EPISODES_DIR, exist_ok=True)
    ep_path = os.path.join(EPISODES_DIR, f"{variant_name}_gt_episodes.csv")
    al_path = os.path.join(EPISODES_DIR, f"{variant_name}_alarm_events.csv")
    with open(ep_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "glasses", "condition", "start_idx", "end_idx",
                    "start_ts", "end_ts", "duration", "n_frames"])
        for se in primary_events:
            for e in se.episodes:
                w.writerow([e.subject, e.glasses, e.condition, e.start_idx,
                            e.end_idx, f"{e.start_ts:.6f}", f"{e.end_ts:.6f}",
                            f"{e.duration:.6f}", e.n_frames])
    with open(al_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "glasses", "condition", "start_idx", "end_idx",
                    "onset_ts", "offset_ts", "duration", "n_frames",
                    "peak_alarm_level", "any_cnn_override",
                    "any_alert_suppressed", "any_face_lost_critical"])
        for se in primary_events:
            for a in se.alarms:
                w.writerow([a.subject, a.glasses, a.condition, a.start_idx,
                            a.end_idx, f"{a.onset_ts:.6f}", f"{a.offset_ts:.6f}",
                            f"{a.duration:.6f}", a.n_frames, a.peak_alarm_level,
                            int(a.any_cnn_override), int(a.any_alert_suppressed),
                            int(a.any_face_lost_critical)])


# ══════════════════════════════════════════════════════════════════════════════
#  CSV summaries
# ══════════════════════════════════════════════════════════════════════════════

def write_csvs(results: Dict, variant_keys: Sequence[str]) -> None:
    """Per-variant and per-subject event-metric CSVs (PRIMARY regime headline)."""
    with open(PER_VARIANT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "name", "regime", "tp", "fp", "fn", "recall",
                    "precision", "miss_rate", "f1", "fa_per_hour_total",
                    "fa_per_hour_alert", "recording_hours"])
        for k in variant_keys:
            name = VARIANTS[k].name
            pm = results["per_variant"][name]["primary"]["pooled_micro"]
            w.writerow([k, name, "primary", pm["tp"], pm["fp"], pm["fn"],
                        f"{pm['recall']:.6f}", f"{pm['precision']:.6f}",
                        f"{pm['miss_rate']:.6f}", f"{pm['f1']:.6f}",
                        f"{pm['fa_per_hour_total']:.6f}",
                        f"{pm['fa_per_hour_alert']:.6f}",
                        f"{pm['recording_hours']:.6f}"])

    with open(PER_SUBJECT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "name", "subject", "regime", "n_recordings",
                    "n_frames", "tp", "fp", "fn", "recall", "precision",
                    "miss_rate", "f1", "fa_per_hour_total", "fa_per_hour_alert",
                    "n_gt_episodes", "n_alarm_events", "alarm_duty_cycle",
                    "latency_median", "latency_n"])
        for k in variant_keys:
            name = VARIANTS[k].name
            ps = results["per_variant"][name]["primary"]["per_subject"]
            for subj, d in ps.items():
                w.writerow([k, name, subj, "primary", d["n_recordings"],
                            d["n_frames"], d["tp"], d["fp"], d["fn"],
                            f"{d['recall']:.6f}", f"{d['precision']:.6f}",
                            f"{d['miss_rate']:.6f}", f"{d['f1']:.6f}",
                            f"{d['fa_per_hour_total']:.6f}",
                            f"{d['fa_per_hour_alert']:.6f}", d["n_gt_episodes"],
                            d["n_alarm_events"], f"{d['alarm_duty_cycle']:.6f}",
                            f"{d['latency']['median']:.6f}", d["latency"]["n"]])
    print(f"[{EXP_ID}] CSVs written -> {PER_VARIANT_CSV}, {PER_SUBJECT_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plots (matplotlib Agg; additive)
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(results: Dict, variant_keys: Sequence[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names = [VARIANTS[k].name for k in variant_keys]
    x = np.arange(len(variant_keys))

    def pooled(name, key):
        return results["per_variant"][name]["primary"]["pooled_micro"][key]

    # (1) FA/hour bars (PRIMARY, pooled-micro, total-time denominator).
    fah = [pooled(n, "fa_per_hour_total") for n in names]
    fah = [0.0 if v != v else v for v in fah]  # NaN -> 0 for display
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.bar(x, fah, 0.6, color="#c0392b")
    plt.xticks(x, variant_keys)
    plt.ylabel("False alarms / hour (pooled, total-time)")
    plt.title("EXP-005 FA/hour by variant (PRIMARY, per-recording reset)",
              fontweight="bold")
    for i, v in enumerate(fah):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fa_per_hour_bars.png"))
    plt.close()

    # (2) Operating-point scatter (recall vs FA/hour), one point per variant.
    recalls = [pooled(n, "recall") for n in names]
    plt.figure(figsize=(6, 5), dpi=300)
    for i, k in enumerate(variant_keys):
        plt.scatter(fah[i], recalls[i], s=60)
        plt.annotate(k, (fah[i], recalls[i]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xlabel("False alarms / hour (pooled)")
    plt.ylabel("Event recall (TP / (TP+FN))")
    plt.title("EXP-005 operating point: recall vs FA/hour (PRIMARY)",
              fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "recall_vs_fa_per_hour.png"))
    plt.close()

    # (3) Per-subject FP-event bars (grouped by variant).
    subjects = list(
        results["per_variant"][names[0]]["primary"]["per_subject"].keys())
    plt.figure(figsize=(8, 4.5), dpi=300)
    width = 0.8 / max(1, len(variant_keys))
    for vi, (k, n) in enumerate(zip(variant_keys, names)):
        ps = results["per_variant"][n]["primary"]["per_subject"]
        fps_ = [ps[s]["fp"] for s in subjects]
        plt.bar(np.arange(len(subjects)) + vi * width, fps_, width, label=k)
    plt.xticks(np.arange(len(subjects)) + width * (len(variant_keys) - 1) / 2,
               subjects)
    plt.xlabel("Subject")
    plt.ylabel("False-alarm events (count)")
    plt.title("EXP-005 per-subject FP-event counts (PRIMARY)", fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_subject_fp_events.png"))
    plt.close()

    # (4) k-sensitivity curves (recall vs k, PRIMARY pooled).
    plt.figure(figsize=(7, 4.5), dpi=300)
    for k in variant_keys:
        n = VARIANTS[k].name
        ks = results["per_variant"][n]["primary"]["k_sweep"]
        xs = [row["k_seconds"] for row in ks]
        ys = [row["recall"] for row in ks]
        plt.plot(xs, ys, marker="o", label=k)
    plt.xlabel("Matching tolerance k (seconds)")
    plt.ylabel("Event recall")
    plt.title("EXP-005 matching-tolerance sensitivity (PRIMARY)",
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "k_sensitivity_recall.png"))
    plt.close()

    print(f"[{EXP_ID}] Plots written -> {PLOTS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  Paired per-subject variant deltas (descriptive; plan §Statistics)
# ══════════════════════════════════════════════════════════════════════════════

def paired_deltas(results: Dict) -> Dict:
    """Per-subject raw differences for the key contrasts (gate: V0 vs V2;
    speech: V0 vs V1; CNN: V3 vs V4) — raw counts, not a test."""
    contrasts = [("gate_V0_vs_V2", "V0", "V2"),
                 ("speech_V0_vs_V1", "V0", "V1"),
                 ("cnn_V3_vs_V4", "V3", "V4")]
    out: Dict = {}
    pv = results["per_variant"]
    for label, ka, kb in contrasts:
        na, nb = VARIANTS[ka].name, VARIANTS[kb].name
        if na not in pv or nb not in pv:
            continue
        pa = pv[na]["primary"]["per_subject"]
        pb = pv[nb]["primary"]["per_subject"]
        rows = []
        for subj in pa:
            if subj not in pb:
                continue
            rows.append({
                "subject": subj,
                "d_fp": pb[subj]["fp"] - pa[subj]["fp"],
                "d_tp": pb[subj]["tp"] - pa[subj]["tp"],
                "d_fn": pb[subj]["fn"] - pa[subj]["fn"],
                "d_fa_per_hour_total": (_nan_sub(pb[subj]["fa_per_hour_total"],
                                                 pa[subj]["fa_per_hour_total"])),
            })
        out[label] = {"comparison": f"{ka} -> {kb}", "per_subject": rows}
    return out


def _nan_sub(a: float, b: float) -> float:
    if a != a or b != b:
        return float("nan")
    return a - b


# ══════════════════════════════════════════════════════════════════════════════
#  Additive events block for measured_results.json (--write, gated)
# ══════════════════════════════════════════════════════════════════════════════

def build_events_block(results: Dict, variant_keys: Sequence[str],
                       gates: Dict, protocol: Dict) -> Dict:
    """The compact ADDITIVE ``events`` block merged into measured_results.json.
    Carries the headline PRIMARY pooled-micro numbers per variant plus protocol
    and gate provenance. Never touches the frozen ``roc`` block."""
    per_variant = OrderedDict()
    for k in variant_keys:
        name = VARIANTS[k].name
        pm = results["per_variant"][name]["primary"]["pooled_micro"]
        per_variant[name] = {
            "variant_key": k,
            "tp": pm["tp"], "fp": pm["fp"], "fn": pm["fn"],
            "recall": round(pm["recall"], 6),
            "precision": round(pm["precision"], 6),
            "miss_rate": round(pm["miss_rate"], 6),
            "f1": round(pm["f1"], 6),
            "fa_per_hour_total": (round(pm["fa_per_hour_total"], 6)
                                  if pm["fa_per_hour_total"] == pm["fa_per_hour_total"]
                                  else None),
        }
    return {
        "experiment_id": EXP_ID,
        "regime": "primary_per_recording_reset",
        "protocol": protocol,
        "observability_gates": gates,
        "per_variant": per_variant,
    }


def merge_events_block(events_block: Dict, out_path: str = RESULTS_PATH) -> None:
    """Additively merge the ``events`` block into measured_results.json. Reads,
    updates only the ``events`` key, and re-writes — the frozen ``roc`` /
    ``latency_ms`` blocks are preserved byte-for-byte."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data: Dict = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
    data["events"] = events_block
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[{EXP_ID}] Additive 'events' block merged into {out_path} "
          "(roc/latency untouched). Log the EXP-005 registry row before citing.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="EXP-005 event-level alarm evaluation (additive orchestrator).")
    ap.add_argument("--variants", nargs="+", default=VARIANT_ORDER,
                    choices=list(VARIANTS.keys()))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Smoke runs only: cap frames per RECORDING "
                         "((subject,glasses,condition) unit), keeping every "
                         "recording — and thus every condition/label regime — "
                         "represented. The frozen full run uses no cap.")
    ap.add_argument("--nthu-root", default=os.path.join("Data", "nthu_ddd"))
    ap.add_argument("--write", action="store_true",
                    help="Merge the additive 'events' block into "
                         "measured_results.json (gated exactly like EXP-004).")
    args = ap.parse_args()

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(STREAMS_DIR, exist_ok=True)
    variant_keys = [k for k in VARIANT_ORDER if k in args.variants]

    t_start = time.perf_counter()

    print(f"[{EXP_ID}] Loading NTHU ground truth from {args.nthu_root} ...",
          flush=True)
    all_frames = enumerate_labelled_frames(args.nthu_root)
    by_subject = group_by_subject(all_frames)
    subjects = list(by_subject.keys())
    print(f"[{EXP_ID}] {len(all_frames)} frames, {len(subjects)} subjects: "
          f"{subjects}", flush=True)

    cfg = SystemConfig()
    min_dur = cfg.alarm.min_alarm_duration
    cooldown = cfg.alarm.cooldown_period
    print(f"[{EXP_ID}] Debounce (frozen cfg.alarm): min_alarm_duration="
          f"{min_dur}s, cooldown_period={cooldown}s", flush=True)

    # Per-variant capture.
    primary_streams: Dict[str, List[FrameRecord]] = {}
    secondary_streams: Dict[str, List[FrameRecord]] = {}
    stream_md5: Dict[str, str] = {}
    results: Dict = {"per_variant": OrderedDict()}

    for key in variant_keys:
        variant = VARIANTS[key]
        print(f"\n[{EXP_ID}] === Variant {key} ({variant.name}) ===", flush=True)
        t0 = time.perf_counter()
        primary, secondary = generate_streams(
            by_subject, cfg, variant, args.fps, args.max_frames)
        dt = time.perf_counter() - t0
        print(f"  [{key}] PRIMARY {len(primary)} frames, "
              f"SECONDARY {len(secondary)} frames in {dt:.1f}s", flush=True)

        primary_streams[key] = primary
        secondary_streams[key] = secondary

        # Debounced views (frozen cfg.alarm params).
        p_deb = debounce_should_alarm(primary, min_dur, cooldown)
        s_deb = debounce_should_alarm(secondary, min_dur, cooldown)

        # Persist raw + debounced per-frame streams.
        stream_path = os.path.join(STREAMS_DIR, f"{variant.name}.csv")
        stream_md5[key] = write_event_stream_csv(
            stream_path, primary, secondary, p_deb, s_deb)

        # Build events + metrics (PRIMARY headline; SECONDARY for contamination).
        primary_events = build_events(primary)
        secondary_events = build_events(secondary)
        write_episode_artifacts(variant.name, primary_events)

        primary_metrics = compute_variant_regime_metrics(
            primary_events, args.fps, strategy="max_overlap",
            k_sweep_seconds=K_SWEEP_SECONDS)
        # SECONDARY reported with greedy-by-onset (robustness cross-check).
        secondary_metrics = compute_variant_regime_metrics(
            secondary_events, args.fps, strategy="greedy_onset",
            k_sweep_seconds=K_SWEEP_SECONDS)

        # Debounced PRIMARY view (deployment-realistic; same core path).
        debounced_records = _apply_debounce_to_records(primary, p_deb)
        debounced_events = build_events(debounced_records)
        debounced_metrics = compute_variant_regime_metrics(
            debounced_events, args.fps, strategy="max_overlap",
            k_sweep_seconds=K_SWEEP_SECONDS)

        contam = contamination_delta(
            primary, secondary, primary_events, secondary_events, args.fps)

        results["per_variant"][variant.name] = {
            "variant_key": key,
            "toggles": {
                "speech_filter": variant.enable_speech_filter,
                "reliability_gate": variant.enable_reliability_gate,
                "cnn": variant.enable_cnn,
            },
            "primary": primary_metrics,
            "secondary": secondary_metrics,
            "debounced_primary": debounced_metrics,
            "contamination_delta": contam,
        }

    # Observability gates (need V0..V4 pairs; skipped rows if absent).
    gate_report = check_observability_gates(
        primary_streams, secondary_streams, variant_keys)

    # Per-subject paired variant deltas (descriptive).
    deltas = paired_deltas(results)

    elapsed = time.perf_counter() - t_start

    protocol = {
        "dataset": "NTHU-DDD",
        "nthu_root": args.nthu_root,
        "subjects": subjects,
        "video_fps": args.fps,
        "segment_unit": "(subject, glasses, condition)",
        "primary_regime": "per-recording reset (fresh FrameProcessor per recording)",
        "secondary_regime": "per-subject concatenation (EXP-004 interleaved order)",
        "matching_primary": "max_overlap (order-independent, one-to-one)",
        "matching_secondary": "greedy_onset (robustness cross-check)",
        "k_sweep_seconds": K_SWEEP_SECONDS,
        "debounce": {
            "source": ("reimplemented on the video clock in "
                       "exp005_event_report.py — NOT src/alarm_controller.py "
                       "(which hard-codes time.monotonic()/datetime.now())"),
            "min_alarm_duration_s": min_dur,
            "cooldown_period_s": cooldown,
        },
        "statistics": ("descriptive only; n=4 subjects; no significance tests; "
                       "bootstrap is a labelled dispersion band, not a test"),
        "label_limitation": ("clip-condition-derived labels — clip-level alarm "
                             "behaviour, not free-driving episodes (see plan "
                             "§'Honest framing')"),
        "sklearn_used": False,
        "max_frames_per_subject": args.max_frames,
    }

    artifact = {
        "experiment_id": EXP_ID,
        "title": "Event-Level Alarm Evaluation (NTHU-DDD, V0–V4)",
        "protocol": protocol,
        "wall_clock_seconds": round(elapsed, 2),
        "event_stream_md5": stream_md5,
        "observability_gates": gate_report,
        "paired_deltas_descriptive": deltas,
        "per_variant": results["per_variant"],
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(artifact, f, indent=2, default=_json_default)
    print(f"\n[{EXP_ID}] Metrics artifact -> {METRICS_JSON}")

    write_csvs(results, variant_keys)
    make_plots(results, variant_keys)

    if args.write:
        events_block = build_events_block(
            results, variant_keys, gate_report, protocol)
        merge_events_block(events_block)
    else:
        print(f"[{EXP_ID}] --write not set; measured_results.json NOT modified.")

    _print_summary(results, gate_report, variant_keys, elapsed)
    return 0


def _json_default(o):
    """Serialize numpy scalars / NaN honestly (NaN -> null via json's allow_nan)."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def _print_summary(results: Dict, gate_report: Dict,
                   variant_keys: Sequence[str], elapsed: float) -> None:
    print("\n" + "=" * 90)
    print(f"  {EXP_ID} SUMMARY  (PRIMARY per-recording reset, pooled-micro)")
    print("=" * 90)
    hdr = (f"{'V':<3}{'name':<20}{'TP':>6}{'FP':>6}{'FN':>6}"
           f"{'Recall':>9}{'Prec':>8}{'FA/hr':>10}{'Dbnc FP':>9}")
    print(hdr)
    for k in variant_keys:
        n = VARIANTS[k].name
        pm = results["per_variant"][n]["primary"]["pooled_micro"]
        db = results["per_variant"][n]["debounced_primary"]["pooled_micro"]
        fah = pm["fa_per_hour_total"]
        fah_s = "nan" if fah != fah else f"{fah:.3f}"
        print(f"{k:<3}{n:<20}{pm['tp']:>6}{pm['fp']:>6}{pm['fn']:>6}"
              f"{pm['recall']:>9.3f}{pm['precision']:>8.3f}{fah_s:>10}"
              f"{db['fp']:>9}")
    print("-" * 90)
    print("  Observability gates (pre-registered):")
    for name, g in gate_report["gates"].items():
        extra = (f"  diff={g['n_diff_frames']} frames "
                 f"subj={g.get('subjects_with_diff')}"
                 if g["status"] in ("PASS", "FAIL") else f"  ({g.get('reason')})")
        print(f"    [{g['status']:<7}] {name}{extra}")
    print(f"  Gates all-ran-pass: {gate_report['all_ran_pass']}")
    print("=" * 90)
    print(f"Total wall-clock: {elapsed/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
