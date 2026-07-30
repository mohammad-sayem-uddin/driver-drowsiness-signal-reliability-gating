"""
EXP-005 — Event-Level Alarm Evaluation: pure metric core
========================================================
This module is the **pure, dependency-light, fully unit-testable core** for the
event-level alarm evaluation (EXP-005). It computes everything the experiment
needs *from arrays already produced elsewhere* — it performs **no I/O, holds no
pipeline logic, imports nothing from ``src/``**, and is **fully deterministic**
(no wall-clock, no unseeded randomness). The orchestrator
(``evaluation/exp005_event_report.py``) is responsible for running the frozen
pipeline, capturing per-frame streams, calling into this module, and writing
artifacts; none of that lives here.

Why event-level (see the EXP-005 plan). EXP-004's frame-level ROC over the
continuous ``fatigue_score`` is structurally blind to the CNN arm and the
reliability gate: both only flip the per-frame boolean ``should_alarm``, never
the swept score. EXP-005 therefore evaluates the system where it was designed to
act — **alarm events** — via:

    * GT drowsy episodes  = maximal runs of ``label == 1`` within one recording.
    * Alarm events        = maximal runs of ``should_alarm == True`` within one
                            recording (rising edge ``False→True`` to falling
                            edge ``True→False``).
    * Matching            = temporal-overlap association between episodes and
                            alarms, one-to-one.
    * Metrics             = event recall / precision / miss-rate, FA/hour,
                            detection latency, plus a subject-stratified
                            descriptive bootstrap dispersion band.

The recording (segment) unit is the full ``(subject, glasses, condition)``
triple — episodes and alarm events never cross a recording boundary (plan
"Methodology"). Overlap is measured in **shared frame indices** because GT
episodes and alarm events are built on the *same* per-segment frame grid, which
makes matching integer-exact, order-independent, and hand-checkable in tests.

No sklearn (absent in the project ``.venv``, matching EXP-002/003/004); every
statistic is hand-rolled numpy with the standard definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the frozen label convention — never redefine it here.
from evaluation.nthu_ground_truth import LABEL_DROWSY, LABEL_ALERT


# ── Input record ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FrameRecord:
    """One captured per-frame decision row for the event evaluation.

    This is the *input* unit the orchestrator produces (one per processed
    frame) and hands to :func:`build_segments`. It carries only what event
    construction needs; it holds no pixels and no pipeline state. ``ts`` is the
    local video-clock timestamp (seconds) for the frame within its recording
    (``local_frame_index / fps``), monotonic within a recording.
    """
    subject: str
    glasses: str
    condition: str
    frame_index: int          # local frame counter within the recording
    ts: float                 # local video-clock timestamp (seconds)
    label: int                # LABEL_ALERT (0) or LABEL_DROWSY (1)
    should_alarm: bool
    alarm_level: int = 0
    alert_suppressed: bool = False
    cnn_override: bool = False
    face_lost_critical: bool = False


# ── Structural units ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Segment:
    """A single recording — the ``(subject, glasses, condition)`` unit.

    ``frames`` are ordered by ``frame_index`` (ascending, stable). All index
    fields on episodes / alarm events built from this segment refer to positions
    in ``frames``.
    """
    subject: str
    glasses: str
    condition: str
    frames: Tuple[FrameRecord, ...]

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.glasses, self.condition)

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class GTEpisode:
    """A maximal contiguous run of drowsy-labelled frames within one recording."""
    subject: str
    glasses: str
    condition: str
    start_idx: int            # first frame position (inclusive) in the segment
    end_idx: int              # last frame position (inclusive) in the segment
    start_ts: float
    end_ts: float
    duration: float           # end_ts - start_ts (span; single-frame run = 0.0)
    n_frames: int


@dataclass(frozen=True)
class AlarmEvent:
    """A maximal contiguous run of ``should_alarm == True`` within one recording."""
    subject: str
    glasses: str
    condition: str
    start_idx: int            # first frame position (inclusive) in the segment
    end_idx: int              # last frame position (inclusive) in the segment
    onset_ts: float
    offset_ts: float
    duration: float           # offset_ts - onset_ts (span; single-frame run = 0.0)
    n_frames: int
    peak_alarm_level: int     # max alarm_level over the run
    any_cnn_override: bool    # any cnn_override True over the run
    any_alert_suppressed: bool
    any_face_lost_critical: bool


@dataclass(frozen=True)
class MatchResult:
    """One-to-one association between GT episodes and alarm events.

    Indices are positions into the ``episodes`` / ``alarms`` lists passed to
    :func:`match_events`. ``pairs`` are ``(episode_i, alarm_i, overlap_frames)``.
    Unmatched episodes are missed events (FN); unmatched alarms are false-alarm
    events (FP).
    """
    pairs: Tuple[Tuple[int, int, int], ...]
    matched_episodes: Tuple[int, ...]
    matched_alarms: Tuple[int, ...]
    unmatched_episodes: Tuple[int, ...]   # FN
    unmatched_alarms: Tuple[int, ...]     # FP
    strategy: str
    min_overlap_frames: int


@dataclass(frozen=True)
class EventMetrics:
    """Event-level confusion counts and derived rates (per fold / pooled)."""
    tp: int
    fp: int
    fn: int
    recall: float             # TP / (TP + FN)  — sensitivity / detection retained
    precision: float          # TP / (TP + FP)
    miss_rate: float          # FN / (TP + FN)  == 1 - recall
    f1: float


@dataclass(frozen=True)
class LatencyStats:
    """Descriptive latency summary (seconds) over matched pairs.

    Latency = ``alarm_onset_ts - episode_start_ts`` on matched pairs. Negative
    latency (alarm before the labelled span begins) is reported honestly, never
    clipped. Empty input yields NaNs (deterministic).
    """
    n: int
    median: float
    q1: float
    q3: float
    iqr: float
    min: float
    max: float
    mean: float


@dataclass(frozen=True)
class BootstrapCI:
    """Subject-stratified descriptive bootstrap dispersion band.

    NOT a hypothesis test: with n=4 subjects this bounds only within-sample
    dispersion, not population generalization (plan "Statistics"). Deterministic
    via a fixed seed.
    """
    point: float
    lo: float
    hi: float
    mean: float
    std: float
    n_boot: int
    ci_level: float
    seed: int


# ── Segmentation ──────────────────────────────────────────────────────────────

def build_segments(frames: Sequence[FrameRecord]) -> List[Segment]:
    """Group per-frame records into recordings on ``(subject, glasses, condition)``.

    The full triple is the true recording unit (the frozen ground-truth sort key
    omits ``glasses`` and would interleave distinct recordings — plan
    "Methodology"). Segments are returned in a deterministic order (sorted by the
    triple); frames within a segment are ordered by ``frame_index`` ascending
    (stable). Pure and side-effect free.
    """
    buckets: "Dict[Tuple[str, str, str], List[FrameRecord]]" = {}
    order: List[Tuple[str, str, str]] = []
    for fr in frames:
        key = (fr.subject, fr.glasses, fr.condition)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(fr)

    segments: List[Segment] = []
    for key in sorted(order):
        rows = sorted(buckets[key], key=lambda r: r.frame_index)
        subject, glasses, condition = key
        segments.append(Segment(
            subject=subject, glasses=glasses, condition=condition,
            frames=tuple(rows),
        ))
    return segments


# ── Run detection (internal) ──────────────────────────────────────────────────

def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive ``(start_idx, end_idx)`` spans of every ``True`` run in a
    boolean array, left-to-right. Deterministic; ``[]`` for an all-False or
    empty array."""
    n = int(mask.shape[0])
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _as_float_ts(ts: Sequence[float]) -> np.ndarray:
    return np.asarray(ts, dtype=np.float64)


# ── Episode & alarm-event construction ────────────────────────────────────────

def build_gt_episodes(labels: Sequence[int], ts: Sequence[float],
                      subject: str = "", glasses: str = "", condition: str = "",
                      positive_label: int = LABEL_DROWSY) -> List[GTEpisode]:
    """Maximal contiguous runs of ``label == positive_label`` as GT episodes.

    ``labels`` and ``ts`` are the per-frame arrays of ONE recording (equal
    length, ``ts`` monotonic within the recording). ``duration`` is the span
    ``end_ts - start_ts`` (a single-frame episode has duration ``0.0``);
    ``n_frames`` is the authoritative count. Pure; empty input -> ``[]``.
    """
    y = np.asarray(labels)
    t = _as_float_ts(ts)
    if y.shape[0] != t.shape[0]:
        raise ValueError("labels and ts must have equal length")
    mask = (y == positive_label)
    episodes: List[GTEpisode] = []
    for s, e in _contiguous_runs(mask):
        episodes.append(GTEpisode(
            subject=subject, glasses=glasses, condition=condition,
            start_idx=s, end_idx=e,
            start_ts=float(t[s]), end_ts=float(t[e]),
            duration=float(t[e] - t[s]), n_frames=e - s + 1,
        ))
    return episodes


def build_alarm_events(should_alarm: Sequence[bool], ts: Sequence[float],
                       alarm_level: Optional[Sequence[int]] = None,
                       alert_suppressed: Optional[Sequence[bool]] = None,
                       cnn_override: Optional[Sequence[bool]] = None,
                       face_lost_critical: Optional[Sequence[bool]] = None,
                       subject: str = "", glasses: str = "",
                       condition: str = "") -> List[AlarmEvent]:
    """Maximal contiguous runs of ``should_alarm == True`` as alarm events.

    All arrays are the per-frame arrays of ONE recording (equal length). The
    optional per-frame channels aggregate over each run: ``peak_alarm_level`` is
    the max level, and the ``any_*`` flags OR the corresponding channel across
    the run. Missing channels default to zeros / ``False``. Pure; empty input ->
    ``[]``.
    """
    a = np.asarray([bool(x) for x in should_alarm], dtype=bool)
    t = _as_float_ts(ts)
    if a.shape[0] != t.shape[0]:
        raise ValueError("should_alarm and ts must have equal length")
    n = a.shape[0]

    lvl = (np.asarray(alarm_level, dtype=np.int64) if alarm_level is not None
           else np.zeros(n, dtype=np.int64))
    supp = (np.asarray([bool(x) for x in alert_suppressed], dtype=bool)
            if alert_suppressed is not None else np.zeros(n, dtype=bool))
    cnn = (np.asarray([bool(x) for x in cnn_override], dtype=bool)
           if cnn_override is not None else np.zeros(n, dtype=bool))
    face = (np.asarray([bool(x) for x in face_lost_critical], dtype=bool)
            if face_lost_critical is not None else np.zeros(n, dtype=bool))
    for arr, name in ((lvl, "alarm_level"), (supp, "alert_suppressed"),
                      (cnn, "cnn_override"), (face, "face_lost_critical")):
        if arr.shape[0] != n:
            raise ValueError(f"{name} must match should_alarm length")

    events: List[AlarmEvent] = []
    for s, e in _contiguous_runs(a):
        events.append(AlarmEvent(
            subject=subject, glasses=glasses, condition=condition,
            start_idx=s, end_idx=e,
            onset_ts=float(t[s]), offset_ts=float(t[e]),
            duration=float(t[e] - t[s]), n_frames=e - s + 1,
            peak_alarm_level=int(lvl[s:e + 1].max()),
            any_cnn_override=bool(cnn[s:e + 1].any()),
            any_alert_suppressed=bool(supp[s:e + 1].any()),
            any_face_lost_critical=bool(face[s:e + 1].any()),
        ))
    return events


# Convenience wrappers that extract the per-frame arrays from a Segment. Kept
# thin so the array-first functions above stay the tested primitives.

def gt_episodes_for_segment(seg: Segment,
                            positive_label: int = LABEL_DROWSY) -> List[GTEpisode]:
    """GT episodes for a whole :class:`Segment` (extracts arrays, calls core)."""
    return build_gt_episodes(
        labels=[f.label for f in seg.frames],
        ts=[f.ts for f in seg.frames],
        subject=seg.subject, glasses=seg.glasses, condition=seg.condition,
        positive_label=positive_label,
    )


def alarm_events_for_segment(seg: Segment) -> List[AlarmEvent]:
    """Alarm events for a whole :class:`Segment` (extracts arrays, calls core)."""
    return build_alarm_events(
        should_alarm=[f.should_alarm for f in seg.frames],
        ts=[f.ts for f in seg.frames],
        alarm_level=[f.alarm_level for f in seg.frames],
        alert_suppressed=[f.alert_suppressed for f in seg.frames],
        cnn_override=[f.cnn_override for f in seg.frames],
        face_lost_critical=[f.face_lost_critical for f in seg.frames],
        subject=seg.subject, glasses=seg.glasses, condition=seg.condition,
    )


# ── Overlap & matching ────────────────────────────────────────────────────────

def _overlap_frames(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Shared-frame-index overlap of two inclusive index spans (0 if disjoint)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def match_events(episodes: Sequence[GTEpisode], alarms: Sequence[AlarmEvent],
                 tolerance: int = 1, strategy: str = "max_overlap") -> MatchResult:
    """Associate GT episodes with alarm events one-to-one within one recording.

    Overlap is counted in shared frame indices (episodes and alarms share the
    segment's frame grid). ``tolerance`` is the minimum overlap in **frames** for
    a pair to be eligible (``1`` = any nonzero overlap, the primary criterion;
    the plan's seconds-based ``k``-sweep is converted to frames by the caller).

    ``strategy``:
      * ``"max_overlap"`` (PRIMARY) — repeatedly bind the eligible pair with the
        largest overlap, ties broken by ``(episode_idx, alarm_idx)``. This is
        order-independent (does not depend on onset order) and one-to-one.
      * ``"greedy_onset"`` (SECONDARY) — sweep alarms by onset and bind each to
        the earliest still-unmatched episode it overlaps. Reported as a
        robustness cross-check.

    Returns a :class:`MatchResult`. TP = matched pairs, FN = unmatched episodes,
    FP = unmatched alarms. Pure and deterministic.
    """
    if tolerance < 1:
        tolerance = 1
    n_ep, n_al = len(episodes), len(alarms)
    ep_used = [False] * n_ep
    al_used = [False] * n_al
    pairs: List[Tuple[int, int, int]] = []

    if strategy == "max_overlap":
        candidates: List[Tuple[int, int, int]] = []
        for ei, ep in enumerate(episodes):
            for ai, al in enumerate(alarms):
                ov = _overlap_frames(ep.start_idx, ep.end_idx,
                                     al.start_idx, al.end_idx)
                if ov >= tolerance:
                    candidates.append((ov, ei, ai))
        # Largest overlap first; deterministic tie-break by (episode, alarm).
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        for ov, ei, ai in candidates:
            if not ep_used[ei] and not al_used[ai]:
                ep_used[ei] = True
                al_used[ai] = True
                pairs.append((ei, ai, ov))
    elif strategy == "greedy_onset":
        order = sorted(range(n_al),
                       key=lambda ai: (alarms[ai].onset_ts, alarms[ai].start_idx))
        for ai in order:
            al = alarms[ai]
            best_ei = -1
            for ei, ep in enumerate(episodes):
                if ep_used[ei]:
                    continue
                ov = _overlap_frames(ep.start_idx, ep.end_idx,
                                     al.start_idx, al.end_idx)
                if ov >= tolerance:
                    best_ei = ei
                    break   # earliest unmatched overlapping episode
            if best_ei >= 0:
                ov = _overlap_frames(episodes[best_ei].start_idx,
                                     episodes[best_ei].end_idx,
                                     al.start_idx, al.end_idx)
                ep_used[best_ei] = True
                al_used[ai] = True
                pairs.append((best_ei, ai, ov))
    else:
        raise ValueError(f"unknown matching strategy: {strategy!r}")

    # Deterministic ordering of the pair list by episode then alarm index.
    pairs.sort(key=lambda p: (p[0], p[1]))
    matched_ep = tuple(p[0] for p in pairs)
    matched_al = tuple(p[1] for p in pairs)
    unmatched_ep = tuple(i for i in range(n_ep) if not ep_used[i])
    unmatched_al = tuple(i for i in range(n_al) if not al_used[i])
    return MatchResult(
        pairs=tuple(pairs),
        matched_episodes=matched_ep,
        matched_alarms=matched_al,
        unmatched_episodes=unmatched_ep,
        unmatched_alarms=unmatched_al,
        strategy=strategy,
        min_overlap_frames=tolerance,
    )


# ── Event metrics ─────────────────────────────────────────────────────────────

def event_metrics_from_counts(tp: int, fp: int, fn: int) -> EventMetrics:
    """Event-level rates from raw counts (the defensible primitive — plan leads
    with raw counts). No event-level TN exists (open-ended negative time), so
    specificity is intentionally absent; use :func:`fa_per_hour` instead."""
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return EventMetrics(tp=int(tp), fp=int(fp), fn=int(fn),
                        recall=recall, precision=precision,
                        miss_rate=miss_rate, f1=f1)


def event_metrics_from_matches(matches: Sequence[MatchResult]) -> EventMetrics:
    """Aggregate one or more :class:`MatchResult` (e.g. per-segment within a
    fold, or pooled across subjects) into event-level metrics. TP = total
    matched pairs, FN = total unmatched episodes, FP = total unmatched alarms."""
    tp = sum(len(m.pairs) for m in matches)
    fn = sum(len(m.unmatched_episodes) for m in matches)
    fp = sum(len(m.unmatched_alarms) for m in matches)
    return event_metrics_from_counts(tp, fp, fn)


# ── Latency ───────────────────────────────────────────────────────────────────

def match_latencies(episodes: Sequence[GTEpisode], alarms: Sequence[AlarmEvent],
                    match: MatchResult) -> List[float]:
    """Per-matched-pair detection latency ``alarm_onset_ts - episode_start_ts``
    (seconds). Negative values (alarm precedes the labelled span) are kept."""
    out: List[float] = []
    for ei, ai, _ in match.pairs:
        out.append(float(alarms[ai].onset_ts - episodes[ei].start_ts))
    return out


def latency_stats(latencies: Sequence[float]) -> LatencyStats:
    """Descriptive latency summary (median / IQR / min / max / mean). Empty input
    -> all-NaN (deterministic). Uses linear-interpolated quantiles."""
    arr = np.asarray(list(latencies), dtype=np.float64)
    if arr.size == 0:
        nan = float("nan")
        return LatencyStats(n=0, median=nan, q1=nan, q3=nan, iqr=nan,
                            min=nan, max=nan, mean=nan)
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    return LatencyStats(
        n=int(arr.size),
        median=float(np.median(arr)),
        q1=q1, q3=q3, iqr=float(q3 - q1),
        min=float(arr.min()), max=float(arr.max()),
        mean=float(arr.mean()),
    )


# ── False-alarm rate ──────────────────────────────────────────────────────────

def hours_from_frames(n_frames: int, fps: float) -> float:
    """Recording exposure in hours from a frame count and the video fps. Each
    frame is treated as ``1/fps`` seconds of exposure (total-time denominator)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (n_frames / fps) / 3600.0


def fa_per_hour(fp_count: int, total_hours: float) -> float:
    """False alarms per hour = ``fp_count / total_hours``. The exposure-normalized
    false-alarm rate that replaces specificity at the event level. Returns NaN
    when exposure is zero (undefined rate, reported honestly)."""
    if total_hours <= 0:
        return float("nan")
    return fp_count / total_hours


# ── Descriptive bootstrap (subject-stratified) ────────────────────────────────

def bootstrap_ci(units: Sequence, combine: Callable[[Sequence], float],
                 n_boot: int = 2000, seed: int = 42,
                 ci_level: float = 0.95) -> BootstrapCI:
    """Subject-stratified percentile bootstrap dispersion band (descriptive only).

    ``units`` is one entry per subject (any type — e.g. ``(fp, hours)`` pairs for
    pooled FA/hour, or ``(tp, fn)`` pairs for pooled recall). ``combine`` maps a
    resampled list of units to the pooled statistic, e.g.
    ``lambda s: sum(fp for fp, _ in s) / sum(h for _, h in s)``. Subjects are
    resampled with replacement using a seeded generator, so results are fully
    reproducible. This is NOT a significance test (plan "Statistics"): with few
    subjects it bounds within-sample dispersion only.
    """
    unit_list = list(units)
    n = len(unit_list)
    if n == 0:
        nan = float("nan")
        return BootstrapCI(point=nan, lo=nan, hi=nan, mean=nan, std=nan,
                           n_boot=n_boot, ci_level=ci_level, seed=seed)
    point = float(combine(unit_list))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = float(combine([unit_list[i] for i in idx]))
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return BootstrapCI(
        point=point, lo=lo, hi=hi,
        mean=float(np.mean(stats)), std=float(np.std(stats)),
        n_boot=n_boot, ci_level=ci_level, seed=seed,
    )
