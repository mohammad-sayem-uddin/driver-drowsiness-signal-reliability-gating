"""
Unit Tests for EXP-005 Event-Level Alarm Evaluation core (evaluation/event_metrics.py)
=======================================================================================
Covers the pure metric core end-to-end with hand-computed expected values:

  * Event construction   — _contiguous_runs, build_gt_episodes, build_alarm_events,
                           build_segments, and the Segment wrappers.
  * Event matching       — _overlap_frames, match_events (both strategies), the
                           one-to-one cases called out in the EXP-005 plan
                           (alarm spanning two episodes; episode with two alarms;
                           alarm starting in alert extending into an episode),
                           tolerance clamping, and unknown-strategy error.
  * Latency              — match_latencies (signed, negatives kept) and
                           latency_stats (median / IQR / min / max / mean, empty→NaN).
  * FA / hour            — hours_from_frames, fa_per_hour (arithmetic, zero-exposure
                           NaN, fps<=0 error).
  * Metrics              — event_metrics_from_counts / _from_matches (rates and
                           zero-denominator guards).
  * Determinism          — bootstrap_ci reproducibility under a fixed seed, and
                           match_events order-independence.

The module under test is deterministic and dependency-light, so every expected
value here is computed by hand rather than by re-deriving it from the module.

Execution:
    python3 -m unittest tests/test_event_metrics.py
"""

import math
import unittest

from evaluation.event_metrics import (
    FrameRecord,
    Segment,
    GTEpisode,
    AlarmEvent,
    build_segments,
    _contiguous_runs,
    build_gt_episodes,
    build_alarm_events,
    gt_episodes_for_segment,
    alarm_events_for_segment,
    _overlap_frames,
    match_events,
    event_metrics_from_counts,
    event_metrics_from_matches,
    match_latencies,
    latency_stats,
    hours_from_frames,
    fa_per_hour,
    bootstrap_ci,
)
from evaluation.nthu_ground_truth import LABEL_DROWSY, LABEL_ALERT

import numpy as np


# ── helpers ─────────────────────────────────────────────────────────────────

def _ts(n, dt=0.1, t0=0.0):
    """A monotonic timestamp grid: t[i] = t0 + i*dt."""
    return [t0 + i * dt for i in range(n)]


def _frame(subject, glasses, condition, idx, ts, label, alarm, **kw):
    return FrameRecord(
        subject=subject, glasses=glasses, condition=condition,
        frame_index=idx, ts=ts, label=label, should_alarm=alarm, **kw,
    )


# ── _contiguous_runs ─────────────────────────────────────────────────────────

class TestContiguousRuns(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_contiguous_runs(np.array([], dtype=bool)), [])

    def test_all_false(self):
        self.assertEqual(_contiguous_runs(np.array([False, False, False])), [])

    def test_all_true_single_run(self):
        self.assertEqual(_contiguous_runs(np.array([True, True, True])), [(0, 2)])

    def test_single_element_runs(self):
        # True, False, True → two length-1 runs at the ends.
        self.assertEqual(_contiguous_runs(np.array([True, False, True])),
                         [(0, 0), (2, 2)])

    def test_multiple_runs_interior(self):
        mask = np.array([False, True, True, False, True, False, False, True])
        self.assertEqual(_contiguous_runs(mask), [(1, 2), (4, 4), (7, 7)])

    def test_run_touching_both_boundaries(self):
        mask = np.array([True, False, False, True])
        self.assertEqual(_contiguous_runs(mask), [(0, 0), (3, 3)])


# ── build_gt_episodes ──────────────────────────────────────────────────────

class TestBuildGtEpisodes(unittest.TestCase):
    def test_basic_run_geometry(self):
        # labels: alert, drowsy, drowsy, drowsy, alert  → one episode idx 1..3
        labels = [LABEL_ALERT, LABEL_DROWSY, LABEL_DROWSY, LABEL_DROWSY, LABEL_ALERT]
        ts = _ts(5, dt=0.5)  # 0.0, 0.5, 1.0, 1.5, 2.0
        eps = build_gt_episodes(labels, ts, subject="s1", glasses="g", condition="c")
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertEqual((ep.start_idx, ep.end_idx), (1, 3))
        self.assertEqual(ep.n_frames, 3)                 # e - s + 1
        self.assertAlmostEqual(ep.start_ts, 0.5)
        self.assertAlmostEqual(ep.end_ts, 1.5)
        self.assertAlmostEqual(ep.duration, 1.0)         # end_ts - start_ts
        self.assertEqual((ep.subject, ep.glasses, ep.condition), ("s1", "g", "c"))

    def test_single_frame_episode_zero_duration(self):
        labels = [LABEL_ALERT, LABEL_DROWSY, LABEL_ALERT]
        ts = _ts(3)
        eps = build_gt_episodes(labels, ts)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0].n_frames, 1)
        self.assertEqual(eps[0].duration, 0.0)

    def test_two_episodes(self):
        labels = [LABEL_DROWSY, LABEL_DROWSY, LABEL_ALERT, LABEL_DROWSY]
        ts = _ts(4)
        eps = build_gt_episodes(labels, ts)
        self.assertEqual([(e.start_idx, e.end_idx) for e in eps], [(0, 1), (3, 3)])

    def test_empty_input(self):
        self.assertEqual(build_gt_episodes([], []), [])

    def test_no_positive_frames(self):
        self.assertEqual(build_gt_episodes([LABEL_ALERT] * 4, _ts(4)), [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_gt_episodes([LABEL_DROWSY, LABEL_DROWSY], _ts(3))

    def test_custom_positive_label(self):
        # positive_label overridable — treat 0 (alert) as the positive class.
        labels = [LABEL_DROWSY, LABEL_ALERT, LABEL_ALERT]
        eps = build_gt_episodes(labels, _ts(3), positive_label=LABEL_ALERT)
        self.assertEqual([(e.start_idx, e.end_idx) for e in eps], [(1, 2)])


# ── build_alarm_events ─────────────────────────────────────────────────────

class TestBuildAlarmEvents(unittest.TestCase):
    def test_basic_geometry_and_defaults(self):
        alarms = [False, True, True, False]
        ts = _ts(4, dt=0.25)  # 0.0, 0.25, 0.5, 0.75
        evs = build_alarm_events(alarms, ts, subject="s", glasses="g", condition="c")
        self.assertEqual(len(evs), 1)
        ev = evs[0]
        self.assertEqual((ev.start_idx, ev.end_idx), (1, 2))
        self.assertEqual(ev.n_frames, 2)
        self.assertAlmostEqual(ev.onset_ts, 0.25)
        self.assertAlmostEqual(ev.offset_ts, 0.5)
        self.assertAlmostEqual(ev.duration, 0.25)
        # Missing channels default to zero / False.
        self.assertEqual(ev.peak_alarm_level, 0)
        self.assertFalse(ev.any_cnn_override)
        self.assertFalse(ev.any_alert_suppressed)
        self.assertFalse(ev.any_face_lost_critical)

    def test_channel_aggregation(self):
        alarms = [True, True, True]
        ts = _ts(3)
        evs = build_alarm_events(
            alarms, ts,
            alarm_level=[1, 3, 2],                 # max = 3
            cnn_override=[False, True, False],      # any = True
            alert_suppressed=[False, False, False], # any = False
            face_lost_critical=[False, False, True],# any = True
        )
        self.assertEqual(len(evs), 1)
        ev = evs[0]
        self.assertEqual(ev.peak_alarm_level, 3)
        self.assertTrue(ev.any_cnn_override)
        self.assertFalse(ev.any_alert_suppressed)
        self.assertTrue(ev.any_face_lost_critical)

    def test_two_events(self):
        alarms = [True, False, True, True]
        evs = build_alarm_events(alarms, _ts(4))
        self.assertEqual([(e.start_idx, e.end_idx) for e in evs], [(0, 0), (2, 3)])

    def test_empty_and_none(self):
        self.assertEqual(build_alarm_events([], []), [])
        self.assertEqual(build_alarm_events([False, False], _ts(2)), [])

    def test_ts_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_alarm_events([True, True], _ts(3))

    def test_channel_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_alarm_events([True, True], _ts(2), alarm_level=[1])
        with self.assertRaises(ValueError):
            build_alarm_events([True, True], _ts(2), cnn_override=[True])
        with self.assertRaises(ValueError):
            build_alarm_events([True, True], _ts(2), alert_suppressed=[True, True, True])
        with self.assertRaises(ValueError):
            build_alarm_events([True, True], _ts(2), face_lost_critical=[True])


# ── build_segments & Segment wrappers ────────────────────────────────────────

class TestBuildSegments(unittest.TestCase):
    def test_bucketing_by_full_triple(self):
        # Same subject+condition, different glasses → two distinct recordings.
        frames = [
            _frame("s1", "none", "c1", 0, 0.0, LABEL_ALERT, False),
            _frame("s1", "glasses", "c1", 0, 0.0, LABEL_ALERT, False),
            _frame("s1", "none", "c1", 1, 0.1, LABEL_DROWSY, True),
        ]
        segs = build_segments(frames)
        self.assertEqual(len(segs), 2)
        self.assertEqual([s.key for s in segs],
                         [("s1", "glasses", "c1"), ("s1", "none", "c1")])  # sorted

    def test_frames_sorted_by_index_stable(self):
        frames = [
            _frame("s", "g", "c", 2, 0.2, LABEL_ALERT, False),
            _frame("s", "g", "c", 0, 0.0, LABEL_ALERT, False),
            _frame("s", "g", "c", 1, 0.1, LABEL_DROWSY, True),
        ]
        segs = build_segments(frames)
        self.assertEqual(len(segs), 1)
        self.assertEqual([f.frame_index for f in segs[0].frames], [0, 1, 2])
        self.assertEqual(len(segs[0]), 3)

    def test_segment_wrappers_delegate_to_core(self):
        frames = [
            _frame("s", "g", "c", 0, 0.0, LABEL_ALERT, False, alarm_level=0),
            _frame("s", "g", "c", 1, 0.1, LABEL_DROWSY, True, alarm_level=2,
                   cnn_override=True),
            _frame("s", "g", "c", 2, 0.2, LABEL_DROWSY, True, alarm_level=1),
        ]
        seg = build_segments(frames)[0]
        eps = gt_episodes_for_segment(seg)
        evs = alarm_events_for_segment(seg)
        self.assertEqual([(e.start_idx, e.end_idx) for e in eps], [(1, 2)])
        self.assertEqual([(e.start_idx, e.end_idx) for e in evs], [(1, 2)])
        self.assertEqual(evs[0].peak_alarm_level, 2)
        self.assertTrue(evs[0].any_cnn_override)


# ── _overlap_frames ──────────────────────────────────────────────────────────

class TestOverlapFrames(unittest.TestCase):
    def test_partial_overlap(self):
        # spans [0..4] and [2..6] share indices 2,3,4 → 3 frames.
        self.assertEqual(_overlap_frames(0, 4, 2, 6), 3)

    def test_identical_spans(self):
        self.assertEqual(_overlap_frames(1, 5, 1, 5), 5)

    def test_nested_span(self):
        self.assertEqual(_overlap_frames(0, 10, 3, 5), 3)

    def test_single_frame_overlap(self):
        # [0..3] and [3..5] share only index 3.
        self.assertEqual(_overlap_frames(0, 3, 3, 5), 1)

    def test_disjoint_returns_zero(self):
        self.assertEqual(_overlap_frames(0, 2, 4, 6), 0)

    def test_touching_but_disjoint(self):
        # [0..2] and [3..5] touch but do not share an index.
        self.assertEqual(_overlap_frames(0, 2, 3, 5), 0)

    def test_symmetry(self):
        self.assertEqual(_overlap_frames(0, 4, 2, 6), _overlap_frames(2, 6, 0, 4))


# ── match_events ─────────────────────────────────────────────────────────────

def _ep(start, end, ts_start=0.0, ts_end=0.0):
    return GTEpisode(subject="s", glasses="g", condition="c",
                     start_idx=start, end_idx=end,
                     start_ts=ts_start, end_ts=ts_end,
                     duration=ts_end - ts_start, n_frames=end - start + 1)


def _al(start, end, onset=0.0, offset=0.0):
    return AlarmEvent(subject="s", glasses="g", condition="c",
                      start_idx=start, end_idx=end,
                      onset_ts=onset, offset_ts=offset,
                      duration=offset - onset, n_frames=end - start + 1,
                      peak_alarm_level=0, any_cnn_override=False,
                      any_alert_suppressed=False, any_face_lost_critical=False)


class TestMatchEvents(unittest.TestCase):
    def test_simple_one_to_one(self):
        eps = [_ep(0, 4), _ep(10, 14)]
        als = [_al(1, 5), _al(11, 13)]
        m = match_events(eps, als)
        self.assertEqual([(ei, ai) for ei, ai, _ in m.pairs], [(0, 0), (1, 1)])
        self.assertEqual(m.unmatched_episodes, ())
        self.assertEqual(m.unmatched_alarms, ())
        self.assertEqual(m.min_overlap_frames, 1)

    def test_largest_overlap_wins(self):
        # a0 overlaps ep0 by 2 and ep1 by 4 → binds ep1; ep0 becomes FN.
        eps = [_ep(0, 2), _ep(4, 9)]
        als = [_al(1, 7)]
        m = match_events(eps, als, strategy="max_overlap")
        self.assertEqual([(ei, ai, ov) for ei, ai, ov in m.pairs], [(1, 0, 4)])
        self.assertEqual(m.unmatched_episodes, (0,))
        self.assertEqual(m.unmatched_alarms, ())

    def test_alarm_spanning_two_episodes_is_one_to_one(self):
        # One long alarm overlaps two episodes; only one pair may form (FN on other).
        eps = [_ep(0, 2), _ep(5, 8)]
        als = [_al(1, 6)]  # overlaps ep0 by 2 (idx 1,2), ep1 by 2 (idx 5,6)
        m = match_events(eps, als)
        self.assertEqual(len(m.pairs), 1)                 # strictly one-to-one
        # Tie on overlap (2 vs 2) → deterministic (episode_idx, alarm_idx) → ep0.
        self.assertEqual(m.pairs[0][:2], (0, 0))
        self.assertEqual(m.unmatched_episodes, (1,))       # the other episode is a miss
        self.assertEqual(m.unmatched_alarms, ())

    def test_episode_with_two_alarms_is_one_to_one(self):
        # One episode overlaps two alarms; one alarm is a false alarm (FP).
        eps = [_ep(0, 9)]
        als = [_al(1, 3), _al(5, 8)]  # overlap 3 and 4 → a1 wins, a0 is FP
        m = match_events(eps, als)
        self.assertEqual(len(m.pairs), 1)
        self.assertEqual(m.pairs[0][:2], (0, 1))
        self.assertEqual(m.unmatched_alarms, (0,))
        self.assertEqual(m.unmatched_episodes, ())

    def test_alarm_starting_in_alert_extending_into_episode(self):
        # Alarm onset precedes the labelled span but overlaps it → a valid match.
        eps = [_ep(5, 10)]
        als = [_al(2, 7)]  # overlap indices 5,6,7 → 3 frames
        m = match_events(eps, als)
        self.assertEqual([(ei, ai, ov) for ei, ai, ov in m.pairs], [(0, 0, 3)])

    def test_tolerance_threshold_excludes_thin_overlap(self):
        # Overlap of exactly 1 frame is excluded when tolerance=2.
        eps = [_ep(0, 3)]
        als = [_al(3, 6)]  # overlap = 1 (only index 3)
        m = match_events(eps, als, tolerance=2)
        self.assertEqual(m.pairs, ())
        self.assertEqual(m.unmatched_episodes, (0,))
        self.assertEqual(m.unmatched_alarms, (0,))
        self.assertEqual(m.min_overlap_frames, 2)

    def test_tolerance_clamped_to_one(self):
        # tolerance < 1 is clamped up to 1 (any nonzero overlap eligible).
        eps = [_ep(0, 3)]
        als = [_al(3, 6)]  # overlap = 1
        m = match_events(eps, als, tolerance=0)
        self.assertEqual(len(m.pairs), 1)
        self.assertEqual(m.min_overlap_frames, 1)

    def test_no_overlap_all_unmatched(self):
        eps = [_ep(0, 2)]
        als = [_al(10, 12)]
        m = match_events(eps, als)
        self.assertEqual(m.pairs, ())
        self.assertEqual(m.unmatched_episodes, (0,))
        self.assertEqual(m.unmatched_alarms, (0,))

    def test_empty_inputs(self):
        self.assertEqual(match_events([], []).pairs, ())
        self.assertEqual(match_events([_ep(0, 2)], []).unmatched_episodes, (0,))
        self.assertEqual(match_events([], [_al(0, 2)]).unmatched_alarms, (0,))

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            match_events([_ep(0, 2)], [_al(0, 2)], strategy="nonsense")

    def test_greedy_onset_binds_earliest_overlapping_episode(self):
        # ep0 has a SMALL overlap, ep1 a LARGER one; the strategies disagree.
        #   greedy_onset -> earliest overlapping episode in list order = ep0
        #   max_overlap  -> globally largest overlap = ep1
        eps = [_ep(4, 5), _ep(6, 10)]
        als = [_al(0, 10)]  # ep0 overlap = 2, ep1 overlap = 5
        m_greedy = match_events(eps, als, strategy="greedy_onset")
        m_max = match_events(eps, als, strategy="max_overlap")
        self.assertEqual(m_greedy.pairs[0][:2], (0, 0))    # earliest episode
        self.assertEqual(m_max.pairs[0][:2], (1, 0))       # largest overlap
        self.assertEqual(m_greedy.strategy, "greedy_onset")

    def test_greedy_onset_sweeps_alarms_by_onset(self):
        eps = [_ep(0, 3), _ep(6, 12)]
        als = [_al(7, 11, onset=0.7), _al(1, 2, onset=0.1)]  # a1 has earlier onset
        m = match_events(eps, als, strategy="greedy_onset")
        # Earlier-onset alarm (a1, idx 2..? here idx1) binds ep0; later binds ep1.
        pairs = sorted((ei, ai) for ei, ai, _ in m.pairs)
        self.assertEqual(pairs, [(0, 1), (1, 0)])
        self.assertEqual(m.unmatched_episodes, ())
        self.assertEqual(m.unmatched_alarms, ())

    def test_pairs_sorted_by_episode_then_alarm(self):
        eps = [_ep(0, 2), _ep(4, 6), _ep(8, 10)]
        als = [_al(8, 10), _al(0, 2), _al(4, 6)]
        m = match_events(eps, als)
        self.assertEqual([(ei, ai) for ei, ai, _ in m.pairs],
                         [(0, 1), (1, 2), (2, 0)])

    def test_match_order_independence(self):
        # Reordering the alarm list must not change the matched pairing (by index
        # identity of the underlying events).
        eps = [_ep(0, 5), _ep(10, 15)]
        als_a = [_al(1, 4), _al(11, 14)]
        als_b = [_al(11, 14), _al(1, 4)]
        m_a = match_events(eps, als_a)
        m_b = match_events(eps, als_b)
        # In als_a: ep0->al0, ep1->al1. In als_b the alarms swap indices, so the
        # same physical pairing is ep0->al1, ep1->al0.
        self.assertEqual([(ei, ai) for ei, ai, _ in m_a.pairs], [(0, 0), (1, 1)])
        self.assertEqual([(ei, ai) for ei, ai, _ in m_b.pairs], [(0, 1), (1, 0)])


# ── event metrics ────────────────────────────────────────────────────────────

class TestEventMetricsFromCounts(unittest.TestCase):
    def test_rates(self):
        m = event_metrics_from_counts(tp=3, fp=1, fn=1)
        self.assertAlmostEqual(m.recall, 0.75)      # 3 / (3+1)
        self.assertAlmostEqual(m.precision, 0.75)   # 3 / (3+1)
        self.assertAlmostEqual(m.miss_rate, 0.25)   # 1 / (3+1)
        self.assertAlmostEqual(m.f1, 0.75)          # 2*.75*.75/1.5
        self.assertAlmostEqual(m.recall + m.miss_rate, 1.0)

    def test_perfect(self):
        m = event_metrics_from_counts(tp=5, fp=0, fn=0)
        self.assertEqual((m.recall, m.precision, m.f1, m.miss_rate),
                         (1.0, 1.0, 1.0, 0.0))

    def test_all_zero_guards(self):
        m = event_metrics_from_counts(tp=0, fp=0, fn=0)
        self.assertEqual((m.recall, m.precision, m.miss_rate, m.f1),
                         (0.0, 0.0, 0.0, 0.0))

    def test_only_false_positives(self):
        m = event_metrics_from_counts(tp=0, fp=4, fn=0)
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.f1, 0.0)

    def test_only_false_negatives(self):
        m = event_metrics_from_counts(tp=0, fp=0, fn=4)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.miss_rate, 1.0)


class TestEventMetricsFromMatches(unittest.TestCase):
    def test_aggregation_over_segments(self):
        # Segment A: 1 pair, 1 missed episode, 1 false alarm.
        a = match_events([_ep(0, 4), _ep(20, 22)], [_al(1, 3), _al(50, 52)])
        # Segment B: 1 clean pair.
        b = match_events([_ep(0, 4)], [_al(0, 4)])
        m = event_metrics_from_matches([a, b])
        self.assertEqual((m.tp, m.fp, m.fn), (2, 1, 1))
        self.assertAlmostEqual(m.recall, 2 / 3)
        self.assertAlmostEqual(m.precision, 2 / 3)

    def test_empty_matches(self):
        m = event_metrics_from_matches([])
        self.assertEqual((m.tp, m.fp, m.fn), (0, 0, 0))


# ── latency ──────────────────────────────────────────────────────────────────

class TestLatency(unittest.TestCase):
    def test_signed_latency_positive_and_negative(self):
        # ep0 starts at t=1.0, its alarm onset at t=1.5 → +0.5 (late detection).
        # ep1 starts at t=5.0, its alarm onset at t=4.5 → -0.5 (alarm precedes).
        # Alarm index spans must overlap the episode index spans to match at all
        # (overlap is by frame index, independent of the timestamps).
        eps = [_ep(10, 14, ts_start=1.0, ts_end=1.4),
               _ep(50, 54, ts_start=5.0, ts_end=5.4)]
        als = [_al(12, 16, onset=1.5, offset=1.8),
               _al(52, 56, onset=4.5, offset=4.8)]
        m = match_events(eps, als)
        lat = match_latencies(eps, als, m)
        self.assertEqual(len(lat), 2)
        # Pairs are sorted by episode index, so latencies align with ep0, ep1.
        self.assertAlmostEqual(lat[0], 0.5)
        self.assertAlmostEqual(lat[1], -0.5)   # negative kept, not clipped

    def test_latency_stats_hand_computed(self):
        stats = latency_stats([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats.n, 5)
        self.assertAlmostEqual(stats.median, 2.0)
        self.assertAlmostEqual(stats.mean, 2.0)
        self.assertAlmostEqual(stats.min, 0.0)
        self.assertAlmostEqual(stats.max, 4.0)
        self.assertAlmostEqual(stats.q1, 1.0)      # linear-interp 25th pct
        self.assertAlmostEqual(stats.q3, 3.0)      # linear-interp 75th pct
        self.assertAlmostEqual(stats.iqr, 2.0)

    def test_latency_stats_single_value(self):
        stats = latency_stats([2.5])
        self.assertEqual(stats.n, 1)
        for v in (stats.median, stats.q1, stats.q3, stats.min, stats.max, stats.mean):
            self.assertAlmostEqual(v, 2.5)
        self.assertAlmostEqual(stats.iqr, 0.0)

    def test_latency_stats_empty_is_nan(self):
        stats = latency_stats([])
        self.assertEqual(stats.n, 0)
        for v in (stats.median, stats.q1, stats.q3, stats.iqr,
                  stats.min, stats.max, stats.mean):
            self.assertTrue(math.isnan(v))

    def test_no_pairs_yields_empty_latencies(self):
        m = match_events([_ep(0, 2)], [_al(10, 12)])  # no overlap
        self.assertEqual(match_latencies([_ep(0, 2)], [_al(10, 12)], m), [])


# ── FA / hour ────────────────────────────────────────────────────────────────

class TestFalseAlarmRate(unittest.TestCase):
    def test_hours_from_frames_arithmetic(self):
        # 3600 frames at 1 fps = 3600 s = 1.0 h.
        self.assertAlmostEqual(hours_from_frames(3600, 1.0), 1.0)
        # 108000 frames at 30 fps = 3600 s = 1.0 h.
        self.assertAlmostEqual(hours_from_frames(108000, 30.0), 1.0)
        self.assertAlmostEqual(hours_from_frames(0, 30.0), 0.0)

    def test_hours_from_frames_bad_fps(self):
        with self.assertRaises(ValueError):
            hours_from_frames(100, 0.0)
        with self.assertRaises(ValueError):
            hours_from_frames(100, -5.0)

    def test_fa_per_hour_arithmetic(self):
        self.assertAlmostEqual(fa_per_hour(10, 2.0), 5.0)
        self.assertAlmostEqual(fa_per_hour(0, 3.0), 0.0)

    def test_fa_per_hour_zero_exposure_is_nan(self):
        self.assertTrue(math.isnan(fa_per_hour(3, 0.0)))
        self.assertTrue(math.isnan(fa_per_hour(3, -1.0)))


# ── bootstrap determinism ────────────────────────────────────────────────────

class TestBootstrapCI(unittest.TestCase):
    def _pooled_fa(self, units):
        # units are (fp, hours) pairs → pooled FA/hour.
        return sum(fp for fp, _ in units) / sum(h for _, h in units)

    def test_point_estimate(self):
        units = [(3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)]  # 18 / 4 = 4.5
        ci = bootstrap_ci(units, self._pooled_fa, n_boot=500, seed=42)
        self.assertAlmostEqual(ci.point, 4.5)
        self.assertEqual(ci.seed, 42)
        self.assertEqual(ci.n_boot, 500)
        self.assertAlmostEqual(ci.ci_level, 0.95)

    def test_reproducible_under_same_seed(self):
        units = [(3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)]
        a = bootstrap_ci(units, self._pooled_fa, n_boot=500, seed=42)
        b = bootstrap_ci(units, self._pooled_fa, n_boot=500, seed=42)
        self.assertEqual((a.lo, a.hi, a.mean, a.std), (b.lo, b.hi, b.mean, b.std))

    def test_different_seed_may_differ_but_stays_bounded(self):
        units = [(3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)]
        a = bootstrap_ci(units, self._pooled_fa, n_boot=500, seed=42)
        c = bootstrap_ci(units, self._pooled_fa, n_boot=500, seed=7)
        # Whatever the seed, the band is ordered and brackets the resample mean.
        self.assertLessEqual(a.lo, a.hi)
        self.assertLessEqual(c.lo, c.hi)
        self.assertLessEqual(a.lo, a.mean)
        self.assertLessEqual(a.mean, a.hi)

    def test_single_subject_band_collapses_to_point(self):
        # With one unit, every resample is identical → zero-width band.
        units = [(5, 1.0)]
        ci = bootstrap_ci(units, self._pooled_fa, n_boot=200, seed=42)
        self.assertAlmostEqual(ci.point, 5.0)
        self.assertAlmostEqual(ci.lo, 5.0)
        self.assertAlmostEqual(ci.hi, 5.0)
        self.assertAlmostEqual(ci.std, 0.0)

    def test_empty_units_is_nan(self):
        ci = bootstrap_ci([], self._pooled_fa, n_boot=100, seed=42)
        for v in (ci.point, ci.lo, ci.hi, ci.mean, ci.std):
            self.assertTrue(math.isnan(v))
        self.assertEqual(ci.seed, 42)


# ── end-to-end integration on a synthetic recording ──────────────────────────

class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_on_one_segment(self):
        # 10 frames @ 10 fps. Drowsy episode over idx 3..6. Alarm over idx 4..7
        # (one frame late onset, extends one frame past the episode).
        n = 10
        ts = _ts(n, dt=0.1)
        labels = [LABEL_ALERT] * n
        for i in range(3, 7):
            labels[i] = LABEL_DROWSY
        alarms = [False] * n
        for i in range(4, 8):
            alarms[i] = True
        frames = [_frame("s1", "g", "c", i, ts[i], labels[i], alarms[i])
                  for i in range(n)]

        seg = build_segments(frames)[0]
        eps = gt_episodes_for_segment(seg)
        evs = alarm_events_for_segment(seg)
        self.assertEqual([(e.start_idx, e.end_idx) for e in eps], [(3, 6)])
        self.assertEqual([(e.start_idx, e.end_idx) for e in evs], [(4, 7)])

        m = match_events(eps, evs)
        self.assertEqual(len(m.pairs), 1)          # detected
        metrics = event_metrics_from_matches([m])
        self.assertEqual((metrics.tp, metrics.fp, metrics.fn), (1, 0, 0))

        lat = match_latencies(eps, evs, m)
        self.assertEqual(len(lat), 1)
        # onset idx 4 (t=0.4) - episode start idx 3 (t=0.3) = +0.1 s.
        self.assertAlmostEqual(lat[0], 0.1)

        # No false alarms → FA/hour is 0 over the recording exposure.
        hours = hours_from_frames(n, fps=10.0)
        self.assertAlmostEqual(fa_per_hour(metrics.fp, hours), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
