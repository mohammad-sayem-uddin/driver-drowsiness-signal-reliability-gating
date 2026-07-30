"""
Headless Smoke Test (no camera, no display)
===========================================
Repeatable, dependency-light check that the system is executable end-to-end
without hardware. It verifies:

  1. `src.main` imports and exposes `main()` (guards the syntax break that
     previously existed at main.py:455 — freeze-report precondition 2).
  2. Every core pipeline stage constructs from `SystemConfig`.
  3. One synthetic frame flows through reliability -> fusion -> state machine
     and yields a valid DriverStatus.
  4. The reliability gate is honestly 3-component (no phantom `tracking`
     field survives — freeze-report precondition 4).

This performs NO measurement and reports NO performance numbers; it only
proves the pipeline runs. Exit code 0 = pass, 1 = fail.

Run:
    python3 tests/smoke_test.py
    python3 -m unittest tests.smoke_test      # also works as a unittest
"""

import os
import sys
import unittest

# Headless/CI safety: never require a display or audio device.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class HeadlessSmokeTest(unittest.TestCase):

    def test_main_module_imports_and_has_entrypoint(self):
        import importlib
        main_mod = importlib.import_module("src.main")
        self.assertTrue(hasattr(main_mod, "main"), "src.main must expose main()")
        self.assertTrue(callable(main_mod.main))

    def test_core_pipeline_constructs_and_runs_one_frame(self):
        from src.config import SystemConfig
        from src.detector import DrowsinessDetector
        from src.temporal_analyzer import TemporalAnalyzer
        from src.fatigue_fusion import FatigueFusionEngine
        from src.robustness import RobustnessGuard, SignalQuality
        from src.state_manager import StateManager, DriverStatus

        cfg = SystemConfig()
        DrowsinessDetector()
        temporal = TemporalAnalyzer(cfg)
        fusion = FatigueFusionEngine(cfg)
        guard = RobustnessGuard(cfg)
        mgr = StateManager(cfg)

        # One synthetic "alert driver" frame through the real data flow.
        sq = SignalQuality(landmark_jitter=1.0, frame_brightness=120.0,
                           face_visible=True)
        snap = guard.update(sq, ear_conf=0.0, mar_conf=0.0, pose_conf=0.0)
        self.assertGreaterEqual(snap.system_reliability, 0.0)
        self.assertLessEqual(snap.system_reliability, 1.0)

        ts = temporal.update(raw_ear=0.30, raw_mar=0.20)
        fusion.update(ts, reliability=snap.system_reliability)
        state = mgr.update(ts, face_detected=True,
                           reliability=snap.system_reliability)
        self.assertIsInstance(state.status, DriverStatus)

    def test_reliability_gate_is_three_component(self):
        from src.robustness import SignalQuality, RobustnessSnapshot
        # Phantom `tracking` fields must not exist (precondition 4).
        self.assertNotIn("tracking_confidence",
                         getattr(SignalQuality, "__dataclass_fields__", {}))
        self.assertNotIn("tracking_quality",
                         getattr(RobustnessSnapshot, "__dataclass_fields__", {}))


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    sys.exit(0 if result.wasSuccessful() else 1)
