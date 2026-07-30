"""
Comprehensive Benchmark & Evaluation Suite (Real Dataset Connected)
=====================================================================
Runs the REAL per-frame pipeline (src.frame_processor.FrameProcessor) over
benchmark datasets and records only measured quantities.

Integrity contract (freeze-report precondition 1):
- No metric is fabricated. Latency is measured on THIS host and labelled with
  the host name; it is NOT a Raspberry Pi 4 number until profiled on-device.
- Accuracy / ROC / FPR@matchedTPR are NOT emitted here: they require the LOSO
  harness and a fixed operating point (frozen protocol §3, work-list step 3).
  This suite emits the measured `latency_ms` artifact and per-frame score
  dumps that the LOSO harness consumes.
"""

import os
import sys
import time
import json
import platform
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SystemConfig
from src.frame_processor import FrameProcessor
from src.data_loaders import NTHUDDDDataLoader, YawDDVideoDataLoader


@dataclass
class LatencyMeasurement:
    """Measured per-frame wall-clock latency for the full pipeline."""
    device: str = ""
    n_frames: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    throughput_fps: float = 0.0


def _host_label() -> str:
    return f"{platform.system()}-{platform.machine()}"


class BenchmarkEvaluator:
    """Benchmark engine driving the shared headless FrameProcessor."""

    def __init__(self, cfg: Optional[SystemConfig] = None):
        self.cfg = cfg or SystemConfig()
        self.nthu_loader = NTHUDDDDataLoader(self.cfg)
        self.yawdd_loader = YawDDVideoDataLoader(self.cfg)

    def measure_nthu_latency(self, max_frames: int = 500,
                             video_fps: float = 30.0,
                             enable_cnn: bool = False) -> LatencyMeasurement:
        """
        Run FrameProcessor over up to ``max_frames`` real NTHU frames and
        MEASURE per-frame pipeline latency on this host. Returns measured
        percentiles — never a hardcoded constant.
        """
        import cv2
        seq = self.nthu_loader.get_sequence_files()
        # Flatten category->frames into an ordered frame list.
        frames: List[str] = []
        for cat in sorted(seq.keys()):
            frames.extend(seq[cat])
        if not frames:
            print("[Benchmark] No NTHU frames found under Data/nthu_ddd.")
            return LatencyMeasurement(device=_host_label())
        frames = frames[:max_frames]

        fp = FrameProcessor(self.cfg, enable_cnn=enable_cnn)
        per_frame_ms: List[float] = []
        try:
            for i, path in enumerate(frames):
                img = cv2.imread(path)
                if img is None:
                    continue
                t0 = time.perf_counter()
                fp.process(img, timestamp=i / video_fps)
                per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        finally:
            fp.close()

        if not per_frame_ms:
            return LatencyMeasurement(device=_host_label())

        arr = np.array(per_frame_ms, dtype=np.float64)
        mean = float(arr.mean())
        return LatencyMeasurement(
            device=_host_label(),
            n_frames=int(arr.size),
            mean_ms=mean,
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            max_ms=float(arr.max()),
            throughput_fps=(1000.0 / mean) if mean > 0 else 0.0,
        )

    def write_latency_artifact(self, meas: LatencyMeasurement,
                               out_path: str = "results/measured_results.json"):
        """
        Merge a measured latency block into results/measured_results.json.
        The stages array is a single aggregate ("full_pipeline") until the
        per-stage profiler (work-list step 5) breaks it down on-device.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data: Dict = {}
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                data = json.load(f)
        lat = data.setdefault("latency_ms", {})
        lat.setdefault("stages", ["full_pipeline"])
        lat[meas.device] = [round(meas.mean_ms, 4)]
        # Provenance so no reader mistakes host latency for Pi 4 latency.
        data.setdefault("_provenance", {})[meas.device] = {
            "measured_on": meas.device,
            "n_frames": meas.n_frames,
            "mean_ms": round(meas.mean_ms, 4),
            "p50_ms": round(meas.p50_ms, 4),
            "p95_ms": round(meas.p95_ms, 4),
            "max_ms": round(meas.max_ms, 4),
            "throughput_fps": round(meas.throughput_fps, 2),
            "note": ("Measured on this host, NOT a Raspberry Pi 4. Pi 4 numbers "
                     "require on-device profiling (frozen protocol step 5)."),
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Benchmark] Wrote measured latency for '{meas.device}' -> {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Measured NTHU pipeline latency.")
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cnn", action="store_true",
                    help="enable the optional CNN ablation arm")
    ap.add_argument("--write", action="store_true",
                    help="merge the measured latency into results/measured_results.json")
    args = ap.parse_args()

    evaluator = BenchmarkEvaluator()
    print("=== NTHU Pipeline Latency Benchmark (measured, this host) ===")
    m = evaluator.measure_nthu_latency(max_frames=args.max_frames,
                                       video_fps=args.fps, enable_cnn=args.cnn)
    print(f"  device        : {m.device}")
    print(f"  frames        : {m.n_frames}")
    print(f"  mean latency  : {m.mean_ms:.3f} ms")
    print(f"  p50 / p95     : {m.p50_ms:.3f} / {m.p95_ms:.3f} ms")
    print(f"  throughput    : {m.throughput_fps:.2f} FPS")
    if args.write:
        evaluator.write_latency_artifact(m)
