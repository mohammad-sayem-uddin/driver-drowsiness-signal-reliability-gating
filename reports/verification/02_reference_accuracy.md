# STAGE 2: REFERENCE ACCURACY & INTERPRETATION AUDIT

**Auditing Body**: Scientific Verification Committee (IEEE Transactions Senior Reviewers)  
**Scope**: Verification of how cited literature was interpreted and represented in prior reports  
**Date**: July 2026

---

## 1. Executive Summary

Prior reports correctly captured the **broad conceptual trajectories** of the literature (e.g., that landmark heuristics run fast but lack temporal robustness, that 3D-CNNs achieve high accuracy but suffer latency overhead, and that MediaPipe is a standard 3D landmark extractor).

However, an in-depth citation audit reveals **three significant interpretation inaccuracies and performance exaggerations** in prior reports:

1. **B. Reddy et al. (2017) Model Performance**: Prior reports claimed Reddy et al. used a heavy 3D ResNet-18 achieving 94.2% accuracy at 125ms latency on Pi 4. In reality, B. Reddy et al. (CVPRW 2017) proposed a **compressed 2D/3D landmark network** running at **14.9 FPS (67ms)** on an NVIDIA Jetson TK1 with **89.5% 3-class accuracy**.
2. **Soukupová & Čech (2016) Scope**: Prior reports implied Soukupová & Čech proposed a full driver drowsiness detection system. In reality, their paper proposed a pure **eye blink detector using 6 landmarks and an SVM**, explicitly noting that drowsiness detection required further temporal tracking (like PERCLOS).
3. **Overstated Novelty vs. Chen et al. (2025)**: Prior reports claimed our system's selective CNN invocation was fundamentally novel. In reality, selective invocation triggered by heuristic uncertainty was already proposed in 2025 edge literature. Our novelty must be reframed around the **`RobustnessGuard` signal quality engine and physical ambiguity filters**.

---

## 2. Detailed Line-by-Line Reference Audit

### Citation 1: Soukupová & Čech (2016)
- **Interpretation in Prior Reports**: Cited as the foundational baseline for Eye Aspect Ratio (EAR) computation.
- **Accuracy Assessment**: **ACCURATE**. Formula $EAR = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \cdot \|P_1 - P_4\|}$ was cited verbatim and implemented faithfully in `src/detector.py`.
- **Omitted Limitation**: Soukupová & Čech explicitly noted that EAR values vary across individuals and recommended dynamic adaptation or thresholding over time—a feature missing in our static $EAR = 0.21$ setup.

### Citation 2: B. Reddy et al. (CVPRW 2017)
- **Interpretation in Prior Reports**: Cited as a heavy 3D ResNet-18 benchmark requiring ~18 GFLOPs and 125ms latency.
- **Accuracy Assessment**: **INACCURATE / EXAGGERATED**. B. Reddy et al. (CVPRW 2017) actually focused on **deep model compression** specifically for low-cost embedded systems (NVIDIA Jetson TK1), achieving 14.9 FPS at 89.5% accuracy.
- **Correction**: The manuscript must cite B. Reddy et al. as an early pioneer of compressed deep models on embedded hardware, rather than a heavy uncompressed 3D network.

### Citation 3: Hassan et al. (2024)
- **Interpretation in Prior Reports**: Cited as a MediaPipe + ML ensemble system running at 32 FPS on Raspberry Pi 4 with 93.8% accuracy.
- **Accuracy Assessment**: **ACCURATE**. Matches reported figures in recent 2024 IEEE Access literature.
- **Key Insight**: Hassan et al. use frame-counting duration logic and 3D Euclidean metrics for mouth landmarks, confirming our research gap regarding 2D/3D depth divergence and FPS variance.

---

## 3. Impact on Paper Positioning

To survive rigorous IEEE T-ITS peer review:
- The manuscript must **correct the performance figures** for Reddy et al. (CVPRW 2017: 14.9 FPS, 89.5% 3-class accuracy).
- The manuscript must **acknowledge that selective CNN invocation** has prior art, and position our main contribution around the **`RobustnessGuard` multi-factor reliability index and physical metric separation (2D vs. 3D Euclidean distances)**.
