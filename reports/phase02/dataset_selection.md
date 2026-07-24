# TASK 2: OFFICIAL DATASET SELECTION & ROLE ASSIGNMENT

**Auditing Body**: Scientific Selection Committee & Research Infrastructure Team  
**Date**: July 2026

---

## 1. Selected Datasets & Role Allocation

```
===================================================================================
                       RESEARCH DATASET ROLE ALLOCATION
===================================================================================

1. MODEL TRAINING & VALIDATION ROLE
   - Selected Dataset : MRL Eye Dataset + Closed Eyes in The Wild (CEW)
   - Data Modality    : Micro $24\times24$ Eye Crops (Open vs Closed)
   - Function         : Train MicroEyeNet TFLite uncertainty resolver CNN (Phase 3).

2. PRIMARY AUTOMOTIVE BENCHMARK ROLE
   - Selected Dataset : NTHU Driver Drowsiness Detection (NTHU-DDD)
   - Data Modality    : Full-frame Infrared & RGB Video Sequences
   - Function         : Frame-by-frame evaluation of full hybrid system (Phase 4).

3. CROSS-DATASET & PHYSICAL AMBIGUITY BENCHMARK ROLE
   - Selected Dataset : YawDD (Yawning Detection Dataset)
   - Data Modality    : Natural Driving Video Sequences under Sunlight & Shadows
   - Function         : Validation of 2D MAR lip depth fix and speech jitter filter.
===================================================================================
```

---

## 2. Evidence-Based Justification for Selection

- **Why MRL + CEW for CNN Training?**: MRL and CEW contain 89,000+ annotated eye crops spanning diverse subjects, illumination levels, infrared lighting, reflections, and glasses. Combining them ensures MicroEyeNet generalizes to edge automotive environments.
- **Why NTHU-DDD for Primary Benchmark?**: NTHU-DDD is the universally recognized gold standard in top-tier transportation venues (IEEE T-ITS, IEEE IV). Its 5 scenarios (Night, Glasses, Bare Face, Yawning, Talking) directly align with our research gaps.
- **Why Exclude OpenEDS and DMD?**: OpenEDS focuses on VR/AR head-mounted displays rather than driver-facing cameras. DMD requires restrictive commercial licensing that hinders open reproducibility.
