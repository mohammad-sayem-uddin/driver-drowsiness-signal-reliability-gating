# TASK 14: FORMAL DATASET CARDS

**Auditing Body**: AI Research Engineer & Academic Documentation Specialist  
**Date**: July 2026

---

## Dataset Card 1: MRL Eye Dataset (Cropped Subset)

### Overview & Purpose
- **Dataset Name**: MRL Eye Dataset
- **Repository Path**: `data/raw/mrl_eyes/` & `data/processed/`
- **Primary Use**: Training and validation of lightweight CNN eye-state classifiers (MicroEyeNet $24\times24$).
- **Modality**: Grayscale & RGB micro eye patch crops.
- **Labels**: Binary (`open` eye vs. `closed` eye), subject ID, glasses (`yes`/`no`), lighting (`normal`/`low_light`).

### Citation & License
- **Citation**: MRL Research Group, *Media Research Lab Eye Dataset*, 2018.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Download URL**: `http://mrl.cs.vsb.cz/eyedataset`

### Biases & Limitations
- **Limitations**: Contains isolated eye crops without full-face context or jaw motion.
- **Known Biases**: Higher proportion of adult male subjects relative to female subjects.

---

## Dataset Card 2: NTHU Driver Drowsiness Detection Dataset (NTHU-DDD)

### Overview & Purpose
- **Dataset Name**: NTHU-DDD
- **Repository Path**: `benchmark/nthu_ddd/` (Target Directory)
- **Primary Use**: Gold-standard frame-by-frame evaluation of full-pipeline driver drowsiness detection systems.
- **Modality**: Full-frame Infrared (NIR) and RGB video streams.
- **Labels**: Frame-level annotations for drowsiness, eye closure, yawning, talking, nodding, and face position.

### Citation & License
- **Citation**: C.-W. Chen et al., *Real-Time Driver Drowsiness Detection Using Facial Landmarks and Deep Learning*, Proc. IEEE CVPR Workshops, 2016.
- **License**: Restricted Academic License Agreement (Requires EULA application).
- **Download URL**: `http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/`

### Biases & Limitations
- **Limitations**: Simulated driving simulator environment (stationary vehicle cabin).
- **Recommended Usage**: Benchmark evaluation set (Phase 4). Do NOT mix into CNN training data.
