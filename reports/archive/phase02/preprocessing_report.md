# DATA PREPROCESSING REPORT

**Target Pipeline**: MicroEyeNet $24\times24$ Input Standardization  
**Source Directory**: `data/raw`  
**Output Directory**: `data/processed`  
**Date**: July 2026

---

## 1. Preprocessing Configuration Matrix

| Step | Operation | Parameters | Rationale |
|:---|:---|:---|:---|
| **1. Color Space** | Grayscale Conversion | `cv2.COLOR_BGR2GRAY` | MicroEyeNet accepts single-channel 8-bit inputs ($24\times24\times1$). |
| **2. Resizing** | Interpolation | `24x24` pixels (`cv2.INTER_AREA`) | Matches MicroEyeNet architecture input shape. |
| **3. Contrast** | Histogram Equalization | `cv2.equalizeHist` | Normalizes illumination variations across day/night driving conditions. |

---

## 2. Processing Summary

- **Total Images Processed**: 200
- **Processing Errors / Skipped**: 0
- **Output Image Shape**: $24 \times 24 \times 1$
- **Data Preservation Guarantee**: Original raw files in `data/raw` remain untouched.
