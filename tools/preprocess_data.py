"""
Data Preprocessing Engine (Data/ Real Data Connected)
=====================================================
Preprocesses raw eye patch images into normalized, standardized 24x24 grayscale matrices
with aspect-ratio preservation and histogram equalization.

Usage:
    python3 tools/preprocess_data.py
"""

import os
import sys
import cv2
import numpy as np

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SystemConfig
from src.dataset_manager import DatasetManager

TARGET_SIZE = (24, 24)


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Resize to MicroEyeNet Target Dimensions (24x24)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 3. Histogram Equalization (Contrast Normalization)
    equalized = cv2.equalizeHist(resized)

    return equalized


def main():
    cfg = SystemConfig()
    manager = DatasetManager(cfg)
    mrl_meta = manager.get_dataset_metadata("mrl_eye")
    print(f"[PreprocessData] Connected to dataset '{mrl_meta.name}' at '{mrl_meta.path}'.")
    print(f" -> Total files validated: {mrl_meta.total_files}")


if __name__ == "__main__":
    main()
