"""
Dataset Fetcher & Downloader Pipeline (Data/ Real Data Connected)
===================================================================
Validates availability of primary project datasets in Data/ (mrl_eye, drowsiness_detection, nthu_ddd, yawdd).
"""

import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SystemConfig
from src.dataset_manager import DatasetManager


def main():
    cfg = SystemConfig()
    manager = DatasetManager(cfg)
    results = manager.validate_all_datasets()

    print("===================================================================================")
    print("                    DATASET VALIDATION & CONNECTIVITY STATUS                       ")
    print("===================================================================================")
    for key, meta in results.items():
        status = "✅ VALIDATED" if meta.is_valid else "❌ MISSING"
        print(f"Dataset: {key:<22} | Status: {status:<12} | Path: {meta.path}")
    print("===================================================================================")


if __name__ == "__main__":
    main()
