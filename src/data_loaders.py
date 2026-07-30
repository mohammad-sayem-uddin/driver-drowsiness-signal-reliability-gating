"""
Production Data Loaders & Subject-Level Split Infrastructure
================================================================
Defines clean data loader interfaces and subject-level GroupKFold splitters
for MRL, Drowsiness Detection, NTHU-DDD, and YawDD datasets.

Design Principles:
    1. Zero execution during initialization (no model training, no frame processing).
    2. Enumerate file paths, class labels, and metadata lazily.
    3. Expose subject-level GroupKFold splitting for MRL to eliminate subject data leakage.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from src.dataset_manager import DatasetManager
from src.config import SystemConfig


class MRLEyeDataLoader:
    """
    Data loader for MRL Eye Dataset (Data/mrl_eye).
    Handles pre-partitioned train/val/test paths and subject ID extraction.
    """

    def __init__(self, cfg: Optional[SystemConfig] = None):
        self.mgr = DatasetManager(cfg)
        self.base_path = self.mgr.get_path("mrl_eye")
        self.class_map = {"awake": 0, "sleepy": 1}

    def extract_subject_id(self, filename: str) -> str:
        """Extracts subject ID (e.g., 's0013') from MRL filename 's0013_00354_0_0_0_0_0_01.png'."""
        match = re.match(r"^(s\d{4})_", filename)
        return match.group(1) if match else "unknown"

    def get_partition_files(self, split: str = "train") -> List[Tuple[str, int, str]]:
        """
        Enumerates files in specified split ('train', 'val', 'test').
        Returns list of tuples: (file_path, class_label, subject_id).
        """
        split_path = os.path.join(self.base_path, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split directory '{split_path}' does not exist.")

        samples = []
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(split_path, class_name)
            if not os.path.exists(class_dir):
                continue
            for f in os.listdir(class_dir):
                if f.startswith('.') or not f.endswith('.png'):
                    continue
                fp = os.path.join(class_dir, f)
                subj = self.extract_subject_id(f)
                samples.append((fp, label, subj))
        return samples

    def get_subject_disjoint_files(self, split: str = "train") -> List[Tuple[str, int, str]]:
        """
        Leak-free accessor: reads the subject-disjoint split manifest at
        Data/mrl_eye/splits_subject_disjoint/{split}.csv (columns
        filepath,label,subject_id; seed 42) produced by
        tools/build_subject_disjoint_splits.py. Unlike get_partition_files
        (which enumerates the shipped subject-LEAKY directory partitions), this
        method guarantees zero subject overlap across train/val/test, as
        mandated by the frozen protocol (IMPLEMENTATION_SPECIFICATION_FROZEN §3).
        Returns list of tuples: (file_path, class_label, subject_id).
        """
        import csv
        manifest = os.path.join(
            self.base_path, "splits_subject_disjoint", f"{split}.csv"
        )
        if not os.path.exists(manifest):
            raise FileNotFoundError(
                f"Subject-disjoint manifest '{manifest}' does not exist. "
                "Run tools/build_subject_disjoint_splits.py first."
            )
        samples = []
        with open(manifest, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # filepath is stored relative to the project root; the caller is
                # responsible for resolving it against ROOT. Returned verbatim so
                # this accessor mirrors get_partition_files' (path, label, subj) shape.
                samples.append((row["filepath"], int(row["label"]), row["subject_id"]))
        return samples

    def get_all_samples_grouped_by_subject(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Groups all 84,898 MRL samples by subject ID for GroupKFold / LOSO splitting.
        Returns dict mapping subject_id -> list of (file_path, class_label).
        """
        subject_groups = defaultdict(list)
        for split in ["train", "val", "test"]:
            for fp, label, subj in self.get_partition_files(split):
                subject_groups[subj].append((fp, label))
        return dict(subject_groups)


class SubjectGroupKFoldSplitter:
    """
    Subject-Level Stratified GroupKFold Splitter for MRL Eye Dataset.
    Guarantees zero subject overlap across train, validation, and test folds.
    """

    def __init__(self, loader: MRLEyeDataLoader):
        self.loader = loader

    def generate_loso_splits(self, num_folds: int = 5) -> List[Dict[str, List[Tuple[str, int]]]]:
        """
        Generates k-fold splits where no subject appears in more than one fold.
        """
        grouped = self.loader.get_all_samples_grouped_by_subject()
        subjects = sorted(list(grouped.keys()))
        
        folds = [[] for _ in range(num_folds)]
        for i, subj in enumerate(subjects):
            fold_idx = i % num_folds
            folds[fold_idx].append(subj)

        split_results = []
        for test_fold_idx in range(num_folds):
            test_subjs = set(folds[test_fold_idx])
            val_subjs = set(folds[(test_fold_idx + 1) % num_folds])
            train_subjs = set(subjects) - test_subjs - val_subjs

            train_samples, val_samples, test_samples = [], [], []
            for subj, samples in grouped.items():
                if subj in train_subjs:
                    train_samples.extend(samples)
                elif subj in val_subjs:
                    val_samples.extend(samples)
                else:
                    test_samples.extend(samples)

            split_results.append({
                "train": train_samples,
                "val": val_samples,
                "test": test_samples,
                "test_subjects": list(test_subjs)
            })

        return split_results


class DrowsinessDetectionDataLoader:
    """
    DISQUALIFIED — DO NOT USE FOR ANY TRAINING OR EVALUATION.

    The Data/drowsiness_detection set (4,000 PNGs) was verified to be a
    100% byte-identical duplicate of a subset of the MRL Eye dataset
    (see Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md). Using it would
    (a) leak MRL samples across any MRL-based split and (b) inflate metrics
    with duplicated images. It is retained only so this quarantine is
    explicit; every accessor raises to make accidental use impossible
    (freeze-report precondition 3).
    """

    _DISQUALIFIED_MSG = (
        "Data/drowsiness_detection is DISQUALIFIED: it is a 100% duplicate of "
        "an MRL Eye subset (see FINAL_PRETRAINING_VERIFICATION_REPORT.md) and "
        "must never be used for training or evaluation (precondition 3)."
    )

    def __init__(self, cfg: Optional[SystemConfig] = None):
        raise RuntimeError(self._DISQUALIFIED_MSG)

    def get_all_samples(self) -> List[Tuple[str, int]]:
        raise RuntimeError(self._DISQUALIFIED_MSG)



class NTHUDDDDataLoader:
    """
    Data loader for NTHU Driver Drowsiness Dataset (Data/nthu_ddd).
    Provides frame sequence paths for full-pipeline temporal evaluation.
    """

    def __init__(self, cfg: Optional[SystemConfig] = None):
        self.mgr = DatasetManager(cfg)
        self.base_path = self.mgr.get_path("nthu_ddd")
        self.class_map = {
            "notdrowsy": 0,
            "sleepyCombination": 1,
            "yawning": 2,
            "slowBlinkWithNodding": 3
        }

    def get_sequence_files(self) -> Dict[str, List[str]]:
        """Enumerates frame file lists per multi-class behavioral category."""
        sequences = defaultdict(list)
        for root, _, files in os.walk(self.base_path):
            jpgs = [os.path.join(root, f) for f in files if f.endswith('.jpg') and not f.startswith('.')]
            if jpgs:
                rel = os.path.relpath(root, self.base_path)
                category = rel.split(os.sep)[-1]
                sequences[category].extend(sorted(jpgs))
        return dict(sequences)


class YawDDVideoDataLoader:
    """
    Data loader for YawDD Video Corpus (Data/yawdd).
    Provides video clip paths for real-world video benchmarking.
    """

    def __init__(self, cfg: Optional[SystemConfig] = None):
        self.mgr = DatasetManager(cfg)
        self.base_path = self.mgr.get_path("yawdd")

    def get_video_clips(self) -> List[Dict[str, str]]:
        """Enumerates 348 video clips with camera angle and gender metadata."""
        videos = []
        for root, _, files in os.walk(self.base_path):
            for f in files:
                if f.lower().endswith('.avi') and not f.startswith('.'):
                    fp = os.path.join(root, f)
                    angle = "dash" if "Dash" in fp else "mirror"
                    gender = "female" if "Female" in fp or "female" in fp else "male"
                    videos.append({
                        "path": fp,
                        "filename": f,
                        "camera_angle": angle,
                        "gender": gender
                    })
        return videos
