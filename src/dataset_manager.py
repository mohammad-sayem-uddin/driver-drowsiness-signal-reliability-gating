"""
Centralized Dataset Manager
============================
Infrastructure component for resolving dataset locations, validating folder structures,
detecting missing datasets, and reporting corpus statistics across Data/ directory datasets.

Design Principles:
    1. Zero image loading, zero preprocessing, zero model training.
    2. Centralized path resolution reading from SystemConfig.
    3. Structural validation of dataset directories and partitions.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.config import SystemConfig, DatasetPathsConfig


@dataclass
class DatasetMetadata:
    """Read-only structural metadata for a validated dataset."""
    name: str
    path: str
    exists: bool
    total_files: int = 0
    subdirectories: List[str] = field(default_factory=list)
    image_formats: List[str] = field(default_factory=list)
    is_valid: bool = False
    validation_notes: str = ""


class DatasetManager:
    """
    Reusable dataset management engine.

    Usage:
        from src.config import SystemConfig
        from src.dataset_manager import DatasetManager

        cfg = SystemConfig()
        manager = DatasetManager(cfg)
        manager.validate_all_datasets()
        meta = manager.get_dataset_metadata("mrl_eye")
    """

    def __init__(self, cfg: Optional[SystemConfig] = None):
        self.cfg = cfg or SystemConfig()
        self.paths = self.cfg.dataset_paths
        self.metadata_cache: Dict[str, DatasetMetadata] = {}

    def get_path(self, dataset_key: str) -> str:
        """Resolves path for a dataset key ('mrl_eye', 'drowsiness_detection', 'nthu_ddd', 'yawdd')."""
        key_map = {
            "mrl_eye": self.paths.mrl_eye_path,
            "drowsiness_detection": self.paths.drowsiness_detection_path,
            "nthu_ddd": self.paths.nthu_ddd_path,
            "yawdd": self.paths.yawdd_path,
        }
        if dataset_key not in key_map:
            raise KeyError(f"Unknown dataset key: '{dataset_key}'. Valid keys: {list(key_map.keys())}")
        return key_map[dataset_key]

    def validate_dataset(self, dataset_key: str) -> DatasetMetadata:
        """Inspects filesystem structure for a dataset without loading image pixels."""
        ds_path = self.get_path(dataset_key)
        if not os.path.exists(ds_path):
            meta = DatasetMetadata(
                name=dataset_key,
                path=ds_path,
                exists=False,
                is_valid=False,
                validation_notes=f"Directory '{ds_path}' does not exist.",
            )
            self.metadata_cache[dataset_key] = meta
            return meta

        file_count = 0
        formats = set()
        subdirs = []

        for root, dirs, files in os.walk(ds_path):
            for d in dirs:
                subdirs.append(os.path.relpath(os.path.join(root, d), ds_path))
            for f in files:
                if f.startswith('.'):
                    continue
                file_count += 1
                ext = os.path.splitext(f)[1].lower()
                if ext:
                    formats.add(ext)

        is_valid = file_count > 0
        notes = f"Validated {file_count} files across {len(subdirs)} subdirectories."

        meta = DatasetMetadata(
            name=dataset_key,
            path=ds_path,
            exists=True,
            total_files=file_count,
            subdirectories=sorted(subdirs),
            image_formats=sorted(list(formats)),
            is_valid=is_valid,
            validation_notes=notes,
        )
        self.metadata_cache[dataset_key] = meta
        return meta

    def validate_all_datasets(self) -> Dict[str, DatasetMetadata]:
        """Validates all 4 primary project datasets."""
        for key in ["mrl_eye", "drowsiness_detection", "nthu_ddd", "yawdd"]:
            self.validate_dataset(key)
        return self.metadata_cache

    def get_dataset_metadata(self, dataset_key: str) -> DatasetMetadata:
        """Returns cached metadata or runs validation."""
        if dataset_key not in self.metadata_cache:
            return self.validate_dataset(dataset_key)
        return self.metadata_cache[dataset_key]
