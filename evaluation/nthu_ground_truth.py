"""
NTHU-DDD Ground-Truth Mapping (frozen protocol — real labels, no fabrication)
=============================================================================
The NTHU Driver Drowsiness Detection dataset ships its labels *in the directory
layout and in every frame filename* — nothing here is invented or inferred.

On-disk layout (as present in ``Data/nthu_ddd``)::

    Data/nthu_ddd/train/notdrowsy/                       -> label 0 (alert)
    Data/nthu_ddd/train/drowsy/sleepyCombination/        -> label 1 (drowsy)
    Data/nthu_ddd/train/drowsy/yawning/                  -> label 1 (drowsy)
    Data/nthu_ddd/train/drowsy/slowBlinkWithNodding/     -> label 1 (drowsy)

Filename grammar (verified against the corpus)::

    <subject>_<glasses>_<condition>_<frameindex>_<label>.jpg
    e.g. 005_noglasses_yawning_2396_drowsy.jpg
         001_noglasses_nonsleepyCombination_860_notdrowsy.jpg

    subject   : leading integer token (subject/driver id) -> LOSO grouping key
    glasses   : {glasses, noglasses}
    condition : recording condition (yawning, sleepyCombination, ...)
    frameindex: integer frame counter within the clip
    label     : trailing token, exactly {drowsy, notdrowsy} -> binary target

Ground-truth policy (frozen, and its limitation stated honestly)
----------------------------------------------------------------
We use the OFFICIAL NTHU per-frame condition label as a *binary* target:
``notdrowsy -> 0`` (negative) and ``drowsy -> 1`` (positive). This is the
label NTHU itself assigns to each frame; we do not relabel or threshold it.

Known limitation (must be reported in the paper): the label is clip-condition
derived, so a physically open-eyed frame inside a "yawning" clip is still
labelled drowsy. This is the standard NTHU frame-level convention; it makes
frame-level FPR a conservative (slightly pessimistic) estimator of the
system's true event-level FPR. The temporal integration in the pipeline is
what turns these noisy per-frame labels into stable event decisions.

This module ENUMERATES and LABELS only. It loads no pixels, computes no
metric, and fabricates nothing.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict, OrderedDict

# subject _ glasses _ condition _ frameidx _ label .jpg
_NTHU_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<glasses>[A-Za-z]+)_(?P<condition>[A-Za-z]+)_"
    r"(?P<frame>\d+)_(?P<label>drowsy|notdrowsy)\.jpg$"
)

LABEL_ALERT = 0    # notdrowsy
LABEL_DROWSY = 1   # drowsy


@dataclass(frozen=True)
class NTHUFrame:
    """One labelled NTHU frame (metadata only; no pixels)."""
    path: str
    subject: str        # LOSO grouping key
    label: int          # 0 = notdrowsy/alert, 1 = drowsy
    condition: str      # e.g. 'yawning', 'nonsleepyCombination'
    glasses: str        # 'glasses' | 'noglasses'
    frame_index: int


def parse_nthu_filename(filename: str) -> Optional[Dict]:
    """
    Parse one NTHU basename into its fields. Returns None if the name does not
    match the NTHU grammar (so unrelated files are skipped, never guessed).
    """
    m = _NTHU_RE.match(os.path.basename(filename))
    if not m:
        return None
    d = m.groupdict()
    return {
        "subject": d["subject"],
        "glasses": d["glasses"],
        "condition": d["condition"],
        "frame_index": int(d["frame"]),
        "label": LABEL_DROWSY if d["label"] == "drowsy" else LABEL_ALERT,
    }


def enumerate_labelled_frames(nthu_root: str) -> List[NTHUFrame]:
    """
    Walk ``nthu_root`` and return every parseable frame with its real label.
    Frames are grouped/ordered by (subject, condition, frame_index) so that
    per-subject temporal ordering is preserved for the LOSO harness.
    """
    frames: List[NTHUFrame] = []
    for root, _, files in os.walk(nthu_root):
        for f in files:
            if not f.endswith(".jpg") or f.startswith("."):
                continue
            parsed = parse_nthu_filename(f)
            if parsed is None:
                continue
            frames.append(NTHUFrame(
                path=os.path.join(root, f),
                subject=parsed["subject"],
                label=parsed["label"],
                condition=parsed["condition"],
                glasses=parsed["glasses"],
                frame_index=parsed["frame_index"],
            ))
    frames.sort(key=lambda fr: (fr.subject, fr.condition, fr.frame_index))
    return frames


def group_by_subject(frames: List[NTHUFrame]) -> "OrderedDict[str, List[NTHUFrame]]":
    """Group labelled frames by subject id (deterministic subject order)."""
    groups: Dict[str, List[NTHUFrame]] = defaultdict(list)
    for fr in frames:
        groups[fr.subject].append(fr)
    return OrderedDict((s, groups[s]) for s in sorted(groups.keys()))


def label_statistics(frames: List[NTHUFrame]) -> Dict:
    """
    Dataset-level FACTS (counts) about the ground truth — not a performance
    result. Safe to log/report: these are properties of the corpus on disk.
    """
    per_subject = group_by_subject(frames)
    n_pos = sum(1 for fr in frames if fr.label == LABEL_DROWSY)
    n_neg = sum(1 for fr in frames if fr.label == LABEL_ALERT)
    cond = defaultdict(int)
    for fr in frames:
        cond[fr.condition] += 1
    return {
        "n_frames": len(frames),
        "n_subjects": len(per_subject),
        "subjects": list(per_subject.keys()),
        "n_drowsy": n_pos,
        "n_notdrowsy": n_neg,
        "conditions": dict(sorted(cond.items())),
        "frames_per_subject": {s: len(v) for s, v in per_subject.items()},
    }


if __name__ == "__main__":
    import sys
    import json
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("Data", "nthu_ddd")
    fr = enumerate_labelled_frames(root)
    stats = label_statistics(fr)
    # Print counts only (dataset facts), never a metric.
    print(json.dumps({k: v for k, v in stats.items()
                      if k != "frames_per_subject"}, indent=2))
