from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import scipy.io


def load_mall_counts(mall_root: str | Path) -> np.ndarray:
    mall_root = Path(mall_root)
    mat = scipy.io.loadmat(mall_root / "mall_gt.mat")
    return mat["count"].squeeze().astype(np.float32)


def estimate_count_from_mask(mask: np.ndarray, min_area: int, person_area: float) -> float:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    big = areas[areas >= min_area]
    if big.size == 0:
        return 0.0
    return float(big.sum()) / float(person_area)


def mog2_mae(
    mall_root: str | Path,
    history: int = 300,
    var_threshold: float = 16.0,
    min_area: int = 150,
    person_area: float = 1800.0,
    max_frames: int = 0,
) -> tuple[float, int]:
    """Simple background-subtraction baseline on Mall frames.

    Returns: (mae, num_frames_evaluated)
    """
    mall_root = Path(mall_root)
    frames_dir = mall_root / "frames"
    image_paths = sorted(frames_dir.glob("seq_*.jpg"))
    counts = load_mall_counts(mall_root)

    n = min(len(image_paths), len(counts))
    if max_frames and max_frames > 0:
        n = min(n, max_frames)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    abs_errors: list[float] = []
    for i in range(n):
        img = cv2.imread(str(image_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fg = subtractor.apply(img)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=1)

        pred = estimate_count_from_mask(fg, min_area=min_area, person_area=person_area)
        gt = float(counts[i])
        abs_errors.append(abs(pred - gt))

    if not abs_errors:
        return float("nan"), 0
    return float(np.mean(abs_errors)), len(abs_errors)
