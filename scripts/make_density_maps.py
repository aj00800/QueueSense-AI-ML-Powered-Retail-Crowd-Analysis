"""Generate Gaussian density maps from Mall dataset ground-truth head positions.

Reads head (x, y) coordinates from mall_gt.mat and creates a proper
Gaussian density map for every frame. These are the ground-truth density
maps used for both model training and visualization.

Output:
    datasets/mall_dataset/density_maps/frame_NNNNNN.npy  — float32 arrays

Usage:
    python scripts/make_density_maps.py
    python scripts/make_density_maps.py --sigma 15 --out-dir datasets/mall_dataset/density_maps
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import scipy.io
import scipy.ndimage


def load_mall_head_positions(mat_path: str | Path) -> list[np.ndarray]:
    """Load head (x, y) positions for each frame.

    Returns:
        List of (N_i, 2) float32 arrays, one per frame.
        Column 0 = x (horizontal), Column 1 = y (vertical).
    """
    mat = scipy.io.loadmat(str(mat_path))
    frame_arr = mat["frame"]  # shape (1, 2000)
    results: list[np.ndarray] = []
    n_frames = frame_arr.shape[1]
    for i in range(n_frames):
        rec = frame_arr[0, i]
        loc = rec["loc"][0, 0]  # (N, 2) float64
        results.append(loc.astype(np.float32))
    return results


def make_gaussian_density(
    height: int,
    width: int,
    head_positions: np.ndarray,
    sigma: float = 15.0,
    adaptive: bool = True,
) -> np.ndarray:
    """Create a Gaussian density map from head positions.

    Each head point contributes a 2D Gaussian blob. The sum of the
    density map equals the number of people in the frame.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        head_positions: (N, 2) array of (x, y) head coordinates.
        sigma: Gaussian std deviation in pixels.
        adaptive: If True, scale sigma by crowd density for better accuracy.

    Returns:
        float32 density map of shape (height, width).
        den.sum() ≈ len(head_positions)
    """
    density = np.zeros((height, width), dtype=np.float32)

    if len(head_positions) == 0:
        return density

    # Adaptive sigma: denser scenes use smaller sigma to avoid overlap
    if adaptive and len(head_positions) > 1:
        from scipy.spatial import KDTree
        tree = KDTree(head_positions[:, ::-1])  # KDTree uses (y, x)
        dists, _ = tree.query(head_positions[:, ::-1], k=min(4, len(head_positions)))
        avg_nn_dist = float(dists[:, 1:].mean()) if dists.shape[1] > 1 else sigma
        sigma = max(5.0, min(sigma, avg_nn_dist * 0.3))

    for x, y in head_positions:
        # Convert to integer pixel center — clamp to frame bounds
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))
        density[cy, cx] += 1.0

    # Apply Gaussian blur — sum is preserved (approximately)
    density = scipy.ndimage.gaussian_filter(density, sigma=sigma, mode="constant", cval=0)

    # Normalize so sum matches actual count
    n_people = len(head_positions)
    dsum = float(density.sum())
    if dsum > 1e-8:
        density = density * (n_people / dsum)

    return density.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GT density maps for Mall dataset.")
    ap.add_argument("--mall-root", type=str, default="datasets/mall_dataset")
    ap.add_argument("--sigma", type=float, default=15.0, help="Gaussian sigma in pixels")
    ap.add_argument("--adaptive", action="store_true", default=True,
                    help="Use adaptive sigma based on nearest-neighbour distance")
    ap.add_argument("--out-dir", type=str, default="datasets/mall_dataset/density_maps")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    args = ap.parse_args()

    mall_root = PROJECT_ROOT / args.mall_root
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_path = mall_root / "mall_gt.mat"
    frames_dir = mall_root / "frames"

    print(f"Loading head positions from {mat_path}")
    all_positions = load_mall_head_positions(mat_path)

    frame_paths = sorted(frames_dir.glob("seq_*.jpg"))
    n = min(len(frame_paths), len(all_positions))
    if args.max_frames and args.max_frames > 0:
        n = min(n, args.max_frames)

    print(f"Generating density maps for {n} frames -> {out_dir}")

    # Get canonical frame size from first image
    sample = cv2.imread(str(frame_paths[0]))
    H, W = sample.shape[:2]
    print(f"Frame size: {W}x{H}")

    for i in range(n):
        positions = all_positions[i]
        density = make_gaussian_density(H, W, positions, sigma=args.sigma, adaptive=args.adaptive)
        out_path = out_dir / f"frame_{i:06d}.npy"
        np.save(str(out_path), density)

        if (i + 1) % 100 == 0 or i == n - 1:
            count_check = float(density.sum())
            gt_count = len(positions)
            print(f"  [{i+1}/{n}] gt_count={gt_count} density_sum={count_check:.2f}")

    print(f"\nDone! Saved {n} density maps to {out_dir}")


if __name__ == "__main__":
    main()
