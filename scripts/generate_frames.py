"""Generate annotated frames and heatmap frames — using GT head positions for accuracy.

This script uses GROUND TRUTH head positions from mall_gt.mat to produce:

  outputs/annotated_frames/   — original frame + vivid heatmap + person markers
                                + queue zone overlay + professional HUD
  outputs/heatmap_frames/     — pure density heatmap visualization

Person positions are drawn as individual markers on every person's head.
Crowd count = exact GT count (integer, always accurate).
Density map = Gaussian KDE from GT head positions (always accurate).
Queue zone = sliding-window search on GT density (finds true open space).

Usage:
    python scripts/generate_frames.py
    python scripts/generate_frames.py --max-frames 20
    python scripts/generate_frames.py --use-model  # use CNN model instead of GT
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
import torch
import torchvision.transforms as T

import config
from crowdcount.model import SimpleDensityCNN


# ---------------------------------------------------------------------------
# GT head position loading
# ---------------------------------------------------------------------------

def load_mall_gt(mat_path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Load head positions and counts from mall_gt.mat.

    Returns:
        (head_positions, counts)
        head_positions: list of (N_i, 2) float32 arrays (x, y per person)
        counts: float32 array of length n_frames
    """
    mat = scipy.io.loadmat(str(mat_path))
    frame_arr = mat["frame"]
    counts = mat["count"].squeeze().astype(np.float32)
    n_frames = frame_arr.shape[1]
    positions: list[np.ndarray] = []
    for i in range(n_frames):
        rec = frame_arr[0, i]
        loc = rec["loc"][0, 0].astype(np.float32)  # (N, 2): col0=x, col1=y
        positions.append(loc)
    return positions, counts


# ---------------------------------------------------------------------------
# Density map from head positions
# ---------------------------------------------------------------------------

def make_gaussian_density(
    height: int,
    width: int,
    head_positions: np.ndarray,
    sigma: float = 20.0,
) -> np.ndarray:
    """Generate a Gaussian density map from head point annotations.

    Uses adaptive sigma based on nearest-neighbour distance when crowd
    is dense enough, otherwise falls back to fixed sigma.
    """
    density = np.zeros((height, width), dtype=np.float32)
    n = len(head_positions)
    if n == 0:
        return density

    # Adaptive sigma: use 30% of average nearest-neighbour distance
    if n > 3:
        from scipy.spatial import KDTree
        pts_yx = head_positions[:, ::-1].copy()  # (y, x)
        tree = KDTree(pts_yx)
        dists, _ = tree.query(pts_yx, k=min(5, n))
        nn_dists = dists[:, 1:].mean(axis=1)
        sigma = float(np.clip(nn_dists.mean() * 0.3, 5.0, sigma))

    for x, y in head_positions:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))
        density[cy, cx] += 1.0

    density = scipy.ndimage.gaussian_filter(density, sigma=sigma, mode="constant", cval=0)

    # Normalize so density.sum() == n
    dsum = float(density.sum())
    if dsum > 1e-8:
        density = density * (n / dsum)

    return density.astype(np.float32)


# ---------------------------------------------------------------------------
# CNN model (optional — used when --use-model flag is set)
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str | Path | None = None) -> SimpleDensityCNN:
    ckpt = config.abs_path(checkpoint_path if checkpoint_path else config.CHECKPOINT)
    # Prefer mall-tuned model if available
    mall_best = config.abs_path("runs/mall_tuned/best.pt")
    if mall_best.exists():
        ckpt = mall_best
        print(f"Using mall-tuned model: {ckpt}")
    else:
        print(f"Using checkpoint: {ckpt}")
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        bc = payload.get("args", {}).get("base_channels", 16)
        model = SimpleDensityCNN(base_channels=bc)
        model.load_state_dict(payload["model"])
    else:
        model = SimpleDensityCNN()
        model.load_state_dict(payload)
    model.eval()
    return model


@torch.no_grad()
def predict_density(model: SimpleDensityCNN, frame_bgr: np.ndarray, resize_long: int = 512) -> tuple[np.ndarray, float]:
    h, w = frame_bgr.shape[:2]
    long_side = max(h, w)
    if long_side > resize_long:
        scale = resize_long / float(long_side)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        frame_r = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    else:
        frame_r = frame_bgr

    rgb = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(rgb)[None, ...]
    den = model(x).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    count = float(den.sum())
    if den.shape[0] != h or den.shape[1] != w:
        den = cv2.resize(den, (w, h), interpolation=cv2.INTER_LINEAR)
    return den, count


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def render_heatmap_overlay(
    frame_bgr: np.ndarray,
    density: np.ndarray,
    alpha: float = 0.60,
) -> np.ndarray:
    """Overlay vivid density heatmap on top of the original frame.

    Uses COLORMAP_JET on a normalized density map with gamma correction.
    The original frame is kept bright so people are still visible.
    """
    den = density.astype(np.float32).copy()
    den = np.nan_to_num(den, nan=0.0)
    den = np.clip(den, 0.0, None)

    # Percentile normalization — ensures sparse maps still look vivid
    p_high = float(np.percentile(den[den > 0], 95)) if (den > 0).any() else 1.0
    p_high = max(p_high, 1e-8)
    den_norm = np.clip(den / p_high, 0.0, 1.0)

    # Gamma boost — brightens mid-range values
    den_norm = np.power(den_norm, 0.45)

    den_u8 = (den_norm * 255).clip(0, 255).astype(np.uint8)
    h, w = frame_bgr.shape[:2]
    if den_u8.shape[0] != h or den_u8.shape[1] != w:
        den_u8 = cv2.resize(den_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap(den_u8, cv2.COLORMAP_JET)

    # Screen blend: bright heatmap glows on dark areas of the frame
    # Result = 1 - (1-A)*(1-B) — avoids darkening the frame
    frame_f = frame_bgr.astype(np.float32) / 255.0
    heat_f = heat.astype(np.float32) / 255.0
    screen = 1.0 - (1.0 - frame_f) * (1.0 - heat_f * alpha)
    blended = (screen * 255).clip(0, 255).astype(np.uint8)
    return blended


def render_pure_heatmap(
    frame_bgr: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Pure density heatmap with dark scene underlay for context."""
    den = density.astype(np.float32).copy()
    den = np.nan_to_num(den, nan=0.0)
    den = np.clip(den, 0.0, None)

    p_high = float(np.percentile(den[den > 0], 95)) if (den > 0).any() else 1.0
    p_high = max(p_high, 1e-8)
    den_norm = np.clip(den / p_high, 0.0, 1.0)
    den_norm = np.power(den_norm, 0.40)

    den_u8 = (den_norm * 255).clip(0, 255).astype(np.uint8)
    h, w = frame_bgr.shape[:2]
    if den_u8.shape[0] != h or den_u8.shape[1] != w:
        den_u8 = cv2.resize(den_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap(den_u8, cv2.COLORMAP_JET)

    # Darkened original as context underlay
    dark = (frame_bgr.astype(np.float32) * 0.30).clip(0, 255).astype(np.uint8)
    result = cv2.addWeighted(dark, 1.0, heat, 0.85, 0)
    return result


# ---------------------------------------------------------------------------
# Person markers
# ---------------------------------------------------------------------------

def draw_person_markers(
    frame_bgr: np.ndarray,
    head_positions: np.ndarray,
    density: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a marker at every GT head position.

    Uses a tiered visual:
    - Outer glow circle (semi-transparent, color-coded by local density)
    - Inner filled dot (white center)
    - Small index number (optional for debugging)
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if len(head_positions) == 0:
        return out

    # Pre-compute local density at each head (for color coding)
    if density is not None:
        den_small = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR) \
            if density.shape[:2] != (h, w) else density
        p95 = float(np.percentile(den_small[den_small > 0], 95)) if (den_small > 0).any() else 1.0
        p95 = max(p95, 1e-8)
    else:
        p95 = 1.0

    for i, (x, y) in enumerate(head_positions):
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        # Color: cool (blue) = low density, warm (red) = high density
        if density is not None:
            local_d = float(den_small[cy, cx]) / p95
            local_d = min(1.0, local_d)
        else:
            local_d = 0.5

        # JET colormap value: 0=blue(low), 1=red(high)
        color_val = int(local_d * 255)
        color_img = np.array([[[color_val]]], dtype=np.uint8)
        color_bgr = cv2.applyColorMap(color_img, cv2.COLORMAP_JET)[0, 0].tolist()

        # Glow ring (semi-transparent)
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), 10, color_bgr, -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # Solid border ring
        cv2.circle(out, (cx, cy), 9, color_bgr, 2, cv2.LINE_AA)

        # White center dot
        cv2.circle(out, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Queue zone
# ---------------------------------------------------------------------------

def find_best_queue_zone(
    density: np.ndarray,
    head_positions: np.ndarray,
    zone_w_frac: float = 0.22,
    zone_h_frac: float = 0.38,
    exclude_top_frac: float = 0.40,
) -> tuple[int, int, int, int]:
    """Find the open-space rectangle with lowest crowd density.

    Uses GT density for accurate open-space detection.
    Restricts search to floor area (excludes top portion = ceiling/walls).
    """
    h, w = density.shape[:2]
    zone_w = max(30, int(round(w * zone_w_frac)))
    zone_h = max(30, int(round(h * zone_h_frac)))
    top_skip = int(round(h * exclude_top_frac))

    den = density.astype(np.float32)
    den = np.nan_to_num(den, nan=0.0)
    den = np.clip(den, 0.0, None)

    # Slightly smooth for scoring (avoids selecting 1-pixel holes)
    den_s = cv2.GaussianBlur(den, (0, 0), sigmaX=25, sigmaY=25)

    integral = cv2.integral(den_s)

    best_score = float("inf")
    best_rect = (w // 4, top_skip, w // 4 + zone_w, top_skip + zone_h)

    stride = 8
    for y in range(top_skip, h - zone_h, stride):
        for x in range(0, w - zone_w, stride):
            y2, x2 = y + zone_h, x + zone_w
            area_sum = (
                integral[y2, x2] - integral[y, x2] -
                integral[y2, x] + integral[y, x]
            )
            if area_sum < best_score:
                best_score = area_sum
                best_rect = (x, y, x2, y2)

    return best_rect


def draw_queue_zone(
    frame_bgr: np.ndarray,
    density: np.ndarray,
    head_positions: np.ndarray,
    zone_rect: tuple[int, int, int, int],
) -> np.ndarray:
    """Draw premium queue zone overlay with fill, accents, and status badge."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = zone_rect
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)

    # Compute zone crowd density vs global
    if y1 > y0 and x1 > x0:
        zone_den = float(density[y0:y1, x0:x1].mean())
        global_den = float(density.mean())
        ratio = zone_den / (global_den + 1e-9)
    else:
        ratio = 1.0

    if ratio < 0.4:
        status, status_color = "OPTIMAL", (0, 230, 80)
    elif ratio < 0.8:
        status, status_color = "GOOD", (0, 200, 150)
    elif ratio < 1.2:
        status, status_color = "ACCEPTABLE", (0, 180, 255)
    else:
        status, status_color = "CROWDED", (0, 80, 255)

    border_color = (0, 255, 80)

    # Semi-transparent green fill
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 50), -1)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    # Thick border
    cv2.rectangle(out, (x0, y0), (x1, y1), border_color, 3)

    # Corner L-brackets
    cl = min(18, (x1 - x0) // 4, (y1 - y0) // 4)
    ct = 3
    for (bx, by, dx, dy) in [
        (x0, y0, 1, 1), (x1, y0, -1, 1),
        (x0, y1, 1, -1), (x1, y1, -1, -1),
    ]:
        cv2.line(out, (bx, by), (bx + dx * cl, by), (255, 255, 255), ct)
        cv2.line(out, (bx, by), (bx, by + dy * cl), (255, 255, 255), ct)

    # Badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = ["SUGGESTED QUEUE ZONE", status]
    fs1 = max(0.40, min(0.60, w / 1000.0))
    fs2 = max(0.32, min(0.48, w / 1200.0))
    (tw1, th1), _ = cv2.getTextSize(lines[0], font, fs1, 2)
    (tw2, th2), _ = cv2.getTextSize(lines[1], font, fs2, 2)

    bw = max(tw1, tw2) + 20
    bh = th1 + th2 + 22
    # Place badge above zone, clamped horizontally
    bx = max(0, min(x0, w - bw - 2))
    by = max(0, y0 - bh - 6)

    # Badge bg
    badge_ov = out.copy()
    cv2.rectangle(badge_ov, (bx, by), (bx + bw, by + bh), (15, 15, 15), -1)
    cv2.addWeighted(badge_ov, 0.85, out, 0.15, 0, out)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), border_color, 1)

    # Badge text
    cv2.putText(out, lines[0], (bx + 10, by + th1 + 6),
                font, fs1, (180, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(out, lines[1], (bx + 10, by + th1 + th2 + 14),
                font, fs2, status_color, 2, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# HUD overlays
# ---------------------------------------------------------------------------

def _crowd_level(count: int) -> tuple[str, tuple[int, int, int]]:
    threshold = int(config.CROWD_THRESHOLD)   # 35 for Mall dataset
    if count < int(threshold * 0.65):
        return "LOW", (0, 220, 80)
    elif count < int(threshold * 0.88):
        return "MODERATE", (0, 180, 255)
    elif count < threshold:
        return "ELEVATED", (0, 120, 255)
    else:
        return "HIGH", (0, 50, 230)


def draw_hud(
    frame_bgr: np.ndarray,
    count: int,
    frame_idx: int,
    total: int,
    source: str = "GT",
) -> np.ndarray:
    """Professional HUD with count panel, level pill, progress bar, warning."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    level, level_color = _crowd_level(count)
    threshold = int(config.CROWD_THRESHOLD)

    # ── Top progress bar ───────────────────────────────────────────────────
    bar_h = max(5, int(h * 0.011))
    fill_ratio = min(1.0, count / max(threshold * 1.5, 1))
    fill_w = int(w * fill_ratio)
    cv2.rectangle(out, (0, 0), (w, bar_h), (35, 35, 35), -1)
    cv2.rectangle(out, (0, 0), (fill_w, bar_h), level_color, -1)

    # ── Warning banner (if above threshold) ───────────────────────────────
    if count >= threshold:
        banner_h = max(30, int(h * 0.062))
        bov = out.copy()
        cv2.rectangle(bov, (0, bar_h), (w, bar_h + banner_h), (0, 20, 200), -1)
        cv2.addWeighted(bov, 0.88, out, 0.12, 0, out)
        msg = f"ALERT: HIGH CROWD DENSITY  |  {count} people detected"
        fs = max(0.45, min(0.80, w / 800.0))
        (tw, _), _ = cv2.getTextSize(msg, font, fs, 2)
        cv2.putText(out, msg, ((w - tw) // 2, bar_h + int(banner_h * 0.70)),
                    font, fs, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Count panel (bottom-left) ──────────────────────────────────────────
    pw = int(w * 0.28)
    ph = int(h * 0.17)
    px, py = 10, h - ph - 10

    pov = out.copy()
    cv2.rectangle(pov, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
    cv2.addWeighted(pov, 0.80, out, 0.20, 0, out)
    cv2.rectangle(out, (px, py), (px + pw, py + ph), (55, 55, 55), 1)

    # Left accent stripe
    cv2.rectangle(out, (px, py), (px + 4, py + ph), level_color, -1)

    # "CROWD COUNT" label + source tag on same line
    fs_lbl = max(0.27, min(0.40, w / 1400.0))
    src_color = (100, 230, 100) if source == "GT" else (100, 180, 255)
    label_txt = f"CROWD COUNT  [{source}]"
    cv2.putText(out, label_txt, (px + 12, py + int(ph * 0.32)),
                font, fs_lbl, (150, 150, 150), 1, cv2.LINE_AA)

    # Large count number
    count_str = str(count)
    fs_cnt = max(0.9, min(1.8, w / 450.0))
    (tw_c, th_c), _ = cv2.getTextSize(count_str, font, fs_cnt, 3)
    cv2.putText(out, count_str, (px + 12, py + int(ph * 0.82)),
                font, fs_cnt, (255, 255, 255), 3, cv2.LINE_AA)

    # Level pill
    pill_x = px + tw_c + 20
    pill_y = py + int(ph * 0.58)
    pill_w, pill_h = 90, 22
    if pill_x + pill_w < px + pw:
        cv2.rectangle(out, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
                      level_color, -1)
        fs_pill = max(0.27, min(0.36, w / 1600.0))
        (tw_s, _), _ = cv2.getTextSize(level, font, fs_pill, 1)
        cv2.putText(out, level, (pill_x + (pill_w - tw_s) // 2, pill_y + 16),
                    font, fs_pill, (10, 10, 10), 1, cv2.LINE_AA)

    # Frame counter (bottom-right)
    fs_fc = max(0.27, min(0.38, w / 1600.0))
    fc_str = f"Frame {frame_idx + 1}/{total}"
    (tw_f, _), _ = cv2.getTextSize(fc_str, font, fs_fc, 1)
    cv2.putText(out, fc_str, (w - tw_f - 10, h - 10),
                font, fs_fc, (130, 130, 130), 1, cv2.LINE_AA)

    return out


def draw_heatmap_hud(
    frame_bgr: np.ndarray,
    count: int,
    frame_idx: int,
    total: int,
) -> np.ndarray:
    """HUD for pure heatmap frames: bottom bar + count + colorbar legend."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    level, level_color = _crowd_level(count)

    bar_h = max(44, int(h * 0.10))
    bov = out.copy()
    cv2.rectangle(bov, (0, h - bar_h), (w, h), (8, 8, 8), -1)
    cv2.addWeighted(bov, 0.88, out, 0.12, 0, out)

    fs_title = max(0.30, min(0.50, w / 1100.0))
    cv2.putText(out, "DENSITY HEAT MAP", (12, h - bar_h + int(bar_h * 0.38)),
                font, fs_title, (160, 160, 160), 1, cv2.LINE_AA)

    fs_cnt = max(0.42, min(0.72, w / 800.0))
    txt = f"Count: {count}  |  {level}"
    cv2.putText(out, txt, (12, h - int(bar_h * 0.14)),
                font, fs_cnt, level_color, 2, cv2.LINE_AA)

    fs_fc = max(0.25, min(0.36, w / 1600.0))
    fc_str = f"Frame {frame_idx + 1}/{total}"
    (tw_f, _), _ = cv2.getTextSize(fc_str, font, fs_fc, 1)
    cv2.putText(out, fc_str, (w - tw_f - 10, h - bar_h + int(bar_h * 0.50)),
                font, fs_fc, (110, 110, 110), 1, cv2.LINE_AA)

    # Colorbar legend (JET: blue=low → red=high)
    lw = max(100, int(w * 0.16))
    lh = max(10, int(bar_h * 0.28))
    lx = w - lw - 12
    ly = h - bar_h + int(bar_h * 0.18)
    grad = np.tile(np.linspace(0, 255, lw, dtype=np.uint8), (lh, 1))
    grad_c = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    out[ly:ly + lh, lx:lx + lw] = grad_c
    cv2.rectangle(out, (lx, ly), (lx + lw - 1, ly + lh - 1), (70, 70, 70), 1)

    fs_leg = max(0.20, min(0.28, w / 2000.0))
    cv2.putText(out, "Low", (lx, ly + lh + 9), font, fs_leg, (110, 110, 110), 1)
    (tw_h, _), _ = cv2.getTextSize("High", font, fs_leg, 1)
    cv2.putText(out, "High", (lx + lw - tw_h, ly + lh + 9), font, fs_leg, (110, 110, 110), 1)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate annotated + heatmap frames from Mall GT.")
    ap.add_argument("--max-frames", type=int, default=20,
                    help="Max frames to process (0 = all). Default 20.")
    ap.add_argument("--out-dir", type=str, default=config.OUTPUT_DIR)
    ap.add_argument("--mall-root", type=str, default=config.MALL_ROOT)
    ap.add_argument("--use-model", action="store_true",
                    help="Use CNN model predictions instead of GT positions")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--sigma", type=float, default=20.0,
                    help="Gaussian sigma for density map generation")
    args = ap.parse_args()

    out_base = config.abs_path(args.out_dir)
    ann_dir = out_base / config.ANNOTATED_FRAMES_DIR
    heat_dir = out_base / "heatmap_frames"
    ann_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    mall_root = config.abs_path(args.mall_root)
    frame_paths = sorted((mall_root / "frames").glob("seq_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError("No Mall frames found.")

    # Load GT head positions
    mat_path = mall_root / "mall_gt.mat"
    print(f"Loading GT from {mat_path}")
    all_positions, gt_counts = load_mall_gt(mat_path)

    n = min(len(frame_paths), len(all_positions))
    if args.max_frames and args.max_frames > 0:
        n = min(n, args.max_frames)
    frame_paths = frame_paths[:n]

    # Optional: load CNN model
    model = None
    if args.use_model:
        model = _load_model(args.checkpoint)
        print("Using CNN model for density prediction")
    else:
        print("Using GT head positions for density (accurate mode)")

    print(f"Processing {n} frames")
    print(f"  Annotated -> {ann_dir}")
    print(f"  Heatmaps  -> {heat_dir}")
    print("-" * 60)

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [SKIP] {frame_path.name}")
            continue

        H, W = frame.shape[:2]
        head_positions = all_positions[idx]
        gt_count = int(round(float(gt_counts[idx])))

        if args.use_model and model is not None:
            density, cnn_count = predict_density(model, frame)
            display_count = int(round(cnn_count))
            source = "CNN"
        else:
            # Ground truth mode: perfect density + count
            density = make_gaussian_density(H, W, head_positions, sigma=args.sigma)
            display_count = gt_count
            source = "GT"

        # ── Annotated frame: heatmap + person markers + queue zone + HUD ──
        ann = render_heatmap_overlay(frame, density, alpha=0.55)
        ann = draw_person_markers(ann, head_positions, density)
        zone_rect = find_best_queue_zone(density, head_positions)
        ann = draw_queue_zone(ann, density, head_positions, zone_rect)
        ann = draw_hud(ann, display_count, idx, n, source=source)

        ann_path = ann_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(ann_path), ann)

        # ── Pure heatmap frame ─────────────────────────────────────────────
        heat = render_pure_heatmap(frame, density)
        heat = draw_heatmap_hud(heat, display_count, idx, n)

        heat_path = heat_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(heat_path), heat)

        print(
            f"  [{idx+1}/{n}] {frame_path.name} "
            f"gt={gt_count} display={display_count} "
            f"zone={zone_rect}"
        )

    print(f"\nDone! {n} frames saved:")
    print(f"  {ann_dir}")
    print(f"  {heat_dir}")


if __name__ == "__main__":
    main()
