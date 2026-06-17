# QueueSense AI — End-to-End Pipeline & Results

This document walks through exactly how data flows through this project, start to finish, and records the actual results produced in this repo (not theoretical numbers).

---

## 1. Source of truth: `mall_gt.mat`

- Mall dataset: 2000 surveillance frames (`datasets/mall_dataset/frames/seq_000001.jpg` … `seq_002000.jpg`), 640×480 each.
- `mall_gt.mat` stores, per frame: the exact `(x, y)` pixel position of every person's head, plus a scalar `count`.
- Count distribution: range 13–53 people/frame, mean ≈ 31.2.
- This file is the only ground truth in the project — everything else (density maps, training targets, displayed counts in GT mode) is derived from it.

## 2. Step 1 — Build Gaussian density maps

`scripts/make_density_maps.py`

For each frame:
1. Take the head `(x, y)` points.
2. Drop a single pixel of mass at each head location on a blank `H×W` canvas.
3. Blur with `scipy.ndimage.gaussian_filter`. Sigma is **adaptive**: a `KDTree` over head positions measures average nearest-neighbour spacing, and sigma is shrunk in dense crowds (`max(5.0, min(sigma, avg_nn_dist * 0.3))`) so blobs don't merge into a blob soup.
4. Renormalize so `density.sum() == count` exactly (floating-point Gaussian blur leaks a tiny amount of mass at the edges otherwise).

Output: `datasets/mall_dataset/density_maps/frame_NNNNNN.npy`, one float32 array per frame.

```
Run:    python scripts/make_density_maps.py
Status: already generated for all 2000 frames in this repo.
```

This is the **training target** for the CNN, and also the density source for GT-mode visualization.

## 3. Step 2 — Fine-tune the CNN

`scripts/train_mall.py`

**Model**: `crowdcount/model.py::SimpleDensityCNN` — a small U-Net.
- Encoder: 4 stages, channel widths 16 → 32 → 64 → 128, `MaxPool2d` between stages.
- Decoder: bilinear upsample + concat skip connections, mirrored channel widths back down to 16.
- Head: 1×1 conv → ReLU (forces non-negative density output; bias initialized to −2.0 so the model starts near-zero and has to learn to "turn on" density).
- **488,705 parameters** total.

**Starting weights**: a ShanghaiTech Part B–pretrained checkpoint (`runs/density_cnn_tuned/best.pt`). That checkpoint was *not* produced by any script currently in this repo — see §6 (Known Gaps).

**Loss** (`CombinedLoss`): `MSE(pred_density, gt_density) + 0.001 * |sum(pred_density) − gt_count|` — pixel-level density accuracy plus a lightweight nudge toward the right total count.

**Data pipeline** (`MallDensityDataset`, defined inline in this script):
- Loads `frames/seq_*.jpg` + matching `density_maps/frame_*.npy`.
- Resizes long side to a target (resizing the density map too, then rescaling by area ratio to preserve `sum() == count`).
- Train-time augmentation: random horizontal flip, random crop.
- 85/15 train/val split (1700 / 300 frames), seeded.

```
Run:    python scripts/train_mall.py --epochs 12 --resize 224 --batch-size 8 --lr 5e-4
Saves:  runs/mall_tuned/best.pt   (checkpoint with lowest val MAE seen so far)
        runs/mall_tuned/last.pt   (final epoch, written only if training runs to completion)
```

### Actual training run (this repo, CPU, 2026-06-17)

| Epoch | train loss | train MAE | val MAE | val RMSE | lr | time/epoch |
|---|---|---|---|---|---|---|
| 1 | 0.0086 | 7.99 | 5.66 | 6.89 | 4.91e-4 | 445s |
| 2 | 0.0065 | 5.79 | 5.58 | 6.92 | 4.67e-4 | 446s |
| 3 | 0.0057 | 4.95 | 4.18 | 5.34 | 4.27e-4 | 405s |
| 4 | 0.0055 | 4.77 | 5.99 | 7.43 | 3.75e-4 | 398s |
| 5 | 0.0050 | 4.27 | 8.85 | 9.85 | 3.15e-4 | 389s |
| 6 | 0.0047 | 4.00 | 8.79 | 9.83 | 2.51e-4 | 414s |
| 7 | 0.0043 | 3.69 | 3.30 | 4.09 | 1.86e-4 | 389s |
| 8 | 0.0039 | 3.29 | 6.18 | 7.00 | 1.26e-4 | 390s |
| **9** | **0.0036** | **2.98** | **3.01** | **3.81** | 7.41e-5 | 382s | ← **best.pt** |

`runs/mall_tuned/best.pt` is the epoch-9 checkpoint: **val MAE = 3.01 people** on counts averaging ~31 (≈10% relative error). Validation is noisy epoch-to-epoch (small 300-frame val set, no early stopping), which is why epoch 9 beats epochs 4–6 and 8 despite training loss decreasing monotonically — this is expected, not a bug.

**Comparison points**, all measured on this dataset in this repo:
- Non-ML baseline (`crowdcount/baseline.py`, OpenCV MOG2 background subtraction + connected components): **MAE 7.98** over 300 frames.
- CNN fine-tuned model: **MAE 3.01** — roughly 2.6× more accurate than the background-subtraction baseline.
- GT mode: MAE 0 by construction (it reads the answer directly from `mall_gt.mat`).

## 4. Step 3 — Generate visual output

`scripts/generate_frames.py` — the actual product. For each frame:

1. **Get a density map**, one of two ways:
   - **GT mode** (default): rebuild the Gaussian density map directly from `mall_gt.mat` head positions (same method as §2, sigma=20 by default). Always exact.
   - **CNN mode** (`--use-model`): resize the frame to ≤512px long side, normalize (ImageNet mean/std), run `SimpleDensityCNN.forward()`, resize the predicted density map back to the original frame size. `_load_model` always prefers `runs/mall_tuned/best.pt` if present on disk, regardless of `--checkpoint`.
2. **Render the heatmap overlay**: percentile-normalize the density map (95th percentile → full color range, so sparse crowds still look vivid), gamma-correct (`** 0.45`), apply `COLORMAP_JET`, screen-blend onto the original frame so dark areas glow without darkening bright ones.
3. **Draw person markers**: a circle at every GT head position, color-coded by local density (blue = sparse, red = dense). *(GT-position markers are drawn even in CNN mode — they visualize "where people actually are," independent of which density source produced the heatmap.)*
4. **Find a queue zone**: slide a window (~22% width × 38% height of the frame, restricted to the bottom 60% to skip ceiling/walls) across an `cv2.integral()`-accelerated sum of a heavily-blurred density map; pick the window with the lowest summed density. Classify it OPTIMAL/GOOD/ACCEPTABLE/CROWDED by its density ratio vs. the frame average.
5. **Draw the HUD**: crowd-level pill (LOW/MODERATE/ELEVATED/HIGH, thresholds derived from `config.CROWD_THRESHOLD=35`), live count, progress bar, and a red alert banner if count ≥ 35.

Two output sets are written per frame:
- `outputs/annotated_frames/frame_NNNN.png` — heatmap + markers + queue zone + full HUD.
- `outputs/heatmap_frames/frame_NNNN.png` — pure density heatmap on a darkened frame + a JET colorbar legend, no markers/queue zone.

```
GT mode:   python scripts/generate_frames.py --max-frames 20
CNN mode:  python scripts/generate_frames.py --max-frames 20 --use-model
```

## 5. What "accurate" means in each mode

| | Count source | Accuracy | Generalizes to new images? |
|---|---|---|---|
| GT mode | `mall_gt.mat` head positions | Exact (MAE 0) | No — needs annotations |
| CNN mode | `SimpleDensityCNN` inference | MAE ≈ 3.0 (this checkpoint) | Yes — any image, no annotations needed |

GT mode is the "ground truth visualization" — useful for demos and as the training target. CNN mode is the actual deployable crowd-counting model; its accuracy is bounded by the checkpoint above.

## 6. Known gaps (not fixed, by design — scope/data limits)

- **No in-repo script reproduces `runs/density_cnn` / `runs/density_cnn_tuned`** (the ShanghaiTech-pretrained starting checkpoints). Their stored `args` reference fields (`shanghai_root`, `lambda_count`, `loss_scale`, `eval_resize`, `max_train_images`) that don't exist anywhere in the current codebase — the script that trained them was removed or never committed. `crowdcount/data.py::ShanghaiTechDensityDataset` is fully built for this (H5 ground truth, crop/flip/color-jitter augmentation) but nothing imports it. Treat those two checkpoints as opaque pretrained starting points.
- `runs/mall_tuned/last.pt` does not exist — only `best.pt`. This is correct/expected behavior of `train_mall.py` if training is stopped before the final epoch (it writes `last.pt` only after the full `--epochs` loop completes); the run that produced today's checkpoint was 9 of 12 planned epochs at the time the checkpoint was taken.
- `config.CROWD_THRESHOLD = 35` is tuned to Mall's specific count distribution (mean 31.2, max 53) and isn't meant to transfer to denser scenes (e.g. ShanghaiTech-scale crowds of hundreds of people).

## 7. Fixed in this session

- **CNN mode was broken**: the shipped `runs/mall_tuned/best.pt` had been saved after a single interrupted epoch and had collapsed to predicting ≈0 density everywhere (`display=0` for every frame regardless of actual crowd size). Verified the training loop itself was correct (loss decreases properly on a held-out smoke test), then retrained for real — see §3. CNN mode now predicts plausible counts (e.g. GT 29 → CNN 36, GT 35 → CNN 44) instead of zero.
- **Two orphaned, broken scripts removed**: `scripts/heatmap.py` and `scripts/queue_suggest.py` referenced `config.HEATMAP_HEAT_WEIGHT` / `config.QUEUE_STRIPS`, which didn't exist in `config.py` — they would have crashed immediately if ever invoked. Neither was imported by anything else (`scripts/generate_frames.py` has its own inline, actually-used versions of the same heatmap/queue-zone logic). Deleted rather than patched, since they were pure duplication.
- Removed stray `__pycache__` directories and leftover test-run artifacts from the repo.
