# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [PIPELINE.md](PIPELINE.md) for a full step-by-step walkthrough of the data flow and the actual training results/metrics achieved in this repo.

## Project Overview

QueueSense AI — an ML-powered retail crowd analysis system. It counts people from head-position ground truth, generates Gaussian KDE density heatmaps, suggests optimal queue zones via sliding-window open-space detection, and trains a CNN (`SimpleDensityCNN`) to predict density maps directly from images.

There are two parallel "modes" throughout the codebase:
- **GT mode** (default): density maps and counts are derived directly from ground-truth head-position annotations (`mall_gt.mat`). Always 100% accurate; used for visualization and as training targets.
- **CNN mode** (`--use-model`): density maps come from `SimpleDensityCNN` inference on raw image pixels. This is the only mode that generalizes to images without annotations.

## Environment

- Python virtualenv lives in `.venv/` (Windows, Python 3.10). Activate with `.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat` (cmd).
- Dependencies: `pip install -r requirements.txt`. Note `torch`/`torchvision` are commented as platform-specific — install matching wheels separately if the default pip install doesn't pick the right build (e.g. CUDA vs CPU).
- All scripts insert `PROJECT_ROOT` onto `sys.path`, so they can be run directly from anywhere via `python scripts/<script>.py`, as long as the working directory contains `config.py` or you run from repo root.

## Common Commands

```bash
# 1. Generate GT Gaussian density maps from head positions (one-time, run first)
python scripts/make_density_maps.py
python scripts/make_density_maps.py --sigma 15 --max-frames 200   # quick subset for testing

# 2. Fine-tune SimpleDensityCNN on Mall GT density maps
python scripts/train_mall.py --epochs 40 --lr 5e-4
python scripts/train_mall.py --epochs 5 --batch-size 2            # quick smoke test

# 3. Generate output frames (annotated + pure heatmap)
python scripts/generate_frames.py --max-frames 20            # GT mode (exact counts)
python scripts/generate_frames.py --max-frames 20 --use-model # CNN inference mode
```

There is no test suite, linter, or CI config in this repo — there's nothing to run beyond the scripts above. Validate changes by running the relevant script on a small `--max-frames`/`--epochs` slice and inspecting `outputs/`.

## Architecture

### Data flow
`mall_gt.mat` (head positions, per-frame `(x, y)` pairs + counts) is the single source of truth for the Mall dataset.
1. `scripts/make_density_maps.py` converts head positions → per-frame Gaussian density `.npy` arrays in `datasets/mall_dataset/density_maps/` (sum of each map ≈ GT count; sigma is adaptive — scaled down in dense crowds based on nearest-neighbour spacing, via `scipy.spatial.KDTree`).
2. `scripts/train_mall.py` fine-tunes `SimpleDensityCNN` against those density maps, starting from a ShanghaiTech-pretrained checkpoint (`config.CHECKPOINT_FALLBACK`). Loss is `MSE(density) + 0.001 * L1(count)` (`CombinedLoss`). Saves `runs/mall_tuned/best.pt` / `last.pt`.
3. `scripts/generate_frames.py` is the main product: for each frame it computes a density map (GT-derived or CNN-predicted), then renders a heatmap overlay, per-person markers, a suggested queue zone, and an info HUD, writing to `outputs/annotated_frames/` and `outputs/heatmap_frames/`.

### `crowdcount/` package (reusable library code)
- `model.py` — `SimpleDensityCNN`: U-Net-style encoder/decoder (4 encoder stages with channel widths `c, 2c, 4c, 8c`; decoder with skip connections; final 1×1 conv + ReLU enforces non-negative density). `base_channels` (default 16) is read back from checkpoint `args` when loading, so different-width checkpoints stay loadable.
- `data.py` — `ShanghaiTechDensityDataset` (H5 density-map ground truth, random crop/flip/color-jitter augmentation, resize-while-preserving-count) and `MallCountDataset` (image + scalar count only, no density map — used where only a count target is needed).
- `checkpoint.py` — thin `torch.save`/`torch.load` wrappers (`weights_only=False`, since checkpoints store a dict with `model`/`epoch`/`args`, not just a state_dict).
- `train_utils.py` — `seed_everything`, `get_device` (`"auto"` → CUDA if available else CPU), `AverageMeter`.
- `metrics.py` — `mae`/`rmse` over batches of predicted vs. true counts.
- `baseline.py` — non-ML baseline: OpenCV MOG2 background subtraction + connected-component counting, for comparison against the CNN (`mog2_mae`).

### Checkpoint resolution
`config.CHECKPOINT` points at the Mall-fine-tuned model; `config.CHECKPOINT_FALLBACK` is the original ShanghaiTech-pretrained one. Checkpoint loading logic differs slightly per script — `scripts/generate_frames.py`'s `_load_model` always prefers `runs/mall_tuned/best.pt` if it exists on disk regardless of `--checkpoint`, while `scripts/train_mall.py`'s `load_pretrained` loads via `strict=False` and reads `base_channels` from the checkpoint's stored `args`. Checkpoints are dicts shaped `{"model": state_dict, "epoch": int, "args": {"base_channels": ...}}`.

### Density map invariant
Every density map operation in this codebase is expected to preserve `density.sum() == count` (a Gaussian blob per head, normalized so the map sums to the true count). When resizing a density map (e.g. `resize_density_preserve_count` in `data.py`), always rescale by the area ratio afterward so this invariant holds — don't just `cv2.resize` and stop.

### Known inconsistencies (don't assume one canonical implementation)
- `config.CROWD_THRESHOLD` (35) is calibrated specifically for the Mall dataset's count distribution (mean 31.2, max 53) — don't reuse it as-is for ShanghaiTech-scale crowds.
- `crowdcount/data.py` defines `ShanghaiTechDensityDataset` and `MallCountDataset`, but nothing in `scripts/` imports them — `scripts/train_mall.py` has its own inline `MallDensityDataset` instead, and there is no script in this repo that trains on ShanghaiTech from scratch (the `runs/density_cnn*` checkpoints were produced by a training script that no longer exists here, judging by `args` stored in those checkpoints referencing fields like `shanghai_root`/`lambda_count` that appear nowhere in the current codebase). Treat `runs/density_cnn_tuned/best.pt` as an opaque pretrained starting point, not something reproducible in-repo.
