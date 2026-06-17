# QueueSense AI — Retail Crowd Queue System

## Overview

An ML-powered retail crowd analysis system that:
- **Counts every person** using ground-truth head position annotations
- **Generates vivid density heatmaps** using Gaussian KDE from head positions
- **Suggests optimal queue zones** using sliding-window open-space detection
- **Trains a CNN model** (SimpleDensityCNN) fine-tuned on Mall dataset GT density maps

---

## Project Structure

```
ML-PROJECT-retial-que-system/
├── config.py                        # Central configuration
├── crowdcount/
│   ├── model.py                     # SimpleDensityCNN (U-Net style encoder-decoder)
│   ├── data.py                      # Datasets: ShanghaiTech + Mall
│   ├── baseline.py                  # MOG2 background subtraction baseline
│   ├── metrics.py                   # MAE, RMSE
│   ├── train_utils.py               # Utilities (seed, device, AverageMeter)
│   └── checkpoint.py                # Checkpoint save/load helpers
├── scripts/
│   ├── make_density_maps.py         # Generate GT Gaussian density maps from head positions
│   ├── train_mall.py                # Fine-tune CNN on Mall GT density maps
│   └── generate_frames.py           # Main output generator (annotated + heatmap frames)
├── datasets/
│   ├── mall_dataset/
│   │   ├── frames/                  # seq_000001.jpg ... seq_002000.jpg
│   │   ├── mall_gt.mat              # GT: head positions (x,y) + counts per frame
│   │   └── density_maps/            # Generated: frame_000000.npy ... (Gaussian KDE)
│   └── shanghaitech_with_people_density_map/
├── runs/
│   ├── density_cnn_tuned/           # Original ShanghaiTech-trained checkpoint
│   └── mall_tuned/                  # Mall-fine-tuned checkpoint (best MAE on Mall)
└── outputs/
    ├── annotated_frames/            # Frame + heatmap overlay + person markers + queue zone + HUD
    └── heatmap_frames/              # Pure density heatmap visualization
```

---

## Quick Start

### Step 1 — Generate GT density maps (one-time)
```bash
python scripts/make_density_maps.py
```
Creates `datasets/mall_dataset/density_maps/frame_NNNNNN.npy` for all 2000 frames.
Each map has `density.sum() == gt_count` (perfectly calibrated).

### Step 2 — Fine-tune CNN on Mall data (optional but improves CNN mode)
```bash
python scripts/train_mall.py --epochs 40 --lr 5e-4
```
Saves best model to `runs/mall_tuned/best.pt`.

### Step 3 — Generate output frames
```bash
# GT mode (100% accurate, uses head positions directly)
python scripts/generate_frames.py --max-frames 20

# CNN mode (uses trained model for inference)
python scripts/generate_frames.py --max-frames 20 --use-model
```

---

## Output Description

### `outputs/annotated_frames/frame_NNNN.png`
- **Heatmap overlay**: COLORMAP_JET screen-blended on original frame  
- **Person markers**: Color-coded circle at every GT head position (red=dense, blue=sparse)
- **Queue zone**: Green rectangle in the lowest-density open area, with status badge
- **HUD**: Count panel, crowd level pill, progress bar, alert banner

### `outputs/heatmap_frames/frame_NNNN.png`
- Pure density heatmap on darkened frame
- Bottom bar with count, crowd level, and JET colorbar legend

---

## Configuration (`config.py`)

| Setting | Value | Description |
|---------|-------|-------------|
| `CROWD_THRESHOLD` | 35 | Alert banner shown when count ≥ this |
| `DENSITY_SIGMA` | 20.0 | Gaussian sigma for density maps (pixels) |
| `CHECKPOINT` | `runs/mall_tuned/best.pt` | Primary model checkpoint |
| `CHECKPOINT_FALLBACK` | `runs/density_cnn_tuned/best.pt` | Fallback if mall-tuned not available |

---

## Model Architecture

`SimpleDensityCNN` — U-Net style fully-convolutional encoder-decoder:
- **Encoder**: 4 stages (3→c→2c→4c→8c channels, MaxPool between stages)
- **Decoder**: Skip connections from encoder, bilinear upsampling
- **Head**: 1×1 conv → ReLU (non-negative density output)
- **Loss**: MSE on density map + L1 count loss

---

## Dataset Details

**Mall Dataset** (2000 frames, 640×480):
- Count range: 13–53 people per frame (mean: 31.2)
- Ground truth: exact (x, y) head positions for every person
- Used for: fine-tuning + visualization

**ShanghaiTech Part B** (400 train / 316 test):
- Highly variable crowd scenes
- Ground truth: Gaussian density maps (H5 files)
- Used for: initial CNN pre-training
