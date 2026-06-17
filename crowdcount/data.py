from __future__ import annotations

import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from torchvision.transforms import v2 as T

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor_normalized(img_rgb: np.ndarray) -> torch.Tensor:
    x = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(x).float()


def resize_density_preserve_count(
    img_rgb: np.ndarray, density: np.ndarray, resize_long: int
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = img_rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= resize_long:
        return img_rgb, density

    scale = resize_long / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if density is not None:
        den_resized = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Rescale so that den_resized.sum() == density.sum()  (preserve total count)
        den_resized = den_resized * ((h * w) / float(new_h * new_w))
        den_resized = den_resized.astype(np.float32)
    else:
        den_resized = None

    return img_resized, den_resized


def random_crop_pair(
    img_rgb: np.ndarray, density: np.ndarray, crop_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img_rgb = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        density = np.pad(density, ((0, pad_h), (0, pad_w)), mode="constant")
        h, w = img_rgb.shape[:2]

    y0 = rng.integers(0, h - crop_size + 1)
    x0 = rng.integers(0, w - crop_size + 1)

    img_c = img_rgb[y0 : y0 + crop_size, x0 : x0 + crop_size]
    den_c = density[y0 : y0 + crop_size, x0 : x0 + crop_size]

    return img_c, den_c


@dataclass
class ShanghaiTechPaths:
    images_dir: Path
    density_h5_dir: Path


def shanghaitech_partb_paths(root: str | Path, split: Literal["train", "test"]) -> ShanghaiTechPaths:
    root = Path(root)
    sub = "train_data" if split == "train" else "test_data"
    base = root / sub
    return ShanghaiTechPaths(images_dir=base / "images", density_h5_dir=base / "ground-truth-h5")


class ShanghaiTechDensityDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"],
        crop_size: int | None = 256,
        hflip_p: float = 0.5,
        color_jitter: float = 0.2,
        resize_long: int | None = None,
        seed: int = 42,
    ):
        self.split = split
        self.crop_size = crop_size if split == "train" else None
        self.resize_long = resize_long
        self.rng = np.random.default_rng(seed)
        self.hflip_p = hflip_p if split == "train" else 0.0

        p = shanghaitech_partb_paths(root, split)
        self.images_dir = p.images_dir
        self.density_dir = p.density_h5_dir

        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        if self.split == "train":
            self.augment = T.Compose(
                [
                    T.ToImage(),
                    T.ColorJitter(
                        brightness=color_jitter,
                        contrast=color_jitter,
                        saturation=color_jitter,
                    ),
                    T.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.augment = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        name = img_path.stem
        h5_path = self.density_dir / f"{name}.h5"

        img_rgb = _read_rgb(img_path)
        with h5py.File(h5_path, "r") as f:
            density = np.array(f["density"], dtype=np.float32)

        # Ensure density map matches image spatial dims (guards against mismatched H5)
        ih, iw = img_rgb.shape[:2]
        dh, dw = density.shape
        if dh != ih or dw != iw:
            density = cv2.resize(density, (iw, ih), interpolation=cv2.INTER_LINEAR)
            density = density * ((dh * dw) / float(ih * iw))

        if self.resize_long is not None:
            img_rgb, density = resize_density_preserve_count(img_rgb, density, self.resize_long)

        if self.split == "train" and self.crop_size is not None:
            img_rgb, density = random_crop_pair(img_rgb, density, self.crop_size, self.rng)

        # Random horizontal flip — flip BOTH image and density together
        if self.split == "train" and self.rng.random() < self.hflip_p:
            img_rgb = np.fliplr(img_rgb).copy()
            density = np.fliplr(density).copy()

        if self.augment:
            img_tensor = self.augment(img_rgb)
            x = T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())(img_tensor)
        else:
            x = _to_tensor_normalized(img_rgb)

        y = torch.from_numpy(density[None, ...]).float()
        gt_count = float(density.sum())

        return {
            "image": x,
            "density": y,
            "gt_count": gt_count,
            "path": str(img_path),
        }


class MallCountDataset(Dataset):
    def __init__(self, root: str | Path, resize_long: int | None = None):
        self.root = Path(root)
        self.resize_long = resize_long

        frames_dir = self.root / "frames"
        self.image_paths = sorted(frames_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No frames found in {frames_dir}")

        import scipy.io

        mat = scipy.io.loadmat(self.root / "mall_gt.mat")
        self.counts = mat["count"].squeeze().astype(np.float32).tolist()

        if len(self.image_paths) != len(self.counts):
            n = min(len(self.image_paths), len(self.counts))
            self.image_paths = self.image_paths[:n]
            self.counts = self.counts[:n]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        gt_count = float(self.counts[idx])

        img_rgb = _read_rgb(img_path)

        if self.resize_long is not None:
            h, w = img_rgb.shape[:2]
            long_side = max(h, w)
            if long_side > self.resize_long:
                scale = self.resize_long / float(long_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x = _to_tensor_normalized(img_rgb)

        return {
            "image": x,
            "gt_count": gt_count,
            "path": str(img_path),
        }
