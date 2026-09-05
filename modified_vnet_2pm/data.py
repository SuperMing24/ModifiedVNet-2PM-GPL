# SPDX-License-Identifier: GPL-3.0-only
"""Damseh/Tian modified V-Net data adapter.

The released TensorFlow pipeline enumerates 128-cubed training patches on a
64-voxel grid, shuffles the grid once per epoch, and does not use foreground
sampling or random flips.  This dataset preserves that sampling contract for
the project NIfTI layout.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def scan_nii_pairs(data_root: str | Path) -> list[dict[str, str]]:
    """Return deterministic image/mask pairs from a split directory."""
    root = Path(data_root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    pairs: list[dict[str, str]] = []
    if not image_dir.is_dir() or not mask_dir.is_dir():
        return pairs
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.name.endswith((".nii", ".nii.gz")):
            continue
        stem = image_path.name.removesuffix(".gz").removesuffix(".nii")
        for suffix in (".nii.gz", ".nii"):
            mask_path = mask_dir / f"{stem}{suffix}"
            if mask_path.is_file():
                pairs.append(
                    {"image": str(image_path), "mask": str(mask_path), "name": stem}
                )
                break
    return pairs


def load_nii(path: str | Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D NIfTI volume, received {data.shape}")
    return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    minimum = float(volume.min())
    maximum = float(volume.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - minimum) / (maximum - minimum)).astype(np.float32)


def load_split(split_root: str | Path) -> list[dict]:
    pairs = scan_nii_pairs(split_root)
    if not pairs:
        raise FileNotFoundError(f"no NIfTI pairs found under {split_root}")
    return [
        {
            "image": normalize_volume(load_nii(pair["image"])),
            "mask": (load_nii(pair["mask"]) > 0.5).astype(np.float32),
            "name": pair["name"],
        }
        for pair in pairs
    ]


def _official_grid_starts(length: int, patch: int, stride: int) -> tuple[int, ...]:
    """Return the endpoint-clamped starts used by the released data loader."""
    if patch < 1 or stride < 1:
        raise ValueError("patch and stride must be positive")
    padded_length = max(int(length), int(patch))
    raw = range(0, padded_length - stride, stride)
    starts = tuple(dict.fromkeys(min(value, padded_length - patch) for value in raw))
    return starts or (0,)


class TianDamsehGridPatchDataset3D(Dataset):
    """Enumerate the released 128-cubed, stride-64 training grid."""

    def __init__(
        self,
        volumes: list[dict],
        patch_size: Sequence[int] = (128, 128, 128),
        stride: Sequence[int] = (64, 64, 64),
    ) -> None:
        if not volumes:
            raise ValueError("volumes must not be empty")
        self.patch_size = tuple(int(value) for value in patch_size)
        self.stride = tuple(int(value) for value in stride)
        if self.patch_size != (128, 128, 128):
            raise ValueError("Tian/Damseh training requires patch_size (128, 128, 128)")
        if self.stride != (64, 64, 64):
            raise ValueError("Tian/Damseh training requires stride (64, 64, 64)")

        self.volumes = []
        self.indices: list[tuple[int, int, int, int]] = []
        patch_d, patch_h, patch_w = self.patch_size
        stride_d, stride_h, stride_w = self.stride

        for volume_index, volume in enumerate(volumes):
            image = self._pad_to_patch(volume["image"])
            mask = self._pad_to_patch(volume["mask"])
            self.volumes.append(dict(volume, image=image, mask=mask))
            height, width, depth = image.shape
            h_starts = _official_grid_starts(height, patch_h, stride_h)
            w_starts = _official_grid_starts(width, patch_w, stride_w)
            d_starts = _official_grid_starts(depth, patch_d, stride_d)
            self.indices.extend(
                (volume_index, h0, w0, d0)
                for h0, w0, d0 in itertools.product(h_starts, w_starts, d_starts)
            )

    def _pad_to_patch(self, array: np.ndarray) -> np.ndarray:
        patch_d, patch_h, patch_w = self.patch_size
        height, width, depth = array.shape
        pads = (
            (0, max(0, patch_h - height)),
            (0, max(0, patch_w - width)),
            (0, max(0, patch_d - depth)),
        )
        if not any(after for _, after in pads):
            return array
        return np.pad(array, pads, mode="constant")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume_index, h0, w0, d0 = self.indices[index]
        volume = self.volumes[volume_index]
        patch_d, patch_h, patch_w = self.patch_size
        slices = (
            slice(h0, h0 + patch_h),
            slice(w0, w0 + patch_w),
            slice(d0, d0 + patch_d),
        )
        image = np.ascontiguousarray(volume["image"][slices].transpose(2, 0, 1))
        mask = np.ascontiguousarray(volume["mask"][slices].transpose(2, 0, 1))
        return (
            torch.from_numpy(image).float().unsqueeze(0),
            torch.from_numpy(mask).float().unsqueeze(0),
        )
