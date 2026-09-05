# SPDX-License-Identifier: GPL-3.0-only
"""Frozen full-volume metrics used only for checkpoint selection."""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def dice_3d(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    prediction = probability > threshold
    truth = target.astype(bool)
    denominator = int(prediction.sum()) + int(truth.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, truth).sum() / denominator)


def cldice_3d(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> float:
    prediction = probability > threshold
    truth = target.astype(bool)
    if not prediction.any() and not truth.any():
        return 1.0
    if not prediction.any() or not truth.any():
        return 0.0
    skeleton_prediction = skeletonize(prediction)
    skeleton_truth = skeletonize(truth)
    smooth = 1e-6
    topology_precision = (
        np.logical_and(skeleton_prediction, truth).sum() + smooth
    ) / (skeleton_prediction.sum() + smooth)
    topology_sensitivity = (
        np.logical_and(skeleton_truth, prediction).sum() + smooth
    ) / (skeleton_truth.sum() + smooth)
    return float(
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity + 1e-8)
    )


def checkpoint_score(rows: list[dict[str, float]]) -> dict[str, float]:
    mean_dice = float(np.mean([row["dice_3d"] for row in rows]))
    mean_cldice = float(np.mean([row["cl_dice_3d"] for row in rows]))
    harmonic = (
        2.0 * mean_dice * mean_cldice / (mean_dice + mean_cldice)
        if mean_dice + mean_cldice > 0
        else 0.0
    )
    return {
        "dice_3d": mean_dice,
        "cl_dice_3d": mean_cldice,
        "checkpoint_score": harmonic,
    }
