# SPDX-License-Identifier: GPL-3.0-only
"""Balanced BCE plus 3D total variation used by the Tian/Damseh model.

Ported from the TensorFlow reference implementation at:
https://github.com/bu-cisl/2PM_Vascular_Segmentation_DNN
revision e56562d303aedfc90124d5a25fba96f325562a82.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TianDamsehTVBCELoss(nn.Module):
    """Faithful NCDHW port of the upstream balanced BCE + Sobel TV loss."""

    def __init__(self, alpha: float = 5e-9, balance_epsilon: float = 1e-3) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if balance_epsilon <= 0:
            raise ValueError("balance_epsilon must be positive")
        self.alpha = float(alpha)
        self.balance_epsilon = float(balance_epsilon)

        derivative = torch.tensor(
            [
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            ],
            dtype=torch.float32,
        )
        kernels = torch.stack(
            (
                derivative,
                derivative.permute(2, 0, 1),
                derivative.permute(1, 2, 0),
            ),
            dim=0,
        ).unsqueeze(1)
        self.register_buffer("sobel_kernels", kernels, persistent=False)

    def balanced_bce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.detach()
        positive_rate = labels.mean()
        positive_weight = (1.0 - positive_rate) / (
            positive_rate + self.balance_epsilon
        )
        weighted = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=positive_weight,
            reduction="none",
        )
        return (weighted * positive_rate).mean()

    def total_variation(self, logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        kernels = self.sobel_kernels.to(
            device=probability.device,
            dtype=probability.dtype,
        )
        return F.conv3d(probability, kernels, stride=1, padding=0).abs().sum()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape != labels.shape:
            raise ValueError(
                f"logits and labels must have the same shape: "
                f"{tuple(logits.shape)} != {tuple(labels.shape)}"
            )
        if logits.ndim != 5 or logits.shape[1] != 1:
            raise ValueError(
                "TianDamsehTVBCELoss expects single-channel NCDHW tensors"
            )
        if any(size < 3 for size in logits.shape[2:]):
            raise ValueError("each spatial dimension must be at least 3")

        return self.balanced_bce(logits, labels) + self.alpha * self.total_variation(
            logits
        )
