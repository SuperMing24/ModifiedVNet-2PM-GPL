# SPDX-License-Identifier: GPL-3.0-only
"""PyTorch architecture port of the Tian/Damseh two-photon V-Net.

Upstream reference:
    https://github.com/bu-cisl/2PM_Vascular_Segmentation_DNN
    revision e56562d303aedfc90124d5a25fba96f325562a82
    GPL-3.0

This standalone GPL package keeps the architecture separate from the host
research project. It may only be redistributed under GPL-3.0-only.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

_UPSTREAM_REVISION = "e56562d303aedfc90124d5a25fba96f325562a82"


class _ResidualConvBlock3d(nn.Module):
    """Contracting-path block with the upstream ReLU -> BatchNorm order."""

    def __init__(self, channels: int, num_convolutions: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                    nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01),
                )
                for _ in range(num_convolutions)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual


class _ResidualMergeBlock3d(nn.Module):
    """Expanding-path block matching the upstream asymmetric first convolution."""

    def __init__(self, channels: int, num_convolutions: int) -> None:
        super().__init__()
        self.first = nn.Conv3d(2 * channels, channels, kernel_size=3, padding=1, bias=True)
        self.remaining = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                    nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01),
                )
                for _ in range(num_convolutions - 1)
            ]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.first(torch.cat((x, skip), dim=1))
        for layer in self.remaining:
            x = layer(x)
        return x + residual


class _DownConv3d(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv3d(channels, 2 * channels, kernel_size=2, stride=2, bias=True),
            nn.ReLU(),
        )


class _UpConv3d(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.ConvTranspose3d(channels, channels // 2, kernel_size=2, stride=2, bias=True),
            nn.ReLU(),
        )


class TianDamsehVNet(nn.Module):
    """Lightweight three-level 3D V-Net used for two-photon vasculature."""

    upstream_revision = _UPSTREAM_REVISION

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        self.input_projection = (
            None
            if in_channels == 1
            else nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=True)
        )

        c = base_channels
        self.contract1 = _ResidualConvBlock3d(c, 1)
        self.down1 = _DownConv3d(c)
        self.contract2 = _ResidualConvBlock3d(2 * c, 2)
        self.down2 = _DownConv3d(2 * c)
        self.contract3 = _ResidualConvBlock3d(4 * c, 3)
        self.down3 = _DownConv3d(4 * c)
        self.bottleneck = _ResidualConvBlock3d(8 * c, 3)

        self.up3 = _UpConv3d(8 * c)
        self.expand3 = _ResidualMergeBlock3d(4 * c, 3)
        self.up2 = _UpConv3d(4 * c)
        self.expand2 = _ResidualMergeBlock3d(2 * c, 2)
        self.up1 = _UpConv3d(2 * c)
        self.expand1 = _ResidualMergeBlock3d(c, 1)
        self.output_conv = nn.Conv3d(c, out_channels, kernel_size=1, bias=True)

        self._initialize_upstream_parameters()

    def _initialize_upstream_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 1.0)

    @staticmethod
    def _validate_spatial_shape(shape: Sequence[int]) -> None:
        if len(shape) != 3 or any(size % 8 != 0 for size in shape):
            raise ValueError(
                "TianDamsehVNet requires each spatial dimension to be divisible by 8; "
                f"received {tuple(shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected NCDHW input, received shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, received {x.shape[1]}"
            )
        self._validate_spatial_shape(x.shape[2:])

        if self.input_projection is None:
            c0 = x.repeat(1, self.base_channels, 1, 1, 1)
        else:
            c0 = F.relu(self.input_projection(x))

        c1 = self.contract1(c0)
        c2 = self.contract2(self.down1(c1))
        c3 = self.contract3(self.down2(c2))
        c4 = self.bottleneck(self.down3(c3))

        e3 = self.expand3(self.up3(c4), c3)
        e2 = self.expand2(self.up2(e3), c2)
        e1 = self.expand1(self.up1(e2), c1)
        return self.output_conv(e1)


def build_tian_damseh_vnet(in_channels: int = 1, cfg: dict | None = None) -> nn.Module:
    cfg = cfg or {}
    return TianDamsehVNet(
        in_channels=in_channels,
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("tian_damseh_base_channels", 4)),
    )
