"""Defect morphology-aware refinement (DMARE).

The implementation follows the project proposal: line-like defects are modeled
at two scales in the horizontal, vertical, 45-degree, and 135-degree directions;
a local branch preserves point-like pinhole features; and a spatial gate chooses
between the two responses at each feature-map location.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class DiagonalDepthwiseConv(nn.Module):
    """Compact learnable depthwise convolution on one image diagonal."""

    def __init__(self, channels: int, kernel_size: int, anti_diagonal: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.anti_diagonal = anti_diagonal
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        padded = F.pad(x, (padding, padding, padding, padding))
        height, width = x.shape[-2:]
        output = torch.zeros_like(x)
        for index in range(self.kernel_size):
            column = self.kernel_size - 1 - index if self.anti_diagonal else index
            sampled = padded[:, :, index : index + height, column : column + width]
            output = output + sampled * self.weight[:, :, index].view(1, self.channels, 1, 1)
        return output


def _directional_branch(channels: int, kernel_size: int, direction: str) -> nn.Sequential:
    if direction == "horizontal":
        convolution: nn.Module = nn.Conv2d(
            channels,
            channels,
            (1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=channels,
            bias=False,
        )
    elif direction == "vertical":
        convolution = nn.Conv2d(
            channels,
            channels,
            (kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=channels,
            bias=False,
        )
    elif direction == "diagonal":
        convolution = DiagonalDepthwiseConv(channels, kernel_size)
    elif direction == "anti_diagonal":
        convolution = DiagonalDepthwiseConv(channels, kernel_size, anti_diagonal=True)
    else:
        raise ValueError(f"Unsupported DMARE direction: {direction}")
    return nn.Sequential(convolution, nn.BatchNorm2d(channels), nn.SiLU(inplace=True))


class DMARE(nn.Module):
    """Morphology-aware feature refinement with residual fusion.

    Args:
        c1: Input channel count.
        c2: Output channel count.
        kernel_small: Short directional kernel length.
        kernel_large: Long directional kernel length.
        reduction: Hidden-channel ratio in the gate.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_small: int = 7,
        kernel_large: int = 11,
        reduction: float = 0.5,
    ) -> None:
        super().__init__()
        kernel_sizes = (kernel_small, kernel_large)
        if any(kernel < 3 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("DMARE kernels must be odd integers of at least 3")
        if not 0 < reduction <= 1:
            raise ValueError("DMARE reduction must be in the interval (0, 1]")
        hidden = max(8, int(c2 * reduction))
        self.project = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        directions = ("horizontal", "vertical", "diagonal", "anti_diagonal")
        self.directional_branches = nn.ModuleList(
            _directional_branch(c2, kernel, direction)
            for kernel in kernel_sizes
            for direction in directions
        )
        self.directional_fuse = nn.Sequential(
            nn.Conv2d(c2 * len(self.directional_branches), c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c2 * 2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
        )
        self.output = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(x)
        directional = self.directional_fuse(
            torch.cat([branch(residual) for branch in self.directional_branches], dim=1)
        )
        local = self.local(residual)
        weights = self.gate(torch.cat((directional, local), dim=1)).softmax(dim=1)
        fused = directional * weights[:, 0:1] + local * weights[:, 1:2]
        return residual + self.output(fused)
