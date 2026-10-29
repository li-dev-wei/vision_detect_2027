"""Defect morphology aware refinement (DMARE).

DMARE combines directional features for cracks/scratches with a local depthwise
branch for point-like defects such as pinholes.  The gate is deliberately
lightweight so the block can be used in the YOLO11n small-object head.
"""

from __future__ import annotations

import torch
from torch import nn


class DMARE(nn.Module):
    """Morphology-aware feature refinement with residual fusion.

    Args:
        c1: Input channel count.
        c2: Output channel count.
        kernel_h: Vertical directional kernel size.
        kernel_w: Horizontal directional kernel size.
        reduction: Hidden-channel ratio in the gate.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_h: int = 7,
        kernel_w: int = 7,
        reduction: float = 0.5,
    ) -> None:
        super().__init__()
        if kernel_h % 2 == 0 or kernel_w % 2 == 0:
            raise ValueError("DMARE kernels must be odd")
        hidden = max(8, int(c2 * reduction))
        self.project = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.direction_h = nn.Sequential(
            nn.Conv2d(c2, c2, (1, kernel_w), padding=(0, kernel_w // 2), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.direction_v = nn.Sequential(
            nn.Conv2d(c2, c2, (kernel_h, 1), padding=(kernel_h // 2, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c2 * 3, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(x)
        directional_h = self.direction_h(residual)
        directional_v = self.direction_v(residual)
        local = self.local(residual)
        fused = self.fuse(torch.cat((directional_h, directional_v, local), dim=1))
        return residual + fused * self.gate(fused)
