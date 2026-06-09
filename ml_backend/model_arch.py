"""
Custom object-detection architecture, defined from raw PyTorch layers and
trained FROM SCRATCH (random initialization, no pretrained weights).

Design: a compact single-stage, anchor-based grid detector ("ScratchDet") in the
spirit of YOLOv2/v3 but written here from primitive Conv/BN/activation blocks so
nothing is borrowed. See TRAINING.md for the full rationale.

Pipeline shape:
    image [B,3,IMG,IMG]
      -> Darknet-style CNN backbone (4x stride-2 downsamples -> stride 16)
      -> detection neck (a few 3x3 convs)
      -> 1x1 head conv producing A*(5+nc) channels on an S x S grid
    where S = IMG/16, A = number of anchors per cell,
    and each anchor predicts [tx, ty, tw, th, objectness, class_1..class_nc].

Three width/depth variants ('tiny' | 'small' | 'medium') let you run the
"train several, keep the best" experiment within the from-scratch constraint.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Anchor priors (w, h) normalized to image size [0,1]. Three shapes covering
# small / medium / large objects. Refine with k-means on your boxes later
# (see scripts/kmeans_anchors.py) — these are sensible construction defaults.
DEFAULT_ANCHORS = [(0.06, 0.08), (0.15, 0.18), (0.35, 0.40)]

# channels per stage (stem, s1, s2, s3) and residual-block counts (s1, s2, s3)
VARIANTS = {
    "tiny":   {"channels": (16, 32, 64, 128),  "depth": (1, 1, 1)},
    "small":  {"channels": (24, 48, 96, 192),  "depth": (1, 2, 2)},
    "medium": {"channels": (32, 64, 128, 256), "depth": (2, 2, 3)},
}

STRIDE = 16  # total downsampling factor of the backbone


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm -> SiLU. The fundamental block; nothing pretrained."""

    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """1x1 squeeze -> 3x3 expand with a skip connection (Darknet bottleneck)."""

    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(ConvBNAct(c, c // 2, 1), ConvBNAct(c // 2, c, 3))

    def forward(self, x):
        return x + self.block(x)


class ScratchDet(nn.Module):
    """
    Custom single-scale anchor-based detector, randomly initialized.

    Args:
        num_classes: current class count (head is sized to this; rebuilt when
                     classes are added mid-annotation).
        variant:     'tiny' | 'small' | 'medium'
        anchors:     list of (w,h) normalized anchor priors.
    """

    def __init__(self, num_classes: int, variant: str = "small", anchors=None):
        super().__init__()
        cfg = VARIANTS[variant]
        c0, c1, c2, c3 = cfg["channels"]
        d1, d2, d3 = cfg["depth"]

        self.num_classes = num_classes
        self.anchors = anchors or DEFAULT_ANCHORS
        self.num_anchors = len(self.anchors)
        self.stride = STRIDE
        self.variant = variant
        self.no = 5 + num_classes  # outputs per anchor

        def stage(c_in, c_out, n_res):
            layers = [ConvBNAct(c_in, c_out, 3, s=2)]  # downsample x2
            layers += [ResidualBlock(c_out) for _ in range(n_res)]
            return nn.Sequential(*layers)

        # Backbone: stem (/2) + 3 stages (/16 total)
        self.stem = ConvBNAct(3, c0, 3, s=2)          # /2
        self.stage1 = stage(c0, c1, d1)               # /4
        self.stage2 = stage(c1, c2, d2)               # /8
        self.stage3 = stage(c2, c3, d3)               # /16

        # Neck: mix features before prediction
        self.neck = nn.Sequential(ConvBNAct(c3, c3, 3), ConvBNAct(c3, c3, 3))

        # Head: 1x1 conv -> A*(5+nc) raw logits
        self.head = nn.Conv2d(c3, self.num_anchors * self.no, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias the objectness channel negative so early training isn't flooded
        # with false positives (standard detector init trick).
        b = self.head.bias.view(self.num_anchors, self.no)
        with torch.no_grad():
            b[:, 4] = -4.0
        self.head.bias = nn.Parameter(b.view(-1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.neck(x)
        x = self.head(x)  # [B, A*no, S, S]
        B, _, S, _ = x.shape
        # -> [B, A, S, S, no]  (no = 5 + nc)
        return x.view(B, self.num_anchors, self.no, S, S).permute(0, 1, 3, 4, 2).contiguous()


def build_model(num_classes: int, variant: str = "small", anchors=None) -> ScratchDet:
    return ScratchDet(num_classes=num_classes, variant=variant, anchors=anchors)
