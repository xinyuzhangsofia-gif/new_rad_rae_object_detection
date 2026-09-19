"""Shared Cartesian detection heads for selectable CenterPoint/RADE modes.

The five experimental backbones (models 7, 8, 12, 13, and 15) intentionally
share this output contract.  A mode changes only the detection head and loss;
both modes regress the same metric Cartesian box
``[x, y, z, length, width, height, yaw]`` on an R/A feature grid.
"""

import torch
import torch.nn as nn


CARTESIAN_HEAD_MODES = {"centerpoint", "radenet"}


def resolve_cartesian_head_mode(loss_mode, default="radenet"):
    """Return a concrete Cartesian head mode and reject unrelated heads."""
    normalized = "auto" if loss_mode is None else str(loss_mode).strip().lower()
    if normalized in {"", "auto"}:
        normalized = default
    if normalized not in CARTESIAN_HEAD_MODES:
        raise ValueError(
            "Cartesian dual-mode models require loss_mode='centerpoint' or "
            f"'radenet', got {loss_mode!r}."
        )
    return normalized


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CartesianCenterPointBoxDecoder(nn.Module):
    """Split metric regression branches used by Cartesian CenterPoint loss."""

    def __init__(self, in_channels=128, hidden_channels=128):
        super().__init__()
        self.shared = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )
        self.center_offset_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.center_height_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.size_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.yaw_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, x):
        feat = self.shared(x)
        center_offset = self.center_offset_head(feat)
        center_height = self.center_height_head(feat)
        size = self.size_head(feat)
        yaw = self.yaw_head(feat)
        return {
            "center_offset": center_offset,
            "center_height": center_height,
            "size": size,
            "yaw": yaw,
            "box_reg": torch.cat(
                [center_offset, center_height, size, yaw], dim=1
            ),
        }


class CartesianCenterPointClsDecoder(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, num_classes=2):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )
        nn.init.constant_(self.decoder[-1].bias, -2.19)

    def forward(self, x):
        return self.decoder(x)


class CartesianCenterPointDecoder(nn.Module):
    """Logit heatmap plus split 8-channel Cartesian regression branches."""

    def __init__(self, in_channels=128, hidden_channels=128, num_classes=2):
        super().__init__()
        self.cls_decoder = CartesianCenterPointClsDecoder(
            in_channels, hidden_channels, num_classes
        )
        self.box_decoder = CartesianCenterPointBoxDecoder(
            in_channels, hidden_channels
        )

    def forward(self, x):
        return {
            "cls_logits": self.cls_decoder(x),
            **self.box_decoder(x),
        }


class CartesianRADEHeatmapHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, x):
        return torch.sigmoid(self.head(x))


class CartesianRADERegressionHead(nn.Module):
    """Linear ``[dx, dy, z, l, w, h, sin(yaw), cos(yaw)]`` prediction."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 8, 1),
        )

    def forward(self, x):
        return self.head(x)


class CartesianRADEDecoder(nn.Module):
    """Probability heatmap plus unified Cartesian RADE regression map."""

    def __init__(self, in_channels=128, hidden_channels=128, num_classes=2):
        super().__init__()
        self.heatmap_head = CartesianRADEHeatmapHead(
            in_channels, hidden_channels, num_classes
        )
        self.regression_head = CartesianRADERegressionHead(
            in_channels, hidden_channels
        )

    def forward(self, x):
        return {
            "heatmap": self.heatmap_head(x),
            "regression": self.regression_head(x),
        }


def build_cartesian_decoder(
        loss_mode,
        in_channels=128,
        hidden_channels=128,
        num_classes=2,
    ):
    """Build one of the two heads while keeping their contracts explicit."""
    mode = resolve_cartesian_head_mode(loss_mode)
    decoder_class = (
        CartesianCenterPointDecoder
        if mode == "centerpoint"
        else CartesianRADEDecoder
    )
    return mode, decoder_class(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
    )
