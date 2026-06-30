import torch
import torch.nn as nn

from .model_radenet_cbam_model13 import (
    DilatedResidualNeck,
    RADRAERADEBackbone,
)


class RADEOfficialHeatmapHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))


class RADEOfficialRegressionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class RADEOfficialDecoder(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, num_classes=2):
        super().__init__()
        self.heatmap_head = RADEOfficialHeatmapHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )
        self.regression_head = RADEOfficialRegressionHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=8,
        )

    def forward(self, x):
        heatmap = self.heatmap_head(x)
        regression = self.regression_head(x)
        return {
            "heatmap": heatmap,
            "regression": regression,
        }


class RADRAERADENetOfficialModel(nn.Module):
    """
    model15: RADE-Net-style backbone/neck with official-style heatmap and
    regression outputs for RADE-Net loss.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=128,
            dropout=0.0,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = RADRAERADEBackbone(
            in_channels=d_in + e_in,
            decoder_channels=128,
            dropout=dropout,
            pad_width=5,
        )
        self.neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))
        self.decoder = RADEOfficialDecoder(
            in_channels=128,
            hidden_channels=decoder_hidden_channels,
            num_classes=num_classes,
        )
        self.register_buffer("_model15_radenet_official_marker", torch.ones(1), persistent=True)

    def forward(self, rad, rae):
        features = self.backbone(rad, rae)
        fused_feat = self.neck(features["backbone_feat"])
        decoded = self.decoder(fused_feat)
        return {
            **features,
            "fused_feat": fused_feat,
            **decoded,
        }
