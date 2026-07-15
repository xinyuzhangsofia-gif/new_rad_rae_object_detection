import torch
import torch.nn as nn

from .model_radenet_official_model15 import RADEOfficialDecoder
from .model_swin_heatmap_model7 import RADRAESwinFPNFusionModel


class RADRAESwinRADENetOfficialModel(nn.Module):
    """
    model16: model7 Swin fusion backbone with official-style RADE-Net decoder.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=128,
            fpn_channels=128,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = RADRAESwinFPNFusionModel(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        self.decoder = RADEOfficialDecoder(
            in_channels=fpn_channels,
            hidden_channels=decoder_hidden_channels,
            num_classes=num_classes,
        )
        self.register_buffer("_model16_swin_radenet_official_marker", torch.ones(1), persistent=True)

    def forward(self, rad, rae):
        features = self.backbone(rad, rae)
        decoded = self.decoder(features["fused_feat"])
        return {
            **features,
            **decoded,
        }
