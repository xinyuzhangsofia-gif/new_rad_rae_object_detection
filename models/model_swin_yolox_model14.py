import torch
import torch.nn as nn

from .model_deform_heatmap_model4 import RADRAEFusion
from .model_swin_heatmap_model7 import SwinFPNEncoder
from .model_yolox_fpn_heatmap_model12 import CenterPointYOLOXDecoder


class RADRAELightweightSwinFPNFusionModel(nn.Module):
    """
    Lightweight Swin fusion backbone for model14.

    Compared with model7, this reduces encoder width and depth, and uses patch_size=4
    so the fused feature map stays at the lighter 8x8-style grid used by model12.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            fpn_channels=96,
            embed_dim=48,
            depths=(2, 2, 1),
            num_heads=(2, 4, 8),
            window_size=4,
            patch_size=4,
        ):
        super().__init__()
        encoder_kwargs = dict(
            fpn_channels=fpn_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            patch_size=patch_size,
        )
        self.rad_encoder = SwinFPNEncoder(in_channels=d_in, **encoder_kwargs)
        self.rae_encoder = SwinFPNEncoder(in_channels=e_in, **encoder_kwargs)
        self.fusion = RADRAEFusion(
            in_channels=fpn_channels,
            fused_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)
        fused_feat = self.fusion(rad_feat, rae_feat)
        return {
            "rad_feat": rad_feat,
            "rae_feat": rae_feat,
            "fused_feat": fused_feat,
        }


class RADRAESwinYOLOXCenterPointModel(nn.Module):
    """
    model14: lightweight Swin-FPN encoder plus the YOLOX-style decoder from model12.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=96,
            fpn_channels=96,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("_model14_swin_yolox_marker", torch.ones(1), persistent=True)
        self.backbone = RADRAELightweightSwinFPNFusionModel(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        self.decoder = CenterPointYOLOXDecoder(
            in_channels=fpn_channels,
            hidden_channels=decoder_hidden_channels,
            num_classes=num_classes,
        )

    def forward(self, rad, rae):
        features = self.backbone(rad, rae)
        decoded = self.decoder(features["fused_feat"])
        return {
            **features,
            **decoded,
        }
