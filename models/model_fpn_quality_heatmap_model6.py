import torch
import torch.nn as nn

from .model_deform_heatmap_model4 import (
    CenterPointBoxDecoder,
    CenterPointClsDecoder,
    ConvBNAct,
)
from .model_fpn_heatmap_model5 import RADRAEFPNDeformFusionModel


class CenterPointQualityDecoder(nn.Module):
    """
    CenterPoint decoder with an extra IoU-aware quality head.
    """

    def __init__(
            self,
            in_channels=128,
            hidden_channels=128,
            num_classes=2,
        ):
        super().__init__()
        self.cls_decoder = CenterPointClsDecoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )
        self.quality_decoder = nn.Sequential(
            ConvBNAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
            ),
        )
        self.box_decoder = CenterPointBoxDecoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )
        nn.init.constant_(self.quality_decoder[-1].bias, 0.0)

    def forward(self, fused_feat):
        cls_logits = self.cls_decoder(fused_feat)
        quality_logits = self.quality_decoder(fused_feat)
        box_outputs = self.box_decoder(fused_feat)
        return {
            "cls_logits": cls_logits,
            "quality_logits": quality_logits,
            **box_outputs,
        }


class RADRAEFPNQualityCenterPointModel(nn.Module):
    """
    model6: model_fpn_heatmap_model5 backbone plus an IoU-aware quality head.
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
        self.backbone = RADRAEFPNDeformFusionModel(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        self.decoder = CenterPointQualityDecoder(
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
            "heatmap_logits": decoded["cls_logits"],
        }
