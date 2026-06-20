import torch
import torch.nn as nn

from .model_deform_heatmap_model4 import (
    CenterPointBoxDecoder,
    CenterPointClsDecoder,
)
from .model_fpn_heatmap_model5 import RADRAEFPNDeformFusionModel


class CenterPointQFLDecoder(nn.Module):
    """
    CenterPoint decoder for QFL-style classification-quality joint scores.

    Unlike model6, this decoder does not predict a separate quality branch. The
    classification logits are trained with Quality Focal Loss and used directly
    as final detection scores at inference time.
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
        self.box_decoder = CenterPointBoxDecoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, fused_feat):
        cls_logits = self.cls_decoder(fused_feat)
        box_outputs = self.box_decoder(fused_feat)
        return {
            "cls_logits": cls_logits,
            "qfl_cls_logits": cls_logits,
            **box_outputs,
        }


class RADRAEQFLFPNCenterPointModel(nn.Module):
    """
    model11: FPN CenterPoint detector trained with Quality Focal Loss.

    It reuses the model5 FPN/deformable backbone, but removes model6's separate
    quality head so the class score itself represents classification confidence
    and localization quality jointly.
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
        self.register_buffer("_qfl_model_marker", torch.ones(1), persistent=True)
        self.backbone = RADRAEFPNDeformFusionModel(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        self.decoder = CenterPointQFLDecoder(
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
