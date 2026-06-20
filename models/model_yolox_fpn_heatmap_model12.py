import torch
import torch.nn as nn

from .model_deform_heatmap_model4 import (
    ConvBNAct,
)
from .model_fpn_heatmap_model5 import RADRAEFPNDeformFusionModel


class CenterPointYOLOXDecoder(nn.Module):
    """
    YOLOX-style decoupled head for the CenterPoint dense detector.

    It uses a 1x1 stem, then two 3x3-conv towers:
        cls tower -> class logits
        reg tower -> box regression + objectness logits
    """

    def __init__(
            self,
            in_channels=128,
            hidden_channels=128,
            num_classes=2,
        ):
        super().__init__()
        self.stem = ConvBNAct(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.cls_tower = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
        )
        self.reg_tower = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
        )

        self.cls_head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.objectness_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.center_offset_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.center_height_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.size_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.yaw_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)

        nn.init.constant_(self.cls_head.bias, -2.19)
        nn.init.constant_(self.objectness_head.bias, -2.19)

    def forward(self, fused_feat):
        stem_feat = self.stem(fused_feat)
        cls_feat = self.cls_tower(stem_feat)
        reg_feat = self.reg_tower(stem_feat)

        cls_logits = self.cls_head(cls_feat)
        objectness_logits = self.objectness_head(reg_feat)
        center_offset = self.center_offset_head(reg_feat)
        center_height = self.center_height_head(reg_feat)
        size = self.size_head(reg_feat)
        yaw = self.yaw_head(reg_feat)
        box_reg = torch.cat(
            [center_offset, center_height, size, yaw],
            dim=1,
        )

        return {
            "cls_logits": cls_logits,
            "objectness_logits": objectness_logits,
            "center_offset": center_offset,
            "center_height": center_height,
            "size": size,
            "yaw": yaw,
            "box_reg": box_reg,
        }


class RADRAEYOLOXFPNCenterPointModel(nn.Module):
    """
    model12: FPN CenterPoint detector with a YOLOX-style decoupled head.
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
        self.register_buffer("_model12_yolox_marker", torch.ones(1), persistent=True)
        self.backbone = RADRAEFPNDeformFusionModel(
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
            "heatmap_logits": decoded["cls_logits"],
        }
