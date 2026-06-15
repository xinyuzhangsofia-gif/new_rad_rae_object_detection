import torch
import torch.nn as nn

from model_deform_heatmap_model4 import (
    CenterPointBoxDecoder,
    CenterPointClsDecoder,
    ConvBNAct,
    gather_topk_features,
    inverse_sigmoid,
)
from model_fpn_heatmap_model5 import RADRAEFPNDeformFusionModel


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
            num_boxes=64,
            fpn_channels=128,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
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
        query_outputs = self._dense_outputs_to_query_outputs(decoded)
        return {
            **features,
            **decoded,
            "heatmap_logits": decoded["cls_logits"],
            **query_outputs,
        }

    def _dense_outputs_to_query_outputs(self, decoded):
        cls_logits = decoded["cls_logits"]
        quality_logits = decoded["quality_logits"]
        center_offset = decoded["center_offset"]
        center_height = decoded["center_height"]
        size = decoded["size"]
        yaw = decoded["yaw"]

        B, _, H, W = cls_logits.shape
        dense_count = H * W
        topk_count = min(self.num_boxes, dense_count)

        cls_scores, _ = cls_logits.sigmoid().flatten(start_dim=2).max(dim=1)
        quality_scores = quality_logits.sigmoid().flatten(start_dim=2).squeeze(1)
        _, topk_indices = (cls_scores * quality_scores).topk(topk_count, dim=1)

        cls_flat = cls_logits.flatten(start_dim=2).transpose(1, 2)
        quality_flat = quality_logits.flatten(start_dim=2).transpose(1, 2)
        center_offset_flat = center_offset.flatten(start_dim=2).transpose(1, 2)
        center_height_flat = center_height.flatten(start_dim=2).transpose(1, 2)
        size_flat = size.flatten(start_dim=2).transpose(1, 2)
        yaw_flat = yaw.flatten(start_dim=2).transpose(1, 2)

        cls_pred = gather_topk_features(cls_flat, topk_indices)
        quality_pred = gather_topk_features(quality_flat, topk_indices)
        center_offset_topk = gather_topk_features(center_offset_flat, topk_indices)
        center_height_topk = gather_topk_features(center_height_flat, topk_indices)
        size_topk = gather_topk_features(size_flat, topk_indices)
        yaw_topk = gather_topk_features(yaw_flat, topk_indices)

        background_logits = cls_pred.new_zeros((B, topk_count, 1))
        cls_pred = torch.cat([cls_pred, background_logits], dim=-1)

        y_idx = (topk_indices // W).to(cls_logits.dtype)
        x_idx = (topk_indices % W).to(cls_logits.dtype)

        offset = center_offset_topk.sigmoid()
        r_center = (y_idx + offset[..., 0]) / max(H, 1)
        a_center = (x_idx + offset[..., 1]) / max(W, 1)
        e_center = center_height_topk[..., 0].sigmoid()

        box_size = size_topk.sigmoid()
        yaw_angle = torch.atan2(yaw_topk[..., 0], yaw_topk[..., 1])
        yaw_norm = (yaw_angle + torch.pi) / (2.0 * torch.pi)

        box_norm = torch.stack(
            [
                r_center,
                a_center,
                e_center,
                box_size[..., 0],
                box_size[..., 1],
                box_size[..., 2],
                yaw_norm,
            ],
            dim=-1,
        ).clamp(min=1e-4, max=1.0 - 1e-4)

        box_pred = inverse_sigmoid(box_norm)
        return {
            "box_pred": box_pred,
            "cls_pred": cls_pred,
            "quality_pred": quality_pred,
        }
