import torch
import torch.nn as nn
import torch.nn.functional as F

from model_con2d_heatmap_model1 import (
    CenterPointDecoder,
    ConvBNAct,
    RADRAEFusion,
)
from utils_dummy.other_helping_dunctions import gather_topk_features, inverse_sigmoid


class FPNNoDeformEncoder(nn.Module):
    """
    FPN-style radar encoder without deformable convolutions.

    This mirrors model_fpn_heatmap_model5.FPNDeformEncoder, but every deformable
    convolution is replaced by a normal ConvBNAct block.
    """

    def __init__(self, in_channels, fpn_channels=128):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=1),
            ConvBNAct(32, 32, kernel_size=3, stride=1),
        )

        self.stage1 = nn.Sequential(
            ConvBNAct(32, 64, kernel_size=3, stride=2),
            ConvBNAct(64, 64, kernel_size=3, stride=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(64, 128, kernel_size=3, stride=2),
            ConvBNAct(128, 128, kernel_size=3, stride=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(128, 256, kernel_size=3, stride=2),
            ConvBNAct(256, 256, kernel_size=3, stride=1),
        )

        self.lateral1 = ConvBNAct(64, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = ConvBNAct(128, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = ConvBNAct(256, fpn_channels, kernel_size=1, stride=1, padding=0)

        self.smooth1 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth2 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth3 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)

        self.output_refine = nn.Sequential(
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        p3 = self.smooth3(self.lateral3(c3))
        p2 = self.lateral2(c2) + F.interpolate(
            p3,
            size=c2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.smooth2(p2)

        p1 = self.lateral1(c1) + F.interpolate(
            p2,
            size=c1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.smooth1(p1)

        fused_at_stride4 = p2 + F.interpolate(
            p1,
            size=p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.output_refine(fused_at_stride4)


class RADRAEFPNNoDeformEncoder(nn.Module):
    """
    Separate RAD and RAE FPN encoders without deformable convolutions.
    """

    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.rad_encoder = FPNNoDeformEncoder(
            in_channels=d_in,
            fpn_channels=fpn_channels,
        )
        self.rae_encoder = FPNNoDeformEncoder(
            in_channels=e_in,
            fpn_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)
        return rad_feat, rae_feat


class RADRAEFPNNoDeformFusionModel(nn.Module):
    """
    FPN no-deform encoder plus the same RAD/RAE fusion used by heatmap models.
    """

    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.encoder = RADRAEFPNNoDeformEncoder(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        self.fusion = RADRAEFusion(
            in_channels=fpn_channels,
            fused_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat, rae_feat = self.encoder(rad, rae)
        fused_feat = self.fusion(rad_feat, rae_feat)
        return {
            "rad_feat": rad_feat,
            "rae_feat": rae_feat,
            "fused_feat": fused_feat,
        }


class RADRAEFPNNoDeformCenterPointModel(nn.Module):
    """
    Full RAD/RAE FPN model without deformable convolutions.

    Output keys match the existing heatmap models for train/evaluation/visualization.
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
        self.backbone = RADRAEFPNNoDeformFusionModel(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        self.decoder = CenterPointDecoder(
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
        center_offset = decoded["center_offset"]
        center_height = decoded["center_height"]
        size = decoded["size"]
        yaw = decoded["yaw"]

        B, _, H, W = cls_logits.shape
        dense_count = H * W
        topk_count = min(self.num_boxes, dense_count)

        score_logits, _ = cls_logits.flatten(start_dim=2).max(dim=1)
        _, topk_indices = score_logits.topk(topk_count, dim=1)

        cls_flat = cls_logits.flatten(start_dim=2).transpose(1, 2)
        center_offset_flat = center_offset.flatten(start_dim=2).transpose(1, 2)
        center_height_flat = center_height.flatten(start_dim=2).transpose(1, 2)
        size_flat = size.flatten(start_dim=2).transpose(1, 2)
        yaw_flat = yaw.flatten(start_dim=2).transpose(1, 2)

        cls_pred = gather_topk_features(cls_flat, topk_indices)
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
        }
