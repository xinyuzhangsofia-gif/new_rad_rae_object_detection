import torch
import torch.nn as nn
import torch.nn.functional as F

from model_con2d_heatmap import (
    CenterPointDecoder,
    ConvBNAct,
    RADRAEFusion,
)
from utils_dummy.other_helping_dunctions import gather_topk_features, inverse_sigmoid


class WeightedFeatureFusion(nn.Module):
    """
    BiFPN-style normalized positive weighted feature fusion.
    """

    def __init__(self, num_inputs, eps=1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, features):
        if len(features) != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} features, got {len(features)}"
            )

        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)

        fused = features[0] * weights[0]
        for idx in range(1, len(features)):
            fused = fused + features[idx] * weights[idx]
        return fused


class BiFPNBlock(nn.Module):
    """
    Three-level BiFPN block.

    Inputs:
        p1: stride-2 feature
        p2: stride-4 feature
        p3: stride-8 feature

    Outputs:
        p1_out, p2_out, p3_out
    """

    def __init__(self, channels):
        super().__init__()
        self.p2_td_fusion = WeightedFeatureFusion(num_inputs=2)
        self.p1_td_fusion = WeightedFeatureFusion(num_inputs=2)
        self.p2_out_fusion = WeightedFeatureFusion(num_inputs=3)
        self.p3_out_fusion = WeightedFeatureFusion(num_inputs=2)

        self.p2_td_refine = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.p1_td_refine = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.p2_out_refine = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.p3_out_refine = ConvBNAct(channels, channels, kernel_size=3, stride=1)

    def forward(self, p1, p2, p3):
        p3_td = p3
        p2_td = self.p2_td_refine(
            self.p2_td_fusion([
                p2,
                F.interpolate(
                    p3_td,
                    size=p2.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            ])
        )
        p1_td = self.p1_td_refine(
            self.p1_td_fusion([
                p1,
                F.interpolate(
                    p2_td,
                    size=p1.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            ])
        )

        p2_out = self.p2_out_refine(
            self.p2_out_fusion([
                p2,
                p2_td,
                F.interpolate(
                    p1_td,
                    size=p2.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            ])
        )
        p3_out = self.p3_out_refine(
            self.p3_out_fusion([
                p3,
                F.max_pool2d(
                    p2_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ])
        )

        return p1_td, p2_out, p3_out


class BiFPNEncoder(nn.Module):
    """
    Non-deformable radar encoder with BiFPN feature fusion.

    Input:
        x: [B, in_channels, R, A]

    Output:
        feat: [B, bifpn_channels, H, W], where H/W follow the stride-4 feature map.
    """

    def __init__(self, in_channels, bifpn_channels=128, num_bifpn_blocks=1):
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

        self.lateral1 = ConvBNAct(64, bifpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = ConvBNAct(128, bifpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = ConvBNAct(256, bifpn_channels, kernel_size=1, stride=1, padding=0)

        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(bifpn_channels)
            for _ in range(num_bifpn_blocks)
        ])

        self.output_refine = nn.Sequential(
            ConvBNAct(bifpn_channels, bifpn_channels, kernel_size=3, stride=1),
            ConvBNAct(bifpn_channels, bifpn_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        p1 = self.lateral1(c1)
        p2 = self.lateral2(c2)
        p3 = self.lateral3(c3)

        for bifpn_block in self.bifpn_blocks:
            p1, p2, p3 = bifpn_block(p1, p2, p3)

        return self.output_refine(p2)


class RADRAEBiFPNEncoder(nn.Module):
    """
    Separate RAD and RAE non-deformable BiFPN encoders.
    """

    def __init__(self, d_in=64, e_in=37, bifpn_channels=128, num_bifpn_blocks=1):
        super().__init__()
        self.rad_encoder = BiFPNEncoder(
            in_channels=d_in,
            bifpn_channels=bifpn_channels,
            num_bifpn_blocks=num_bifpn_blocks,
        )
        self.rae_encoder = BiFPNEncoder(
            in_channels=e_in,
            bifpn_channels=bifpn_channels,
            num_bifpn_blocks=num_bifpn_blocks,
        )

    def forward(self, rad, rae):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)
        return rad_feat, rae_feat


class RADRAEBiFPNFusionModel(nn.Module):
    """
    BiFPN encoder + same RAD/RAE fusion used by the other CenterPoint models.

    Output dict:
        rad_feat:   [B, 128, H, W]
        rae_feat:   [B, 128, H, W]
        fused_feat: [B, 128, H, W]
    """

    def __init__(self, d_in=64, e_in=37, bifpn_channels=128, num_bifpn_blocks=1):
        super().__init__()
        self.encoder = RADRAEBiFPNEncoder(
            d_in=d_in,
            e_in=e_in,
            bifpn_channels=bifpn_channels,
            num_bifpn_blocks=num_bifpn_blocks,
        )
        self.fusion = RADRAEFusion(
            in_channels=bifpn_channels,
            fused_channels=bifpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat, rae_feat = self.encoder(rad, rae)
        fused_feat = self.fusion(rad_feat, rae_feat)
        return {
            "rad_feat": rad_feat,
            "rae_feat": rae_feat,
            "fused_feat": fused_feat,
        }


class RADRAEBiFPNCenterPointModel(nn.Module):
    """
    Full RAD/RAE model with non-deformable BiFPN encoders, RAD/RAE fusion,
    and decoupled CenterPoint decoder.

    Output keys match the existing heatmap models for train/evaluation/visualization.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=128,
            num_boxes=64,
            bifpn_channels=128,
            num_bifpn_blocks=1,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.backbone = RADRAEBiFPNFusionModel(
            d_in=d_in,
            e_in=e_in,
            bifpn_channels=bifpn_channels,
            num_bifpn_blocks=num_bifpn_blocks,
        )
        self.decoder = CenterPointDecoder(
            in_channels=bifpn_channels,
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

