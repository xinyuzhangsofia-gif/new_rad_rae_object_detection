import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformer

from model_deform_heatmap_model4 import (
    CenterPointDecoder,
    ConvBNAct,
    RADRAEFusion,
    gather_topk_features,
    inverse_sigmoid,
)


class SwinFPNEncoder(nn.Module):
    """
    Swin Transformer encoder with the same stride-4 FPN output contract as model5.
    """

    def __init__(
            self,
            in_channels,
            fpn_channels=128,
            embed_dim=64,
            depths=(2, 2, 2),
            num_heads=(2, 4, 8),
            window_size=4,
            patch_size=4,
            mlp_ratio=4.0,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=0.1,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.swin = SwinTransformer(
            patch_size=[patch_size, patch_size],
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=[window_size, window_size],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=1,
        )
        self.swin.features[0][0] = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

        c1_channels = embed_dim
        c2_channels = embed_dim * 2
        c3_channels = embed_dim * 4

        self.lateral1 = ConvBNAct(c1_channels, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = ConvBNAct(c2_channels, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = ConvBNAct(c3_channels, fpn_channels, kernel_size=1, stride=1, padding=0)

        self.smooth1 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth2 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth3 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)

        self.output_refine = nn.Sequential(
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
        )

    @staticmethod
    def _channels_first(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        pad_h = (self.patch_size - x.shape[-2] % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - x.shape[-1] % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        features = self.swin.features
        x = features[0](x)
        x = features[1](x)
        c1 = self._channels_first(x)

        x = features[2](x)
        x = features[3](x)
        c2 = self._channels_first(x)

        x = features[4](x)
        x = features[5](x)
        c3 = self._channels_first(x)

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

        return self.output_refine(p1)


class RADRAESwinFPNEncoder(nn.Module):
    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.rad_encoder = SwinFPNEncoder(
            in_channels=d_in,
            fpn_channels=fpn_channels,
        )
        self.rae_encoder = SwinFPNEncoder(
            in_channels=e_in,
            fpn_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)
        return rad_feat, rae_feat


class RADRAESwinFPNFusionModel(nn.Module):
    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.encoder = RADRAESwinFPNEncoder(
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


class RADRAESwinFPNCenterPointModel(nn.Module):
    """
    model7: model5-style FPN heatmap detector with Swin Transformer encoders.
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
        self.backbone = RADRAESwinFPNFusionModel(
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
