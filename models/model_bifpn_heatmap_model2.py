import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_con2d_heatmap_model1 import (
    CenterPointBoxDecoder,
    CenterPointClsDecoder,
    ConvBNAct,
    RADRAEFusion,
)


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


class PyramidEncoder(nn.Module):
    """
    Non-deformable three-stage encoder.

    Output:
        c1: stride-2 feature
        c2: stride-4 feature
        c3: stride-8 feature
    """

    def __init__(self, in_channels):
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

    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return c1, c2, c3


class RADRAEBiFPNEncoder(nn.Module):
    """
    Separate RAD and RAE non-deformable pyramid encoders.
    """

    def __init__(self, d_in=64, e_in=37):
        super().__init__()
        self.rad_encoder = PyramidEncoder(in_channels=d_in)
        self.rae_encoder = PyramidEncoder(in_channels=e_in)

    def forward(self, rad, rae):
        rad_c1, rad_c2, rad_c3 = self.rad_encoder(rad)
        rae_c1, rae_c2, rae_c3 = self.rae_encoder(rae)
        return {
            "rad_c1": rad_c1,
            "rad_c2": rad_c2,
            "rad_c3": rad_c3,
            "rae_c1": rae_c1,
            "rae_c2": rae_c2,
            "rae_c3": rae_c3,
        }


class RADRAEBiFPNFusionModel(nn.Module):
    """
    RAD/RAE pyramid encoder, per-scale RAD/RAE fusion, then BiFPN.

    Output dict:
        p1:         fused stride-2 feature before BiFPN
        p2:         fused stride-4 feature before BiFPN
        p3:         fused stride-8 feature before BiFPN
        p1_out:     BiFPN stride-2 output
        p2_out:     BiFPN stride-4 output
        p3_out:     BiFPN stride-8 output
        decoder_feat: high-resolution decoder feature fused from p1/p2/p3 outputs
    """

    def __init__(self, d_in=64, e_in=37, bifpn_channels=128, num_bifpn_blocks=1):
        super().__init__()
        self.encoder = RADRAEBiFPNEncoder(d_in=d_in, e_in=e_in)

        self.p1_fusion = RADRAEFusion(
            in_channels=64,
            fused_channels=bifpn_channels,
        )
        self.p2_fusion = RADRAEFusion(
            in_channels=128,
            fused_channels=bifpn_channels,
        )
        self.p3_fusion = RADRAEFusion(
            in_channels=256,
            fused_channels=bifpn_channels,
        )

        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(bifpn_channels)
            for _ in range(num_bifpn_blocks)
        ])

        self.decoder_fusion = nn.Sequential(
            ConvBNAct(
                bifpn_channels * 3,
                bifpn_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ConvBNAct(bifpn_channels, bifpn_channels, kernel_size=3, stride=1),
            ConvBNAct(bifpn_channels, bifpn_channels, kernel_size=3, stride=1),
        )

    def forward(self, rad, rae):
        features = self.encoder(rad, rae)

        p1 = self.p1_fusion(features["rad_c1"], features["rae_c1"])
        p2 = self.p2_fusion(features["rad_c2"], features["rae_c2"])
        p3 = self.p3_fusion(features["rad_c3"], features["rae_c3"])

        p1_out, p2_out, p3_out = p1, p2, p3
        for bifpn_block in self.bifpn_blocks:
            p1_out, p2_out, p3_out = bifpn_block(p1_out, p2_out, p3_out)

        p2_to_p1 = F.interpolate(
            p2_out,
            size=p1_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p3_to_p1 = F.interpolate(
            p3_out,
            size=p1_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_feat = self.decoder_fusion(
            torch.cat([p1_out, p2_to_p1, p3_to_p1], dim=1)
        )

        return {
            **features,
            "rad_feat": features["rad_c2"],
            "rae_feat": features["rae_c2"],
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p1_out": p1_out,
            "p2_out": p2_out,
            "p3_out": p3_out,
            "decoder_feat": decoder_feat,
            "heatmap_feat": decoder_feat,
            "box_feat": decoder_feat,
            "fused_feat": decoder_feat,
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
            bifpn_channels=128,
            num_bifpn_blocks=1,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = RADRAEBiFPNFusionModel(
            d_in=d_in,
            e_in=e_in,
            bifpn_channels=bifpn_channels,
            num_bifpn_blocks=num_bifpn_blocks,
        )
        self.cls_decoder = CenterPointClsDecoder(
            in_channels=bifpn_channels,
            hidden_channels=decoder_hidden_channels,
            num_classes=num_classes,
        )
        self.box_decoder = CenterPointBoxDecoder(
            in_channels=bifpn_channels,
            hidden_channels=decoder_hidden_channels,
        )

    def forward(self, rad, rae):
        features = self.backbone(rad, rae)
        decoded = {
            "cls_logits": self.cls_decoder(features["heatmap_feat"]),
            **self.box_decoder(features["box_feat"]),
        }
        return {
            **features,
            **decoded,
            "heatmap_logits": decoded["cls_logits"],
        }
