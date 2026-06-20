import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_bifpn_heatmap_model2 import BiFPNBlock
from .model_cfe_heatmap_model8 import ConvolutionalFeatureEnhancement
from .model_deform_heatmap_model4 import (
    CenterPointDecoder,
    ConvBNAct,
    DeformConvBNAct,
    RADRAEFusion,
)


class CFEDeformPyramidEncoder(nn.Module):
    """
    model5-style deformable pyramid encoder with a CFE module after each stage.

    Outputs:
        c1: stride-2 feature, 64 channels
        c2: stride-4 feature, 128 channels
        c3: stride-8 feature, 256 channels
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
            DeformConvBNAct(128, 128, kernel_size=3, stride=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(128, 256, kernel_size=3, stride=2),
            DeformConvBNAct(256, 256, kernel_size=3, stride=1),
        )

        self.cfe1 = ConvolutionalFeatureEnhancement(64)
        self.cfe2 = ConvolutionalFeatureEnhancement(128)
        self.cfe3 = ConvolutionalFeatureEnhancement(256)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.cfe1(self.stage1(x))
        c2 = self.cfe2(self.stage2(c1))
        c3 = self.cfe3(self.stage3(c2))
        return c1, c2, c3


class RADRAECFEBiFPNEncoder(nn.Module):
    """
    Separate RAD and RAE CFE-enhanced pyramid encoders.
    """

    def __init__(self, d_in=64, e_in=37):
        super().__init__()
        self.rad_encoder = CFEDeformPyramidEncoder(in_channels=d_in)
        self.rae_encoder = CFEDeformPyramidEncoder(in_channels=e_in)

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


class RADRAECFEBiFPNFusionModel(nn.Module):
    """
    CFE-enhanced RAD/RAE pyramids, per-scale RAD/RAE fusion, then BiFPN.

    The decoder feature is built at stride 4 to keep target/loss/eval behavior
    comparable with model5/model8 while still using all BiFPN output levels.
    """

    def __init__(self, d_in=64, e_in=37, bifpn_channels=128, num_bifpn_blocks=1):
        super().__init__()
        self.encoder = RADRAECFEBiFPNEncoder(d_in=d_in, e_in=e_in)

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

        p1_to_p2 = F.interpolate(
            p1_out,
            size=p2_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p3_to_p2 = F.interpolate(
            p3_out,
            size=p2_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_feat = self.decoder_fusion(
            torch.cat([p1_to_p2, p2_out, p3_to_p2], dim=1)
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


class RADRAECFEBiFPNCenterPointModel(nn.Module):
    """
    model9: model8-style CFE stage enhancement plus model2-style BiFPN fusion.

    Output keys match the existing dense heatmap models. Intermediate features
    are omitted by default to reduce DataParallel gather memory.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=128,
            bifpn_channels=128,
            num_bifpn_blocks=1,
            return_features=False,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.return_features = return_features
        self.backbone = RADRAECFEBiFPNFusionModel(
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
        outputs = {
            **decoded,
            "heatmap_logits": decoded["cls_logits"],
        }
        if self.return_features:
            outputs = {
                **features,
                **outputs,
            }
        return outputs
