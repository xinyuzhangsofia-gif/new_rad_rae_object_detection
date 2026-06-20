import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_deform_heatmap_model4 import (
    CenterPointDecoder,
    ConvBNAct,
    DeformConvBNAct,
    RADRAEFusion,
)


class SpatialConvBNAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            dilation=1,
        ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DilatedConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvolutionalFeatureEnhancement(nn.Module):
    """
    CFE module from C-AFBiFPN, adapted to preserve the input channel count.
    """

    def __init__(self, channels):
        super().__init__()
        branch_channels = max(8, (channels + 2) // 3)
        concat_channels = branch_channels * 3

        self.branch1 = nn.Sequential(
            ConvBNAct(channels, branch_channels, kernel_size=1, stride=1, padding=0),
            SpatialConvBNAct(branch_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1)),
            SpatialConvBNAct(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
            DilatedConvBNAct(branch_channels, branch_channels),
        )
        self.branch2 = nn.Sequential(
            ConvBNAct(channels, branch_channels, kernel_size=1, stride=1, padding=0),
            SpatialConvBNAct(branch_channels, branch_channels, kernel_size=(1, 5), padding=(0, 2)),
            SpatialConvBNAct(branch_channels, branch_channels, kernel_size=(5, 1), padding=(2, 0)),
            DilatedConvBNAct(branch_channels, branch_channels),
        )
        self.branch3 = nn.Sequential(
            ConvBNAct(channels, branch_channels, kernel_size=1, stride=1, padding=0),
            SpatialConvBNAct(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
            SpatialConvBNAct(branch_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1)),
            DeformConvBNAct(branch_channels, branch_channels, kernel_size=3, stride=1),
        )
        self.residual = ConvBNAct(channels, concat_channels, kernel_size=1, stride=1, padding=0)
        self.output_project = ConvBNAct(concat_channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enhanced = torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
            ],
            dim=1,
        )
        enhanced = enhanced + self.residual(x)
        return self.output_project(enhanced)


class FPNCFEEncoder(nn.Module):
    """
    model5 FPN-deform encoder with CFE modules inserted after each backbone stage.
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
            DeformConvBNAct(128, 128, kernel_size=3, stride=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(128, 256, kernel_size=3, stride=2),
            DeformConvBNAct(256, 256, kernel_size=3, stride=1),
        )

        self.cfe1 = ConvolutionalFeatureEnhancement(64)
        self.cfe2 = ConvolutionalFeatureEnhancement(128)
        self.cfe3 = ConvolutionalFeatureEnhancement(256)

        self.lateral1 = ConvBNAct(64, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = ConvBNAct(128, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = ConvBNAct(256, fpn_channels, kernel_size=1, stride=1, padding=0)

        self.smooth1 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth2 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth3 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)

        self.output_refine = nn.Sequential(
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
            DeformConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        c1 = self.cfe1(self.stage1(x))
        c2 = self.cfe2(self.stage2(c1))
        c3 = self.cfe3(self.stage3(c2))

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


class RADRAEFPNCFEEncoder(nn.Module):
    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.rad_encoder = FPNCFEEncoder(
            in_channels=d_in,
            fpn_channels=fpn_channels,
        )
        self.rae_encoder = FPNCFEEncoder(
            in_channels=e_in,
            fpn_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)
        return rad_feat, rae_feat


class RADRAEFPNCFEFusionModel(nn.Module):
    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.encoder = RADRAEFPNCFEEncoder(
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


class RADRAEFPNCFECenterPointModel(nn.Module):
    """
    model8: model5 FPN heatmap detector with CFE-enhanced backbone stages.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=128,
            fpn_channels=128,
            return_features=False,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.return_features = return_features
        self.backbone = RADRAEFPNCFEFusionModel(
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
