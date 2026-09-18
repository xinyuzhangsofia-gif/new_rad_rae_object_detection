import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None
        ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CenterPointBoxDecoder(nn.Module):
    """
    CenterPoint-style box decoder.

    Output components:
        center_offset: [B, 2, H, W]  # local range/azimuth offset
        center_height: [B, 1, H, W]  # elevation/height center
        size:          [B, 3, H, W]  # r/a/e box size
        yaw:           [B, 2, H, W]  # sin(yaw), cos(yaw)

    Concatenated:
        box_reg:       [B, 8, H, W]
    """

    def __init__(self, in_channels=128, hidden_channels=128):
        super().__init__()
        self.shared = nn.Sequential(
            ConvBNAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1
            ),
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1
            ),
        )
        self.center_offset_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.center_height_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.size_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.yaw_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, fused_feat):
        feat = self.shared(fused_feat)
        center_offset = self.center_offset_head(feat)
        center_height = self.center_height_head(feat)
        size = self.size_head(feat)
        yaw = self.yaw_head(feat)
        box_reg = torch.cat(
            [center_offset, center_height, size, yaw],
            dim=1
        )

        return {
            "center_offset": center_offset,
            "center_height": center_height,
            "size": size,
            "yaw": yaw,
            "box_reg": box_reg,
        }


class CenterPointClsDecoder(nn.Module):
    """
    Center heatmap/classification decoder.

    Input:
        fused_feat: [B, 128, H, W]

    Output:
        cls_logits: [B, num_classes, H, W]
    """

    def __init__(self, in_channels=128, hidden_channels=128, num_classes=2):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBNAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1
            ),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=num_classes,
                kernel_size=1
            )
        )
        nn.init.constant_(self.decoder[-1].bias, -2.19)

    def forward(self, fused_feat):
        return self.decoder(fused_feat)


class DeformConvBNAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None
        ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.offset_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.deform_conv = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualConvRefinement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.refine(x))


class RADRAEFusion(nn.Module):
    """
    Fuse RAD and RAE features.

    Input:
        rad_feat: [B, 128, H, W]
        rae_feat: [B, 128, H, W]

    Output:
        fused_feat: [B, 128, H, W]
    """

    def __init__(self, in_channels=128, fused_channels=128):
        super().__init__()
        self.channel_fusion = ConvBNAct(
            in_channels=in_channels * 2,
            out_channels=fused_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.refinement = ResidualConvRefinement(channels=fused_channels)

    def forward(self, rad_feat, rae_feat):
        if rad_feat.shape[-2:] != rae_feat.shape[-2:]:
            raise ValueError(
                f"RAD/RAE feature map sizes must match, got "
                f"rad={tuple(rad_feat.shape)} and rae={tuple(rae_feat.shape)}"
            )

        fused_feat = torch.cat([rad_feat, rae_feat], dim=1)
        fused_feat = self.channel_fusion(fused_feat)
        fused_feat = self.refinement(fused_feat)
        return fused_feat


class FPNDeformPyramidEncoder(nn.Module):
    """
    FPN-deform encoder based on model_fpn_heatmap_model5, but returns all FPN levels.

    Output:
        p1: stride-2 feature, high spatial resolution
        p2: stride-4 feature
        p3: stride-8 feature, low spatial resolution
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

        self.lateral1 = ConvBNAct(64, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = ConvBNAct(128, fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = ConvBNAct(256, fpn_channels, kernel_size=1, stride=1, padding=0)

        self.smooth1 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth2 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)
        self.smooth3 = ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1)

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

        return {
            "p1": p1,
            "p2": p2,
            "p3": p3,
        }


class RADRAEFPNDeformPyramidEncoder(nn.Module):
    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.rad_encoder = FPNDeformPyramidEncoder(
            in_channels=d_in,
            fpn_channels=fpn_channels,
        )
        self.rae_encoder = FPNDeformPyramidEncoder(
            in_channels=e_in,
            fpn_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_pyramid = self.rad_encoder(rad)
        rae_pyramid = self.rae_encoder(rae)
        return rad_pyramid, rae_pyramid


class RADRAEMultiScaleFusion(nn.Module):
    def __init__(self, fpn_channels=128):
        super().__init__()
        self.fusion_p1 = RADRAEFusion(
            in_channels=fpn_channels,
            fused_channels=fpn_channels,
        )
        self.fusion_p2 = RADRAEFusion(
            in_channels=fpn_channels,
            fused_channels=fpn_channels,
        )
        self.fusion_p3 = RADRAEFusion(
            in_channels=fpn_channels,
            fused_channels=fpn_channels,
        )

    def forward(self, rad_pyramid, rae_pyramid):
        return {
            "p1": self.fusion_p1(rad_pyramid["p1"], rae_pyramid["p1"]),
            "p2": self.fusion_p2(rad_pyramid["p2"], rae_pyramid["p2"]),
            "p3": self.fusion_p3(rad_pyramid["p3"], rae_pyramid["p3"]),
        }


class FPNFeaturePairMixer(nn.Module):
    def __init__(self, channels=128, use_deform=True):
        super().__init__()
        refine_layers = [
            ConvBNAct(channels, channels, kernel_size=3, stride=1),
        ]
        if use_deform:
            refine_layers.append(
                DeformConvBNAct(channels, channels, kernel_size=3, stride=1)
            )
        else:
            refine_layers.append(
                ConvBNAct(channels, channels, kernel_size=3, stride=1)
            )
        self.refine = nn.Sequential(*refine_layers)

    def forward(self, high_resolution_feat, low_resolution_feat):
        mixed = high_resolution_feat + F.interpolate(
            low_resolution_feat,
            size=high_resolution_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(mixed)


class SplitFPNCenterPointDecoder(nn.Module):
    """
    CenterPoint heads with separate FPN feature inputs:
        cls head: p2 + upsample(p3), output stride-4 heatmap
        box head: p1 + upsample(p2), output stride-2 regression map
    """

    def __init__(
            self,
            channels=128,
            hidden_channels=128,
            num_classes=2,
        ):
        super().__init__()
        self.cls_feature_mixer = FPNFeaturePairMixer(
            channels=channels,
            use_deform=True,
        )
        self.reg_feature_mixer = FPNFeaturePairMixer(
            channels=channels,
            use_deform=True,
        )
        self.cls_decoder = CenterPointClsDecoder(
            in_channels=channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )
        self.box_decoder = CenterPointBoxDecoder(
            in_channels=channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, fused_pyramid):
        cls_feat = self.cls_feature_mixer(
            high_resolution_feat=fused_pyramid["p2"],
            low_resolution_feat=fused_pyramid["p3"],
        )
        reg_feat = self.reg_feature_mixer(
            high_resolution_feat=fused_pyramid["p1"],
            low_resolution_feat=fused_pyramid["p2"],
        )
        cls_logits = self.cls_decoder(cls_feat)
        box_outputs = self.box_decoder(reg_feat)
        return {
            "cls_feat": cls_feat,
            "reg_feat": reg_feat,
            "cls_logits": cls_logits,
            **box_outputs,
        }


class RADRAEFPNMultiFeatureCenterPointModel(nn.Module):
    """
    model10: model5-style FPN-deform backbone with split FPN features per head.

    Classification uses p2/p3, regression uses p1/p2. Dense train/eval code can
    handle the different cls/reg spatial sizes.
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
        self.encoder = RADRAEFPNDeformPyramidEncoder(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        self.fusion = RADRAEMultiScaleFusion(fpn_channels=fpn_channels)
        self.decoder = SplitFPNCenterPointDecoder(
            channels=fpn_channels,
            hidden_channels=decoder_hidden_channels,
            num_classes=num_classes,
        )

    def forward(self, rad, rae):
        rad_pyramid, rae_pyramid = self.encoder(rad, rae)
        fused_pyramid = self.fusion(rad_pyramid, rae_pyramid)
        decoded = self.decoder(fused_pyramid)
        return {
            "cls_logits": decoded["cls_logits"],
            "center_offset": decoded["center_offset"],
            "center_height": decoded["center_height"],
            "size": decoded["size"],
            "yaw": decoded["yaw"],
            "heatmap_logits": decoded["cls_logits"],
        }
