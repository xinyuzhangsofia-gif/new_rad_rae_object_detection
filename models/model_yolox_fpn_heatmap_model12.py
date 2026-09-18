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


class FPNDeformEncoder(nn.Module):
    """
    FPN-style radar encoder with deformable convolutions.

    Input:
        x: [B, in_channels, R, A]

    Output:
        feat: [B, 128, H, W], where H/W follow the stride-4 feature map.
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

        self.output_refine = nn.Sequential(
            ConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
            DeformConvBNAct(fpn_channels, fpn_channels, kernel_size=3, stride=1),
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


class RADRAEFPNDeformEncoder(nn.Module):
    """
    Separate RAD and RAE FPN-deform encoders.

    RAD:
        [B, D, R, A] -> [B, 128, H, W]

    RAE:
        [B, E, R, A] -> [B, 128, H, W]
    """

    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.rad_encoder = FPNDeformEncoder(
            in_channels=d_in,
            fpn_channels=fpn_channels,
        )
        self.rae_encoder = FPNDeformEncoder(
            in_channels=e_in,
            fpn_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)
        return rad_feat, rae_feat


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


class RADRAEFPNDeformFusionModel(nn.Module):
    """
    FPN-deform encoder + same RAD/RAE fusion used by model_deform_heatmap_model4.

    Output dict:
        rad_feat:   [B, 128, H, W]
        rae_feat:   [B, 128, H, W]
        fused_feat: [B, 128, H, W]
    """

    def __init__(self, d_in=64, e_in=37, fpn_channels=128):
        super().__init__()
        self.encoder = RADRAEFPNDeformEncoder(
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
            hidden_channels=64,     # want a smaller hidden channel to make the model lighter , past 128
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
            decoder_hidden_channels=64,
            fpn_channels=64,
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
        }
