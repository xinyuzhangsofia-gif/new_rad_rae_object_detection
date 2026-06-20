import torch
import torch.nn as nn

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
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StageBackboneEncoder(nn.Module):
    """
    Stage-based radar encoder without deformable convolution.

    Input:
        x: [B, in_channels, R, A]

    Output:
        feat: [B, 128, H, W]
    """

    def __init__(self, in_channels):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            ConvBNAct(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
        )

        self.stage1 = nn.Sequential(
            ConvBNAct(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            ConvBNAct(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
        )

        self.stage2 = nn.Sequential(
            ConvBNAct(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
            ),
            ConvBNAct(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
            ),
        )

        self.stage3 = nn.Sequential(
            ConvBNAct(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
            ),
            ConvBNAct(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
            ),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class RADRAEStageEncoder(nn.Module):
    """
    Separate RAD and RAE encoders with the same non-deformable stage architecture.

    RAD:
        [B, D, R, A] -> [B, 128, H, W]

    RAE:
        [B, E, R, A] -> [B, 128, H, W]
    """

    def __init__(self, d_in=64, e_in=37):
        super().__init__()
        self.rad_encoder = StageBackboneEncoder(in_channels=d_in)
        self.rae_encoder = StageBackboneEncoder(in_channels=e_in)

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
                stride=1,
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
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
            padding=0,
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


class RADRAEStageFusionModel(nn.Module):
    """
    Encoder + fusion backbone.

    Input:
        rad: [B, D, R, A]
        rae: [B, E, R, A]

    Output dict:
        rad_feat:   [B, 128, H, W]
        rae_feat:   [B, 128, H, W]
        fused_feat: [B, 128, H, W]
    """

    def __init__(self, d_in=64, e_in=37):
        super().__init__()
        self.encoder = RADRAEStageEncoder(d_in=d_in, e_in=e_in)
        self.fusion = RADRAEFusion(in_channels=128, fused_channels=128)

    def forward(self, rad, rae):
        rad_feat, rae_feat = self.encoder(rad, rae)
        fused_feat = self.fusion(rad_feat, rae_feat)
        return {
            "rad_feat": rad_feat,
            "rae_feat": rae_feat,
            "fused_feat": fused_feat,
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
                stride=1,
            ),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=num_classes,
                kernel_size=1,
            ),
        )
        nn.init.constant_(self.decoder[-1].bias, -2.19)

    def forward(self, fused_feat):
        return self.decoder(fused_feat)


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
                stride=1,
            ),
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
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
            dim=1,
        )

        return {
            "center_offset": center_offset,
            "center_height": center_height,
            "size": size,
            "yaw": yaw,
            "box_reg": box_reg,
        }


class CenterPointDecoder(nn.Module):
    """
    Decoupled CenterPoint-style decoder:
        cls decoder + box decoder.
    """

    def __init__(
            self,
            in_channels=128,
            hidden_channels=128,
            num_classes=2
        ):
        super().__init__()
        self.cls_decoder = CenterPointClsDecoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )
        self.box_decoder = CenterPointBoxDecoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, fused_feat):
        cls_logits = self.cls_decoder(fused_feat)
        box_outputs = self.box_decoder(fused_feat)
        return {
            "cls_logits": cls_logits,
            **box_outputs,
        }


class RADRAEStageCenterPointModel(nn.Module):
    """
    Full RAD/RAE model with non-deformable stage encoders, fusion,
    and CenterPoint decoder.

    Input:
        rad: [B, D, R, A]
        rae: [B, E, R, A]

    Output dict:
        rad_feat:       [B, 128, H, W]
        rae_feat:       [B, 128, H, W]
        fused_feat:     [B, 128, H, W]
        cls_logits:     [B, num_classes, H, W]
        center_offset:  [B, 2, H, W]
        center_height:  [B, 1, H, W]
        size:           [B, 3, H, W]
        yaw:            [B, 2, H, W]
        box_reg:        [B, 8, H, W]
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=128,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = RADRAEStageFusionModel(d_in=d_in, e_in=e_in)
        self.decoder = CenterPointDecoder(
            in_channels=128,
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
