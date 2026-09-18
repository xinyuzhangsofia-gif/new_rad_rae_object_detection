import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformer


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
            patch_size=2,
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


class RADRAELightweightSwinFPNFusionModel(nn.Module):
    """
    Lightweight Swin fusion backbone for model14.

    Compared with model7, this reduces encoder width and depth, and uses patch_size=4
    so the fused feature map stays at the lighter 8x8-style grid used by model12.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            fpn_channels=96,
            embed_dim=48,
            depths=(2, 2, 1),
            num_heads=(2, 4, 8),
            window_size=4,
            patch_size=4,
        ):
        super().__init__()
        encoder_kwargs = dict(
            fpn_channels=fpn_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            patch_size=patch_size,
        )
        self.rad_encoder = SwinFPNEncoder(in_channels=d_in, **encoder_kwargs)
        self.rae_encoder = SwinFPNEncoder(in_channels=e_in, **encoder_kwargs)
        self.fusion = RADRAEFusion(
            in_channels=fpn_channels,
            fused_channels=fpn_channels,
        )

    def forward(self, rad, rae):
        rad_feat = self.rad_encoder(rad)
        rae_feat = self.rae_encoder(rae)
        fused_feat = self.fusion(rad_feat, rae_feat)
        return {
            "rad_feat": rad_feat,
            "rae_feat": rae_feat,
            "fused_feat": fused_feat,
        }


class RADRAESwinYOLOXCenterPointModel(nn.Module):
    """
    model14: lightweight Swin-FPN encoder plus the YOLOX-style decoder from model12.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=96,
            fpn_channels=96,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("_model14_swin_yolox_marker", torch.ones(1), persistent=True)
        self.backbone = RADRAELightweightSwinFPNFusionModel(
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
