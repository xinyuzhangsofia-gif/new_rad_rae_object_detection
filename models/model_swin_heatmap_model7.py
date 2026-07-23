import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformer

from coordinate_modes import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    validate_box_coordinate_mode,
)
from .model_deform_heatmap_model4 import (
    CenterPointDecoder,
    ConvBNAct,
    RADRAEFusion,
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


class Model7RADEHeatmapHead(nn.Module):
    """Original RADE-Net expanded heatmap head, copied into model7."""

    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))


class Model7RADERegressionHead(nn.Module):
    """Original RADE-Net [dx,dy,dz,l,w,h,sin(yaw),cos(yaw)] head."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 8, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class Model7RADECartesianDecoder(nn.Module):
    """RADE-Net Cartesian detector implemented directly inside model7."""

    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.heatmap_head = Model7RADEHeatmapHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )
        self.regression_head = Model7RADERegressionHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, x):
        return {
            "heatmap": self.heatmap_head(x),
            "regression": self.regression_head(x),
        }


class RADRAESwinFPNCenterPointModel(nn.Module):
    """
    model7 with one coordinate-aware detection implementation:

    - Polar mode: the existing split CenterPoint decoder.
    - Cartesian + RADE-Net: original RADE-Net heatmap and 8-channel Cartesian
      regression heads copied directly into model7.
    - Cartesian + CenterPoint: CenterPoint dense branches decoded as metric
      Cartesian boxes by the Cartesian CenterPoint loss.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
        decoder_hidden_channels=128,
        fpn_channels=128,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        loss_mode="auto",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.box_coordinate_mode = validate_box_coordinate_mode(
            box_coordinate_mode
        )
        self.backbone = RADRAESwinFPNFusionModel(
            d_in=d_in,
            e_in=e_in,
            fpn_channels=fpn_channels,
        )
        if self.box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            if loss_mode == "centerpoint":
                # Cartesian CenterPoint uses the same dense branch contract
                # as the other CenterPoint models, while its loss decodes the
                # branches into metric Cartesian boxes.
                self.decoder = CenterPointDecoder(
                    in_channels=fpn_channels,
                    hidden_channels=decoder_hidden_channels,
                    num_classes=num_classes,
                )
                self.register_buffer(
                    "_model7_cartesian_centerpoint_marker",
                    torch.ones(1),
                    persistent=True,
                )
            else:
                self.decoder = Model7RADECartesianDecoder(
                    in_channels=fpn_channels,
                    hidden_channels=decoder_hidden_channels,
                    num_classes=num_classes,
                )
                self.register_buffer(
                    "_model7_cartesian_radenet_marker",
                    torch.ones(1),
                    persistent=True,
                )
        else:
            self.decoder = CenterPointDecoder(
                in_channels=fpn_channels,
                hidden_channels=decoder_hidden_channels,
                num_classes=num_classes,
            )

    def forward(self, rad, rae):
        features = self.backbone(rad, rae)
        decoded = self.decoder(features["fused_feat"])
        outputs = {
            **features,
            **decoded,
        }
        if self.box_coordinate_mode == BOX_COORDINATE_POLAR:
            outputs["heatmap_logits"] = decoded["cls_logits"]
        return outputs
