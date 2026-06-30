import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0, num_groups=32):
        super().__init__()
        if out_channels == 1:
            norm = nn.Identity()
        else:
            norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm,
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.single_conv(x)


class DoubleConvResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            SingleConv(in_channels, out_channels, kernel_size, padding, dropout),
            SingleConv(out_channels, out_channels, kernel_size, padding, dropout),
        )
        if in_channels != out_channels:
            self.adjust_input = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.adjust_input = nn.Identity()

    def forward(self, x):
        return self.double_conv(x) + self.adjust_input(x)


class Downsample(nn.Module):
    def __init__(self, kernel_size=2, padding=0):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=padding)

    def forward(self, x):
        return self.pooling(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, padding=0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
        )

    def forward(self, x):
        return self.upsample(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, in_channels)

    def forward(self, x):
        max_pool = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        avg_pool = self.avg_pool(x)
        max_pool = max_pool.view(max_pool.size(0), -1)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        weights = torch.sigmoid(max_out + avg_out).view(x.size(0), -1, 1, 1)
        return x * weights


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention = self.convolution(torch.cat([max_pool, avg_pool], dim=1))
        return torch.sigmoid(attention)


class ConvolutionalBlockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.start_conv = SingleConv(in_channels, out_channels, kernel_size, padding, dropout)
        self.channel_attention = ChannelAttentionModule(out_channels)
        self.spatial_attention = SpatialAttentionModule()
        self.end_conv = SingleConv(out_channels, out_channels, kernel_size, padding, dropout)

    def forward(self, x):
        start_conv = self.start_conv(x)
        channel_weighted = self.channel_attention(start_conv)
        spatial_attention = self.spatial_attention(channel_weighted)
        residual = channel_weighted * spatial_attention + start_conv
        return self.end_conv(residual)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.conv1 = DoubleConvResidual(in_channels, out_channels, kernel_size, padding, dropout)
        self.conv2 = DoubleConvResidual(out_channels, out_channels, kernel_size, padding, dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=128, kernel_size=3, dilation=1, groups=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=1, dilation=1)
        self.gn = nn.GroupNorm(groups, in_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.gn(out)
        return self.act(out + x)


class DilatedResidualNeck(nn.Module):
    def __init__(self, in_channels=128, dilation=(1, 2, 3)):
        super().__init__()
        self.neck = nn.Sequential(*[ResidualBlock(in_channels, dilation=d) for d in dilation])

    def forward(self, x):
        return self.neck(x)


class ExpandedCenterHead(nn.Module):
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
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, x):
        return self.head(x)


class ExpandedRegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=8):
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
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class RADRAERADEBackbone(nn.Module):
    """
    RADE-Net-style radar-only U-Net with CBAM skip attention.
    """

    def __init__(self, in_channels=101, decoder_channels=128, dropout=0.0, pad_width=5):
        super().__init__()
        self.pad_width = pad_width
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.encoder_block_1 = DoubleConvResidual(128, 128, dropout=dropout)
        self.downsample_1 = Downsample(kernel_size=2, padding=0)
        self.encoder_block_2 = DoubleConvResidual(128, 256, dropout=dropout)
        self.downsample_2 = Downsample(kernel_size=2, padding=0)
        self.encoder_block_3 = DoubleConvResidual(256, 512, dropout=dropout)
        self.downsample_3 = Downsample(kernel_size=2, padding=0)
        self.bottleneck = Bottleneck(512, 512, dropout=dropout)
        self.upsample_1 = Upsample(512, 512, kernel_size=2, padding=0)
        self.cbam_1 = ConvolutionalBlockAttention(512, 512, dropout=dropout)
        self.decoder_block_1 = DoubleConvResidual(1024, 256, dropout=dropout)
        self.upsample_2 = Upsample(256, 256, kernel_size=2, padding=0)
        self.cbam_2 = ConvolutionalBlockAttention(256, 256, dropout=dropout)
        self.decoder_block_2 = DoubleConvResidual(512, 128, dropout=dropout)
        self.upsample_3 = Upsample(128, 128, kernel_size=2, padding=0)
        self.cbam_3 = ConvolutionalBlockAttention(128, 128, dropout=dropout)
        self.decoder_block_3 = DoubleConvResidual(256, decoder_channels, dropout=dropout)

    def forward(self, rad, rae):
        x = torch.cat([rad, rae], dim=1)
        if self.pad_width > 0:
            x = F.pad(x, (0, self.pad_width, 0, 0))

        x = self.start_conv(x)
        encoder_1 = self.encoder_block_1(x)
        down_1 = self.downsample_1(encoder_1)
        encoder_2 = self.encoder_block_2(down_1)
        down_2 = self.downsample_2(encoder_2)
        encoder_3 = self.encoder_block_3(down_2)
        down_3 = self.downsample_3(encoder_3)

        bottleneck = self.bottleneck(down_3)

        upsample_1 = self.upsample_1(bottleneck)
        if upsample_1.shape[-2:] != encoder_3.shape[-2:]:
            upsample_1 = F.interpolate(
                upsample_1,
                size=encoder_3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        concat_1 = torch.cat([upsample_1, self.cbam_1(encoder_3)], dim=1)
        decoder_1 = self.decoder_block_1(concat_1)

        upsample_2 = self.upsample_2(decoder_1)
        if upsample_2.shape[-2:] != encoder_2.shape[-2:]:
            upsample_2 = F.interpolate(
                upsample_2,
                size=encoder_2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        concat_2 = torch.cat([upsample_2, self.cbam_2(encoder_2)], dim=1)
        decoder_2 = self.decoder_block_2(concat_2)

        upsample_3 = self.upsample_3(decoder_2)
        if upsample_3.shape[-2:] != encoder_1.shape[-2:]:
            upsample_3 = F.interpolate(
                upsample_3,
                size=encoder_1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        concat_3 = torch.cat([upsample_3, self.cbam_3(encoder_1)], dim=1)
        decoder_3 = self.decoder_block_3(concat_3)

        return {
            "backbone_feat": decoder_3,
        }


class RADECenterPointDecoder(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, num_classes=2):
        super().__init__()
        self.cls_head = ExpandedCenterHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )
        self.reg_head = ExpandedRegHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=8,
        )

    def forward(self, x):
        cls_logits = self.cls_head(x)
        box_reg = self.reg_head(x)
        return {
            "cls_logits": cls_logits,
            "center_offset": box_reg[:, 0:2],
            "center_height": box_reg[:, 2:3],
            "size": box_reg[:, 3:6],
            "yaw": box_reg[:, 6:8],
            "box_reg": box_reg,
        }


class RADRAERADENetCenterPointModel(nn.Module):
    """
    model13: RADE-Net-style CBAM U-Net backbone with the paper's dilated neck and
    decoupled CenterPoint-style heads, adapted to MVRSS RAD/RAE inputs.
    """

    def __init__(
            self,
            d_in=64,
            e_in=37,
            num_classes=2,
            decoder_hidden_channels=128,
            dropout=0.0,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = RADRAERADEBackbone(
            in_channels=d_in + e_in,
            decoder_channels=128,
            dropout=dropout,
            pad_width=5,
        )
        self.neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))
        self.decoder = RADECenterPointDecoder(
            in_channels=128,
            hidden_channels=decoder_hidden_channels,
            num_classes=num_classes,
        )
        self.register_buffer("_model13_radenet_marker", torch.ones(1), persistent=True)

    def forward(self, rad, rae):
        features = self.backbone(rad, rae)
        fused_feat = self.neck(features["backbone_feat"])
        decoded = self.decoder(fused_feat)
        return {
            **features,
            "fused_feat": fused_feat,
            **decoded,
            "heatmap_logits": decoded["cls_logits"],
        }
