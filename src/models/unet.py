import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Compact U-Net for binary tumor/background segmentation."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()

        b = base_channels

        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(
            b * 16,
            b * 8,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = DoubleConv(b * 16, b * 8)

        self.up3 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = DoubleConv(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(
            b * 2,
            b,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = DoubleConv(b * 2, b)

        self.out_conv = nn.Conv2d(
            b,
            out_channels,
            kernel_size=1,
        )

    @staticmethod
    def _align(skip, x):
        """Center-crop skip connection if odd dimensions differ."""
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)

        if diff_h > 0 or diff_w > 0:
            start_h = max(diff_h // 2, 0)
            start_w = max(diff_w // 2, 0)

            skip = skip[
                :,
                :,
                start_h:start_h + x.size(2),
                start_w:start_w + x.size(3),
            ]

        return skip

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.up4(bottleneck)
        e4 = self._align(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self._align(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self._align(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self._align(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
