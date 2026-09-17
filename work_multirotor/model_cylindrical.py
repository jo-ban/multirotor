import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicConv2d(nn.Module):
    """방위각(theta) 방향은 circular, 높이(z) 방향은 replicate padding."""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("현재 구현은 kernel_size=3만 지원합니다.")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=bias)

    def forward(self, x):
        # 마지막 축이 theta이므로 좌우 끝을 순환 연결한다.
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return self.conv(x)


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            PeriodicConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            PeriodicConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CylindricalUNet2D(nn.Module):
    """원통 전개면(theta-z)의 속도 성분 하나를 예측하는 2D U-Net.

    입력 shape : (B, 5, Nz, Ntheta)
    출력 shape : (B, 1, Nz, Ntheta)
    """

    def __init__(self, in_channels=5, out_channels=1, base_channels=16):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv2D(in_channels, b)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv2D(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv2D(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv2D(b * 4, b * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv2D(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = DoubleConv2D(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv2D(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv2D(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv2D(b * 2, b)
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat((self.up4(bottleneck), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.out_conv(d1)
