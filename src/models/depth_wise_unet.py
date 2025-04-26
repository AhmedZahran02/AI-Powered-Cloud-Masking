import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class EfficientDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_depthwise=True):
        super().__init__()
        
        if use_depthwise:
            self.double_conv = nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels),
                DepthwiseSeparableConv(out_channels, out_channels)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_depthwise=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            EfficientDoubleConv(in_channels, out_channels, use_depthwise)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_depthwise=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = EfficientDoubleConv(in_channels, out_channels, use_depthwise)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = EfficientDoubleConv(in_channels, out_channels, use_depthwise)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class EfficientUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EfficientUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Reduced feature channels throughout the network
        base_filters = 32  # Starting with fewer filters (was 64)
        
        # First layer uses standard convolutions as the input channels are just 4
        self.inc = EfficientDoubleConv(n_channels, base_filters, use_depthwise=False)
        
        # Use depthwise separable convolutions for deeper layers
        self.down1 = Down(base_filters, base_filters*2, use_depthwise=True)  # 32 -> 64
        self.down2 = Down(base_filters*2, base_filters*4, use_depthwise=True)  # 64 -> 128
        self.down3 = Down(base_filters*4, base_filters*8, use_depthwise=True)  # 128 -> 256
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters*8, base_filters*16 // factor, use_depthwise=True)  # 256 -> 512 or 256
        
        self.up1 = Up(base_filters*16, base_filters*8 // factor, bilinear, use_depthwise=True)  # 512 -> 256 or 128
        self.up2 = Up(base_filters*8, base_filters*4 // factor, bilinear, use_depthwise=True)  # 256 -> 128 or 64
        self.up3 = Up(base_filters*4, base_filters*2 // factor, bilinear, use_depthwise=True)  # 128 -> 64 or 32
        self.up4 = Up(base_filters*2, base_filters, bilinear, use_depthwise=True)  # 64 -> 32
        
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
