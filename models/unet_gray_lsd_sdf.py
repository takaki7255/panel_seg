"""
U-Net with Gray + LSD + SDF Input for Manga Panel Segmentation

3-channel input U-Net:
- Channel 0: Grayscale image
- Channel 1: LSD (Line Segment Detector) line map
- Channel 2: SDF (Signed Distance Field) - distance to nearest line

The combination of LSD and SDF provides strong boundary cues,
improving fine-grained edge detection and reducing thin line breaks.

Usage:
    model = UNetGrayLSDSDF(in_channels=3, n_classes=1)
    # Input: (B, 3, H, W) where [gray, lsd, sdf]
    output = model(input)  # (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetGrayLSDSDF(nn.Module):
    """
    U-Net with Gray + LSD + SDF Input
    
    Args:
        in_channels (int): Number of input channels (default: 3 for Gray+LSD+SDF)
        n_classes (int): Number of output classes (default: 1)
        base_channels (int): Base number of channels (default: 64)
    """
    def __init__(self, in_channels=3, n_classes=1, base_channels=64):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, n_classes, 1)
        
        print(f"UNetGrayLSDSDF initialized:")
        print(f"  - Input channels: {in_channels} (Gray + LSD + SDF)")
        print(f"  - Output classes: {n_classes}")
        print(f"  - Base channels: {base_channels}")
    
    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        x = self.out_conv(x)
        
        return x
    
    def get_model_info(self):
        return {
            'model': 'UNetGrayLSDSDF',
            'in_channels': self.in_channels,
            'n_classes': self.n_classes,
            'base_channels': self.base_channels,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


if __name__ == "__main__":
    print("="*60)
    print("UNetGrayLSDSDF Model Test")
    print("="*60)
    
    model = UNetGrayLSDSDF(in_channels=3, n_classes=1, base_channels=64)
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Total parameters: {info['total_params']:,}")
    
    x = torch.randn(2, 3, 384, 512)  # (B, 3, H, W)
    with torch.no_grad():
        out = model(x)
    print(f"\nForward pass: {x.shape} -> {out.shape}")
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
