"""
U-Net with Gray + LSD Input for Manga Panel Segmentation

2-channel input U-Net:
- Channel 1: Grayscale image
- Channel 2: LSD (Line Segment Detector) line map

The LSD line map helps the model focus on panel boundaries,
improving edge detection and reducing connection errors.

Architecture:
    - Input: 2 channels (Gray + LSD)
    - Encoder: 4 downsampling blocks
    - Bottleneck: Conv block at lowest resolution
    - Decoder: 4 upsampling blocks with skip connections
    - Output: Single channel segmentation mask

Usage:
    model = UNetGrayLSD(in_channels=2, n_classes=1)
    # Input: (B, 2, H, W) where channel 0 = gray, channel 1 = LSD
    output = model(input)  # (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double Convolution Block"""
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
    """Encoder Block: ConvBlock -> MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder Block: Upsample -> Concat -> ConvBlock"""
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


class UNetGrayLSD(nn.Module):
    """
    U-Net with Gray + LSD Input
    
    Args:
        in_channels (int): Number of input channels (default: 2 for Gray+LSD)
        n_classes (int): Number of output classes (default: 1 for binary segmentation)
        base_channels (int): Base number of channels (default: 64)
    """
    def __init__(self, in_channels=2, n_classes=1, base_channels=64):
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
        
        print(f"UNetGrayLSD initialized:")
        print(f"  - Input channels: {in_channels} (Gray + LSD)")
        print(f"  - Output classes: {n_classes}")
        print(f"  - Base channels: {base_channels}")
    
    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)  # 1/2
        x, skip2 = self.enc2(x)  # 1/4
        x, skip3 = self.enc3(x)  # 1/8
        x, skip4 = self.enc4(x)  # 1/16
        
        # Bottleneck
        x = self.bottleneck(x)  # 1/16
        
        # Decoder
        x = self.dec4(x, skip4)  # 1/8
        x = self.dec3(x, skip3)  # 1/4
        x = self.dec2(x, skip2)  # 1/2
        x = self.dec1(x, skip1)  # 1/1
        
        # Output
        x = self.out_conv(x)
        
        return x
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model': 'UNetGrayLSD',
            'in_channels': self.in_channels,
            'n_classes': self.n_classes,
            'base_channels': self.base_channels,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# Test code
if __name__ == "__main__":
    print("="*60)
    print("UNetGrayLSD Model Test")
    print("="*60)
    
    # Create model
    model = UNetGrayLSD(in_channels=2, n_classes=1, base_channels=64)
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    
    # Forward pass test
    print(f"\nForward pass test:")
    # Channel 0: Gray, Channel 1: LSD
    x = torch.randn(2, 2, 384, 512)  # (B, C, H, W)
    with torch.no_grad():
        out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    # Test with different input sizes
    print(f"\nDifferent input sizes:")
    test_sizes = [(256, 256), (512, 512), (384, 512)]
    for h, w in test_sizes:
        x = torch.randn(1, 2, h, w)
        with torch.no_grad():
            out = model(x)
        print(f"  {h}x{w} -> {out.shape}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
