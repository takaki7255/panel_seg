"""
Models package for manga panel segmentation

Available models:
- ResNetUNet: ResNet-based U-Net with RGB input
- UNetGray: U-Net with grayscale input
- UNetGrayLSD: U-Net with grayscale + LSD input
- UNetGrayLSDSDF: U-Net with grayscale + LSD + SDF input
- SegFormer: Transformer-based segmentation (coming soon)
- Mask2Former: Instance segmentation (coming soon)
"""

from .unet import ResNetUNet
from .unet_gray import UNetGray
from .unet_gray_lsd import UNetGrayLSD
from .unet_gray_lsd_sdf import UNetGrayLSDSDF
from .losses import DiceLoss, BoundaryLoss, CombinedLoss

__all__ = [
    'ResNetUNet',
    'UNetGray',
    'UNetGrayLSD',
    'UNetGrayLSDSDF',
    'DiceLoss',
    'BoundaryLoss',
    'CombinedLoss',
]
