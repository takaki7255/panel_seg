"""
Models package for manga panel segmentation

Available models:
- ResNetUNet: ResNet-based U-Net with RGB input
- UNetGray: U-Net with grayscale input
- UNetGrayLSD: U-Net with grayscale + LSD input
- UNetGrayLSDSDF: U-Net with grayscale + LSD + SDF input
- SegFormerPanel: Transformer-based segmentation with Gray+LSD+SDF
- SegFormerGray: Transformer-based segmentation with grayscale only
- Mask2FormerPanel: Instance segmentation with Gray+LSD+SDF (3ch)
- Mask2FormerGray: Instance segmentation with grayscale only (1ch)
- MaskRCNNPanel: Instance segmentation with Gray+LSD+SDF (3ch)
- MaskRCNNGray: Instance segmentation with grayscale only (1ch)
"""

from .unet import ResNetUNet
from .unet_gray import UNetGray
from .unet_gray_lsd import UNetGrayLSD
from .unet_gray_lsd_sdf import UNetGrayLSDSDF
from .segformer import SegFormerPanel
from .segformer_gray import SegFormerGray
from .losses import DiceLoss, BoundaryLoss, CombinedLoss

# Mask2Former requires transformers library
try:
    from .mask2former import Mask2FormerPanel, Mask2FormerPanelSimple, create_mask2former
    from .mask2former_gray import Mask2FormerGray, create_mask2former_gray
    MASK2FORMER_AVAILABLE = True
except ImportError:
    MASK2FORMER_AVAILABLE = False
    Mask2FormerPanel = None
    Mask2FormerPanelSimple = None
    create_mask2former = None
    Mask2FormerGray = None
    create_mask2former_gray = None

# Mask R-CNN uses torchvision
try:
    from .maskrcnn import MaskRCNNPanel, create_maskrcnn
    from .maskrcnn_gray import MaskRCNNGray, create_maskrcnn_gray
    MASKRCNN_AVAILABLE = True
except ImportError:
    MASKRCNN_AVAILABLE = False
    MaskRCNNPanel = None
    create_maskrcnn = None
    MaskRCNNGray = None
    create_maskrcnn_gray = None

__all__ = [
    'ResNetUNet',
    'UNetGray',
    'UNetGrayLSD',
    'UNetGrayLSDSDF',
    'SegFormerPanel',
    'SegFormerGray',
    'DiceLoss',
    'BoundaryLoss',
    'CombinedLoss',
    'Mask2FormerPanel',
    'Mask2FormerPanelSimple',
    'create_mask2former',
    'Mask2FormerGray',
    'create_mask2former_gray',
    'MASK2FORMER_AVAILABLE',
    'MaskRCNNPanel',
    'create_maskrcnn',
    'MaskRCNNGray',
    'create_maskrcnn_gray',
    'MASKRCNN_AVAILABLE'
]
