"""
Mask2Former with Swin-T backbone for Panel Instance Segmentation (Grayscale input)

This model uses Mask2Former architecture with Swin Transformer backbone
for instance segmentation of manga panels.

Input: 1 channel (Grayscale) - internally replicated to 3 channels for Swin-T
Output: Instance masks for each panel

Reference:
- Mask2Former: https://arxiv.org/abs/2112.01527
- Swin Transformer: https://arxiv.org/abs/2103.14030
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from transformers import Mask2FormerImageProcessor
import numpy as np


class Mask2FormerGray(nn.Module):
    """
    Mask2Former for Panel Instance Segmentation with Grayscale input
    
    Input: 1 channel grayscale image
    The grayscale image is replicated to 3 channels internally to match
    the Swin-T backbone's expected input.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-tiny-coco-instance",
        num_labels: int = 1,  # Only "panel" class
        pretrained: bool = True,
        dropout: float = 0.0  # Note: dropout not applied to avoid attention mask issues
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load config - Note: do NOT set config.dropout as it causes attention mask size mismatch
        config = Mask2FormerConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        # config.dropout = dropout  # Disabled: causes RuntimeError with attention mask
        
        if pretrained:
            # Load pretrained Mask2Former with modified config
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            # Create from config
            self.model = Mask2FormerForUniversalSegmentation(config)
        
        # Image processor for post-processing
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    
    def _expand_grayscale(self, x):
        """
        Expand grayscale input to 3 channels
        
        Args:
            x: (B, 1, H, W) grayscale input
        
        Returns:
            (B, 3, H, W) replicated to 3 channels
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x
    
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        """
        Forward pass
        
        Args:
            pixel_values: (B, 1, H, W) grayscale input images
            mask_labels: List of (num_instances, H, W) instance masks for training
            class_labels: List of (num_instances,) class labels for training
        
        Returns:
            If training: loss dict
            If inference: model outputs with masks_queries_logits and class_queries_logits
        """
        # Expand grayscale to 3 channels
        pixel_values = self._expand_grayscale(pixel_values)
        
        if mask_labels is not None and class_labels is not None:
            # Training mode
            outputs = self.model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
            return outputs
        else:
            # Inference mode
            outputs = self.model(pixel_values=pixel_values)
            return outputs
    
    @torch.no_grad()
    def predict_instances(self, pixel_values, threshold=0.5):
        """
        Predict instance segmentation
        
        Args:
            pixel_values: (B, 1, H, W) grayscale input
            threshold: Score threshold for predictions
        
        Returns:
            instance_maps: List of (H, W) arrays with instance IDs (0=background)
            instance_infos: List of segment information per image
        """
        self.eval()
        
        # Expand grayscale to 3 channels
        pixel_values_3ch = self._expand_grayscale(pixel_values)
        
        outputs = self.model(pixel_values=pixel_values_3ch)
        
        B, _, H, W = pixel_values.shape
        target_sizes = [(H, W)] * B
        
        predictions = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )
        
        instance_maps = []
        instance_infos = []
        
        for pred in predictions:
            instance_maps.append(pred['segmentation'])
            instance_infos.append(pred['segments_info'])
        
        return instance_maps, instance_infos


def create_mask2former_gray(
    model_name: str = "facebook/mask2former-swin-tiny-coco-instance",
    num_labels: int = 1,
    pretrained: bool = True,
    dropout: float = 0.0
):
    """
    Factory function to create Mask2Former model for grayscale input
    
    Args:
        model_name: Pretrained model name
            - "facebook/mask2former-swin-tiny-coco-instance" (Swin-T, recommended)
            - "facebook/mask2former-swin-small-coco-instance" (Swin-S)
            - "facebook/mask2former-swin-base-coco-instance" (Swin-B)
        num_labels: Number of classes (1 for panel only)
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization (0.0 to 0.5 recommended)
    
    Returns:
        Mask2FormerGray model
    """
    return Mask2FormerGray(
        model_name=model_name,
        num_labels=num_labels,
        pretrained=pretrained,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test model creation
    print("Testing Mask2FormerGray model creation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = create_mask2former_gray(pretrained=True).to(device)
    print("✅ Model created")
    
    # Test forward pass with grayscale input (1 channel)
    dummy_input = torch.randn(2, 1, 384, 512).to(device)
    
    # Inference mode
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"✅ Forward pass successful")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - masks_queries_logits shape: {outputs.masks_queries_logits.shape}")
    print(f"   - class_queries_logits shape: {outputs.class_queries_logits.shape}")
    
    # Test prediction
    instance_maps, instance_infos = model.predict_instances(dummy_input)
    print(f"✅ Prediction successful")
    print(f"   - Instance maps: {len(instance_maps)} images")
    for i, (imap, info) in enumerate(zip(instance_maps, instance_infos)):
        print(f"   - Image {i}: {len(info)} instances detected")
