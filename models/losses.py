"""
Loss functions for manga panel segmentation

Includes:
- BCE Loss
- Dice Loss
- Boundary Loss (weighted Dice or BCE on boundary region)
- Combined Loss (BCE + Dice + λ * Boundary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 1, H, W) - logits
            y_true: (B, 1, H, W) - binary mask [0, 1]
        """
        y_pred = torch.sigmoid(y_pred)
        
        # Flatten
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        y_true_flat = y_true.view(y_true.size(0), -1)
        
        intersection = (y_pred_flat * y_true_flat).sum(dim=1)
        union = y_pred_flat.sum(dim=1) + y_true_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss: focuses on 2-3px boundary region of GT mask
    
    Two modes:
    - 'dice': Boundary-weighted Dice loss
    - 'bce': Boundary-weighted BCE loss with higher weight on boundary
    """
    def __init__(self, mode='dice', boundary_width=3, boundary_weight=3.0, smooth=1e-7):
        super().__init__()
        self.mode = mode
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.smooth = smooth
    
    def extract_boundary(self, mask):
        """
        Extract boundary region from binary mask
        
        Args:
            mask: (B, 1, H, W) - binary mask [0, 1]
        
        Returns:
            boundary: (B, 1, H, W) - boundary region [0, 1]
        """
        # Erosion and dilation to find boundary
        kernel_size = self.boundary_width
        padding = kernel_size // 2
        
        # Max pooling for dilation
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # -Max pooling for erosion (invert, max pool, invert back)
        eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Boundary = dilated - eroded
        boundary = dilated - eroded
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 1, H, W) - logits
            y_true: (B, 1, H, W) - binary mask [0, 1]
        """
        # Extract boundary region
        boundary_mask = self.extract_boundary(y_true)
        
        if self.mode == 'dice':
            # Boundary-weighted Dice
            y_pred_sig = torch.sigmoid(y_pred)
            
            # Weight map: higher weight on boundary
            weight_map = 1.0 + (self.boundary_weight - 1.0) * boundary_mask
            
            # Weighted intersection and union
            y_pred_flat = (y_pred_sig * weight_map).view(y_pred_sig.size(0), -1)
            y_true_flat = (y_true * weight_map).view(y_true.size(0), -1)
            
            intersection = (y_pred_flat * y_true_flat).sum(dim=1)
            union = y_pred_flat.sum(dim=1) + y_true_flat.sum(dim=1)
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()
            
        elif self.mode == 'bce':
            # Boundary-weighted BCE
            weight_map = 1.0 + (self.boundary_weight - 1.0) * boundary_mask
            
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
            weighted_bce = bce * weight_map
            
            return weighted_bce.mean()
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'dice' or 'bce'")


class CombinedLoss(nn.Module):
    """
    Combined Loss: BCE + Dice + λ * Boundary
    
    Default configuration:
    - BCE weight: 0.5
    - Dice weight: 0.5
    - Boundary weight (λ): 0.2-0.4
    """
    def __init__(self, 
                 bce_weight=0.5, 
                 dice_weight=0.5, 
                 boundary_lambda=0.3,
                 boundary_mode='dice',
                 boundary_width=3,
                 boundary_weight=3.0):
        super().__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_lambda = boundary_lambda
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(
            mode=boundary_mode,
            boundary_width=boundary_width,
            boundary_weight=boundary_weight
        )
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 1, H, W) - logits
            y_true: (B, 1, H, W) - binary mask [0, 1]
        """
        loss_bce = self.bce(y_pred, y_true)
        loss_dice = self.dice(y_pred, y_true)
        loss_boundary = self.boundary(y_pred, y_true)
        
        total_loss = (self.bce_weight * loss_bce + 
                      self.dice_weight * loss_dice + 
                      self.boundary_lambda * loss_boundary)
        
        return total_loss, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'boundary': loss_boundary.item(),
            'total': total_loss.item()
        }


# Test code
if __name__ == "__main__":
    print("="*60)
    print("Loss Functions Test")
    print("="*60)
    
    # Create dummy data
    B, C, H, W = 4, 1, 128, 128
    y_pred = torch.randn(B, C, H, W)  # logits
    y_true = torch.randint(0, 2, (B, C, H, W)).float()  # binary mask
    
    print(f"\nInput shapes:")
    print(f"  y_pred (logits): {y_pred.shape}")
    print(f"  y_true (mask): {y_true.shape}")
    
    # Test Dice Loss
    print("\n1. Dice Loss:")
    dice_loss = DiceLoss()
    loss = dice_loss(y_pred, y_true)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Boundary Loss (Dice mode)
    print("\n2. Boundary Loss (Dice mode):")
    boundary_loss_dice = BoundaryLoss(mode='dice', boundary_width=3, boundary_weight=3.0)
    loss = boundary_loss_dice(y_pred, y_true)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Boundary Loss (BCE mode)
    print("\n3. Boundary Loss (BCE mode):")
    boundary_loss_bce = BoundaryLoss(mode='bce', boundary_width=3, boundary_weight=3.0)
    loss = boundary_loss_bce(y_pred, y_true)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    print("\n4. Combined Loss:")
    combined_loss = CombinedLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        boundary_lambda=0.3,
        boundary_mode='dice'
    )
    total_loss, loss_dict = combined_loss(y_pred, y_true)
    print(f"   BCE: {loss_dict['bce']:.4f}")
    print(f"   Dice: {loss_dict['dice']:.4f}")
    print(f"   Boundary: {loss_dict['boundary']:.4f}")
    print(f"   Total: {loss_dict['total']:.4f}")
    
    # Test backward pass
    print("\n5. Backward pass test:")
    y_pred.requires_grad = True
    total_loss, _ = combined_loss(y_pred, y_true)
    total_loss.backward()
    print(f"   Gradient computed: {y_pred.grad is not None}")
    print(f"   Gradient shape: {y_pred.grad.shape}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
