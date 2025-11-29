"""
Mask R-CNN for Panel Instance Segmentation (Grayscale input)

This model uses Mask R-CNN with ResNet-50-FPN backbone for instance 
segmentation of manga panels.

Input: 1 channel (Grayscale) - replicated to 3 channels internally
Output: Instance masks, bounding boxes, and scores for each panel

Reference:
- Mask R-CNN: https://arxiv.org/abs/1703.06870
"""

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np


class MaskRCNNGray(nn.Module):
    """
    Mask R-CNN for Panel Instance Segmentation with Grayscale input
    
    Input: 1 channel grayscale image (replicated to 3 channels internally)
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # Background + Panel
        pretrained: bool = True,
        trainable_backbone_layers: int = 3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained:
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            self.model = maskrcnn_resnet50_fpn(
                weights=weights,
                trainable_backbone_layers=trainable_backbone_layers
            )
        else:
            self.model = maskrcnn_resnet50_fpn(
                weights=None,
                trainable_backbone_layers=trainable_backbone_layers
            )
        
        # Replace the box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace the mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    
    def _expand_grayscale(self, images):
        """
        Expand grayscale images to 3 channels
        
        Args:
            images: List of (1, H, W) tensors or (B, 1, H, W) tensor
        
        Returns:
            List of (3, H, W) tensors
        """
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                # Batched input
                images = [img for img in images]
            else:
                images = [images]
        
        expanded = []
        for img in images:
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            expanded.append(img)
        
        return expanded
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of (1, H, W) tensors or batched (B, 1, H, W) tensor
            targets: List of dicts with 'boxes', 'labels', 'masks' for training
        
        Returns:
            Training: dict with losses
            Inference: List of dicts with 'boxes', 'labels', 'scores', 'masks'
        """
        images = self._expand_grayscale(images)
        return self.model(images, targets)
    
    @torch.no_grad()
    def predict(self, images, score_threshold=0.5, mask_threshold=0.5):
        """
        Predict instances
        
        Args:
            images: List of (1, H, W) tensors or (B, 1, H, W) tensor
            score_threshold: Minimum score for predictions
            mask_threshold: Threshold for binary masks
        
        Returns:
            List of dicts with 'boxes', 'labels', 'scores', 'masks'
        """
        self.eval()
        
        images = self._expand_grayscale(images)
        outputs = self.model(images)
        
        results = []
        for output in outputs:
            keep = output['scores'] >= score_threshold
            
            boxes = output['boxes'][keep]
            labels = output['labels'][keep]
            scores = output['scores'][keep]
            masks = output['masks'][keep]
            
            masks = (masks > mask_threshold).squeeze(1).cpu().numpy()
            
            results.append({
                'boxes': boxes.cpu().numpy(),
                'labels': labels.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'masks': masks
            })
        
        return results
    
    @torch.no_grad()
    def predict_instances(self, images, score_threshold=0.5, mask_threshold=0.5):
        """
        Predict and return instance maps (compatible with Mask2Former interface)
        
        Args:
            images: (B, 1, H, W) tensor
        
        Returns:
            instance_maps: List of (H, W) arrays with instance IDs
            instance_infos: List of dicts with instance information
        """
        self.eval()
        
        # Store original dimensions
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            B, C, H, W = images.shape
        else:
            H, W = images[0].shape[1:]
        
        images_3ch = self._expand_grayscale(images)
        outputs = self.model(images_3ch)
        
        instance_maps = []
        instance_infos = []
        
        for i, output in enumerate(outputs):
            instance_map = np.zeros((H, W), dtype=np.int32)
            
            keep = output['scores'] >= score_threshold
            
            boxes = output['boxes'][keep].cpu().numpy()
            scores = output['scores'][keep].cpu().numpy()
            masks = output['masks'][keep]
            
            masks = (masks > mask_threshold).squeeze(1).cpu().numpy()
            
            infos = []
            for j, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                instance_id = j + 1
                instance_map[mask > 0] = instance_id
                
                infos.append({
                    'id': instance_id,
                    'score': float(score),
                    'bbox': box.tolist(),
                    'area': int(mask.sum())
                })
            
            instance_maps.append(instance_map)
            instance_infos.append(infos)
        
        return instance_maps, instance_infos


def create_maskrcnn_gray(
    num_classes: int = 2,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3
):
    """
    Factory function to create Mask R-CNN model for grayscale input
    
    Args:
        num_classes: Number of classes (2 = background + panel)
        pretrained: Whether to use COCO pretrained weights
        trainable_backbone_layers: Number of trainable backbone layers (0-5)
    
    Returns:
        MaskRCNNGray model
    """
    return MaskRCNNGray(
        num_classes=num_classes,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers
    )


if __name__ == '__main__':
    print("Testing Mask R-CNN Gray model creation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = create_maskrcnn_gray(pretrained=True).to(device)
    model.eval()
    print("✅ Model created")
    
    # Test forward pass with grayscale input (1 channel)
    dummy_images = torch.randn(2, 1, 384, 512).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"✅ Forward pass successful")
    print(f"   - Input shape: {dummy_images.shape}")
    print(f"   - Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"   - Image {i}: {len(out['boxes'])} detections")
    
    # Test predict_instances
    instance_maps, instance_infos = model.predict_instances(dummy_images)
    print(f"✅ predict_instances successful")
    for i, (imap, info) in enumerate(zip(instance_maps, instance_infos)):
        print(f"   - Image {i}: {len(info)} instances")
