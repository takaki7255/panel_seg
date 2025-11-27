"""
SegFormer with Grayscale Input for Manga Panel Segmentation

Uses Hugging Face transformers implementation with:
- 1-channel input (Grayscale only)
- Modified first patch embedding for 1ch input
- Pretrained MiT-B2 backbone (optional)
- Simpler and more efficient than 3-channel version

Architecture:
    - Encoder: Mix Transformer (MiT-B2 or MiT-B3)
    - Decoder: All-MLP decoder
    - Input: 1 channel (Grayscale)
    - Output: Single channel segmentation mask

Benefits over multi-channel version:
    - Simpler input - no preprocessing needed
    - Faster training and inference
    - Better suited for high-contrast manga images
    - Avoids information redundancy issues

Usage:
    # Create model
    model = SegFormerGray(
        model_name='nvidia/mit-b2',
        pretrained=True
    )
    
    # Forward pass
    output = model(gray_images)  # (B, 1, H, W) -> (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormerGray(nn.Module):
    """
    SegFormer for manga panel segmentation with grayscale input
    
    Args:
        model_name (str): Pretrained model name or path
            - 'nvidia/mit-b0': Smallest (3.7M params)
            - 'nvidia/mit-b1': Small (13.7M params)
            - 'nvidia/mit-b2': Medium (24.7M params) - Recommended
            - 'nvidia/mit-b3': Large (44.6M params)
            - 'nvidia/mit-b4': Very large (61.4M params)
            - 'nvidia/mit-b5': Largest (81.9M params)
        num_labels (int): Number of output classes (default: 1 for binary segmentation)
        pretrained (bool): Use pretrained weights (default: True)
        freeze_encoder (bool): Freeze encoder during initial training (default: False)
    """
    def __init__(self, 
                 model_name='nvidia/mit-b2',
                 num_labels=1,
                 pretrained=True,
                 freeze_encoder=False):
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = 1
        self.num_labels = num_labels
        self.pretrained = pretrained
        
        # Load pretrained model or create from config
        if pretrained:
            try:
                self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True
                )
                print(f"✅ Loaded pretrained SegFormer: {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to load pretrained model: {e}")
                print(f"   Creating model from scratch...")
                config = SegformerConfig.from_pretrained(model_name)
                config.num_labels = num_labels
                self.segformer = SegformerForSemanticSegmentation(config)
                pretrained = False
        else:
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            self.segformer = SegformerForSemanticSegmentation(config)
            print(f"✅ Created SegFormer from scratch: {model_name}")
        
        # Modify first patch embedding for 1-channel input
        self._modify_first_conv()
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
            print(f"❄️  Encoder frozen for transfer learning")
        
        print(f"SegFormerGray initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Input channels: 1 (Grayscale)")
        print(f"  - Output labels: {num_labels}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Freeze encoder: {freeze_encoder}")
    
    def _modify_first_conv(self):
        """
        Modify the first convolutional layer to accept grayscale input
        
        Averages the RGB weights into a single channel for better initialization
        """
        first_patch_embed = self.segformer.segformer.encoder.patch_embeddings[0]
        old_proj = first_patch_embed.proj
        
        # Create new conv layer with 1 input channel
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )
        
        # Initialize: average RGB weights if pretrained
        if self.pretrained:
            with torch.no_grad():
                # Average across RGB channels (dim=1)
                avg_weight = old_proj.weight.mean(dim=1, keepdim=True)
                new_proj.weight.copy_(avg_weight)
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)
        
        # Replace projection layer
        first_patch_embed.proj = new_proj
        print(f"  Modified first patch embedding: 3ch -> 1ch (grayscale)")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters for transfer learning"""
        for param in self.segformer.segformer.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.segformer.segformer.encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, 1, H, W) - grayscale input image
        
        Returns:
            logits: (B, num_labels, H, W) - output logits
        """
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits
        
        # Upsample to original size
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return logits
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model': 'SegFormerGray',
            'backbone': self.model_name,
            'in_channels': self.in_channels,
            'num_labels': self.num_labels,
            'pretrained': self.pretrained,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


if __name__ == "__main__":
    print("="*60)
    print("SegFormerGray Model Test")
    print("="*60)
    
    print("\n1. Creating SegFormerGray (MiT-B2):")
    try:
        model = SegFormerGray(
            model_name='nvidia/mit-b2',
            num_labels=1,
            pretrained=True
        )
        
        info = model.get_model_info()
        print(f"\nModel Info:")
        print(f"  Backbone: {info['backbone']}")
        print(f"  Total parameters: {info['total_params']:,}")
        print(f"  Trainable parameters: {info['trainable_params']:,}")
        
        print(f"\n2. Forward pass test:")
        x = torch.randn(2, 1, 384, 512)  # Grayscale input
        with torch.no_grad():
            out = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        
        print(f"\n3. Different input sizes:")
        test_sizes = [(256, 256), (512, 512), (384, 512)]
        for h, w in test_sizes:
            x = torch.randn(1, 1, h, w)
            with torch.no_grad():
                out = model(x)
            print(f"  {h}x{w} -> {out.shape}")
        
        print("\n" + "="*60)
        print("✅ Test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nInstall transformers: pip install transformers")
