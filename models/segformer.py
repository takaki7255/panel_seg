"""
SegFormer for Manga Panel Segmentation

Uses Hugging Face transformers implementation with:
- 3-channel input (Gray + LSD + SDF)
- Modified first patch embedding for 3ch input
- Pretrained MiT-B2/B3 backbone (optional)
- Boundary loss integration

Architecture:
    - Encoder: Mix Transformer (MiT-B2 or MiT-B3)
    - Decoder: All-MLP decoder
    - Input: 3 channels (Gray + LSD + SDF)
    - Output: Single channel segmentation mask

Benefits:
    - Wide receptive field (good for dense pages)
    - Lightweight and fast
    - Better context modeling than U-Net
    - Reduces connection/separation errors

Usage:
    # Create model
    model = SegFormerPanel(
        model_name='nvidia/mit-b2',
        in_channels=3,
        num_labels=1,
        pretrained=True
    )
    
    # Forward pass
    output = model(images)  # (B, 3, H, W) -> (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormerPanel(nn.Module):
    """
    SegFormer for manga panel segmentation with 3-channel input
    
    Args:
        model_name (str): Pretrained model name or path
            - 'nvidia/mit-b0': Smallest (3.7M params)
            - 'nvidia/mit-b1': Small (13.7M params)
            - 'nvidia/mit-b2': Medium (24.7M params) - Recommended
            - 'nvidia/mit-b3': Large (44.6M params)
            - 'nvidia/mit-b4': Very large (61.4M params)
            - 'nvidia/mit-b5': Largest (81.9M params)
        in_channels (int): Number of input channels (default: 3 for Gray+LSD+SDF)
        num_labels (int): Number of output classes (default: 1 for binary segmentation)
        pretrained (bool): Use pretrained weights (default: True)
        freeze_encoder (bool): Freeze encoder during initial training (default: False)
    """
    def __init__(self, 
                 model_name='nvidia/mit-b2',
                 in_channels=3, 
                 num_labels=1,
                 pretrained=True,
                 freeze_encoder=False):
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_labels = num_labels
        self.pretrained = pretrained
        
        # Load pretrained model or create from config
        if pretrained:
            try:
                self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True  # Allow different num_labels
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
            # Create from config without pretrained weights
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            self.segformer = SegformerForSemanticSegmentation(config)
            print(f"✅ Created SegFormer from scratch: {model_name}")
        
        # Modify first patch embedding for 3-channel input (instead of 3ch RGB)
        # SegFormer uses overlapping patch embeddings in the encoder
        # We need to replace the first patch embedding layer
        if in_channels != 3:
            self._modify_first_conv(in_channels)
        
        # Freeze encoder if requested (for transfer learning)
        if freeze_encoder:
            self._freeze_encoder()
            print(f"❄️  Encoder frozen for transfer learning")
        
        print(f"SegFormerPanel initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Input channels: {in_channels}")
        print(f"  - Output labels: {num_labels}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Freeze encoder: {freeze_encoder}")
    
    def _modify_first_conv(self, in_channels):
        """
        Modify the first convolutional layer to accept different number of input channels
        
        SegFormer uses patch embeddings in stages. We modify the first stage.
        """
        # Get the first patch embedding layer
        first_patch_embed = self.segformer.segformer.encoder.patch_embeddings[0]
        
        # Get original conv layer
        old_proj = first_patch_embed.proj
        
        # Create new conv layer with same params but different in_channels
        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )
        
        # Initialize new conv layer
        if self.pretrained and in_channels == 3:
            # If in_channels is 3, copy weights directly
            with torch.no_grad():
                new_proj.weight.copy_(old_proj.weight)
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)
        elif self.pretrained:
            # If in_channels differs, average RGB weights for each input channel
            with torch.no_grad():
                # Average the RGB weights
                avg_weight = old_proj.weight.mean(dim=1, keepdim=True)
                # Repeat for in_channels
                new_proj.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)
        # else: random initialization (when not pretrained)
        
        # Replace the projection layer
        first_patch_embed.proj = new_proj
        
        print(f"  Modified first patch embedding: 3ch -> {in_channels}ch")
    
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
            x: (B, in_channels, H, W) - input image
        
        Returns:
            logits: (B, num_labels, H, W) - output logits
        """
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits  # (B, num_labels, H/4, W/4)
        
        # Upsample to original size
        logits = F.interpolate(
            logits,
            size=x.shape[2:],  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        return logits
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model': 'SegFormerPanel',
            'backbone': self.model_name,
            'in_channels': self.in_channels,
            'num_labels': self.num_labels,
            'pretrained': self.pretrained,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# Test code
if __name__ == "__main__":
    print("="*60)
    print("SegFormerPanel Model Test")
    print("="*60)
    
    # Test with MiT-B2
    print("\n1. Creating SegFormerPanel (MiT-B2):")
    try:
        model = SegFormerPanel(
            model_name='nvidia/mit-b2',
            in_channels=3,
            num_labels=1,
            pretrained=True,
            freeze_encoder=False
        )
        
        # Model info
        info = model.get_model_info()
        print(f"\nModel Info:")
        print(f"  Backbone: {info['backbone']}")
        print(f"  Total parameters: {info['total_params']:,}")
        print(f"  Trainable parameters: {info['trainable_params']:,}")
        
        # Forward pass test
        print(f"\n2. Forward pass test:")
        x = torch.randn(2, 3, 384, 512)  # (B, C, H, W)
        with torch.no_grad():
            out = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        
        # Test with different sizes
        print(f"\n3. Different input sizes:")
        test_sizes = [(256, 256), (512, 512), (384, 512)]
        for h, w in test_sizes:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                out = model(x)
            print(f"  {h}x{w} -> {out.shape}")
        
        print("\n" + "="*60)
        print("✅ Test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        print("\nNote: Make sure 'transformers' is installed:")
        print("  pip install transformers")
        print("\nFor offline use, download models first:")
        print("  from transformers import SegformerForSemanticSegmentation")
        print("  SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b2')")
