"""
Training script for Mask2Former Panel Instance Segmentation

This script trains Mask2Former with Swin-T backbone for manga panel
instance segmentation.

Input: 3 channels (Grayscale + LSD + SDF)
Output: Instance masks for each panel

Usage:
    python train_mask2former.py \
        --root ./frame_dataset/1000_instance \
        --epochs 100 \
        --batch 4 \
        --lr 1e-4 \
        --output ./panel_models/mask2former

Requirements:
    pip install transformers accelerate
"""

import argparse
import time
import glob
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.mask2former import create_mask2former


# ============================================================================
# Dataset
# ============================================================================
class PanelInstanceDataset(Dataset):
    """
    Dataset for instance segmentation with Gray + LSD + SDF input
    
    Expects:
        - images/: Original images
        - instance_masks/: Instance segmentation masks (each panel has unique ID)
        - lsd/: LSD line detection maps
        - sdf/: SDF distance maps
    """
    
    def __init__(self, root_dir, split='train', img_size=(384, 512), augment=True):
        self.root = Path(root_dir) / split
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment and (split == 'train')
        
        # Check if split directory exists
        if not self.root.exists():
            self.root = Path(root_dir)
        
        self.img_dir = self.root / 'images'
        self.instance_dir = self.root / 'instance_masks'
        self.lsd_dir = self.root / 'lsd'
        self.sdf_dir = self.root / 'sdf'
        
        # Get image list
        self.img_paths = sorted(glob.glob(str(self.img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(self.img_dir / "*.png")))
        
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        
        # Check auxiliary directories
        self.has_lsd = self.lsd_dir.exists()
        self.has_sdf = self.sdf_dir.exists()
        
        print(f"Dataset: {len(self.img_paths)} images")
        print(f"  LSD: {'✅' if self.has_lsd else '❌'}")
        print(f"  SDF: {'✅' if self.has_sdf else '❌'}")
        
        # Transforms
        self.resize = transforms.Resize(self.img_size)
        self.to_tensor = transforms.ToTensor()
        
        # Augmentations
        if self.augment:
            self.aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
            ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        stem = Path(img_path).stem
        
        # Load grayscale image
        img_gray = Image.open(img_path).convert('L')
        img_gray = self.resize(img_gray)
        
        # Load LSD
        if self.has_lsd:
            lsd_path = self.lsd_dir / f"{stem}_lsd.png"
            if lsd_path.exists():
                lsd = Image.open(lsd_path).convert('L')
                lsd = self.resize(lsd)
            else:
                lsd = Image.new('L', self.img_size[::-1], 0)
        else:
            lsd = Image.new('L', self.img_size[::-1], 0)
        
        # Load SDF
        if self.has_sdf:
            sdf_path = self.sdf_dir / f"{stem}_sdf.png"
            if sdf_path.exists():
                sdf = Image.open(sdf_path).convert('L')
                sdf = self.resize(sdf)
            else:
                sdf = Image.new('L', self.img_size[::-1], 0)
        else:
            sdf = Image.new('L', self.img_size[::-1], 0)
        
        # Load instance mask
        instance_path = self.instance_dir / f"{stem}_instance.png"
        instance_mask = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
        if instance_mask is None:
            raise FileNotFoundError(f"Instance mask not found: {instance_path}")
        
        # Resize instance mask
        instance_mask = cv2.resize(
            instance_mask, 
            (self.img_size[1], self.img_size[0]),  # (width, height)
            interpolation=cv2.INTER_NEAREST
        )
        
        # Apply augmentations (if enabled)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img_gray = img_gray.transpose(Image.FLIP_LEFT_RIGHT)
                lsd = lsd.transpose(Image.FLIP_LEFT_RIGHT)
                sdf = sdf.transpose(Image.FLIP_LEFT_RIGHT)
                instance_mask = np.fliplr(instance_mask).copy()
            
            # Random vertical flip
            if np.random.random() > 0.7:
                img_gray = img_gray.transpose(Image.FLIP_TOP_BOTTOM)
                lsd = lsd.transpose(Image.FLIP_TOP_BOTTOM)
                sdf = sdf.transpose(Image.FLIP_TOP_BOTTOM)
                instance_mask = np.flipud(instance_mask).copy()
        
        # Convert to tensors
        img_gray_t = self.to_tensor(img_gray)
        lsd_t = self.to_tensor(lsd)
        sdf_t = self.to_tensor(sdf)
        
        # Stack channels: (3, H, W)
        pixel_values = torch.cat([img_gray_t, lsd_t, sdf_t], dim=0)
        
        # Process instance mask for Mask2Former
        # Get unique instance IDs (excluding background 0)
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        # Create mask for each instance
        masks = []
        class_labels = []
        
        for inst_id in instance_ids:
            mask = (instance_mask == inst_id).astype(np.float32)
            masks.append(torch.from_numpy(mask))
            class_labels.append(0)  # All panels are class 0
        
        if len(masks) > 0:
            mask_labels = torch.stack(masks)  # (num_instances, H, W)
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            # No instances - create dummy
            mask_labels = torch.zeros(1, self.img_size[0], self.img_size[1])
            class_labels = torch.tensor([0], dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'mask_labels': mask_labels,
            'class_labels': class_labels,
            'stem': stem,
            'num_instances': len(instance_ids)
        }


def collate_fn(batch):
    """Custom collate function for variable number of instances"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    mask_labels = [item['mask_labels'] for item in batch]  # List of tensors
    class_labels = [item['class_labels'] for item in batch]  # List of tensors
    stems = [item['stem'] for item in batch]
    num_instances = [item['num_instances'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'mask_labels': mask_labels,
        'class_labels': class_labels,
        'stems': stems,
        'num_instances': num_instances
    }


# ============================================================================
# Training
# ============================================================================
def train_one_epoch(model, loader, optimizer, device, epoch, scheduler=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        mask_labels = [m.to(device) for m in batch['mask_labels']]
        class_labels = [c.to(device) for c in batch['class_labels']]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Validating"):
        pixel_values = batch['pixel_values'].to(device)
        mask_labels = [m.to(device) for m in batch['mask_labels']]
        class_labels = [c.to(device) for c in batch['class_labels']]
        
        outputs = model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        
        total_loss += outputs.loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train(args):
    """Main training function"""
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print("\n" + "=" * 60)
    print("Mask2Former Training for Panel Instance Segmentation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.root}")
    print(f"Dropout: {args.dropout}")
    print("=" * 60 + "\n")
    
    # Create model
    model = create_mask2former(
        model_name=args.model_name,
        num_labels=1,  # Only "panel" class
        pretrained=args.pretrained,
        dropout=args.dropout
    ).to(device)
    print("✅ Model created")
    
    # Datasets
    img_size = tuple(args.img_size)
    
    train_dataset = PanelInstanceDataset(
        args.root, split='train', img_size=img_size, augment=True
    )
    val_dataset = PanelInstanceDataset(
        args.root, split='val', img_size=img_size, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch, 
        shuffle=True, 
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images\n")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        step_scheduler = False
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1
        )
        step_scheduler = True
    else:
        scheduler = None
        step_scheduler = False
    
    # WandB
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="panel-seg-mask2former",
            name=f"mask2former-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            scheduler=scheduler if step_scheduler else None
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Step scheduler (epoch-based)
        if scheduler is not None and not step_scheduler:
            scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        
        # Log to WandB
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                output_dir / 'mask2former_best.pt'
            )
            print(f"✅ Best model saved (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch{epoch}.pt')
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'mask2former_final.pt')
    print(f"\n✅ Final model saved to {output_dir / 'mask2former_final.pt'}")
    
    # Save training config
    config = vars(args)
    config['best_val_loss'] = best_val_loss
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask2Former for Panel Segmentation')
    
    # Data
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of instance dataset')
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512],
                        help='Image size (H W)')
    
    # Model
    parser.add_argument('--model-name', type=str, 
                        default='facebook/mask2former-swin-tiny-coco-instance',
                        help='Pretrained model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Train from scratch')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'onecycle', 'none'])
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0.0-0.5 recommended for regularization)')
    parser.add_argument('--workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output', type=str, default='./panel_models/mask2former')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--wandb', action='store_true', help='Use WandB logging')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
