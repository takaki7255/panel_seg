"""
Training script for Mask R-CNN Panel Instance Segmentation

Supports both:
- 3-channel input (Gray + LSD + SDF)
- 1-channel input (Grayscale only)

Usage:
    # 3-channel (Gray + LSD + SDF)
    python train_maskrcnn.py \
        --root ./frame_dataset/1000_instance \
        --input-type 3ch \
        --epochs 50 \
        --batch 4 \
        --output ./panel_models/maskrcnn

    # Grayscale only
    python train_maskrcnn.py \
        --root ./frame_dataset/1000_instance \
        --input-type gray \
        --epochs 50 \
        --batch 4 \
        --output ./panel_models/maskrcnn_gray
"""

import argparse
import time
import glob
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.maskrcnn import create_maskrcnn
from models.maskrcnn_gray import create_maskrcnn_gray


# ============================================================================
# Dataset
# ============================================================================
class PanelInstanceDataset(Dataset):
    """
    Dataset for Mask R-CNN training
    
    Supports both 3-channel (Gray+LSD+SDF) and 1-channel (Gray) input
    """
    
    def __init__(self, root_dir, split='train', img_size=(384, 512), 
                 input_type='gray', augment=True):
        self.root = Path(root_dir) / split
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.input_type = input_type  # 'gray' or '3ch'
        self.augment = augment and (split == 'train')
        
        if not self.root.exists():
            self.root = Path(root_dir)
        
        self.img_dir = self.root / 'images'
        self.instance_dir = self.root / 'instance_masks'
        self.lsd_dir = self.root / 'lsd'
        self.sdf_dir = self.root / 'sdf'
        
        self.img_paths = sorted(glob.glob(str(self.img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(self.img_dir / "*.png")))
        
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        
        self.has_lsd = self.lsd_dir.exists()
        self.has_sdf = self.sdf_dir.exists()
        
        if input_type == '3ch' and (not self.has_lsd or not self.has_sdf):
            print("⚠️  Warning: LSD/SDF not found, falling back to grayscale")
            self.input_type = 'gray'
        
        print(f"Dataset: {len(self.img_paths)} images")
        print(f"Input type: {self.input_type}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        stem = Path(img_path).stem
        
        # Load grayscale image
        img_gray = Image.open(img_path).convert('L')
        img_gray = img_gray.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img_gray_np = np.array(img_gray, dtype=np.float32) / 255.0
        
        # Load LSD/SDF if 3-channel
        if self.input_type == '3ch':
            lsd_path = self.lsd_dir / f"{stem}_lsd.png"
            sdf_path = self.sdf_dir / f"{stem}_sdf.png"
            
            if lsd_path.exists():
                lsd = Image.open(lsd_path).convert('L')
                lsd = lsd.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
                lsd_np = np.array(lsd, dtype=np.float32) / 255.0
            else:
                lsd_np = np.zeros_like(img_gray_np)
            
            if sdf_path.exists():
                sdf = Image.open(sdf_path).convert('L')
                sdf = sdf.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
                sdf_np = np.array(sdf, dtype=np.float32) / 255.0
            else:
                sdf_np = np.zeros_like(img_gray_np)
            
            # Stack to 3 channels
            img_np = np.stack([img_gray_np, lsd_np, sdf_np], axis=0)
        else:
            # Single channel
            img_np = img_gray_np[np.newaxis, ...]
        
        # Load instance mask
        instance_path = self.instance_dir / f"{stem}_instance.png"
        instance_mask = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
        if instance_mask is None:
            raise FileNotFoundError(f"Instance mask not found: {instance_path}")
        
        instance_mask = cv2.resize(
            instance_mask,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                img_np = np.flip(img_np, axis=2).copy()
                instance_mask = np.fliplr(instance_mask).copy()
        
        # Convert to tensor
        image = torch.from_numpy(img_np).float()
        
        # Extract instances
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        boxes = []
        masks = []
        labels = []
        
        for inst_id in instance_ids:
            mask = (instance_mask == inst_id).astype(np.uint8)
            
            # Get bounding box
            pos = np.where(mask)
            if len(pos[0]) == 0:
                continue
            
            ymin, ymax = pos[0].min(), pos[0].max()
            xmin, xmax = pos[1].min(), pos[1].max()
            
            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(mask)
            labels.append(1)  # Panel class
        
        if len(boxes) == 0:
            # No valid instances - create dummy
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, self.img_size[0], self.img_size[1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target


def collate_fn(batch):
    """Custom collate function for Mask R-CNN"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================================
# Training
# ============================================================================
def train_one_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Forward pass returns losses in training mode
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'loss_cls': f'{loss_dict.get("loss_classifier", 0):.3f}',
            'loss_mask': f'{loss_dict.get("loss_mask", 0):.3f}'
        })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, device):
    """Validate model - compute average loss"""
    model.train()  # Keep in train mode to get losses
    
    total_loss = 0
    num_batches = 0
    
    for images, targets in tqdm(loader, desc="Validating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
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
    print("Mask R-CNN Training for Panel Instance Segmentation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input type: {args.input_type}")
    print(f"Dataset: {args.root}")
    print("=" * 60 + "\n")
    
    # Create model
    if args.input_type == 'gray':
        model = create_maskrcnn_gray(
            num_classes=2,
            pretrained=args.pretrained,
            trainable_backbone_layers=args.trainable_layers
        )
    else:
        model = create_maskrcnn(
            num_classes=2,
            pretrained=args.pretrained,
            trainable_backbone_layers=args.trainable_layers
        )
    
    model = model.to(device)
    print("✅ Model created")
    
    # Datasets
    img_size = tuple(args.img_size)
    
    train_dataset = PanelInstanceDataset(
        args.root, split='train', img_size=img_size,
        input_type=args.input_type, augment=True
    )
    val_dataset = PanelInstanceDataset(
        args.root, split='val', img_size=img_size,
        input_type=args.input_type, augment=False
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
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == 'sgd':
        optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # WandB
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="panel-seg-maskrcnn",
            name=f"maskrcnn-{args.input_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Step scheduler
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
            model_name = f'maskrcnn_{args.input_type}_best.pt'
            torch.save(model.state_dict(), output_dir / model_name)
            print(f"✅ Best model saved (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch{epoch}.pt')
    
    # Save final model
    model_name = f'maskrcnn_{args.input_type}_final.pt'
    torch.save(model.state_dict(), output_dir / model_name)
    print(f"\n✅ Final model saved to {output_dir / model_name}")
    
    # Save config
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
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for Panel Segmentation')
    
    # Data
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of instance dataset')
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512],
                        help='Image size (H W)')
    parser.add_argument('--input-type', type=str, default='gray',
                        choices=['gray', '3ch'],
                        help='Input type: gray (1ch) or 3ch (Gray+LSD+SDF)')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--trainable-layers', type=int, default=3,
                        help='Number of trainable backbone layers (0-5)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step'])
    parser.add_argument('--workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output', type=str, default='./panel_models/maskrcnn')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--wandb', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
