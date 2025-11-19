"""
Training script for UNetGray (Grayscale input U-Net)

Usage:
    # Basic training
    python train_unet_gray.py --root ./panel_dataset --dataset panel_seg

    # Custom parameters
    python train_unet_gray.py \
        --root ./panel_dataset \
        --dataset panel_seg \
        --batch 16 \
        --epochs 150 \
        --lr 1e-4 \
        --base-channels 64

    # Resume training
    python train_unet_gray.py --resume ./panel_models/panel_seg-unetgray-01.pt
"""

import glob, random, time, os, sys, argparse, gc
from pathlib import Path
import psutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import wandb

# Import models and losses
from models.unet_gray import UNetGray
from models.losses import CombinedLoss

# ============================================================================
# Configuration
# ============================================================================
CFG = {
    # Dataset
    "ROOT": Path("./panel_dataset"),
    "IMG_SIZE": (384, 512),  # (height, width)
    
    # Model
    "BASE_CHANNELS": 64,
    
    # Loss
    "BCE_WEIGHT": 0.5,
    "DICE_WEIGHT": 0.5,
    "BOUNDARY_LAMBDA": 0.3,
    "BOUNDARY_MODE": "dice",  # 'dice' or 'bce'
    "BOUNDARY_WIDTH": 3,
    "BOUNDARY_WEIGHT": 3.0,
    
    # Training
    "BATCH": 8,
    "EPOCHS": 100,
    "LR": 1e-4,
    "PATIENCE": 15,
    "SEED": 42,
    
    # Memory management
    "ENABLE_EMERGENCY_STOP": True,
    "EMERGENCY_GPU_THRESHOLD": 0.95,
    "EMERGENCY_RAM_THRESHOLD": 0.90,
    
    # Wandb
    "WANDB_PROJ": "panel-seg-unet",
    "DATASET": "panel_seg",
    "RUN_NAME": "",
    
    # Output
    "MODELS_DIR": Path("panel_models"),
    "SAVE_PRED_EVERY": 10,
    "PRED_SAMPLE_N": 4,
    "PRED_DIR": "predictions",
    
    # Resume
    "RESUME": "",
}

# ============================================================================
# Utilities
# ============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_memory_safety(threshold_gpu=0.90, threshold_ram=0.85):
    """Check memory usage"""
    warnings = []
    
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_usage = mem_reserved / total_memory
        
        if gpu_usage > threshold_gpu:
            warnings.append(f"⚠️ GPU usage critical: {gpu_usage*100:.1f}% ({mem_reserved:.2f}GB/{total_memory:.2f}GB)")
    
    ram = psutil.virtual_memory()
    ram_usage = ram.percent / 100.0
    
    if ram_usage > threshold_ram:
        warnings.append(f"⚠️ RAM usage critical: {ram_usage*100:.1f}% ({ram.used/1e9:.2f}GB/{ram.total/1e9:.2f}GB)")
    
    is_safe = len(warnings) == 0
    warning_message = "\n".join(warnings) if warnings else ""
    
    return is_safe, warning_message

def emergency_save_and_exit(model, cfg, epoch, reason):
    """Emergency save and exit"""
    print("\n" + "="*60)
    print("🚨 Emergency Stop")
    print("="*60)
    print(f"Reason: {reason}")
    print(f"Current epoch: {epoch}")
    
    emergency_path = f"emergency_epoch{epoch}.pth"
    try:
        torch.save(model.state_dict(), emergency_path)
        print(f"✅ Emergency save: {emergency_path}")
    except Exception as e:
        print(f"❌ Emergency save failed: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\nExiting safely...")
    print("="*60)
    sys.exit(0)

def next_version(models_dir, prefix):
    """Get next version number"""
    models_dir.mkdir(exist_ok=True, parents=True)
    exist = sorted(models_dir.glob(f"{prefix}-*.pt"))
    if not exist:
        return "01"
    last = int(exist[-1].stem.split("-")[-1])
    return f"{last+1:02d}"

# ============================================================================
# Dataset
# ============================================================================
class PanelDataset(Dataset):
    """Dataset for grayscale manga panel segmentation"""
    def __init__(self, img_dir, mask_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        self.mask_dir = mask_dir
        
        if isinstance(img_size, tuple):
            resize_size = img_size
        else:
            resize_size = (img_size, img_size)
        
        # Grayscale transforms (no ImageNet normalization)
        self.img_tf = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
        
        self.mask_tf = transforms.Compose([
            transforms.Resize(resize_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem = Path(img_p).stem
        mask_p = self.mask_dir / f"{stem}_mask.png"
        
        # Load as grayscale
        img = self.img_tf(Image.open(img_p).convert("L"))  # Grayscale
        mask = self.mask_tf(Image.open(mask_p).convert("L"))
        mask = (mask > 0.5).float()
        
        return img, mask, stem

# ============================================================================
# Training & Evaluation
# ============================================================================
def train_epoch(model, loader, loss_fn, opt, dev, cfg, current_epoch):
    model.train()
    total_loss = 0
    loss_components = {'bce': 0, 'dice': 0, 'boundary': 0}
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="train", leave=False)):
        if batch_idx % 10 == 0:
            is_safe, warning = check_memory_safety(
                threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95),
                threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90)
            )
            if not is_safe and cfg.get("ENABLE_EMERGENCY_STOP", True):
                print(f"\n[Batch {batch_idx}] {warning}")
                emergency_save_and_exit(model, cfg, current_epoch, warning)
        
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss, loss_dict = loss_fn(logits, y)
        loss.backward()
        opt.step()
        
        total_loss += loss.item() * x.size(0)
        for k in loss_components:
            loss_components[k] += loss_dict[k] * x.size(0)
        
        del x, y, logits, loss
        
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    n = len(loader.dataset)
    avg_loss = total_loss / n
    avg_components = {k: v / n for k, v in loss_components.items()}
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return avg_loss, avg_components

@torch.no_grad()
def eval_epoch(model, loader, dev, cfg, current_epoch):
    model.eval()
    dice = iou = 0
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="eval ", leave=False)):
        if batch_idx % 15 == 0:
            is_safe, warning = check_memory_safety(
                threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95),
                threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90)
            )
            if not is_safe and cfg.get("ENABLE_EMERGENCY_STOP", True):
                print(f"\n[Eval Batch {batch_idx}] {warning}")
                emergency_save_and_exit(model, cfg, current_epoch, warning)
        
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        p = torch.sigmoid(model(x))
        pb = (p > 0.5).float()
        
        inter = (pb * y).sum((2, 3))
        union = (pb + y - pb * y).sum((2, 3))
        
        dice_den = (pb + y).sum((2, 3)) + 1e-7
        dice += torch.where(dice_den > 1e-6, 2*inter/dice_den, torch.ones_like(dice_den)).mean().item() * x.size(0)
        
        iou_den = union + 1e-7
        iou += torch.where(iou_den > 1e-6, inter/iou_den, torch.ones_like(iou_den)).mean().item() * x.size(0)
        
        del x, y, p, pb, inter, union
        
        if batch_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    n = len(loader.dataset)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return dice/n, iou/n

@torch.no_grad()
def save_predictions(model, loader, cfg, epoch, run_dir, device):
    """Save prediction samples"""
    if epoch % cfg["SAVE_PRED_EVERY"] != 0:
        return
    
    model.eval()
    pred_dir = run_dir / cfg["PRED_DIR"]
    pred_dir.mkdir(exist_ok=True)
    images_for_wandb = []
    
    cnt = 0
    for x, y, stem in loader:
        for i in range(len(x)):
            if cnt >= cfg["PRED_SAMPLE_N"]:
                break
            
            img = x[i:i+1].to(device)
            gt = y[i]
            
            pred = torch.sigmoid(model(img))[0, 0]
            pred_bin = (pred > 0.5).cpu().numpy() * 255
            gt_np = gt[0].cpu().numpy() * 255
            
            out_path = pred_dir / f"pred_{epoch:03}_{stem[i]}.png"
            Image.fromarray(pred_bin.astype(np.uint8)).save(out_path)
            
            # Visualize
            orig_np = (img[0].cpu()[0].numpy() * 255).astype(np.uint8)
            h, w = orig_np.shape[:2]
            display_h, display_w = h // 2, w // 2
            
            orig_small = Image.fromarray(orig_np).resize((display_w, display_h))
            gt_small = Image.fromarray(gt_np.astype(np.uint8)).resize((display_w, display_h))
            pred_small = Image.fromarray(pred_bin.astype(np.uint8)).resize((display_w, display_h))
            
            trio = np.concatenate([
                np.stack([np.array(orig_small)]*3, 2),
                np.stack([np.array(gt_small)]*3, 2),
                np.stack([np.array(pred_small)]*3, 2)
            ], axis=1)
            
            images_for_wandb.append(wandb.Image(trio, caption=f"ep{epoch:03}-{stem[i]}"))
            cnt += 1
            
            del img, gt, pred, pred_bin, gt_np, orig_np
            
        if cnt >= cfg["PRED_SAMPLE_N"]:
            break
    
    if images_for_wandb:
        wandb.log({f"pred_samples_epoch_{epoch}": images_for_wandb})
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='UNetGray Training Script')
    
    parser.add_argument('--root', type=str, help='Dataset root directory')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--base-channels', type=int, help='Base channels for U-Net')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--boundary-lambda', type=float, help='Boundary loss weight')
    parser.add_argument('--models-dir', type=str, help='Models directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--wandb-proj', type=str, help='Wandb project name')
    parser.add_argument('--run-name', type=str, help='Wandb run name')
    
    return parser.parse_args()

# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    cfg = CFG.copy()
    
    # Update config from args
    if args.root: cfg["ROOT"] = Path(args.root)
    if args.dataset: cfg["DATASET"] = args.dataset
    if args.base_channels: cfg["BASE_CHANNELS"] = args.base_channels
    if args.batch: cfg["BATCH"] = args.batch
    if args.epochs: cfg["EPOCHS"] = args.epochs
    if args.lr: cfg["LR"] = args.lr
    if args.patience: cfg["PATIENCE"] = args.patience
    if args.boundary_lambda: cfg["BOUNDARY_LAMBDA"] = args.boundary_lambda
    if args.models_dir: cfg["MODELS_DIR"] = Path(args.models_dir)
    if args.resume: cfg["RESUME"] = args.resume
    if args.wandb_proj: cfg["WANDB_PROJ"] = args.wandb_proj
    if args.run_name: cfg["RUN_NAME"] = args.run_name
    
    seed_everything(cfg["SEED"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*60)
    print("UNetGray Training")
    print("="*60)
    print(f"Device: {dev}")
    print(f"Dataset: {cfg['ROOT']}")
    print(f"Base channels: {cfg['BASE_CHANNELS']}")
    print(f"Boundary λ: {cfg['BOUNDARY_LAMBDA']}")
    print("="*60 + "\n")
    
    # Model naming
    dataset_name = cfg["DATASET"].replace("/", "-").replace("\\", "-")
    prefix = f"{dataset_name}-unetgray"
    version = next_version(cfg["MODELS_DIR"], prefix)
    model_tag = f"{prefix}-{version}"
    
    if not cfg["RUN_NAME"]:
        cfg["RUN_NAME"] = model_tag
    
    # Wandb init
    wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
    run_dir = Path(wandb.run.dir)
    
    # Dataset
    root = cfg["ROOT"]
    train_ds = PanelDataset(root/"train/images", root/"train/masks", cfg["IMG_SIZE"])
    val_ds = PanelDataset(root/"val/images", root/"val/masks", cfg["IMG_SIZE"])
    
    dl_tr = DataLoader(train_ds, batch_size=cfg["BATCH"], shuffle=True, num_workers=0, pin_memory=False)
    dl_va = DataLoader(val_ds, batch_size=cfg["BATCH"], shuffle=False, num_workers=0, pin_memory=False)
    
    # Model
    model = UNetGray(in_channels=1, n_classes=1, base_channels=cfg["BASE_CHANNELS"]).to(dev)
    
    if cfg["RESUME"]:
        model.load_state_dict(torch.load(cfg["RESUME"], map_location=dev))
        print(f"✅ Resumed from: {cfg['RESUME']}")
    
    # Log model info
    model_info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}\n")
    wandb.config.update(model_info)
    
    # Loss and optimizer
    loss_fn = CombinedLoss(
        bce_weight=cfg["BCE_WEIGHT"],
        dice_weight=cfg["DICE_WEIGHT"],
        boundary_lambda=cfg["BOUNDARY_LAMBDA"],
        boundary_mode=cfg["BOUNDARY_MODE"],
        boundary_width=cfg["BOUNDARY_WIDTH"],
        boundary_weight=cfg["BOUNDARY_WEIGHT"]
    )
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"])
    
    # Training loop
    best_iou = 0
    patience = 0
    
    for ep in range(1, cfg["EPOCHS"] + 1):
        t = time.time()
        
        # Memory check
        is_safe, warning = check_memory_safety(
            threshold_gpu=cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95) - 0.02,
            threshold_ram=cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90) - 0.02
        )
        if not is_safe and cfg.get("ENABLE_EMERGENCY_STOP", True):
            print(f"\n[Epoch {ep} Start] {warning}")
            emergency_save_and_exit(model, cfg, ep, f"Epoch start memory issue\n{warning}")
        
        # Train
        tr_loss, tr_components = train_epoch(model, dl_tr, loss_fn, opt, dev, cfg, ep)
        va_dice, va_iou = eval_epoch(model, dl_va, dev, cfg, ep)
        sched.step()
        
        # Save predictions
        save_predictions(model, dl_va, cfg, ep, run_dir, dev)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Log
        wandb.log({
            "epoch": ep,
            "loss": tr_loss,
            "loss_bce": tr_components['bce'],
            "loss_dice": tr_components['dice'],
            "loss_boundary": tr_components['boundary'],
            "val_dice": va_dice,
            "val_iou": va_iou,
            "lr": sched.get_last_lr()[0]
        })
        
        print(f"[{ep:03}] loss={tr_loss:.4f} (bce={tr_components['bce']:.3f} dice={tr_components['dice']:.3f} bdry={tr_components['boundary']:.3f}) "
              f"dice={va_dice:.4f} iou={va_iou:.4f}  {time.time()-t:.1f}s")
        
        # Save best
        if va_iou > best_iou:
            best_iou, patience = va_iou, 0
            ckpt_wandb = run_dir / f"best_ep{ep:03}_iou{va_iou:.4f}.pt"
            torch.save(model.state_dict(), ckpt_wandb)
            
            cfg["MODELS_DIR"].mkdir(parents=True, exist_ok=True)
            ckpt_models = cfg["MODELS_DIR"] / f"{model_tag}.pt"
            torch.save(model.state_dict(), ckpt_models)
            print(f"  ✅ Best model saved: {model_tag}.pt")
        else:
            patience += 1
            if patience >= cfg["PATIENCE"]:
                print(f"Early stopping at epoch {ep}")
                break
    
    print("\n" + "="*60)
    print(f"Training completed! Best IoU: {best_iou:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
