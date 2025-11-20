"""
Training script for SegFormer (Gray + LSD + SDF input)

Uses Hugging Face transformers SegFormer with:
- 3-channel input (Gray + LSD + SDF)
- Combined loss (BCE + Dice + Boundary)
- Optional encoder freezing for transfer learning

Usage:
    # Basic training with MiT-B2
    python train_segformer.py \
        --root ./panel_dataset_processed \
        --dataset panel_seg \
        --model-name nvidia/mit-b2

    # MiT-B3 (larger model)
    python train_segformer.py \
        --root ./panel_dataset_processed \
        --dataset panel_seg \
        --model-name nvidia/mit-b3 \
        --batch 4 \
        --lr 5e-5

    # Transfer learning (freeze encoder initially)
    python train_segformer.py \
        --root ./panel_dataset_processed \
        --dataset panel_seg \
        --freeze-encoder \
        --freeze-epochs 10

    # Resume training
    python train_segformer.py --resume ./panel_models/panel_seg-segformer-b2-01.pt
"""

import glob, random, time, sys, argparse, gc
from pathlib import Path
import psutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

from models.segformer import SegFormerPanel
from models.losses import CombinedLoss

# ============================================================================
# Configuration
# ============================================================================
CFG = {
    "ROOT": Path("./panel_dataset_processed"),
    "IMG_SIZE": (512, 512),  # SegFormer works well with 512x512
    
    # Model
    "MODEL_NAME": "nvidia/mit-b2",  # or 'nvidia/mit-b3'
    "PRETRAINED": True,
    "FREEZE_ENCODER": False,
    "FREEZE_EPOCHS": 0,  # Number of epochs to freeze encoder
    
    # Loss
    "BCE_WEIGHT": 0.5,
    "DICE_WEIGHT": 0.5,
    "BOUNDARY_LAMBDA": 0.3,
    "BOUNDARY_MODE": "dice",
    "BOUNDARY_WIDTH": 3,
    "BOUNDARY_WEIGHT": 3.0,
    
    # Training
    "BATCH": 4,  # Smaller batch for larger model
    "EPOCHS": 100,
    "LR": 5e-5,  # Lower LR for pretrained transformer
    "PATIENCE": 20,
    "SEED": 42,
    
    # Memory
    "ENABLE_EMERGENCY_STOP": True,
    "EMERGENCY_GPU_THRESHOLD": 0.95,
    "EMERGENCY_RAM_THRESHOLD": 0.90,
    
    # Wandb
    "WANDB_PROJ": "panel-seg-segformer",
    "DATASET": "panel_seg",
    "RUN_NAME": "",
    
    # Output
    "MODELS_DIR": Path("panel_models"),
    "SAVE_PRED_EVERY": 10,
    "PRED_SAMPLE_N": 4,
    "PRED_DIR": "predictions",
    
    "RESUME": "",
}

# ============================================================================
# Utilities (same as U-Net scripts)
# ============================================================================
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

def check_memory_safety(threshold_gpu=0.90, threshold_ram=0.85):
    warnings = []
    if torch.cuda.is_available():
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_reserved / total_memory > threshold_gpu:
            warnings.append(f"⚠️ GPU: {mem_reserved/total_memory*100:.1f}%")
    ram = psutil.virtual_memory()
    if ram.percent / 100.0 > threshold_ram:
        warnings.append(f"⚠️ RAM: {ram.percent}%")
    return len(warnings) == 0, "\n".join(warnings)

def emergency_save_and_exit(model, cfg, epoch, reason):
    print(f"\n🚨 Emergency: {reason}")
    torch.save(model.state_dict(), f"emergency_segformer_epoch{epoch}.pth")
    sys.exit(0)

def next_version(models_dir, prefix):
    models_dir.mkdir(exist_ok=True, parents=True)
    exist = sorted(models_dir.glob(f"{prefix}-*.pt"))
    if not exist: return "01"
    return f"{int(exist[-1].stem.split('-')[-1])+1:02d}"

# ============================================================================
# Dataset
# ============================================================================
class PanelDatasetLSDSDF(Dataset):
    def __init__(self, img_dir, mask_dir, lsd_dir, sdf_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images in {img_dir}")
        self.mask_dir, self.lsd_dir, self.sdf_dir = mask_dir, lsd_dir, sdf_dir
        
        resize_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.img_tf = transforms.Compose([transforms.Resize(resize_size), transforms.ToTensor()])
        self.mask_tf = transforms.Compose([transforms.Resize(resize_size, interpolation=Image.NEAREST), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem = Path(img_p).stem
        
        img_gray = self.img_tf(Image.open(img_p).convert("L"))
        lsd = self.img_tf(Image.open(self.lsd_dir / f"{stem}_lsd.png").convert("L"))
        sdf = self.img_tf(Image.open(self.sdf_dir / f"{stem}_sdf.png").convert("L"))
        img = torch.cat([img_gray, lsd, sdf], dim=0)  # (3, H, W)
        
        mask = self.mask_tf(Image.open(self.mask_dir / f"{stem}_mask.png").convert("L"))
        mask = (mask > 0.5).float()
        
        return img, mask, stem

# ============================================================================
# Training & Evaluation
# ============================================================================
def train_epoch(model, loader, loss_fn, opt, dev, cfg, ep):
    model.train()
    total_loss = 0
    loss_components = {'bce': 0, 'dice': 0, 'boundary': 0}
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="train", leave=False)):
        if batch_idx % 10 == 0:
            is_safe, warning = check_memory_safety(cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95), cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90))
            if not is_safe and cfg.get("ENABLE_EMERGENCY_STOP", True):
                emergency_save_and_exit(model, cfg, ep, warning)
        
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
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
    
    n = len(loader.dataset)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return total_loss / n, {k: v / n for k, v in loss_components.items()}

@torch.no_grad()
def eval_epoch(model, loader, dev, cfg, ep):
    model.eval()
    dice = iou = 0
    
    for batch_idx, (x, y, _) in enumerate(tqdm(loader, desc="eval ", leave=False)):
        if batch_idx % 15 == 0:
            is_safe, warning = check_memory_safety(cfg.get("EMERGENCY_GPU_THRESHOLD", 0.95), cfg.get("EMERGENCY_RAM_THRESHOLD", 0.90))
            if not is_safe and cfg.get("ENABLE_EMERGENCY_STOP", True):
                emergency_save_and_exit(model, cfg, ep, warning)
        
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
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
    
    n = len(loader.dataset)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return dice/n, iou/n

@torch.no_grad()
def save_predictions(model, loader, cfg, epoch, run_dir, device):
    if epoch % cfg["SAVE_PRED_EVERY"] != 0: return
    model.eval()
    pred_dir = run_dir / cfg["PRED_DIR"]
    pred_dir.mkdir(exist_ok=True)
    images_for_wandb = []
    
    cnt = 0
    for x, y, stem in loader:
        for i in range(len(x)):
            if cnt >= cfg["PRED_SAMPLE_N"]: break
            img = x[i:i+1].to(device)
            gt = y[i]
            pred = torch.sigmoid(model(img))[0, 0]
            pred_bin = (pred > 0.5).cpu().numpy() * 255
            gt_np = gt[0].cpu().numpy() * 255
            
            Image.fromarray(pred_bin.astype(np.uint8)).save(pred_dir / f"pred_{epoch:03}_{stem[i]}.png")
            
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
        if cnt >= cfg["PRED_SAMPLE_N"]: break
    
    if images_for_wandb:
        wandb.log({f"pred_samples_epoch_{epoch}": images_for_wandb})
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SegFormer Training')
    parser.add_argument('--root', type=str, help='Dataset root (preprocessed)')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model-name', type=str, help='SegFormer model (nvidia/mit-b2, nvidia/mit-b3, etc.)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--no-pretrained', action='store_true', help='Train from scratch')
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze encoder initially')
    parser.add_argument('--freeze-epochs', type=int, help='Epochs to freeze encoder')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--patience', type=int, help='Patience')
    parser.add_argument('--boundary-lambda', type=float, help='Boundary lambda')
    parser.add_argument('--models-dir', type=str, help='Models dir')
    parser.add_argument('--resume', type=str, help='Resume checkpoint')
    parser.add_argument('--wandb-proj', type=str, help='Wandb project')
    parser.add_argument('--run-name', type=str, help='Run name')
    return parser.parse_args()

# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    cfg = CFG.copy()
    
    if args.root: cfg["ROOT"] = Path(args.root)
    if args.dataset: cfg["DATASET"] = args.dataset
    if args.model_name: cfg["MODEL_NAME"] = args.model_name
    if args.pretrained: cfg["PRETRAINED"] = True
    if args.no_pretrained: cfg["PRETRAINED"] = False
    if args.freeze_encoder: cfg["FREEZE_ENCODER"] = True
    if args.freeze_epochs is not None: cfg["FREEZE_EPOCHS"] = args.freeze_epochs
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
    
    # Device selection (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    
    print("\n" + "="*60)
    print("SegFormer Training (Gray + LSD + SDF)")
    print("="*60)
    print(f"Device: {dev}")
    print(f"Model: {cfg['MODEL_NAME']}")
    print(f"Dataset: {cfg['ROOT']}")
    print(f"Pretrained: {cfg['PRETRAINED']}")
    print("="*60 + "\n")
    
    # Model naming
    dataset_name = cfg["DATASET"].replace("/", "-").replace("\\", "-")
    model_short = cfg["MODEL_NAME"].split("/")[-1]  # e.g., 'mit-b2'
    prefix = f"{dataset_name}-segformer-{model_short}"
    version = next_version(cfg["MODELS_DIR"], prefix)
    model_tag = f"{prefix}-{version}"
    
    if not cfg["RUN_NAME"]: cfg["RUN_NAME"] = model_tag
    
    wandb.init(project=cfg["WANDB_PROJ"], name=cfg["RUN_NAME"], config=cfg)
    run_dir = Path(wandb.run.dir)
    
    # Dataset
    root = cfg["ROOT"]
    train_ds = PanelDatasetLSDSDF(root/"train/images", root/"train/masks", root/"train/lsd", root/"train/sdf", cfg["IMG_SIZE"])
    val_ds = PanelDatasetLSDSDF(root/"val/images", root/"val/masks", root/"val/lsd", root/"val/sdf", cfg["IMG_SIZE"])
    
    dl_tr = DataLoader(train_ds, batch_size=cfg["BATCH"], shuffle=True, num_workers=0, pin_memory=False)
    dl_va = DataLoader(val_ds, batch_size=cfg["BATCH"], shuffle=False, num_workers=0, pin_memory=False)
    
    # Model
    model = SegFormerPanel(
        model_name=cfg["MODEL_NAME"],
        in_channels=3,
        num_labels=1,
        pretrained=cfg["PRETRAINED"],
        freeze_encoder=cfg["FREEZE_ENCODER"]
    ).to(dev)
    
    if cfg["RESUME"]:
        model.load_state_dict(torch.load(cfg["RESUME"], map_location=dev))
        print(f"✅ Resumed from: {cfg['RESUME']}")
    
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
    
    best_iou = 0
    patience = 0
    
    for ep in range(1, cfg["EPOCHS"] + 1):
        t = time.time()
        
        # Unfreeze encoder after freeze_epochs
        if ep == cfg["FREEZE_EPOCHS"] + 1 and cfg["FREEZE_EPOCHS"] > 0:
            print(f"\n🔓 Unfreezing encoder at epoch {ep}")
            model.unfreeze_encoder()
            # Recreate optimizer to include encoder params
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"]-ep)
        
        tr_loss, tr_components = train_epoch(model, dl_tr, loss_fn, opt, dev, cfg, ep)
        va_dice, va_iou = eval_epoch(model, dl_va, dev, cfg, ep)
        sched.step()
        
        save_predictions(model, dl_va, cfg, ep, run_dir, dev)
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
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
        
        if va_iou > best_iou:
            best_iou, patience = va_iou, 0
            ckpt_wandb = run_dir / f"best_ep{ep:03}_iou{va_iou:.4f}.pt"
            torch.save(model.state_dict(), ckpt_wandb)
            
            cfg["MODELS_DIR"].mkdir(parents=True, exist_ok=True)
            ckpt_models = cfg["MODELS_DIR"] / f"{model_tag}.pt"
            torch.save(model.state_dict(), ckpt_models)
            print(f"  ✅ Best: {model_tag}.pt")
        else:
            patience += 1
            if patience >= cfg["PATIENCE"]:
                print(f"Early stopping at epoch {ep}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best IoU: {best_iou:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
