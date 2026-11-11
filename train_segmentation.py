"""
============================================================================
LunaTwin AI — 4-class lunar hazard SEGMENTATION training (Colab-friendly)
============================================================================
Trains a pixel-wise multi-class segmentation model:
    input : lunar image (RGB or grayscale, resized to IMG_SIZE)
    output: per-pixel logits over 4 classes
        0 = smooth_safe, 1 = crater, 2 = boulder, 3 = rough_terrain
Loss   : nn.CrossEntropyLoss with class weights (pixel-wise, ignore 255)
Arch   : --arch unet        compact custom U-Net  (fast, CPU-friendly)
          --arch deeplabv3  torchvision DeepLabV3+ mobilenet_v3_large
Saves  : a checkpoint dict {"arch": ..., "state_dict": ...} readable by
         luna_core/segmentation_model.load_trained_model().

--------------------------------------------------------------------------
RUNNING IN GOOGLE COLAB (Runtime > Change runtime type > T4 GPU)
--------------------------------------------------------------------------
Run the whole file cell-by-cell. If you have the Kaggle YOLO crater dataset
prepared as data/lunar/{images,masks}/{train,val} it is used directly;
otherwise the script AUTO-GENERATES a synthetic lunar dataset (perfect
pixel labels) so you get a working, spatially-varying model in minutes.

--------------------------------------------------------------------------
RUNNING LOCALLY (CPU)
--------------------------------------------------------------------------
    python train_segmentation.py --arch unet --epochs 20 --img-size 256 \
        --out hazard_unet.pth
============================================================================
"""

# ---------------------------------------------------------------------------
# CELL 1 — Install (Colab)
# ---------------------------------------------------------------------------
# !pip install -q torch torchvision albumentations opencv-python-headless tqdm scikit-image

# ---------------------------------------------------------------------------
# CELL 2 — Imports + config
# ---------------------------------------------------------------------------
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose
from albumentations import (HorizontalFlip, VerticalFlip, RandomRotate90,
                            RandomBrightnessContrast, GaussNoise, Resize,
                            RandomResizedCrop, Normalize)
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from luna_core import segmentation_model

NUM_CLASSES = 4
CLASS_NAMES = segmentation_model.CLASS_NAMES
IGNORE_INDEX = 255

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet", choices=["unet", "deeplabv3"])
    ap.add_argument("--img-size", type=int, default=None,
                    help="default: 384 for deeplabv3, 256 for unet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data-dir", default="data/lunar")
    ap.add_argument("--out", default="hazard_unet.pth")
    ap.add_argument("--class-weights", default="1.0,2.5,2.0,1.5",
                    help="loss weights for safe, crater, boulder, rough")
    return ap.parse_args()

ARGS = parse_args()
IMG_SIZE = ARGS.img_size or (384 if ARGS.arch == "deeplabv3" else 256)
EPOCHS = ARGS.epochs
BATCH_SIZE = ARGS.batch_size
LR = ARGS.lr
OUT_PATH = ARGS.out
ARCH = ARGS.arch
CLASS_WEIGHTS = torch.tensor([float(x) for x in ARGS.class_weights.split(",")])

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] arch={ARCH} size={IMG_SIZE} epochs={EPOCHS} batch={BATCH_SIZE} "
      f"lr={LR} device={DEVICE} class_weights={CLASS_WEIGHTS.tolist()}")

# ---------------------------------------------------------------------------
# CELL 3 — SYNTHETIC quick-start dataset (perfect pixel labels)
# Used only when no real data/lunar folder exists. Gives a demonstrative model
# immediately; replace with real pseudo-masks for production (see CELL 4).
# ---------------------------------------------------------------------------
def _smooth_bg(size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    bg = 0.5 * (x / size) + 0.4 * (y / size)
    for _ in range(3):
        fx = random.randint(1, 4); fy = random.randint(1, 4)
        amp = random.uniform(6, 16)
        bg += amp * np.sin(x * fx * 2 * np.pi / size) * np.cos(y * fy * 2 * np.pi / size)
    bg += np.random.default_rng().normal(0, 3, (size, size)).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _add_rough(img, mask, rng):
    for _ in range(random.randint(2, 5)):
        c = random.randint(40, img.shape[1] - 40)
        r = random.randint(20, img.shape[0] - 20)
        rad = random.randint(25, 60)
        cell = img[c - rad:c + rad, r - rad:r + rad]
        mcell = mask[c - rad:c + rad, r - rad:r + rad]
        if cell.size == 0:
            continue
        noise = rng.normal(0, random.uniform(20, 40), cell.shape).astype(np.float32)
        cell[:] = np.clip(cell.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        mcell[:] = 3  # rough_terrain


def _add_craters(img, mask, rng):
    for _ in range(random.randint(1, 5)):
        cx, cy = random.randint(40, img.shape[1] - 40), random.randint(40, img.shape[0] - 40)
        r = random.randint(8, 35)
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        inside = d2 <= r * r
        rim = (d2 <= (r + 3) ** 2) & ~inside
        img[inside] = np.clip(img[inside].astype(np.float32) - random.uniform(50, 90), 0, 255).astype(np.uint8)
        img[rim] = np.clip(img[rim].astype(np.float32) + random.uniform(25, 50), 0, 255).astype(np.uint8)
        mask[inside] = 1  # crater


def _add_boulders(img, mask, rng):
    for _ in range(random.randint(3, 12)):
        bx, by = random.randint(5, img.shape[1] - 6), random.randint(5, img.shape[0] - 6)
        r = random.randint(2, 6)
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        d2 = (x - bx) ** 2 + (y - by) ** 2
        inside = d2 <= r * r
        sign = random.choice([-1, 1])
        img[inside] = np.clip(img[inside].astype(np.float32) + sign * random.uniform(60, 110), 0, 255).astype(np.uint8)
        mask[inside] = 2  # boulder


def make_synthetic_sample(size):
    rng = np.random.default_rng()
    img = np.dstack([_smooth_bg(size)] * 3)
    mask = np.zeros((size, size), dtype=np.uint8)  # 0 = smooth_safe
    _add_rough(img, mask, rng)
    _add_craters(img, mask, rng)
    _add_boulders(img, mask, rng)
    # boulders/rough can overwrite craters last-write-wins; enforce priority:
    priority = np.full((size, size), -1, dtype=np.int32)
    priority[mask == 3] = 1
    priority[mask == 2] = 2
    priority[mask == 1] = 3
    mask[:] = 0
    mask[priority >= 1] = 3
    mask[priority >= 2] = 2
    mask[priority >= 3] = 1
    return img, mask


def build_synthetic_dataset(n_train, n_val, size, root="data/synth"):
    root = Path(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
    n = {"train": n_train, "val": n_val}
    for split, count in n.items():
        for i in range(count):
            img, mask = make_synthetic_sample(size)
            cv2.imwrite(str(root / "images" / split / f"{i:04d}.png"), img[:, :, ::-1])
            cv2.imwrite(str(root / "masks" / split / f"{i:04d}.png"), mask)
    print(f"[data] generated synthetic dataset at {root} "
          f"(train={n_train}, val={n_val}, size={size})")
    return str(root)


# ---------------------------------------------------------------------------
# CELL 4 — Dataset classes (real pseudo-mask folder or synthetic folder)
# ---------------------------------------------------------------------------
def train_transform():
    return Compose([
        RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        GaussNoise(p=0.2),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def val_transform():
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class LunarHazardDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.files = [f for f in sorted(os.listdir(images_dir))
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.images_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_dir, os.path.splitext(fname)[0] + ".png"),
                          cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        augmented = self.transform(image=img, mask=mask)
        return augmented["image"], augmented["mask"].long()


def load_datasets(data_dir):
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    train_img, val_img = os.path.join(images_dir, "train"), os.path.join(images_dir, "val")
    train_mask, val_mask = os.path.join(masks_dir, "train"), os.path.join(masks_dir, "val")

    if (os.path.isdir(train_img) and os.path.isdir(train_mask)
            and os.listdir(train_img)):
        print(f"[data] using real dataset from {data_dir}")
        train_ds = LunarHazardDataset(train_img, train_mask, train_transform())
        val_ds = LunarHazardDataset(val_img, val_mask, val_transform())
        return train_ds, val_ds

    # No real data -> auto-generate a synthetic quick-start dataset.
    synth_dir = build_synthetic_dataset(40, 10, IMG_SIZE)
    return load_datasets(synth_dir)


DATA_DIR = ARGS.data_dir
train_ds, val_ds = load_datasets(DATA_DIR)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"[data] train={len(train_ds)} val={len(val_ds)}")


# ---------------------------------------------------------------------------
# CELL 5 — Model, loss, optimizer, scheduler
# ---------------------------------------------------------------------------
model = segmentation_model.build_model(ARCH, pretrained_backbone=False).to(DEVICE)

# Inverse-frequency-ish prior; safe dominates frames so it is down-weighted.
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE), ignore_index=IGNORE_INDEX)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
print(f"[model] {ARCH} | params={sum(p.numel() for p in model.parameters()):,}")


# ---------------------------------------------------------------------------
# CELL 6 — Training loop
# ---------------------------------------------------------------------------
def pixel_accuracy(pred, target):
    valid = target != IGNORE_INDEX
    if not valid.any():
        return float("nan")
    return float((pred[valid] == target[valid]).float().mean().item())


BEST_VAL_LOSS = float("inf")
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_acc, n = 0.0, 0.0, 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        out = out["out"] if isinstance(out, dict) else out
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        train_acc += pixel_accuracy(out.argmax(dim=1), masks) * imgs.size(0)
        n += imgs.size(0)
    train_loss /= max(n, 1)
    train_acc /= max(n, 1)

    model.eval()
    val_loss, val_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            out = out["out"] if isinstance(out, dict) else out
            loss = criterion(out, masks)
            val_loss += loss.item() * imgs.size(0)
            val_acc += pixel_accuracy(out.argmax(dim=1), masks) * imgs.size(0)
            n += imgs.size(0)
    val_loss /= max(n, 1)
    val_acc /= max(n, 1)

    scheduler.step()
    print(f"Epoch {epoch+1:3d}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
          f"  val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if val_loss < BEST_VAL_LOSS:
        BEST_VAL_LOSS = val_loss
        checkpoint = {"arch": ARCH, "state_dict": model.state_dict(),
                      "class_weights": CLASS_WEIGHTS.tolist(),
                      "epoch": epoch + 1, "val_loss": float(val_loss)}
        torch.save(checkpoint, OUT_PATH)
        print(f"  -> saved new best checkpoint -> {OUT_PATH} (val_loss={val_loss:.4f})")

print("Training complete.")

# ---------------------------------------------------------------------------
# CELL 7 — Quick visual validation on one val image (shows spatial variation)
# ---------------------------------------------------------------------------
def quick_validate(model, val_ds, index=0):
    model.eval()
    img, mask = val_ds[index]
    with torch.no_grad():
        out = model(img.unsqueeze(0).to(DEVICE))
        out = out["out"] if isinstance(out, dict) else out
        probs = F.softmax(out, dim=1)[0].cpu()
        pred = probs.argmax(dim=0).numpy()
        conf = probs.max(dim=0).values.numpy()
    gt = mask.numpy()
    print("[debug] ground-truth class counts:", np.unique(gt, return_counts=True))
    print("[debug] predicted   class counts:", np.unique(pred, return_counts=True))
    print("[debug] predicted class percentages:",
          {CLASS_NAMES[c]: f"{100 * (pred == c).mean():.1f}%" for c in range(NUM_CLASSES)})
    print("[debug] confidence range:", conf.min(), "-", conf.max())
    vis = np.hstack([cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)])
    return pred, conf


if len(val_ds) > 0:
    quick_validate(model, val_ds)

# ---------------------------------------------------------------------------
# CELL 8 — Download weights (Colab)
# ---------------------------------------------------------------------------
# from google.colab import files
# files.download(OUT_PATH)
# Then place the file in the project root and set Home's sidebar weights path.
