"""
Segmentation model definitions + inference helpers for LunaTwin AI.

Two architectures are available:
  * deeplabv3  - torchvision's deeplabv3_mobilenet_v3_large (smallest DeepLabV3
                 variant torchvision ships; good accuracy/speed trade-off).
  * unet       - a compact custom U-Net (DoubleConv + skip connections). Much
                 lighter than DeepLabV3, trains fast on CPU/Colab.

Both output per-pixel logits over 4 classes:
  0 = smooth_safe, 1 = crater, 2 = boulder, 3 = rough_terrain   (see pseudo_labels.py)

Confidence: returned as *per-pixel* signals directly from the model's output:
  confidence        = max class probability (argmax's softmax score)   [0,1]
  confidence_entropy = 1 - normalized softmax entropy                   [0,1]
Either way it is a full HxW map, NOT a single scalar — that is what makes the
confidence heatmap spatially vary instead of being a uniform color.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from . import pseudo_labels

NUM_CLASSES = 4
CLASS_NAMES = ["smooth_safe", "crater", "boulder", "rough_terrain"]
CLASS_ID2NAME = {i: n for i, n in enumerate(CLASS_NAMES)}
IGNORE_INDEX = 255

ARCH_NAMES = ("deeplabv3", "unet")


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------
def build_model(arch: str = "deeplabv3", pretrained_backbone: bool = True) -> nn.Module:
    arch = (arch or "deeplabv3").lower()
    if arch == "deeplabv3":
        model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1" if pretrained_backbone else None,
            num_classes=NUM_CLASSES,
        )
        return model
    if arch == "unet":
        return UNet(in_ch=3, num_classes=NUM_CLASSES, base=32)
    raise ValueError(f"Unknown arch '{arch}'. Choose one of {ARCH_NAMES}.")


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _Up(nn.Module):
    """2x bilinear upsample + conv — alignment-safe, unlike stride-2 transposed convs."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.act(self.norm(self.up(x)))


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=4, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.e1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.e2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 8, base * 16))

        self.up1 = _Up(base * 16 + base * 8, base * 8)
        self.up2 = _Up(base * 8 + base * 4, base * 4)
        self.up3 = _Up(base * 4 + base * 2, base * 2)
        self.up4 = _Up(base * 2 + base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e0 = self.inc(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        d = self.up1(e4, e3)
        d = self.up2(d, e2)
        d = self.up3(d, e1)
        d = self.up4(d, e0)
        return self.out(d)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def resolve_weights_path(weights_path: str | None) -> Path | None:
    if not weights_path:
        return None

    candidate = Path(weights_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    search_paths = [
        candidate,
        Path.cwd() / candidate,
        Path(__file__).resolve().parent.parent / candidate,
    ]
    for path in search_paths:
        if path.exists():
            return path
    return None


def _detect_arch_from_state(state: dict) -> str:
    """Inspect key names to figure out whether a state_dict is U-Net or DeepLabV3."""
    for key in state:
        if key.startswith("inc.") or key.startswith("up1."):
            return "unet"
        if key.startswith("backbone.") or key.startswith("classifier."):
            return "deeplabv3"
    return "deeplabv3"  # default assumption


def load_trained_model(weights_path: str, device: str = "cpu"):
    """
    Loads trained weights. Returns (model, info) where info is a dict:
      {"loaded": bool, "arch": str|None, "path": str|None, "message": str}

    It never silently hands back a random model without telling the caller —
    callers (pipeline/Home) use info["loaded"] to decide whether to run the
    trained-model engine or fall back to the classical pseudo-label engine.
    """
    model = build_model("deeplabv3", pretrained_backbone=False)
    resolved_path = resolve_weights_path(weights_path)

    if resolved_path is None:
        model.to(device).eval()
        return model, {
            "loaded": False,
            "arch": None,
            "path": None,
            "message": f"weights file '{weights_path}' not found",
        }

    try:
        state = torch.load(str(resolved_path), map_location="cpu")

        arch = None
        if isinstance(state, dict) and "arch" in state:
            arch = str(state["arch"]).lower()
            state = state.get("state_dict", state)
        elif isinstance(state, dict) and "state_dict" in state:
            arch = _detect_arch_from_state(state["state_dict"])
            state = state["state_dict"]
        elif isinstance(state, dict):
            arch = _detect_arch_from_state(state)

        arch = arch if arch in ARCH_NAMES else "deeplabv3"
        model = build_model(arch, pretrained_backbone=False)
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    except (FileNotFoundError, RuntimeError, ValueError, EOFError) as exc:
        model.to(device).eval()
        return model, {
            "loaded": False,
            "arch": arch if "arch" in locals() else None,
            "path": str(resolved_path),
            "message": f"weights file '{resolved_path}' could not be loaded ({exc})",
        }

    model.to(device).eval()
    return model, {
        "loaded": True,
        "arch": arch,
        "path": str(resolved_path),
        "message": f"loaded {arch} weights from {resolved_path}",
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model: nn.Module, image_bgr: np.ndarray, device: str = "cpu", img_size: int = 512) -> dict:
    """
    Forward pass producing per-pixel outputs at the *original* image resolution.

    Returns dict:
      class_map          (H, W) uint8  argmax class id 0..3 per pixel
      confidence         (H, W) float  max class probability per pixel [0, 1]
      confidence_entropy (H, W) float  1 - normalized softmax entropy   [0, 1]
      probs              (4, H, W) float32  full softmax probabilities
      logits             (4, H, W) float32  raw model logits (for debugging)
    """
    h0, w0 = image_bgr.shape[:2]
    img_rgb = image_bgr[:, :, ::-1].copy()  # BGR -> RGB
    resized = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    resized = F.interpolate(resized.unsqueeze(0), size=(img_size, img_size),
                            mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    inp = ((resized - mean) / std).to(device)

    out = model(inp)
    if isinstance(out, dict):  # DeepLabV3 returns {'out': ...}
        out = out["out"]

    probs = F.softmax(out, dim=1)[0].cpu()          # (C, img_size, img_size)
    logits = out[0].cpu()                           # (C, img_size, img_size)

    class_map = probs.argmax(dim=0).byte().numpy()                     # (s, s)
    confidence = probs.max(dim=0).values.numpy().astype(np.float32)    # (s, s)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=0)
    confidence_entropy = (1.0 - entropy / np.log(probs.shape[0])).numpy().astype(np.float32)
    probs_np = probs.numpy().astype(np.float32)
    logits_np = logits.numpy().astype(np.float32)

    import cv2
    class_map = cv2.resize(class_map, (w0, h0), interpolation=cv2.INTER_NEAREST)
    confidence = cv2.resize(confidence, (w0, h0), interpolation=cv2.INTER_LINEAR)
    confidence_entropy = cv2.resize(confidence_entropy, (w0, h0), interpolation=cv2.INTER_LINEAR)
    probs_np = cv2.resize(probs_np.transpose(1, 2, 0), (w0, h0)).transpose(2, 0, 1)

    return {
        "class_map": class_map,
        "confidence": confidence,
        "confidence_entropy": confidence_entropy,
        "probs": probs_np,
        "logits": logits_np,
    }
