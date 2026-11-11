"""
Phase 1-2: Preprocessing + Enhancement.
Classical, fast, no model weights needed -> runs instantly on every uploaded image.
"""
import cv2
import numpy as np


def load_and_resize(image_bgr: np.ndarray, size: int = 640) -> np.ndarray:
    """Resize to a fixed working resolution (keeps YOLO input + downstream steps predictable)."""
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # pad to square so the digital-twin / grid overlays stay simple
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off, x_off = (size - new_h) // 2, (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def denoise(image_bgr: np.ndarray, method: str = "gaussian") -> np.ndarray:
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(image_bgr, None, 6, 6, 7, 21)
    return cv2.GaussianBlur(image_bgr, (3, 3), 0)


def clahe_enhance(image_bgr: np.ndarray, clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB so color balance stays natural."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gamma_correct(image_bgr: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)


def sharpen(image_bgr: np.ndarray, amount: float = 0.6) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), 3)
    return cv2.addWeighted(image_bgr, 1 + amount, blurred, -amount, 0)


def enhance_pipeline(image_bgr: np.ndarray, clip_limit=3.0, gamma=0.6, do_sharpen=True) -> np.ndarray:
    """Full enhancement chain used before both the YOLO detector and the terrain-risk pass."""
    img = load_and_resize(image_bgr)
    img = denoise(img, "gaussian")
    img = clahe_enhance(img, clip_limit=clip_limit)
    img = gamma_correct(img, gamma=gamma)
    if do_sharpen:
        img = sharpen(img)
    return img
