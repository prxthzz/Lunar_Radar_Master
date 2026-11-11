"""
Pseudo-mask generation for the 3 classes that have no real ground truth:
  boulder        -> blob/contour detection on the enhanced image
  rough_terrain   -> high edge density + high LBP/Gabor texture
  smooth_safe     -> low edge density + low texture
Crater pixels (from the real Kaggle boxes) are excluded from all three so classes
don't overlap. This module is used OFFLINE to build the training dataset — it does
NOT run at inference time; at inference the trained DeepLabV3+ model predicts all
4 classes directly from the image.

Class ids (must match segmentation_model.py and train_segmentation.py):
  0 = smooth_safe
  1 = crater
  2 = boulder
  3 = rough_terrain
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

SMOOTH_SAFE, CRATER, BOULDER, ROUGH_TERRAIN = 0, 1, 2, 3


def detect_craters_hough(enhanced_gray: np.ndarray, min_radius: int = 8,
                         max_radius: int = 0, sensitivity: float = 0.6):
    """
    Inference-time crater detection via HoughCircles — used when no YOLO labels
    exist (classical fallback engine / `make_pseudo_masks.py --no-labels`).

    Lunar craters appear as roughly circular dark depressions often with a bright
    rim. Each Hough candidate is validated: the interior disc must be meaningfully
    darker than the surrounding annulus (dark floor + bright rim), which kills the
    false positives Hough produces on random noise/rough regolith.

    Returns (mask, circles) where circles is (N,3) (cx, cy, r) or None.
    """
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    h, w = blurred.shape
    if max_radius <= 0:
        max_radius = max(16, min(h, w) // 6)
    if min_radius >= max_radius:
        min_radius = max(4, max_radius // 4)

    param2 = max(20.0, 38.0 * (1.0 - sensitivity))
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.1, minDist=max(12, min(h, w) // 8),
        param1=120, param2=param2, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return np.zeros(blurred.shape, dtype=bool), None

    mask = np.zeros(blurred.shape, dtype=bool)
    kept = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for (cx, cy, r) in np.round(circles[0]).astype(int):
        r = int(r)
        if r < min_radius:
            continue
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        interior = d2 <= (r * 0.7) ** 2
        annulus = (d2 <= (r * 1.3) ** 2) & (d2 >= (r * 0.9) ** 2)
        if not interior.any() or not annulus.any():
            continue
        interior_mean = float(blurred[interior].mean())
        annulus_mean = float(blurred[annulus].mean())
        if interior_mean < annulus_mean - 6.0:  # dark floor vs brighter rim
            mask[interior | annulus] = True
            kept.append((cx, cy, r))

    return mask, (np.array(kept).reshape(-1, 3) if kept else None)


def crater_mask_from_yolo_boxes(image_shape, boxes_xyxy) -> np.ndarray:
    """Fill an ellipse inside each YOLO box — closer to a real crater's rounded
    footprint than a hard rectangle, and keeps some background pixels inside the
    box (a crater's bounding box is not fully crater)."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for (x1, y1, x2, y2) in boxes_xyxy:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        ax, ay = max(1, int((x2 - x1) / 2 * 0.85)), max(1, int((y2 - y1) / 2 * 0.85))
        cv2.ellipse(mask.view(np.uint8), (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    return mask


def boulder_mask_from_blobs(enhanced_gray: np.ndarray, exclude_mask: np.ndarray) -> np.ndarray:
    """
    Boulders: small, compact, high-contrast blobs, distinct from the larger/more
    circular crater shapes already excluded via exclude_mask.
    """
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(enhanced_gray, dtype=bool)
    h, w = enhanced_gray.shape
    max_area = (h * w) * 0.01  # boulders are small relative to the frame; large blobs are terrain, not rocks
    min_area = 20

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity < 0.35:  # skip elongated/irregular shapes (more likely terrain edges)
            continue
        cv2.drawContours(mask.view(np.uint8), [c], -1, 1, -1)

    mask &= ~exclude_mask
    return mask


def rough_smooth_masks(enhanced_gray: np.ndarray, exclude_mask: np.ndarray, tile: int = 32):
    """Grid-tiled texture/edge analysis -> rough_terrain vs smooth_safe, excluding craters+boulders."""
    h, w = enhanced_gray.shape
    rough = np.zeros((h, w), dtype=bool)
    safe = np.zeros((h, w), dtype=bool)

    for i in range(0, h, tile):
        for j in range(0, w, tile):
            cell = enhanced_gray[i:i + tile, j:j + tile]
            if cell.size == 0:
                continue
            edges = cv2.Canny(cell, 50, 150)
            edge_density = edges.mean() / 255.0
            lbp = local_binary_pattern(cell, P=8, R=1, method="uniform")
            texture = min(lbp.var() / 40.0, 1.0)
            score = 0.5 * edge_density + 0.5 * texture

            region = (slice(i, min(i + tile, h)), slice(j, min(j + tile, w)))
            if score > 0.45:
                rough[region] = True
            elif score < 0.20:
                safe[region] = True
            # mid-range cells are left unlabeled (ambiguous) -> excluded from loss via ignore_index

    rough &= ~exclude_mask
    safe &= ~exclude_mask
    return rough, safe


def build_pseudo_mask(enhanced_bgr: np.ndarray, crater_boxes_xyxy, ignore_index: int = 255) -> np.ndarray:
    """
    Returns a single (H, W) uint8 mask with class ids 0-3, and `ignore_index` for
    pixels that don't confidently belong to any class (excluded from the training loss).
    """
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    crater = crater_mask_from_yolo_boxes((h, w), crater_boxes_xyxy)
    boulder = boulder_mask_from_blobs(gray, exclude_mask=crater)
    rough, safe = rough_smooth_masks(gray, exclude_mask=(crater | boulder))

    out = np.full((h, w), ignore_index, dtype=np.uint8)
    out[safe] = SMOOTH_SAFE
    out[rough] = ROUGH_TERRAIN
    out[boulder] = BOULDER
    out[crater] = CRATER  # craters last -> highest priority if masks overlap
    return out


def build_pseudo_mask_from_image(enhanced_bgr: np.ndarray, ignore_index: int = 255):
    """
    Self-contained pseudo-mask generation for a single enhanced image with NO
    YOLO labels. Used by the classical fallback engine at inference time (and by
    `make_pseudo_masks.py --no-labels`).

    Crater class comes from Hough circle detection (instead of YOLO boxes);
    boulder / rough / smooth come from the same blob + texture logic as the
    training-time pipeline.

    Returns dict:
      class_map   (H, W) uint8  class ids 0..3 (+ ignore_index for ambiguous)
      crater_mask, boulder_mask, rough_mask, safe_mask  bool maps (for debug)
      craters     (N,3) circles or None (for drawing)
    """
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    crater, craters = detect_craters_hough(gray)
    boulder = boulder_mask_from_blobs(gray, exclude_mask=crater)
    rough, safe = rough_smooth_masks(gray, exclude_mask=(crater | boulder))

    out = np.full((h, w), ignore_index, dtype=np.uint8)
    out[safe] = SMOOTH_SAFE
    out[rough] = ROUGH_TERRAIN
    out[boulder] = BOULDER
    out[crater] = CRATER

    return {
        "class_map": out,
        "crater_mask": crater,
        "boulder_mask": boulder,
        "rough_mask": rough,
        "safe_mask": safe,
        "craters": craters,
    }


def fallback_confidence_map(enhanced_bgr: np.ndarray, fallback: dict,
                            grid_size: int = 32) -> np.ndarray:
    """
    Per-pixel confidence for the classical fallback engine, so the heatmap is
    spatially varying too. Confidence = how decisive the local evidence was:
      - rough terrain: higher where texture/edge score is strongly rough
      - safe terrain : higher where the region is strongly smooth
      - boulder      : stronger for high-contrast, edge-rich blobs
      - crater       : higher on crater rims (strong edges)
    """
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Per-pixel roughness evidence (upsampled from a fine grid).
    from . import terrain_risk
    risk_grid = terrain_risk.compute_terrain_risk_grid(enhanced_bgr, grid_size=grid_size)
    roughness = terrain_risk.risk_grid_to_fullsize_mask(risk_grid, h, w)

    # Per-pixel edge magnitude (normalized) — strong edges = decisive boundary.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_mag = np.clip(edge_mag / (edge_mag.max() + 1e-6), 0, 1)

    conf = np.zeros((h, w), dtype=np.float32)
    class_map = fallback["class_map"]

    safe_m = class_map == SMOOTH_SAFE
    rough_m = class_map == ROUGH_TERRAIN
    boulder_m = class_map == BOULDER
    crater_m = class_map == CRATER

    conf[safe_m] = (1.0 - roughness[safe_m]).astype(np.float32)
    conf[rough_m] = roughness[rough_m].astype(np.float32)
    conf[boulder_m] = np.clip(0.3 + 0.7 * edge_mag[boulder_m], 0, 1).astype(np.float32)
    conf[crater_m] = np.clip(0.3 + 0.7 * edge_mag[crater_m], 0, 1).astype(np.float32)
    return conf
