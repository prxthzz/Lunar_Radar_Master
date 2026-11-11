"""
Fusion for the segmentation-based pipeline (v3).
Takes the model's per-pixel class map + confidence map and produces:
  - a coarse hazard grid (for the Landing Zones tab)
  - landing score + verdict
  - per-class stats (counts, area %, where concentrated)
  - a risk breakdown per hazard type (level + reason)
  - explanation lines
"""
import numpy as np

from .segmentation_model import CLASS_NAMES, NUM_CLASSES

# Per-class base risk weight (0 = no risk, 1 = maximum risk) if that class dominates a cell
CLASS_RISK_WEIGHT = {
    0: 0.05,   # smooth_safe
    1: 0.95,   # crater
    2: 0.75,   # boulder
    3: 0.55,   # rough_terrain
}


def class_area_stats(class_map: np.ndarray) -> dict:
    total = class_map.size
    stats = {}
    for cls_id, name in enumerate(CLASS_NAMES):
        count = int((class_map == cls_id).sum())
        stats[name] = {"pixel_count": count, "area_pct": round(100 * count / total, 1)}
    return stats


def compute_hazard_grid(class_map: np.ndarray, confidence_map: np.ndarray, grid_size: int = 10):
    """Per-cell hazard score = weighted mix of every class present in the cell, not just the dominant one."""
    h, w = class_map.shape
    ch, cw = h // grid_size, w // grid_size
    hazard_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    dominant_class_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    confidence_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            cell = class_map[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            conf_cell = confidence_map[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            if cell.size == 0:
                continue
            counts = np.bincount(cell.flatten(), minlength=NUM_CLASSES)
            fractions = counts / cell.size
            blended_risk = sum(fractions[c] * CLASS_RISK_WEIGHT[c] for c in range(NUM_CLASSES))
            avg_conf = float(conf_cell.mean())

            hazard_grid[i, j] = blended_risk
            dominant_class_grid[i, j] = int(np.argmax(counts))
            confidence_grid[i, j] = avg_conf

    return hazard_grid, dominant_class_grid, confidence_grid


def landing_score_and_verdict(hazard_grid: np.ndarray, confidence_grid: np.ndarray):
    avg_hazard = float(hazard_grid.mean())
    worst_hazard = float(np.percentile(hazard_grid, 90))
    avg_conf = float(confidence_grid.mean())
    central_hazard = float(np.mean(hazard_grid[1:-1, 1:-1])) if hazard_grid.size > 4 else avg_hazard
    combined = 0.45 * avg_hazard + 0.25 * worst_hazard + 0.20 * (1.0 - avg_conf) + 0.10 * central_hazard
    score = round(float(np.clip(100 * (1 - combined), 0, 100)), 1)

    if score >= 70:
        verdict = "Safe"
    elif score >= 40:
        verdict = "Caution"
    else:
        verdict = "Risky"

    overall_confidence = round(avg_conf * 100, 1)
    return score, verdict, overall_confidence


def risk_breakdown(stats: dict) -> dict:
    def level(pct, thresholds=(15, 5)):
        if pct >= thresholds[0]:
            return "High"
        if pct >= thresholds[1]:
            return "Medium"
        return "Low"

    crater_pct = stats["crater"]["area_pct"]
    boulder_pct = stats["boulder"]["area_pct"]
    rough_pct = stats["rough_terrain"]["area_pct"]

    return {
        "crater_risk": {"level": level(crater_pct, (10, 3)), "reason": f"Craters cover {crater_pct}% of the frame."},
        "boulder_risk": {"level": level(boulder_pct, (8, 2)), "reason": f"Boulder-like regions cover {boulder_pct}% of the frame."},
        "roughness_risk": {"level": level(rough_pct, (25, 10)), "reason": f"Rough/textured terrain covers {rough_pct}% of the frame."},
    }


def build_explanations(stats: dict, dominant_class_grid: np.ndarray, hazard_grid: np.ndarray):
    explanations = []
    worst_cells = np.dstack(np.unravel_index(np.argsort(-hazard_grid.ravel()), hazard_grid.shape))[0][:4]
    for (i, j) in worst_cells:
        cls_id = int(dominant_class_grid[i, j])
        cls_name = CLASS_NAMES[cls_id]
        explanations.append(
            f"Cell ({i},{j}) dominated by {cls_name.replace('_', ' ')} "
            f"(hazard score {hazard_grid[i, j]:.2f}) \u2014 flagged "
            f"{'Risky' if hazard_grid[i, j] > 0.6 else 'Caution'}."
        )
    if stats["smooth_safe"]["area_pct"] > 50:
        explanations.append(f"Over half the frame ({stats['smooth_safe']['area_pct']}%) is classified smooth/safe terrain.")
    return explanations
