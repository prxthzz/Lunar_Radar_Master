"""
Classical terrain-risk analysis for regions the YOLO detector does NOT flag as craters.
This is what splits "non-crater" into safe vs rough — there is no ground-truth label
for this in the dataset, so it is explicitly rule-based, not learned.

Grid-based: the enhanced image is divided into cells; each cell gets a roughness score
from local texture (LBP variance) + edge density + local contrast.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def _cell_roughness(cell_gray: np.ndarray) -> float:
    # Edge density
    edges = cv2.Canny(cell_gray, 50, 150)
    edge_density = edges.mean() / 255.0

    # Local Binary Pattern texture variance (higher = rougher / more rubbly)
    lbp = local_binary_pattern(cell_gray, P=8, R=1, method="uniform")
    lbp_var = lbp.var() / 40.0  # empirical normalization

    # Local contrast (std of intensity)
    contrast = cell_gray.std() / 128.0

    score = 0.45 * edge_density + 0.35 * min(lbp_var, 1.0) + 0.20 * min(contrast, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_terrain_risk_grid(enhanced_bgr: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Returns an (grid_size x grid_size) array of roughness scores in [0, 1]."""
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ch, cw = h // grid_size, w // grid_size
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            if cell.size == 0:
                continue
            grid[i, j] = _cell_roughness(cell)
    return grid


def risk_grid_to_fullsize_mask(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Upsample the coarse grid to a per-pixel roughness map for overlay/heatmap rendering."""
    return cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
