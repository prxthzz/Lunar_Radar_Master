"""
Orchestrates the full v3 (segmentation-based) LunaTwin AI pipeline.
Single entry point the multi-page Streamlit app calls.

Two detection engines:
  * "model"     - runs the trained DeepLabV3+ / U-Net (per-pixel softmax).
  * "classical" - runs the pseudo-label feature pipeline directly on the image
                  (Hough craters + blob boulders + texture rough/smooth). Always
                  available, no weights needed, and spatially varying by design.
  * "auto"      - uses the trained model if weights actually loaded, otherwise
                  falls back to "classical" (never a random model).
"""
import cv2
import numpy as np

from . import enhancement, fusion, mission_briefing, pseudo_labels, segmentation_model, terrain3d

GRID_SIZE = 10


def load_model(weights_path: str, device: str = "cpu"):
    return segmentation_model.load_trained_model(weights_path, device=device)


def _debug_class_counts(class_map: np.ndarray) -> dict:
    """Debug helper: np.unique(pred_mask, return_counts=True) + class percentages."""
    vals, counts = np.unique(class_map, return_counts=True)
    total = max(1, class_map.size)
    by_value = {int(v): int(c) for v, c in zip(vals, counts)}
    by_name = {}
    for v, c in zip(vals, counts):
        cls_id = int(v)
        if 0 <= cls_id < len(segmentation_model.CLASS_NAMES):
            by_name[segmentation_model.CLASS_NAMES[cls_id]] = f"{100.0 * int(c) / total:.1f}%"
        else:
            by_name[f"unknown_{cls_id}"] = f"{100.0 * int(c) / total:.1f}%"
    print("[debug] np.unique(pred_mask, return_counts=True):", by_value)
    print("[debug] class percentages:", by_name)
    return by_name


def _run_model_engine(enhanced: np.ndarray, model, device: str) -> dict:
    pred = segmentation_model.predict(model, enhanced, device=device)
    print("[debug] engine=trained model |", pred["class_map"].shape)
    return {
        "class_map": pred["class_map"],
        "confidence_map": pred["confidence"],
        "confidence_entropy_map": pred["confidence_entropy"],
        "probs": pred["probs"],
        "logits": pred["logits"],
    }


def _run_classical_engine(enhanced: np.ndarray) -> dict:
    fallback = pseudo_labels.build_pseudo_mask_from_image(enhanced)
    confidence = pseudo_labels.fallback_confidence_map(enhanced, fallback)
    print("[debug] engine=classical pseudo-label | craters detected:",
          0 if fallback["craters"] is None else len(fallback["craters"]),
          "| rough px:", int(fallback["rough_mask"].sum()),
          "| boulder px:", int(fallback["boulder_mask"].sum()))
    return {
        "class_map": fallback["class_map"],
        "confidence_map": confidence,
        "confidence_entropy_map": confidence,
        "probs": None,
        "logits": None,
        "craters": fallback["craters"],
    }


def run_pipeline(image_bgr: np.ndarray, model, clahe_clip=3.0, gamma=0.6,
                 device="cpu", engine="auto", model_info=None) -> dict:
    model_info = model_info or {"loaded": False, "arch": None, "path": None, "message": ""}
    enhanced = enhancement.enhance_pipeline(image_bgr, clip_limit=clahe_clip, gamma=gamma)

    if engine == "classical":
        pred_result = _run_classical_engine(enhanced)
        engine_used = "classical pseudo-label pipeline"
    elif engine == "model":
        if model_info.get("loaded"):
            pred_result = _run_model_engine(enhanced, model, device)
            engine_used = f"trained model ({model_info.get('arch')})"
        else:
            print("[warn] trained-model engine requested but weights are unavailable —",
                  "falling back to classical pseudo-label pipeline.")
            pred_result = _run_classical_engine(enhanced)
            engine_used = "classical pseudo-label pipeline (model weights unavailable)"
    else:  # auto
        if model_info.get("loaded"):
            pred_result = _run_model_engine(enhanced, model, device)
            engine_used = f"trained model ({model_info.get('arch')})"
        else:
            print("[info] no trained weights — using classical pseudo-label pipeline.")
            pred_result = _run_classical_engine(enhanced)
            engine_used = "classical pseudo-label pipeline"

    class_map = pred_result["class_map"]
    confidence_map = pred_result["confidence_map"]
    probs = pred_result.get("probs")
    logits = pred_result.get("logits")

    # ------------------------------------------------------------------
    # DEBUG OUTPUT — proves the map is spatially varying, not uniform
    # ------------------------------------------------------------------
    _debug_class_counts(class_map)
    print("[debug] confidence_map range:", float(confidence_map.min()),
          "-", float(confidence_map.max()),
          "| std:", float(confidence_map.std()),
          "| unique pixels w/ different conf:", int((np.diff(np.unique(confidence_map)) > 0).sum() + 1) if confidence_map.size else 0)

    stats = fusion.class_area_stats(class_map)
    class_counts = np.bincount(class_map.ravel(), minlength=segmentation_model.NUM_CLASSES)
    class_distribution = {
        segmentation_model.CLASS_NAMES[i]: {
            "count": int(class_counts[i]),
            "percent": round(100.0 * class_counts[i] / max(1, class_map.size), 2),
        }
        for i in range(segmentation_model.NUM_CLASSES)
    }

    hazard_grid, dominant_class_grid, confidence_grid = fusion.compute_hazard_grid(
        class_map, confidence_map, grid_size=GRID_SIZE
    )
    score, verdict, confidence_pct = fusion.landing_score_and_verdict(hazard_grid, confidence_grid)
    risk = fusion.risk_breakdown(stats)
    explanations = fusion.build_explanations(stats, dominant_class_grid, hazard_grid)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    height_map = terrain3d.estimate_height_map(gray)

    hazards_count = sum(
        1 for name in ("crater", "boulder", "rough_terrain") if stats[name]["pixel_count"] > 0
    )

    result = {
        "original": image_bgr,
        "enhanced": enhanced,
        "class_map": class_map,
        "class_distribution": class_distribution,
        "confidence_map": confidence_map,
        "confidence_entropy_map": pred_result.get("confidence_entropy_map"),
        "hazard_grid": hazard_grid,
        "dominant_class_grid": dominant_class_grid,
        "confidence_grid": confidence_grid,
        "class_stats": stats,
        "risk_breakdown": risk,
        "explanations": explanations,
        "score": score,
        "verdict": verdict,
        "confidence_pct": confidence_pct,
        "hazards_count": hazards_count,
        "height_map": height_map,
        "engine_used": engine_used,
        "model_info": model_info,
        "probs": probs,
        "logits": logits,
        "craters": pred_result.get("craters"),
    }
    result["briefing"] = mission_briefing.generate_briefing(result)
    return result
