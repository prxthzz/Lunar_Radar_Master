"""
Lightweight pseudo-3D "digital twin" view.
NOT real DEM data — estimated from image shading (shape-from-shading style).
Always label it as such in the UI.
"""
import cv2
import numpy as np
import plotly.graph_objects as go


def estimate_height_map(enhanced_gray: np.ndarray, blur_sigma: float = 3.0) -> np.ndarray:
    """Brighter = higher/sun-facing, darker = lower/shadowed. Sobel relief sharpens the result."""
    gray = enhanced_gray.astype(np.float32) / 255.0
    smoothed = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=5)
    relief = np.sqrt(gx ** 2 + gy ** 2)
    relief = relief / (relief.max() + 1e-6)

    height = 0.7 * smoothed + 0.3 * relief
    height = cv2.GaussianBlur(height, (0, 0), 1.5)  # remove residual per-pixel noise
    return height


# Discrete colorscale mapping class id -> color. Values in each band [k, k+1)
# get a solid color: bands at 0/0.25/0.5/0.75/1.0 with cmin=-0.5, cmax=3.5.
CLASS_SURFACE_COLORSCALE = [
    [0.00, "#3ac85a"], [0.25, "#3ac85a"],   # 0 = smooth_safe  (green)
    [0.25, "#ef4b4b"], [0.50, "#ef4b4b"],   # 1 = crater       (red)
    [0.50, "#f5a524"], [0.75, "#f5a524"],   # 2 = boulder      (orange)
    [0.75, "#f5d524"], [1.00, "#f5d524"],   # 3 = rough_terrain (yellow)
]
CLASS_SURFACE_CMIN, CLASS_SURFACE_CMAX = -0.5, 3.5


def build_surface_figure(height_map: np.ndarray, color_overlay: np.ndarray = None,
                         class_map: np.ndarray = None,
                         downsample: int = 4, z_exaggeration: float = 25.0) -> go.Figure:
    """
    Downsamples for performance (a 640x640 mesh is too heavy for smooth in-browser rotation).

    Coloring (highest priority first):
      * class_map  -> per-cell hazard class with a discrete 4-color colorscale, so
                      crater/boulder/rough/safe regions each get their own color.
      * color_overlay -> drapes the enhanced/hazard-overlay image as the surface color.
      * otherwise  -> grayscale height.
    """
    h, w = height_map.shape
    hm = cv2.resize(height_map, (w // downsample, h // downsample), interpolation=cv2.INTER_AREA)

    if class_map is not None:
        cls = cv2.resize(class_map.astype(np.float32), (w // downsample, h // downsample),
                         interpolation=cv2.INTER_NEAREST).astype(np.float32)
        fig = go.Figure(data=[go.Surface(
            z=hm * z_exaggeration,
            surfacecolor=cls,
            colorscale=CLASS_SURFACE_COLORSCALE,
            cmin=CLASS_SURFACE_CMIN, cmax=CLASS_SURFACE_CMAX,
            showscale=False,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.3, roughness=0.6),
            lightposition=dict(x=100, y=200, z=300),
        )])
    else:
        if color_overlay is not None:
            overlay_small = cv2.resize(color_overlay, (w // downsample, h // downsample))
            surfacecolor = cv2.cvtColor(overlay_small, cv2.COLOR_BGR2GRAY)
        else:
            surfacecolor = hm
        fig = go.Figure(data=[go.Surface(
            z=hm * z_exaggeration,
            surfacecolor=surfacecolor,
            colorscale="Inferno",
            showscale=False,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.3, roughness=0.6),
            lightposition=dict(x=100, y=200, z=300),
        )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=1, z=0.35),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        scene_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def hillshade_fallback(height_map: np.ndarray, azimuth_deg: float = 315, altitude_deg: float = 45) -> np.ndarray:
    """2D fallback if 3D rendering is ever too slow for the demo machine."""
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    gy, gx = np.gradient(height_map)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = np.clip(shaded, 0, 1)
    return (shaded * 255).astype(np.uint8)
