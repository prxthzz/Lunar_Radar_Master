"""
Shared theme, CSS animations, and small UI helpers used by every page.
Import and call inject_theme() at the top of every page script.
"""
import cv2
import numpy as np
import streamlit as st

CLASS_COLORS_BGR = {
    0: (60, 200, 90),    # smooth_safe - green
    1: (60, 60, 239),    # crater - red
    2: (30, 140, 245),   # boulder - orange
    3: (0, 210, 245),    # rough_terrain - yellow
}
CLASS_COLORS_HEX = {0: "#3ac85a", 1: "#ef4b4b", 2: "#f5a524", 3: "#f5d524"}
CLASS_LABELS = {0: "Smooth / Safe", 1: "Crater", 2: "Boulder", 3: "Rough Terrain"}


def inject_theme():
    st.markdown("""
    <style>
    :root {
        --bg: #05070d; --panel: #0d1220;
        --safe: #29d38b; --caution: #f5a524; --risky: #ef4b4b;
        --accent: #3b82f6; --text: #e8ecf5;
    }
    .stApp { background: radial-gradient(circle at 20% 0%, #0d1730 0%, #05070d 60%); color: var(--text); }
    section[data-testid="stSidebar"] { background: #080b16; border-right: 1px solid #1a2036; }

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(16px);} to { opacity: 1; transform: translateY(0);} }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .page-enter { animation: fadeInUp .5s ease-out both; }
    .fade-in { animation: fadeIn .6s ease-out both; }

    .metric-card {
        background: linear-gradient(145deg, #0f1730, #0a0f20);
        border: 1px solid #1e2946; border-radius: 14px; padding: 18px 20px; text-align: center;
        box-shadow: 0 0 22px rgba(59,130,246,0.06); animation: fadeInUp .5s ease-out both;
    }
    .metric-label { font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; color: #8b93ad; margin-bottom: 6px; }
    .metric-value { font-size: 34px; font-weight: 700; }
    .badge { display:inline-block; padding: 4px 14px; border-radius: 999px; font-weight: 600; font-size: 14px; }
    .badge-safe { background: rgba(41,211,139,.15); color: var(--safe); border: 1px solid var(--safe); }
    .badge-caution { background: rgba(245,165,36,.15); color: var(--caution); border: 1px solid var(--caution); }
    .badge-risky { background: rgba(239,75,75,.15); color: var(--risky); border: 1px solid var(--risky); }
    .level-high { color: var(--risky); font-weight: 700; }
    .level-medium { color: var(--caution); font-weight: 700; }
    .level-low { color: var(--safe); font-weight: 700; }

    .twin-title { font-size: 30px; font-weight: 800; letter-spacing: .5px;
        background: linear-gradient(90deg, #7db8ff, #a78bfa); -webkit-background-clip: text; color: transparent; }
    .twin-sub { color: #8b93ad; font-size: 14px; margin-top: -8px; }

    .explain-line { padding: 10px 14px; border-left: 3px solid var(--accent); background: #0c1226;
        border-radius: 6px; margin-bottom: 8px; animation: fadeInUp .4s ease-out both; }
    .legend-chip { display:inline-flex; align-items:center; gap:6px; margin-right:16px; font-size: 13px; }
    .legend-swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
    .briefing-box { background: #0c1226; border: 1px solid #1e2946; border-radius: 10px; padding: 22px 26px;
        white-space: pre-wrap; line-height: 1.65; font-size: 15px; animation: fadeInUp .5s ease-out both; }
    </style>
    """, unsafe_allow_html=True)


def count_up_html(target_value, suffix="", color="var(--accent)", elem_id="v1", duration_ms=900):
    display_value = (
        f"{float(target_value):.1f}"
        if isinstance(target_value, (float, np.floating))
        else str(target_value)
    )
    return f"""
    <div class="metric-value" id="{elem_id}" style="color:{color}">{display_value}{suffix}</div>
    """


def verdict_badge(verdict):
    cls = {"Safe": "badge-safe", "Caution": "badge-caution", "Risky": "badge-risky"}[verdict]
    return f'<span class="badge {cls}">{verdict}</span>'


def level_span(level):
    return f'<span class="level-{level.lower()}">{level}</span>'


def render_metric_row(result):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Landing Score</div>'
                     f'{count_up_html(result["score"], "", "var(--accent)", "score")}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Verdict</div>'
                     f'<div style="margin-top:6px;">{verdict_badge(result["verdict"])}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div>'
                     f'{count_up_html(result["confidence_pct"], "%", "#a78bfa", "conf")}</div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Hazards Detected</div>'
                     f'{count_up_html(result["hazards_count"], "", "var(--risky)", "haz")}</div>', unsafe_allow_html=True)


def class_legend():
    chips = "".join(
        f'<span class="legend-chip"><span class="legend-swatch" style="background:{CLASS_COLORS_HEX[c]}"></span>{CLASS_LABELS[c]}</span>'
        for c in range(4)
    )
    st.markdown(f'<div class="fade-in">{chips}</div>', unsafe_allow_html=True)


def colorize_class_map(class_map: np.ndarray) -> np.ndarray:
    h, w = class_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_BGR.items():
        out[class_map == cls_id] = color
    return out


def hazard_overlay_image(enhanced_bgr, class_map, alpha=0.45):
    colored = colorize_class_map(class_map)
    return cv2.addWeighted(enhanced_bgr, 1 - alpha, colored, alpha, 0)


def require_result():
    """Guard used at the top of every results page."""
    if "result" not in st.session_state or st.session_state.result is None:
        st.info("No analysis yet \u2014 go to **Home**, upload a lunar image, and click Analyze.")
        st.stop()
    return st.session_state.result
