import cv2
import streamlit as st

from luna_core.ui_theme import (
    class_legend,
    colorize_class_map,
    hazard_overlay_image,
    inject_theme,
    level_span,
    require_result,
)

st.set_page_config(page_title="Hazard Detection", page_icon="\u26A0\uFE0F", layout="wide")
inject_theme()
result = require_result()

st.markdown('<div class="page-enter"><h2>Hazard Detection & Explanation</h2>'
            '<p class="twin-sub">Multi-class hazard overlay, area breakdown, and risk levels.</p></div>',
            unsafe_allow_html=True)
st.write("")

class_legend()
st.write("")

st.markdown("#### Debug: raw predicted mask")
raw_mask = colorize_class_map(result["class_map"])
st.image(cv2.cvtColor(raw_mask, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Raw class map before any UI blending")
st.caption(f"Engine used: **{result.get('engine_used', 'n/a')}**")

craters = result.get("craters")
if craters is not None and len(craters):
    debug_img = result["enhanced"].copy()
    for (cx, cy, r) in craters:
        cv2.circle(debug_img, (int(cx), int(cy)), int(r), (60, 60, 239), 2)
    st.markdown("#### Debug: classical crater detections (Hough)")
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True,
             caption=f"{len(craters)} circular craters detected")

class_distribution = result.get("class_distribution", {})
if class_distribution:
    st.markdown("#### Class distribution")
    cols = st.columns(4)
    for idx, col in enumerate(cols):
        name = list(class_distribution.keys())[idx]
        with col:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{class_distribution[name]['percent']:.1f}%",
                delta=f"{class_distribution[name]['count']} px",
            )
    st.caption("If smooth/safe dominates the distribution, the model is likely predicting safe terrain too often.")

st.write("")
overlay = hazard_overlay_image(result["enhanced"], result["class_map"])
st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

st.write("")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("#### Hazard Analysis")
    stats = result["class_stats"]
    for name, s in stats.items():
        label = name.replace("_", " ").title()
        st.markdown(f"**{label}** \u2014 {s['area_pct']}% of frame")
        st.progress(min(s["area_pct"] / 100, 1.0))

with col2:
    st.markdown("#### Risk Breakdown")
    risk = result["risk_breakdown"]
    st.markdown(f"**Crater risk:** {level_span(risk['crater_risk']['level'])}  \n"
                f"<span class='twin-sub'>{risk['crater_risk']['reason']}</span>", unsafe_allow_html=True)
    st.write("")
    st.markdown(f"**Boulder risk:** {level_span(risk['boulder_risk']['level'])}  \n"
                f"<span class='twin-sub'>{risk['boulder_risk']['reason']}</span>", unsafe_allow_html=True)
    st.write("")
    st.markdown(f"**Roughness risk:** {level_span(risk['roughness_risk']['level'])}  \n"
                f"<span class='twin-sub'>{risk['roughness_risk']['reason']}</span>", unsafe_allow_html=True)

st.write("")
st.markdown("#### Explanation")
for line in result["explanations"]:
    st.markdown(f'<div class="explain-line">{line}</div>', unsafe_allow_html=True)
