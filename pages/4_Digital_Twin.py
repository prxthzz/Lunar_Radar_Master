import streamlit as st

from luna_core import terrain3d
from luna_core.ui_theme import inject_theme, require_result, hazard_overlay_image

st.set_page_config(page_title="3D Digital Twin", page_icon="\U0001F30D", layout="wide")
inject_theme()
result = require_result()

st.markdown('<div class="page-enter"><h2>3D Digital Twin</h2>'
            '<p class="twin-sub">Estimated terrain, draped with the hazard classification.</p></div>',
            unsafe_allow_html=True)
st.write("")

overlay = hazard_overlay_image(result["enhanced"], result["class_map"], alpha=0.5)
fig = terrain3d.build_surface_figure(
    result["height_map"], color_overlay=overlay, class_map=result["class_map"]
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Surface color = per-pixel hazard class "
           "(green = smooth/safe, red = crater, orange = boulder, yellow = rough terrain).")

st.warning(
    "\u26A0\uFE0F This terrain is **estimated** from shape-from-shading (how light and shadow fall across "
    "the image) \u2014 it is not real elevation/DEM data. Treat it as a visualization aid, not a survey."
)

c1, c2, c3 = st.columns(3)
c1.metric("Landing Score", f"{result['score']}/100")
c2.metric("Verdict", result["verdict"])
c3.metric("Avg. Confidence", f"{result['confidence_pct']}%")
