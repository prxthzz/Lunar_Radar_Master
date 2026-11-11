import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from luna_core.ui_theme import inject_theme, require_result, verdict_badge

st.set_page_config(page_title="Confidence & Landing Zones", page_icon="\U0001F3AF", layout="wide")
inject_theme()
result = require_result()

st.markdown('<div class="page-enter"><h2>Confidence & Landing Zones</h2>'
            '<p class="twin-sub">Where the model is sure, where it isn\'t, and which cells are safest to target.</p></div>',
            unsafe_allow_html=True)
st.write("")

tab1, tab2 = st.tabs(["Confidence Heatmap", "Landing Zone Grid"])

with tab1:
    heat = (result["confidence_map"] * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    blended = cv2.addWeighted(result["enhanced"], 0.45, heat, 0.55, 0)
    st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption("Bright = higher per-pixel model confidence (max softmax probability) — "
               "craters/boulders/rough patches should stand out as spatially distinct regions.")

    st.markdown("#### Raw per-pixel confidence heatmap (unblended)")
    st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), use_container_width=True)

    entropy_map = result.get("confidence_entropy_map")
    if entropy_map is not None:
        st.markdown("#### Entropy-based confidence (1 \u2212 normalized softmax entropy)")
        ent = (np.clip(entropy_map, 0, 1) * 255).astype(np.uint8)
        st.image(cv2.cvtColor(cv2.applyColorMap(ent, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB),
                 use_container_width=True)

with tab2:
    hazard_grid = result["hazard_grid"]
    conf_grid = result["confidence_grid"]

    def cell_verdict(h):
        if h < 0.35:
            return "Safe"
        if h < 0.6:
            return "Caution"
        return "Risky"

    verdict_grid = np.vectorize(cell_verdict)(hazard_grid)
    colorscale = [[0, "#29d38b"], [0.5, "#f5a524"], [1, "#ef4b4b"]]

    fig = go.Figure(data=go.Heatmap(
        z=hazard_grid, colorscale=colorscale, zmin=0, zmax=1,
        text=verdict_grid, texttemplate="%{text}", textfont={"size": 10, "color": "black"},
        showscale=True, colorbar=dict(title="Hazard"),
        hovertemplate="Row %{y}, Col %{x}<br>Hazard: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8ecf5", margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed", showticklabels=False),
        xaxis=dict(showticklabels=False),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    n_safe = int((verdict_grid == "Safe").sum())
    n_caution = int((verdict_grid == "Caution").sum())
    n_risky = int((verdict_grid == "Risky").sum())
    total = verdict_grid.size

    st.write("")
    st.markdown("#### Landing Zone Assessment")
    c1, c2, c3 = st.columns(3)
    c1.metric("Safe cells", f"{n_safe}/{total}")
    c2.metric("Caution cells", f"{n_caution}/{total}")
    c3.metric("Risky cells", f"{n_risky}/{total}")

    best_i, best_j = np.unravel_index(np.argmin(hazard_grid + (1 - conf_grid) * 0.3), hazard_grid.shape)
    st.markdown(
        f"**Overall:** {verdict_badge(result['verdict'])} &nbsp; (score {result['score']}/100)  \n"
        f"**Recommended cell for touchdown targeting:** row {best_i}, col {best_j} "
        f"(lowest hazard combined with confidence).",
        unsafe_allow_html=True,
    )
