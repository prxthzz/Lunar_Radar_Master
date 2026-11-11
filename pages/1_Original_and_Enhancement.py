import cv2
import streamlit as st

from luna_core.ui_theme import inject_theme, require_result

st.set_page_config(page_title="Original & Enhancement", page_icon="\U0001F5BC\uFE0F", layout="wide")
inject_theme()
result = require_result()

st.markdown('<div class="page-enter"><h2>Original & Enhancement</h2>'
            '<p class="twin-sub">Side-by-side comparison and what the enhancement step does.</p></div>',
            unsafe_allow_html=True)
st.write("")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Original**")
    st.image(cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.markdown("**Enhanced**")
    st.image(cv2.cvtColor(result["enhanced"], cv2.COLOR_BGR2RGB), use_container_width=True)

st.write("")
with st.expander("How enhancement works", expanded=True):
    st.markdown("""
- **Denoising** — light Gaussian smoothing removes sensor noise without destroying crater rims.
- **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) — pulls out detail hidden in shadowed
  crater interiors by boosting *local* contrast rather than a single global adjustment.
- **Gamma correction** — non-linearly brightens dark regions while leaving already-bright terrain
  largely untouched.
- **Sharpening** — an unsharp mask pass makes crater rims and rock edges crisper for the segmentation
  model to key off.
    """)
