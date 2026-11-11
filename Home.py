"""
LunaTwin AI / LUNA-SAGE — Home.
Run:  streamlit run Home.py
"""
import time

import cv2
import numpy as np
import streamlit as st

from luna_core import pipeline
from luna_core.ui_theme import inject_theme

st.set_page_config(page_title="LunaTwin AI \u2014 Home", page_icon="\U0001F311", layout="wide")
inject_theme()

with st.sidebar:
    st.markdown('<div class="twin-title">LunaTwin AI</div><div class="twin-sub">LUNA-SAGE mission console</div>', unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader("Upload a lunar image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    weights_path = st.text_input("Segmentation model weights (.pt)", value="hazard_unet.pth")
    engine = st.selectbox(
        "Detection engine",
        ["auto", "model", "classical"],
        index=0,
        help="auto = use the trained model if weights load, else the classical pseudo-label engine. "
             "model = force the trained model (falls back if weights missing). "
             "classical = pure feature-based detection, no weights needed.",
    )
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    with st.expander("Pipeline parameters"):
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 6.0, 3.0, 0.5)
        gamma = st.slider("Gamma (lower = brighter shadows)", 0.3, 1.0, 0.6, 0.05)
    run_btn = st.button("\U0001F680 Analyze", use_container_width=True)
    with st.expander("About LunaTwin AI"):
        st.caption(
            "DeepLabV3+ / U-Net segments every pixel into smooth/safe, crater, boulder, "
            "or rough terrain. Crater labels come from a real annotated dataset; "
            "boulder/rough/safe labels are pseudo-labels (blob detection + texture "
            "analysis) since no pixel-level ground truth exists for them yet. "
            "Per-pixel confidence = max softmax probability. If no trained weights are "
            "available the app transparently falls back to the classical pseudo-label "
            "engine. The 3D view is an estimated shape-from-shading terrain, not real "
            "elevation data."
        )

st.markdown(
    '<div class="page-enter"><h1 class="twin-title">Mission Console</h1>'
    '<p class="twin-sub">Upload a lunar image, then Analyze. Results appear on every page in the sidebar nav.</p></div>',
    unsafe_allow_html=True,
)
st.write("")

if "result" not in st.session_state:
    st.session_state.result = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_info" not in st.session_state:
    st.session_state.model_info = {"loaded": False, "arch": None, "path": None, "message": ""}
if "weights_path" not in st.session_state:
    st.session_state.weights_path = None

if run_btn:
    if uploaded is None:
        st.warning("Upload an image first.")
    else:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        progress_msgs = [
            "Enhancing image\u2026", "Running hazard segmentation\u2026",
            "Estimating per-pixel confidence\u2026", "Fusing hazard grid\u2026",
            "Building digital twin\u2026", "Writing mission briefing\u2026",
        ]
        bar = st.progress(0, text=progress_msgs[0])
        for i, msg in enumerate(progress_msgs):
            bar.progress(int((i + 1) / len(progress_msgs) * 100), text=msg)
            time.sleep(0.12)

        if st.session_state.model is None or st.session_state.weights_path != weights_path:
            try:
                loaded = pipeline.load_model(weights_path, device=device)
                if isinstance(loaded, tuple):
                    st.session_state.model, st.session_state.model_info = loaded
                else:
                    st.session_state.model = loaded
                    st.session_state.model_info = {"loaded": False, "arch": None, "path": None, "message": "loaded fallback"}
            except Exception as exc:
                st.session_state.model = None
                st.session_state.model_info = {"loaded": False, "arch": None, "path": None, "message": str(exc)}
                st.warning(f"Could not load weights from '{weights_path}'. Falling back to the pseudo-label pipeline. ({exc})")
            st.session_state.weights_path = weights_path

        info = st.session_state.model_info or {}
        if not info.get("loaded"):
            st.warning(
                f"**{info.get('message', 'Model weights unavailable')}.** "
                "Using the **classical pseudo-label engine** instead (Hough craters + blob boulders + "
                "texture rough/smooth) so the overlays are still per-pixel and spatially varying. "
                "Train `hazard_unet.pth` (see `train_segmentation.py`) to use the neural segmentation model."
            )
        else:
            st.info(info.get("message", "Model loaded."))

        result = pipeline.run_pipeline(
            image_bgr,
            st.session_state.model,
            clahe_clip,
            gamma,
            device=device,
            engine=engine,
            model_info=st.session_state.model_info,
        )
        st.session_state.result = result
        bar.empty()
        st.success(f"Analysis complete (engine: **{result['engine_used']}**) — open any page in the sidebar to view results.")

result = st.session_state.result
if result:
    from luna_core.ui_theme import render_metric_row
    st.write("")
    render_metric_row(result)
    st.write("")
    st.image(cv2.cvtColor(result["enhanced"], cv2.COLOR_BGR2RGB), caption="Enhanced input", use_container_width=True)
else:
    st.info("No analysis yet. Upload an image in the sidebar and click **Analyze**.")
