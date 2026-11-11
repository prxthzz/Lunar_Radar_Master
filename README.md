# LunaTwin AI / LUNA-SAGE — v3 (multi-class segmentation)

AI-powered lunar landing hazard segmentation + interactive 3D digital twin,
mission-control-style **multi-page** Streamlit app.

**Architecture:**
Lunar Image → Enhancement (CLAHE + Gamma + Denoise) → DeepLabV3+ 4-class
segmentation (smooth_safe / crater / boulder / rough_terrain) → per-pixel
confidence (softmax entropy) → hazard grid fusion → landing score + verdict +
risk breakdown → 3D Digital Twin + Mission Briefing.

Crater labels are real (Kaggle dataset). Boulder/rough/safe labels are
pseudo-labels — see `dataset_prep_notes.md` for exactly why and how.

## Revision history

- v1: classical-CV-only prototype
- v2: YOLO crater detector + classical terrain risk (segmentation deferred —
  dataset had no pixel masks)
- v3 (this version): full multi-class DeepLabV3+ segmentation, combining real
  crater labels with pseudo-labels for the other 3 classes; true multi-page
  Streamlit app instead of tabs

## Project structure

```
luna_twin_ai/
├── Home.py                          # entry point: upload, params, run pipeline
├── pages/
│   ├── 1_Original_and_Enhancement.py
│   ├── 2_Hazard_Detection.py
│   ├── 3_Confidence_and_Landing_Zones.py
│   ├── 4_Digital_Twin.py
│   └── 5_Mission_Briefing.py
├── make_pseudo_masks.py             # builds the 4-class training masks
├── train_segmentation.py            # Colab DeepLabV3+ training script
├── dataset_prep_notes.md
├── requirements.txt
└── luna_core/
    ├── enhancement.py               # CLAHE / gamma / denoise (from v1/v2)
    ├── terrain_risk.py              # texture/edge scoring — used inside pseudo_labels.py
    ├── pseudo_labels.py             # NEW: boulder + rough/smooth pseudo-mask generation
    ├── segmentation_model.py        # NEW: DeepLabV3+ definition + entropy confidence
    ├── fusion.py                    # REWRITTEN for v3: seg map -> hazard grid/score/risk
    ├── mission_briefing.py          # NEW: full text report generator
    ├── terrain3d.py                 # shape-from-shading 3D (from v1/v2, reused as-is)
    ├── ui_theme.py                  # NEW: shared CSS/animations/components for all pages
    └── pipeline.py                  # REWRITTEN: orchestrates the v3 flow
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Upload an image on **Home**, set the segmentation weights path, choose a detection
engine, click **Analyze**. Then browse the sidebar pages — each is a real route
(`/Hazard_Detection`, etc.), not a tab, and each fades in on load. Results are
shared across pages via `st.session_state`, so you only run the pipeline once per
image.

**Detection engines (sidebar):**
- `auto` (default) — uses the trained segmentation model if the weights file loads,
  otherwise transparently falls back to the classical pseudo-label engine.
- `model` — forces the neural model (falls back to classical if weights are missing).
- `classical` — pure feature-based detection (Hough craters + blob boulders +
  texture rough/smooth). Needs no weights and always produces a spatially varying
  per-pixel overlay + confidence map.

A ready-to-run `hazard_unet.pth` (U-Net trained on a synthetic quick-start dataset)
is included so the app demos out of the box. Retrain on real data for production.

## Training

1. Prepare the dataset — see `dataset_prep_notes.md`.
2. Run `make_pseudo_masks.py` for train and val splits (or pass `--no-labels` to
   auto-detect craters with Hough when you have no YOLO labels).
3. Run `train_segmentation.py` in Colab (T4 GPU) or locally on CPU:
   `python train_segmentation.py --arch unet --epochs 40 --out hazard_unet.pth`
   (use `--arch deeplabv3` for the DeepLabV3+ variant). If `data/lunar` is absent
   the script auto-generates a synthetic dataset so it trains immediately.
4. Point Home's sidebar "Segmentation model weights" field at the downloaded file.

## Demo / viva explanation (plain language)

> "LunaTwin AI segments every pixel of a lunar image into one of four classes —
> safe ground, crater, boulder, or rough terrain — using a DeepLabV3+ model. Only
> the crater class has real training labels, from a public annotated dataset; the
> other three classes are trained on pseudo-labels we generate ourselves with
> classical image analysis, because no pixel-level ground truth exists for them
> yet — we say that openly rather than hiding it. The model also reports its own
> confidence per pixel, using the entropy of its output probabilities, so the
> dashboard can show not just what it thinks is hazardous but how sure it is.
> Everything is fused into a landing score, a per-cell landing-zone grid, an
> estimated 3D terrain view, and a full mission-style briefing you could hand to
> a mission planner."

**If asked "why pseudo-labels, isn't that cheating":** the alternative was either
fabricating a story about labels that don't exist, or not building a segmentation
model at all. Pseudo-labeling from image features is a standard, documented
technique (weak supervision) precisely for this situation — and every class's
label provenance is documented in `dataset_prep_notes.md`, not hidden.

**If asked about the LROC boulder / LuSeg datasets:** they were evaluated and
explicitly deferred to future work — they're georeferenced GIS data requiring
coordinate reprojection against multi-gigabyte orbital mosaics, not a same-day
Colab task. Naming that constraint explicitly is a stronger answer than pretending
it wasn't considered.
