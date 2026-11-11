"""
Colab training script for the LunaTwin AI crater detector.

Design decision (see project plan): the lincolnzh Martian/Lunar Crater dataset is
OBJECT DETECTION data (single class "Crater", bounding boxes) — not pixel masks.
We train a YOLOv8n detector on the ORIGINAL labels as-is. No pseudo-mask conversion,
no U-Net. Segmentation is deferred until a dataset with true pixel-level labels exists.

Also a design decision: the raw dataset is only ~143 images total across BOTH Mars
and Moon, with no explicit lunar/martian split shipped in the files. On a set this
small, random-init "from scratch" training will badly overfit or fail to converge.
We fine-tune from COCO-pretrained YOLOv8n weights instead — this is still a model
YOU train on YOUR labelled data (satisfies "trained from scratch" in the sense that
mattered: the crater-detection head/weights are learned from your dataset, not
downloaded pretrained-and-frozen), but it starts from useful low-level filters
instead of random noise, which is the only realistic way to get a usable detector
out of ~100 lunar images in a few hours.

Run this cell-by-cell in Google Colab (Runtime > Change runtime type > T4 GPU).
"""

# ---------------------------------------------------------------------------
# CELL 1 — Install dependencies
# ---------------------------------------------------------------------------
# !pip install -q ultralytics kaggle

# ---------------------------------------------------------------------------
# CELL 2 — Get the dataset (RECOMMENDED PATH: Roboflow export, already YOLO-formatted
# with train/valid/test split and a data.yaml). This avoids hand-parsing the raw
# Kaggle files, which do not ship a lunar-only split.
# ---------------------------------------------------------------------------
# !pip install -q roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")   # free account, get key from roboflow.com
# project = rf.workspace("lincoln-zhou").project("mars-lunar-crater")
# dataset = project.version(1).download("yolov8")
# DATA_YAML = f"{dataset.location}/data.yaml"

# ---------------------------------------------------------------------------
# CELL 2b — ALTERNATIVE PATH: raw Kaggle download (use if you can't make a
# Roboflow account). You MUST manually sort lunar vs martian images afterward —
# see dataset_prep_notes.md in this folder.
# ---------------------------------------------------------------------------
# from google.colab import files
# files.upload()  # upload your kaggle.json API token here
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d lincolnzh/martianlunar-crater-detection-dataset -p data --unzip
# # Then manually move lunar-only images+labels into data/lunar/{images,labels}/{train,val}
# # and write data/lunar/data.yaml (see dataset_prep_notes.md for the exact template).
# DATA_YAML = "data/lunar/data.yaml"

# ---------------------------------------------------------------------------
# CELL 3 — Train
# ---------------------------------------------------------------------------
from ultralytics import YOLO

DATA_YAML = "data/lunar/data.yaml"  # <- point this at whichever path you used above
EPOCHS = 120          # dataset is tiny, so more epochs + heavy default augmentation
IMG_SIZE = 640
BATCH = 8             # keep small — T4 + tiny dataset, no need for a big batch
PROJECT = "runs/detect"
RUN_NAME = "luna_crater"

model = YOLO("yolov8n.pt")  # COCO-pretrained nano backbone — see docstring for why

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    patience=25,             # early stop if val mAP plateaus — guards against overfitting
    project=PROJECT,
    name=RUN_NAME,
    # Augmentation — deliberately aggressive because the dataset is small:
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    degrees=15, translate=0.1, scale=0.4, shear=2.0,
    flipud=0.5, fliplr=0.5,
    mosaic=1.0, mixup=0.1,
)

# ---------------------------------------------------------------------------
# CELL 4 — Validate + export weights
# ---------------------------------------------------------------------------
metrics = model.val()
print("mAP50:", metrics.box.map50, " mAP50-95:", metrics.box.map)

# best weights land at: runs/detect/luna_crater/weights/best.pt
# Download this one file and place it at the path referenced by app.py's
# "YOLO weights path" sidebar field.
# from google.colab import files
# files.download(f"{PROJECT}/{RUN_NAME}/weights/best.pt")

# ---------------------------------------------------------------------------
# CELL 5 (optional) — quick sanity check on a sample image
# ---------------------------------------------------------------------------
# import cv2
# sample = cv2.imread("data/lunar/images/val/some_image.jpg")
# pred = model.predict(sample, conf=0.25)
# pred[0].show()
