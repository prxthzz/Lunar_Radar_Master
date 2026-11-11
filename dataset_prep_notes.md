# Dataset preparation notes — v3 (multi-class segmentation)

## Real labels: craters only

Source: `lincolnzh/martianlunar-crater-detection-dataset` on Kaggle (mirrored at
`roboflow.com/lincoln-zhou/mars-lunar-crater`).

- 143 images total, Mars + Moon combined, **no shipped lunar-only split**
- Single class `Crater`, YOLO bounding boxes

**Recommended path:** Roboflow export (format: YOLOv8) — gives clean `train/valid`
folders. Manually filter to lunar-only images (~20–30 min by eye, don't script this
for 143 images). See the v2 addendum in the plan doc for the full walkthrough.

**Fallback path:** raw Kaggle download via `kaggle datasets download -d
lincolnzh/martianlunar-crater-detection-dataset`.

Either way, end up with:
```
data/lunar/
├── images/{train,val}/   # lunar images only
└── labels/{train,val}/   # matching YOLO .txt files, one per image
```

## Pseudo-labels: boulder, rough_terrain, smooth_safe

No real pixel-level labels exist for these. Real georeferenced options (LROC/Zenodo
boulder GIS datasets, LuSeg/LuSNAR) were evaluated and explicitly deferred — see the
Revision 3 chapter in the plan doc for why. Instead:

1. Run `make_pseudo_masks.py` on the same `images/train` and `images/val` folders —
   it enhances each image, converts the crater boxes to filled-ellipse masks, and
   generates boulder (blob/contour) and rough/smooth (edge density + LBP texture)
   pseudo-labels for everything that isn't a crater:

   ```bash
   python make_pseudo_masks.py --images data/lunar/images/train \
                                --labels data/lunar/labels/train \
                                --out data/lunar/masks/train

   python make_pseudo_masks.py --images data/lunar/images/val \
                                --labels data/lunar/labels/val \
                                --out data/lunar/masks/val
   ```

2. This produces one PNG mask per image (pixel values 0–3 + 255 for "ignore/
   ambiguous"), matching the `images/` folder structure expected by
   `train_segmentation.py`.

## Optional secondary signal (not required)

`LROCNet Moon Classifier` on Zenodo (record 7041842) has patch-level fresh/old/none
crater labels. It's patch classification, not pixel masks, so it can only be used as
weak extra supervision (e.g. pretraining a classifier head) — worth mentioning in a
report as a future enhancement, not required for this prototype to work.

## Class imbalance

Expect `smooth_safe` to dominate most frames. `train_segmentation.py` starts with
manually-set class weights in the loss — inspect a few generated masks first
(`cv2.imshow` or just check the per-class pixel counts) and adjust the weights if
one class is barely represented.
