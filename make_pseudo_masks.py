"""
Run this BEFORE train_segmentation.py.

Reads a YOLO-format crater dataset (images/ + labels/*.txt, one box per line:
`class x_center y_center width height`, all normalized 0-1 as YOLO expects) and
writes a combined 4-class pseudo-mask PNG per image into masks/, using
luna_core.pseudo_labels for boulder/rough/smooth.

Usage:
    python make_pseudo_masks.py --images data/lunar/images/train \
                                 --labels data/lunar/labels/train \
                                 --out data/lunar/masks/train
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from luna_core import enhancement, pseudo_labels


def yolo_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, xc, yc, w, h = parts[:5]
    xc, yc, w, h = float(xc) * img_w, float(yc) * img_h, float(w) * img_w, float(h) * img_h
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def main(images_dir, labels_dir, out_dir, no_labels=False):
    os.makedirs(out_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fname in tqdm(image_files, desc="Building pseudo-masks"):
        img_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"skip (unreadable): {fname}")
            continue

        enhanced = enhancement.enhance_pipeline(img)  # also resizes to 640x640, keep masks aligned to THIS size
        h, w = enhanced.shape[:2]

        if no_labels or labels_dir is None or not os.path.exists(label_path):
            # No YOLO boxes -> fall back to Hough-based crater detection.
            fallback = pseudo_labels.build_pseudo_mask_from_image(enhanced)
            mask = fallback["class_map"]
        else:
            boxes = []
            with open(label_path) as f:
                for line in f:
                    box = yolo_to_xyxy(line, w, h)
                    if box:
                        boxes.append(box)
            mask = pseudo_labels.build_pseudo_mask(enhanced, boxes)

        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")
        cv2.imwrite(out_path, mask)

    print(f"Done. Masks written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None,
                    help="YOLO labels dir (optional; omit or use --no-labels to use Hough crater detection)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-labels", action="store_true",
                    help="Skip YOLO labels and detect craters with Hough circles instead")
    args = ap.parse_args()
    main(args.images, args.labels, args.out, no_labels=args.no_labels)
