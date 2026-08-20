#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic mIoU evaluation on real Cityscapes validation set (19 trainIds).

Runs the trained checkpoint over a Cityscapes split, argmaxes the semantic
logits into a trainId map, and accumulates a confusion matrix against the
trainId ground truth from `cityscapes_labelIds_to_trainIds.py`.

Note: no surviving script ran mIoU against real Cityscapes val end-to-end;
this reuses the project's confusion-matrix routine (fast_hist/per_class_iu)
with a new inference loop.

Usage:
    python eval_miou_cityscapes.py \
        --config-file ../../configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml \
        --weights /path/to/model_final.pth \
        --cityscapes-root /path/to/cityscapes \
        --split val --output /path/to/output_dir
"""

import os
import glob
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config

NUM_CLASSES = 19


def fast_hist(true: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (true >= 0) & (true < num_classes)
    return np.bincount(
        num_classes * true[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def per_class_iu(hist: np.ndarray) -> np.ndarray:
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)


def build_predictor(config_file: str, weights: str) -> DefaultPredictor:
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.freeze()
    return DefaultPredictor(cfg)


def find_pairs(cityscapes_root: str, split: str):
    img_paths = sorted(glob.glob(os.path.join(cityscapes_root, "leftImg8bit", split, "*", "*_leftImg8bit.png")))
    pairs = []
    for img_path in img_paths:
        base = os.path.basename(img_path).replace("_leftImg8bit.png", "")
        city = os.path.basename(os.path.dirname(img_path))
        gt_path = os.path.join(cityscapes_root, "gtFine_trainIds", split, city, f"{base}_gtFine_trainIds.png")
        if os.path.isfile(gt_path):
            pairs.append((img_path, gt_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cityscapes-root", required=True, help="dir containing leftImg8bit/ and gtFine_trainIds/ (see cityscapes_labelIds_to_trainIds.py)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    predictor = build_predictor(args.config_file, args.weights)

    pairs = find_pairs(args.cityscapes_root, args.split)
    if not pairs:
        raise RuntimeError(f"No image/GT pairs found under {args.cityscapes_root} (split={args.split})")
    print(f"[MIOU] {len(pairs)} image/GT pairs")

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img_path, gt_path in tqdm(pairs):
        image = np.array(Image.open(img_path).convert("RGB"))
        with torch.no_grad():
            outputs = predictor(image[:, :, ::-1])  # DefaultPredictor expects BGR
        pred = outputs["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.int64)
        gt = np.array(Image.open(gt_path), dtype=np.int64)
        hist += fast_hist(gt, pred, NUM_CLASSES)

    ious = per_class_iu(hist)
    miou = np.nanmean(ious)

    print("\n===== IoU per class =====")
    lines = [f"Class {i:2d}: {iou:.4f}" for i, iou in enumerate(ious)]
    for line in lines:
        print(line)
    lines.append(f"\nmIoU: {miou:.4f}")
    print(lines[-1])

    result_path = os.path.join(args.output, "miou_result.txt")
    with open(result_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
