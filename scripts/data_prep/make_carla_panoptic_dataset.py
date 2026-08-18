#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the CARLA COCO-panoptic training dataset used in the thesis.

Reads raw per-frame CARLA capture output (RGB + trainId19-or-raw semantic PNG
+ panoptic-ID PNG, one folder per Town x weather scenario — produced by the
CARLA PythonAPI capture loop, not included in this repo, see thesis Sec. III-2
for capture parameters: Town01-03, SUNNY weather, FOV 90, 2048x1024) and packs
it into a COCO panoptic dataset: copies/hardlinks the RGB images, converts
semantic labels to Cityscapes trainId19 (auto-detecting CARLA's old/new tag
sets via `carla_label_mapping.py`), renders panoptic-ID maps to COCO's
id2rgb PNG encoding, and writes the `panoptic_{train,val}.json` COCO panoptic
annotation files Detectron2 expects.

Final split used in the thesis: Town01 + Town02 -> train, Town03 -> val,
SUNNY_GLARE_DAY weather only, every available frame (no random subsampling).

Usage:
    python make_carla_panoptic_dataset.py \
        --in-root /path/to/raw_carla_capture \
        --out-root /path/to/final_dataset \
        --train-towns Town01 Town02 --val-towns Town03 \
        --scenarios SUNNY_GLARE_DAY
"""

import os
import json
import glob
import shutil
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from panopticapi.utils import id2rgb

from carla_label_mapping import (
    THING_TRAINIDS,
    CATEGORIES,
    map_to_train19,
)

IN_SEM_SUFFIX = "_gtFine_trainIds19.png"  # raw semantic label file suffix
IN_PAN_SUFFIX = "_panopticId.png"         # raw panoptic-ID file suffix
ROAD_NEW_PANOPTIC_ID = 26                 # re-mapped panoptic id used for the road ("stuff", segment id 0 is reserved for void)


def safe_link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)  # hardlink when on the same filesystem
    except OSError:
        shutil.copy2(src, dst)


def bbox_from_mask(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]


def list_bases_with_all(in_root, town, scenario):
    left_dir = os.path.join(in_root, town, scenario, "leftImg8bit")
    sem_dir = os.path.join(in_root, town, scenario, "gtFine")
    pano_dir = os.path.join(in_root, town, scenario, "panoptic")

    if not (os.path.isdir(left_dir) and os.path.isdir(sem_dir) and os.path.isdir(pano_dir)):
        print(f"[WARN] dirs missing in {town}/{scenario}")
        return [], left_dir, sem_dir, pano_dir

    left_bases = {os.path.basename(p).replace("_leftImg8bit.png", "")
                  for p in glob.glob(os.path.join(left_dir, "*_leftImg8bit.png"))}
    sem_bases = {os.path.basename(p).replace(IN_SEM_SUFFIX, "")
                 for p in glob.glob(os.path.join(sem_dir, f"*{IN_SEM_SUFFIX}"))}
    pano_bases = {os.path.basename(p).replace(IN_PAN_SUFFIX, "")
                  for p in glob.glob(os.path.join(pano_dir, f"*{IN_PAN_SUFFIX}"))}

    bases = sorted(left_bases & sem_bases & pano_bases)
    if not bases and left_bases:
        print(f"[WARN] no matching GTs for {town}/{scenario}. Check suffixes ({IN_SEM_SUFFIX} / {IN_PAN_SUFFIX}).")
    return bases, left_dir, sem_dir, pano_dir


def build_samples_for_towns(towns, scenarios, index):
    """No subsampling — every available frame for the given town/scenario combos, in order."""
    samples = []
    for scn in scenarios:
        for town in towns:
            for base in index[(town, scn)]["bases"]:
                samples.append((town, scn, base))
    return samples


def write_coco_panoptic(out_root, split_name, samples, index):
    out_img_dir = os.path.join(out_root, "leftImg8bit", split_name)
    out_sem_dir = os.path.join(out_root, "gtFine", split_name)
    out_pano_dir = os.path.join(out_root, "panoptic_gt_id", split_name)
    out_json = os.path.join(out_root, "panoptic_json", f"panoptic_{split_name}.json")

    for d in (out_img_dir, out_sem_dir, out_pano_dir, os.path.dirname(out_json)):
        os.makedirs(d, exist_ok=True)

    images, annotations = [], []

    for (town, scn, base) in tqdm(samples, desc=f"[{split_name}] copy+convert"):
        left_dir = index[(town, scn)]["left_dir"]
        sem_dir = index[(town, scn)]["sem_dir"]
        pano_dir = index[(town, scn)]["pano_dir"]

        rgb_src = os.path.join(left_dir, f"{base}_leftImg8bit.png")
        sem_src = os.path.join(sem_dir, f"{base}{IN_SEM_SUFFIX}")
        pan16_src = os.path.join(pano_dir, f"{base}{IN_PAN_SUFFIX}")

        rgb_name = f"{town}_{scn}_{base}_leftImg8bit.png"
        sem_name = f"{town}_{scn}_{base}_gtFine_trainIds19.png"
        pan_gt_name = f"{town}_{scn}_{base}_panopticGT.png"

        rgb_dst = os.path.join(out_img_dir, rgb_name)
        sem_dst = os.path.join(out_sem_dir, sem_name)
        pan_gt_dst = os.path.join(out_pano_dir, pan_gt_name)

        if not (os.path.exists(rgb_src) and os.path.exists(sem_src) and os.path.exists(pan16_src)):
            print(f"[SKIP] missing files for {town}/{scn}/{base}")
            continue

        safe_link_or_copy(rgb_src, rgb_dst)

        pan16 = cv2.imread(pan16_src, cv2.IMREAD_UNCHANGED)
        sem_orig = cv2.imread(sem_src, cv2.IMREAD_UNCHANGED)
        if pan16 is None or sem_orig is None:
            print(f"[SKIP] failed to read GTs for {pan16_src} or {sem_src}")
            continue
        if pan16.ndim != 2:
            pan16 = pan16[..., 0]

        sem19 = map_to_train19(sem_orig, strategy="auto")
        cv2.imwrite(sem_dst, sem19.astype(np.uint8))

        H, W = pan16.shape[:2]

        # segment id 0 is COCO panoptic "void"; CARLA also encodes the road as id 0,
        # so re-map road -> ROAD_NEW_PANOPTIC_ID to disambiguate it from true void/ignore pixels.
        void_mask = (sem19 == 255)
        road_mask = (pan16 == 0) & (~void_mask)
        pan16_remapped = pan16.copy()
        pan16_remapped[void_mask] = 0
        pan16_remapped[road_mask] = ROAD_NEW_PANOPTIC_ID

        sem19_safe = sem19.copy()
        sem19_safe[void_mask] = 0

        max_sid = int(pan16_remapped.max()) if pan16_remapped.size else 0
        if max_sid >= (1 << 24):
            raise ValueError(f"[FATAL] seg_id >= 2^24 detected: max={max_sid} at {pan16_src}")

        if not os.path.exists(pan_gt_dst):
            rgb = id2rgb(pan16_remapped.astype(np.int64))
            cv2.imwrite(pan_gt_dst, rgb[..., ::-1])

        img_id = f"{town}_{scn}_{base}"
        images.append({
            "id": img_id,
            "file_name": os.path.relpath(rgb_dst, start=out_root).replace("\\", "/"),
            "height": H, "width": W,
        })

        seg_infos = []
        for sid in np.unique(pan16_remapped):
            sid = int(sid)
            if sid == 0:
                continue  # panoptic void

            m = (pan16_remapped == sid)
            area = int(m.sum())
            if area == 0:
                continue

            tid_candidates = sem19_safe[m]
            if tid_candidates.size == 0:
                continue
            tid = int(np.bincount(tid_candidates).argmax())
            if tid == 255:
                tid = 0
            if not (0 <= tid <= 18):
                continue

            if sid == ROAD_NEW_PANOPTIC_ID and tid != 0:
                tid = 0
            elif sid != ROAD_NEW_PANOPTIC_ID and tid == 0:
                continue  # non-road segment collapsed onto tid 0 -> void overlap, drop it

            seg_infos.append({
                "id": sid,
                "category_id": tid,
                "isthing": 1 if tid in THING_TRAINIDS else 0,
                "area": area,
                "bbox": bbox_from_mask(m),
                "iscrowd": 0,
            })

        annotations.append({
            "image_id": img_id,
            "file_name": os.path.relpath(pan_gt_dst, start=out_pano_dir).replace("\\", "/"),
            "segments_info": seg_infos,
        })

    with open(out_json, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES}, f, ensure_ascii=False)
    print(f"[{split_name}] images={len(images)}  json -> {out_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-root", required=True, help="raw CARLA capture root (Town.../scenario/{leftImg8bit,gtFine,panoptic})")
    ap.add_argument("--out-root", required=True, help="output dataset root (COCO panoptic layout)")
    ap.add_argument("--train-towns", nargs="+", default=["Town01", "Town02"])
    ap.add_argument("--val-towns", nargs="+", default=["Town03"])
    ap.add_argument("--scenarios", nargs="+", default=["SUNNY_GLARE_DAY"])
    args = ap.parse_args()

    index = {}
    all_towns = sorted(set(args.train_towns + args.val_towns))
    for town in tqdm(all_towns, desc="Indexing"):
        for scn in args.scenarios:
            bases, left_dir, sem_dir, pano_dir = list_bases_with_all(args.in_root, town, scn)
            index[(town, scn)] = {"bases": bases, "left_dir": left_dir, "sem_dir": sem_dir, "pano_dir": pano_dir}

    train_samples = build_samples_for_towns(args.train_towns, args.scenarios, index)
    val_samples = build_samples_for_towns(args.val_towns, args.scenarios, index)
    print(f"[SUMMARY] train samples: {len(train_samples)}, val samples: {len(val_samples)}")

    write_coco_panoptic(args.out_root, "train", train_samples, index)
    write_coco_panoptic(args.out_root, "val", val_samples, index)
    print("DONE.")


if __name__ == "__main__":
    main()
