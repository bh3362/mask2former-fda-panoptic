#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detectron2 dataset registration for the CARLA panoptic train/val split.

This is the single, parameterized version of what used to be two
near-duplicate blocks: the original `resister.py` (registered the raw,
Non-FDA `leftImg8bit` images) and an inline copy of the same logic baked into
`train_from_scratch_FDA.py` (registered `leftImg8bit_fda` instead). The only
real difference between "baseline" and "FDA" runs is which image subfolder
gets registered as the dataset's `image_root` — everything else (panoptic
JSON/PNG paths, category/thing/stuff id maps) is identical, so both
`train_baseline.py` and `train_fda.py` now call `register_carla_panoptic()`
with a different `image_subdir`.
"""

import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog

LABEL_DIVISOR = 1000
IGNORE_LABEL = 255

THING_CLASSES = ["traffic light", "traffic sign", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
STUFF_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky"]
THING_IDS = [6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
STUFF_IDS = [0, 1, 2, 3, 4, 5, 8, 9, 10]  # 'road' = 0


def _safe_remove(name: str):
    if hasattr(DatasetCatalog, "remove"):
        try:
            DatasetCatalog.remove(name)
        except KeyError:
            pass
    reg = getattr(DatasetCatalog, "_REGISTERED", None) or getattr(DatasetCatalog, "REGISTERED", None)
    if isinstance(reg, dict):
        reg.pop(name, None)
    meta = getattr(MetadataCatalog, "_NAME_TO_META", None)
    if isinstance(meta, dict):
        meta.pop(name, None)


def _to_png_candidates(image_root, rel_or_base, split_hint=None):
    base = os.path.basename(rel_or_base).replace("\\", "/")
    base_noext = os.path.splitext(base)[0]
    if not base_noext.endswith("_leftImg8bit"):
        base_noext += "_leftImg8bit"
    png = base_noext + ".png"
    rel = rel_or_base.replace("\\", "/")
    cands = [
        os.path.join(image_root, rel),
        os.path.join(image_root, split_hint or "", png),
        os.path.join(image_root, "train", png),
        os.path.join(image_root, "val", png),
        os.path.join(image_root, png),
    ]
    out, seen = [], set()
    for c in cands:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _pick_panoptic_abs(pan_root, ann_file_name):
    pan_abs = os.path.join(pan_root, ann_file_name)
    if os.path.exists(pan_abs):
        return pan_abs
    base_rel = ann_file_name.replace("\\", "/")
    cands = [os.path.join(pan_root, base_rel)]
    if base_rel.endswith("_panopticGT.png"):
        stem = base_rel[:-len("_panopticGT.png")]
        cands.append(os.path.join(pan_root, stem + "_panopticGT_trainIds19.png"))
        cands.append(os.path.join(pan_root, stem + "_trainIds19.png"))
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def _make_loader(pjson, image_root, pan_root, ds_name, split_hint):
    def _loader():
        with open(pjson, "r", encoding="utf-8") as f:
            j = json.load(f)
        assert len(j.get("categories", [])) == 19, f"{ds_name}: categories != 19 in {pjson}"
        id2img = {im["id"]: im for im in j["images"]}
        imgid2an = {an["image_id"]: an for an in j["annotations"]}
        dataset, miss_img, miss_pan = [], 0, 0
        cache = {}
        exists = lambda p: cache.setdefault(p, os.path.exists(p))
        for iid, info in id2img.items():
            an = imgid2an.get(iid)
            if an is None:
                continue
            rel = info["file_name"]
            direct_path = os.path.join(image_root, rel)
            abs_img = direct_path if exists(direct_path) else None
            if abs_img is None:
                for c in _to_png_candidates(image_root, rel, split_hint):
                    if exists(c):
                        abs_img = c
                        break
            if abs_img is None:
                miss_img += 1
                continue
            pan_abs = _pick_panoptic_abs(pan_root, an["file_name"])
            if pan_abs is None:
                miss_pan += 1
                continue
            segs = an.get("segments_info", [])
            for s in segs:
                s.setdefault("iscrowd", 0)
            dataset.append({
                "image_id": iid,
                "file_name": abs_img,
                "pan_seg_file_name": pan_abs,
                "segments_info": segs,
                "height": int(info.get("height", 0)),
                "width": int(info.get("width", 0)),
            })
        print(f"[register_datasets] {ds_name}: total={len(dataset)}, missing_image={miss_img}, missing_panoptic_gt_id={miss_pan}")
        return dataset
    return _loader


def register_carla_panoptic(root: str, image_subdir: str, train_name: str, val_name: str):
    """Register `<train_name>`/`<val_name>` from a COCO-panoptic-formatted CARLA dataset root.

    root: dataset root, e.g. ".../final_dataset5" (contains panoptic_json/, panoptic_gt_id/, and `image_subdir`)
    image_subdir: "leftImg8bit" for the raw/Non-FDA images, or "leftImg8bit_fda" for the FDA-styled ones
    """
    image_root = os.path.join(root, image_subdir)
    pan_train_root = os.path.join(root, "panoptic_gt_id", "train")
    pan_val_root = os.path.join(root, "panoptic_gt_id", "val")
    pan_train_json = os.path.join(root, "panoptic_json", "panoptic_train.json")
    pan_val_json = os.path.join(root, "panoptic_json", "panoptic_val.json")

    thing_map = {i: i for i in THING_IDS}
    stuff_map = {i: i for i in STUFF_IDS}
    pan_d2c = {**thing_map, **stuff_map}

    for name, pjson, pan_root, split_hint in (
        (train_name, pan_train_json, pan_train_root, "train"),
        (val_name, pan_val_json, pan_val_root, "val"),
    ):
        _safe_remove(name)
        DatasetCatalog.register(name, _make_loader(pjson, image_root, pan_root, name, split_hint))
        MetadataCatalog.get(name).set(
            image_root=image_root,
            panoptic_root=pan_root,
            panoptic_json=pjson,
            evaluator_type="coco_panoptic_seg",
            ignore_label=IGNORE_LABEL,
            thing_classes=THING_CLASSES,
            stuff_classes=STUFF_CLASSES,
            thing_dataset_id_to_contiguous_id=thing_map,
            stuff_dataset_id_to_contiguous_id=stuff_map,
            panoptic_dataset_id_to_contiguous_id=pan_d2c,
            panoptic_contiguous_id_to_dataset_id={v: k for k, v in pan_d2c.items()},
            label_divisor=LABEL_DIVISOR,
            panoptic_label_divisor=LABEL_DIVISOR,
        )

    print(f"[register_datasets] registered '{train_name}' / '{val_name}' from root={root} (images={image_subdir})")
    return train_name, val_name
