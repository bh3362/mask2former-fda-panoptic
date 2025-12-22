#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA panoptic/instance GT converter
- 입력 구조 (예):
  <base>/TownXX/<SCENARIO>/leftImg8bit/frame_xxx_leftImg8bit.png
                         /gtFine      /frame_xxx_gtFine_labelIds.png
                         /panoptic    /frame_xxx_panopticId.png (uint16)

- 출력:
  <out>/panoptic_json/panoptic_{split}.json
  <out>/panoptic_pngs/TownXX/SCENARIO/frame_xxx_panoptic.png   (COCO 포맷 RGB encoding)
  <out>/instances_json/instances_{split}.json   (pycocotools 있으면 생성)

카테고리: Cityscapes-19(trainIds) 기준. 'thing'은 {person,rider,car,truck,bus,train,motorcycle,bicycle}.
"""
import os, glob, json, argparse, math
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
import cv2

# optional: pycocotools (없으면 인스턴스 JSON은 skip)
try:
    from pycocotools import mask as cocomask
    HAS_COCO = True
except Exception:
    HAS_COCO = False

IGNORE = 255

# ---------------- Cityscapes-19 ----------------
CLS19 = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
PALETTE19_RGB = np.array([
    [128,  64, 128],[244,  35, 232],[ 70,  70,  70],[102, 102, 156],[190, 153, 153],
    [153, 153, 153],[250, 170,  30],[220, 220,   0],[107, 142,  35],[152, 251, 152],
    [ 70, 130, 180],[220,  20,  60],[255,   0,   0],[  0,   0, 142],[  0,   0,  70],
    [  0,  60, 100],[  0,  80, 100],[  0,   0, 230],[119,  11,  32],
], dtype=np.uint8)
# Cityscapes "thing" 집합(인스턴스 GT로 뽑을 클래스들)
THING_TRAINIDS = {11,12,13,14,15,16,17,18}  # person..bicycle

# Cityscapes labelIds -> trainIds
LABELIDS_TO_TRAINIDS_BASE = {
     7:0,  8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
    23:10, 24:11,25:12,26:13,27:14,28:15,31:16,32:17,33:18
}

# CARLA NEW enum (너가 준 CityObjectLabel: 1..28)
CARLA_IDS_TO_TRAINIDS_NEW = {
     1:0, 24:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8,
    10:9, 11:10, 12:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18
}
# CARLA OLD fallback
CARLA_IDS_TO_TRAINIDS_OLD = {
     7:0, 6:0, 8:1, 1:2, 11:3, 2:4, 5:5, 18:6, 12:7, 9:8, 22:9, 13:10, 4:11, 10:13
}

def detect_gt_kind(uvals: Set[int]) -> str:
    if uvals <= (set(range(19)) | {IGNORE}): return "trainIds"
    if max(uvals) > 28: return "city_labelIds"
    if {24,25,26,27,28} & uvals: return "carla_new"
    if {1,2,3} & uvals: return "carla_new"
    if 10 in uvals: return "carla_old"
    return "carla_new"

def _apply_table(arr: np.ndarray, table: dict,
                 absorb_ground: Optional[str]=None, ground_id: Optional[int]=None) -> np.ndarray:
    out = np.full_like(arr, IGNORE, dtype=np.uint16)
    for k, v in table.items():
        out[arr==k] = v
    if absorb_ground in ("road","terrain") and ground_id is not None:
        tgt = 0 if absorb_ground=="road" else 9
        out[arr==ground_id] = tgt
    return out

def map_to_train19(gt: np.ndarray, absorb_roadlines: bool=True, absorb_ground: Optional[str]=None) -> np.ndarray:
    u = set(np.unique(gt).tolist())
    kind = detect_gt_kind(u)
    if kind == "trainIds": return gt.astype(np.uint16)
    if kind == "city_labelIds":
        return _apply_table(gt, LABELIDS_TO_TRAINIDS_BASE)
    if kind == "carla_new":
        tb = dict(CARLA_IDS_TO_TRAINIDS_NEW)
        if not absorb_roadlines: tb.pop(24, None)
        return _apply_table(gt, tb, absorb_ground=absorb_ground, ground_id=25)
    tb = dict(CARLA_IDS_TO_TRAINIDS_OLD)
    if not absorb_roadlines: tb.pop(6, None)
    return _apply_table(gt, tb, absorb_ground=absorb_ground, ground_id=14)

def id2rgb(seg_id: int) -> Tuple[int,int,int]:
    r = seg_id % 256
    g = (seg_id // 256) % 256
    b = (seg_id // 65536) % 256
    return (r, g, b)

def mask_to_rle(mask: np.ndarray) -> dict:
    # mask: HxW, {0,1}
    if not HAS_COCO:
        return {}
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if len(xs)==0: return [0,0,0,0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1-x0+1, y1-y0+1]

def build_categories_panoptic() -> List[dict]:
    cats = []
    for cid, name in enumerate(CLS19):
        cats.append({
            "id": cid,
            "name": name,
            "isthing": 1 if cid in THING_TRAINIDS else 0,
            "color": PALETTE19_RGB[cid].tolist()
        })
    return cats

def parse_args():
    ap = argparse.ArgumentParser("CARLA → COCO Panoptic/Instance converter")
    ap.add_argument("--base", required=True, help="CARLA root. e.g. /media/vip-dell/HC/_output3")
    ap.add_argument("--out",  default="", help="output root (default: <base>/_coco)")
    ap.add_argument("--split", default="val", help="split name in output filenames")
    ap.add_argument("--towns", default="", help="Town filter, comma-separated. e.g. Town01,Town02")
    ap.add_argument("--scenarios", default="", help="Scenario filter, comma-separated.")
    ap.add_argument("--first_n_per_group", type=int, default=0, help="first N per (Town,Scenario). 0=all")
    ap.add_argument("--min_area", type=int, default=32, help="drop tiny segments (<pixels)")
    ap.add_argument("--absorb_roadlines", action="store_true", help="map RoadLines to road")
    ap.add_argument("--absorb_ground", choices=["none","road","terrain"], default="none")
    return ap.parse_args()

def main():
    args = parse_args()
    base = args.base
    out_root = args.out or os.path.join(base, "_coco")
    pan_json_dir = os.path.join(out_root, "panoptic_json")
    pan_png_root = os.path.join(out_root, "panoptic_pngs")
    ins_json_dir = os.path.join(out_root, "instances_json")
    os.makedirs(pan_json_dir, exist_ok=True)
    os.makedirs(pan_png_root, exist_ok=True)
    if HAS_COCO: os.makedirs(ins_json_dir, exist_ok=True)

    absorb_ground = None if args.absorb_ground=="none" else args.absorb_ground
    absorb_roadlines = True if args.absorb_roadlines else False

    # 이미지 리스트 수집
    patt = os.path.join(base, "Town*", "*", "leftImg8bit", "frame_*_leftImg8bit.png")
    img_list = sorted(glob.glob(patt))
    # 필터
    if args.towns.strip():
        allow_t = set(x.strip() for x in args.towns.split(",") if x.strip())
        img_list = [p for p in img_list if os.path.relpath(p, base).split(os.sep)[0] in allow_t]
    if args.scenarios.strip():
        allow_s = set(x.strip() for x in args.scenarios.split(",") if x.strip())
        img_list = [p for p in img_list if os.path.relpath(p, base).split(os.sep)[1] in allow_s]

    # 그룹별 first-N
    if args.first_n_per_group > 0:
        buckets = defaultdict(list)
        for p in img_list:
            rel = os.path.relpath(p, base).split(os.sep)  # [Town, Scenario, leftImg8bit, fname]
            if len(rel) < 4: continue
            key = (rel[0], rel[1])
            buckets[key].append(p)
        img_list = []
        for key, items in buckets.items():
            img_list.extend(sorted(items)[:args.first_n_per_group])

    # JSON 뼈대
    images = []
    pan_annotations = []
    ins_annotations = []
    categories = build_categories_panoptic()
    cat_ids_set = {c["id"] for c in categories}

    ann_id_counter = 1  # instance ann id
    img_id_counter = 1  # image id

    for ip in img_list:
        rel = os.path.relpath(ip, base).split(os.sep)
        if len(rel) < 4: continue
        town, scen = rel[0], rel[1]
        fn_core = os.path.basename(ip).replace("_leftImg8bit.png", "")

        gt_path  = os.path.join(base, town, scen, "gtFine",   f"{fn_core}_gtFine_labelIds.png")
        pan_path = os.path.join(base, town, scen, "panoptic", f"{fn_core}_panopticId.png")
        if not (os.path.exists(gt_path) and os.path.exists(pan_path)):
            continue

        rgb = cv2.imread(ip)
        if rgb is None: continue
        h, w = rgb.shape[:2]

        # 등록
        image_rec = {
            "id": img_id_counter,
            "file_name": os.path.join(town, scen, "leftImg8bit", f"{fn_core}_leftImg8bit.png").replace("\\","/"),
            "height": h, "width": w
        }
        images.append(image_rec)

        # GT 로드 & 매핑
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)  # uint16
        pan = cv2.imread(pan_path, cv2.IMREAD_UNCHANGED)  # uint16 segment id map
        if gt is None or pan is None: 
            img_id_counter += 1
            continue
        gt19 = map_to_train19(gt, absorb_roadlines=absorb_roadlines, absorb_ground=absorb_ground)
        gt19[(gt19!=IGNORE) & ((gt19<0)|(gt19>=19))] = IGNORE

        # panoptic PNG 생성
        pan_png = np.zeros((h, w, 3), np.uint8)
        seg_infos = []
        used_seg_ids = set(int(x) for x in np.unique(pan).tolist())

        for sid in sorted(used_seg_ids):
            mask = (pan == sid)
            area = int(mask.sum())
            if area < args.min_area:
                continue
            # 세그먼트의 주 클래스(majority vote in trainIds19)
            vals, cnts = np.unique(gt19[mask], return_counts=True)
            # IGNORE(255)만 있으면 스킵
            vals = [int(v) for v in vals]
            cnts = [int(c) for c in cnts]
            # 유효 클래스만 남기기
            keep = [(v,c) for v,c in zip(vals,cnts) if (v in cat_ids_set)]
            if not keep:
                continue
            keep.sort(key=lambda x: x[1], reverse=True)
            cid = keep[0][0]  # category id (0..18)

            # panoptic PNG에 색 넣기
            r,g,b = id2rgb(int(sid))
            pan_png[mask] = (r,g,b)

            seg_infos.append({
                "id": int(sid),
                "category_id": int(cid),
                "iscrowd": 0,
                "area": area,
            })

            # 인스턴스 JSON: thing만
            if HAS_COCO and (cid in THING_TRAINIDS):
                rle = mask_to_rle(mask)
                bbox = bbox_from_mask(mask)
                ins_annotations.append({
                    "id": ann_id_counter,
                    "image_id": image_rec["id"],
                    "category_id": int(cid),
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                    "segmentation": rle,
                })
                ann_id_counter += 1

        # panoptic png 저장 (타운/시나리오 하위로)
        out_pan_dir = os.path.join(pan_png_root, town, scen)
        os.makedirs(out_pan_dir, exist_ok=True)
        pan_png_name = f"{fn_core}_panoptic.png"
        cv2.imwrite(os.path.join(out_pan_dir, pan_png_name), pan_png)

        pan_annotations.append({
            "image_id": image_rec["id"],
            "file_name": os.path.join(town, scen, pan_png_name).replace("\\","/"),
            "segments_info": seg_infos
        })

        img_id_counter += 1

    # Panoptic JSON 저장
    pan_json = {
        "images": images,
        "annotations": pan_annotations,
        "categories": categories
    }
    pan_json_path = os.path.join(pan_json_dir, f"panoptic_{args.split}.json")
    with open(pan_json_path, "w") as f:
        json.dump(pan_json, f)

    # Instance JSON 저장 (있을 때만)
    if HAS_COCO:
        thing_cats = [c for c in categories if c["id"] in THING_TRAINIDS]
        ins_json = {
            "images": images,
            "annotations": ins_annotations,
            "categories": thing_cats
        }
        ins_json_path = os.path.join(ins_json_dir, f"instances_{args.split}.json")
        with open(ins_json_path, "w") as f:
            json.dump(ins_json, f)

    print("[DONE]")
    print(f" Panoptic JSON : {pan_json_path}")
    print(f" Panoptic PNGs : {pan_png_root}")
    if HAS_COCO:
        print(f" Instance JSON : {ins_json_path}")
    else:
        print(" Instance JSON : (pycocotools 미설치 → 생략됨)")

if __name__ == "__main__":
    main()
