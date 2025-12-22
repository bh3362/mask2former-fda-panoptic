#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, itertools
import numpy as np
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog

# [수정] 방금 생성한 'final_dataset' 경로를 정확히 지정합니다.
ROOT = "/media/vip-dell/HC/final_dataset5"

IMAGE_ROOT = f"{ROOT}/leftImg8bit"
PAN_TRAIN_ROOT = f"{ROOT}/panoptic_gt_id/train" # 'panoptic_gt_id' 사용 (id2rgb로 변환된 COCO-RGB)
PAN_VAL_ROOT   = f"{ROOT}/panoptic_gt_id/val"
PAN_TRAIN_JSON = f"{ROOT}/panoptic_json/panoptic_train.json"
PAN_VAL_JSON   = f"{ROOT}/panoptic_json/panoptic_val.json"

# [수정] 데이터셋 이름을 명확하게 변경 (예: carla_final_panoptic_train)
TRAIN_NAME = "carla_final_panoptic_train"
VAL_NAME   = "carla_final_panoptic_val"

LABEL_DIVISOR = 1000
IGNORE_LABEL  = 255

# --- (이하 코드는 원본 Script B와 거의 동일) ---

def _safe_remove(name):
    if hasattr(DatasetCatalog, "remove"):
        try: DatasetCatalog.remove(name)
        except KeyError: pass
    reg = getattr(DatasetCatalog, "_REGISTERED", None) or getattr(DatasetCatalog, "REGISTERED", None)
    if isinstance(reg, dict): reg.pop(name, None)
    meta = getattr(MetadataCatalog, "_NAME_TO_META", None)
    if isinstance(meta, dict): meta.pop(name, None)

def _to_png_candidates(image_root, rel_or_base, split_hint=None):
    # 전처리 스크립트가 JSON에 상대경로를 잘 넣어주므로, 경로 탐색이 더 간단해집니다.
    # (Script B의 원본 로직 유지)
    base = os.path.basename(rel_or_base).replace("\\","/")
    base_noext = os.path.splitext(base)[0]
    if not base_noext.endswith("_leftImg8bit"):
        base_noext += "_leftImg8bit"
    png = base_noext + ".png"
    rel = rel_or_base.replace("\\","/")
    cands = [
        os.path.join(ROOT, rel), # JSON이 ROOT 기준 상대경로 (전처리 스크립트가 이렇게 만듦)
        os.path.join(image_root, rel),
        os.path.join(image_root, split_hint or "", png),
        os.path.join(image_root, "train", png),
        os.path.join(image_root, "val", png),
        os.path.join(image_root, png),
    ]
    out, seen = [], set()
    for c in cands:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out

def _pick_panoptic_abs(pan_root, ann_file_name):
    # 전처리 스크립트가 panoptic_gt_id/train/*.png 경로에 파일을 생성하고
    # JSON의 file_name에도 이 파일명(예: Town01_..._panopticGT.png)을 넣어줍니다.
    pan_abs = os.path.join(pan_root, ann_file_name)
    if os.path.exists(pan_abs):
        return pan_abs
    
    # (원본 Script B의 호환성 로직 유지)
    base_rel = ann_file_name.replace("\\", "/")
    cands = [os.path.join(pan_root, base_rel)]
    if base_rel.endswith("_panopticGT.png"):
        stem = base_rel[:-len("_panopticGT.png")]
        cands.append(os.path.join(pan_root, stem + "_panopticGT_trainIds19.png"))
        cands.append(os.path.join(pan_root, stem + "_trainIds19.png"))
    for p in cands:
        if os.path.exists(p):
            return p
    return None # 찾기 실패

def _make_loader(pjson, image_root, pan_root, ds_name, split_hint):
    def _loader():
        with open(pjson, "r", encoding="utf-8") as f:
            j = json.load(f)
        
        # 전처리 스크립트가 이미 19개 카테고리로 JSON을 생성했는지 확인
        assert len(j.get("categories",[]))==19, f"{ds_name}: categories != 19. 'final_dataset'의 JSON이 맞는지 확인하세요."

        id2img = {im["id"]: im for im in j["images"]}
        imgid2an = {an["image_id"]: an for an in j["annotations"]}

        dataset, miss_img, miss_pan = [], 0, 0
        cache = {}
        exists = lambda p: cache.setdefault(p, os.path.exists(p))

        for iid, info in id2img.items():
            an = imgid2an.get(iid)
            if an is None: continue

            # image
            abs_img = None
            rel = info["file_name"] # 예: leftImg8bit/train/Town01...png
            
            # 1. JSON의 상대경로(rel)는 ROOT(final_dataset) 기준
            direct_path = os.path.join(ROOT, rel)
            if exists(direct_path):
                abs_img = direct_path
            else:
                # 2. 혹시 모르니 Script B의 원래 후보군도 탐색
                for c in _to_png_candidates(image_root, rel, split_hint):
                    if exists(c): abs_img=c; break
            
            if abs_img is None:
                miss_img += 1
                continue

            # panoptic (id2rgb로 변환된 COCO-RGB PNG)
            # an["file_name"] = 예: Town01_..._panopticGT.png
            pan_abs = _pick_panoptic_abs(pan_root, an["file_name"])
            if pan_abs is None:
                miss_pan += 1
                continue

            segs = an.get("segments_info", [])
            for s in segs:
                if "iscrowd" not in s: s["iscrowd"]=0
                # 전처리 스크립트가 category_id를 0-18로 이미 변환했으므로, 여기서 변환 불필요!

            H = int(info.get("height",0)); W=int(info.get("width",0))
            dataset.append({
                "image_id": iid,
                "file_name": abs_img,
                "pan_seg_file_name": pan_abs, # id-encoded RGB PNG
                "segments_info": segs,
                "height": H, "width": W,
            })

        print(f"[custom-loader] {ds_name}: total={len(dataset)}, missing_image={miss_img}, missing_panoptic_gt_id={miss_pan}")
        return dataset
    return _loader

# --- Cityscapes 19 클래스 정보 (Script B와 동일) ---
# (주의: 전처리 스크립트의 CATEGORIES 순서와 정확히 일치해야 함)
thing_classes = ["traffic light","traffic sign","person","rider","car","truck","bus","train","motorcycle","bicycle"]
stuff_classes = ["road","sidewalk","building","wall","fence","pole","vegetation","terrain","sky"]

# !! [중요] !!
# Detectron2는 Cityscapes 기본 파이프라인에서
# 1. Stuff (0-10)을 먼저,
# 2. Thing (11-18)을 나중에 매핑하는 경향이 있습니다.
#    (Script B의 thing/stuff 순서와 다름)
# Cityscapes 표준 trainId (0-18)를 따르는 것이 가장 안전합니다.
thing_classes = [
    "person", "rider", "car", "truck", "bus", 
    "train", "motorcycle", "bicycle"
]
stuff_classes = [
    "road", "sidewalk", "building", "wall", "fence", 
    "pole", "traffic light", "traffic sign", "vegetation", 
    "terrain", "sky"
]

# dataset-id (0-18) → contiguous id (0-18)
# 전처리 스크립트가 이미 0-18을 사용하므로, 여기서는 "항등 매핑(identity mapping)"을 수행합니다.
stuff_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10} # 0-10
thing_map = {11:0, 12:1, 13:2, 14:3, 15:4, 16:5, 17:6, 18:7} # 11-18 -> 0-7

# 하지만 Cityscapes Panoptic 모델은 19개 클래스를 하나의 맵(0-18)으로 처리합니다.
# Script B의 원본 로직 (항등 매핑)이 더 정확할 수 있습니다.
thing_classes_std = ["person","rider","car","truck","bus","train","motorcycle","bicycle"]
stuff_classes_std = ["road","sidewalk","building","wall","fence","pole","traffic light","traffic sign","vegetation","terrain","sky"]

# 전처리 JSON의 'categories' (id: 0-18)와 순서가 같아야 합니다.
# (전처리 스크립트의 CATEGORIES 리스트 기준)
thing_classes_final = ["traffic light", "traffic sign", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
stuff_classes_final = ["road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky"]

# dataset-id (0-18) -> contiguous-id (0-18) (항등 매핑)
# 전처리 스크립트가 0-18 ID를 사용했으므로, 맵도 0-18 -> 0-18 이어야 합니다.
thing_ids_final = [6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
stuff_ids_final = [0, 1, 2, 3, 4, 5, 8, 9, 10]
thing_map_final = {i: i for i in thing_ids_final}
stuff_map_final = {i: i for i in stuff_ids_final}
pan_d2c = {**thing_map_final, **stuff_map_final}

# --- 등록 실행 ---
for n in (TRAIN_NAME, VAL_NAME): _safe_remove(n)

DatasetCatalog.register(TRAIN_NAME, _make_loader(PAN_TRAIN_JSON, IMAGE_ROOT, PAN_TRAIN_ROOT, TRAIN_NAME, "train"))
mt = MetadataCatalog.get(TRAIN_NAME)
mt.set(
    image_root=IMAGE_ROOT,
    panoptic_root=PAN_TRAIN_ROOT,
    panoptic_json=PAN_TRAIN_JSON,
    evaluator_type="coco_panoptic_seg", # Cityscapes도 COCO Panoptic 평가 형식을 따름
    ignore_label=IGNORE_LABEL,
    thing_classes=thing_classes_final,
    stuff_classes=stuff_classes_final,
    thing_dataset_id_to_contiguous_id=thing_map_final,
    stuff_dataset_id_to_contiguous_id=stuff_map_final,
    panoptic_dataset_id_to_contiguous_id=pan_d2c,
    panoptic_contiguous_id_to_dataset_id={v:k for k,v in pan_d2c.items()},
    label_divisor=LABEL_DIVISOR, # Mask2Former가 id/instance 분리 시 사용
    panoptic_label_divisor=LABEL_DIVISOR,
)

DatasetCatalog.register(VAL_NAME, _make_loader(PAN_VAL_JSON, IMAGE_ROOT, PAN_VAL_ROOT, VAL_NAME, "val"))
mv = MetadataCatalog.get(VAL_NAME)
mv.set(
    image_root=IMAGE_ROOT,
    panoptic_root=PAN_VAL_ROOT,
    panoptic_json=PAN_VAL_JSON,
    evaluator_type="coco_panoptic_seg",
    ignore_label=IGNORE_LABEL,
    thing_classes=thing_classes_final,
    stuff_classes=stuff_classes_final,
    thing_dataset_id_to_contiguous_id=thing_map_final,
    stuff_dataset_id_to_contiguous_id=stuff_map_final,
    panoptic_dataset_id_to_contiguous_id=pan_d2c,
    panoptic_contiguous_id_to_dataset_id={v:k for k,v in pan_d2c.items()},
    label_divisor=LABEL_DIVISOR,
    panoptic_label_divisor=LABEL_DIVISOR,
)

print(f"[OK] Detectron2에 데이터셋 등록 완료:")
print(f"  > Train: {TRAIN_NAME}")
print(f"  > Val:   {VAL_NAME}")