# === add at the end of resister.py (or right after you register datasets) ===
import json, os
from detectron2.data import MetadataCatalog

# 경로
DATASET_ROOT = "/media/vip-dell/HC/train_data_3k_2"
PAN_JSON_TRAIN = f"{DATASET_ROOT}/panoptic_json/panoptic_train.json"
PAN_JSON_VAL   = f"{DATASET_ROOT}/panoptic_json/panoptic_val.json"
PAN_ROOT_TRAIN = f"{DATASET_ROOT}/panoptic_gt_id/train"
PAN_ROOT_VAL   = f"{DATASET_ROOT}/panoptic_gt_id/val"

TRAIN_NAME = "carla_panoptic_train_png_v4"
VAL_NAME   = "carla_panoptic_val_png_v4"

def _build_meta(json_path, pan_root):
    with open(json_path, "r") as f:
        j = json.load(f)
    cats = j["categories"]

    # isthing 플래그 기준으로 things/stuff 분리 (없으면 thing=0으로 간주)
    thing_ids, stuff_ids = [], []
    thing_classes, stuff_classes = [], []
    for c in cats:
        if c.get("isthing", 0) == 1:
            thing_ids.append(c["id"]); thing_classes.append(c["name"])
        else:
            stuff_ids.append(c["id"]); stuff_classes.append(c["name"])

    # id → 0..K-1 연속 매핑
    thing_ids_sorted = sorted(thing_ids)
    stuff_ids_sorted = sorted(stuff_ids)
    thing_map = {did: i for i, did in enumerate(thing_ids_sorted)}
    stuff_map = {did: i for i, did in enumerate(stuff_ids_sorted)}

    meta = {
        "thing_dataset_id_to_contiguous_id": thing_map,
        "stuff_dataset_id_to_contiguous_id": stuff_map,
        "thing_classes": thing_classes,
        "stuff_classes": stuff_classes,
        "ignore_label": 255,
        "evaluator_type": "coco_panoptic",   # panoptic evaluator 사용
        "panoptic_root": pan_root,
        "panoptic_json": json_path,
    }
    # 전체 semantic class 개수(thing+stuff)
    num_semantic = len(thing_ids_sorted) + len(stuff_ids_sorted)
    return meta, num_semantic

# train/val 메타데이터 세팅
train_meta, train_num_classes = _build_meta(PAN_JSON_TRAIN, PAN_ROOT_TRAIN)
val_meta,   val_num_classes   = _build_meta(PAN_JSON_VAL,   PAN_ROOT_VAL)

# 등록된 메타데이터에 주입
MetadataCatalog.get(TRAIN_NAME).set(**train_meta)
MetadataCatalog.get(VAL_NAME).set(**val_meta)

print(f"[resister] TRAIN classes={train_num_classes}, VAL classes={val_num_classes}")
