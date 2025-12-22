#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WildDash2 (Cityscapes 19-class subset) 평가 스크립트

- 입력:
  * WD2_ROOT: /media/vip-dell/HC/wd_public_v2p0
  * images/: 원본 jpg
  * panoptic/: COCO panoptic RGB png
  * panoptic_cs19_mapped.json: 앞에서 만든 19-class 전용 JSON

- 출력:
  * COCOPanopticEvaluator로 PQ/SQ/RQ (All / Things / Stuff)
  * 우리 커스텀 루틴으로 mIoU (19 trainIds 기준, 0~18)
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import cv2

# 프로젝트 경로
sys.path.append("/home/vip-dell/NoRo_AD")

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    DatasetMapper,
)
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.evaluation import COCOPanopticEvaluator, inference_on_dataset
from detectron2.modeling import build_model

from mask2former import add_maskformer2_config
from panopticapi.utils import rgb2id

# -------------------------
# 1) WildDash2(19-class) 데이터셋 등록
# -------------------------

WD2_ROOT      = "/media/vip-dell/HC/wd_public_v2p0"
WD2_IMG_ROOT  = os.path.join(WD2_ROOT, "images")
WD2_PAN_ROOT  = os.path.join(WD2_ROOT, "panoptic")
WD2_PAN_JSON  = os.path.join(WD2_ROOT, "panoptic_cs19_mapped.json")  # 앞에서 만든 파일
DATASET_NAME  = "wilddash2_panoptic_cs19"

LABEL_DIVISOR = 1000
IGNORE_LABEL  = 255


def _safe_remove(name: str):
    """DatasetCatalog / MetadataCatalog에 기존에 등록된 이름 있으면 제거"""
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


def _register_wd2_city19():
    assert os.path.isfile(WD2_PAN_JSON), f"panoptic_cs19_mapped.json이 없습니다: {WD2_PAN_JSON}"
    assert os.path.isdir(WD2_PAN_ROOT),  f"panoptic 디렉토리가 없습니다: {WD2_PAN_ROOT}"
    assert os.path.isdir(WD2_IMG_ROOT),  f"images 디렉토리가 없습니다: {WD2_IMG_ROOT}"

    print(f"[REG] WD2_PAN_JSON = {WD2_PAN_JSON}")
    print(f"[REG] WD2_PAN_ROOT = {WD2_PAN_ROOT}")
    print(f"[REG] WD2_IMG_ROOT = {WD2_IMG_ROOT}")

    with open(WD2_PAN_JSON, "r", encoding="utf-8") as f:
        j = json.load(f)

    cats = j.get("categories", [])
    print(f"[REG] categories in JSON: {len(cats)} (expect 19)")
    assert len(cats) == 19, "[FATAL] panoptic_cs19_mapped.json categories != 19"

    # -----------------------------
    # ① id → category dict
    # -----------------------------
    id_to_cat = {int(c["id"]): c for c in cats}
    all_ids   = sorted(id_to_cat.keys())
    print(f"[REG] all_ids = {all_ids}")

    # -----------------------------
    # ② 우리가 직접 정의하는 thing / stuff 분할
    #    (Cityscapes 19 trainId 기준)
    # -----------------------------
    #   11~18 : person, rider, car, truck, bus, train, motorcycle, bicycle
    #   나머지(0~10) : 전부 stuff로 취급
    thing_ids = [i for i in [11, 12, 13, 14, 15, 16, 17, 18] if i in all_ids]
    stuff_ids = [i for i in all_ids if i not in thing_ids]

    thing_classes = [id_to_cat[i]["name"] for i in thing_ids]
    stuff_classes = [id_to_cat[i]["name"] for i in stuff_ids]

    # dataset-id -> contiguous-id (그냥 항등 매핑)
    thing_map = {i: i for i in thing_ids}
    stuff_map = {i: i for i in stuff_ids}
    pan_d2c   = {i: i for i in all_ids}

    print(f"[REG] thing_ids = {thing_ids}")
    print(f"[REG] stuff_ids = {stuff_ids}")
    print(f"[REG] thing_classes = {thing_classes}")
    print(f"[REG] stuff_classes = {stuff_classes}")

    # -----------------------------
    # ③ image_id → image / annotation 매핑
    # -----------------------------
    img_map = {im["id"]: im for im in j["images"]}
    ann_map = {an["image_id"]: an for an in j["annotations"]}

    def loader():
        dataset = []
        miss_img, miss_pan = 0, 0

        for img_id, im in img_map.items():
            ann = ann_map.get(img_id, None)
            if ann is None:
                continue

            rel = im.get("file_name", "")
            img_path = None
            if rel:
                cand = [
                    os.path.join(WD2_IMG_ROOT, rel),
                    os.path.join(WD2_IMG_ROOT, os.path.basename(rel)),
                ]
            else:
                base = str(im.get("id", "")) + ".jpg"
                cand = [os.path.join(WD2_IMG_ROOT, base)]

            for c in cand:
                if c and os.path.exists(c):
                    img_path = c
                    break
            if img_path is None:
                miss_img += 1
                continue

            pan_file = ann["file_name"]
            pan_path = None
            pan_cand = [
                os.path.join(WD2_PAN_ROOT, pan_file),
                os.path.join(WD2_PAN_ROOT, os.path.basename(pan_file)),
            ]
            for c in pan_cand:
                if c and os.path.exists(c):
                    pan_path = c
                    break
            if pan_path is None:
                miss_pan += 1
                continue

            segs = ann.get("segments_info", [])
            for s in segs:
                if "iscrowd" not in s:
                    s["iscrowd"] = 0

            H = int(im.get("height", 720))
            W = int(im.get("width", 1280))

            dataset.append({
                "image_id": img_id,
                "file_name": img_path,
                "height": H,
                "width": W,
                "pan_seg_file_name": pan_path,
                "segments_info": segs,
            })

        print(
            f"[REG-LOADER] {DATASET_NAME}: total={len(dataset)}, "
            f"missing_image={miss_img}, missing_panoptic={miss_pan}"
        )
        return dataset

    _safe_remove(DATASET_NAME)
    DatasetCatalog.register(DATASET_NAME, loader)

    meta = MetadataCatalog.get(DATASET_NAME)
    meta.set(
        image_root=WD2_IMG_ROOT,
        panoptic_root=WD2_PAN_ROOT,
        panoptic_json=WD2_PAN_JSON,
        evaluator_type="coco_panoptic_seg",
        ignore_label=IGNORE_LABEL,

        thing_classes=thing_classes,
        stuff_classes=stuff_classes,

        thing_dataset_id_to_contiguous_id=thing_map,
        stuff_dataset_id_to_contiguous_id=stuff_map,
        panoptic_dataset_id_to_contiguous_id=pan_d2c,
        panoptic_contiguous_id_to_dataset_id={v: k for k, v in pan_d2c.items()},
        label_divisor=LABEL_DIVISOR,
        panoptic_label_divisor=LABEL_DIVISOR,
    )

    print(f"[REG] Dataset '{DATASET_NAME}' 등록 완료.")




# -------------------------
# 2) cfg 생성
# -------------------------

def build_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.DATASETS.TEST = (DATASET_NAME,)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19
    cfg.INPUT.FORMAT = "RGB"

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    if not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = args.output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg


# -------------------------
# 3) mIoU 계산 (panoptic → semantic)
# -------------------------

@torch.no_grad()
def compute_mIoU_from_panoptic(model, cfg, dataset_name, num_classes=19, ignore_label=255):
    """
    WD2 19-class subset에 대해, panoptic prediction과 GT panoptic으로 mIoU 계산.

    - GT: rgb2id + GT segments_info.category_id (0~18)로 semantic 복원
    - Pred: panoptic_seg + segments_info.category_id (0~18)로 semantic 추출
    """
    print("[mIoU] Start second pass over dataset for semantic IoU (19-class)...")

    augs = [T.ResizeShortestEdge(
        short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
        max_size=cfg.INPUT.MAX_SIZE_TEST
    )]

    mapper = DatasetMapper(
        cfg,
        is_train=False,
        augmentations=augs,
        image_format=cfg.INPUT.FORMAT,
        use_instance_mask=False,
        use_keypoint=False,
    )

    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    k = num_classes
    conf_mat = np.zeros((k, k), dtype=np.int64)

    for batch in data_loader:
        outputs = model(batch)

        for inp, out in zip(batch, outputs):
            if "panoptic_seg" not in out:
                continue

            # ----- 예측 semantic -----
            pan_pred, segments_info = out["panoptic_seg"]  # pan_pred: (H,W)
            pan_pred = pan_pred.to("cpu").numpy().astype(np.int64)

            sem_pred = np.full_like(pan_pred, fill_value=ignore_label, dtype=np.int64)

            for seg in segments_info:
                seg_id = seg["id"]
                cid    = seg["category_id"]   # 0~18
                if not (0 <= cid < k):
                    continue
                mask = (pan_pred == seg_id)
                if not np.any(mask):
                    continue
                sem_pred[mask] = cid

            # ----- GT semantic -----
            gt_pan_path = inp.get("pan_seg_file_name", None)
            if not gt_pan_path or (not os.path.isfile(gt_pan_path)):
                continue

            gt_rgb = cv2.imread(gt_pan_path, cv2.IMREAD_COLOR)
            if gt_rgb is None:
                continue
            gt_rgb = gt_rgb[:, :, ::-1]  # BGR -> RGB

            gt_id = rgb2id(gt_rgb).astype(np.int64)  # (H_gt, W_gt)
            sem_gt = np.full_like(gt_id, fill_value=ignore_label, dtype=np.int64)

            gt_segs = inp.get("segments_info", [])
            for seg in gt_segs:
                seg_id = seg["id"]
                cid    = seg["category_id"]   # 0~18
                if not (0 <= cid < k):
                    continue
                mask = (gt_id == seg_id)
                if not np.any(mask):
                    continue
                sem_gt[mask] = cid

            # 사이즈 맞추기 (GT -> prediction 크기)
            h_pred, w_pred = sem_pred.shape
            h_gt, w_gt = sem_gt.shape
            if (h_pred, w_pred) != (h_gt, w_gt):
                sem_gt = cv2.resize(
                    sem_gt.astype(np.int32),
                    (w_pred, h_pred),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int64)

            valid = (sem_gt != ignore_label)
            if not np.any(valid):
                continue

            gt_flat   = sem_gt[valid].ravel()
            pred_flat = sem_pred[valid].ravel()

            mask_valid = (gt_flat >= 0) & (gt_flat < k)
            mask_valid &= (pred_flat >= 0) & (pred_flat < k)

            gt_flat   = gt_flat[mask_valid]
            pred_flat = pred_flat[mask_valid]
            if gt_flat.size == 0:
                continue

            ind = k * gt_flat + pred_flat
            binc = np.bincount(ind, minlength=k * k)
            conf_mat += binc.reshape(k, k)

    intersection = np.diag(conf_mat).astype(np.float64)
    gt_sum = conf_mat.sum(axis=1).astype(np.float64)
    pred_sum = conf_mat.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - intersection

    iou_per_class = np.zeros(k, dtype=np.float64)
    valid_classes = union > 0
    iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]
    mIoU = float(iou_per_class[valid_classes].mean()) if np.any(valid_classes) else 0.0

    print("\n================ WD2 mIoU (Cityscapes 19-class subset) ================")
    print(f"mIoU: {mIoU:.4f}")
    print("IoU per class (0~18 trainId):")
    for cid, val in enumerate(iou_per_class):
        print(f"  class {cid:2d}: {val:.4f}")
    print("=====================================================================\n")

    return {
        "mIoU": mIoU,
        "IoU_per_class": iou_per_class.tolist(),
    }


# -------------------------
# 4) 메인: PQ/SQ/RQ + mIoU 평가
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()

    # 1) 데이터셋 등록
    _register_wd2_city19()

    # 2) cfg 구성
    cfg = build_cfg(args)
    default_setup(cfg, args)

    # 3) 모델 생성 + weight 로드
    print("[MODEL] build_model & load weights...")
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # 4) Panoptic PQ/SQ/RQ 평가
    evaluator = COCOPanopticEvaluator(
        DATASET_NAME,
        output_dir=args.output,
    )
    data_loader = build_detection_test_loader(cfg, DATASET_NAME)

    print("[EVAL] start inference_on_dataset (PQ/SQ/RQ)...")
    t0 = time.time()
    with torch.no_grad():
        results = inference_on_dataset(model, data_loader, evaluator)
    dt = time.time() - t0

    print("\n================ WD2 PANOPTIC EVAL (Cityscapes 19-class subset) ================")
    print(f"Time: {dt:.1f} sec")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("===============================================================================\n")

    # 5) mIoU 추가 계산
    compute_mIoU_from_panoptic(model, cfg, DATASET_NAME, num_classes=19, ignore_label=255)


if __name__ == "__main__":
    main()
