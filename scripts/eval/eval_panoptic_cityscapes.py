#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panoptic PQ/SQ/RQ evaluation on real Cityscapes validation set (19 trainIds).

Registers a Cityscapes panoptic-trainId dataset from the JSON+PNGs produced by
`../data_prep/prepare_cityscapes_panoptic.py`, loads a trained checkpoint, and
runs Detectron2's `COCOPanopticEvaluator` — this produces the exact
All/Things/Stuff PQ/SQ/RQ columns reported in the thesis's result tables.

Usage:
    python eval_panoptic_cityscapes.py \
        --config-file ../../configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml \
        --weights /path/to/model_final.pth --output /path/to/output_dir
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(_THIS_DIR)))  # repo root, for `mask2former`

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.evaluation import COCOPanopticEvaluator, inference_on_dataset
from detectron2.modeling import build_model

from mask2former import add_maskformer2_config


# -------------------------
# 1) Cityscapes panoptic (trainId) 데이터셋 등록
# -------------------------

CITY_ROOT   = os.environ.get("CITYSCAPES_EVAL_ROOT", "/media/vip-dell/HC")
LEFT_ROOT   = os.path.join(CITY_ROOT, "leftImg8bit")
PAN_JSON    = os.path.join(CITY_ROOT, "cityscapes_panoptic_trainId_val.json")
PAN_ROOT    = os.path.join(CITY_ROOT, "cityscapes_panoptic_trainId_val")

DATASET_NAME = "cityscapes_panoptic_val_trainId"


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


def _register_cityscapes_panoptic_trainId():
    assert os.path.isfile(PAN_JSON), f"panoptic json이 없습니다: {PAN_JSON}"
    assert os.path.isdir(PAN_ROOT),  f"panoptic png 디렉토리가 없습니다: {PAN_ROOT}"
    assert os.path.isdir(LEFT_ROOT), f"leftImg8bit 디렉토리가 없습니다: {LEFT_ROOT}"

    print(f"[REG] PAN_JSON = {PAN_JSON}")
    print(f"[REG] PAN_ROOT = {PAN_ROOT}")
    print(f"[REG] LEFT_ROOT = {LEFT_ROOT}")

    # ---- JSON 읽기 ----
    with open(PAN_JSON, "r") as f:
        j = json.load(f)

    cats = j.get("categories", [])
    print(f"[REG] categories in JSON: {len(cats)} (expect 19)")

    # ==== ★ 핵심 수정: thing/stuff 매핑을 모두 0~18 identity로 열어둔다 ====
    all_ids = [c["id"] for c in cats]        # 보통 [0,1,...,18]
    thing_ids = list(all_ids)
    stuff_ids = list(all_ids)

    thing_map = {i: i for i in thing_ids}    # dataset-id -> contiguous-id
    stuff_map = {i: i for i in stuff_ids}    # dataset-id -> contiguous-id
    # panoptic 전체도 identity
    pan_d2c   = {i: i for i in all_ids}

    # ---- Loader 정의 ----
    def loader():
        id2img = {im["id"]: im for im in j["images"]}
        annos = j["annotations"]

        dataset = []
        miss_img, miss_pan = 0, 0

        for ann in annos:
            img_info = id2img[ann["image_id"]]

            # ---- 이미지 경로 찾기 ----
            rel = img_info.get("file_name", "")
            candidates = []

            if rel:
                # leftImg8bit/val/xxx/... 형태일 수도 있고, val/xxx/...일 수도 있음
                candidates.append(os.path.join(CITY_ROOT, rel))
                candidates.append(os.path.join(LEFT_ROOT, rel))

            # id 기반 fallback
            base = os.path.basename(rel) if rel else (img_info.get("id", "") + "_leftImg8bit.png")
            if not base.endswith("_leftImg8bit.png"):
                base = base.split("_gtFine")[0]
                if not base.endswith("_leftImg8bit.png"):
                    base = base + "_leftImg8bit.png"

            city = base.split("_")[0]
            candidates.append(os.path.join(LEFT_ROOT, "val", city, base))

            img_path = None
            for c in candidates:
                if c and os.path.exists(c):
                    img_path = c
                    break

            if img_path is None:
                miss_img += 1
                continue

            # ---- panoptic GT PNG 경로 ----
            pan_file = ann["file_name"]  # frankfurt_000000_...._panoptic.png
            pan_candidates = [
                os.path.join(PAN_ROOT, pan_file),
                os.path.join(PAN_ROOT, os.path.basename(pan_file)),
            ]
            pan_path = None
            for c in pan_candidates:
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

            H = int(img_info.get("height", 1024))
            W = int(img_info.get("width", 2048))
            dataset.append(
                {
                    "image_id": ann["image_id"],
                    "file_name": img_path,
                    "height": H,
                    "width": W,
                    "pan_seg_file_name": pan_path,
                    "segments_info": segs,
                }
            )

        print(
            f"[REG-LOADER] {DATASET_NAME}: total={len(dataset)}, "
            f"missing_image={miss_img}, missing_panoptic={miss_pan}"
        )
        return dataset

    # ---- Catalog 등록 ----
    _safe_remove(DATASET_NAME)
    DatasetCatalog.register(DATASET_NAME, loader)

    # ---- 메타데이터 설정 ----
    # 이름은 JSON에서 isthing 기준으로 나누고, 매핑은 위에서 만든 identity 사용
    thing_classes = [c["name"] for c in cats if c.get("isthing", 0) == 1]
    stuff_classes = [c["name"] for c in cats if c.get("isthing", 0) == 0]

    meta = MetadataCatalog.get(DATASET_NAME)
    meta.set(
        image_root=LEFT_ROOT,
        panoptic_root=PAN_ROOT,
        panoptic_json=PAN_JSON,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,

        thing_dataset_id_to_contiguous_id=thing_map,
        stuff_dataset_id_to_contiguous_id=stuff_map,

        panoptic_dataset_id_to_contiguous_id=pan_d2c,
        panoptic_contiguous_id_to_dataset_id={v: k for k, v in pan_d2c.items()},
        label_divisor=1000,
        panoptic_label_divisor=1000,
    )

    print(f"[REG] Dataset '{DATASET_NAME}' 등록 완료.")
    print(f"      thing_ids = {thing_ids}")
    print(f"      stuff_ids = {stuff_ids}")


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

    # 학습이랑 맞추기
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    if not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = args.output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg


# -------------------------
# 3) 메인: PQ/SQ/RQ 평가
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
    _register_cityscapes_panoptic_trainId()

    # 2) cfg 구성
    cfg = build_cfg(args)
    default_setup(cfg, args)

    # 3) 모델 생성 + weight 로드
    print("[MODEL] build_model & load weights...")
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # 4) 평가 세팅
    evaluator = COCOPanopticEvaluator(
        DATASET_NAME,
        output_dir=args.output,
    )
    data_loader = build_detection_test_loader(cfg, DATASET_NAME)

    # 5) 평가 실행
    print("[EVAL] start inference_on_dataset (PQ/SQ/RQ)...")
    t0 = time.time()
    with torch.no_grad():
        results = inference_on_dataset(model, data_loader, evaluator)
    dt = time.time() - t0

    print("\n================ CITYSCAPES PANOPTIC EVAL (trainId, val) ================")
    print(f"Time: {dt:.1f} sec")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("=======================================================================")


if __name__ == "__main__":
    main()
