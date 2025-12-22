# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import sys
# import json
# import argparse
# import time

# import numpy as np
# import torch
# import cv2

# # 프로젝트 경로
# sys.path.append("/home/vip-dell/NoRo_AD")

# from detectron2.config import get_cfg
# from detectron2.utils.logger import setup_logger
# from detectron2.data import (
#     MetadataCatalog,
#     DatasetCatalog,
#     build_detection_test_loader,
#     DatasetMapper,
# )
# from detectron2.data import transforms as T
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.engine import default_setup
# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.evaluation import COCOPanopticEvaluator, inference_on_dataset
# from detectron2.modeling import build_model

# from mask2former import add_maskformer2_config
# from panopticapi.utils import rgb2id


# # -------------------------
# # 1) Cityscapes panoptic (trainId) 데이터셋 등록
# # -------------------------

# CITY_ROOT   = "/media/vip-dell/HC"
# LEFT_ROOT   = os.path.join(CITY_ROOT, "leftImg8bit")
# PAN_JSON    = os.path.join(CITY_ROOT, "cityscapes_panoptic_trainId_val.json")
# PAN_ROOT    = os.path.join(CITY_ROOT, "cityscapes_panoptic_trainId_val")

# DATASET_NAME = "cityscapes_panoptic_val_trainId"


# def _safe_remove(name: str):
#     """DatasetCatalog / MetadataCatalog에 기존에 등록된 이름 있으면 제거"""
#     if hasattr(DatasetCatalog, "remove"):
#         try:
#             DatasetCatalog.remove(name)
#         except KeyError:
#             pass
#     reg = getattr(DatasetCatalog, "_REGISTERED", None) or getattr(DatasetCatalog, "REGISTERED", None)
#     if isinstance(reg, dict):
#         reg.pop(name, None)
#     meta = getattr(MetadataCatalog, "_NAME_TO_META", None)
#     if isinstance(meta, dict):
#         meta.pop(name, None)


# def _register_cityscapes_panoptic_trainId():
#     assert os.path.isfile(PAN_JSON), f"panoptic json이 없습니다: {PAN_JSON}"
#     assert os.path.isdir(PAN_ROOT),  f"panoptic png 디렉토리가 없습니다: {PAN_ROOT}"
#     assert os.path.isdir(LEFT_ROOT), f"leftImg8bit 디렉토리가 없습니다: {LEFT_ROOT}"

#     print(f"[REG] PAN_JSON = {PAN_JSON}")
#     print(f"[REG] PAN_ROOT = {PAN_ROOT}")
#     print(f"[REG] LEFT_ROOT = {LEFT_ROOT}")

#     # ---- JSON 읽기 ----
#     with open(PAN_JSON, "r") as f:
#         j = json.load(f)

#     cats = j.get("categories", [])
#     print(f"[REG] categories in JSON: {len(cats)} (expect 19)")

#     # ==== ★ thing/stuff/panoptic 모두 0~18 identity 매핑 ====
#     all_ids   = [c["id"] for c in cats]       # 보통 [0,1,...,18]
#     thing_ids = list(all_ids)
#     stuff_ids = list(all_ids)

#     thing_map = {i: i for i in thing_ids}     # dataset-id -> contiguous-id
#     stuff_map = {i: i for i in stuff_ids}     # dataset-id -> contiguous-id
#     pan_d2c   = {i: i for i in all_ids}       # 전체도 identity

#     # ---- Loader 정의 ----
#     def loader():
#         id2img = {im["id"]: im for im in j["images"]}
#         annos = j["annotations"]

#         dataset = []
#         miss_img, miss_pan = 0, 0

#         for ann in annos:
#             img_info = id2img[ann["image_id"]]

#             # ---- 이미지 경로 찾기 ----
#             rel = img_info.get("file_name", "")
#             candidates = []

#             if rel:
#                 candidates.append(os.path.join(CITY_ROOT, rel))
#                 candidates.append(os.path.join(LEFT_ROOT, rel))

#             # id 기반 fallback
#             base = os.path.basename(rel) if rel else (img_info.get("id", "") + "_leftImg8bit.png")
#             if not base.endswith("_leftImg8bit.png"):
#                 base = base.split("_gtFine")[0]
#                 if not base.endswith("_leftImg8bit.png"):
#                     base = base + "_leftImg8bit.png"

#             city = base.split("_")[0]
#             candidates.append(os.path.join(LEFT_ROOT, "val", city, base))

#             img_path = None
#             for c in candidates:
#                 if c and os.path.exists(c):
#                     img_path = c
#                     break

#             if img_path is None:
#                 miss_img += 1
#                 continue

#             # ---- panoptic GT PNG 경로 ----
#             pan_file = ann["file_name"]  # frankfurt_000000_...._panoptic.png
#             pan_candidates = [
#                 os.path.join(PAN_ROOT, pan_file),
#                 os.path.join(PAN_ROOT, os.path.basename(pan_file)),
#             ]
#             pan_path = None
#             for c in pan_candidates:
#                 if c and os.path.exists(c):
#                     pan_path = c
#                     break
#             if pan_path is None:
#                 miss_pan += 1
#                 continue

#             segs = ann.get("segments_info", [])
#             for s in segs:
#                 if "iscrowd" not in s:
#                     s["iscrowd"] = 0

#             H = int(img_info.get("height", 1024))
#             W = int(img_info.get("width", 2048))
#             dataset.append(
#                 {
#                     "image_id": ann["image_id"],
#                     "file_name": img_path,
#                     "height": H,
#                     "width": W,
#                     "pan_seg_file_name": pan_path,
#                     "segments_info": segs,   # 🔥 GT segments_info 그대로 넣어둠
#                 }
#             )

#         print(
#             f"[REG-LOADER] {DATASET_NAME}: total={len(dataset)}, "
#             f"missing_image={miss_img}, missing_panoptic={miss_pan}"
#         )
#         return dataset

#     # ---- Catalog 등록 ----
#     _safe_remove(DATASET_NAME)
#     DatasetCatalog.register(DATASET_NAME, loader)

#     # ---- 메타데이터 설정 ----
#     thing_classes = [c["name"] for c in cats if c.get("isthing", 0) == 1]
#     stuff_classes = [c["name"] for c in cats if c.get("isthing", 0) == 0]

#     meta = MetadataCatalog.get(DATASET_NAME)
#     meta.set(
#         image_root=LEFT_ROOT,
#         panoptic_root=PAN_ROOT,
#         panoptic_json=PAN_JSON,
#         evaluator_type="coco_panoptic_seg",
#         ignore_label=255,
#         thing_classes=thing_classes,
#         stuff_classes=stuff_classes,

#         thing_dataset_id_to_contiguous_id=thing_map,
#         stuff_dataset_id_to_contiguous_id=stuff_map,

#         panoptic_dataset_id_to_contiguous_id=pan_d2c,
#         panoptic_contiguous_id_to_dataset_id={v: k for k, v in pan_d2c.items()},
#         # div는 “모델 예측” 내부 포맷용이고 GT에는 안 쓸 수도 있음
#         label_divisor=1000,
#         panoptic_label_divisor=1000,
#     )

#     print(f"[REG] Dataset '{DATASET_NAME}' 등록 완료.")
#     print(f"      thing_ids = {thing_ids}")
#     print(f"      stuff_ids = {stuff_ids}")


# # -------------------------
# # 2) cfg 생성
# # -------------------------

# def build_cfg(args):
#     cfg = get_cfg()
#     add_deeplab_config(cfg)
#     add_maskformer2_config(cfg)

#     cfg.merge_from_file(args.config_file)
#     if args.opts:
#         cfg.merge_from_list(args.opts)

#     cfg.defrost()
#     cfg.DATASETS.TEST = (DATASET_NAME,)
#     cfg.INPUT.MASK_FORMAT = "bitmask"
#     cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19

#     # 학습이랑 맞추기
#     cfg.INPUT.FORMAT = "RGB"
#     if args.weights:
#         cfg.MODEL.WEIGHTS = args.weights

#     if not cfg.OUTPUT_DIR:
#         cfg.OUTPUT_DIR = args.output
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     cfg.freeze()
#     return cfg


# # -------------------------
# # 3) mIoU 계산 (panoptic → semantic 변환)
# # -------------------------

# @torch.no_grad()
# def compute_mIoU_from_panoptic(model, cfg, dataset_name, num_classes=19, ignore_label=255):
#     """
#     COCOPanopticEvaluator로 PQ/SQ/RQ 계산한 동일 모델/데이터셋에 대해
#     panoptic prediction과 GT panoptic을 이용해서 trainId 기준 mIoU 계산.

#     🔥 GT는 rgb2id + GT segments_info로부터 category_id를 채워넣어서 semantic으로 복원한다.
#        (더 이상 `gt_id // div` 같은 가정 안 함)
#     """
#     print("[mIoU] Start second pass over dataset for semantic IoU (trainId)...")

#     meta = MetadataCatalog.get(dataset_name)

#     # 테스트용 mapper (PQ랑 동일하게 ResizeShortestEdge만 사용)
#     augs = [T.ResizeShortestEdge(
#         short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
#         max_size=cfg.INPUT.MAX_SIZE_TEST
#     )]

#     mapper = DatasetMapper(
#         cfg,
#         is_train=False,
#         augmentations=augs,
#         image_format=cfg.INPUT.FORMAT,
#         use_instance_mask=False,
#         use_keypoint=False,
#     )

#     data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

#     model.eval()
#     device = torch.device(cfg.MODEL.DEVICE)
#     model.to(device)

#     k = num_classes
#     conf_mat = np.zeros((k, k), dtype=np.int64)

#     for batch in data_loader:
#         outputs = model(batch)

#         for inp, out in zip(batch, outputs):
#             if "panoptic_seg" not in out:
#                 continue

#             # ---------------------- 예측 semantic ----------------------
#             pan_pred, segments_info = out["panoptic_seg"]   # pan_pred: (H,W)
#             pan_pred = pan_pred.to("cpu").numpy().astype(np.int64)

#             sem_pred = np.full_like(pan_pred, fill_value=ignore_label, dtype=np.int64)

#             # segments_info: pred에서 각 segment의 id & category_id
#             for seg in segments_info:
#                 seg_id = seg["id"]
#                 cid    = seg["category_id"]      # 0~18 (trainId 기준)
#                 mask = (pan_pred == seg_id)
#                 if not np.any(mask):
#                     continue
#                 sem_pred[mask] = cid

#             # ---------------------- GT semantic ----------------------
#             gt_pan_path = inp.get("pan_seg_file_name", None)
#             if not gt_pan_path or (not os.path.isfile(gt_pan_path)):
#                 continue

#             gt_rgb = cv2.imread(gt_pan_path, cv2.IMREAD_COLOR)
#             if gt_rgb is None:
#                 continue
#             gt_rgb = gt_rgb[:, :, ::-1]  # BGR -> RGB

#             gt_id = rgb2id(gt_rgb).astype(np.int64)  # (H_gt, W_gt), segment_id map
#             sem_gt = np.full_like(gt_id, fill_value=ignore_label, dtype=np.int64)

#             # 🔥 GT의 segments_info는 loader에서 이미 넣어둔 걸 그대로 사용
#             gt_segs = inp.get("segments_info", [])
#             for seg in gt_segs:
#                 seg_id = seg["id"]               # GT panoptic PNG 안에서의 segment id
#                 cid    = seg["category_id"]      # 0~18 (trainId 기준)
#                 mask = (gt_id == seg_id)
#                 if not np.any(mask):
#                     continue
#                 sem_gt[mask] = cid

#             # 사이즈 맞추기: GT를 prediction 사이즈로 downsample (nearest)
#             h_pred, w_pred = sem_pred.shape
#             h_gt, w_gt = sem_gt.shape
#             if (h_pred, w_pred) != (h_gt, w_gt):
#                 sem_gt = cv2.resize(
#                     sem_gt.astype(np.int32),
#                     (w_pred, h_pred),
#                     interpolation=cv2.INTER_NEAREST
#                 ).astype(np.int64)

#             # ignore label 제외
#             valid = (sem_gt != ignore_label)
#             if not np.any(valid):
#                 continue

#             gt_flat = sem_gt[valid].ravel()
#             pred_flat = sem_pred[valid].ravel()

#             # 범위 밖의 클래스는 모두 무시
#             mask_valid = (gt_flat >= 0) & (gt_flat < k)
#             mask_valid &= (pred_flat >= 0) & (pred_flat < k)

#             gt_flat = gt_flat[mask_valid]
#             pred_flat = pred_flat[mask_valid]
#             if gt_flat.size == 0:
#                 continue

#             # confusion matrix 업데이트
#             ind = k * gt_flat + pred_flat
#             binc = np.bincount(ind, minlength=k * k)
#             conf_mat += binc.reshape(k, k)

#     # IoU 계산
#     intersection = np.diag(conf_mat).astype(np.float64)
#     gt_sum = conf_mat.sum(axis=1).astype(np.float64)
#     pred_sum = conf_mat.sum(axis=0).astype(np.float64)
#     union = gt_sum + pred_sum - intersection

#     iou_per_class = np.zeros(k, dtype=np.float64)
#     valid_classes = union > 0
#     iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]
#     mIoU = float(iou_per_class[valid_classes].mean()) if np.any(valid_classes) else 0.0

#     print("\n================ CITYSCAPES mIoU (trainId, val) ================")
#     print(f"mIoU: {mIoU:.4f}")
#     print("IoU per class (0~18 trainId):")
#     for cid, val in enumerate(iou_per_class):
#         print(f"  class {cid:2d}: {val:.4f}")
#     print("===============================================================")

#     return {
#         "mIoU": mIoU,
#         "IoU_per_class": iou_per_class.tolist(),
#     }


# # -------------------------
# # 4) 메인: PQ/SQ/RQ + mIoU 평가
# # -------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config-file", required=True)
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--output", required=True)
#     parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
#     args = parser.parse_args()

#     setup_logger()

#     # 1) 데이터셋 등록
#     _register_cityscapes_panoptic_trainId()

#     # 2) cfg 구성
#     cfg = build_cfg(args)
#     default_setup(cfg, args)

#     # 3) 모델 생성 + weight 로드
#     print("[MODEL] build_model & load weights...")
#     model = build_model(cfg)
#     model.eval()
#     DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

#     # 4) Panoptic PQ/SQ/RQ 평가
#     evaluator = COCOPanopticEvaluator(
#         DATASET_NAME,
#         output_dir=args.output,
#     )
#     data_loader = build_detection_test_loader(cfg, DATASET_NAME)

#     print("[EVAL] start inference_on_dataset (PQ/SQ/RQ)...")
#     t0 = time.time()
#     with torch.no_grad():
#         results = inference_on_dataset(model, data_loader, evaluator)
#     dt = time.time() - t0

#     print("\n================ CITYSCAPES PANOPTIC EVAL (trainId, val) ================")
#     print(f"Time: {dt:.1f} sec")
#     for k, v in results.items():
#         print(f"{k}: {v}")
#     print("=======================================================================")

#     # 5) mIoU 추가 계산
#     compute_mIoU_from_panoptic(model, cfg, DATASET_NAME, num_classes=19, ignore_label=255)


# if __name__ == "__main__":
#     main()
# # ss

# panoptic GT/Pred json을 읽고 category_id별 등장 여부 체크#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


# =======================================================
# 0) trainId별 고정 컬러 팔레트 + colorize 함수
# =======================================================

# Cityscapes trainId 팔레트 (BGR) – 필요하면 색 바꿔도 됨
TRAINID_COLOR_MAP = {
    0:  (128,  64, 128),  # road
    1:  (244,  35, 232),  # sidewalk
    2:  (70,   70,  70),  # building
    3:  (102, 102, 156),  # wall
    4:  (190, 153, 153),  # fence
    5:  (153, 153, 153),  # pole
    6:  (250, 170,  30),  # traffic light
    7:  (220, 220,   0),  # traffic sign
    8:  (107, 142,  35),  # vegetation
    9:  (152, 251, 152),  # terrain
    10: (70,  130, 180),  # sky
    11: (220,  20,  60),  # person
    12: (255,   0,   0),  # rider
    13: (0,     0, 142),  # car
    14: (0,     0,  70),  # truck
    15: (0,    60, 100),  # bus
    16: (0,    80, 100),  # train
    17: (0,     0, 230),  # motorcycle
    18: (119,  11,  32),  # bicycle
}


def colorize_trainId(sem_map: np.ndarray, ignore_label: int = 255) -> np.ndarray:
    """
    sem_map: HxW, trainId (0~18 / ignore_label)
    return: HxWx3, BGR uint8 (cv2.imwrite용)
    """
    h, w = sem_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for tid, bgr in TRAINID_COLOR_MAP.items():
        mask = (sem_map == tid)
        if not np.any(mask):
            continue
        color[mask] = np.array(bgr, dtype=np.uint8)

    # ignore 영역은 검정 (원하면 다른 색으로 변경 가능)
    color[sem_map == ignore_label] = (0, 0, 0)
    return color


# =======================================================
# 1) Cityscapes panoptic (trainId) 데이터셋 등록
# =======================================================

CITY_ROOT   = "/media/vip-dell/HC"
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

    # ==== ★ thing/stuff/panoptic 모두 0~18 identity 매핑 ====
    all_ids   = [c["id"] for c in cats]       # 보통 [0,1,...,18]
    thing_ids = list(all_ids)
    stuff_ids = list(all_ids)

    thing_map = {i: i for i in thing_ids}     # dataset-id -> contiguous-id
    stuff_map = {i: i for i in stuff_ids}     # dataset-id -> contiguous-id
    pan_d2c   = {i: i for i in all_ids}       # 전체도 identity

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
                    "segments_info": segs,   # 🔥 GT segments_info 그대로 넣어둠
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
        # div는 “모델 예측” 내부 포맷용이고 GT에는 안 쓸 수도 있음
        label_divisor=1000,
        panoptic_label_divisor=1000,
    )

    print(f"[REG] Dataset '{DATASET_NAME}' 등록 완료.")
    print(f"      thing_ids = {thing_ids}")
    print(f"      stuff_ids = {stuff_ids}")


# =======================================================
# 2) cfg 생성
# =======================================================

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
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    if not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = args.output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg


# =======================================================
# 3) mIoU 계산 + trainId 고정색 시각화
# =======================================================

@torch.no_grad()
def compute_mIoU_from_panoptic(model, cfg, dataset_name, num_classes=19, ignore_label=255):
    """
    COCOPanopticEvaluator로 PQ/SQ/RQ 계산한 동일 모델/데이터셋에 대해
    panoptic prediction과 GT panoptic을 이용해서 trainId 기준 mIoU 계산.

    🔥 GT는 rgb2id + GT segments_info로부터 category_id를 채워넣어서 semantic으로 복원한다.
       (더 이상 `gt_id // div` 같은 가정 안 함)

    🔥 추가: 예측 semantic(trainId)을 라벨별 고정색으로 저장
    """
    print("[mIoU] Start second pass over dataset for semantic IoU (trainId)...")

    meta = MetadataCatalog.get(dataset_name)

    # 테스트용 mapper (PQ랑 동일하게 ResizeShortestEdge만 사용)
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

    # ---- 시각화 설정 ----
    vis_dir = os.path.join(cfg.OUTPUT_DIR, "semantic_vis_fixedcolor")
    os.makedirs(vis_dir, exist_ok=True)
    vis_limit = 30  # 최대 30장까지 저장
    vis_count = 0

    for batch in data_loader:
        outputs = model(batch)

        for inp, out in zip(batch, outputs):
            if "panoptic_seg" not in out:
                continue

            # ---------------------- 예측 semantic ----------------------
            pan_pred, segments_info = out["panoptic_seg"]   # pan_pred: (H,W)
            pan_pred = pan_pred.to("cpu").numpy().astype(np.int64)

            sem_pred = np.full_like(pan_pred, fill_value=ignore_label, dtype=np.int64)

            # segments_info: pred에서 각 segment의 id & category_id
            for seg in segments_info:
                seg_id = seg["id"]
                cid    = seg["category_id"]      # 0~18 (trainId 기준)
                mask = (pan_pred == seg_id)
                if not np.any(mask):
                    continue
                sem_pred[mask] = cid

            # ---------------------- GT semantic ----------------------
            gt_pan_path = inp.get("pan_seg_file_name", None)
            if not gt_pan_path or (not os.path.isfile(gt_pan_path)):
                continue

            gt_rgb = cv2.imread(gt_pan_path, cv2.IMREAD_COLOR)
            if gt_rgb is None:
                continue
            gt_rgb = gt_rgb[:, :, ::-1]  # BGR -> RGB

            gt_id = rgb2id(gt_rgb).astype(np.int64)  # (H_gt, W_gt), segment_id map
            sem_gt = np.full_like(gt_id, fill_value=ignore_label, dtype=np.int64)

            # 🔥 GT의 segments_info는 loader에서 이미 넣어둔 걸 그대로 사용
            gt_segs = inp.get("segments_info", [])
            for seg in gt_segs:
                seg_id = seg["id"]               # GT panoptic PNG 안에서의 segment id
                cid    = seg["category_id"]      # 0~18 (trainId 기준)
                mask = (gt_id == seg_id)
                if not np.any(mask):
                    continue
                sem_gt[mask] = cid

            # 사이즈 맞추기: GT를 prediction 사이즈로 downsample (nearest)
            h_pred, w_pred = sem_pred.shape
            h_gt, w_gt = sem_gt.shape
            if (h_pred, w_pred) != (h_gt, w_gt):
                sem_gt = cv2.resize(
                    sem_gt.astype(np.int32),
                    (w_pred, h_pred),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int64)

            # ignore label 제외
            valid = (sem_gt != ignore_label)
            if not np.any(valid):
                continue

            gt_flat = sem_gt[valid].ravel()
            pred_flat = sem_pred[valid].ravel()

            # 범위 밖의 클래스는 모두 무시
            mask_valid = (gt_flat >= 0) & (gt_flat < k)
            mask_valid &= (pred_flat >= 0) & (pred_flat < k)

            gt_flat = gt_flat[mask_valid]
            pred_flat = pred_flat[mask_valid]
            if gt_flat.size == 0:
                continue

            # confusion matrix 업데이트
            ind = k * gt_flat + pred_flat
            binc = np.bincount(ind, minlength=k * k)
            conf_mat += binc.reshape(k, k)

            # ---------------------- 시각화 저장 (trainId 고정색) ----------------------
            if vis_count < vis_limit:
                img_id = inp.get("image_id", None)
                if img_id is None:
                    base = os.path.splitext(os.path.basename(inp["file_name"]))[0]
                    img_id = base

                color_pred = colorize_trainId(sem_pred, ignore_label=ignore_label)
                out_path = os.path.join(vis_dir, f"{img_id}_pred_trainId.png")
                cv2.imwrite(out_path, color_pred)
                print(f"[VIS] wrote {out_path}")
                vis_count += 1

    # IoU 계산
    intersection = np.diag(conf_mat).astype(np.float64)
    gt_sum = conf_mat.sum(axis=1).astype(np.float64)
    pred_sum = conf_mat.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - intersection

    iou_per_class = np.zeros(k, dtype=np.float64)
    valid_classes = union > 0
    iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]
    mIoU = float(iou_per_class[valid_classes].mean()) if np.any(valid_classes) else 0.0

    print("\n================ CITYSCAPES mIoU (trainId, val) ================")
    print(f"mIoU: {mIoU:.4f}")
    print("IoU per class (0~18 trainId):")
    for cid, val in enumerate(iou_per_class):
        print(f"  class {cid:2d}: {val:.4f}")
    print("===============================================================")

    return {
        "mIoU": mIoU,
        "IoU_per_class": iou_per_class.tolist(),
    }


# =======================================================
# 4) 메인: PQ/SQ/RQ + mIoU 평가
# =======================================================

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

    print("\n================ CITYSCAPES PANOPTIC EVAL (trainId, val) ================")
    print(f"Time: {dt:.1f} sec")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("=======================================================================")

    # 5) mIoU + trainId 고정색 시각화
    compute_mIoU_from_panoptic(model, cfg, DATASET_NAME, num_classes=19, ignore_label=255)


if __name__ == "__main__":
    main()
