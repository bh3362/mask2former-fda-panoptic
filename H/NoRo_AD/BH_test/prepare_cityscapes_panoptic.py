#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cityscapes gtFine + leftImg8bit -> Panoptic(trainId) 포맷 변환 스크립트

출력:
  <out_root>/cityscapes_panoptic_trainId_train.json
  <out_root>/cityscapes_panoptic_trainId_val.json
  <out_root>/cityscapes_panoptic_trainId_train/*.png
  <out_root>/cityscapes_panoptic_trainId_val/*.png
"""

import os
import argparse
import json
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm


# -----------------------------
# Cityscapes labelId -> trainId 매핑
# (공식 cityscapesScripts 기준)
# -----------------------------
LABELID_TO_TRAINID = {
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectification border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255,
    255: 255,
}

# trainId(0~18)에 해당하는 카테고리 정의
# (COCO Panoptic categories 형식)
CATEGORIES = [
    {"id": 0,  "name": "road",          "isthing": 0, "color": [128, 64, 128]},
    {"id": 1,  "name": "sidewalk",      "isthing": 0, "color": [244, 35, 232]},
    {"id": 2,  "name": "building",      "isthing": 0, "color": [70, 70, 70]},
    {"id": 3,  "name": "wall",          "isthing": 0, "color": [102, 102, 156]},
    {"id": 4,  "name": "fence",         "isthing": 0, "color": [190, 153, 153]},
    {"id": 5,  "name": "pole",          "isthing": 0, "color": [153, 153, 153]},
    {"id": 6,  "name": "traffic light", "isthing": 1, "color": [250, 170, 30]},
    {"id": 7,  "name": "traffic sign",  "isthing": 1, "color": [220, 220, 0]},
    {"id": 8,  "name": "vegetation",    "isthing": 0, "color": [107, 142, 35]},
    {"id": 9,  "name": "terrain",       "isthing": 0, "color": [152, 251, 152]},
    {"id": 10, "name": "sky",           "isthing": 0, "color": [70, 130, 180]},
    {"id": 11, "name": "person",        "isthing": 1, "color": [220, 20, 60]},
    {"id": 12, "name": "rider",         "isthing": 1, "color": [255, 0, 0]},
    {"id": 13, "name": "car",           "isthing": 1, "color": [0, 0, 142]},
    {"id": 14, "name": "truck",         "isthing": 1, "color": [0, 0, 70]},
    {"id": 15, "name": "bus",           "isthing": 1, "color": [0, 60, 100]},
    {"id": 16, "name": "train",         "isthing": 1, "color": [0, 80, 100]},
    {"id": 17, "name": "motorcycle",    "isthing": 1, "color": [0, 0, 230]},
    {"id": 18, "name": "bicycle",       "isthing": 1, "color": [119, 11, 32]},
]

THING_TRAINIDS = {6, 7, 11, 12, 13, 14, 15, 16, 17, 18}  # 라이트/싸인 + 사람/차류


def id2rgb(id_map: np.ndarray) -> np.ndarray:
    """
    COCO panopticapi와 호환되는 id2rgb 구현
    id: int32 2D -> RGB uint8 3D
    """
    id_map = id_map.astype(np.int64)
    r = id_map % 256
    g = (id_map // 256) % 256
    b = (id_map // (256 * 256)) % 256
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb


def process_split(left_root, gt_root, out_root, split):
    """
    한 split(train 또는 val)에 대해 panoptic PNG + JSON 생성
    """
    left_split = os.path.join(left_root, split)
    gt_split = os.path.join(gt_root, split)
    assert os.path.isdir(left_split), f"leftImg8bit {split} 폴더가 없습니다: {left_split}"
    assert os.path.isdir(gt_split), f"gtFine {split} 폴더가 없습니다: {gt_split}"

    panoptic_dir = os.path.join(out_root, f"cityscapes_panoptic_trainId_{split}")
    os.makedirs(panoptic_dir, exist_ok=True)

    images = []
    annotations = []
    seg_id_counter = 1  # 전역 segment id (1부터 시작)

    img_id_counter = 1

    # city별로 순회
    cities = sorted(os.listdir(left_split))
    for city in cities:
        city_img_dir = os.path.join(left_split, city)
        city_gt_dir = os.path.join(gt_split, city)
        if not os.path.isdir(city_img_dir):
            continue

        img_files = sorted(
            [f for f in os.listdir(city_img_dir) if f.endswith("_leftImg8bit.png")]
        )

        for img_file in tqdm(img_files, desc=f"{split}/{city}", ncols=80):
            img_path = os.path.join(city_img_dir, img_file)

            # base name 예: berlin_000000_000019
            base = img_file.replace("_leftImg8bit.png", "")

            label_path = os.path.join(
                city_gt_dir, base + "_gtFine_labelIds.png"
            )
            inst_path = os.path.join(
                city_gt_dir, base + "_gtFine_instanceIds.png"
            )

            if not (os.path.isfile(label_path) and os.path.isfile(inst_path)):
                # 혹시 빠진 파일이 있으면 스킵
                print(f"[WARN] label/instance 파일 없음, skip: {img_file}")
                continue

            # 이미지 크기
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)

            if label is None or inst is None:
                print(f"[WARN] 읽기 실패, skip: {img_file}")
                continue

            # PNG는 HxW (단일 채널)이어야 함
            if label.ndim == 3:
                label = label[:, :, 0]
            if inst.ndim == 3:
                inst = inst[:, :, 0]

            h, w = label.shape

            # images 항목 추가 (file_name은 leftImg8bit 기준 상대경로)
            image_id = img_id_counter
            img_id_counter += 1

            rel_img_path = os.path.join(split, city, img_file)  # 예: val/frankfurt/xxx_leftImg8bit.png

            images.append({
                "id": image_id,
                "width": int(w),
                "height": int(h),
                "file_name": rel_img_path,
            })

            # panoptic id map (각 segment에 고유 id 할당)
            pan_map = np.zeros((h, w), dtype=np.int32)
            segments_info = []

            # instanceIds의 유니크 값 기준으로 segment 생성
            unique_inst_ids = np.unique(inst)
            for inst_id in unique_inst_ids:
                inst_id = int(inst_id)
                if inst_id == 0:
                    # 0은 void/배경 (unlabeled) 취급
                    continue

                # Cityscapes instanceId -> labelId
                if inst_id < 1000:
                    label_id = inst_id
                else:
                    label_id = inst_id // 1000

                train_id = LABELID_TO_TRAINID.get(label_id, 255)
                if train_id == 255 or train_id < 0:
                    # ignore class
                    continue

                mask = (inst == inst_id)
                area = int(mask.sum())
                if area == 0:
                    continue

                ys, xs = np.where(mask)
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

                seg_id = seg_id_counter
                seg_id_counter += 1

                pan_map[mask] = seg_id

                isthing = 1 if train_id in THING_TRAINIDS else 0

                segments_info.append({
                    "id": int(seg_id),
                    "category_id": int(train_id),
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })

            # panoptic PNG 저장 (COCO 스타일 id2rgb)
            pan_rgb = id2rgb(pan_map)
            pan_name = base + "_panoptic_trainId.png"
            pan_path = os.path.join(panoptic_dir, pan_name)
            cv2.imwrite(pan_path, pan_rgb[:, :, ::-1])  # RGB -> BGR

            annotations.append({
                "image_id": image_id,
                "file_name": pan_name,
                "segments_info": segments_info,
            })

    # JSON 저장
    out_json = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    out_json_path = os.path.join(
        out_root, f"cityscapes_panoptic_trainId_{split}.json"
    )
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {split} split panoptic 생성 완료:")
    print(f"  JSON: {out_json_path}")
    print(f"  PNG dir: {panoptic_dir}")
    print(f"  images={len(images)}, annotations={len(annotations)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leftImg8bit-dir", required=True,
                    help="Cityscapes leftImg8bit 루트 디렉토리")
    ap.add_argument("--gtFine-dir", required=True,
                    help="Cityscapes gtFine 루트 디렉토리")
    ap.add_argument("--output-dir", required=True,
                    help="panoptic JSON/PNG를 저장할 루트 디렉토리")
    ap.add_argument("--splits", default="train,val",
                    help="콤마로 구분된 split 목록 (기본: train,val)")
    args = ap.parse_args()

    left_root = args.leftImg8bit_dir
    gt_root = args.gtFine_dir
    out_root = args.output_dir

    os.makedirs(out_root, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for sp in splits:
        process_split(left_root, gt_root, out_root, sp)


if __name__ == "__main__":
    main()
