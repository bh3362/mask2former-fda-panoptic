#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob

import numpy as np
import cv2

# --- 여기만 네 환경에 맞게 수정 ---
CITY_ROOT = os.environ.get("CITYSCAPES_EVAL_ROOT", "/media/vip-dell/HC")  # dir containing leftImg8bit/, gtFine/
IN_GT_DIR = os.path.join(CITY_ROOT, "gtFine")            # gtFine/{train,val,test}
OUT_GT_DIR = os.path.join(CITY_ROOT, "gtFine_trainIds")  # 새로 만들 폴더
# ---------------------------------


# Cityscapes 공식 labelId -> trainId 매핑
LABELID_TO_TRAINID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
    # 나머지는 전부 255 (ignore)
}


def build_lut():
    """
    0~255 정수에 대해, labelId -> trainId 매핑을 담은 lookup table 생성.
    """
    lut = np.full(256, 255, dtype=np.uint8)  # default = 255 (ignore)
    for lid, tid in LABELID_TO_TRAINID.items():
        lut[lid] = tid
    return lut


def convert_split(split, lut):
    """
    한 split(train/val/test)에 대해:
      gtFine/<split>/<city>/*_gtFine_labelIds.png
    → gtFine_trainIds/<split>/<city>/*_gtFine_trainIds.png 으로 변환.
    """
    in_split_dir = os.path.join(IN_GT_DIR, split)
    if not os.path.isdir(in_split_dir):
        print(f"[WARN] 입력 디렉토리 없음: {in_split_dir}")
        return

    out_split_dir = os.path.join(OUT_GT_DIR, split)
    os.makedirs(out_split_dir, exist_ok=True)

    cities = sorted(os.listdir(in_split_dir))
    print(f"[INFO] split='{split}', cities={len(cities)}")

    for city in cities:
        in_city_dir = os.path.join(in_split_dir, city)
        if not os.path.isdir(in_city_dir):
            continue

        out_city_dir = os.path.join(out_split_dir, city)
        os.makedirs(out_city_dir, exist_ok=True)

        # *_gtFine_labelIds.png 패턴 찾기
        pattern = os.path.join(in_city_dir, "*_gtFine_labelIds.png")
        files = sorted(glob.glob(pattern))

        if not files:
            continue

        print(f"  [CITY] {split}/{city}: {len(files)} files")

        for in_path in files:
            filename = os.path.basename(in_path)
            out_filename = filename.replace("_labelIds", "_trainIds")
            out_path = os.path.join(out_city_dir, out_filename)

            # 이미 있으면 스킵하고 싶으면 아래 if 해제
            # if os.path.isfile(out_path):
            #     continue

            lbl = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            if lbl is None:
                print(f"    [WARN] fail to read: {in_path}")
                continue

            if lbl.ndim != 2:
                print(f"    [WARN] not single-channel: {in_path}, shape={lbl.shape}")
                continue

            # LUT로 매핑
            train = lut[lbl]

            cv2.imwrite(out_path, train)

    print(f"[DONE] split='{split}' 변환 완료 -> {out_split_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="변환할 split 리스트 (예: train val test)",
    )
    args = parser.parse_args()

    lut = build_lut()

    print("[INFO] labelId -> trainId LUT 요약:")
    for lid, tid in sorted(LABELID_TO_TRAINID.items()):
        print(f"  labelId {lid:2d} -> trainId {tid:2d}")
    print("  others -> 255 (ignore)")

    for split in args.splits:
        convert_split(split, lut)


if __name__ == "__main__":
    main()
