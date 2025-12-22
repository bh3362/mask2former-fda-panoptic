#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import json
import argparse

import numpy as np
import cv2
import torch

# --- 프로젝트 루트 추가 ---
sys.path.insert(0, "/home/vip-dell/NoRo_AD")

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from panopticapi.utils import id2rgb

# DefaultPredictor 래퍼 (기존 demo/predictor 그대로 사용)
from demo.predictor import VisualizationDemo


def build_cfg(args):
    """
    Mask2Former 설정 로드 + 테스트용 세팅.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19

    # 학습 때랑 맞춰서 RGB + 512
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    # DATASETS.TEST 는 굳이 안 써도 되지만 비워두면 에러나는 경우가 있어서 dummy 로 세팅
    if not len(cfg.DATASETS.TEST):
        cfg.DATASETS.TEST = ("carla_final_panoptic_val",)

    cfg.freeze()
    return cfg


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def colorize_panoptic(pan_seg_np: np.ndarray) -> np.ndarray:
    """
    panopticapi.utils.id2rgb 를 이용해서
    panoptic id (instance+class)별로 고유 색을 입힌 RGB 이미지 생성.
    같은 class라도 id 가 다르면 색이 달라짐.
    """
    # pan_seg_np: (H, W) uint32
    color_rgb = id2rgb(pan_seg_np)  # (H, W, 3) RGB, uint8
    return color_rgb


def blend_overlay(img_bgr: np.ndarray, color_rgb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    원본 BGR 이미지와 panoptic 색상 맵(RGB)을 합성해서 overlay 생성.
    """
    color_bgr = color_rgb[:, :, ::-1]  # RGB -> BGR
    overlay = (img_bgr.astype(np.float32) * (1.0 - alpha) +
               color_bgr.astype(np.float32) * alpha)
    return overlay.astype(np.uint8)


@torch.inference_mode()
def run_on_image(predictor, img_bgr):
    """
    - predictor 로 panoptic_seg, segments_info 얻기
    - panoptic id map (np.uint32) 반환
    """
    outputs = predictor(img_bgr)

    if "panoptic_seg" not in outputs:
        return img_bgr.copy(), None, None

    pan_seg, segments_info = outputs["panoptic_seg"]  # pan_seg: (H, W) torch.Tensor
    pan_np = pan_seg.to("cpu").numpy().astype(np.uint32)

    # color_map 은 나중에 main 에서 만들 거라 여기선 pan_np 만 넘김
    return img_bgr, pan_np, segments_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, nargs="+",
                        help="이미지 파일 / 디렉토리 / glob 패턴")
    parser.add_argument("--output", required=True,
                        help="결과를 저장할 디렉토리")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = build_cfg(args)

    # DefaultPredictor 래퍼
    demo = VisualizationDemo(cfg)
    predictor = demo.predictor

    out_root = ensure_dir(args.output)
    out_overlay = ensure_dir(os.path.join(out_root, "overlay"))
    out_panpng = ensure_dir(os.path.join(out_root, "panoptic_color"))
    out_segjson = ensure_dir(os.path.join(out_root, "segments_info"))

    # ----- 입력 경로 수집 -----
    paths = []
    for p in args.input:
        if any(c in p for c in ["*", "?", "[", "]"]):
            paths += sorted(glob.glob(p))
        elif os.path.isdir(p):
            for r, _, fs in os.walk(p):
                for f in fs:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        paths.append(os.path.join(r, f))
        else:
            paths.append(p)

    paths = sorted(set(paths))
    print(f"[INFO] Found {len(paths)} images.")

    for ip in paths:
        if not os.path.isfile(ip):
            continue

        img = read_image(ip, format="BGR")
        t0 = time.time()

        _, pan_id, segs = run_on_image(predictor, img)

        stem = os.path.splitext(os.path.basename(ip))[0]

        if pan_id is not None:
            # 1) panoptic id -> instance별 고유 색 RGB 맵
            color_rgb = colorize_panoptic(pan_id)

            # 2) 원본 + 색맵 overlay
            overlay_bgr = blend_overlay(img, color_rgb, alpha=0.6)

            # 3) 저장
            cv2.imwrite(os.path.join(out_overlay, f"{stem}_overlay.png"), overlay_bgr)
            cv2.imwrite(
                os.path.join(out_panpng, f"{stem}_panoptic.png"),
                color_rgb[:, :, ::-1]  # RGB -> BGR
            )

            with open(os.path.join(out_segjson, f"{stem}_segments.json"), "w") as f:
                json.dump({"segments_info": segs}, f, ensure_ascii=False)
        else:
            # panoptic_seg 없으면 원본만 복사
            cv2.imwrite(os.path.join(out_overlay, f"{stem}_overlay.png"), img)

        print(f"[OK] {os.path.basename(ip)}  {time.time() - t0:.2f}s")

    print(f"[DONE] results saved under {out_root}")


if __name__ == "__main__":
    main()
