#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##최종코드 (같은 라벨은 항상 같은 색으로 시각화)

import os, sys, glob, time, json, argparse
import numpy as np
import cv2
import torch

# --- 프로젝트 경로 & 데이터셋 등록 (infer.py와 동일) ---
sys.path.insert(0, "/home/vip-dell/NoRo_AD")
import resister  # carla_final_panoptic_{train,val} 등록용

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.utils.visualizer import Visualizer   # 더 이상 사용 X
from mask2former import add_maskformer2_config
from panopticapi.utils import id2rgb

# --- Cityscapes 19-class용 고정 색 팔레트 (RGB 기준) ---
# trainId: 0~18
CITYSCAPES_TRAINID_COLORS_RGB = {
    0:  (128,  64, 128),  # road
    1:  (244,  35, 232),  # sidewalk
    2:  ( 70,  70,  70),  # building
    3:  (102, 102, 156),  # wall
    4:  (190, 153, 153),  # fence
    5:  (153, 153, 153),  # pole
    6:  (250, 170,  30),  # traffic light
    7:  (220, 220,   0),  # traffic sign
    8:  (107, 142,  35),  # vegetation
    9:  (152, 251, 152),  # terrain
    10: ( 70, 130, 180),  # sky
    11: (220,  20,  60),  # person
    12: (255,   0,   0),  # rider
    13: (  0,   0, 142),  # car
    14: (  0,   0,  70),  # truck
    15: (  0,  60, 100),  # bus
    16: (  0,  80, 100),  # train
    17: (  0,   0, 230),  # motorcycle
    18: (119,  11,  32),  # bicycle
}

# OpenCV는 BGR을 쓰니까 BGR로 한 번 변환해 두자
CITYSCAPES_TRAINID_COLORS_BGR = {
    k: (v[2], v[1], v[0]) for k, v in CITYSCAPES_TRAINID_COLORS_RGB.items()
}

# --- (참고) Cityscapes trainId 기준 THING/STUFF 구분 (필요하면 사용) ---
THING_IDS = {6, 7, 11, 12, 13, 14, 15, 16, 17, 18}
STUFF_IDS = {0, 1, 2, 3, 4, 5, 8, 9, 10}

# ----------------------------------------------------

def build_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19

    # meta 가져오기 위해 등록
    cfg.DATASETS.TEST = ("carla_final_panoptic_val",)

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

    # infer.py 와 동일하게 RGB + 512 사이즈
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    cfg.freeze()
    return cfg


@torch.inference_mode()
def run_on_image(predictor, img_bgr):
    """
    - predictor에서 panoptic_seg + segments_info 받기
    - trainId 기준 고정 팔레트로 색칠 (같은 라벨 == 항상 같은 색)
    - overlay(BGR), panoptic id 맵(np.uint32), segments_info 반환
    """
    outputs = predictor(img_bgr)

    if "panoptic_seg" not in outputs:
        return img_bgr.copy(), None, None

    pan_seg, segments_info = outputs["panoptic_seg"]  # pan_seg: (H,W) torch.Tensor
    pan_np = pan_seg.to("cpu").numpy().astype(np.uint32)

    h, w = pan_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 세그먼트별로 해당 id 영역에 클래스 색을 칠해줌
    for seg in segments_info:
        cid = seg["category_id"]  # 이게 trainId(0~18)라고 가정
        seg_id = seg["id"]

        if cid not in CITYSCAPES_TRAINID_COLORS_BGR:
            # 혹시 19 클래스 밖이면 스킵
            continue

        bgr = CITYSCAPES_TRAINID_COLORS_BGR[cid]
        mask = (pan_np == seg_id)
        color_mask[mask] = bgr

    # 원본 위에 반투명 overlay
    overlay = cv2.addWeighted(img_bgr, 0.5, color_mask, 0.5, 0.0)

    return overlay, pan_np, segments_info


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="이미지 파일/디렉토리/글롭 패턴",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="결과를 저장할 디렉토리",
    )
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = build_cfg(args)

    from demo.predictor import VisualizationDemo
    demo = VisualizationDemo(cfg)
    predictor = demo.predictor

    # meta 는 이제 안 써도 됨 (색은 우리가 직접 정의)
    # meta = MetadataCatalog.get("carla_final_panoptic_val")

    out_root = ensure_dir(args.output)
    out_overlay = ensure_dir(os.path.join(out_root, "overlay"))
    out_panpng = ensure_dir(os.path.join(out_root, "panoptic_color"))
    out_segjson = ensure_dir(os.path.join(out_root, "segments_info"))

    # ---- 입력 리스트 만들기 (파일 / 디렉토리 / 글롭 모두 지원) ----
    paths = []
    for p in args.input:
        if any(c in p for c in ["*", "?", "[", "]"]):
            paths += sorted(glob.glob(p))
        elif os.path.isdir(p):
            for r, d, fs in os.walk(p):
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

        overlay, pan_id, segs = run_on_image(predictor, img)

        stem = os.path.splitext(os.path.basename(ip))[0]

        # overlay 저장 (같은 라벨은 항상 같은 색으로)
        cv2.imwrite(os.path.join(out_overlay, f"{stem}_overlay.png"), overlay)

        # panoptic 색상 PNG + segments_info JSON 저장 (이건 예전처럼 id2rgb)
        if pan_id is not None:
            color = id2rgb(pan_id)[:, :, ::-1]  # RGB -> BGR
            cv2.imwrite(os.path.join(out_panpng, f"{stem}_panoptic.png"), color)

            with open(os.path.join(out_segjson, f"{stem}_segments.json"), "w") as f:
                json.dump({"segments_info": segs}, f, ensure_ascii=False)

        print(f"[OK] {os.path.basename(ip)}  {time.time()-t0:.2f}s")

    print(f"[DONE] results saved under {out_root}")


if __name__ == "__main__":
    main()
