#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from detectron2.utils.visualizer import Visualizer
from mask2former import add_maskformer2_config
from panopticapi.utils import id2rgb

# --- Visualizer용 ID 매핑 (infer.py와 동일) ---
THING_ID_TO_LOCAL_IDX = {
    6: 0,  # traffic light
    7: 1,  # traffic sign
    11: 2, # person
    12: 3, # rider
    13: 4, # car
    14: 5, # truck
    15: 6, # bus
    16: 7, # train
    17: 8, # motorcycle
    18: 9, # bicycle
}

STUFF_ID_TO_LOCAL_IDX = {
    0: 0,  # road
    1: 1,  # sidewalk
    2: 2,  # building
    3: 3,  # wall
    4: 4,  # fence
    5: 5,  # pole
    8: 6,  # vegetation
    9: 7,  # terrain
    10: 8, # sky
}

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

    # 굳이 TEST 데이터셋 안 써도 되지만, meta 가져오기 위해 등록해 둠
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
def run_on_image(predictor, img_bgr, meta):
    """
    infer.py 와 거의 동일한 함수:
    - panoptic_seg + segments_info 로 overlay 그리기
    - panoptic id 맵(np.uint32)도 함께 반환
    """
    outputs = predictor(img_bgr)

    if "panoptic_seg" not in outputs:
        return img_bgr.copy(), None, None

    pan_seg, segments_info = outputs["panoptic_seg"]  # pan_seg: (H,W) torch.Tensor

    # ---- Visualizer 를 위한 ID 재매핑 (infer.py 로직 그대로) ----
    mapped_segments = []
    for seg in segments_info:
        seg_new = seg.copy()
        cid = seg_new["category_id"]
        if seg_new["isthing"]:
            if cid in THING_ID_TO_LOCAL_IDX:
                seg_new["category_id"] = THING_ID_TO_LOCAL_IDX[cid]
                mapped_segments.append(seg_new)
        else:
            if cid in STUFF_ID_TO_LOCAL_IDX:
                seg_new["category_id"] = STUFF_ID_TO_LOCAL_IDX[cid]
                mapped_segments.append(seg_new)
    # ----------------------------------------------------------

    vis = Visualizer(img_bgr[:, :, ::-1], metadata=meta)  # BGR -> RGB
    vis = vis.draw_panoptic_seg_predictions(pan_seg.to("cpu"), mapped_segments)
    overlay = vis.get_image()[:, :, ::-1]  # 다시 RGB -> BGR

    pan_np = pan_seg.to("cpu").numpy().astype(np.uint32)
    return overlay, pan_np, segments_info


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--input", required=True, nargs="+",
                        help="이미지 파일/디렉토리/글롭 패턴")
    parser.add_argument("--output", required=True,
                        help="결과를 저장할 디렉토리")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = build_cfg(args)

    from demo.predictor import VisualizationDemo
    demo = VisualizationDemo(cfg)
    predictor = demo.predictor

    # meta 는 CARLA 기준이지만, 우리는 그냥 클래스 이름/색 팔레트만 쓰는 수준이라 괜찮음
    meta = MetadataCatalog.get("carla_final_panoptic_val")

    out_root = ensure_dir(args.output)
    out_overlay = ensure_dir(os.path.join(out_root, "overlay"))
    out_panpng = ensure_dir(os.path.join(out_root, "panoptic_color"))
    out_segjson = ensure_dir(os.path.join(out_root, "segments_info"))

    # ---- 입력 리스트 만들기 (파일 / 디렉토리 / 글롭 모두 지원) ----
    paths = []
    for p in args.input:
        if any(c in p for c in ["*", "?", "[" ,"]"]):
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

        overlay, pan_id, segs = run_on_image(predictor, img, meta)

        stem = os.path.splitext(os.path.basename(ip))[0]

        # overlay 저장
        cv2.imwrite(os.path.join(out_overlay, f"{stem}_overlay.png"), overlay)

        # panoptic 색상 PNG + segments_info JSON 저장
        if pan_id is not None:
            color = id2rgb(pan_id)[:, :, ::-1]  # RGB -> BGR
            cv2.imwrite(os.path.join(out_panpng, f"{stem}_panoptic.png"), color)

            with open(os.path.join(out_segjson, f"{stem}_segments.json"), "w") as f:
                json.dump({"segments_info": segs}, f, ensure_ascii=False)

        print(f"[OK] {os.path.basename(ip)}  {time.time()-t0:.2f}s")

    print(f"[DONE] results saved under {out_root}")


if __name__ == "__main__":
    main()
