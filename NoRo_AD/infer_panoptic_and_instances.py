# _output4 데이터셋에 대해 Mask2Former panoptic 모델로 추론 수행 (ETA 표시 추가)

import os, glob, argparse, re
import cv2, torch
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from mask2former import add_maskformer2_config

IGNORE = 255
THING_TRAINIDS = set([11,12,13,14,15,16,17,18,6,7])  # person~bicycle + tl,ts

def add_mask2former_panoptic_cfg(cfg):
    add_maskformer2_config(cfg)

def build_predictor(cfg_yaml: str, weights_path: str, device: str = "cuda"):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_mask2former_panoptic_cfg(cfg)
    cfg.merge_from_file(cfg_yaml)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    # 반드시 panoptic 모드
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    return DefaultPredictor(cfg)

def save_png16(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, arr.astype(np.uint16))

def frame_index_from_path(p: str):
    m = re.search(r'frame_(\d+)_leftImg8bit\.png$', p)
    return int(m.group(1)) if m else None

def parse_args():
    ap = argparse.ArgumentParser("Panoptic(+Instance) inference on CARLA _output4 (with ETA)")
    ap.add_argument("--base", required=True, help="입력 루트 (예: /media/vip-dell/HC/_output4)")
    ap.add_argument("--pred_root", default="", help="출력 루트(비우면 base/_pred_panoptic)")
    ap.add_argument("--cfg", required=True, help="Mask2Former panoptic cfg yaml")
    ap.add_argument("--weights", required=True, help=".pth")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--towns", type=str, default="")
    ap.add_argument("--scenarios", type=str, default="")
    ap.add_argument("--index_min", type=int, default=None)
    ap.add_argument("--index_max", type=int, default=None)
    ap.add_argument("--first_k_per_group", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    base = args.base
    pred_root = args.pred_root or os.path.join(base, "_pred_panoptic")
    out_pan = pred_root                             # panoptic PNG 저장
    out_inst = os.path.join(base, "_pred_instances")# instance PNG 저장
    os.makedirs(out_pan, exist_ok=True)
    os.makedirs(out_inst, exist_ok=True)

    predictor = build_predictor(args.cfg, args.weights, device=args.device)

    pattern = os.path.join(base, "Town*", "*", "leftImg8bit", "frame_*_leftImg8bit.png")
    all_imgs = sorted(glob.glob(pattern))

    # 타운/시나리오 필터
    if args.towns.strip():
        allow_t = set(t.strip() for t in args.towns.split(",") if t.strip())
        all_imgs = [p for p in all_imgs
                    if (lambda pr: len(pr)>=4 and pr[0] in allow_t)(os.path.relpath(p, base).split(os.sep))]
    if args.scenarios.strip():
        allow_s = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        all_imgs = [p for p in all_imgs
                    if (lambda pr: len(pr)>=4 and pr[1] in allow_s)(os.path.relpath(p, base).split(os.sep))]

    items = []
    for p in all_imgs:
        idx = frame_index_from_path(p)
        if idx is None: continue
        if args.index_min is not None and idx < args.index_min: continue
        if args.index_max is not None and idx > args.index_max: continue
        parts = os.path.relpath(p, base).split(os.sep)
        if len(parts) < 4: continue
        town, scen = parts[0], parts[1]
        items.append(((town, scen), idx, p))

    if args.first_k_per_group and args.first_k_per_group > 0:
        grouped = {}
        for key, idx, path in items:
            grouped.setdefault(key, []).append((idx, path))
        selected = []
        for key, lst in grouped.items():
            lst.sort(key=lambda x: x[0])
            take = lst[:args.first_k_per_group]
            selected.extend([(key, i, p) for (i, p) in take])
        items = selected
    else:
        items.sort(key=lambda x: (x[0][0], x[0][1], x[1]))

    paths = [p for (_,_,p) in items]
    if args.stride > 1: paths = paths[::args.stride]
    if args.limit  > 0: paths = paths[:args.limit]

    print(f"[INFO] 대상 이미지: {len(paths)}장")
    if not paths: return

    with torch.no_grad():
        # tqdm: ETA, 평균 속도, 진행률 표시
        for ip in tqdm(paths, ncols=100, unit="img", desc="Infer", leave=True):
            img = cv2.imread(ip)
            if img is None:
                continue
            out = predictor(img)

            # Detectron2 panoptic 결과: dict with "panoptic_seg"=(H,W), segments_info
            pan_seg, segments = out["panoptic_seg"]
            pan_seg = pan_seg.cpu().numpy().astype(np.int32)

            # 각 segment 처리
            h, w = pan_seg.shape
            pan_png = np.zeros((h, w), np.uint16)
            inst_png = np.zeros((h, w), np.uint16)
            per_class_counter = {}

            for seg in segments:
                seg_id = int(seg["id"])
                clsid = int(seg["category_id"])
                isthing = bool(seg.get("isthing", True))
                mask = (pan_seg == seg_id)

                train_id = clsid
                if train_id == IGNORE:
                    continue

                if isthing and (train_id in THING_TRAINIDS):
                    k = per_class_counter.get(train_id, 0) + 1
                    per_class_counter[train_id] = k
                    inst_id = min(k, 999)
                else:
                    inst_id = 0

                pan_val = int(train_id) * 1000 + inst_id
                pan_png[mask] = pan_val

                if inst_id > 0:
                    inst_png[mask] = inst_id

            parts = os.path.relpath(ip, base).split(os.sep)
            town, scen = parts[0], parts[1]
            fn_core = os.path.basename(ip).replace("_leftImg8bit.png", "")

            save_png16(os.path.join(out_pan, town, scen, f"{fn_core}_pred_panopticId.png"), pan_png)
            save_png16(os.path.join(out_inst, town, scen, f"{fn_core}_pred_instanceId.png"), inst_png)

if __name__ == "__main__":
    main()
