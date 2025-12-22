# infer_instances_only.py (fixed & filtered)
# CARLA _output3에서 "인스턴스 결과만" 저장 (파놉틱 미계산/미저장)
import os, glob, argparse, json, re
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import torch

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.engine import DefaultPredictor
from mask2former import add_maskformer2_config

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None

# ---------- helpers ----------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def frame_index_from_path(p: str):
    m = re.search(r'frame_(\d+)_leftImg8bit\.png$', p)
    return int(m.group(1)) if m else None

def encode_binary_mask(bin_mask: np.ndarray):
    if mask_utils is None: return None
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def clamp_box_xyxy(box, w, h):
    x1,y1,x2,y2 = box
    x1 = max(0.0, min(float(x1), w - 1.0))
    y1 = max(0.0, min(float(y1), h - 1.0))
    x2 = max(0.0, min(float(x2), w - 1.0))
    y2 = max(0.0, min(float(y2), h - 1.0))
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1
    return [x1,y1,x2,y2]

def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def add_fan_config(cfg: CN):
    cfg.MODEL.FAN = CN()
    cfg.MODEL.FAN.PATCH_SIZE = 4
    cfg.MODEL.FAN.IN_CHANS = 3
    cfg.MODEL.FAN.NUM_CLASSES = 19
    cfg.MODEL.FAN.EMBED_DIM = 480
    cfg.MODEL.FAN.DEPTH = 22
    cfg.MODEL.FAN.OUT_IDX = 18
    cfg.MODEL.FAN.NUM_HEADS = 10
    cfg.MODEL.FAN.MLP_RATIO = 4.0
    cfg.MODEL.FAN.DROP_RATE = 0.0
    cfg.MODEL.FAN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.FAN.DROP_PATH_RATE = 0.1
    cfg.MODEL.FAN.CLS_ATTN_LAYERS = 2
    cfg.MODEL.FAN.ETA = 1.0
    cfg.MODEL.FAN.OUT_FEATURES = ["res2","res3","res4","res5"]

def add_resnets_compat(cfg: CN):
    cfg.MODEL.RESNETS = CN()
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STEM_TYPE = ""
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1,1,1]
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.NUM_GROUPS = 1
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    cfg.MODEL.RESNETS.BOTTLENECK_CHANNELS = 64
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2","res3","res4","res5"]

def build_predictor(cfg_yaml: str, weights_path: str, device: str = "cuda") -> DefaultPredictor:
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_fan_config(cfg)
    add_resnets_compat(cfg)

    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_yaml)
    cfg.set_new_allowed(False)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 인스턴스만 ON
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    return DefaultPredictor(cfg)

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser("Instance-only inference on CARLA (_output3)")
    ap.add_argument("--base", default="/media/vip-dell/HC/_output3")
    ap.add_argument("--cfg",  required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--out_inst", default="", help="기본 <base>/_pred_instances")
    ap.add_argument("--towns", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--first_k_per_group", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score_thresh", type=float, default=0.0, help="저장 최소 score")
    ap.add_argument("--min_mask_area", type=int, default=0, help="마스크 면적(px) 미만 버림")
    ap.add_argument("--min_box_area", type=int, default=0, help="bbox 면적(px) 미만 버림")
    ap.add_argument("--use_mask_bbox", action="store_true", help="마스크 bbox를 저장 bbox로 사용")
    return ap.parse_args()

def main():
    args = parse_args()
    base = args.base
    out_inst = args.out_inst or os.path.join(base, "_pred_instances")
    ensure_dir(out_inst)

    predictor = build_predictor(args.cfg, args.weights, args.device)
    input_format = str(getattr(predictor.cfg.INPUT, "FORMAT", "BGR")).upper()

    pattern = os.path.join(base, "Town*", "*", "leftImg8bit", "frame_*_leftImg8bit.png")
    all_imgs = sorted(glob.glob(pattern))

    if args.towns.strip():
        allow_t = set(t.strip() for t in args.towns.split(",") if t.strip())
        all_imgs = [p for p in all_imgs if os.path.relpath(p, base).split(os.sep)[0] in allow_t]
    if args.scenarios.strip():
        allow_s = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        all_imgs = [p for p in all_imgs if os.path.relpath(p, base).split(os.sep)[1] in allow_s]

    groups: Dict[Tuple[str,str], List[Tuple[int,str]]] = {}
    for ip in all_imgs:
        parts = os.path.relpath(ip, base).split(os.sep)
        if len(parts) < 4: continue
        town, scen = parts[0], parts[1]
        idx = frame_index_from_path(ip)
        if idx is None: continue
        groups.setdefault((town,scen), []).append((idx, ip))

    selected_paths: List[str] = []
    k = max(0, int(args.first_k_per_group))
    for key, lst in groups.items():
        lst.sort(key=lambda x: x[0])
        take = lst if k == 0 else lst[:k]
        selected_paths.extend([p for _, p in take])

    if args.stride > 1: selected_paths = selected_paths[::args.stride]
    if args.limit and args.limit > 0: selected_paths = selected_paths[:args.limit]

    print(f"[INFO] 대상 이미지: {len(selected_paths)}장 (instance-only)")

    torch.backends.cudnn.benchmark = True
    autocast_device = "cuda" if predictor.cfg.MODEL.DEVICE == "cuda" else "cpu"

    with torch.inference_mode(), torch.autocast(device_type=autocast_device, enabled=(autocast_device=="cuda")):
        for ip in selected_paths:
            img_bgr = cv2.imread(ip)
            if img_bgr is None: continue
            H, W = img_bgr.shape[:2]
            img_in = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if input_format == "RGB" else img_bgr

            outputs = predictor(img_in)
            if "instances" not in outputs: continue

            inst = outputs["instances"].to("cpu")
            boxes = inst.pred_boxes.tensor.numpy().tolist() if inst.has("pred_boxes") else []
            classes = inst.pred_classes.numpy().tolist() if inst.has("pred_classes") else []
            scores = inst.scores.numpy().tolist() if inst.has("scores") else []

            # 마스크 → RLE
            masks_rle = []
            if inst.has("pred_masks") and mask_utils is not None:
                pm = inst.pred_masks.numpy()  # [N,H,W]
                for i in range(pm.shape[0]):
                    masks_rle.append(encode_binary_mask(pm[i]))
            elif inst.has("pred_masks"):
                masks_rle = [None]*len(classes)

            # 경로
            parts = os.path.relpath(ip, base).split(os.sep)
            town, scen = parts[0], parts[1]
            stem = os.path.basename(ip).replace("_leftImg8bit.png", "")

            # 레코드 작성 (bbox clamp + 점수/면적 필터)
            inst_recs = []
            for i in range(len(classes)):
                sc = float(scores[i]) if scores else 1.0
                if sc < args.score_thresh:
                    continue

                # 기본 bbox는 모델 bbox
                xyxy = clamp_box_xyxy(boxes[i], W, H)
                bw = max(0.0, xyxy[2]-xyxy[0]); bh = max(0.0, xyxy[3]-xyxy[1])

                # 마스크가 있고 옵션 켠 경우, 마스크 bbox로 대체
                rle = None
                if i < len(masks_rle) and masks_rle[i] is not None:
                    rle = masks_rle[i]
                    if args.min_mask_area > 0:
                        try:
                            if float(mask_utils.area(rle)) < args.min_mask_area:
                                continue
                        except Exception:
                            pass
                    if args.use_mask_bbox:
                        try:
                            mb = mask_utils.toBbox(rle).tolist()  # [x,y,w,h]
                            xyxy = [mb[0], mb[1], mb[0]+mb[2], mb[1]+mb[3]]
                            xyxy = clamp_box_xyxy(xyxy, W, H)
                            bw = max(0.0, xyxy[2]-xyxy[0]); bh = max(0.0, xyxy[3]-xyxy[1])
                        except Exception:
                            pass

                if args.min_box_area > 0 and (bw*bh) < args.min_box_area:
                    continue

                rec = {
                    "bbox_xyxy": xyxy,
                    "category_id": int(classes[i]),
                    "score": sc,
                }
                if rle is not None:
                    rec["segmentation"] = rle
                inst_recs.append(rec)

            # 저장(이미지별 1 JSON)
            inst_dir = os.path.join(out_inst, town, scen)
            ensure_dir(inst_dir)
            save_json(os.path.join(inst_dir, f"{stem}_instances_pred.json"), {"instances": inst_recs})

    print("[DONE] Instance-only inference finished.]")

if __name__ == "__main__":
    main()
