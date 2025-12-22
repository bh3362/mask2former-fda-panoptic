# infer_save_all.py
# ---------------------------------------------------------------
# CARLA _output3 구조용 인퍼런스 스크립트
# - 입력: /media/<USER>/<DISK>/_output3/TownXX/SCENARIO/leftImg8bit/frame_***_leftImg8bit.png
# - 출력(기본): /media/<USER>/<DISK>/_output3/_pred_masks_19/TownXX/SCENARIO/*.png
# - 예측 마스크: Cityscapes trainIds(0..18, 255=IGNORE) 기준
# - 옵션: city11 매핑본 저장, 컬러 프리뷰 저장
# - 추가: 인덱스 범위(--index_min/max), 그룹별 앞 K장(--first_k_per_group)
# ---------------------------------------------------------------

import os, glob, argparse, re
import cv2, torch
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from mask2former import add_maskformer2_config

IGNORE_ID = 255

# (선택) Cityscapes-19 -> 11 매핑
CITY19_TO_11 = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10}

# Cityscapes-19 팔레트 (BGR)
PALETTE19_BGR = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170,  30], [220, 220,   0], [107, 142,  35], [152, 251, 152],
    [ 70, 130, 180], [220,  20,  60], [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
    [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)[:, ::-1]

def city19_to_11(arr: np.ndarray, ignore: int = IGNORE_ID) -> np.ndarray:
    out = np.full_like(arr, ignore, dtype=np.uint16)
    for s, d in CITY19_TO_11.items():
        out[arr == s] = d
    return out

def colorize_train19(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), np.uint8)
    for cid in range(19):
        out[mask == cid] = PALETTE19_BGR[cid]
    return out

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

def build_predictor(cfg_yaml: str, weights_path: str, device: str = "cuda"):
    cfg = get_cfg()
    # 일부 yaml 충돌 방지용 dummy
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

    add_fan_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_yaml)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    return DefaultPredictor(cfg)

def save_id_png(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, arr.astype(np.uint16))

def save_color_png(path: str, img_bgr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True
    )
    cv2.imwrite(path, img_bgr)

def frame_index_from_path(p: str):
    m = re.search(r'frame_(\d+)_leftImg8bit\.png$', p)
    return int(m.group(1)) if m else None

def parse_args():
    ap = argparse.ArgumentParser("Infer & save Cityscapes-19 masks on CARLA _output3")
    ap.add_argument("--base", default="/media/vip-dell/HC/_output3", help="입력 루트(_output3)")
    ap.add_argument("--pred_root", default="", help="출력 루트(비우면 base/_pred_masks_19)")
    ap.add_argument("--cfg",  required=True, help="Detectron2/Mask2Former cfg yaml 경로")
    ap.add_argument("--weights", required=True, help="가중치 경로 (.pth)")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--limit", type=int, default=0, help="0=전체, >0=상위 N장만(전역)")
    ap.add_argument("--stride", type=int, default=1, help="프레임 샘플 간격(>=1)")
    ap.add_argument("--scenarios", type=str, default="", help="쉼표구분 시나리오 필터 (비우면 전체)")
    ap.add_argument("--towns", type=str, default="", help="쉼표구분 타운 필터 (비우면 전체)")
    ap.add_argument("--save_city11", action="store_true", help="Cityscapes-11 매핑본도 저장")
    ap.add_argument("--save_color", action="store_true", help="paletted color 미리보기 PNG도 저장")
    # 추가 옵션
    ap.add_argument("--index_min", type=int, default=None, help="프레임 인덱스 하한(예: 0)")
    ap.add_argument("--index_max", type=int, default=None, help="프레임 인덱스 상한(예: 19)")
    ap.add_argument("--first_k_per_group", type=int, default=0,
                    help="각 (Town,Scenario) 그룹에서 앞 K장만 선택(파일명 인덱스 기준). 0=비활성")
    return ap.parse_args()

def main():
    args = parse_args()

    base = args.base
    pred_root = args.pred_root or os.path.join(base, "_pred_masks_19")
    os.makedirs(pred_root, exist_ok=True)

    predictor = build_predictor(args.cfg, args.weights, device=args.device)
    input_format = str(getattr(predictor.cfg.INPUT, "FORMAT", "BGR")).upper()

    # 입력 이미지 수집
    pattern = os.path.join(base, "Town*", "*", "leftImg8bit", "frame_*_leftImg8bit.png")
    all_imgs = sorted(glob.glob(pattern))

    # 타운/시나리오 필터
    if args.towns.strip():
        allow_t = set(t.strip() for t in args.towns.split(",") if t.strip())
        all_imgs = [p for p in all_imgs
                    if (lambda pr: len(pr)>=4 and pr[0] in allow_t)(
                        os.path.relpath(p, base).split(os.sep)
                    )]
    if args.scenarios.strip():
        allow_s = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        all_imgs = [p for p in all_imgs
                    if (lambda pr: len(pr)>=4 and pr[1] in allow_s)(
                        os.path.relpath(p, base).split(os.sep)
                    )]

    # 인덱스 부여 + 범위 필터
    items = []
    for p in all_imgs:
        idx = frame_index_from_path(p)
        if idx is None:
            continue
        if args.index_min is not None and idx < args.index_min:
            continue
        if args.index_max is not None and idx > args.index_max:
            continue
        parts = os.path.relpath(p, base).split(os.sep)
        if len(parts) < 4:  # Town/Scenario/leftImg8bit/...
            continue
        town, scen = parts[0], parts[1]
        items.append(((town, scen), idx, p))

    # 그룹별 앞 K장 선택 (옵션)
    if args.first_k_per_group and args.first_k_per_group > 0:
        grouped = {}
        for key, idx, path in items:
            grouped.setdefault(key, []).append((idx, path))
        selected = []
        for key, lst in grouped.items():
            lst.sort(key=lambda x: x[0])  # 인덱스 오름차순
            take = lst[:args.first_k_per_group]
            selected.extend([(key, i, p) for (i, p) in take])
        items = selected
    else:
        # 인덱스 기준으로 전체 정렬 (안전 차원)
        items.sort(key=lambda x: (x[0][0], x[0][1], x[1]))

    # stride / limit 적용
    paths = [p for (_, _, p) in items]
    if args.stride > 1:
        paths = paths[::args.stride]
    if args.limit > 0:
        paths = paths[:args.limit]

    print(f"[INFO] 대상 이미지: {len(paths)}장  (base={base})")
    if not paths:
        return

    with torch.no_grad():
        for ip in tqdm(paths, ncols=90):
            img = cv2.imread(ip)  # BGR
            if img is None:
                continue
            in_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if input_format == "RGB" else img

            out = predictor(in_img)
            pred19 = out["sem_seg"].argmax(0).cpu().numpy().astype(np.uint16)

            parts = os.path.relpath(ip, base).split(os.sep)
            if len(parts) < 4 or parts[2] != "leftImg8bit":
                continue
            town, scen = parts[0], parts[1]
            fn_core = os.path.basename(ip).replace("_leftImg8bit.png", "")

            out_dir_19 = os.path.join(pred_root, town, scen)
            os.makedirs(out_dir_19, exist_ok=True)
            save_id_png(os.path.join(out_dir_19, f"{fn_core}_pred_mask_19.png"), pred19)

            if args.save_city11:
                pred11 = city19_to_11(pred19)
                out_dir_11 = os.path.join(os.path.dirname(pred_root), "_pred_masks_city11", town, scen)
                os.makedirs(out_dir_11, exist_ok=True)
                save_id_png(os.path.join(out_dir_11, f"{fn_core}_pred_mask_city11.png"), pred11)

            if args.save_color:
                color = colorize_train19(pred19)
                out_dir_col = os.path.join(os.path.dirname(pred_root), "_pred_color_19", town, scen)
                os.makedirs(out_dir_col, exist_ok=True)
                save_color_png(os.path.join(out_dir_col, f"{fn_core}_pred_color_19.png"), color)

if __name__ == "__main__":
    main()
