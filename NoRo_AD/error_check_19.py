# error_check_carla_19.py
# -----------------------------------------------------------
# Cityscapes-19(trainIds) 평가/시각화 (CARLA/Cityscapes GT 자동 매핑)
# - GT:   <base>/TownXX/SCENARIO/gtFine/frame_***_gtFine_labelIds.png
# - RGB:  <base>/TownXX/SCENARIO/leftImg8bit/frame_***_leftImg8bit.png
# - PRED: <pred_root>/TownXX/SCENARIO/frame_***_pred_mask_19.png
# - OUT:  <out>/TownXX/SCENARIO/frame_***_combo.png
# - REPORT: 요약 TXT(기본 <out>/_metrics_summary.txt)
# -----------------------------------------------------------

import os, glob, argparse
from typing import Tuple, Optional, Set, Dict, List
import numpy as np
import cv2
from PIL import Image

VERSION = "error_check_carla_19 v2.3"
IGNORE = 255

CLASSES19 = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
PALETTE19_RGB = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
    [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
    [0,60,100],[0,80,100],[0,0,230],[119,11,32],
], dtype=np.uint8)
PALETTE19_BGR = PALETTE19_RGB[:, ::-1]
NCLS = len(CLASSES19)

# Cityscapes labelIds -> trainIds
LABELIDS_TO_TRAINIDS_BASE = {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,31:16,32:17,33:18}
# CARLA enum (new)
CARLA_IDS_TO_TRAINIDS_NEW = {1:0,24:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13,15:14,16:15,17:16,18:17,19:18}
# CARLA old fallback
CARLA_IDS_TO_TRAINIDS_OLD = {7:0,6:0,8:1,1:2,11:3,2:4,5:5,18:6,12:7,9:8,22:9,13:10,4:11,10:13}

# ---------- IO ----------
def load_id_png(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("P","L","I;16","I;16B","I"):
        arr = np.array(im)
    elif im.mode == "RGB":
        raise ValueError(f"[ERR] {path} 는 RGB PNG입니다. *_labelIds.png / ID PNG를 넣으세요.")
    else:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[...,0]
    return arr.astype(np.uint16)

def save_png(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, arr)

# ---------- 매핑 ----------
def _apply_table(arr: np.ndarray, table: dict,
                 absorb_ground: Optional[str] = None,
                 ground_id: Optional[int] = None) -> np.ndarray:
    out = np.full_like(arr, IGNORE, dtype=np.uint16)
    for k, v in table.items():
        out[arr == k] = v
    if absorb_ground in ("road","terrain") and ground_id is not None:
        tgt = 0 if absorb_ground == "road" else 9
        out[arr == ground_id] = tgt
    return out

def detect_gt_kind(uvals: Set[int]) -> str:
    if uvals <= (set(range(NCLS)) | {IGNORE}): return "trainIds"
    if max(uvals) > 28: return "city_labelIds"
    if {24,25,26,27,28} & uvals: return "carla_new"
    if {1,2,3} & uvals: return "carla_new"
    if 10 in uvals: return "carla_old"
    return "carla_new"

def map_gt_to_train19(gt: np.ndarray, *, absorb_roadlines: bool=True, absorb_ground: Optional[str]=None, debug: bool=False) -> np.ndarray:
    u = set(np.unique(gt).tolist())
    kind = detect_gt_kind(u)
    if debug: print(f"[MAP] kind={kind}, uniques(sample)={sorted(list(u))[:16]}...")
    if kind == "trainIds": return gt.astype(np.uint16)
    if kind == "city_labelIds": return _apply_table(gt, LABELIDS_TO_TRAINIDS_BASE)
    if kind == "carla_new":
        table = dict(CARLA_IDS_TO_TRAINIDS_NEW)
        if not absorb_roadlines: table.pop(24, None)
        return _apply_table(gt, table, absorb_ground=absorb_ground, ground_id=25)
    table = dict(CARLA_IDS_TO_TRAINIDS_OLD)
    if not absorb_roadlines: table.pop(6, None)
    return _apply_table(gt, table, absorb_ground=absorb_ground, ground_id=14)

def sanitize_train19(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    bad = (out != IGNORE) & ((out < 0) | (out >= NCLS))
    if np.any(bad): out[bad] = IGNORE
    return out

# ---------- 비주얼 ----------
def colorize_train19(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h,w,3), np.uint8)
    for cid in range(NCLS): out[mask == cid] = PALETTE19_BGR[cid]
    return out

def draw_label_box(img: np.ndarray, text: str, xy: Tuple[int,int]) -> None:
    x, y = int(xy[0]), int(xy[1])
    font = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.5
    (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
    pad = 4
    x0 = max(0, x - tw//2 - pad); y0 = max(0, y - th//2 - pad)
    x1 = min(img.shape[1]-1, x + tw//2 + pad); y1 = min(img.shape[0]-1, y + th//2 + pad)
    cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,255), -1)
    cv2.putText(img, text, (x0+pad, y1-pad), font, fs, (0,0,0), 1, cv2.LINE_AA)

def overlay_labels_one_per_class(mask: np.ndarray, img: np.ndarray, min_area: int=100):
    for cid in range(NCLS):
        m = (mask == cid)
        if not np.any(m): continue
        comp = m.astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(comp, connectivity=8)
        if n <= 1:
            ys, xs = np.where(m)
            if len(xs) == 0: continue
            draw_label_box(img, CLASSES19[cid], (int(np.median(xs)), int(np.median(ys))))
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        if areas[idx-1] < min_area: continue
        cx, cy = centroids[idx]
        draw_label_box(img, CLASSES19[cid], (int(cx), int(cy)))

def overlay_labels_grid(mask: np.ndarray, img: np.ndarray, step: int=96, min_cover: float=0.25, min_dom: float=0.60):
    h, w = mask.shape
    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            y0,y1 = max(0, y-step//2), min(h, y+step//2)
            x0,x1 = max(0, x-step//2), min(w, x+step//2)
            crop = mask[y0:y1, x0:x1]
            total = crop.size
            valid = (crop != IGNORE) & (crop < NCLS)
            vcnt = int(valid.sum())
            if vcnt == 0 or vcnt/total < min_cover: continue
            vals, cnts = np.unique(crop[valid], return_counts=True)
            cid = int(vals[int(np.argmax(cnts))])
            dom = int(np.max(cnts)) / float(total)
            if dom < min_dom: continue
            draw_label_box(img, CLASSES19[cid], (x, y))

def make_error_map(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    h, w = gt.shape
    err = np.zeros((h,w,3), np.uint8)
    valid_gt = (gt != IGNORE); valid_pr = (pr != IGNORE)
    same = (gt == pr) & valid_gt & valid_pr
    err[same] = (0,255,0)              # TP (green)
    err[valid_pr & (~same)] = (0,0,255)  # FP (red)
    err[valid_gt & (~same)] = (255,0,0)  # FN (blue)
    return err

def stack4(rgb: np.ndarray, pr_col: np.ndarray, gt_col: np.ndarray, err: np.ndarray) -> np.ndarray:
    cv2.putText(pr_col, "Pred(trainIds19)", (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(gt_col, "GT(trainIds19)",   (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(rgb,    "RGB",              (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
    h = min(rgb.shape[0], pr_col.shape[0], gt_col.shape[0], err.shape[0])
    return np.hstack([rgb[:h], pr_col[:h], gt_col[:h], err[:h]])

def assert_train19(name: str, arr: np.ndarray):
    u = set(np.unique(arr).tolist())
    allowed = set(range(NCLS)) | {IGNORE}
    assert u.issubset(allowed), f"{name}: 0..18,255 이외 값 존재 -> {sorted(u - allowed)}"

# ---------- metrics ----------
def iou_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    ious = np.where(denom > 0, tp / denom, np.nan)
    miou = np.nanmean(ious)
    return ious, miou

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser(VERSION + " | RGB | Pred(19) | GT(19) | Error + mIoU report")
    ap.add_argument("--base", default="/media/vip-dell/HC/_output3", help="CARLA 생성 루트(_output3)")
    ap.add_argument("--pred_root", default="", help="예측 19 루트(기본 <base>/_pred_masks_19)")
    ap.add_argument("--out", default="_viz_combo_carla_19", help="저장 루트(타운/시나리오 하위로 저장)")
    ap.add_argument("--report", default="", help="요약 TXT 저장경로(비우면 <out>/_metrics_summary.txt)")
    ap.add_argument("--pred_pattern", default="*_pred_mask_19.png", help="예측 파일 패턴")
    ap.add_argument("--towns", default="", help="쉼표구분 타운 필터 (예: Town01,Town02)")
    ap.add_argument("--scenarios", default="", help="쉼표구분 시나리오 필터")
    ap.add_argument("--first_n_per_group", type=int, default=0, help="타운×시나리오별 상위 N장만 사용 (0=제한 없음)")
    ap.add_argument("--limit", type=int, default=0, help="전체 저장 개수 제한(0=무제한)")
    ap.add_argument("--stride", type=int, default=1, help="프레임 샘플 간격(>=1)")
    # 라벨 오버레이: 기본 ON, --no-labels 로 끄기
    ap.add_argument("--labels", dest="show_labels", action="store_true")
    ap.add_argument("--no-labels", dest="show_labels", action="store_false")
    ap.set_defaults(show_labels=True)
    ap.add_argument("--label_mode", choices=["one","grid"], default="one")
    ap.add_argument("--min_area", type=int, default=150)
    ap.add_argument("--compute_cm", action="store_true", help="혼동행렬/IoU 콘솔 출력(전체)")
    ap.add_argument("--save_cm", default="", help="혼동행렬 .npy 저장 경로(옵션)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no_absorb_roadlines", action="store_true")
    ap.add_argument("--absorb_ground", choices=["none","road","terrain"], default="none")
    return ap.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    print(f"[INFO] {VERSION}")

    pred_root = args.pred_root or os.path.join(args.base, "_pred_masks_19")
    absorb_ground = None if args.absorb_ground == "none" else args.absorb_ground
    absorb_roadlines = not args.no_absorb_roadlines
    report_path = args.report or os.path.join(args.out, "_metrics_summary.txt")

    # 1) 예측 파일 기준 스캔 (예측 존재 샘플만 평가)
    pred_list = sorted(glob.glob(os.path.join(pred_root, "Town*", "*", args.pred_pattern)))
    # 필터
    if args.towns.strip():
        allow_t = set(t.strip() for t in args.towns.split(",") if t.strip())
        pred_list = [p for p in pred_list
                     if (lambda pr: len(pr)>=3 and pr[0] in allow_t)(os.path.relpath(p, pred_root).split(os.sep))]
    if args.scenarios.strip():
        allow_s = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        pred_list = [p for p in pred_list
                     if (lambda pr: len(pr)>=3 and pr[1] in allow_s)(os.path.relpath(p, pred_root).split(os.sep))]

    # 타운×시나리오별 첫 N장
    if args.first_n_per_group and args.first_n_per_group > 0:
        grouped: Dict[Tuple[str,str], List[str]] = {}
        for pp in pred_list:
            pr = os.path.relpath(pp, pred_root).split(os.sep)  # [TownXX, SCENARIO, file.png]
            if len(pr) < 3: continue
            key = (pr[0], pr[1])
            grouped.setdefault(key, []).append(pp)
        tmp = []
        for key, items in grouped.items():
            tmp.extend(sorted(items)[:args.first_n_per_group])  # frame_000000... 순서
        pred_list = tmp

    # stride/limit
    if args.stride > 1:
        pred_list = pred_list[::args.stride]
    if args.limit and args.limit > 0:
        pred_list = pred_list[:args.limit]

    # 혼동행렬들
    overall_cm = np.zeros((NCLS, NCLS), dtype=np.int64)
    group_cm: Dict[Tuple[str,str], np.ndarray] = {}   # (Town,Scenario)
    town_cm:  Dict[str, np.ndarray] = {}
    scen_cm:  Dict[str, np.ndarray] = {}

    # 카운트
    group_count: Dict[Tuple[str,str], int] = {}
    town_count:  Dict[str, int] = {}
    scen_count:  Dict[str, int] = {}

    used = 0
    for pr_path in pred_list:
        parts = os.path.relpath(pr_path, pred_root).split(os.sep)
        if len(parts) < 3: continue
        town, scen = parts[0], parts[1]
        core = os.path.basename(pr_path).replace("_pred_mask_19.png","")

        gt_path  = os.path.join(args.base, town, scen, "gtFine",      f"{core}_gtFine_labelIds.png")
        rgb_path = os.path.join(args.base, town, scen, "leftImg8bit", f"{core}_leftImg8bit.png")
        if not (os.path.exists(gt_path) and os.path.exists(rgb_path)):
            continue

        rgb = cv2.imread(rgb_path)
        if rgb is None: continue
        gt_raw = load_id_png(gt_path)
        pr19   = load_id_png(pr_path)

        gt19 = map_gt_to_train19(gt_raw, absorb_roadlines=absorb_roadlines, absorb_ground=absorb_ground, debug=args.debug)
        gt19 = sanitize_train19(gt19); pr19 = sanitize_train19(pr19)
        assert_train19("GT_19_MAPPED", gt19); assert_train19("PR_19", pr19)

        # 해상도 정합
        if pr19.shape != gt19.shape:
            pr19 = cv2.resize(pr19, (gt19.shape[1], gt19.shape[0]), interpolation=cv2.INTER_NEAREST)
        if rgb.shape[:2] != gt19.shape:
            rgb  = cv2.resize(rgb,  (gt19.shape[1], gt19.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 비주얼(라벨 오버레이 기본 ON)
        pr_col = colorize_train19(pr19)
        gt_col = colorize_train19(gt19)
        if args.show_labels:
            if args.label_mode == "one":
                overlay_labels_one_per_class(pr19, pr_col, min_area=args.min_area)
                overlay_labels_one_per_class(gt19, gt_col, min_area=args.min_area)
            else:
                overlay_labels_grid(pr19, pr_col)
                overlay_labels_grid(gt19, gt_col)

        err = make_error_map(gt19, pr19)
        combo = stack4(rgb.copy(), pr_col, gt_col, err)

        out_dir = os.path.join(args.out, town, scen)
        os.makedirs(out_dir, exist_ok=True)
        save_png(os.path.join(out_dir, f"{core}_combo.png"), combo)
        used += 1

        # 혼동행렬 누적
        m = (gt19 != IGNORE) & (pr19 != IGNORE)
        if np.any(m):
            idx = gt19[m] * NCLS + pr19[m]
            binc = np.bincount(idx, minlength=NCLS*NCLS)
            mat = binc.reshape(NCLS, NCLS)

            overall_cm += mat
            group_cm.setdefault((town,scen), np.zeros((NCLS,NCLS), np.int64))[:] += mat
            town_cm.setdefault(town, np.zeros((NCLS,NCLS), np.int64))[:] += mat
            scen_cm.setdefault(scen, np.zeros((NCLS,NCLS), np.int64))[:] += mat

            group_count[(town,scen)] = group_count.get((town,scen), 0) + 1
            town_count[town] = town_count.get(town, 0) + 1
            scen_count[scen] = scen_count.get(scen, 0) + 1

    print(f"[DONE] {used} imgs saved to {os.path.abspath(args.out)}")

    # 콘솔 출력(선택)
    if args.compute_cm:
        ious, miou = iou_from_cm(overall_cm)
        print("\n[IoU per class] (overall)")
        for i, (cls, iou) in enumerate(zip(CLASSES19, ious)):
            val = float(iou) if not np.isnan(iou) else float('nan')
            print(f"{i:2d} {cls:14s}: {val:.4f}")
        print(f"[mIoU] overall: {miou:.4f}")

    if args.save_cm:
        np.save(args.save_cm, overall_cm)
        print(f"[SAVED] confusion matrix -> {os.path.abspath(args.save_cm)}")

    # TXT 리포트 저장
    os.makedirs(args.out, exist_ok=True)
    lines: List[str] = []
    lines.append(f"{VERSION}")
    lines.append(f"Base={os.path.abspath(args.base)}")
    lines.append(f"PredRoot={os.path.abspath(pred_root)}")
    lines.append(f"OutDir={os.path.abspath(args.out)}")
    lines.append(f"Used images={used}")
    lines.append("")

    def fmt(x):
        return "nan" if np.isnan(x) else f"{x:.4f}"

    # 1) Overall
    ious_all, miou_all = iou_from_cm(overall_cm)
    lines.append("[OVERALL]")
    lines.append(f"mIoU: {fmt(miou_all)}")
    lines.append("IoU per class:")
    for i, (cls, v) in enumerate(zip(CLASSES19, ious_all)):
        lines.append(f"  {i:2d} {cls:14s}: {fmt(float(v))}")
    lines.append("")

    # 2) Town x Scenario
    lines.append("[PER TOWN x SCENARIO]")
    for (town, scen) in sorted(group_cm.keys()):
        ious_g, miou_g = iou_from_cm(group_cm[(town,scen)])
        cnt = group_count.get((town,scen), 0)
        lines.append(f"{town}/{scen}: mIoU={fmt(miou_g)}  (imgs={cnt})")
    lines.append("")

    # 3) Town aggregated
    lines.append("[PER TOWN]")
    for town in sorted(town_cm.keys()):
        ious_t, miou_t = iou_from_cm(town_cm[town])
        cnt = town_count.get(town, 0)
        lines.append(f"{town}: mIoU={fmt(miou_t)}  (imgs={cnt})")
    lines.append("")

    # 4) Scenario aggregated
    lines.append("[PER SCENARIO]")
    for scen in sorted(scen_cm.keys()):
        ious_s, miou_s = iou_from_cm(scen_cm[scen])
        cnt = scen_count.get(scen, 0)
        lines.append(f"{scen}: mIoU={fmt(miou_s)}  (imgs={cnt})")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] report -> {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()
