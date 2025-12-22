# new_matrix_0815.py
# Cityscapes-11 평가 (CARLA GT / City19 or City11 preds 호환)
# - GT: *_gtFine_labelIds.png  -> City11로 사후 매핑
# - PR: 우선 _pred_masks_city11/**/frame_*_pred_mask_city11.png 사용
#       없으면 _pred_masks_19/**/frame_*_pred_mask_19.png 읽어 City19->City11 변환
# - 해상도 불일치 시 NEAREST로 정합
# - 출력: 전체 IoU/mIoU + 시나리오별 mIoU

import os
import glob
import argparse
import numpy as np
import cv2

# -------------------------------
# 설정
# -------------------------------
IGNORE = 255
NCLS = 11
CLASSES = [
    "road","sidewalk","building","wall","fence",
    "pole","traffic light","traffic sign","vegetation","terrain","sky"
]

# CARLA raw labelIds -> City11
CARLA2CITY11 = {7:0, 8:1, 10:2, 12:3, 13:4, 14:5, 18:6, 19:7, 20:8, 21:9, 22:10}

# Cityscapes 19 trainIds -> City11 (thing 11~18은 ignore)
CITY19_TO_11 = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10}


# -------------------------------
# 유틸
# -------------------------------
def load_id(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def carla_to_11(arr: np.ndarray, ignore: int = IGNORE) -> np.ndarray:
    out = np.full_like(arr, ignore)
    for s, d in CARLA2CITY11.items():
        out[arr == s] = d
    return out

def city19_to_11(arr: np.ndarray, ignore: int = IGNORE) -> np.ndarray:
    out = np.full_like(arr, ignore)
    for s, d in CITY19_TO_11.items():
        out[arr == s] = d
    return out

def sanitize_11(arr: np.ndarray, name="arr") -> np.ndarray:
    """0~10과 255만 남도록 강제 정리 + 범위 밖 값 한 번 경고"""
    a = arr.copy()
    bad_mask = (a != IGNORE) & ((a < 0) | (a >= NCLS))
    if np.any(bad_mask):
        bad_vals = np.unique(a[bad_mask])
        print(f"[WARN] {name}에 City11 범위 밖 값 발견:", bad_vals[:20], "…")
        a[bad_mask] = IGNORE
    return a

def confmat(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    gt = sanitize_11(gt, "gt")
    pr = sanitize_11(pr, "pr")
    mask = (gt != IGNORE) & (pr != IGNORE)
    if not np.any(mask):
        return np.zeros((NCLS, NCLS), dtype=np.int64)
    a = gt[mask].ravel().astype(np.int64)
    b = pr[mask].ravel().astype(np.int64)
    idx = NCLS * a + b  # 0..120
    h = np.bincount(idx, minlength=NCLS * NCLS).reshape(NCLS, NCLS)
    return h

def iou_from_conf(C: np.ndarray) -> np.ndarray:
    TP = np.diag(C)
    FP = C.sum(axis=0) - TP
    FN = C.sum(axis=1) - TP
    denom = TP + FP + FN
    iou = np.where(denom > 0, TP / denom, 0.0)
    return iou


# -------------------------------
# 메인
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate Cityscapes-11 (CARLA GT vs preds)")
    ap.add_argument("--base",
        default="/home/vip-dell/CARLA_0.9.15/PythonAPI/examples/_output2",
        help="원본 RGB/GT 상위 폴더 (scenario/gtFine, scenario/leftImg8bit 포함)")
    ap.add_argument("--gt_pattern", default="*_gtFine_labelIds.png",
        help="GT 파일 패턴")
    ap.add_argument("--pred_city11_root", default=None,
        help="(선택) City11 예측 폴더 (기본: <base>/_pred_masks_city11)")
    ap.add_argument("--pred_city19_root", default=None,
        help="(선택) City19 예측 폴더 (기본: <base>/_pred_masks_19)")
    ap.add_argument("--no_scenario_breakdown", action="store_true",
        help="시나리오별 mIoU 출력 생략")
    return ap.parse_args()

def main():
    args = parse_args()
    base = args.base
    pr11_root = args.pred_city11_root or os.path.join(base, "_pred_masks_city11")
    pr19_root = args.pred_city19_root or os.path.join(base, "_pred_masks_19")

    # GT 리스트 수집
    gt_files = sorted(glob.glob(os.path.join(base, "**", "gtFine", args.gt_pattern), recursive=True))
    if not gt_files:
        print("[WARN] GT 파일을 찾지 못했습니다.")
        return

    C_overall = np.zeros((NCLS, NCLS), dtype=np.int64)
    per_scn = {}

    miss_cnt = 0
    use_pr11_cnt = 0
    use_pr19_cnt = 0

    for g in gt_files:
        # 시나리오 및 frame 코어명
        parts = g.split(os.sep)
        # .../<scenario>/gtFine/frame_xxxxxx_gtFine_labelIds.png
        try:
            scenario = parts[-3]
        except IndexError:
            scenario = "unknown"

        core = os.path.basename(g).split("_gtFine_")[0]  # frame_xxxxxx

        # 우선 City11 예측을 찾고, 없으면 City19 예측을 찾아 변환
        pr11 = os.path.join(pr11_root, scenario, f"{core}_pred_mask_city11.png")
        pr19 = os.path.join(pr19_root, scenario, f"{core}_pred_mask_19.png")

        pred = None
        if os.path.exists(pr11):
            pred = load_id(pr11)
            use_pr11_cnt += 1
        elif os.path.exists(pr19):
            pred = city19_to_11(load_id(pr19))
            use_pr19_cnt += 1
        else:
            miss_cnt += 1
            continue

        # GT 로드 및 City11 매핑
        gt = carla_to_11(load_id(g))

        # 해상도 정합 (NEAREST)
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 혼동행렬 누적
        CM = confmat(gt, pred)
        C_overall += CM
        if scenario not in per_scn:
            per_scn[scenario] = np.zeros((NCLS, NCLS), dtype=np.int64)
        per_scn[scenario] += CM

    # 결과 출력
    print(f"\n[INFO] 총 GT 파일: {len(gt_files)} | 사용 PR(11): {use_pr11_cnt} | 사용 PR(19->11): {use_pr19_cnt} | 매칭 실패: {miss_cnt}")

    iou = iou_from_conf(C_overall)
    miou = float(iou.mean())

    print("\n[IoU by class] (overall)")
    for k, v in enumerate(iou):
        print(f"{k:02d} {CLASSES[k]:<14}: {v:.4f}")
    print(f"\n[mIoU overall]: {miou:.4f}")

    if not args.no_scenario_breakdown and per_scn:
        print("\n[Per-scenario mIoU]")
        for scn in sorted(per_scn.keys()):
            i = iou_from_conf(per_scn[scn])
            print(f"{scn:<20}: {float(i.mean()):.4f}")

if __name__ == "__main__":
    main()
