# check_semantic_performance.py
# ---------------------------------------------------------------
# Cityscapes-19 시맨틱 성능 평가 (mIoU + per-class IoU)
# - GT:   /BASE/TownXX/SCENARIO/gtFine/*_gtFine_trainIds19.png
# - Pred: /BASE/_pred_masks_19/TownXX/SCENARIO/*_pred_mask_19.png
# - 옵션: 타운/시나리오 필터, 그룹당 앞 K장만 평가, 결과 파일 저장
# ---------------------------------------------------------------

import os, argparse, re, glob
import numpy as np
import cv2
from tqdm import tqdm

IGNORE = 255
NCLS = 19
CLASS_NAMES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]

def read_png16(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img if img is not None else None

def fast_confusion_matrix(gt, pred, ncls=NCLS, ignore=IGNORE):
    mask = (gt != ignore)
    gt_v = gt[mask]
    pr_v = pred[mask]
    if gt_v.size == 0:
        return np.zeros((ncls, ncls), dtype=np.int64)
    gt_v = np.clip(gt_v, 0, ncls-1)
    pr_v = np.clip(pr_v, 0, ncls-1)
    cm = np.bincount(gt_v * ncls + pr_v, minlength=ncls*ncls).reshape(ncls, ncls)
    return cm

def miou_from_confmat(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, 0.0)
    valid = denom > 0
    miou = iou[valid].mean() if valid.any() else 0.0
    return miou, iou

def frame_index_from_path(p: str):
    m = re.search(r'frame_(\d+)_', os.path.basename(p))
    return int(m.group(1)) if m else None

def parse_args():
    ap = argparse.ArgumentParser("Semantic mIoU evaluator for _pred_masks_19")
    ap.add_argument("--base", required=True, help="데이터 루트 (예: /media/vip-dell/HC/_output4)")
    ap.add_argument("--pred_root", default="", help="예측 루트(기본: base/_pred_masks_19)")
    ap.add_argument("--towns", type=str, default="", help="필터: 쉼표구분 (비우면 전체)")
    ap.add_argument("--scenarios", type=str, default="", help="필터: 쉼표구분 (비우면 전체)")
    ap.add_argument("--first_k_per_group", type=int, default=0, help="그룹별 앞 K장만 평가 (0=전체)")
    ap.add_argument("--save_txt", default="semantic_eval_summary.txt", help="요약 저장 파일명")
    ap.add_argument("--save_tsv", default="semantic_eval_details.tsv", help="상세 TSV 파일명")
    return ap.parse_args()

def main():
    args = parse_args()
    base = args.base
    pred_root = args.pred_root or os.path.join(base, "_pred_masks_19")

    towns = sorted([d for d in os.listdir(base) if d.startswith("Town") and os.path.isdir(os.path.join(base, d))])
    if args.towns.strip():
        allow_t = set(t.strip() for t in args.towns.split(",") if t.strip())
        towns = [t for t in towns if t in allow_t]

    lines = []
    detail_rows = []  # TSV: town,scenario,miou, then per-class IoUs

    for town in towns:
        scen_dirs = sorted([s for s in os.listdir(os.path.join(base, town))
                            if os.path.isdir(os.path.join(base, town, s))])
        if args.scenarios.strip():
            allow_s = set(s.strip() for s in args.scenarios.split(",") if s.strip())
            scen_dirs = [s for s in scen_dirs if s in allow_s]

        for scen in scen_dirs:
            gt_dir   = os.path.join(base, town, scen, "gtFine")
            pred_dir = os.path.join(pred_root, town, scen)
            if not (os.path.isdir(gt_dir) and os.path.isdir(pred_dir)):
                continue

            gt_files = sorted(glob.glob(os.path.join(gt_dir, "frame_*_gtFine_trainIds19.png")))
            # pred 파일 존재로 필터
            pairs = []
            for gt_path in gt_files:
                fn_core = os.path.basename(gt_path).replace("_gtFine_trainIds19.png", "")
                pred_path = os.path.join(pred_dir, f"{fn_core}_pred_mask_19.png")
                if os.path.isfile(pred_path):
                    idx = frame_index_from_path(gt_path)
                    pairs.append((idx, gt_path, pred_path))

            if not pairs:
                continue

            # 그룹별 앞 K장
            pairs.sort(key=lambda x:(x[0] if x[0] is not None else 1e12))
            if args.first_k_per_group and args.first_k_per_group > 0:
                pairs = pairs[:args.first_k_per_group]

            confmat = np.zeros((NCLS, NCLS), dtype=np.int64)
            for _, gt_p, pr_p in tqdm(pairs, desc=f"{town}/{scen}", ncols=90):
                gt = read_png16(gt_p)
                pred = read_png16(pr_p)
                if gt is None or pred is None:
                    continue
                confmat += fast_confusion_matrix(gt, pred, NCLS)

            miou, ious = miou_from_confmat(confmat)
            line = f"{town}/{scen}: mIoU={miou:.3f}"
            print(line)
            lines.append(line)
            detail_rows.append((town, scen, miou, ious))

    # 저장
    txt_path = os.path.join(base, args.save_txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("### Semantic Evaluation Summary ###\n\n")
        for line in lines:
            f.write(line + "\n")
    print(f"✅ 요약 저장: {txt_path}")

    tsv_path = os.path.join(base, args.save_tsv)
    with open(tsv_path, "w", encoding="utf-8") as f:
        header = ["town","scenario","mIoU"] + [f"IoU_{i}_{name}" for i, name in enumerate(CLASS_NAMES)]
        f.write("\t".join(header) + "\n")
        for town, scen, miou, ious in detail_rows:
            row = [town, scen, f"{miou:.6f}"] + [f"{float(v):.6f}" for v in ious]
            f.write("\t".join(row) + "\n")
    print(f"✅ 상세 TSV 저장: {tsv_path}")

if __name__ == "__main__":
    main()
