# check_performance_gpu.py
# ---------------------------------------------------------------
# _output4 전용: Panoptic/Instance/ Semantic 성능 평가
# - mIoU : GPU(PyTorch) 가속 (fast_confusion_matrix_gpu)
# - PQ/RQ/SQ : NumPy 기반 근사(클래스별 매칭)
# - 기본: (Town,Scenario) 전 파일 평가 / --samples_per_group 로 제한 가능
# - 결과: performance_summary.txt 저장
# ---------------------------------------------------------------

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch

IGNORE = 255
NCLS = 19
THING_TRAINIDS = set([11,12,13,14,15,16,17,18])  # Cityscapes thing

def read_png16(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img if img is not None else None

# -----------------------------
# GPU mIoU (confusion matrix)
# -----------------------------
@torch.inference_mode()
def fast_confusion_matrix_gpu(gt_np, pred_np, ncls=NCLS, ignore=IGNORE, device="cuda"):
    """
    gt_np, pred_np: np.ndarray (H,W) uint16/uint8
    return: torch.Tensor [ncls, ncls] (on CPU)
    """
    gt = torch.from_numpy(gt_np.astype(np.int64))
    pred = torch.from_numpy(pred_np.astype(np.int64))

    if device == "cuda" and torch.cuda.is_available():
        gt = gt.cuda(non_blocking=True)
        pred = pred.cuda(non_blocking=True)

    mask = (gt != ignore)
    gt = gt[mask].clamp_(0, ncls - 1)
    pred = pred[mask].clamp_(0, ncls - 1)

    idx = gt * ncls + pred
    cm = torch.bincount(idx, minlength=ncls * ncls)
    cm = cm.reshape(ncls, ncls)

    return cm.cpu()

def miou_from_confmat(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, 0.0)
    valid = (denom > 0)
    miou = iou[valid].mean() if valid.any() else 0.0
    return miou, iou

# -----------------------------
# PQ 근사 계산 (NumPy)
# -----------------------------
def segments_from_panoptic(pan_png):
    ids = np.unique(pan_png)
    segs = {}
    for v in ids:
        t = int(v // 1000)
        i = int(v % 1000)
        if t == IGNORE:
            continue
        segs[(t, i)] = (pan_png == v)
    return segs

def pq_match_and_score(gt_pan, pred_pan):
    gt_segs = segments_from_panoptic(gt_pan)
    pred_segs = segments_from_panoptic(pred_pan)
    by_cls_gt, by_cls_pr = {}, {}
    for (t, i), m in gt_segs.items():
        by_cls_gt.setdefault(t, []).append(((t, i), m))
    for (t, i), m in pred_segs.items():
        by_cls_pr.setdefault(t, []).append(((t, i), m))

    sum_iou_all = tp_all = fp_all = fn_all = 0.0
    sum_iou_th = tp_th = fp_th = fn_th = 0.0
    sum_iou_st = tp_st = fp_st = fn_st = 0.0

    for t in set(list(by_cls_gt.keys()) + list(by_cls_pr.keys())):
        gt_list = by_cls_gt.get(t, [])
        pr_list = by_cls_pr.get(t, [])
        if len(gt_list) == len(pr_list) == 0:
            continue

        iou_mat = np.zeros((len(gt_list), len(pr_list)), dtype=np.float32)
        g_areas = [m.sum() for _, m in gt_list]
        p_areas = [m.sum() for _, m in pr_list]

        for gi, (_, gmask) in enumerate(gt_list):
            g_area = g_areas[gi]
            if g_area == 0:
                continue
            for pj, (_, pmask) in enumerate(pr_list):
                inter = np.logical_and(gmask, pmask).sum()
                if inter == 0:
                    continue
                union = g_area + p_areas[pj] - inter
                iou_mat[gi, pj] = inter / max(union, 1)

        matched_gt, matched_pr, matches = set(), set(), []
        pairs = [(iou_mat[gi, pj], gi, pj)
                 for gi in range(len(gt_list)) for pj in range(len(pr_list))]
        pairs.sort(reverse=True, key=lambda x: x[0])
        for iou, gi, pj in pairs:
            if iou <= 0.5:
                break
            if gi in matched_gt or pj in matched_pr:
                continue
            matched_gt.add(gi); matched_pr.add(pj)
            matches.append(iou)

        tp, fp, fn = len(matches), len(pr_list) - len(matches), len(gt_list) - len(matches)
        sum_iou = float(np.sum(matches))

        sum_iou_all += sum_iou; tp_all += tp; fp_all += fp; fn_all += fn
        if t in THING_TRAINIDS:
            sum_iou_th += sum_iou; tp_th += tp; fp_th += fp; fn_th += fn
        else:
            sum_iou_st += sum_iou; tp_st += tp; fp_st += fp; fn_st += fn

    def pq_from(sum_iou, tp, fp, fn):
        denom = tp + 0.5 * fp + 0.5 * fn
        pq = (sum_iou / denom) if denom > 0 else 0.0
        rq = (tp / denom) if denom > 0 else 0.0
        sq = (sum_iou / max(tp, 1)) if tp > 0 else 0.0
        return pq, rq, sq

    PQ, RQ, SQ = pq_from(sum_iou_all, tp_all, fp_all, fn_all)
    PQ_th, RQ_th, SQ_th = pq_from(sum_iou_th, tp_th, fp_th, fn_th)
    PQ_st, RQ_st, SQ_st = pq_from(sum_iou_st, tp_st, fp_st, fn_st)
    return {"PQ": PQ, "RQ": RQ, "SQ": SQ, "PQ_th": PQ_th, "PQ_st": PQ_st}

# -----------------------------
# Runner
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate panoptic/instance/semantic with GPU-accelerated mIoU")
    ap.add_argument("--base", default="/media/vip-dell/HC/_output4", help="dataset root (_output4)")
    ap.add_argument("--pred_pan", default="", help="pred panoptic root (default: <base>/_pred_panoptic)")
    ap.add_argument("--samples_per_group", type=int, default=0,
                    help="0=ALL, >0이면 각 (Town,Scenario) 앞 K장만 평가")
    ap.add_argument("--save_path", default="", help="summary txt (default: <base>/performance_summary.txt)")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="mIoU 가속 디바이스")
    return ap.parse_args()

def main():
    args = parse_args()
    BASE = args.base
    PRED_PAN = args.pred_pan or os.path.join(BASE, "_pred_panoptic")
    SAVE_PATH = args.save_path or os.path.join(BASE, "performance_summary.txt")
    K = int(args.samples_per_group)

    towns = sorted([d for d in os.listdir(BASE) if d.startswith("Town")])
    lines = []
    print(f"[INFO] Base={BASE}  PredPan={PRED_PAN}  Device(mIoU)={args.device}  Samples/Group={K or 'ALL'}")

    for town in towns:
        scen_root = os.path.join(BASE, town)
        if not os.path.isdir(scen_root):
            continue
        scen_dirs = sorted([d for d in os.listdir(scen_root) if os.path.isdir(os.path.join(scen_root, d))])

        for scen in scen_dirs:
            gt_dir = os.path.join(BASE, town, scen, "gtFine")
            pan_gt_dir = os.path.join(BASE, town, scen, "panoptic")
            pred_pan_dir = os.path.join(PRED_PAN, town, scen)
            if not (os.path.isdir(gt_dir) and os.path.isdir(pan_gt_dir) and os.path.isdir(pred_pan_dir)):
                continue

            gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith("_gtFine_trainIds19.png")])
            # GT ∩ Pred 교집합만 선택
            candidates = []
            for gf in gt_files:
                base = gf.replace("_gtFine_trainIds19.png", "")
                if os.path.isfile(os.path.join(pred_pan_dir, f"{base}_pred_panopticId.png")):
                    candidates.append(base)

            if K > 0:
                selected = candidates[:K]
            else:
                selected = candidates

            if not selected:
                continue

            # GPU confusion matrix 누적 (torch 텐서로 합산)
            confmat_torch = torch.zeros((NCLS, NCLS), dtype=torch.int64)

            pq_list = []
            for base in tqdm(selected, desc=f"{town}/{scen} (K={len(selected)})", ncols=90):
                gt = read_png16(os.path.join(gt_dir, f"{base}_gtFine_trainIds19.png"))
                if gt is None:
                    continue

                pred_pan = read_png16(os.path.join(pred_pan_dir, f"{base}_pred_panopticId.png"))
                if pred_pan is None:
                    continue
                pred_train = (pred_pan // 1000).astype(np.uint16)

                # mIoU: GPU 가속 누적
                cm_gpu = fast_confusion_matrix_gpu(gt, pred_train, NCLS, IGNORE, args.device)
                confmat_torch += cm_gpu

                # PQ: NumPy 근사
                gt_pan = read_png16(os.path.join(pan_gt_dir, f"{base}_panopticId.png"))
                if gt_pan is None:
                    continue
                pq_list.append(pq_match_and_score(gt_pan, pred_pan))

            # mIoU 계산 (CPU로 변환)
            confmat = confmat_torch.numpy()
            miou, _ = miou_from_confmat(confmat)

            if pq_list:
                PQ = float(np.mean([x["PQ"] for x in pq_list]))
                RQ = float(np.mean([x["RQ"] for x in pq_list]))
                SQ = float(np.mean([x["SQ"] for x in pq_list]))
                PQ_th = float(np.mean([x["PQ_th"] for x in pq_list]))
                PQ_st = float(np.mean([x["PQ_st"] for x in pq_list]))
            else:
                PQ=RQ=SQ=PQ_th=PQ_st=0.0

            line = f"{town}/{scen}: mIoU={miou:.3f} | PQ={PQ:.3f} (RQ={RQ:.3f}, SQ={SQ:.3f}) | PQ_th={PQ_th:.3f}, PQ_st={PQ_st:.3f}"
            print(f"\n🏙️ {line}")
            lines.append(line)

    # 저장
    print("\n📊 요약 결과 저장 중...")
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        hdr = "### Mask2Former Evaluation Summary (GPU mIoU, samples_per_group={}) ###\n\n".format(K or "ALL")
        f.write(hdr)
        for line in lines:
            f.write(line + "\n")
    print(f"✅ 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    main()
