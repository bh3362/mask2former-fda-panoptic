# eval_semantic_per_class_town03.py
# Town03의 모든 시나리오에 대해 클래스별 IoU + mIoU 계산
# 출력: 화면 로그 + TSV 저장

import os, glob, cv2, numpy as np

BASE = "/media/vip-dell/HC/_output4"
TOWN = "Town03"
PRED_ROOT = os.path.join(BASE, "_pred_masks_19")
SAVE_TSV = os.path.join(BASE, "_semantic_eval_details", f"{TOWN}_per_class_iou.tsv")
os.makedirs(os.path.dirname(SAVE_TSV), exist_ok=True)

NCLS = 19
IGNORE = 255

CS19_NAMES = [
 "road","sidewalk","building","wall","fence","pole","traffic_light","traffic_sign",
 "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]

def read_id(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def fast_confmat(gt, pr, ncls=NCLS, ignore=IGNORE):
    m = (gt != ignore)
    gt = gt[m]; pr = pr[m]
    gt = np.clip(gt, 0, ncls-1); pr = np.clip(pr, 0, ncls-1)
    cm = np.bincount(gt*ncls + pr, minlength=ncls*ncls).reshape(ncls, ncls)
    return cm

def iou_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom>0, tp/denom, 0.0)
    miou = iou[denom>0].mean() if np.any(denom>0) else 0.0
    return iou, miou

def main():
    scen_dirs = sorted([d for d in os.listdir(os.path.join(BASE, TOWN)) if os.path.isdir(os.path.join(BASE, TOWN, d))])

    lines = []
    header = ["town","scenario","mIoU"] + [f"IoU_{n}" for n in CS19_NAMES]
    lines.append("\t".join(header))

    for scen in scen_dirs:
        gt_dir   = os.path.join(BASE, TOWN, scen, "gtFine")
        pred_dir = os.path.join(PRED_ROOT, TOWN, scen)
        if not (os.path.isdir(gt_dir) and os.path.isdir(pred_dir)):
            continue

        gt_list = sorted(glob.glob(os.path.join(gt_dir, "frame_*_gtFine_trainIds19.png")))
        cm = np.zeros((NCLS, NCLS), np.int64)

        # GT에 맞춰 pred 매칭
        used = 0
        for g in gt_list:
            base = os.path.basename(g).replace("_gtFine_trainIds19.png","")
            p = os.path.join(pred_dir, f"{base}_pred_mask_19.png")
            if not os.path.exists(p): 
                continue
            gt = read_id(g); pr = read_id(p)
            if gt is None or pr is None: 
                continue
            cm += fast_confmat(gt, pr)
            used += 1

        iou, miou = iou_from_cm(cm)
        # 화면 요약
        worst = np.argsort(iou)[:3]
        print(f"{TOWN}/{scen}: mIoU={miou:.3f} | worst classes:", ", ".join([f"{CS19_NAMES[i]}={iou[i]:.2f}" for i in worst]))

        row = [TOWN, scen, f"{miou:.6f}"] + [f"{v:.6f}" for v in iou]
        lines.append("\t".join(row))

    with open(SAVE_TSV, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n✅ 저장:", SAVE_TSV)

if __name__ == "__main__":
    main()
