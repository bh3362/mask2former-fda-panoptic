import os
import csv
import numpy as np
from PIL import Image
from glob import glob
from collections import Counter, OrderedDict

NUM_CLASSES = 11
IGNORE = 255

BASE_INPUT_DIR = "/home/vip-dell/CARLA_0.9.15/PythonAPI/examples/_output"
BASE_PRED_DIR  = os.path.join(BASE_INPUT_DIR, "_pred_masks")
LEFT_DIR_NAME  = "leftImg8bit"
FRAMES_TO_DEBUG = 2
FRAMES_FOR_CALIB = 8   # 매핑 학습에 쓸 프레임 수(시나리오당 최대)

SCENARIOS = ("clear","dust","fog","rain")

# ----- CARLA → Cityscapes 11-class (네 GT 값 기준) -----
CARLA2CITY = {
    0: IGNORE, 1:0, 2:1, 3:2, 6:5, 7:6, 8:7, 9:8, 11:10,
    13:IGNORE, 14:IGNORE, 15:IGNORE, 18:IGNORE, 20:IGNORE, 21:IGNORE, 22:IGNORE,
    24:0, 25:9, 255:IGNORE
}

# Cityscapes 11-class 색상 (인퍼런스 색칠용)
COLOR_MAP = OrderedDict({
    0:(128, 64,128), 1:(244, 35,232), 2:(70,70,70), 3:(102,102,156), 4:(190,153,153),
    5:(153,153,153), 6:(250,170, 30), 7:(220,220,  0), 8:(107,142, 35), 9:(152,251,152), 10:(70,130,180)
})
COLOR_MAP_INV_RGB = {COLOR_MAP[k]: k for k in COLOR_MAP}
COLOR_MAP_INV_BGR = {(b,g,r): cid for (cid,(r,g,b)) in COLOR_MAP.items()}

def key_of(path: str) -> str:
    name = os.path.basename(path)
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else name

def resize_like(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if arr.shape == ref.shape:
        return arr
    return np.array(Image.fromarray(arr).resize((ref.shape[1], ref.shape[0]), Image.NEAREST), dtype=np.uint8)

def decode_color_mask(img_rgb: np.ndarray, palette_inv: dict) -> np.ndarray:
    h,w,_ = img_rgb.shape
    ids = np.full((h,w), IGNORE, dtype=np.uint8)
    for (r,g,b), cid in palette_inv.items():
        m = (img_rgb[:,:,0]==r) & (img_rgb[:,:,1]==g) & (img_rgb[:,:,2]==b)
        ids[m] = cid
    return ids

def load_gt_as_class_ids(gt_path: str) -> np.ndarray:
    arr = np.array(Image.open(gt_path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:,:,0]
    mapped = np.full(arr.shape, IGNORE, dtype=np.uint8)
    for v in np.unique(arr):
        mapped[arr == v] = CARLA2CITY.get(int(v), IGNORE)
    return mapped

def load_pr_as_class_ids(pr_dir: str, key: str) -> np.ndarray:
    id_path = os.path.join(pr_dir, f"frame_{key}_pred_mask.png")
    if os.path.exists(id_path):
        return np.array(Image.open(id_path)).astype(np.uint8)
    color_path = os.path.join(pr_dir, f"frame_{key}_color_mask.png")
    rgb = np.array(Image.open(color_path).convert("RGB"))
    ids_rgb = decode_color_mask(rgb, COLOR_MAP_INV_RGB)
    ids_bgr = decode_color_mask(rgb, COLOR_MAP_INV_BGR)
    return ids_rgb if (ids_rgb != IGNORE).mean() >= (ids_bgr != IGNORE).mean() else ids_bgr

def compute_iou_masked(gt_ids: np.ndarray, pr_ids: np.ndarray, num_classes=NUM_CLASSES):
    valid = (gt_ids != IGNORE)
    if valid.sum() == 0:
        return np.zeros(num_classes, dtype=float)
    gt = gt_ids[valid]
    pr = pr_ids[valid]
    ious = []
    for c in range(num_classes):
        inter = np.logical_and(gt == c, pr == c).sum()
        union = np.logical_or(gt == c, pr == c).sum()
        ious.append((inter/union) if union > 0 else 0.0)
    return np.array(ious, dtype=float)

# ---------- 자동 라벨 보정(간단 greedy) ----------
def calibrate_label_mapping(scn_gt_dir, scn_pr_dir, keys):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # rows: GT, cols: PR
    used = 0
    for k in keys[:FRAMES_FOR_CALIB]:
        gt_path = os.path.join(scn_gt_dir, f"frame_{k}_gtFine_instanceIds.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(scn_gt_dir, f"frame_{k}_gtFine_labelIds.png")
        if not os.path.exists(gt_path):
            continue
        gt_ids = load_gt_as_class_ids(gt_path)
        pr_ids = load_pr_as_class_ids(scn_pr_dir, k)
        pr_ids = resize_like(pr_ids, gt_ids)
        valid = (gt_ids != IGNORE)
        if valid.sum() == 0:
            continue
        gt = gt_ids[valid].astype(np.int64)
        pr = pr_ids[valid].astype(np.int64)
        mask = (pr != IGNORE)
        gt, pr = gt[mask], pr[mask]
        idx = gt * NUM_CLASSES + pr
        binc = np.bincount(idx, minlength=NUM_CLASSES*NUM_CLASSES)
        conf += binc.reshape(NUM_CLASSES, NUM_CLASSES)
        used += 1

    if used == 0:
        return {i:i for i in range(NUM_CLASSES)}

    mapping = {}
    conf_copy = conf.copy()
    for _ in range(NUM_CLASSES):
        i, j = divmod(conf_copy.argmax(), NUM_CLASSES)
        if conf_copy[i, j] == 0:
            break
        mapping[j] = i     # PR j → GT i
        conf_copy[i, :] = -1
        conf_copy[:, j] = -1
    for j in range(NUM_CLASSES):
        mapping.setdefault(j, j)
    print("[CALIB] PR→GT mapping:", mapping)
    return mapping

def apply_mapping(pr_ids, mapping):
    out = np.full_like(pr_ids, IGNORE, dtype=np.uint8)
    for pr_lbl, gt_lbl in mapping.items():
        out[pr_ids == pr_lbl] = gt_lbl
    return out

def evaluate(base_input_dir=BASE_INPUT_DIR, base_pred_dir=BASE_PRED_DIR, scenarios=SCENARIOS, save_csv="eval_summary.csv"):
    results = {}  # scenario -> dict(mIoU=float, per_class=list, n_frames=int)

    all_means = []
    for sc in scenarios:
        sc_gt_dir  = os.path.join(base_input_dir, sc, "gtFine")
        sc_pr_dir  = os.path.join(base_pred_dir, sc)

        gt_files = sorted(glob(os.path.join(sc_gt_dir, "frame_*_gtFine_instanceIds.png")) +
                          glob(os.path.join(sc_gt_dir, "frame_*_gtFine_labelIds.png")))
        pr_files = sorted(glob(os.path.join(sc_pr_dir, "frame_*_color_mask.png")) +
                          glob(os.path.join(sc_pr_dir, "frame_*_pred_mask.png")))
        if not gt_files or not pr_files:
            print(f"[{sc}] No data found.")
            continue

        gt_map = {key_of(f): f for f in gt_files}
        pr_keys = {key_of(f) for f in pr_files}
        common = sorted(set(gt_map.keys()) & pr_keys)
        if not common:
            print(f"[{sc}] No matched frames.")
            continue

        # 1) 시나리오별 자동 매핑 학습
        mapping = calibrate_label_mapping(sc_gt_dir, sc_pr_dir, common)

        # 2) 앞 2장 디버그
        for k in common[:FRAMES_TO_DEBUG]:
            gt_path = gt_map[k]
            pr_before = load_pr_as_class_ids(sc_pr_dir, k)
            gt_ids = load_gt_as_class_ids(gt_path)
            pr_before = resize_like(pr_before, gt_ids)
            pr_after  = apply_mapping(pr_before, mapping)

            cov_gt = (gt_ids != IGNORE).mean()
            cov_pr = (pr_after != IGNORE).mean()
            print(f"[DEBUG {sc}-{k}] shapes GT/PR: {gt_ids.shape} / {pr_after.shape}")
            print(f"[DEBUG {sc}-{k}] GT uniq: {np.unique(gt_ids)} | valid: {cov_gt:.3f}")
            print(f"[DEBUG {sc}-{k}] PR uniq(after map): {np.unique(pr_after)} | covered: {cov_pr:.3f}")

            gt_hist = Counter(gt_ids[gt_ids != IGNORE].flatten())
            pr_hist = Counter(pr_after[pr_after != IGNORE].flatten())
            print(f"[DEBUG {sc}-{k}] GT hist (top): {dict(sorted(gt_hist.items(), key=lambda x:-x[1])[:6])}")
            print(f"[DEBUG {sc}-{k}] PR hist (top, mapped): {dict(sorted(pr_hist.items(), key=lambda x:-x[1])[:6])}")

            ious = compute_iou_masked(gt_ids, pr_after)
            print(f"[DEBUG {sc}-{k}] frame mIoU(after map): {ious.mean():.4f} | per-class: {' '.join(f'{v:.3f}' for v in ious)}")

        # 3) 전체 평가(매핑 적용)
        sc_ious = []
        for k in common:
            gt_path = gt_map[k]
            pr_raw = load_pr_as_class_ids(sc_pr_dir, k)
            gt_ids = load_gt_as_class_ids(gt_path)
            pr_raw = resize_like(pr_raw, gt_ids)
            pr_ids = apply_mapping(pr_raw, mapping)
            sc_ious.append(compute_iou_masked(gt_ids, pr_ids))

        sc_ious = np.array(sc_ious)
        mean_c = sc_ious.mean(axis=0)
        mIoU = float(mean_c.mean())
        print(f"[{sc}] mIoU (mapped): {mIoU:.4f} | per-class: {' '.join(f'{x:.3f}' for x in mean_c)}")

        results[sc] = {
            "mIoU": mIoU,
            "per_class": [float(x) for x in mean_c],
            "n_frames": len(sc_ious)
        }
        all_means.append(mean_c)

    # 4) 전체 평균
    if all_means:
        all_means = np.vstack(all_means)
        m = all_means.mean(axis=0)
        print(f"\n[ALL] mIoU (mapped): {m.mean():.4f}")
        for i,v in enumerate(m):
            print(f"class {i:02d} IoU: {v:.4f}")

    # 5) 요약/랭킹 출력
    if results:
        print("\n==== Scenario-wise Summary ====")
        # 정렬
        ranked = sorted(results.items(), key=lambda kv: kv[1]["mIoU"], reverse=True)
        # 표 형태
        header = f"{'Scenario':<18} {'mIoU':>8} {'Frames':>8}"
        print(header)
        print("-"*len(header))
        for sc, info in ranked:
            print(f"{sc:<18} {info['mIoU']:>8.4f} {info['n_frames']:>8d}")

        best_sc, best_info = ranked[0]
        worst_sc, worst_info = ranked[-1]
        print("\n▶ 요약 문장")
        print(f"· 가장 성능이 좋은 날씨는 **{best_sc}** (mIoU {best_info['mIoU']:.3f}) 입니다.")
        if len(ranked) > 1:
            print(f"· 가장 낮은 성능은 **{worst_sc}** (mIoU {worst_info['mIoU']:.3f}) 입니다.")

    # 6) CSV 저장
    if results and save_csv:
        csv_path = os.path.abspath(save_csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "mIoU", "n_frames"] + [f"class_{i}" for i in range(NUM_CLASSES)])
            for sc, info in sorted(results.items()):
                row = [sc, info["mIoU"], info["n_frames"]] + info["per_class"]
                writer.writerow(row)
        print(f"\n📄 Summary saved to: {csv_path}")

    return results

if __name__ == "__main__":
    evaluate()
