import os
import numpy as np
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm

NUM_CLASSES = 19  # 클래스 수

def fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    return np.bincount(
        num_classes * true[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)

def find_all_gt_images(gt_root):
    # gtFine_labelIds.png 파일을 재귀적으로 정렬해서 수집
    return sorted(glob(os.path.join(gt_root, "temp_sem_seg_masks", "*_sem_seg.png")))
    
def evaluate(gt_dir, pred_dir, num_classes):
    gt_paths = find_all_gt_images(gt_dir)
    pred_paths = sorted(glob(os.path.join(pred_dir, "semseg_*.png")))
    pred_paths = [p for p in pred_paths if "vis" not in p]  # ✅ 이 줄 추가

    assert len(gt_paths) == len(pred_paths), f"[❌] GT({len(gt_paths)}) != Pred({len(pred_paths)})"

    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    for gt_path, pred_path in tqdm(zip(gt_paths, pred_paths), total=len(gt_paths)):
        gt_img = Image.open(gt_path)
        pred_img = Image.open(pred_path)

        # ➤ ensure label maps are 2D (H, W)
        gt = np.array(gt_img.convert("L"), dtype=np.int64)
        pred = np.array(pred_img.convert("L"), dtype=np.int64)




        hist += fast_hist(gt, pred, num_classes)


    ious = per_class_iu(hist)
    miou = np.nanmean(ious)

    # 출력 + 저장
    print("\n===== IoU per class =====")
    result_lines = []
    for i, iou in enumerate(ious):
        line = f"Class {i:2d}: {iou:.4f}"
        print(line)
        result_lines.append(line)

    miou_line = f"\n✅ mIoU: {miou:.4f}"
    print(miou_line)
    result_lines.append(miou_line)

    result_path = os.path.join(pred_dir, "evaluation_result.txt")
    with open(result_path, "w") as f:
        for line in result_lines:
            f.write(line + "\n")
    print(f"\n📁 Results saved to: {result_path}")

    return miou

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Path to _output directory (contains clear/, dust/, etc.)")
    parser.add_argument("--pred_dir", required=True, help="Path to predicted semseg_*.png files")
    parser.add_argument("--num_classes", type=int, default=19)
    args = parser.parse_args()

    evaluate(args.gt_dir, args.pred_dir, args.num_classes)
