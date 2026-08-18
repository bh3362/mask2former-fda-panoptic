import os, sys, argparse
import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # scripts/inference
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)                        # scripts/
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)                       # repo root
sys.path.append(_SCRIPTS_DIR)
sys.path.append(_REPO_ROOT)
from register_datasets import register_carla_panoptic

from detectron2.data import DatasetCatalog
from panopticapi.utils import id2rgb, rgb2id

CARLA_DATASET_ROOT = os.environ.get("CARLA_DATASET_ROOT", "/media/vip-dell/HC/final_dataset5")
CARLA_IMAGE_SUBDIR = os.environ.get("CARLA_IMAGE_SUBDIR", "leftImg8bit")  # or "leftImg8bit_fda"

def load_id_or_rgb_png(path):
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return rgb2id(arr)
    return arr.astype(np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carla_final_panoptic_val")
    ap.add_argument("--pred_dir", default="./output/inference_carla_final_panoptic_val/panoptic_pred")
    ap.add_argument("--out_dir",  default="./output/vis_pred_vs_gt_val")
    ap.add_argument("--max_vis", type=int, default=50)
    args = ap.parse_args()

    register_carla_panoptic(CARLA_DATASET_ROOT, CARLA_IMAGE_SUBDIR, "carla_final_panoptic_train", "carla_final_panoptic_val")
    recs = DatasetCatalog.get(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    n = min(len(recs), args.max_vis)
    for i in range(n):
        r = recs[i]
        gt_png = r["pan_seg_file_name"]
        pred_png = os.path.join(args.pred_dir, f"{i+1}.png")
        if not os.path.isfile(pred_png):
            print(f"[SKIP] pred missing: {pred_png}")
            continue

        gt_id = load_id_or_rgb_png(gt_png)
        pred_id = load_id_or_rgb_png(pred_png)

        gt_rgb = id2rgb(gt_id)
        pred_rgb = id2rgb(pred_id)

        h = max(gt_rgb.shape[0], pred_rgb.shape[0])
        w = gt_rgb.shape[1] + pred_rgb.shape[1]
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[:gt_rgb.shape[0], :gt_rgb.shape[1]] = gt_rgb
        canvas[:pred_rgb.shape[0], gt_rgb.shape[1]:] = pred_rgb

        out = os.path.join(args.out_dir, f"{i+1:04d}.png")
        Image.fromarray(canvas).save(out)
        print(f"[SAVE] {out}")

    print(f"[DONE] {args.out_dir}")

if __name__ == "__main__":
    main()
