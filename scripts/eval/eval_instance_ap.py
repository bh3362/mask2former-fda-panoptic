# eval_instance_from_saved_preds.py  (HC/_output3 전용 편의 + 디버그 강화)
import os, json, argparse, glob
from collections import defaultdict, Counter
import numpy as np
import cv2

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

# ---------------- utils ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def read_panoptic_idmap(abs_png):
    im = cv2.imread(abs_png, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        return im.astype(np.uint32)
    if im.shape[2] == 4:
        im = im[:, :, :3]
    rgb = im[:, :, ::-1]  # BGR->RGB
    return (rgb[:, :, 0].astype(np.uint32)
            + rgb[:, :, 1].astype(np.uint32) * 256
            + rgb[:, :, 2].astype(np.uint32) * 256 * 256)

def rle_from_mask(bin_mask):
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def build_id_maps(gt_json):
    def stem_from_rgb_path(p): return os.path.basename(p).replace("_leftImg8bit.png", "")
    id2triplet = {
        im["id"]: (im["file_name"].split("/")[0],
                   im["file_name"].split("/")[1],
                   stem_from_rgb_path(im["file_name"]))
        for im in gt_json["images"]
    }
    cats = gt_json["categories"]
    cat_isthing = {c["id"]: int(c.get("isthing", 0)) == 1 for c in cats}
    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}
    coco_categories = [{
        "id": c["id"], "name": c.get("name", str(c["id"])),
        "supercategory": c.get("supercategory", "thing")
    } for c in cats if int(c.get("isthing", 0)) == 1]
    return id2triplet, cat_isthing, cat_name, coco_categories

def collect_pred_files(pred_root):
    patt = os.path.join(pred_root, "Town*", "*", "*_instances_pred.json")
    return sorted(glob.glob(patt))

def parse_contig2gt_map(s: str):
    """Parse string like '0->9,1->9,2->2' into {0:9,1:9,2:2}"""
    m = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "->" not in pair:
            continue
        a, b = pair.split("->")
        m[int(a.strip())] = int(b.strip())
    return m


def bbox_from_rle(seg):
    box = mask_utils.toBbox(seg).tolist()
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

# ---------------- main ----------------
def parse_args():
    ap = argparse.ArgumentParser("COCO-style instance AP from saved per-image predictions (HC/_output3 편의세팅)")
    ap.add_argument("--base", default="/media/vip-dell/HC/_output3",
                    help="실험 루트(기본: /media/vip-dell/HC/_output3)")
    ap.add_argument("--gt_json", default="", help="(선택) panoptic_all.json 경로. 비우면 <base>/_coco/panoptic_json/panoptic_all.json")
    ap.add_argument("--gt_png_root", default="", help="(선택) panoptic PNG 루트. 비우면 <base>")
    ap.add_argument("--pred_root", default="", help="(선택) *_instances_pred.json 루트. 비우면 <base>/_pred_instances")
    ap.add_argument("--out_dir", default="", help="(선택) 출력 폴더. 비우면 <base>/_eval_instance_from_saved")

    # 평가 옵션
    ap.add_argument("--all_classes", action="store_true",
                    help="things+stuff 전부 (기본은 thing만)")
    ap.add_argument("--require_masks_for_segm", action="store_true", default=False,
                    help="True면 예측 중 마스크 없는 게 하나라도 있으면 segm 평가 스킵")
    ap.add_argument("--contig2gt", default="", help="연속 id -> GT id 매핑. 예: '11,12,13,14,15,16,17,18'")
    ap.add_argument("--use_mask_bbox", action="store_true",
                    help="bbox AP를 마스크에서 구한 bbox로 평가")
    ap.add_argument("--score_thr", type=float, default=0.0,
                    help="이 값 미만 score 예측은 버림(디버그용)")
    ap.add_argument("--debug_stats", action="store_true",
                    help="bbox sanity/분포 통계 출력 및 저장")
    return ap.parse_args()

def main():
    args = parse_args()

    # 경로 자동 세팅
    base = args.base
    gt_json = args.gt_json or os.path.join(base, "_coco", "panoptic_json", "panoptic_all.json")
    gt_png_root = args.gt_png_root or base
    pred_root = args.pred_root or os.path.join(base, "_pred_instances")
    out_dir = args.out_dir or os.path.join(base, "_eval_instance_from_saved")
    ensure_dir(out_dir)

    print("[PATH]")
    print(" base        :", base)
    print(" gt_json     :", gt_json)
    print(" gt_png_root :", gt_png_root)
    print(" pred_root   :", pred_root)
    print(" out_dir     :", out_dir)

    # GT 적재
    gt = json.load(open(gt_json, "r"))
    id2triplet, cat_isthing, cat_name, coco_categories = build_id_maps(gt)
    CONTIG2GT = parse_contig2gt_map(args.contig2gt)
    print("[MAP] contiguous->GT:", CONTIG2GT)

    # 예측 파일 수집
    pred_files = collect_pred_files(pred_root)
    if not pred_files:
        raise RuntimeError(f"No *_instances_pred.json under {pred_root}")

    # (Town,Scenario,stem) -> pred json
    tss2pred = {}
    for pf in pred_files:
        fn = os.path.basename(pf)
        if not fn.endswith("_instances_pred.json"):
            continue
        stem = fn[:-len("_instances_pred.json")]
        parts = os.path.relpath(pf, pred_root).split(os.sep)
        if len(parts) < 3:
            continue
        town, scen = parts[0], parts[1]
        tss2pred[(town, scen, stem)] = pf

    # GT 이미지 중 예측과 매칭되는 것만
    selected_images = []
    for im in gt["images"]:
        img_id = im["id"]
        town, scen, stem = id2triplet[img_id]
        if (town, scen, stem) in tss2pred:
            selected_images.append(im)
    if not selected_images:
        raise RuntimeError("No GT images match your instance prediction files.")

    # ---- COCO GT 생성 ----
    evaluating_things_only = not args.all_classes
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": coco_categories if evaluating_things_only else [
            {"id": c["id"], "name": c.get("name", str(c["id"])), "supercategory": c.get("supercategory", "")}
            for c in gt["categories"]
        ]
    }
    ann_id = 1
    selected_ids = {im["id"] for im in selected_images}
    anns_by_img = defaultdict(list)
    for a in gt["annotations"]:
        if a["image_id"] in selected_ids:
            anns_by_img[a["image_id"]].append(a)

    for im in selected_images:
        img_id = im["id"]
        rel_png, segments = None, []
        for a in anns_by_img[img_id]:
            rel_png = a["file_name"]; segments = a["segments_info"]
        if rel_png is None:
            continue
        idmap = read_panoptic_idmap(os.path.join(gt_png_root, rel_png))
        if idmap is None:
            continue

        coco_gt["images"].append({
            "id": img_id,
            "file_name": im["file_name"],
            "width": int(im.get("width", idmap.shape[1])),
            "height": int(im.get("height", idmap.shape[0]))
        })
        for s in segments:
            cid = int(s["category_id"])
            if evaluating_things_only and not cat_isthing.get(cid, False):
                continue
            sid = int(s["id"])
            bin_mask = (idmap == sid)
            area = int(bin_mask.sum())
            if area == 0:
                continue
            rle = rle_from_mask(bin_mask)
            bbox = mask_utils.toBbox(rle).tolist()
            coco_gt["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "segmentation": rle, "area": area, "bbox": bbox,
                "iscrowd": int(s.get("iscrowd", 0))
            })
            ann_id += 1

    gt_inst_json = os.path.join(out_dir, "coco_gt_instances.json")
    with open(gt_inst_json, "w") as f:
        json.dump(coco_gt, f, indent=2)
    print(f"[WRITE] COCO GT -> {gt_inst_json}  (images={len(coco_gt['images'])}, anns={len(coco_gt['annotations'])})")

    # ---- COCO preds 생성 ----
    coco_preds_bbox, coco_preds_segm = [], []
    missing_masks = False

    # 디버그 통계
    bad_w = bad_h = 0
    tot = 0
    iou_sum = 0.0
    cnt_iou = 0
    pred_cat_hist = Counter()
    score_list = []

    for im in selected_images:
        img_id = im["id"]
        town, scen, stem = id2triplet[img_id]
        pred = json.load(open(tss2pred[(town, scen, stem)], "r"))
        insts = pred.get("instances", [])

        for det in insts:
            # 스코어 필터
            score = float(det.get("score", 1.0))
            if score < args.score_thr:
                continue

            # 카테고리 매핑 (contiguous -> GT)
            cid_raw = int(det.get("category_id", 0))
            cid = int(CONTIG2GT.get(cid_raw, cid_raw))
            if evaluating_things_only and not cat_isthing.get(cid, False):
                continue

            pred_cat_hist[cid] += 1
            score_list.append(score)

            # segmentation (optional)
            seg = det.get("segmentation", None)
            if isinstance(seg, dict) and "counts" in seg and "size" in seg:
                coco_preds_segm.append({
                    "image_id": img_id, "category_id": cid,
                    "segmentation": seg, "score": score
                })
            else:
                seg = None
                missing_masks = True

            # bbox: 마스크에서 추출 or 모델 제공
            if args.use_mask_bbox and seg is not None:
                xywh = bbox_from_rle(seg)
            else:
                xyxy = det.get("bbox_xyxy", None)
                if xyxy is None or len(xyxy) != 4:
                    if seg is None:
                        continue
                    xywh = bbox_from_rle(seg)
                else:
                    xywh = xyxy_to_xywh(xyxy)

            # bbox sanity
            tot += 1
            if xywh[2] <= 0: bad_w += 1
            if xywh[3] <= 0: bad_h += 1

            # bbox vs mask bbox IoU
            if args.debug_stats and seg is not None:
                mb = bbox_from_rle(seg)
                ax1, ay1, aw, ah = xywh; bx1, by1, bw, bh = mb
                ax2, ay2 = ax1 + aw, ay1 + ah
                bx2, by2 = bx1 + bw, by1 + bh
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                union = max(aw * ah, 0.0) + max(bw * bh, 0.0) - inter if (aw > 0 and ah > 0 and bw > 0 and bh > 0) else 0.0
                if union > 0:
                    iou_sum += inter / union
                    cnt_iou += 1

            coco_preds_bbox.append({
                "image_id": img_id, "category_id": cid,
                "bbox": xywh, "score": score
            })

    pred_bbox_json = os.path.join(out_dir, "coco_pred_bbox.json")
    with open(pred_bbox_json, "w") as f:
        json.dump(coco_preds_bbox, f, indent=2)
    print(f"[WRITE] COCO PRED (bbox) -> {pred_bbox_json}  (dets={len(coco_preds_bbox)})")

    pred_segm_json = None
    if coco_preds_segm:
        pred_segm_json = os.path.join(out_dir, "coco_pred_segm.json")
        with open(pred_segm_json, "w") as f:
            json.dump(coco_preds_segm, f, indent=2)
        print(f"[WRITE] COCO PRED (segm) -> {pred_segm_json}  (dets={len(coco_preds_segm)})")

    # 디버그 통계 저장
    if args.debug_stats:
        avg_iou = (iou_sum / max(1, cnt_iou)) if cnt_iou > 0 else 0.0
        dbg = {
            "total_bbox": tot,
            "neg_width": bad_w,
            "neg_height": bad_h,
            "mask_bbox_iou_avg": avg_iou,
            "pred_category_hist": dict(sorted(pred_cat_hist.items())),
            "score_mean": float(np.mean(score_list)) if score_list else 0.0,
            "score_std": float(np.std(score_list)) if score_list else 0.0,
        }
        with open(os.path.join(out_dir, "pred_debug_stats.json"), "w") as f:
            json.dump(dbg, f, indent=2)
        print("[DEBUG] saved ->", os.path.join(out_dir, "pred_debug_stats.json"))

    # ---- COCOeval ----
    cocoGt = COCO(gt_inst_json)

    print("\n[COCOeval] BBOX ...")
    bbox_res = {}
    if len(coco_preds_bbox):
        E = COCOeval(cocoGt, cocoGt.loadRes(pred_bbox_json), iouType='bbox')
        E.evaluate(); E.accumulate(); E.summarize()
        bbox_res = {
            "AP": float(E.stats[0]), "AP50": float(E.stats[1]),
            "AP75": float(E.stats[2]), "APs": float(E.stats[3]),
            "APm": float(E.stats[4]), "APl": float(E.stats[5])
        }

    segm_res = {}
    if pred_segm_json and (not args.require_masks_for_segm or not missing_masks):
        print("\n[COCOeval] SEGM ...")
        E2 = COCOeval(cocoGt, cocoGt.loadRes(pred_segm_json), iouType='segm')
        E2.evaluate(); E2.accumulate(); E2.summarize()
        segm_res = {
            "AP": float(E2.stats[0]), "AP50": float(E2.stats[1]),
            "AP75": float(E2.stats[2]), "APs": float(E2.stats[3]),
            "APm": float(E2.stats[4]), "APl": float(E2.stats[5])
        }
    else:
        print("\n[COCOeval] SEGM skipped (no masks).")

    # summary 저장
    summary = {
        "paths": {
            "base": base, "gt_json": gt_json, "gt_png_root": gt_png_root,
            "pred_root": pred_root, "out_dir": out_dir
        },
        "num_images": len(coco_gt["images"]),
        "num_gt_instances": len(coco_gt["annotations"]),
        "bbox": bbox_res, "segm": segm_res,
        "things_only": evaluating_things_only,
        "notes": "SEGM skipped if no predicted masks." if not segm_res else ""
    }
    out_json = os.path.join(out_dir, "instance_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] results -> {out_json}")

    # txt 리포트
    out_txt = os.path.join(out_dir, "instance_eval_summary.txt")
    lines = []
    lines.append("Instance evaluation (COCO AP)")
    lines.append("--------------------------------")
    lines.append(f"GT (from panoptic) : {os.path.abspath(gt_inst_json)}")
    lines.append(f"Pred bbox          : {os.path.abspath(pred_bbox_json)}")
    if pred_segm_json: lines.append(f"Pred segm          : {os.path.abspath(pred_segm_json)}")
    lines.append(f"Images             : {summary['num_images']}")
    lines.append(f"GT instances       : {summary['num_gt_instances']}")
    lines.append("")
    if bbox_res:
        lines.append("[BBox] AP={AP:.4f}  AP50={AP50:.4f}  AP75={AP75:.4f}  APs={APs:.4f}  APm={APm:.4f}  APl={APl:.4f}".format(
            **{k: (v if not np.isnan(v) else 0.0) for k, v in bbox_res.items()}
        ))
    if segm_res:
        lines.append("[Segm] AP={AP:.4f}  AP50={AP50:.4f}  AP75={AP75:.4f}  APs={APs:.4f}  APm={APm:.4f}  APl={APl:.4f}".format(
            **{k: (v if not np.isnan(v) else 0.0) for k, v in segm_res.items()}
        ))
    if not segm_res:
        lines.append("[Segm] skipped (no masks)")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SAVED] summary -> {out_txt}")

if __name__ == "__main__":
    main()
