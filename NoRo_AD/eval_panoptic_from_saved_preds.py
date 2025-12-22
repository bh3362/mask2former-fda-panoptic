# eval_panoptic_from_saved_preds.py
import os, json, argparse, random
import numpy as np
import cv2
from collections import defaultdict, Counter
from panopticapi.evaluation import pq_compute
from panopticapi.utils import rgb2id, id2rgb

def parse_args():
    ap = argparse.ArgumentParser("PQ eval from saved panoptic preds (robust + optional auto remap)")
    ap.add_argument("--gt_json", required=True, help="GT panoptic JSON (e.g., panoptic_all.json)")
    ap.add_argument("--gt_png_root", required=True, help="GT panoptic PNG root (joins with annotations[*].file_name)")
    ap.add_argument("--pred_root", required=True, help="root of saved predictions (_pred_panoptic)")
    ap.add_argument("--out_dir", required=True, help="output dir (predictions.json, filtered_gt.json, pq_results.json, summary.txt)")
    ap.add_argument("--compute_pred_area", action="store_true",
                    help="compute & fill area for each pred segment (safe but a bit slower)")
    # NEW: 자동 리맵 옵션
    ap.add_argument("--auto_remap_pred_categories", action="store_true",
                    help="build pred->GT category id map from overlaps and remap predictions before PQ")
    ap.add_argument("--remap_sample", type=int, default=50,
                    help="number of images to use for building the remap (if enabled)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_as_idmap(abs_png):
    im = cv2.imread(abs_png, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        return im.astype(np.uint32)
    if im.ndim == 3:
        if im.shape[2] == 4:
            im = im[:, :, :3]
        rgb = im[:, :, ::-1]
        return rgb2id(rgb).astype(np.uint32)
    return None

def materialize_rgb_panoptic(abs_in, abs_out):
    im = cv2.imread(abs_in, cv2.IMREAD_UNCHANGED)
    if im is None:
        return False, "read-failed"
    ensure_dir(os.path.dirname(abs_out))
    if im.ndim == 2:
        idmap = im.astype(np.uint32)
        rgb = id2rgb(idmap)
        bgr = rgb[:, :, ::-1]
        ok = cv2.imwrite(abs_out, bgr)
        return bool(ok), "converted-single-channel->rgb"
    else:
        ok = cv2.imwrite(abs_out, im)
        return bool(ok), "copied-3ch"

def safe_int(x):
    if isinstance(x, (np.generic,)):
        return int(x.item())
    return int(x)

def compute_segment_areas_from_png(abs_pred_png, segments_info):
    idmap = read_as_idmap(abs_pred_png)
    if idmap is None:
        return {}
    areas = {}
    for s in segments_info:
        sid = safe_int(s["id"])
        areas[sid] = int((idmap == sid).sum())
    return areas

def _get_nested(d, *keys, default=np.nan):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur

def format_float(x):
    try:
        xf = float(x)
        if np.isnan(xf):
            return "nan"
        return f"{xf:.4f}"
    except Exception:
        return "nan"

def write_summary_txt(out_dir, res, num_images, paths):
    pq_all   = _get_nested(res, "All", "pq",   default=_get_nested(res, "pq"))
    sq_all   = _get_nested(res, "All", "sq",   default=_get_nested(res, "sq"))
    rq_all   = _get_nested(res, "All", "rq",   default=_get_nested(res, "rq"))
    n_all    = _get_nested(res, "All", "n",    default=_get_nested(res, "n"))
    pq_th    = _get_nested(res, "Things", "pq", default=np.nan)
    sq_th    = _get_nested(res, "Things", "sq", default=np.nan)
    rq_th    = _get_nested(res, "Things", "rq", default=np.nan)
    n_th     = _get_nested(res, "Things", "n",  default=np.nan)
    pq_st    = _get_nested(res, "Stuff", "pq",  default=np.nan)
    sq_st    = _get_nested(res, "Stuff", "sq",  default=np.nan)
    rq_st    = _get_nested(res, "Stuff", "rq",  default=np.nan)
    n_st     = _get_nested(res, "Stuff", "n",   default=np.nan)

    if isinstance(n_all, float) and np.isnan(n_all):
        n_all = num_images

    lines = []
    lines.append("Panoptic PQ evaluation summary")
    lines.append("--------------------------------")
    lines.append(f"GT JSON      : {os.path.abspath(paths['gt_json'])}")
    lines.append(f"GT PNG root  : {os.path.abspath(paths['gt_png_root'])}")
    lines.append(f"Pred root    : {os.path.abspath(paths['pred_root'])}")
    lines.append(f"Work dir     : {os.path.abspath(paths['out_dir'])}")
    lines.append(f"predictions  : {os.path.abspath(paths['pred_json'])}")
    lines.append(f"gt_filtered  : {os.path.abspath(paths['gt_fixed_json'])}")
    lines.append("")
    lines.append(f"Images evaluated: {int(n_all)}")
    lines.append("")
    lines.append("== All ==")
    lines.append(f"PQ={format_float(pq_all)}  SQ={format_float(sq_all)}  RQ={format_float(rq_all)}  N={int(n_all)}")
    lines.append("")
    lines.append("== Things ==")
    lines.append(f"PQ={format_float(pq_th)}  SQ={format_float(sq_th)}  RQ={format_float(rq_th)}  N={format_float(n_th)}")
    lines.append("")
    lines.append("== Stuff ==")
    lines.append(f"PQ={format_float(pq_st)}  SQ={format_float(sq_st)}  RQ={format_float(rq_st)}  N={format_float(n_st)}")
    lines.append("")

    out_txt = os.path.join(out_dir, "pq_results_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_txt

# -------------------- NEW: 자동 리맵 도우미 --------------------
def build_gt_sid2cid_map_per_image(gt_annotations):
    """image_id -> {segment_id -> gt_category_id}"""
    per_img = defaultdict(dict)
    for ann in gt_annotations:
        img_id = ann["image_id"]
        for s in ann["segments_info"]:
            per_img[img_id][int(s["id"])] = int(s["category_id"])
    return per_img

def build_pred_sid2cid_map_per_image(pred_annotations):
    """image_id -> {segment_id -> pred_category_id}"""
    per_img = defaultdict(dict)
    for ann in pred_annotations:
        img_id = ann["image_id"]
        for s in ann["segments_info"]:
            per_img[img_id][int(s["id"])] = int(s["category_id"])
    return per_img

def learn_pred_to_gt_cid_map(sample_img_ids, fixed_gt_root, pred_root,
                             gt_sid2cid_per_img, pred_sid2cid_per_img):
    """
    픽셀 overlap로 pred category -> gt category 매핑을 학습.
    여러 이미지에서 누적 다수결로 최종 매핑을 만든다.
    """
    votes = defaultdict(Counter)  # pred_cid -> Counter(gt_cid: pixels)

    for img_id in sample_img_ids:
        # 파일 경로 구성: GT/Pred PNG 상대경로는 gt_annotations의 file_name, pred_annotations의 file_name을 통해 알 수 있지만
        # 여기선 idmap만 필요하므로 PNG만 읽는다.
        # GT 경로는 filtered_gt_fixed.json의 ann file_name을 유지해야 하지만,
        # 이 함수는 상위에서 만든 경로를 그대로 재사용하기 위해 호출 위치에서 abs 경로를 넘겨도 된다.
        # 다만 여기서는 형식을 통일하기 위해 per_img map만 활용하고 PNG는 폴더에서 직접 찾는다.

        # GT PNG 찾기: gt_sid2cid_per_img 에는 세그먼트만 있고 파일 경로 정보가 없으므로
        # 실제 경로는 상위에서 전달해야 하지만, 호출부에서 이미 fixed_gt_root/<rel>로 저장되므로
        # rel을 알기 위해 한 번 더 이미지 id -> rel file_name 매핑이 필요하다.
        # 간단히는 호출부에서 abs path를 함께 넘기는 구조가 좋지만, 여기서는 파일명 패턴을 유도할 수 없으므로
        # 이 함수는 "외부에서 abs 경로 리스트로 호출"하도록 설계하는게 이상적이다.
        # => 구현 단순화를 위해, 이 함수 호출부에서 abs 경로를 dict로 전달하도록 변경.
        pass

# ==== (위쪽 동일) import, utils, write_summary_txt 까지 동일 ====

# -------------------- NEW: 자동 리맵 도우미 --------------------
def build_gt_sid2cid_map_per_image(gt_annotations):
    per_img = defaultdict(dict)
    for ann in gt_annotations:
        img_id = ann["image_id"]
        for s in ann["segments_info"]:
            per_img[img_id][int(s["id"])] = int(s["category_id"])
    return per_img

def build_pred_sid2cid_map_per_image(pred_annotations):
    per_img = defaultdict(dict)
    for ann in pred_annotations:
        img_id = ann["image_id"]
        for s in ann["segments_info"]:
            per_img[img_id][int(s["id"])] = int(s["category_id"])
    return per_img

def learn_pred_to_gt_cid_map(sample_img_ids, gt_png_by_imgid, pred_png_by_imgid,
                             gt_sid2cid_per_img, pred_sid2cid_per_img, max_images=50):
    from collections import Counter, defaultdict
    votes = defaultdict(Counter)  # pred_cid -> Counter(gt_cid: pixels)

    used = 0
    for img_id in sample_img_ids[:max_images]:
        gt_png = gt_png_by_imgid.get(img_id)
        pred_png = pred_png_by_imgid.get(img_id)
        if not (gt_png and pred_png and os.path.exists(gt_png) and os.path.exists(pred_png)):
            continue

        gt_idmap   = read_as_idmap(gt_png)
        pred_idmap = read_as_idmap(pred_png)
        if gt_idmap is None or pred_idmap is None:
            continue
        if gt_idmap.shape != pred_idmap.shape:
            # 해상도 다르면 스킵
            continue

        gt_sid2cid   = gt_sid2cid_per_img.get(img_id, {})
        pred_sid2cid = pred_sid2cid_per_img.get(img_id, {})

        # 빠른 교집합 처리를 위해 유니크 세그먼트 id 집합
        pred_sids = np.unique(pred_idmap)
        # 각 pred 세그먼트별로, 해당 픽셀의 GT 세그먼트 id를 따라 GT 카테고리로 투표
        for psid in pred_sids:
            pred_mask = (pred_idmap == psid)
            if pred_mask.sum() == 0: 
                continue
            pcid = pred_sid2cid.get(int(psid))
            if pcid is None:
                continue
            gt_sids_on_mask = gt_idmap[pred_mask]
            if gt_sids_on_mask.size == 0:
                continue
            # gt 세그먼트 id -> gt 카테고리 id로 매핑
            # 최빈의 gt 카테고리에 면적만큼 표 투표
            gt_cids = [gt_sid2cid.get(int(x), None) for x in gt_sids_on_mask.tolist()]
            gt_cids = [g for g in gt_cids if g is not None]
            if not gt_cids:
                continue
            # 다수결
            cnt = Counter(gt_cids)
            # 표를 누적(픽셀수 기반)
            for gcid, n in cnt.items():
                votes[pcid][gcid] += n
        used += 1

    # 최종 매핑: pred_cid -> argmax gt_cid
    mapping = {}
    for pcid, ctr in votes.items():
        if len(ctr) == 0:
            continue
        gcid, _ = ctr.most_common(1)[0]
        mapping[int(pcid)] = int(gcid)

    return mapping, used

def apply_category_remap(pred_annotations, cid_map):
    if not cid_map:
        return pred_annotations
    out = []
    for ann in pred_annotations:
        ann2 = dict(ann)
        segs = []
        for s in ann["segments_info"]:
            s2 = dict(s)
            s2["category_id"] = int(cid_map.get(int(s["category_id"]), int(s["category_id"])))
            segs.append(s2)
        ann2["segments_info"] = segs
        out.append(ann2)
    return out

# -------------------- main --------------------
def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    random.seed(args.seed)

    # 0) GT JSON 로드
    gt = json.load(open(args.gt_json, "r"))

    # id -> (Town, Scenario, stem)
    def stem_from_rgb_path(p): return os.path.basename(p).replace("_leftImg8bit.png", "")
    id2triplet = {
        im["id"]: (im["file_name"].split("/")[0],
                   im["file_name"].split("/")[1],
                   stem_from_rgb_path(im["file_name"]))
        for im in gt["images"]
    }

    # 2) 예측 파일 존재 필터
    have_pred_ids, raw_preds = set(), {}
    for im in gt["images"]:
        img_id = im["id"]
        town, scen, stem = id2triplet[img_id]
        pred_png = os.path.join(args.pred_root, town, scen, f"{stem}_panoptic_pred.png")
        seg_js   = os.path.join(args.pred_root, town, scen, f"{stem}_segments.json")
        if os.path.exists(pred_png) and os.path.exists(seg_js):
            rel_png = os.path.relpath(pred_png, args.pred_root).replace("\\","/")
            raw_preds[img_id] = (rel_png, seg_js)
            have_pred_ids.add(img_id)

    if not have_pred_ids:
        raise RuntimeError("예측 PNG/segments.json 이 하나도 없습니다. --pred_root를 확인하세요.")

    # 3) GT 필터
    gt_images_f = [im for im in gt["images"] if im["id"] in have_pred_ids]
    gt_anns_f   = [ann for ann in gt["annotations"] if ann["image_id"] in have_pred_ids]
    gt_f = {"images": gt_images_f, "annotations": gt_anns_f, "categories": gt["categories"]}

    # 4) GT PNG를 RGB로 정규화해 복제
    fixed_gt_root = os.path.join(args.out_dir, "_fixed_gt_root")
    ensure_dir(fixed_gt_root)

    updated_gt_anns, valid_image_ids = [], set()
    converted, copied, failed = 0, 0, 0
    # image_id -> fixed GT png abs path (리맵용)
    gt_png_by_imgid = {}

    for ann in gt_f["annotations"]:
        rel = ann["file_name"]
        abs_src = os.path.join(args.gt_png_root, rel)
        abs_dst = os.path.join(fixed_gt_root, rel)
        ok, how = materialize_rgb_panoptic(abs_src, abs_dst)
        if not ok:
            failed += 1
            continue
        if how.startswith("converted"): converted += 1
        else: copied += 1

        ann2 = dict(ann)
        ann2["file_name"] = rel
        updated_gt_anns.append(ann2)
        valid_image_ids.add(ann2["image_id"])
        gt_png_by_imgid[ann2["image_id"]] = abs_dst

    gt_f2 = {
        "images": [im for im in gt_f["images"] if im["id"] in valid_image_ids],
        "annotations": updated_gt_anns,
        "categories": gt["categories"],
    }

    # 5) preds 정리(+ area 옵션)
    preds_anns = []
    pred_png_by_imgid = {}
    for img_id in sorted(valid_image_ids):
        rel_png, seg_js = raw_preds[img_id]
        segs = json.load(open(seg_js, "r"))
        safe_segments = []
        for s in segs:
            rec = {
                "id": safe_int(s["id"]),
                "category_id": safe_int(s["category_id"]),
                "iscrowd": safe_int(s.get("iscrowd", 0)),
            }
            safe_segments.append(rec)

        if args.compute_pred_area:
            abs_pred_png = os.path.join(args.pred_root, rel_png)
            areas = compute_segment_areas_from_png(abs_pred_png, safe_segments)
            for r in safe_segments:
                r["area"] = int(areas.get(r["id"], 0))

        preds_anns.append({
            "image_id": safe_int(img_id),
            "file_name": rel_png,
            "segments_info": safe_segments,
        })
        pred_png_by_imgid[img_id] = os.path.join(args.pred_root, rel_png)

    print(f"[GT PNG FIX] converted={converted}, copied={copied}, failed={failed}")
    print(f"[INFO] final images: {len(gt_f2['images'])}, preds: {len(preds_anns)}")

    # ---------- (옵션) 자동 리맵 ----------
    if args.auto_remap_pred_categories:
        # 이미지 id -> {seg_id -> cat_id} 맵 구성
        gt_sid2cid_per_img   = build_gt_sid2cid_map_per_image(gt_f2["annotations"])
        pred_sid2cid_per_img = build_pred_sid2cid_map_per_image(preds_anns)

        # 샘플 이미지 선택
        img_ids_list = list(valid_image_ids)
        random.shuffle(img_ids_list)
        mapping, used = learn_pred_to_gt_cid_map(
            img_ids_list, gt_png_by_imgid, pred_png_by_imgid,
            gt_sid2cid_per_img, pred_sid2cid_per_img, max_images=args.remap_sample
        )
        print(f"[REMAP] learned mapping (pred->GT) from {used} images:")
        for k in sorted(mapping.keys()):
            print(f"  {k} -> {mapping[k]}")

        # 매핑 적용
        preds_anns = apply_category_remap(preds_anns, mapping)

    # 6) JSON 저장
    pred_json_path = os.path.join(args.out_dir, "predictions.json")
    gt_f_json_path = os.path.join(args.out_dir, "filtered_gt_fixed.json")
    with open(pred_json_path, "w") as f:
        json.dump({"annotations": preds_anns}, f, indent=2)
    with open(gt_f_json_path, "w") as f:
        json.dump(gt_f2, f, indent=2)

    print(f"[WRITE] predictions -> {pred_json_path}")
    print(f"[WRITE] filtered+fixed GT -> {gt_f_json_path}")

    # 7) PQ 계산 (panopticapi: 인자 4개 버전)
    res = pq_compute(gt_f_json_path, pred_json_path, fixed_gt_root, args.pred_root)

    out_res = os.path.join(args.out_dir, "pq_results.json")
    with open(out_res, "w") as f:
        json.dump(res, f, indent=2)

    # 8) 요약 TXT
    summary_txt = write_summary_txt(
        args.out_dir, res, len(gt_f2["images"]),
        paths=dict(
            gt_json=args.gt_json,
            gt_png_root=args.gt_png_root,
            pred_root=args.pred_root,
            out_dir=args.out_dir,
            pred_json=pred_json_path,
            gt_fixed_json=gt_f_json_path,
        )
    )
    print(f"[DONE] results -> {out_res}")
    print(f"[SAVED] summary -> {summary_txt}")

if __name__ == "__main__":
    main()
