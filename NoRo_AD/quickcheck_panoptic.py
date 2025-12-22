#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random, sys
import numpy as np
import cv2
from collections import Counter

# ---- COCO rgb2id: panopticapi 없을 때 폴백 ----
try:
    from panopticapi.utils import rgb2id as coco_rgb2id
    def rgb2id(rgb):
        return coco_rgb2id(rgb)
except Exception:
    def rgb2id(rgb):
        # rgb: (H,W,3) uint8, RGB 순서 가정
        r = rgb[..., 0].astype(np.int64)
        g = rgb[..., 1].astype(np.int64)
        b = rgb[..., 2].astype(np.int64)
        return r + (g << 8) + (b << 16)

def fixed_color_for_cat(cid: int) -> tuple:
    # 카테고리 id 고정색 (BGR, OpenCV용)
    rnd = np.random.RandomState(cid * 12345 + 7)
    c = rnd.randint(60, 220, size=3).tolist()
    return (int(c[2]), int(c[1]), int(c[0]))

def draw_overlay(img_bgr, seg_id_map, seg_infos, cat_id_to_name, alpha=0.45):
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    # 색칠
    for s in seg_infos:
        sid = int(s.get("id", 0))
        if sid == 0:  # 배경/void는 스킵
            continue
        m = (seg_id_map == sid)
        if not np.any(m):
            continue
        cid = int(s.get("category_id", 0))
        color = fixed_color_for_cat(cid)
        overlay[m] = (0.55 * np.array(color) + 0.45 * overlay[m]).astype(np.uint8)

    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    # 라벨 텍스트
    for s in seg_infos:
        sid = int(s.get("id", 0))
        if sid == 0:
            continue
        m = (seg_id_map == sid)
        if not np.any(m):
            continue
        ys, xs = np.where(m)
        y1, x1, y2, x2 = ys.min(), xs.min(), ys.max(), xs.max()
        cx, cy = int((x1 + x2) / 2), max(15, int((y1 + y2) / 2))
        cid = int(s.get("category_id", 0))
        name = cat_id_to_name.get(cid, str(cid))
        color = fixed_color_for_cat(cid)
        cv2.putText(out, f"{name}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (/media/.../train_data_3k_2)")
    ap.add_argument("--split", default="val", choices=["train","val"])
    ap.add_argument("--num", type=int, default=20, help="시각화/검증 샘플 수 (0이면 시각화 스킵)")
    ap.add_argument("--out", default="./qc_out", help="오버레이/리포트 저장 폴더")
    ap.add_argument("--stats-only", dest="stats_only", action="store_true",
                    help="통계만 계산(시각화 스킵). 전수 검사 권장.")
    args = ap.parse_args()

    # 디버그: 인자 확인
    # print(vars(args))

    ROOT = args.root
    SPLIT = args.split
    OUT_DIR = args.out
    ensure_dir(OUT_DIR)

    IMG_DIR  = os.path.join(ROOT, "leftImg8bit", SPLIT)
    PAN_DIR  = os.path.join(ROOT, "panoptic_gt_id", SPLIT)
    JSON_FP  = os.path.join(ROOT, "panoptic_json", f"panoptic_{SPLIT}.json")

    if not os.path.isfile(JSON_FP):
        print(f"[FATAL] JSON not found: {JSON_FP}")
        sys.exit(1)

    JJ = load_json(JSON_FP)
    images = JJ.get("images", [])
    anns   = JJ.get("annotations", [])
    cats   = JJ.get("categories", [])
    img_by_id = {im["id"]: im for im in images}
    ann_by_img = {an["image_id"]: an for an in anns}
    cat_id_to_name = {int(c["id"]): c["name"] for c in cats}
    cat_id_to_thing = {int(c["id"]): int(c.get("isthing",0)) for c in cats}

    print(f"[INFO] images={len(images)} annotations={len(anns)} categories={len(cats)} split={SPLIT}")

    # ---- 샘플 리스트 결정 ----
    sample_ids = list(ann_by_img.keys())
    random.shuffle(sample_ids)
    if args.stats_only:
        N = len(sample_ids)               # 전수 통계
    else:
        N = len(sample_ids) if args.num <= 0 else min(args.num, len(sample_ids))
    sample_ids = sample_ids[:N]

    # ---- 리포트 파일 ----
    rpt_path = os.path.join(OUT_DIR, f"report_{SPLIT}.txt")
    with open(rpt_path, "w", encoding="utf-8") as rpt:
        rpt.write(f"[SPLIT] {SPLIT}\nJSON: {JSON_FP}\nIMG_DIR: {IMG_DIR}\nPAN_DIR: {PAN_DIR}\n")
        rpt.write(f"images={len(images)} annotations={len(anns)} categories={len(cats)}\n\n")

        seg_count_by_cat = Counter()
        px_area_by_cat = Counter()
        thing_count = 0
        stuff_count = 0
        bad = 0

        for i, img_id in enumerate(sample_ids, 1):
            iminfo = img_by_id.get(img_id)
            an = ann_by_img.get(img_id)
            if iminfo is None or an is None:
                continue

            img_rel = iminfo["file_name"]
            pan_rel = an["file_name"]
            img_abs = os.path.join(ROOT, img_rel)
            pan_abs = os.path.join(PAN_DIR, pan_rel)

            im = cv2.imread(img_abs, cv2.IMREAD_COLOR)
            panrgb_bgr = cv2.imread(pan_abs, cv2.IMREAD_COLOR)
            if im is None or panrgb_bgr is None:
                rpt.write(f"[MISS] {img_rel} or {pan_rel}\n")
                bad += 1
                continue

            # cv2는 BGR → RGB 변환 후 rgb2id
            pan_id_map = rgb2id(cv2.cvtColor(panrgb_bgr, cv2.COLOR_BGR2RGB)).astype(np.int64)
            png_ids = set(np.unique(pan_id_map).tolist())
            seginfo_ids = {int(s.get("id", 0)) for s in an.get("segments_info", []) if int(s.get("id", 0)) != 0}

            inter = len(png_ids & seginfo_ids)
            miss_in_png = sorted(list(seginfo_ids - png_ids))[:10]
            orphan_in_png = sorted([x for x in (png_ids - seginfo_ids) if x != 0])[:10]

            rpt.write(f"[{i}/{N}] {os.path.basename(img_rel)} | seginfo={len(seginfo_ids)} uniqPNG={len(png_ids)} inter={inter}\n")
            if miss_in_png:
                rpt.write(f"  - seginfo.id but not in PNG: {miss_in_png}\n")
            if orphan_in_png:
                rpt.write(f"  - PNG ids not in seginfo: {orphan_in_png}\n")

            # 통계 누적
            for s in an.get("segments_info", []):
                sid = int(s.get("id", 0))
                if sid == 0:
                    continue
                cid = int(s.get("category_id", 0))
                area = int(s.get("area", 0))
                seg_count_by_cat[cid] += 1
                px_area_by_cat[cid] += area
                if cat_id_to_thing.get(cid, 0) == 1:
                    thing_count += 1
                else:
                    stuff_count += 1

            # 시각화 저장 (stats-only면 스킵)
            if not args.stats_only:
                overlay = draw_overlay(im, pan_id_map, an.get("segments_info", []), cat_id_to_name)
                save_p = os.path.join(OUT_DIR, f"{os.path.splitext(os.path.basename(img_rel))[0]}_overlay.png")
                cv2.imwrite(save_p, overlay)

        # 최종 통계 요약
        rpt.write("\n=== SUMMARY ===\n")
        rpt.write(f"checked_samples={N}, missing_files={bad}\n")
        rpt.write(f"thing_segments={thing_count}, stuff_segments={stuff_count}\n")
        rpt.write(f"per-category counts & pixel-areas:\n")
        for cid in sorted(cat_id_to_name.keys()):
            name = cat_id_to_name[cid]
            rpt.write(f"  - [{cid:02d}] {name:>12s}  segs={seg_count_by_cat[cid]:6d}  px={px_area_by_cat[cid]:10d}\n")

    print(f"[DONE] report -> {rpt_path}")
    if not args.stats_only:
        print(f"[DONE] overlays -> {OUT_DIR}")

if __name__ == "__main__":
    main()
