#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WildDash2 panoptic.json -> Cityscapes 19-class 전용 panoptic_cs19_mapped.json 생성

핵심:
1) wilddash2-meta.json 안에서
   - label "name" 기반으로 wd2_label_id -> Cityscapes trainId(0~18) 매핑 테이블을 만든다.
2) panoptic.json의 segments_info.category_id를
   - wd2_label_id -> cs_trainId 로 치환한다.
3) categories 는 Cityscapes 19 클래스로 덮어쓴다.
"""

import os
import json

ROOT = "/media/vip-dell/HC/wd_public_v2p0"

SRC_PAN_JSON = os.path.join(ROOT, "panoptic.json")
META_JSON    = os.path.join(ROOT, "wilddash2-meta.json")
DST_PAN_JSON = os.path.join(ROOT, "panoptic_cs19_mapped.json")

# --- Cityscapes 19 trainId 정의 (id == trainId) ---
CITYSCAPES_19 = [
    (0,  "road",          0),
    (1,  "sidewalk",      0),
    (2,  "building",      0),
    (3,  "wall",          0),
    (4,  "fence",         0),
    (5,  "pole",          0),
    (6,  "traffic light", 1),
    (7,  "traffic sign",  1),
    (8,  "vegetation",    0),
    (9,  "terrain",       0),
    (10, "sky",           0),
    (11, "person",        1),
    (12, "rider",         1),
    (13, "car",           1),
    (14, "truck",         1),
    (15, "bus",           1),
    (16, "train",         1),
    (17, "motorcycle",    1),
    (18, "bicycle",       1),
]

# WD2 label "name" → Cityscapes trainId
WD2_NAME_TO_CS_TRAINID = {
    "road":            0,
    "sidewalk":        1,
    "building":        2,
    "wall":            3,
    "fence":           4,
    "pole":            5,
    "trafficlight":    6,   # WD2 이름
    "trafficsignfront":7,   # WD2 이름
    "vegetation":      8,
    "terrain":         9,
    "sky":             10,
    "person":          11,
    "rider":           12,
    "car":             13,
    "truck":           14,
    "bus":             15,
    "train":           16,
    "motorcycle":      17,
    "bicycle":         18,
}


def main():
    # ------------------------------------------------------
    # 1) wilddash2-meta.json 에서 wd2_id -> cs_trainId 맵핑 만들기 (이름 기반)
    # ------------------------------------------------------
    assert os.path.isfile(META_JSON), f"meta 파일 없음: {META_JSON}"
    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    labels = meta.get("labels", meta.get("classes", []))
    if not labels:
        raise RuntimeError("wilddash2-meta.json 안에 'labels' 또는 'classes' 키가 없음. 구조 먼저 확인 필요.")

    wd2_to_cs = {}  # wd2_label_id -> cs_trainId(0~18) or 255(ignore)

    for lb in labels:
        wid = lb["id"]
        name = lb["name"]
        cs_tid = WD2_NAME_TO_CS_TRAINID.get(name, 255)
        wd2_to_cs[wid] = cs_tid

    print("[MAP] WD2 label id -> Cityscapes trainId (0~18, 255=ignore):")
    for k in sorted(wd2_to_cs.keys()):
        print(f"  wd2_id={k:2d} ({labels[k]['name']}) -> cs_trainId={wd2_to_cs[k]}")

    # ------------------------------------------------------
    # 2) 원본 panoptic.json을 읽어서 segments_info.category_id를 치환
    # ------------------------------------------------------
    assert os.path.isfile(SRC_PAN_JSON), f"원본 panoptic.json 없음: {SRC_PAN_JSON}"
    with open(SRC_PAN_JSON, "r", encoding="utf-8") as f:
        j = json.load(f)

    anns = j.get("annotations", [])
    new_anns = []
    total_segments = 0
    dropped_segments = 0

    for ann in anns:
        segs = ann.get("segments_info", [])
        total_segments += len(segs)
        new_segs = []

        for s in segs:
            wd_cid = s.get("category_id", -1)
            cs_tid = wd2_to_cs.get(wd_cid, 255)
            if cs_tid == 255:
                # Cityscapes 19에 직접 대응 안 되는 라벨 → PQ/mIoU에서 무시
                dropped_segments += 1
                continue

            ns = dict(s)
            ns["category_id"] = cs_tid  # ★ WD2 id → Cityscapes trainId 로 치환
            new_segs.append(ns)

        ann_new = dict(ann)
        ann_new["segments_info"] = new_segs
        new_anns.append(ann_new)

    print(f"[INFO] 원본 segments_info 총 개수: {total_segments}")
    print(f"[INFO] Cityscapes 19에 매핑 안 돼서 드롭된 segment 수: {dropped_segments}")

    # ------------------------------------------------------
    # 3) categories 는 아예 Cityscapes 19개로 덮어쓰기
    # ------------------------------------------------------
    new_cats = []
    for tid, name, isthing in CITYSCAPES_19:
        new_cats.append({
            "id": tid,        # trainId == id
            "name": name,
            "isthing": isthing,
        })

    out = {
        "images": j.get("images", []),
        "annotations": new_anns,
        "categories": new_cats,
    }

    print(f"[SAVE] Cityscapes 19 전용 panoptic json: {DST_PAN_JSON}")
    with open(DST_PAN_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[DONE] WildDash2 -> Cityscapes19 서브셋 JSON 생성 완료.")


if __name__ == "__main__":
    main()
