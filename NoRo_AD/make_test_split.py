# make_test_split.py
# - 기존 train/val 생성 로직을 그대로 이용해서
#   "train, val에서 쓰지 않은 남은 프레임" 중에서
#   타운×날씨 균등 샘플링으로 test 세트를 만든다.
# - 출력 루트는 /media/vip-dell/HC/train_bh_test
import os, glob, json, shutil, random, cv2
import numpy as np
from tqdm import tqdm

# ======= 기존 설정 복붙 (필요한 부분) =======
IN_ROOT   = "/media/vip-dell/HC/train_data"          # 원본 (Town01~Town05)
OUT_ROOT  = "/media/vip-dell/HC/train_data_3k_2"     # 기존 train/val 출력 루트
OUT_ROOT_TEST = "/media/vip-dell/HC/train_bh_test"   # ★ test 출력 루트

TRAIN_TOWNS = ["Town01","Town02","Town03","Town04"]
VAL_TOWNS   = ["Town05"]
SCENARIOS   = ["SUNNY_GLARE_DAY","SUPER_FOG","HARD_RAIN_WET_DAY","HARD_RAIN_WET_NIGHT"]

# 기존에 만든 개수(참조용): train=3000, val=500
TEST_TOTAL  = 500         # ★ 원하는 test 총 이미지 수
RANDOM_SEED = 42

THING_TRAINIDS = set([6,7,11,12,13,14,15,16,17,18])
APPLY_AUTO_MAP = True

CARLA_NEW_TO_TRAIN19 = {
     1: 0,  24: 0,
     2: 1,
     3: 2,  4: 3, 5: 4,
     6: 5,  7: 6,  8: 7,
     9: 8, 10: 9, 11:10,
    12:11, 13:12,
    14:13, 15:14, 16:15,
    17:16, 18:17, 19:18
}
CARLA_OLD_TO_TRAIN19 = {
     7: 0, 6: 0,  8: 1, 1: 2, 11: 3, 2: 4,
     5: 5, 18: 6, 12: 7, 9: 8, 22: 9, 13:10,
     4:11, 10:13
}

CATEGORIES = [
    {"id": 0, "name":"road", "isthing":0}, {"id": 1, "name":"sidewalk","isthing":0},
    {"id": 2, "name":"building","isthing":0}, {"id": 3, "name":"wall","isthing":0},
    {"id": 4, "name":"fence","isthing":0}, {"id": 5, "name":"pole","isthing":0},
    {"id": 6, "name":"traffic light","isthing":1}, {"id": 7, "name":"traffic sign","isthing":1},
    {"id": 8, "name":"vegetation","isthing":0}, {"id": 9, "name":"terrain","isthing":0},
    {"id":10, "name":"sky","isthing":0}, {"id":11, "name":"person","isthing":1},
    {"id":12, "name":"rider","isthing":1}, {"id":13, "name":"car","isthing":1},
    {"id":14, "name":"truck","isthing":1}, {"id":15, "name":"bus","isthing":1},
    {"id":16, "name":"train","isthing":1}, {"id":17, "name":"motorcycle","isthing":1},
    {"id":18, "name":"bicycle","isthing":1},
]

import numpy as np, glob, os, shutil
def safe_link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst): return
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)

def bbox_from_mask(m):
    ys, xs = np.where(m)
    if len(xs) == 0: return [0,0,0,0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)]

def list_bases_with_all(town, scenario):
    left_dir = os.path.join(IN_ROOT, town, scenario, "leftImg8bit")
    sem_dir  = os.path.join(IN_ROOT, town, scenario, "gtFine")
    pano_dir = os.path.join(IN_ROOT, town, scenario, "panoptic")
    if not (os.path.isdir(left_dir) and os.path.isdir(sem_dir) and os.path.isdir(pano_dir)):
        print(f"[WARN] Dirs missing in {town}/{scenario}"); return [], left_dir, sem_dir, pano_dir
    left_bases = set(os.path.basename(p).replace("_leftImg8bit.png","")
                     for p in glob.glob(os.path.join(left_dir, "*_leftImg8bit.png")))
    sem_bases  = set(os.path.basename(p).replace("_gtFine_trainIds19.png","")
                     for p in glob.glob(os.path.join(sem_dir, "*_gtFine_trainIds19.png")))
    pano_bases = set(os.path.basename(p).replace("_panopticId.png","")
                     for p in glob.glob(os.path.join(pano_dir, "*_panopticId.png")))
    bases = sorted(left_bases & sem_bases & pano_bases)
    if len(bases)==0 and len(left_bases)>0:
        print(f"[WARN] No matching GTs for {town}/{scenario}. Check suffixes.")
    return bases, left_dir, sem_dir, pano_dir

def even_split(total, n):
    base = total // n; rem = total % n
    return [base + (1 if i<rem else 0) for i in range(n)]

TRAIN19_SET = set(range(19))
VALID_TRAIN19 = TRAIN19_SET | {255}
def is_already_train19(arr: np.ndarray) -> bool:
    return set(np.unique(arr).tolist()).issubset(VALID_TRAIN19)

def build_lut(mapping: dict, default_val: int = 255) -> np.ndarray:
    lut = np.full(65536, default_val, dtype=np.uint16)
    for s,d in mapping.items(): lut[int(s)] = int(d)
    return lut
LUT_NEW = build_lut(CARLA_NEW_TO_TRAIN19, 255)
LUT_OLD = build_lut(CARLA_OLD_TO_TRAIN19, 255)

def map_to_train19(arr: np.ndarray, strategy="auto") -> np.ndarray:
    arr = arr.astype(np.uint16, copy=False)
    if strategy=="auto":
        if is_already_train19(arr): return arr
        uniq = set(np.unique(arr).tolist())
        mapped = LUT_NEW[arr] if (24 in uniq or 19 in uniq or 17 in uniq) else LUT_OLD[arr]
        return mapped
    elif strategy=="new": return LUT_NEW[arr]
    elif strategy=="old": return LUT_OLD[arr]
    else: raise ValueError

from panopticapi.utils import id2rgb

def write_coco_panoptic(split_name, samples, index, OUT_DIR):
    out_img_dir  = os.path.join(OUT_DIR, "leftImg8bit", split_name)
    out_sem_dir  = os.path.join(OUT_DIR, "gtFine", split_name)
    out_pano_dir = os.path.join(OUT_DIR, "panoptic_gt_id", split_name)
    out_json     = os.path.join(OUT_DIR, "panoptic_json", f"panoptic_{split_name}.json")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_sem_dir, exist_ok=True)
    os.makedirs(out_pano_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    images, annotations = [], []
    for (town, scn, base) in tqdm(samples, desc=f"[{split_name}] copy+convert"):
        left_dir = index[(town, scn)]["left_dir"]
        sem_dir  = index[(town, scn)]["sem_dir"]
        pano_dir = index[(town, scn)]["pano_dir"]

        rgb_src   = os.path.join(left_dir, f"{base}_leftImg8bit.png")
        sem_src   = os.path.join(sem_dir,  f"{base}_gtFine_trainIds19.png")
        pan16_src = os.path.join(pano_dir, f"{base}_panopticId.png")

        rgb_name    = f"{town}_{scn}_{base}_leftImg8bit.png"
        sem_name    = f"{town}_{scn}_{base}_gtFine_trainIds19.png"
        pan_gt_name = f"{town}_{scn}_{base}_panopticGT.png"

        rgb_dst    = os.path.join(out_img_dir,  rgb_name)
        sem_dst    = os.path.join(out_sem_dir,  sem_name)
        pan_gt_dst = os.path.join(out_pano_dir, pan_gt_name)

        if not (os.path.exists(rgb_src) and os.path.exists(sem_src) and os.path.exists(pan16_src)):
            print(f"[SKIP] Missing files for {town}/{scn}/{base}"); continue

        safe_link_or_copy(rgb_src, rgb_dst)
        safe_link_or_copy(sem_src, sem_dst)

        pan16 = cv2.imread(pan16_src, cv2.IMREAD_UNCHANGED)
        sem19 = cv2.imread(sem_src,   cv2.IMREAD_UNCHANGED)
        if pan16 is None or sem19 is None:
            print(f"[SKIP] Failed to read GTs for {pan16_src} or {sem_src}"); continue
        if pan16.ndim != 2:
            pan16 = pan16[...,0]

        if APPLY_AUTO_MAP:
            sem19 = map_to_train19(sem19, "auto")

        H, W = pan16.shape[:2]
        void_mask = (sem19 == 255)
        if np.any(void_mask):
            pan16 = pan16.copy(); pan16[void_mask] = 0
            sem19 = sem19.copy(); sem19[void_mask] = 0

        max_sid = int(pan16.max()) if pan16.size else 0
        if max_sid >= (1<<24):
            raise ValueError(f"[FATAL] seg_id >= 2^24: max={max_sid}")

        if not os.path.exists(pan_gt_dst):
            rgb = id2rgb(pan16.astype(np.int64))
            cv2.imwrite(pan_gt_dst, rgb[..., ::-1])

        img_id = f"{town}_{scn}_{base}"
        images.append({
            "id": img_id,
            "file_name": os.path.relpath(rgb_dst, start=OUT_DIR).replace("\\","/"),
            "height": H, "width": W,
        })

        seg_infos = []
        for sid in np.unique(pan16):
            sid = int(sid)
            if sid == 0: continue
            m = (pan16 == sid)
            ar = int(m.sum()) 
            if ar == 0: continue
            tid = int(np.bincount(sem19[m]).argmax())
            if tid == 255: tid = 0
            if not (0 <= tid <= 18): continue
            seg_infos.append({
                "id": sid,
                "category_id": tid,
                "isthing": 1 if tid in THING_TRAINIDS else 0,
                "area": ar,
                "bbox": bbox_from_mask(m),
                "iscrowd": 0,
            })

        annotations.append({
            "image_id": img_id,
            "file_name": os.path.relpath(pan_gt_dst, start=out_pano_dir).replace("\\","/"),
            "segments_info": seg_infos
        })

    with open(out_json, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES},
                  f, ensure_ascii=False)
    print(f"✔ {split_name}: images={len(images)}  json→ {out_json}")
    print(f"  > RGB images: {out_img_dir}")
    print(f"  > Panoptic GT (ID-RGB): {out_pano_dir}")
    print(f"  > Semantic GT (trainId19): {out_sem_dir}")

def main():
    # 1) 모든 프레임 인덱스
    index = {}
    all_towns = sorted(list(set(TRAIN_TOWNS + VAL_TOWNS)))
    for town in tqdm(all_towns, desc="Indexing"):
        for scn in SCENARIOS:
            bases, left_dir, sem_dir, pano_dir = list_bases_with_all(town, scn)
            index[(town, scn)] = {
                "bases": bases,
                "left_dir": left_dir,
                "sem_dir":  sem_dir,
                "pano_dir": pano_dir
            }

    # 2) 기존 train/val JSON에서 사용된 이미지 id를 수집 (중복 제외용)
    used_set = set()
    for split in ["train","val"]:
        json_path = os.path.join(OUT_ROOT, "panoptic_json", f"panoptic_{split}.json")
        if not os.path.isfile(json_path):
            print(f"[WARN] cannot find {json_path}; skip dedup from {split}")
            continue
        with open(json_path,"r") as f:
            jj = json.load(f)
        for im in jj.get("images", []):
            # 이미지 id는 f"{town}_{scn}_{base}"
            used_set.add(im["id"])

    # 3) 남은 풀(pool)에서 town×scenario 균등 샘플링
    random.seed(RANDOM_SEED)
    per_scn = even_split(TEST_TOTAL, len(SCENARIOS))
    samples = []
    for scn, scn_need in zip(SCENARIOS, per_scn):
        per_town = even_split(scn_need, len(all_towns))
        for town, need in zip(all_towns, per_town):
            pool = []
            for b in index[(town, scn)]["bases"]:
                img_id = f"{town}_{scn}_{b}"
                if img_id not in used_set:  # train/val에 없는 것만
                    pool.append(b)
            if len(pool) == 0:
                print(f"[WARN] no remaining frames for {town}/{scn}")
                continue
            if len(pool) < need:
                print(f"[WARN] {town}/{scn}: need {need}, have {len(pool)}. sampling with replacement.")
                pick = random.choices(pool, k=need)
            else:
                pick = random.sample(pool, need)
            for b in pick:
                samples.append((town, scn, b))

    # 4) test COCO-panoptic 생성 (OUT_ROOT_TEST에 저장)
    write_coco_panoptic("test", samples, index, OUT_ROOT_TEST)
    print("DONE (test).")

if __name__ == "__main__":
    main()
