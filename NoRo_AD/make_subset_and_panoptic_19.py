# import os, json, glob, cv2, numpy as np, random, shutil
# from tqdm import tqdm
# from collections import defaultdict
# from panopticapi.utils import id2rgb  # COCO 표준 변환 (seg_id -> RGB)

# # ================== 설정 ==================
# IN_ROOT  = "/media/vip-dell/HC/train_data"
# # [사용자 요청] 출력 디렉토리 변경
# OUT_ROOT = "/media/vip-dell/HC/final_dataset2"

# # [개선] 원본 파일 접미사 (유지보수용)
# IN_SEM_SUFFIX = "_gtFine_trainIds19.png" # 원본 시맨틱 라벨 파일
# IN_PAN_SUFFIX = "_panopticId.png"        # 원본 panoptic ID 파일

# TRAIN_TOWNS = ["Town01","Town02","Town03","Town04"]
# VAL_TOWNS   = ["Town05"]
# SCENARIOS   = ["SUNNY_GLARE_DAY","SUPER_FOG","HARD_RAIN_WET_DAY","HARD_RAIN_WET_NIGHT"]

# TRAIN_TOTAL = 3000
# VAL_TOTAL   = 500
# RANDOM_SEED = 42

# # Cityscapes-19 thing trainIds
# THING_TRAINIDS = set([6,7,11,12,13,14,15,16,17,18])

# # ---- (중요) sem19가 원본 라벨일 수도 있으니 자동 매핑 지원 ----
# APPLY_AUTO_MAP = True  # True 권장: 이미 0..18/255면 그대로 통과, 아니면 NEW/OLD 중 하나로 변환

# # [사용자 요청] CARLA → trainId19 매핑 테이블 (NEW/OLD)
# # ★[수정] 누락된 'ground'(25)와 'guard rail'(28) 추가
# CARLA_NEW_TO_TRAIN19 = {
#     1: 0, 24: 0,  # Road, RoadLine
#     2: 1,         # Sidewalk
#     3: 2,         # Building
#     4: 3,         # Wall
#     5: 4, 28: 4,  # Fence, Guard Rail (★추가)
#     6: 5,         # Pole
#     7: 6,         # TrafficLight
#     8: 7,         # TrafficSign
#     9: 8,         # Vegetation
#     10: 9, 25: 9, # Terrain, Ground (★추가)
#     11: 10,       # Sky
#     12: 11,       # Pedestrian
#     13: 12,       # Rider
#     14: 13,       # Car
#     15: 14,       # Truck
#     16: 15,       # Bus
#     17: 16,       # Train
#     18: 17,       # Motorcycle
#     19: 18        # Bicycle
#     # 0, 20-23, 26, 27은 build_lut의 default_val(255)로 자동 매핑됨
# }
# CARLA_OLD_TO_TRAIN19 = {
#      7: 0, 6: 0,  8: 1, 1: 2, 11: 3, 2: 4,
#      5: 5, 18: 6, 12: 7, 9: 8, 22: 9, 13:10,
#      4:11, 10:13
# }

# CATEGORIES = [
#     {"id": 0, "name":"road", "isthing":0}, {"id": 1, "name":"sidewalk","isthing":0},
#     {"id": 2, "name":"building","isthing":0}, {"id": 3, "name":"wall","isthing":0},
#     {"id": 4, "name":"fence","isthing":0}, {"id": 5, "name":"pole","isthing":0},
#     {"id": 6, "name":"traffic light","isthing":1}, {"id": 7, "name":"traffic sign","isthing":1},
#     {"id": 8, "name":"vegetation","isthing":0}, {"id": 9, "name":"terrain","isthing":0},
#     {"id":10, "name":"sky","isthing":0}, {"id":11, "name":"person","isthing":1},
#     {"id":12, "name":"rider","isthing":1}, {"id":13, "name":"car","isthing":1},
#     {"id":14, "name":"truck","isthing":1}, {"id":15, "name":"bus","isthing":1},
#     {"id":16, "name":"train","isthing":1}, {"id":17, "name":"motorcycle","isthing":1},
#     {"id":18, "name":"bicycle","isthing":1},
# ]
# # ==========================================

# def safe_link_or_copy(src, dst):
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     if os.path.exists(dst):
#         return
#     try:
#         os.link(src, dst)  # 하드링크 (빠름, 같은 파티션일 때)
#     except OSError:
#         shutil.copy2(src, dst)  # 파티션 다르면 복사

# def bbox_from_mask(m):
#     ys, xs = np.where(m)
#     if len(xs) == 0:
#         return [0, 0, 0, 0]
#     x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
#     return [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]

# def list_bases_with_all(town, scenario):
#     left_dir = os.path.join(IN_ROOT, town, scenario, "leftImg8bit")
#     sem_dir  = os.path.join(IN_ROOT, town, scenario, "gtFine")      # 시맨틱 GT (trainId19 or 원본)
#     pano_dir = os.path.join(IN_ROOT, town, scenario, "panoptic")    # panopticId (정수 세그먼트 ID)

#     if not (os.path.isdir(left_dir) and os.path.isdir(sem_dir) and os.path.isdir(pano_dir)):
#         print(f"[WARN] Dirs missing in {town}/{scenario}")
#         return [], left_dir, sem_dir, pano_dir

#     left_bases = set(os.path.basename(p).replace("_leftImg8bit.png","")
#                        for p in glob.glob(os.path.join(left_dir, "*_leftImg8bit.png")))

#     # [개선] 하드코딩된 접미사 대신 변수 사용
#     sem_bases  = set(os.path.basename(p).replace(IN_SEM_SUFFIX,"")
#                        for p in glob.glob(os.path.join(sem_dir, f"*{IN_SEM_SUFFIX}")))
#     pano_bases = set(os.path.basename(p).replace(IN_PAN_SUFFIX,"")
#                        for p in glob.glob(os.path.join(pano_dir, f"*{IN_PAN_SUFFIX}")))

#     bases = sorted(left_bases & sem_bases & pano_bases)
#     if len(bases) == 0 and len(left_bases) > 0:
#         print(f"[WARN] No matching GTs for {town}/{scenario}. Check suffixes ({IN_SEM_SUFFIX} / {IN_PAN_SUFFIX}).")

#     return bases, left_dir, sem_dir, pano_dir

# def even_split(total, n):
#     base = total // n
#     rem  = total % n
#     return [base + (1 if i < rem else 0) for i in range(n)]

# # ---------- trainId19 자동 매핑 유틸 ----------
# TRAIN19_SET = set(range(19))
# VALID_TRAIN19 = TRAIN19_SET | {255}  # 255=void

# def is_already_train19(arr: np.ndarray) -> bool:
#     """라벨 배열이 이미 0..18/255 형태인지 검사"""
#     uniq = set(np.unique(arr).tolist())
#     return uniq.issubset(VALID_TRAIN19)

# def build_lut(mapping: dict, default_val: int = 255) -> np.ndarray:
#     """0..65535 범위 LUT 생성 (원본 라벨 → trainId19/255)"""
#     lut = np.full(65536, default_val, dtype=np.uint16)
#     for src, dst in mapping.items():
#         lut[int(src)] = int(dst)
#     return lut

# LUT_NEW = build_lut(CARLA_NEW_TO_TRAIN19, default_val=255)
# LUT_OLD = build_lut(CARLA_OLD_TO_TRAIN19, default_val=255)

# def map_to_train19(arr: np.ndarray, strategy: str = "auto") -> np.ndarray:
#     """
#     arr: 원본 시맨틱 라벨(정수)
#     strategy: "auto" | "new" | "old"
#       - auto: 이미 trainId19면 그대로 반환
#               아니면 힌트 규칙으로 new/old 추정 후 매핑
#     """
#     arr = arr.astype(np.uint16, copy=False)
#     if strategy == "auto":
#         if is_already_train19(arr):
#             return arr  # 이미 0..18/255
#         uniq = set(np.unique(arr).tolist())
#         # 힌트: NEW 체계에 24,19,17 등이 자주 등장
#         if 24 in uniq or 19 in uniq or 17 in uniq:
#             mapped = LUT_NEW[arr]
#         else:
#             mapped = LUT_OLD[arr]
#         return mapped
#     elif strategy == "new":
#         return LUT_NEW[arr]
#     elif strategy == "old":
#         return LUT_OLD[arr]
#     else:
#         raise ValueError("strategy must be one of {'auto','new','old'}")

# # -------------------------------------------

# def sample_subset(total, towns, scenarios, index):
#     random.seed(RANDOM_SEED)
#     per_scn = even_split(total, len(scenarios))
#     samples = []
#     for scn, scn_need in zip(scenarios, per_scn):
#         per_town = even_split(scn_need, len(towns))
#         for town, need in zip(towns, per_town):
#             pool = index[(town, scn)]["bases"]
#             if len(pool) < need:
#                 print(f"[WARN] Not enough frames in {town}/{scn}: need {need}, have {len(pool)}. Sampling with replacement.")
#                 pick = random.choices(pool, k=need)  # 중복 허용
#             else:
#                 pick = random.sample(pool, need)
#             for b in pick:
#                 samples.append((town, scn, b))
#     return samples

# def write_coco_panoptic(split_name, samples, index):
#     out_img_dir  = os.path.join(OUT_ROOT, "leftImg8bit", split_name)
#     out_sem_dir  = os.path.join(OUT_ROOT, "gtFine", split_name)
#     out_pano_dir = os.path.join(OUT_ROOT, "panoptic_gt_id", split_name)
#     out_json     = os.path.join(OUT_ROOT, "panoptic_json", f"panoptic_{split_name}.json")

#     os.makedirs(out_img_dir, exist_ok=True)
#     os.makedirs(out_sem_dir, exist_ok=True)
#     os.makedirs(out_pano_dir, exist_ok=True)
#     os.makedirs(os.path.dirname(out_json), exist_ok=True)

#     images, annotations = [], []

#     for (town, scn, base) in tqdm(samples, desc=f"[{split_name}] copy+convert"):
#         left_dir = index[(town, scn)]["left_dir"]
#         sem_dir  = index[(town, scn)]["sem_dir"]
#         pano_dir = index[(town, scn)]["pano_dir"]

#         rgb_src   = os.path.join(left_dir, f"{base}_leftImg8bit.png")
#         sem_src   = os.path.join(sem_dir,  f"{base}{IN_SEM_SUFFIX}")
#         pan16_src = os.path.join(pano_dir, f"{base}{IN_PAN_SUFFIX}")

#         rgb_name    = f"{town}_{scn}_{base}_leftImg8bit.png"
#         sem_name    = f"{town}_{scn}_{base}_gtFine_trainIds19.png"
#         pan_gt_name = f"{town}_{scn}_{base}_panopticGT.png"

#         rgb_dst    = os.path.join(out_img_dir,  rgb_name)
#         sem_dst    = os.path.join(out_sem_dir,  sem_name)
#         pan_gt_dst = os.path.join(out_pano_dir, pan_gt_name)

#         if not (os.path.exists(rgb_src) and os.path.exists(sem_src) and os.path.exists(pan16_src)):
#             print(f"[SKIP] Missing files for {town}/{scn}/{base}")
#             continue

#         safe_link_or_copy(rgb_src, rgb_dst)

#         pan16 = cv2.imread(pan16_src, cv2.IMREAD_UNCHANGED)
#         sem_orig = cv2.imread(sem_src,   cv2.IMREAD_UNCHANGED)
#         if pan16 is None or sem_orig is None:
#             print(f"[SKIP] Failed to read GTs for {pan16_src} or {sem_src}")
#             continue

#         if pan16.ndim != 2:
#             print(f"[WARN] {pan16_src} is not single-channel. shape={pan16.shape}. Using first channel.")
#             pan16 = pan16[..., 0]

#         if APPLY_AUTO_MAP:
#             sem19 = map_to_train19(sem_orig, strategy="auto")
#         else:
#             sem19 = sem_orig.astype(np.uint16)

#         cv2.imwrite(sem_dst, sem19.astype(np.uint8))

#         H, W = pan16.shape[:2]

#         void_mask = (sem19 == 255)
#         if np.any(void_mask):
#             pan16 = pan16.copy(); pan16[void_mask] = 0
#             sem19 = sem19.copy(); sem19[void_mask] = 0

#         max_sid = int(pan16.max()) if pan16.size else 0
#         if max_sid >= (1 << 24):
#             raise ValueError(f"[FATAL] seg_id >= 2^24 detected: max={max_sid} at {pan16_src}")

#         if not os.path.exists(pan_gt_dst):
#             rgb = id2rgb(pan16.astype(np.int64))
#             cv2.imwrite(pan_gt_dst, rgb[..., ::-1])
#         else:
#             tmp = cv2.imread(pan_gt_dst, cv2.IMREAD_UNCHANGED)
#             if tmp is None:
#                 print(f"[SKIP] Failed to read existing {pan_gt_dst}")
#                 continue

#         img_id = f"{town}_{scn}_{base}"
#         images.append({
#             "id": img_id,
#             "file_name": os.path.relpath(rgb_dst, start=OUT_ROOT).replace("\\","/"),
#             "height": H, "width": W,
#         })

#         seg_infos = []
#         seg_ids = np.unique(pan16)
#         for sid in seg_ids:
#             sid = int(sid)
            
#             # [!!! 수정된 부분 !!!]
#             # 'sid == 0' (도로)를 건너뛰던 버그를 수정합니다.
#             # COCO 파놀틱 형식에서 id 0은 'void'를 의미할 수 있으나,
#             # 우리 데이터 생성 파이프라인(generate_data.py)은 'road'를 
#             # panoptic_id = 0 (trainId=0 * 1000 + instId=0)으로 인코딩합니다.
#             # 따라서 'road'를 JSON에 포함시키려면 이 continue를 제거해야 합니다.
#             # if sid == 0:
#             #     continue
#             # [!!! 수정 완료 !!!]

#             m = (pan16 == sid)
#             ar = int(m.sum())
#             if ar == 0:
#                 continue

#             tid_candidates = sem19[m]
#             if tid_candidates.size == 0:
#                 continue
#             tid = int(np.bincount(tid_candidates).argmax())
#             if tid == 255: 
#                 tid = 0 

#             if not (0 <= tid <= 18):
#                 print(f"[WARN] Invalid tid {tid} from {sem_src} for sid {sid}. Skip this segment.")
#                 continue

#             # `sid=0` (도로)인 경우, tid도 `0`이어야 합니다.
#             if sid == 0 and tid != 0:
#                 print(f"[WARN] sid=0 (road/void) mapped to tid={tid}! Forcing tid=0. @ {img_id}")
#                 tid = 0

#             seg_infos.append({
#                 "id": sid,
#                 "category_id": tid,
#                 "isthing": 1 if tid in THING_TRAINIDS else 0,
#                 "area": ar,
#                 "bbox": bbox_from_mask(m),
#                 "iscrowd": 0,
#             })

#         annotations.append({
#             "image_id": img_id,
#             "file_name": os.path.relpath(pan_gt_dst, start=out_pano_dir).replace("\\","/"),
#             "segments_info": seg_infos
#         })

#     with open(out_json, "w") as f:
#         json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES},
#                   f, ensure_ascii=False)
#     print(f"✔ {split_name}: images={len(images)}   json→ {out_json}")
#     print(f"  > RGB images: {out_img_dir}")
#     print(f"  > Panoptic GT (ID-RGB): {out_pano_dir}")
#     print(f"  > Semantic GT (trainId19): {out_sem_dir}")

# def main():
#     if os.path.exists(OUT_ROOT):
#         print(f"[WARN] Output directory {OUT_ROOT} already exists.")
#         print("         This script will add/overwrite files. Continuing...")

#     index = {}
#     all_towns = sorted(list(set(TRAIN_TOWNS + VAL_TOWNS)))
#     for town in tqdm(all_towns, desc="Indexing"):
#         for scn in SCENARIOS:
#             bases, left_dir, sem_dir, pano_dir = list_bases_with_all(town, scn)
#             index[(town, scn)] = {
#                 "bases": bases,
#                 "left_dir": left_dir,
#                 "sem_dir": sem_dir,
#                 "pano_dir": pano_dir
#             }

#     print("Sampling subsets...")
#     train_samples = sample_subset(TRAIN_TOTAL, TRAIN_TOWNS, SCENARIOS, index)
#     val_samples   = sample_subset(VAL_TOTAL,   VAL_TOWNS,   SCENARIOS, index)

#     write_coco_panoptic("train", train_samples, index)
#     write_coco_panoptic("val",   val_samples,   index)
#     print("DONE.")

# if __name__ == "__main__":
#     main()

import os, json, glob, cv2, numpy as np, shutil
from tqdm import tqdm
from panopticapi.utils import id2rgb  # COCO 표준 변환 (seg_id -> RGB)

# ================== 설정 ==================
IN_ROOT  = "/media/vip-dell/HC/sunny_only"
OUT_ROOT = "/media/vip-dell/HC/final_dataset5"

IN_SEM_SUFFIX = "_gtFine_trainIds19.png"  # 시맨틱 라벨 파일
IN_PAN_SUFFIX = "_panopticId.png"         # panoptic ID 파일

TRAIN_TOWNS = ["Town01","Town02"]
VAL_TOWNS   = ["Town03"]
SCENARIOS   = ["SUNNY_GLARE_DAY"]

# Cityscapes-19 thing trainIds
THING_TRAINIDS = set([6,7,11,12,13,14,15,16,17,18])

APPLY_AUTO_MAP = True  # sem이 이미 0..18/255면 그대로, 아니면 NEW/OLD 매핑

CARLA_NEW_TO_TRAIN19 = {
    1: 0, 24: 0,  # Road, RoadLine
    2: 1,         # Sidewalk
    3: 2,         # Building
    4: 3,         # Wall
    5: 4, 28: 4,  # Fence, Guard Rail
    6: 5,         # Pole
    7: 6,         # TrafficLight
    8: 7,         # TrafficSign
    9: 8,         # Vegetation
    10: 9, 25: 9, # Terrain, Ground
    11: 10,       # Sky
    12: 11,       # Pedestrian
    13: 12,       # Rider
    14: 13,       # Car
    15: 14,       # Truck
    16: 15,       # Bus
    17: 16,       # Train
    18: 17,       # Motorcycle
    19: 18        # Bicycle
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
# ==========================================

def safe_link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)   # 같은 파티션이면 하드링크
    except OSError:
        shutil.copy2(src, dst)

def bbox_from_mask(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]

def list_bases_with_all(town, scenario):
    left_dir = os.path.join(IN_ROOT, town, scenario, "leftImg8bit")
    sem_dir  = os.path.join(IN_ROOT, town, scenario, "gtFine")
    pano_dir = os.path.join(IN_ROOT, town, scenario, "panoptic")

    if not (os.path.isdir(left_dir) and os.path.isdir(sem_dir) and os.path.isdir(pano_dir)):
        print(f"[WARN] Dirs missing in {town}/{scenario}")
        return [], left_dir, sem_dir, pano_dir

    left_bases = set(os.path.basename(p).replace("_leftImg8bit.png","")
                     for p in glob.glob(os.path.join(left_dir, "*_leftImg8bit.png")))
    sem_bases  = set(os.path.basename(p).replace(IN_SEM_SUFFIX,"")
                     for p in glob.glob(os.path.join(sem_dir, f"*{IN_SEM_SUFFIX}")))
    pano_bases = set(os.path.basename(p).replace(IN_PAN_SUFFIX,"")
                     for p in glob.glob(os.path.join(pano_dir, f"*{IN_PAN_SUFFIX}")))

    bases = sorted(left_bases & sem_bases & pano_bases)
    if len(bases) == 0 and len(left_bases) > 0:
        print(f"[WARN] No matching GTs for {town}/{scenario}. Check suffixes ({IN_SEM_SUFFIX} / {IN_PAN_SUFFIX}).")

    return bases, left_dir, sem_dir, pano_dir

# ---------- trainId19 자동 매핑 유틸 ----------
TRAIN19_SET = set(range(19))
VALID_TRAIN19 = TRAIN19_SET | {255}

def is_already_train19(arr: np.ndarray) -> bool:
    uniq = set(np.unique(arr).tolist())
    return uniq.issubset(VALID_TRAIN19)

def build_lut(mapping: dict, default_val: int = 255) -> np.ndarray:
    lut = np.full(65536, default_val, dtype=np.uint16)
    for src, dst in mapping.items():
        lut[int(src)] = int(dst)
    return lut

LUT_NEW = build_lut(CARLA_NEW_TO_TRAIN19, default_val=255)
LUT_OLD = build_lut(CARLA_OLD_TO_TRAIN19, default_val=255)

def map_to_train19(arr: np.ndarray, strategy: str = "auto") -> np.ndarray:
    arr = arr.astype(np.uint16, copy=False)
    if strategy == "auto":
        if is_already_train19(arr):
            return arr
        uniq = set(np.unique(arr).tolist())
        if 24 in uniq or 19 in uniq or 17 in uniq:
            mapped = LUT_NEW[arr]
        else:
            mapped = LUT_OLD[arr]
        return mapped
    elif strategy == "new":
        return LUT_NEW[arr]
    elif strategy == "old":
        return LUT_OLD[arr]
    else:
        raise ValueError("strategy must be one of {'auto','new','old'}")

# ---------- 샘플링: 랜덤 없음, 전부 사용 ----------
def build_samples_for_towns(towns, scenarios, index):
    """
    towns, scenarios 조합에서
    index[(town, scn)]["bases"] 에 있는 걸 '순서대로 전부' 사용
    """
    samples = []
    for scn in scenarios:
        for town in towns:
            bases = index[(town, scn)]["bases"]
            for b in bases:   # sorted 되어 있음
                samples.append((town, scn, b))
    return samples
# -------------------------------------------

def write_coco_panoptic(split_name, samples, index):
    out_img_dir  = os.path.join(OUT_ROOT, "leftImg8bit", split_name)
    out_sem_dir  = os.path.join(OUT_ROOT, "gtFine", split_name)
    out_pano_dir = os.path.join(OUT_ROOT, "panoptic_gt_id", split_name)
    out_json     = os.path.join(OUT_ROOT, "panoptic_json", f"panoptic_{split_name}.json")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_sem_dir, exist_ok=True)
    os.makedirs(out_pano_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    images, annotations = [], []

    if 0 in THING_TRAINIDS:
        print("[WARN] THING_TRAINIDS should not contain 0. Removing it.")
        THING_TRAINIDS.discard(0)
    
    ROAD_NEW_PANOPTIC_ID = 26  # road용 새로운 panoptic id

    for (town, scn, base) in tqdm(samples, desc=f"[{split_name}] copy+convert"):
        left_dir = index[(town, scn)]["left_dir"]
        sem_dir  = index[(town, scn)]["sem_dir"]
        pano_dir = index[(town, scn)]["pano_dir"]

        rgb_src   = os.path.join(left_dir, f"{base}_leftImg8bit.png")
        sem_src   = os.path.join(sem_dir,  f"{base}{IN_SEM_SUFFIX}")
        pan16_src = os.path.join(pano_dir, f"{base}{IN_PAN_SUFFIX}")

        rgb_name    = f"{town}_{scn}_{base}_leftImg8bit.png"
        sem_name    = f"{town}_{scn}_{base}_gtFine_trainIds19.png"
        pan_gt_name = f"{town}_{scn}_{base}_panopticGT.png"

        rgb_dst    = os.path.join(out_img_dir,  rgb_name)
        sem_dst    = os.path.join(out_sem_dir,  sem_name)
        pan_gt_dst = os.path.join(out_pano_dir, pan_gt_name)

        if not (os.path.exists(rgb_src) and os.path.exists(sem_src) and os.path.exists(pan16_src)):
            print(f"[SKIP] Missing files for {town}/{scn}/{base}")
            continue

        safe_link_or_copy(rgb_src, rgb_dst)

        pan16 = cv2.imread(pan16_src, cv2.IMREAD_UNCHANGED)
        sem_orig = cv2.imread(sem_src,   cv2.IMREAD_UNCHANGED)
        if pan16 is None or sem_orig is None:
            print(f"[SKIP] Failed to read GTs for {pan16_src} or {sem_src}")
            continue

        if pan16.ndim != 2:
            print(f"[WARN] {pan16_src} is not single-channel. shape={pan16.shape}. Using first channel.")
            pan16 = pan16[..., 0]

        if APPLY_AUTO_MAP:
            sem19 = map_to_train19(sem_orig, strategy="auto")
        else:
            sem19 = sem_orig.astype(np.uint16)

        cv2.imwrite(sem_dst, sem19.astype(np.uint8))

        H, W = pan16.shape[:2]

        # road / void 재매핑
        pan16_remapped = pan16.copy()
        void_mask = (sem19 == 255)
        road_mask = (pan16_remapped == 0) & (~void_mask)

        pan16_remapped[void_mask] = 0
        pan16_remapped[road_mask] = ROAD_NEW_PANOPTIC_ID

        sem19_safe = sem19.copy()
        sem19_safe[void_mask] = 0

        max_sid = int(pan16_remapped.max()) if pan16_remapped.size else 0
        if max_sid >= (1 << 24):
            raise ValueError(f"[FATAL] seg_id >= 2^24 detected: max={max_sid} at {pan16_src}")

        if not os.path.exists(pan_gt_dst):
            rgb = id2rgb(pan16_remapped.astype(np.int64))
            cv2.imwrite(pan_gt_dst, rgb[..., ::-1])
        else:
            tmp = cv2.imread(pan_gt_dst, cv2.IMREAD_UNCHANGED)
            if tmp is None:
                print(f"[SKIP] Failed to read existing {pan_gt_dst}")
                continue

        img_id = f"{town}_{scn}_{base}"
        images.append({
            "id": img_id,
            "file_name": os.path.relpath(rgb_dst, start=OUT_ROOT).replace("\\","/"),
            "height": H, "width": W,
        })

        seg_infos = []
        seg_ids = np.unique(pan16_remapped)
        for sid in seg_ids:
            sid = int(sid)
            if sid == 0:
                continue  # panoptic void

            m = (pan16_remapped == sid)
            ar = int(m.sum())
            if ar == 0:
                continue

            tid_candidates = sem19_safe[m]
            if tid_candidates.size == 0:
                continue
            tid = int(np.bincount(tid_candidates).argmax())
            if tid == 255:
                tid = 0

            if not (0 <= tid <= 18):
                print(f"[WARN] Invalid tid {tid} from {sem_src} for sid {sid}. Skip this segment.")
                continue

            if sid == ROAD_NEW_PANOPTIC_ID and tid != 0:
                print(f"[WARN] sid={ROAD_NEW_PANOPTIC_ID} (road) mapped to tid={tid}! Forcing tid=0. @ {img_id}")
                tid = 0
            elif sid != ROAD_NEW_PANOPTIC_ID and tid == 0:
                print(f"[WARN] Non-road sid={sid} mapped to tid=0! (Likely void overlap). Skipping segment. @ {img_id}")
                continue

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
    print(f"✔ {split_name}: images={len(images)}   json→ {out_json}")
    print(f"  > RGB images: {out_img_dir}")
    print(f"  > Panoptic GT (ID-RGB): {out_pano_dir}")
    print(f"  > Semantic GT (trainId19): {out_sem_dir}")

def main():
    if os.path.exists(OUT_ROOT):
        print(f"[WARN] Output directory {OUT_ROOT} already exists.")
        print("         This script will add/overwrite files. Continuing...")

    index = {}
    all_towns = sorted(list(set(TRAIN_TOWNS + VAL_TOWNS)))
    for town in tqdm(all_towns, desc="Indexing"):
        for scn in SCENARIOS:
            bases, left_dir, sem_dir, pano_dir = list_bases_with_all(town, scn)
            index[(town, scn)] = {
                "bases": bases,
                "left_dir": left_dir,
                "sem_dir": sem_dir,
                "pano_dir": pano_dir
            }

    # 여기서부터는 랜덤 없음, 전부 사용
    train_samples = build_samples_for_towns(TRAIN_TOWNS, SCENARIOS, index)
    val_samples   = build_samples_for_towns(VAL_TOWNS,   SCENARIOS, index)

    print(f"[SUMMARY] train samples: {len(train_samples)}, val samples: {len(val_samples)}")

    write_coco_panoptic("train", train_samples, index)
    write_coco_panoptic("val",   val_samples,   index)
    print("DONE.")

if __name__ == "__main__":
    main()
