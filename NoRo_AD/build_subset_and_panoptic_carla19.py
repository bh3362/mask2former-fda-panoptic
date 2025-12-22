# build_subset_and_panoptic_carla19.py
# 60k 원본에서 균등 샘플링하여 /train_data_3k 구조로 내보내고,
# Detectron2 COCO Panoptic(JSON)까지 한 번에 생성 (19-class, PQ 평가용)

import os, json, glob, cv2, numpy as np, random, shutil
from tqdm import tqdm
from collections import defaultdict

# ================== 사용자 설정 ==================
# 원본(60k) 루트: TownXX/SCENARIO/{leftImg8bit, panoptic}
IN_ROOT   = "/media/vip-dell/HC/train_data"

# 출력(3k) 루트: 우리가 학습에 쓰는 폴더
OUT_ROOT  = "/media/vip-dell/HC/train_data_3k_2"

TRAIN_TOWNS = ["Town01", "Town02", "Town03", "Town04"]
VAL_TOWNS   = ["Town05"]

SCENARIOS = ["SUNNY_GLARE_DAY", "SUPER_FOG", "HARD_RAIN_WET_DAY", "HARD_RAIN_WET_NIGHT"]

# Train 2975장 분할(시나리오별)
TRAIN_SCN_TARGET = {
    "SUNNY_GLARE_DAY":    744,
    "SUPER_FOG":          744,
    "HARD_RAIN_WET_DAY":  744,
    "HARD_RAIN_WET_NIGHT":743,
}
# Val 500장(시나리오별 125 균등)
VAL_PER_SCENARIO = 125

RANDOM_SEED = 42
THING_TRAINIDS = set([6,7,11,12,13,14,15,16,17,18])

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
# =================================================

def id_to_rgb(seg_id: int):
    r = seg_id % 256
    g = (seg_id // 256) % 256
    b = (seg_id // 65536) % 256
    return (r, g, b)

def bbox_from_mask(m):
    ys, xs = np.where(m)
    if len(xs)==0: return [0,0,0,0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)]

def list_frames(town, scenario):
    left_dir = os.path.join(IN_ROOT, town, scenario, "leftImg8bit")
    pano_dir = os.path.join(IN_ROOT, town, scenario, "panoptic")
    if not (os.path.isdir(left_dir) and os.path.isdir(pano_dir)):
        return [], left_dir, pano_dir
    left_bases = set(os.path.basename(p).replace("_leftImg8bit.png","")
                     for p in glob.glob(os.path.join(left_dir, "*_leftImg8bit.png")))
    pano_bases = set(os.path.basename(p).replace("_panopticId.png","")
                     for p in glob.glob(os.path.join(pano_dir, "*_panopticId.png")))
    bases = sorted(left_bases & pano_bases)
    return bases, left_dir, pano_dir

def split_quota(total, towns):
    n = len(towns); base = total // n; rem  = total % n
    return {town: (base + (1 if i < rem else 0)) for i, town in enumerate(towns)}

def sample_frames():
    random.seed(RANDOM_SEED)
    index = {}
    for town in TRAIN_TOWNS + VAL_TOWNS:
        for scn in SCENARIOS:
            bases, left_dir, pano_dir = list_frames(town, scn)
            index[(town, scn)] = {"bases": bases, "left_dir": left_dir, "pano_dir": pano_dir}
            if not bases:
                print(f"[WARN] No frames: {town}/{scn}")

    train_samples, val_samples = [], []

    # train
    for scn, total in TRAIN_SCN_TARGET.items():
        quota = split_quota(total, TRAIN_TOWNS)
        for town in TRAIN_TOWNS:
            pool = index[(town, scn)]["bases"]
            need = quota[town]
            if len(pool) < need:
                raise RuntimeError(f"Not enough frames in {town}/{scn}: need {need}, have {len(pool)}")
            pick = random.sample(pool, need)
            train_samples += [(town, scn, b) for b in pick]

    # val
    for scn in SCENARIOS:
        quota = split_quota(VAL_PER_SCENARIO, VAL_TOWNS)
        for town in VAL_TOWNS:
            pool = index[(town, scn)]["bases"]
            need = quota[town]
            if len(pool) < need:
                raise RuntimeError(f"Not enough frames in {town}/{scn} (val): need {need}, have {len(pool)}")
            pick = random.sample(pool, need)
            val_samples += [(town, scn, b) for b in pick]

    print("\n=== Sampling summary ===")
    print("Train totals per scenario:", dict(TRAIN_SCN_TARGET))
    print("Val per scenario:", VAL_PER_SCENARIO)
    print("Sampled Train:", len(train_samples), "  Val:", len(val_samples))
    return index, train_samples, val_samples

def ensure_dirs():
    for p in [
        os.path.join(OUT_ROOT, "leftImg8bit", "train"),
        os.path.join(OUT_ROOT, "leftImg8bit", "val"),
        os.path.join(OUT_ROOT, "panoptic_color", "train"),
        os.path.join(OUT_ROOT, "panoptic_color", "val"),
        os.path.join(OUT_ROOT, "panoptic_json"),
    ]:
        os.makedirs(p, exist_ok=True)

def copy_leftimg_and_build_json(split_name, samples, index):
    """
    leftImg8bit/{train|val} 에 평평한 파일명으로 복사하고,
    panoptic_color/{train|val}에 color png 생성,
    panoptic_json/panoptic_{split}_fixed_paths.v4.json 생성.
    """
    out_img_dir = os.path.join(OUT_ROOT, "leftImg8bit", split_name)
    out_pan_dir = os.path.join(OUT_ROOT, "panoptic_color", split_name)
    out_json    = os.path.join(OUT_ROOT, "panoptic_json", f"panoptic_{split_name}_fixed_paths.v4.json")

    images, annotations = [], []
    copied, skipped = 0, 0

    for (town, scn, base) in tqdm(samples, desc=f"[{split_name}] copy & convert"):
        left_dir = index[(town, scn)]["left_dir"]
        pano_dir = index[(town, scn)]["pano_dir"]

        src_img = os.path.join(left_dir, f"{base}_leftImg8bit.png")
        src_p16 = os.path.join(pano_dir, f"{base}_panopticId.png")
        if not (os.path.exists(src_img) and os.path.exists(src_p16)):
            skipped += 1
            continue

        # 평평한 파일명
        flat_base = f"{town}_{scn}_{base}"
        dst_img_name = f"{flat_base}_leftImg8bit.png"
        dst_img = os.path.join(out_img_dir, dst_img_name)

        # leftImg8bit 복사 (존재하면 스킵)
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        copied += 1

        # panoptic_color 생성
        pan16 = cv2.imread(src_p16, cv2.IMREAD_UNCHANGED)
        if pan16 is None:
            skipped += 1
            continue
        H, W = pan16.shape[:2]
        trainId = (pan16 // 1000).astype(np.uint16)
        void_mask = (trainId == 255)
        if np.any(void_mask):
            pan16 = pan16.copy()
            pan16[void_mask] = 0
            trainId[void_mask] = 0

        seg_ids = np.unique(pan16); seg_ids = seg_ids[seg_ids > 0]
        color = np.zeros((H, W, 3), np.uint8)
        for sid in seg_ids:
            color[pan16 == sid] = id_to_rgb(int(sid))
        color_name = f"{flat_base}.png"
        color_path = os.path.join(out_pan_dir, color_name)
        cv2.imwrite(color_path, color)

        seg_infos = []
        for sid in seg_ids:
            m = (pan16 == sid)
            ar = int(m.sum())
            if ar == 0: 
                continue
            tid = int(sid // 1000)
            seg_infos.append({
                "id": int(sid),
                "category_id": tid,
                "isthing": 1 if tid in THING_TRAINIDS else 0,
                "area": ar,
                "bbox": bbox_from_mask(m),
            })

        # Detectron2 기대 포맷:
        # - images[*].file_name: "train/..." or "val/..." (IMAGE_ROOT 하위 상대경로)
        # - annotations[*].file_name: panoptic_color/{split}/... (panoptic_root 하위 파일명)
        img_id = flat_base  # 문자열 id 허용
        images.append({
            "id": img_id,
            "file_name": f"{split_name}/{dst_img_name}",
            "height": H, "width": W,
        })
        annotations.append({
            "image_id": img_id,
            "file_name": color_name,
            "segments_info": seg_infos
        })

    with open(out_json, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES}, f)

    print(f"✔ {split_name}: copied={copied}, skipped={skipped}")
    print(f"   JSON: {out_json}")
    print(f"   leftImg8bit dir: {out_img_dir}")
    print(f"   panoptic_color dir: {out_pan_dir}")

def main():
    ensure_dirs()
    index, train_samples, val_samples = sample_frames()
    copy_leftimg_and_build_json("train", train_samples, index)
    copy_leftimg_and_build_json("val",   val_samples,   index)
    print("DONE.")

if __name__ == "__main__":
    main()
