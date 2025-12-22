# new_mapping.py
# ---------------------------------------------
# CARLA labelIds -> Cityscapes-11(0~10) 정규화
# - GT만 있어도 동작
# - PR(예측)도 함께 처리 가능: city11 / city19 / carla 중 선택
# - GT 해상도에 PR을 NEAREST로 맞춰 저장
# - 디렉토리 트리(시나리오 구조) 보존
# ---------------------------------------------

import os, glob, argparse
from typing import Dict, Tuple, List
import numpy as np
import cv2
from PIL import Image

IGNORE_ID = 255
NCLS = 11

# ---- Classes (참고용) ----
CLASSES11 = [
    "road","sidewalk","building","wall","fence",
    "pole","traffic light","traffic sign","vegetation","terrain","sky"
]

# ---- Mappings ----
# CARLA raw IDs -> City11
CARLA2CITY11: Dict[int, int] = {
     7: 0,  # road
     8: 1,  # sidewalk
    10: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 5,  # pole
    18: 6,  # traffic light
    19: 7,  # traffic sign
    20: 8,  # vegetation
    21: 9,  # terrain
    22: 10, # sky
}

# Cityscapes "trainIds" 19(0..18) -> City11 (stuff-only 0..10 유지)
# 0..10 그대로 두고, 11..18(thing)은 ignore
CITY19_TO_11_KEEP: Dict[int, int] = {i: i for i in range(11)}

# ---------- IO ----------
def load_id_png(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expect (H,W) ID image, got {arr.shape} for {path}")
    return arr.astype(np.uint16)

def save_id_png(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint16)).save(path)

# ---------- mapping ----------
def carla_to_city11(src: np.ndarray, ignore: int = IGNORE_ID) -> np.ndarray:
    out = np.full_like(src, ignore, dtype=np.uint16)
    for s, d in CARLA2CITY11.items():
        out[src == s] = d
    return out

def city19_to_city11(src: np.ndarray, ignore: int = IGNORE_ID) -> np.ndarray:
    out = np.full_like(src, ignore, dtype=np.uint16)
    for s, d in CITY19_TO_11_KEEP.items():
        out[src == s] = d
    return out

def sanitize_city11(arr: np.ndarray, ignore: int = IGNORE_ID, name: str = "arr") -> np.ndarray:
    a = arr.copy()
    bad = (a != ignore) & ((a < 0) | (a >= NCLS))
    if np.any(bad):
        vals = np.unique(a[bad])
        print(f"[WARN] {name} has out-of-range labels:", vals[:20], "…")
        a[bad] = ignore
    return a

# ---------- resize ----------
def resize_label_nearest(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    H, W = arr.shape[:2]
    if (W, H) == (width, height):
        return arr
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST).astype(arr.dtype)

# ---------- helpers ----------
def rel_to(path: str, root: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root))

def index_by_core(files: List[str], kind: str) -> dict:
    """
    core name: frame_XXXXXX
    - GT: *_gtFine_labelIds.png -> core = frame_XXXXXX
    - PR: *_pred_*.png         -> core = frame_XXXXXX
    """
    idx = {}
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if kind == "gt" and "_gtFine_" in stem:
            core = stem.split("_gtFine_")[0]
        elif kind == "pr" and "_pred" in stem:
            core = stem.split("_pred")[0]
        else:
            # fallback
            parts = stem.split("_")
            core = "_".join(parts[:2]) if len(parts) >= 2 else stem
        idx.setdefault(core, []).append(p)
    return idx

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("CARLA labelIds -> Cityscapes-11 mapper (GT-only or GT+PR)")
    ap.add_argument("--gt_root", required=True, help="GT 라벨 상위 폴더 (재귀)")
    ap.add_argument("--out_root", required=True, help="출력 상위 폴더 (구조 보존)")
    ap.add_argument("--gt_pattern", default="*_gtFine_labelIds.png", help="GT 파일 패턴")

    ap.add_argument("--pr_root", default=None, help="(선택) 예측 ID 상위 폴더")
    # 기본값을 실제 파일명에 맞게!
    ap.add_argument("--pr_pattern", default="*_pred_mask_city11.png",
                    help="PR 파일 패턴 (예: *_pred_mask_city11.png / *_pred_mask_19.png / carla 라벨이면 *_pred_mask.png)")
    ap.add_argument("--pr_kind", default="city11", choices=["city11","city19","carla"],
                    help="PR 라벨의 원본 종류")

    ap.add_argument("--no_stats", action="store_true")
    args = ap.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_root, "**", args.gt_pattern), recursive=True))
    if not gt_files:
        print("[WARN] GT 파일을 못 찾았습니다."); return
    print(f"[INFO] GT 파일 {len(gt_files)}개 발견")

    # PR 인덱스
    pr_idx = {}
    if args.pr_root:
        pr_files = sorted(glob.glob(os.path.join(args.pr_root, "**", args.pr_pattern), recursive=True))
        if pr_files:
            pr_idx = index_by_core(pr_files, "pr")
            print(f"[INFO] PR 파일 {len(pr_files)}개 인덱싱 (코어 매칭)")
        else:
            print("[WARN] pr_root에서 PR 파일을 못 찾았습니다. GT만 처리합니다.")
            args.pr_root = None

    # 처리
    for gt_path in gt_files:
        gt_raw = load_id_png(gt_path)
        gt11 = carla_to_city11(gt_raw)
        gt11 = sanitize_city11(gt11, name="gt11")

        # GT 출력 경로
        rel_gt = rel_to(gt_path, args.gt_root)
        gt_out = os.path.join(args.out_root, rel_gt).replace("_gtFine_labelIds", "_city11_labelIds")
        save_id_png(gt_out, gt11)

        if args.pr_root:
            core = os.path.splitext(os.path.basename(gt_path))[0].split("_gtFine_")[0]
            pr_path = pr_idx.get(core, [None])[0]
            if pr_path is not None:
                pr_raw = load_id_png(pr_path)

                # PR 변환 종류에 따라 처리
                if args.pr_kind == "city11":
                    pr11 = sanitize_city11(pr_raw, name="pred(city11)")
                elif args.pr_kind == "city19":
                    pr11 = city19_to_city11(pr_raw)
                    pr11 = sanitize_city11(pr11, name="pred(19->11)")
                elif args.pr_kind == "carla":
                    pr11 = carla_to_city11(pr_raw)
                    pr11 = sanitize_city11(pr11, name="pred(carla->11)")
                else:
                    raise ValueError(f"Unknown pr_kind: {args.pr_kind}")

                H, W = gt11.shape
                pr11 = resize_label_nearest(pr11, width=W, height=H)

                rel_pr = rel_to(pr_path, args.pr_root)
                # 파일명 유지 + 접미사 추가
                if rel_pr.endswith(".png"):
                    pr_out = os.path.join(args.out_root, rel_pr[:-4] + "_city11.png")
                else:
                    pr_out = os.path.join(args.out_root, rel_pr + "_city11.png")
                save_id_png(pr_out, pr11)

                if not args.no_stats:
                    print(f"[OK] {rel_gt} + {rel_pr} -> saved (GT/PR city11)")
            else:
                if not args.no_stats:
                    print(f"[OK] {rel_gt} -> saved (GT city11) | PR 없음")
        elif not args.no_stats:
            print(f"[OK] {rel_gt} -> saved (GT city11)")

    print(f"\n[DONE] 출력 위치: {args.out_root}")

if __name__ == "__main__":
    main()
# new_mapping.py
# ---------------------------------------------
# CARLA labelIds -> Cityscapes-11(0~10) 정규화
# - GT만 있어도 동작
# - PR(예측)도 함께 처리 가능: city11 / city19 / carla 중 선택
# - GT 해상도에 PR을 NEAREST로 맞춰 저장
# - 디렉토리 트리(시나리오 구조) 보존
# ---------------------------------------------

import os, glob, argparse
from typing import Dict, Tuple, List
import numpy as np
import cv2
from PIL import Image

IGNORE_ID = 255
NCLS = 11

# ---- Classes (참고용) ----
CLASSES11 = [
    "road","sidewalk","building","wall","fence",
    "pole","traffic light","traffic sign","vegetation","terrain","sky"
]

# ---- Mappings ----
# CARLA raw IDs -> City11
CARLA2CITY11: Dict[int, int] = {
     7: 0,  # road
     8: 1,  # sidewalk
    10: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 5,  # pole
    18: 6,  # traffic light
    19: 7,  # traffic sign
    20: 8,  # vegetation
    21: 9,  # terrain
    22: 10, # sky
}

# Cityscapes "trainIds" 19(0..18) -> City11 (stuff-only 0..10 유지)
# 0..10 그대로 두고, 11..18(thing)은 ignore
CITY19_TO_11_KEEP: Dict[int, int] = {i: i for i in range(11)}

# ---------- IO ----------
def load_id_png(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expect (H,W) ID image, got {arr.shape} for {path}")
    return arr.astype(np.uint16)

def save_id_png(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint16)).save(path)

# ---------- mapping ----------
def carla_to_city11(src: np.ndarray, ignore: int = IGNORE_ID) -> np.ndarray:
    out = np.full_like(src, ignore, dtype=np.uint16)
    for s, d in CARLA2CITY11.items():
        out[src == s] = d
    return out

def city19_to_city11(src: np.ndarray, ignore: int = IGNORE_ID) -> np.ndarray:
    out = np.full_like(src, ignore, dtype=np.uint16)
    for s, d in CITY19_TO_11_KEEP.items():
        out[src == s] = d
    return out

def sanitize_city11(arr: np.ndarray, ignore: int = IGNORE_ID, name: str = "arr") -> np.ndarray:
    a = arr.copy()
    bad = (a != ignore) & ((a < 0) | (a >= NCLS))
    if np.any(bad):
        vals = np.unique(a[bad])
        print(f"[WARN] {name} has out-of-range labels:", vals[:20], "…")
        a[bad] = ignore
    return a

# ---------- resize ----------
def resize_label_nearest(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    H, W = arr.shape[:2]
    if (W, H) == (width, height):
        return arr
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST).astype(arr.dtype)

# ---------- helpers ----------
def rel_to(path: str, root: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root))

def index_by_core(files: List[str], kind: str) -> dict:
    """
    core name: frame_XXXXXX
    - GT: *_gtFine_labelIds.png -> core = frame_XXXXXX
    - PR: *_pred_*.png         -> core = frame_XXXXXX
    """
    idx = {}
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if kind == "gt" and "_gtFine_" in stem:
            core = stem.split("_gtFine_")[0]
        elif kind == "pr" and "_pred" in stem:
            core = stem.split("_pred")[0]
        else:
            # fallback
            parts = stem.split("_")
            core = "_".join(parts[:2]) if len(parts) >= 2 else stem
        idx.setdefault(core, []).append(p)
    return idx

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("CARLA labelIds -> Cityscapes-11 mapper (GT-only or GT+PR)")
    ap.add_argument("--gt_root", required=True, help="GT 라벨 상위 폴더 (재귀)")
    ap.add_argument("--out_root", required=True, help="출력 상위 폴더 (구조 보존)")
    ap.add_argument("--gt_pattern", default="*_gtFine_labelIds.png", help="GT 파일 패턴")

    ap.add_argument("--pr_root", default=None, help="(선택) 예측 ID 상위 폴더")
    # 기본값을 실제 파일명에 맞게!
    ap.add_argument("--pr_pattern", default="*_pred_mask_city11.png",
                    help="PR 파일 패턴 (예: *_pred_mask_city11.png / *_pred_mask_19.png / carla 라벨이면 *_pred_mask.png)")
    ap.add_argument("--pr_kind", default="city11", choices=["city11","city19","carla"],
                    help="PR 라벨의 원본 종류")

    ap.add_argument("--no_stats", action="store_true")
    args = ap.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_root, "**", args.gt_pattern), recursive=True))
    if not gt_files:
        print("[WARN] GT 파일을 못 찾았습니다."); return
    print(f"[INFO] GT 파일 {len(gt_files)}개 발견")

    # PR 인덱스
    pr_idx = {}
    if args.pr_root:
        pr_files = sorted(glob.glob(os.path.join(args.pr_root, "**", args.pr_pattern), recursive=True))
        if pr_files:
            pr_idx = index_by_core(pr_files, "pr")
            print(f"[INFO] PR 파일 {len(pr_files)}개 인덱싱 (코어 매칭)")
        else:
            print("[WARN] pr_root에서 PR 파일을 못 찾았습니다. GT만 처리합니다.")
            args.pr_root = None

    # 처리
    for gt_path in gt_files:
        gt_raw = load_id_png(gt_path)
        gt11 = carla_to_city11(gt_raw)
        gt11 = sanitize_city11(gt11, name="gt11")

        # GT 출력 경로
        rel_gt = rel_to(gt_path, args.gt_root)
        gt_out = os.path.join(args.out_root, rel_gt).replace("_gtFine_labelIds", "_city11_labelIds")
        save_id_png(gt_out, gt11)

        if args.pr_root:
            core = os.path.splitext(os.path.basename(gt_path))[0].split("_gtFine_")[0]
            pr_path = pr_idx.get(core, [None])[0]
            if pr_path is not None:
                pr_raw = load_id_png(pr_path)

                # PR 변환 종류에 따라 처리
                if args.pr_kind == "city11":
                    pr11 = sanitize_city11(pr_raw, name="pred(city11)")
                elif args.pr_kind == "city19":
                    pr11 = city19_to_city11(pr_raw)
                    pr11 = sanitize_city11(pr11, name="pred(19->11)")
                elif args.pr_kind == "carla":
                    pr11 = carla_to_city11(pr_raw)
                    pr11 = sanitize_city11(pr11, name="pred(carla->11)")
                else:
                    raise ValueError(f"Unknown pr_kind: {args.pr_kind}")

                H, W = gt11.shape
                pr11 = resize_label_nearest(pr11, width=W, height=H)

                rel_pr = rel_to(pr_path, args.pr_root)
                # 파일명 유지 + 접미사 추가
                if rel_pr.endswith(".png"):
                    pr_out = os.path.join(args.out_root, rel_pr[:-4] + "_city11.png")
                else:
                    pr_out = os.path.join(args.out_root, rel_pr + "_city11.png")
                save_id_png(pr_out, pr11)

                if not args.no_stats:
                    print(f"[OK] {rel_gt} + {rel_pr} -> saved (GT/PR city11)")
            else:
                if not args.no_stats:
                    print(f"[OK] {rel_gt} -> saved (GT city11) | PR 없음")
        elif not args.no_stats:
            print(f"[OK] {rel_gt} -> saved (GT city11)")

    print(f"\n[DONE] 출력 위치: {args.out_root}")

if __name__ == "__main__":
    main()
