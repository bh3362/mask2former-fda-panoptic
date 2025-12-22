# fix_panoptic_file_names.py
import json
from pathlib import Path

ROOT = Path("/media/vip-dell/HC/train_data_3k")
IMG_ROOT = ROOT / "leftImg8bit"     # <-- file_name은 이 루트 기준의 상대경로여야 함 (예: "train/xxx.png")
PAN_DIR  = ROOT / "panoptic_json"

IN_JSONS  = ["panoptic_train_fixed.json", "panoptic_val_fixed.json"]
OUT_JSONS = ["panoptic_train_fixed_paths.json", "panoptic_val_fixed_paths.json"]

def index_images(img_root: Path):
    idx = {}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            stem = p.stem                            # 확장자 뺀 이름
            rel  = str(p.relative_to(img_root))      # leftImg8bit 기준 상대경로 (예: "train/Town01_...png")
            idx.setdefault(stem, []).append(rel)
    return idx

def fix_one(in_path: Path, out_path: Path, img_index: dict):
    data = json.loads(in_path.read_text())
    fixed, missing = 0, 0

    for im in data.get("images", []):
        orig = im.get("file_name", "")
        stem = Path(orig).stem

        # 1) 정확히 같은 stem
        cand = img_index.get(stem)

        # 2) 흔한 케이스: 원래 JSON이 "_leftImg8bit" 붙이기 전 이름을 가짐
        if not cand:
            cand = img_index.get(f"{stem}_leftImg8bit")

        # 3) 마지막 시도: .jpg → .png 치환 후 직접 존재 확인
        if not cand and orig.lower().endswith(".jpg"):
            maybe = orig[:-4] + ".png"
            p = (IMG_ROOT / maybe)
            if p.is_file():
                cand = [str(p.relative_to(IMG_ROOT))]

        if cand:
            # 여러 후보면 .png 우선
            cand.sort(key=lambda x: (not x.lower().endswith(".png"), x))
            im["file_name"] = cand[0]
            fixed += 1
        else:
            missing += 1

    out_path.write_text(json.dumps(data))
    print(f"[{in_path.name}] fixed={fixed}, missing={missing}")

def main():
    img_index = index_images(IMG_ROOT)
    print(f"Indexed images: {sum(len(v) for v in img_index.values())}")

    for in_name, out_name in zip(IN_JSONS, OUT_JSONS):
        fix_one(PAN_DIR / in_name, PAN_DIR / out_name, img_index)

if __name__ == "__main__":
    main()
