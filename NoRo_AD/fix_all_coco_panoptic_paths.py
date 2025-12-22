# fix_all_coco_panoptic_paths.py
# COCO panoptic JSON의 images[*].file_name을 일괄 교정:
#   - 확장자: *_leftImg8bit.png 로 강제
#   - 프리픽스: train/ 또는 val/ 로 강제
# 또한 images[*].id / annotations[*].image_id 가 문자열이면 정수로 일관화(선택).
import os, json, sys

def fix_one(json_path: str, split_hint: str, make_numeric_ids: bool = True):
    assert split_hint in ("train", "val"), "split_hint must be 'train' or 'val'"
    if not os.path.exists(json_path):
        print(f"[ERR] not found: {json_path}")
        return

    with open(json_path, "r") as f:
        d = json.load(f)

    images = d.get("images", [])
    anns   = d.get("annotations", [])
    cats   = d.get("categories", [])

    changed_path = 0
    for im in images:
        fn = im.get("file_name", "")
        base = os.path.splitext(os.path.basename(fn))[0]  # 확장자 제거한 파일명만
        # *_leftImg8bit.png 보장
        if base.lower().endswith("_leftimg8bit"):
            fixed_rel = f"{split_hint}/{base}.png"
        else:
            fixed_rel = f"{split_hint}/{base}_leftImg8bit.png"
        if fn != fixed_rel:
            im["file_name"] = fixed_rel
            changed_path += 1

    # (선택) ID를 정수로 통일
    changed_id = 0
    if make_numeric_ids:
        id_map = {}
        new_images = []
        for new_id, im in enumerate(images, start=1):
            old_id = im["id"]
            if old_id != new_id:
                changed_id += 1
            id_map[old_id] = new_id
            im = dict(im)
            im["id"] = new_id
            new_images.append(im)
        new_anns = []
        for an in anns:
            an = dict(an)
            old_img_id = an["image_id"]
            if old_img_id not in id_map:
                # 혹시 키 타입 불일치 방지(문자/정수 섞임)
                if str(old_img_id) in id_map:
                    an["image_id"] = id_map[str(old_img_id)]
                else:
                    raise KeyError(f"image_id {old_img_id} missing in images")
            else:
                an["image_id"] = id_map[old_img_id]
            new_anns.append(an)
        d["images"] = new_images
        d["annotations"] = new_anns

    out_path = os.path.splitext(json_path)[0] + ".fixed_all.json"
    with open(out_path, "w") as f:
        json.dump(d, f)
    print(f"[OK] wrote: {out_path}")
    print(f"     images: {len(images)}  anns: {len(anns)}  cats: {len(cats)}")
    print(f"     changed file_name: {changed_path}, changed IDs: {changed_id}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_all_coco_panoptic_paths.py <json_path> <train|val> [--keep-string-id]")
        sys.exit(1)
    jp = sys.argv[1]
    split = sys.argv[2]
    keep_string = ("--keep-string-id" in sys.argv)
    fix_one(jp, split, make_numeric_ids=(not keep_string))
