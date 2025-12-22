# visualize_panoptic_gt.py
import os, json, argparse, random
import numpy as np
import cv2

def segid_from_rgb(rgb):
    return (rgb[:,:,0].astype(np.int64)
            + 256*rgb[:,:,1].astype(np.int64)
            + 256*256*rgb[:,:,2].astype(np.int64))

def rand_color(seed):
    rng = np.random.RandomState(seed)
    return [int(x) for x in rng.randint(0,255,3)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True)   # .../leftImg8bit
    ap.add_argument("--pan_png_root", required=True) # .../panoptic_color
    ap.add_argument("--pan_json", required=True)     # .../panoptic_json/panoptic_{split}.json
    ap.add_argument("--split", default="val", choices=["train","val"])
    ap.add_argument("--out", default="./_viz_gt")
    ap.add_argument("--num", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.pan_json, "r") as f:
        pan = json.load(f)

    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in pan["categories"]}
    ann_by_file = {a["file_name"]: a for a in pan["annotations"]}

    # 이미지-파놉틱 매칭 후보 수집
    candidates = []
    img_split_root = os.path.join(args.image_root, args.split)
    pan_split_root = os.path.join(args.pan_png_root, args.split)
    for root, _, files in os.walk(img_split_root):
        for fn in files:
            if not fn.endswith("_leftImg8bit.png"):
                continue
            stem = fn[:-len("_leftImg8bit.png")]
            pan_fn = stem + "_panoptic.png"
            pan_path = os.path.join(pan_split_root, pan_fn)
            if os.path.exists(pan_path):
                candidates.append((os.path.join(root, fn), pan_path))
    random.shuffle(candidates)
    candidates = candidates[:args.num]

    for img_path, pan_path in candidates:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        pan_rgb = cv2.imread(pan_path, cv2.IMREAD_COLOR)
        if img is None or pan_rgb is None:
            print("skip read fail:", img_path, pan_path); continue

        seg = segid_from_rgb(pan_rgb)
        pan_file = os.path.basename(pan_path)
        ann = ann_by_file.get(pan_file)
        if ann is None:
            print("no annotation for:", pan_file); continue

        # segment_id -> category_id
        seginfo = {s["id"]: s["category_id"] for s in ann["segments_info"]}
        used_colors = {}
        overlay = img.copy()

        for sid, cid in seginfo.items():
            if cid not in used_colors:
                used_colors[cid] = rand_color(cid+31)
            mask = (seg == sid)
            if mask.any():
                overlay[mask] = (0.6*overlay[mask] + 0.4*np.array(used_colors[cid])).astype(np.uint8)

        vis = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)

        # 간단 범례
        legend = np.ones((220, 420, 3), dtype=np.uint8)*255
        y = 22
        for i, (cid, col) in enumerate(list(used_colors.items())[:12]):
            name = cat_id_to_name.get(cid, f"id{cid}")
            cv2.rectangle(legend, (10, y-12), (30, y+8), col, -1)
            cv2.putText(legend, f"{cid}: {name}", (40, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
            y += 18

        H = max(vis.shape[0], legend.shape[0])
        canvas = np.ones((H, vis.shape[1]+legend.shape[1]+10, 3), dtype=np.uint8)*255
        canvas[:vis.shape[0], :vis.shape[1]] = vis
        canvas[:legend.shape[0], vis.shape[1]+10:] = legend

        out_name = os.path.join(args.out, os.path.basename(pan_path).replace("_panoptic.png", "_gtviz.jpg"))
        cv2.imwrite(out_name, canvas)
        print("saved:", out_name)

if __name__ == "__main__":
    main()
