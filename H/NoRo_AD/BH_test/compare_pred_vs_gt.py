# compare_pred_vs_gt.py
import os, json, argparse, cv2, numpy as np

def segid_from_rgb(rgb):
    return rgb[:,:,0].astype(np.int64) + 256*rgb[:,:,1].astype(np.int64) + 256*256*rgb[:,:,2].astype(np.int64)

def make_colormap(ids):
    cmap = {}
    for i, cid in enumerate(sorted(ids)):
        rng = np.random.RandomState(cid+73)
        cmap[cid] = [int(x) for x in rng.randint(0,255,3)]
    return cmap

def colorize(seg, seginfo):
    # seginfo: list of dicts with {id, category_id}
    sid2cid = {s["id"]: s["category_id"] for s in seginfo}
    cids = set(sid2cid.values())
    cmap = make_colormap(cids) if cids else {}
    rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for sid, cid in sid2cid.items():
        rgb[seg==sid] = cmap.get(cid, (255,0,255))
    return rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt_pan_png", required=True)
    ap.add_argument("--gt_pan_json", required=True)
    ap.add_argument("--pred_pan_png", required=True)
    ap.add_argument("--pred_json", required=True, help="detectron2 panoptic pred json")
    ap.add_argument("--out", default="./_viz_cmp.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    gt_png = cv2.imread(args.gt_pan_png, cv2.IMREAD_COLOR)
    pred_png = cv2.imread(args.pred_pan_png, cv2.IMREAD_COLOR)

    with open(args.gt_pan_json,"r") as f: gtj = json.load(f)
    with open(args.pred_json,"r") as f: predj = json.load(f)

    # 파일명 키 매칭 (간단화를 위해 png 파일명 키 일치 가정)
    gt_ann = None
    for a in gtj["annotations"]:
        if a["file_name"] == os.path.basename(args.gt_pan_png):
            gt_ann = a; break

    pred_ann = None
    for a in predj:
        # detectron2 panoptic evaluator는 {image_id,file_name,segments_info} 구조
        if os.path.basename(a["file_name"]) == os.path.basename(args.pred_pan_png):
            pred_ann = a; break

    assert gt_ann is not None and pred_ann is not None, "ann not found"

    gt_seg = segid_from_rgb(gt_png)
    pr_seg = segid_from_rgb(pred_png)

    gt_col = colorize(gt_seg, gt_ann["segments_info"])
    pr_col = colorize(pr_seg, pred_ann["segments_info"])

    # overlay
    gt_overlay = cv2.addWeighted(img, 0.4, gt_col, 0.6, 0)
    pr_overlay = cv2.addWeighted(img, 0.4, pr_col, 0.6, 0)

    top = np.hstack([img, gt_overlay, pr_overlay])
    cv2.imwrite(args.out, top)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
