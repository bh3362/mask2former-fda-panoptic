# viz_label_overlay.py
import os, glob, argparse
import numpy as np
import cv2
from PIL import Image

# City11 팔레트 & 이름
NAMES = ["road","sidewalk","building","wall","fence",
         "pole","traffic light","traffic sign","vegetation","terrain","sky"]
PALETTE = np.array([
    [128, 64,128],[244, 35,232],[ 70, 70, 70],[102,102,156],[190,153,153],
    [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180]
], dtype=np.uint8)
IGNORE = 255

def load_id(p):
    a = np.array(Image.open(p))
    if a.ndim==3: a=a[...,0]
    return a.astype(np.uint16)

def colorize(mask):
    h,w = mask.shape
    out = np.zeros((h,w,3), np.uint8)
    for cid in range(len(NAMES)):
        out[mask==cid] = PALETTE[cid]
    out[mask==IGNORE] = (0,0,0)
    return out

def draw_label_box(img, text, org, font_scale=0.6, txt_color=(0,0,0), bg_color=(0,255,255)):
    """OpenCV로 가독성 좋은 텍스트 박스 그리기"""
    x,y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 2)
    pad = 4
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+baseline+pad), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, txt_color, 2, cv2.LINE_AA)

def annotate_by_components(mask, canvas_bgr, min_area=800):
    """클래스별 연결요소 중심에 라벨 텍스트 표시"""
    out = canvas_bgr.copy()
    for cid, name in enumerate(NAMES):
        m = (mask==cid).astype(np.uint8)
        if m.sum()==0: continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: 
                continue
            # 중심점 계산
            M = cv2.moments(cnt)
            if M["m00"] == 0: 
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            # 경계선도 살짝
            cv2.drawContours(out, [cnt], -1, (0,0,0), 2)
            cv2.drawContours(out, [cnt], -1, (int(PALETTE[cid][2]), int(PALETTE[cid][1]), int(PALETTE[cid][0])), 1)
            # 라벨 박스
            draw_label_box(out, name, (max(0,cx-30), max(20,cy)), bg_color=(0,255,255))
    return out

def overlay(rgb, color_mask_bgr, alpha=0.5):
    return cv2.addWeighted(rgb, 1-alpha, color_mask_bgr, alpha, 0)

def main():
    ap = argparse.ArgumentParser("RGB에 세그 오버레이 + 클래스 라벨 저장")
    ap.add_argument("--base", required=True, help="시나리오 루트 (예: .../_output2)")
    ap.add_argument("--pred_root", required=True, help="예측(city11) 루트 (예: .../_pred_masks_city11)")
    ap.add_argument("--out_dir", default="viz_overlay", help="결과 저장 폴더")
    ap.add_argument("--scenarios", default="", help="쉼표구분. 비우면 전체")
    ap.add_argument("--limit", type=int, default=0, help="저장 개수 제한(0=제한없음)")
    ap.add_argument("--stride", type=int, default=1, help="프레임 샘플 간격")
    ap.add_argument("--alpha", type=float, default=0.55, help="오버레이 투명도")
    ap.add_argument("--min_area", type=int, default=800, help="라벨 표시 최소 영역 픽셀 수")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rgb_list = sorted(glob.glob(os.path.join(args.base, "**", "leftImg8bit", "frame_*_leftImg8bit.png"), recursive=True))
    if args.scenarios.strip():
        allow = set([s.strip() for s in args.scenarios.split(",") if s.strip()])
        rgb_list = [p for p in rgb_list if os.path.relpath(p, args.base).split(os.sep)[0] in allow]
    if args.stride > 1:
        rgb_list = rgb_list[::args.stride]
    if args.limit and args.limit > 0:
        rgb_list = rgb_list[:args.limit]

    used = 0
    for rp in rgb_list:
        rel = os.path.relpath(rp, args.base)
        scen = rel.split(os.sep)[0]
        core = os.path.basename(rp).replace("_leftImg8bit.png","")

        # 경로들
        rgb_bgr = cv2.imread(rp)
        if rgb_bgr is None: 
            continue
        gt_path = os.path.join(args.base, scen, "gtFine", f"{core}_gtFine_labelIds.png")
        pr_path = os.path.join(args.pred_root, scen, f"{core}_pred_mask_city11.png")
        if not os.path.exists(pr_path):
            continue

        pred = load_id(pr_path)
        # 컬러 마스크 & 오버레이
        pred_col = colorize(pred)                            # RGB
        pred_col_bgr = cv2.cvtColor(pred_col, cv2.COLOR_RGB2BGR)
        over = overlay(rgb_bgr, pred_col_bgr, alpha=args.alpha)
        # 클래스 라벨 주석
        labeled = annotate_by_components(pred, over, min_area=args.min_area)

        out_dir_s = os.path.join(args.out_dir, scen)
        os.makedirs(out_dir_s, exist_ok=True)
        outp = os.path.join(out_dir_s, f"{core}_overlay_labeled.png")
        cv2.imwrite(outp, labeled)
        used += 1

    print(f"[DONE] 저장 {used}장  ->  {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
