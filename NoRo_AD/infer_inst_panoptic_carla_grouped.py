# infer_inst_panoptic_carla_grouped.py
# ---------------------------------------------------------------
# CARLA _output3 구조에서 "인스턴스 + 파놉틱"만 인퍼런스하여 저장
# - 입력: <base>/TownXX/SCENARIO/leftImg8bit/frame_***_leftImg8bit.png
# - 파놉틱 출력:
#     <out_pan>/<Town>/<Scenario>/<stem>_panoptic_pred.png     (RGB 인코딩 panoptic PNG)
#     <out_pan>/<Town>/<Scenario>/<stem>_segments.json         (segments_info 리스트; 원본 ID 기준)
#     <out_pan>/<Town>/<Scenario>/<stem>_segments_contig.json  (시각화용 연속 ID 사본)
#     [옵션] <out_pan>/_vis/...                                (원본+예측 시각화)
# - 인스턴스 출력:
#     <out_inst>/<Town>/<Scenario>/<stem>_instances_pred.json  (COCO-like per-image)
#     [옵션] <out_inst>/_vis/...                               (시각화)
# - 그룹 선택: 기본 타운×시나리오당 앞 20장만
# ---------------------------------------------------------------
import os, glob, argparse, json, re
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import torch

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from mask2former import add_maskformer2_config

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None  # 없으면 마스크 RLE는 생략

# ---------- helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def frame_index_from_path(p: str):
    m = re.search(r'frame_(\d+)_leftImg8bit\.png$', p)
    return int(m.group(1)) if m else None

def id_to_rgb(id_: int) -> Tuple[int, int, int]:
    # COCO panoptic RGB encoding (24-bit)
    r = id_ % 256
    g = (id_ // 256) % 256
    b = (id_ // 65536) % 256
    return (r, g, b)

def panoptic_idmap_to_rgb(seg: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    out = np.zeros((h, w, 3), np.uint8)
    for sid in np.unique(seg).tolist():
        out[seg == sid] = id_to_rgb(int(sid))
    return out

def add_fan_config(cfg: CN):
    # FAN 노드(키) 선등록: YAML 병합 시 Unknown key 방지
    cfg.MODEL.FAN = CN()
    cfg.MODEL.FAN.PATCH_SIZE = 4
    cfg.MODEL.FAN.IN_CHANS = 3
    cfg.MODEL.FAN.NUM_CLASSES = 19
    cfg.MODEL.FAN.EMBED_DIM = 480
    cfg.MODEL.FAN.DEPTH = 22
    cfg.MODEL.FAN.OUT_IDX = 18
    cfg.MODEL.FAN.NUM_HEADS = 10
    cfg.MODEL.FAN.MLP_RATIO = 4.0
    cfg.MODEL.FAN.DROP_RATE = 0.0
    cfg.MODEL.FAN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.FAN.DROP_PATH_RATE = 0.1
    cfg.MODEL.FAN.CLS_ATTN_LAYERS = 2
    cfg.MODEL.FAN.ETA = 1.0
    cfg.MODEL.FAN.OUT_FEATURES = ["res2","res3","res4","res5"]

def add_resnets_compat(cfg: CN):
    # RESNETS 더미 노드: STEM_TYPE 포함해 선등록
    cfg.MODEL.RESNETS = CN()
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STEM_TYPE = ""             # <- 이 키 때문에 보통 에러 발생
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.NUM_GROUPS = 1
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    cfg.MODEL.RESNETS.BOTTLENECK_CHANNELS = 64
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

def build_predictor(cfg_yaml: str, weights_path: str, device: str = "cuda") -> DefaultPredictor:
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_fan_config(cfg)
    add_resnets_compat(cfg)

    # 병합 시 미등록 키 허용 → 병합 후 다시 막기
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_yaml)
    cfg.set_new_allowed(False)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 인스턴스 + 파놉틱만
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

    return DefaultPredictor(cfg)

def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def encode_binary_mask(bin_mask: np.ndarray):
    if mask_utils is None:
        return None
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

# --- NEW: visualizer metadata & seg-info remap helpers ---
def get_vis_metadata_from_cfg(predictor: DefaultPredictor):
    """모델이 사용한 dataset 메타데이터를 가져온다."""
    cfg = predictor.cfg
    ds_name = None
    if getattr(cfg.DATASETS, "TEST", None) and len(cfg.DATASETS.TEST) > 0:
        ds_name = cfg.DATASETS.TEST[0]
    elif getattr(cfg.DATASETS, "TRAIN", None) and len(cfg.DATASETS.TRAIN) > 0:
        ds_name = cfg.DATASETS.TRAIN[0]

    if ds_name is not None:
        meta = MetadataCatalog.get(ds_name)
        # 최소 필수 키 보장
        if not hasattr(meta, "thing_classes"): meta.thing_classes = []
        if not hasattr(meta, "stuff_classes"): meta.stuff_classes = []
        return meta, ds_name

    # 완전 fallback
    fallback_name = "fallback_dummy_meta"
    if fallback_name not in MetadataCatalog.list():
        MetadataCatalog.get(fallback_name).set(thing_classes=[], stuff_classes=[])
    return MetadataCatalog.get(fallback_name), fallback_name

def remap_segments_info_to_contiguous(seg_info_py, meta):
    """
    seg_info의 category_id가 dataset 원래 ID일 수 있으므로
    meta의 *_dataset_id_to_contiguous_id로 연속 ID로 변환한 사본을 돌려준다.
    범위 밖은 드롭해 시각화 오류 방지.
    """
    thing_map = getattr(meta, "thing_dataset_id_to_contiguous_id", None) or {}
    stuff_map = getattr(meta, "stuff_dataset_id_to_contiguous_id", None) or {}

    out = []
    for s in seg_info_py:
        t = dict(s)
        isthing = t.get("isthing", False)
        cid = int(t.get("category_id", -1))

        if isthing:
            if thing_map:
                cid = thing_map.get(cid, cid)
            if not (0 <= cid < len(getattr(meta, "thing_classes", []))):
                # 범위 밖 → 스킵
                continue
        else:
            if stuff_map:
                cid = stuff_map.get(cid, cid)
            if not (0 <= cid < len(getattr(meta, "stuff_classes", []))):
                # 범위 밖 → 스킵
                continue

        t["category_id"] = int(cid)
        out.append(t)
    return out

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser("Instance + Panoptic inference on CARLA grouped (first N per Town×Scenario)")
    ap.add_argument("--base", default="/media/vip-dell/HC/_output3", help="입력 루트(_output3)")
    ap.add_argument("--cfg",  required=True, help="Mask2Former cfg yaml 경로 (FAN 백본 yaml 가능)")
    ap.add_argument("--weights", required=True, help="가중치 경로 (.pth)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--out_pan",  default="", help="파놉틱 출력 루트(기본 <base>/_pred_panoptic)")
    ap.add_argument("--out_inst", default="", help="인스턴스 출력 루트(기본 <base>/_pred_instances)")
    ap.add_argument("--towns", default="", help="쉼표구분 타운 필터 (예: Town01,Town02)")
    ap.add_argument("--scenarios", default="", help="쉼표구분 시나리오 필터")
    ap.add_argument("--first_k_per_group", type=int, default=20, help="타운×시나리오별 앞 K장")
    ap.add_argument("--stride", type=int, default=1, help="샘플 간격(>=1)")
    ap.add_argument("--limit", type=int, default=0, help="전역 상위 N장 제한(0=무제한)")
    ap.add_argument("--save_vis", action="store_true", help="예측 시각화 PNG도 저장")
    return ap.parse_args()

def main():
    args = parse_args()

    base = args.base
    out_pan = args.out_pan or os.path.join(base, "_pred_panoptic")
    out_inst = args.out_inst or os.path.join(base, "_pred_instances")
    ensure_dir(out_pan); ensure_dir(out_inst)

    predictor = build_predictor(args.cfg, args.weights, args.device)
    input_format = str(getattr(predictor.cfg.INPUT, "FORMAT", "BGR")).upper()

    # NEW: 모델이 실제로 쓴 dataset 메타데이터를 사용
    vis_meta, meta_name = get_vis_metadata_from_cfg(predictor)

    # 모든 이미지 수집
    pattern = os.path.join(base, "Town*", "*", "leftImg8bit", "frame_*_leftImg8bit.png")
    all_imgs = sorted(glob.glob(pattern))

    # 타운/시나리오 필터
    if args.towns.strip():
        allow_t = set(t.strip() for t in args.towns.split(",") if t.strip())
        all_imgs = [p for p in all_imgs
                    if (lambda pr: len(pr)>=4 and pr[0] in allow_t)(
                        os.path.relpath(p, base).split(os.sep)
                    )]
    if args.scenarios.strip():
        allow_s = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        all_imgs = [p for p in all_imgs
                    if (lambda pr: len(pr)>=4 and pr[1] in allow_s)(
                        os.path.relpath(p, base).split(os.sep)
                    )]

    # 인덱스 추출 후 그룹핑
    groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for ip in all_imgs:
        parts = os.path.relpath(ip, base).split(os.sep)  # Town/Scenario/leftImg8bit/...
        if len(parts) < 4:
            continue
        town, scen = parts[0], parts[1]
        idx = frame_index_from_path(ip)
        if idx is None:
            continue
        groups.setdefault((town, scen), []).append((idx, ip))

    # 각 그룹에서 앞 K장 선택
    selected_paths: List[str] = []
    k = max(0, int(args.first_k_per_group))
    for key, lst in groups.items():
        lst.sort(key=lambda x: x[0])  # 프레임 번호 오름차순
        take = lst if k == 0 else lst[:k]
        selected_paths.extend([p for _, p in take])

    # stride / limit 적용
    if args.stride > 1:
        selected_paths = selected_paths[::args.stride]
    if args.limit and args.limit > 0:
        selected_paths = selected_paths[:args.limit]

    print(f"[INFO] 대상 이미지: {len(selected_paths)}장  (그룹별 앞 {k}장, stride={args.stride})")

    # GPU 추론(자동 캐스트), 그래디언트 비활성화
    torch.backends.cudnn.benchmark = True
    autocast_device = "cuda" if predictor.cfg.MODEL.DEVICE == "cuda" else "cpu"

    with torch.inference_mode(), torch.autocast(device_type=autocast_device, enabled=(autocast_device=="cuda")):
        for ip in selected_paths:
            img_bgr = cv2.imread(ip)
            if img_bgr is None:
                continue
            img_in = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if input_format == "RGB" else img_bgr

            outputs = predictor(img_in)

            # 경로 파싱
            parts = os.path.relpath(ip, base).split(os.sep)
            town, scen = parts[0], parts[1]
            stem = os.path.basename(ip).replace("_leftImg8bit.png", "")

            # -------- Panoptic 저장 --------
            if "panoptic_seg" in outputs:
                pan_seg_t, seg_info = outputs["panoptic_seg"]   # pan_seg_t: torch.Tensor
                pan_seg_t = pan_seg_t.to("cpu")
                pan_seg_np = pan_seg_t.numpy().astype(np.int32)

                # RGB 인코딩 PNG
                pan_rgb = panoptic_idmap_to_rgb(pan_seg_np)
                pan_dir = os.path.join(out_pan, town, scen)
                ensure_dir(pan_dir)
                cv2.imwrite(
                    os.path.join(pan_dir, f"{stem}_panoptic_pred.png"),
                    cv2.cvtColor(pan_rgb, cv2.COLOR_RGB2BGR)
                )

                # segments_info JSON (직렬화 가능한 기본형으로 변환) - 원본 ID 기준
                seg_info_py = []
                for s in seg_info:
                    d = {}
                    for k, v in s.items():
                        if hasattr(v, "item"):
                            d[k] = v.item()
                        elif isinstance(v, np.generic):
                            d[k] = v.item()
                        else:
                            d[k] = v
                    seg_info_py.append(d)
                save_json(os.path.join(pan_dir, f"{stem}_segments.json"), seg_info_py)

                # NEW: 시각화용으로 contiguous ID로 리매핑한 사본도 저장
                seg_info_contig = remap_segments_info_to_contiguous(seg_info_py, vis_meta)
                save_json(os.path.join(pan_dir, f"{stem}_segments_contig.json"), seg_info_contig)

                # 시각화(옵션) — Visualizer는 torch.Tensor를 기대하므로 pan_seg_t 사용
                if args.save_vis:
                    v = Visualizer(img_bgr[:, :, ::-1], vis_meta, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
                    # ★ remapped 사본을 사용해 IndexError 방지
                    vis = v.draw_panoptic_seg(pan_seg_t, seg_info_contig).get_image()[:, :, ::-1]
                    vis_dir = os.path.join(pan_dir, "_vis")
                    ensure_dir(vis_dir)
                    cv2.imwrite(os.path.join(vis_dir, f"{stem}_panoptic_vis.png"), vis)

            # -------- Instance 저장 --------
            if "instances" in outputs:
                inst = outputs["instances"].to("cpu")
                boxes = inst.pred_boxes.tensor.numpy().tolist() if inst.has("pred_boxes") else []
                classes = inst.pred_classes.numpy().tolist() if inst.has("pred_classes") else []
                scores = inst.scores.numpy().tolist() if inst.has("scores") else []
                masks_rle = []
                if inst.has("pred_masks") and mask_utils is not None:
                    pm = inst.pred_masks.numpy()  # [N, H, W] bool
                    for i in range(pm.shape[0]):
                        rle = encode_binary_mask(pm[i])
                        masks_rle.append(rle)
                elif inst.has("pred_masks"):
                    masks_rle = [None] * len(classes)

                inst_recs = []
                for i in range(len(classes)):
                    rec = {
                        "bbox_xyxy": boxes[i],
                        "category_id": int(classes[i]),
                        "score": float(scores[i]),
                    }
                    if i < len(masks_rle) and masks_rle[i] is not None:
                        rec["segmentation"] = masks_rle[i]
                    inst_recs.append(rec)

                inst_dir = os.path.join(out_inst, town, scen)
                ensure_dir(inst_dir)
                save_json(os.path.join(inst_dir, f"{stem}_instances_pred.json"), {"instances": inst_recs})

                if args.save_vis:
                    v = Visualizer(img_bgr[:, :, ::-1], vis_meta, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
                    vis = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
                    vis_dir = os.path.join(inst_dir, "_vis")
                    ensure_dir(vis_dir)
                    cv2.imwrite(os.path.join(vis_dir, f"{stem}_instances_vis.png"), vis)

    print("[DONE] Inference finished.")

if __name__ == "__main__":
    main()
