# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os, sys, glob, time, json, argparse
# import numpy as np, cv2, torch

# # [유지] resister.py를 임포트하여 데이터셋을 등록합니다.
# sys.path.insert(0, "/home/vip-dell/NoRo_AD")
# import resister 

# from detectron2.config import get_cfg
# from detectron2.utils.logger import setup_logger
# from detectron2.data.detection_utils import read_image
# from detectron2.data import MetadataCatalog
# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.utils.visualizer import Visualizer
# from mask2former import add_maskformer2_config
# from panopticapi.utils import id2rgb

# # --- [필수] Visualizer를 위한 ID -> Index 매핑 테이블 ---
# # resister.py / train_from_scratch.py에서 정의한 순서와 정확히 일치해야 함

# # Global ID (0-18) -> Local Index (0-9)
# THING_ID_TO_LOCAL_IDX = {
#     6: 0,  # "traffic light"
#     7: 1,  # "traffic sign"
#     11: 2, # "person"
#     12: 3, # "rider"
#     13: 4, # "car"
#     14: 5, # "truck"
#     15: 6, # "bus"
#     16: 7, # "train"
#     17: 8, # "motorcycle"
#     18: 9  # "bicycle"
# }

# # Global ID (0-18) -> Local Index (0-8)
# STUFF_ID_TO_LOCAL_IDX = {
#     0: 0,  # "road"
#     1: 1,  # "sidewalk"
#     2: 2,  # "building"
#     3: 3,  # "wall"
#     4: 4,  # "fence"
#     5: 5,  # "pole"
#     8: 6,  # "vegetation"
#     9: 7,  # "terrain"
#     10: 8  # "sky"
# }
# # ----------------------------------------------------

# def build_cfg(args):
#     cfg = get_cfg(); add_deeplab_config(cfg); add_maskformer2_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     if args.opts: cfg.merge_from_list(args.opts)
#     cfg.defrost()
#     cfg.INPUT.MASK_FORMAT = "bitmask"
#     cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19
#     cfg.DATASETS.TEST = ("carla_final_panoptic_val",) # 등록된 실제 이름
#     cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
#     cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
#     cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
#     cfg.INPUT.FORMAT = "BGR"
#     cfg.INPUT.MIN_SIZE_TEST = 512
#     cfg.INPUT.MAX_SIZE_TEST = 512
#     if args.weights: cfg.MODEL.WEIGHTS = args.weights
#     cfg.freeze(); return cfg

# @torch.inference_mode()
# def run_on_image(predictor, img_bgr, meta):
#     out = predictor(img_bgr)
#     if "panoptic_seg" not in out:
#         return img_bgr.copy(), None, None
    
#     pan_seg, segs = out["panoptic_seg"] # 'segs'는 글로벌 ID (0-18)를 포함

#     # --- [유지] Visualizer에 전달하기 전 ID를 로컬 인덱스로 변환 ---
#     mapped_segs = []
#     for segment in segs:
#         new_seg = segment.copy() # 원본 복사
#         global_cid = new_seg["category_id"]
        
#         if new_seg["isthing"]:
#             # 'Things' 매핑
#             if global_cid in THING_ID_TO_LOCAL_IDX:
#                 new_seg["category_id"] = THING_ID_TO_LOCAL_IDX[global_cid]
#                 mapped_segs.append(new_seg)
#         else:
#             # 'Stuff' 매핑
#             if global_cid in STUFF_ID_TO_LOCAL_IDX:
#                 new_seg["category_id"] = STUFF_ID_TO_LOCAL_IDX[global_cid]
#                 mapped_segs.append(new_seg)
#     # --------------------------------------------------------

#     vis = Visualizer(img_bgr[:,:,::-1], metadata=meta)
    
#     # 'segs' 대신 변환된 'mapped_segs'를 전달하여 IndexError 해결
#     vis = vis.draw_panoptic_seg_predictions(pan_seg.to("cpu"), mapped_segs) 
    
#     overlay = vis.get_image()[:,:,::-1]
    
#     # 원본 'segs' (글로벌 ID 0-18)는 JSON 저장을 위해 반환
#     return overlay, pan_seg.to("cpu").numpy().astype(np.uint32), segs

# def ensure(p): os.makedirs(p, exist_ok=True); return p

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config-file", required=True)
    
#     # [수정] 'addGument' -> 'add_argument' 오타 수정
#     ap.add_argument("--input", required=True, nargs="+") 
    
#     ap.add_argument("--output", required=True)
#     ap.add_argument("--weights", default="")
#     ap.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
#     args = ap.parse_args()

#     setup_logger(); cfg = build_cfg(args)
#     from demo.predictor import VisualizationDemo
#     demo = VisualizationDemo(cfg); predictor = demo.predictor

#     meta = MetadataCatalog.get("carla_final_panoptic_val") # 등록된 실제 이름
    
#     # [검증] 메타데이터와 수동 매핑 테이블이 일치하는지 확인
#     try:
#         assert meta.thing_classes == ["traffic light", "traffic sign", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
#         assert meta.stuff_classes == ["road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky"]
#         print("[INFO] Metadata check passed.")
#     except AssertionError:
#         print("[FATAL] Metadata mismatch! 'resister.py'의 클래스 순서와 'infer.py'의 ID 매핑 테이블이 다릅니다.")
#         sys.exit(1)


#     root = ensure(args.output)
#     out_overlay = ensure(os.path.join(root, "overlay"))
#     out_panpng = ensure(os.path.join(root, "panoptic_color"))
#     out_segjson = ensure(os.path.join(root, "segments_info"))

#     paths = []
#     for s in args.input:
#         if any(x in s for x in ["*","?","[","]"]): paths += sorted(glob.glob(s))
#         else: paths.append(s)
        
#     # [수정] 'os.Dpath' -> 'os.path' 오타 수정
#     if len(paths)==1 and os.path.isdir(paths[0]): 
#         tmp=[]
#         for _r,_d,fs in os.walk(paths[0]): tmp += [os.path.join(_r,f) for f in fs]
#         paths = sorted(tmp)

#     for ip in paths:
#         if not os.path.isfile(ip): continue
#         img = read_image(ip, format="BGR")
#         t0 = time.time()
        
#         overlay, pan_id, segs = run_on_image(predictor, img, meta)
        
#         stem = os.path.splitext(os.path.basename(ip))[0]
#         cv2.imwrite(os.path.join(out_overlay, f"{stem}_overlay.png"), overlay)
        
#         if pan_id is not None:
#             color = id2rgb(pan_id)[:,:,::-1]
#             cv2.imwrite(os.path.join(out_panpng, f"{stem}_panoptic.png"), color)
            
#             with open(os.path.join(out_segjson, f"{stem}_segments.json"), "w") as f:
#                 json.dump({"segments_info": segs}, f, ensure_ascii=False)
                
#         print(f"[OK] {os.path.basename(ip)} {(time.time()-t0):.2f}s")
#     print(f"[DONE] {root}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, time, json, argparse
import numpy as np, cv2, torch

# [유지] resister.py를 임포트하여 데이터셋을 등록합니다.
sys.path.insert(0, "/home/vip-dell/NoRo_AD")
import resister 

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer
from mask2former import add_maskformer2_config
from panopticapi.utils import id2rgb

# --- [유지] Visualizer를 위한 ID -> Index 매핑 테이블 ---
THING_ID_TO_LOCAL_IDX = {
    6: 0, #"traffic light"
    7: 1, # "traffic sign"
    11: 2, # "person"
    12: 3, # "rider"
    13: 4, # "car"
    14: 5, # "truck"
    15: 6, # "bus"
    16: 7, # "train"
    17: 8, # "motorcycle"
    18: 9   # "bicycle"
}
STUFF_ID_TO_LOCAL_IDX = {
    0: 0, # "road"
    1: 1, # "sidewalk"
    2: 2, # "building"
    3: 3, # "wall"
    4: 4, # "fence"
    5: 5, # "pole"
    8: 6, # "vegetation"
    9: 7, # "terrain"
    10: 8 # "sky"
}
# ----------------------------------------------------

def build_cfg(args):
    cfg = get_cfg(); add_deeplab_config(cfg); add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts: cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19
    cfg.DATASETS.TEST = ("carla_final_panoptic_val",) # 등록된 실제 이름
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    
    # [!!! 핵심 수정 !!!]
    cfg.INPUT.FORMAT = "RGB" 
    
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    if args.weights: cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze(); return cfg

# 🌟 [추가] 추론 결과 Panoptic ID 재매핑 함수 (Numpy 버전)
def _remap_prediction_ids_np(pan_seg_np: np.ndarray, sem_seg_tensor: torch.Tensor) -> np.ndarray:
    ROAD_INFERENCE_ID = 26
    
    pan_remapped = pan_seg_np.copy()
    sem_seg_np = sem_seg_tensor.numpy()

    # Panoptic ID 0이면서 Semantic ID 0인 영역을 road로 간주하고 99999로 재매핑
    road_mask = (pan_remapped == 0) & (sem_seg_np == 0)
    pan_remapped[road_mask] = ROAD_INFERENCE_ID
    
    return pan_remapped

@torch.inference_mode()
# 🌟 [수정] sem_seg를 반환하도록 변경
def run_on_image(predictor, img_bgr, meta): 
    out = predictor(img_bgr)
    # pan_seg, segs, sem_seg 순서로 반환
    if "panoptic_seg" not in out:
        return img_bgr.copy(), None, None, None 
    
    pan_seg, segs = out["panoptic_seg"] # 'pan_seg'는 torch.Tensor
    sem_seg = out.get("semantic_seg", None) # torch.Tensor or None

    # --- [유지] Visualizer(IndexError)를 위한 ID 변환 로직 ---
    mapped_segs = []
    for segment in segs:
        new_seg = segment.copy() 
        global_cid = new_seg["category_id"]
        
        if new_seg["isthing"]:
            if global_cid in THING_ID_TO_LOCAL_IDX:
                new_seg["category_id"] = THING_ID_TO_LOCAL_IDX[global_cid]
                mapped_segs.append(new_seg)
        else:
            if global_cid in STUFF_ID_TO_LOCAL_IDX:
                new_seg["category_id"] = STUFF_ID_TO_LOCAL_IDX[global_cid]
                mapped_segs.append(new_seg)
    # --------------------------------------------------------

    # Visualizer는 BGR -> RGB 변환이 필요합니다.
    vis = Visualizer(img_bgr[:,:,::-1], metadata=meta)
    
    vis = vis.draw_panoptic_seg_predictions(pan_seg.to("cpu"), mapped_segs) 
    overlay = vis.get_image()[:,:,::-1] # 다시 BGR로 변환
    
    # 🌟 [수정]: pan_seg (np), segs, sem_seg (torch)를 모두 반환
    return (
        overlay, 
        pan_seg.to("cpu").numpy().astype(np.uint32), 
        segs, 
        sem_seg.to("cpu") if sem_seg is not None else None
    )

def ensure(p): os.makedirs(p, exist_ok=True); return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--input", required=True, nargs="+") 
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = ap.parse_args()

    setup_logger(); cfg = build_cfg(args)
    from demo.predictor import VisualizationDemo
    demo = VisualizationDemo(cfg); predictor = demo.predictor

    meta = MetadataCatalog.get("carla_final_panoptic_val") # 등록된 실제 이름
    
    # [검증] 메타데이터와 수동 매핑 테이블이 일치하는지 확인
    try:
        assert meta.thing_classes == ["traffic light", "traffic sign", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        assert meta.stuff_classes == ["road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky"]
        print("[INFO] Metadata check passed.")
    except AssertionError:
        print("[FATAL] Metadata mismatch! 'resister.py'의 클래스 순서와 'infer.py'의 ID 매핑 테이블이 다릅니다.")
        sys.exit(1)


    root = ensure(args.output)
    out_overlay = ensure(os.path.join(root, "overlay"))
    out_panpng = ensure(os.path.join(root, "panoptic_color"))
    out_segjson = ensure(os.path.join(root, "segments_info"))

    paths = []
    for s in args.input:
        if any(x in s for x in ["*","?","[","]"]): paths += sorted(glob.glob(s))
        else: paths.append(s)
        
    if len(paths)==1 and os.path.isdir(paths[0]): 
        tmp=[]
        for _r,_d,fs in os.walk(paths[0]): tmp += [os.path.join(_r,f) for f in fs]
        paths = sorted(tmp)

    for ip in paths:
        if not os.path.isfile(ip): continue
        
        # [유지] 이미지는 BGR로 읽어옵니다. (OpenCV/Visualizer 호환)
        img = read_image(ip, format="BGR") 
        t0 = time.time()
        
        # 🌟 [수정]: run_on_image 반환 값 변경 (sem_seg 추가)
        overlay, pan_id, segs, sem_seg = run_on_image(predictor, img, meta) 
        
        stem = os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(out_overlay, f"{stem}_overlay.png"), overlay)
        
        # 🌟 [수정]: pan_id와 sem_seg가 모두 있을 때만 재매핑 및 저장
        if pan_id is not None and sem_seg is not None:
            
            # 🌟 [추가]: Panoptic ID 0을 99999로 재매핑
            pan_remapped = _remap_prediction_ids_np(pan_id, sem_seg)

            # 🌟 [추가]: 재매핑된 ID 맵을 사용하여 컬러맵 생성 및 저장
            color = id2rgb(pan_remapped)[:,:,::-1]
            cv2.imwrite(os.path.join(out_panpng, f"{stem}_panoptic.png"), color)
            
            with open(os.path.join(out_segjson, f"{stem}_segments.json"), "w") as f:
                json.dump({"segments_info": segs}, f, ensure_ascii=False)
                
        print(f"[OK] {os.path.basename(ip)} {(time.time()-t0):.2f}s")
    print(f"[DONE] {root}")

if __name__ == "__main__":
    main()