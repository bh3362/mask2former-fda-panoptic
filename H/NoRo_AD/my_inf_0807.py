import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from detectron2.config import get_cfg
from mask2former.modeling.backbone.fan import add_fan_config
from detectron2.engine import DefaultPredictor


from detectron2.config import CfgNode as CN

def add_fan_config(cfg):
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
    cfg.MODEL.FAN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


# ========== 1. config 설정 ==========
cfg = get_cfg()
add_fan_config(cfg) 
cfg.merge_from_file("/home/vip-dell/H/NoRo_AD/configs/cityscapes/panoptic-segmentation/fan/maskformer2_fan_hybrid_large_IN21k_384_bs16_90k.yaml")  # 너의 config
cfg.MODEL.WEIGHTS = "/home/vip-dell/H/NoRo_AD/checkpoints/rand.pth"  # ✔️ 너의 학습된 weight로 바꿔
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 11        # ✔️ CARLA class 수
cfg.MODEL.DEVICE = "cuda"  # or "cpu"
predictor = DefaultPredictor(cfg)

# ========== 2. 시나리오별 inference 루프 ==========
base_input_dir = "/home/vip-dell/CARLA_0.9.15/PythonAPI/examples/_output"
base_output_dir = os.path.join(base_input_dir, "_pred_masks")
os.makedirs(base_output_dir, exist_ok=True)

weather_scenarios = ['clear', 'dust', 'fog', 'rain']
frames_per_scenario = 50

for scenario in weather_scenarios:
    print(f"\n🌤️ 인퍼런스 시작: {scenario.upper()}")
    input_dir = os.path.join(base_input_dir, scenario, "leftImg8bit")
    output_dir = os.path.join(base_output_dir, scenario)
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(frames_per_scenario), desc=f"{scenario}", ncols=80):
        filename = f"frame_{i:06d}_leftImg8bit.png"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"frame_{i:06d}_pred_mask.png")

        if not os.path.exists(input_path):
            print(f"❌ 파일 없음: {input_path}")
            continue

        image = cv2.imread(input_path)
        outputs = predictor(image)
        sem_mask = outputs["sem_seg"].argmax(dim=0).cpu().numpy()

        # 0~10 class로 저장 (uint8)
        Image.fromarray(sem_mask.astype(np.uint8)).save(output_path)

print("\n✅ 모든 인퍼런스 완료!")
