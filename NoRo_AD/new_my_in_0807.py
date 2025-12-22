import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode as CN

from mask2former import add_maskformer2_config
from collections import Counter


# FAN 백본 설정 함수
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


# =========================
# 1. Config 설정
# =========================

cfg = get_cfg()

# 🔧 ResNet 관련 dummy 설정 (FAN을 쓸 때도 _BASE_에 존재하므로 필요함)
cfg.MODEL.RESNETS = CN()
cfg.MODEL.RESNETS.DEPTH = 50
cfg.MODEL.RESNETS.STEM_TYPE = ""
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

# FAN + Mask2Former 설정 등록
add_fan_config(cfg)
add_maskformer2_config(cfg)

# config 및 weights 설정
cfg.merge_from_file("/home/vip-dell/NoRo_AD/configs/cityscapes/panoptic-segmentation/fan/maskformer2_fan_hybrid_large_IN21k_384_bs16_90k.yaml")
cfg.MODEL.WEIGHTS = "/home/vip-dell/H/NoRo_AD/checkpoints/rand.pth"
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 11
cfg.MODEL.MASK_FORMER.TASKS = ["semantic"]
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# =========================
# 2. 컬러 매핑 설정
# =========================

COLOR_MAP = {
    0: (128, 64, 128),   # road
    1: (244, 35, 232),   # sidewalk
    2: (70, 70, 70),     # building
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),   # traffic light
    7: (220, 220, 0),    # traffic sign
    8: (107, 142, 35),   # vegetation
    9: (152, 251, 152),  # terrain
    10: (70, 130, 180),  # sky
}

# =========================
# 3. 인퍼런스 실행
# =========================

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
        output_mask_path = os.path.join(output_dir, f"frame_{i:06d}_pred_mask.png")
        output_color_path = os.path.join(output_dir, f"frame_{i:06d}_color_mask.png")

        if not os.path.exists(input_path):
            print(f"❌ 파일 없음: {input_path}")
            continue

        image = cv2.imread(input_path)
        image = image.copy()##
        outputs = predictor(image)
        sem_mask = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
        class_counts = Counter(sem_mask.flatten())
        print(f"[{scenario.upper()} | frame {i:06d}] 클래스 분포: {dict(class_counts)}")

        # [1] 흑백 클래스 마스크 저장
        Image.fromarray(sem_mask.astype(np.uint8)).save(output_mask_path)

        # [2] 컬러 마스크 생성 및 저장
        color_mask = np.zeros((sem_mask.shape[0], sem_mask.shape[1], 3), dtype=np.uint8)
        for cls_id, color in COLOR_MAP.items():
            color_mask[sem_mask == cls_id] = color
        Image.fromarray(color_mask).save(output_color_path)




print("\n✅ 모든 인퍼런스 및 저장 완료!")
