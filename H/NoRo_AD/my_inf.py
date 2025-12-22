import os
import cv2
import torch
import argparse
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from my_mapping import register_carla_semantic
import numpy as np  # ✅ 누락되어 있으므로 추가
from detectron2.data import MetadataCatalog



import sys
sys.path.insert(0, "/home/vip-dell/H/NoRo_AD")  # ✅ mask2former 디렉토리가 포함된 상위 디렉토리

from mask2former import add_maskformer2_config  # ✅ mask2former/config.py 안에 정의된 함수



def setup_cfg(config_path: str, weights_path: str):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DATASETS.TEST = ("carla_sem_seg",)
    cfg.MODEL.MASK_FORMER.TASKS = ["semantic"]  # ✅ semantic 모델임을 명시

    return cfg


def inference_semantic(cfg, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = MetadataCatalog.get(dataset_name)
    predictor = DefaultPredictor(cfg)
        # ✅ 모델 디바이스 확인
    device = next(predictor.model.parameters()).device
    print(f"[INFO] Model is using device: {device}")

    dataset_dicts = DatasetCatalog.get(dataset_name)

    for idx, d in enumerate(dataset_dicts):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        sem_seg = outputs["sem_seg"].argmax(dim=0).to("cpu").numpy()

        # ✅ [1] label 값 (0~18)을 저장 (평가용)
        label_path = os.path.join(output_dir, f"semseg_{idx:04d}.png")
        cv2.imwrite(label_path, sem_seg.astype(np.uint8))

        # ✅ [2] 시각화 이미지 저장 (참고용)
        colorized = np.zeros((sem_seg.shape[0], sem_seg.shape[1], 3), dtype=np.uint8)
        for id, color in enumerate(metadata.stuff_colors):
            colorized[sem_seg == id] = color
        vis_path = os.path.join(output_dir, f"semseg_vis_{idx:04d}.png")
        cv2.imwrite(vis_path, colorized)

        print(f"[INFO] Saved label: {label_path}")
        print(f"[INFO] Saved vis:   {vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, help="Path to the config.yaml")
    parser.add_argument("--weights", required=True, help="Path to the model .pth weights")
    parser.add_argument("--output", default="./results", help="Output directory to save visualized results")
    args = parser.parse_args()

    # CARLA 데이터셋 최초 1회 등록 (주석 해제 시 사용)
    ROOT = "/home/vip-dell/CARLA_0.9.15/PythonAPI/examples/_output"
    register_carla_semantic("carla_sem_seg", ROOT)

    cfg = setup_cfg(args.config_file, args.weights)
    inference_semantic(cfg, "carla_sem_seg", args.output)

