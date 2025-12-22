import os
import glob
import numpy as np
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog

# CARLA semantic ID → contiguous Detectron2 ID 매핑
CARLA_SEM2CITY_ID = {
    7: 0, 8: 1, 10: 2, 12: 3, 13: 4, 14: 5,
    18: 6, 19: 7, 20: 8, 21: 9, 22: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18
}
_CARLA_IGNORE_LABEL = 255
CARLA_CATEGORY_IDS = list(CARLA_SEM2CITY_ID.keys())

def get_carla_sem_seg_dicts(root_dir):
    dataset_dicts = []
    temp_dir = os.path.join(root_dir, "temp_sem_seg_masks")
    os.makedirs(temp_dir, exist_ok=True)

    contig_map = {orig: i for i, orig in enumerate(CARLA_CATEGORY_IDS)}
    used_ids = set()
    global_idx = 0

    for weather in ["clear", "dust", "fog", "rain"]:
        print(f"\n[INFO] Processing weather folder: {weather}")
        img_dir = os.path.join(root_dir, weather, "leftImg8bit")
        gt_dir  = os.path.join(root_dir, weather, "gtFine")

        label_paths = glob.glob(os.path.join(gt_dir, "*_gtFine_instanceIds.png"))
        print(f"[INFO] Found {len(label_paths)} label files in {gt_dir}")

        for label_path in sorted(label_paths):
            try:
                stem = os.path.basename(label_path).replace("_gtFine_instanceIds.png", "")
                img_path = os.path.join(img_dir, f"{stem}_leftImg8bit.png")
                if not os.path.exists(img_path):
                    print(f"[WARN] Image not found: {img_path}")
                    continue

                # ✅ instanceIds.png → 정수형 2D 이미지로 강제 변환
                sem_mask = np.array(Image.open(label_path).convert("I"))
                if sem_mask is None or sem_mask.ndim != 2:
                    print(f"[ERROR] Failed to load or invalid mask: {label_path}, shape={sem_mask.shape}")
                    continue

                # instanceId → semanticId (CARLA: semantic_id * 1000 + instance_id)
                sem_mask = (sem_mask // 1000).astype(np.uint8)
                print(f"[DEBUG] {label_path} semantic IDs: {np.unique(sem_mask)}")

                # semanticId → contiguous ID 리매핑
                sem_mask_remapped = np.full_like(sem_mask, _CARLA_IGNORE_LABEL, dtype=np.uint8)
                for orig_id, contig_id in contig_map.items():
                    sem_mask_remapped[sem_mask == orig_id] = contig_id
                    used_ids.add(contig_id)

                sem_path = os.path.join(temp_dir, f"{global_idx:04d}_sem_seg.png")
                Image.fromarray(sem_mask_remapped).save(sem_path)

                h, w = sem_mask.shape
                dataset_dicts.append({
                    "file_name": img_path,
                    "sem_seg_file_name": sem_path,
                    "image_id": global_idx,
                    "height": h,
                    "width": w,
                })
                global_idx += 1

            except Exception as e:
                print(f"[EXCEPTION] {label_path}: {e}")
                continue

    print(f"\n✅ 총 처리된 이미지 수: {len(dataset_dicts)}")
    print(f"[INFO] Used remapped class IDs: {sorted(list(used_ids))}")
    return dataset_dicts

def register_carla_semantic(name, root_dir):
    DatasetCatalog.register(name, lambda: get_carla_sem_seg_dicts(root_dir))
    stuff_names = list(CARLA_SEM2CITY_ID.keys())  # 실제 semantic ID 목록
    stuff_colors = [(i*13 % 256, i*7 % 256, i*19 % 256) for i in range(len(stuff_names))]

    MetadataCatalog.get(name).set(
        evaluator_type="sem_seg",
        ignore_label=_CARLA_IGNORE_LABEL,
        stuff_classes=stuff_names,
        stuff_colors=stuff_colors,
        stuff_dataset_id_to_contiguous_id=CARLA_SEM2CITY_ID
    )

    print(f"[INFO] '{name}' 등록 완료. 총 샘플 수 = {len(DatasetCatalog.get(name))}")

if __name__ == "__main__":
    ROOT = "/home/vip-dell/CARLA_0.9.15/PythonAPI/examples/_output"
    register_carla_semantic("carla_sem_seg", ROOT)

    dataset = DatasetCatalog.get("carla_sem_seg")
    print(f"[INFO] 샘플 수: {len(dataset)}")
    if dataset:
        print("[INFO] 첫 번째 항목 예시:")
        print(dataset[0])
