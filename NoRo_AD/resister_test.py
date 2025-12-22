from detectron2.data import MetadataCatalog, DatasetCatalog

# --- TEST 등록 (train_bh_test) ---
OUT_ROOT_TEST = "/media/vip-dell/HC/train_bh_test"
TEST_NAME     = "carla_panoptic_test_png_v4"

IMAGE_TEST_DIR = f"{OUT_ROOT_TEST}/leftImg8bit/test"
SEM_TEST_DIR   = f"{OUT_ROOT_TEST}/gtFine/test"
PAN_TEST_ROOT  = f"{OUT_ROOT_TEST}/panoptic_gt_id/test"
PAN_TEST_JSON  = f"{OUT_ROOT_TEST}/panoptic_json/panoptic_test.json"

# (train/val 때와 동일한 방식으로) DatasetCatalog.register(TEST_NAME, ...) 해주고
# MetadataCatalog 설정:
m = MetadataCatalog.get(TEST_NAME)
m.set(
    evaluator_type="coco_panoptic_seg",
    image_root=IMAGE_TEST_DIR,
    panoptic_root=PAN_TEST_ROOT,
    panoptic_json=PAN_TEST_JSON,
    sem_seg_root=SEM_TEST_DIR,
    ignore_label=255,
    # 19-class identity mapping 보장
    panoptic_dataset_id_to_contiguous_id={i:i for i in range(19)},
)
