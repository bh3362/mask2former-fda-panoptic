from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated

def register_wilddash2_panoptic():
    # 경로 설정
    image_root = "./datasets/wilddash2/images/val"
    panoptic_root = "./datasets/wilddash2/annotations/panoptic"
    json_file = "./datasets/wilddash2/annotations/wilddash2_panoptic.json"

    name = "wilddash2_panoptic_val"
    
    register_coco_panoptic_separated(
        name,
        {},
        image_root,
        panoptic_root,
        json_file,
    )

    MetadataCatalog.get(name).set(
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        image_root=image_root,
        panoptic_root=panoptic_root,
        panoptic_json=json_file,
        thing_classes=[
            "person", "car", "bicycle", "motorcycle", "bus", "truck", "rider", "animal"
        ],
        stuff_classes=[
            "road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "ground", "dynamic"
        ],
        classes=[
            "road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "ground", "dynamic",
            "person", "car", "bicycle", "motorcycle", "bus", "truck", "rider", "animal"
        ]
    )
