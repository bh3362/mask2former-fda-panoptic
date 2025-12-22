from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes_panoptic import load_cityscapes_panoptic

def register_cityscapes_panoptic(root="datasets/cityscapes"):
    for split in ["train", "val"]:
        name = f"cityscapes_panoptic_{split}"
        image_dir = f"{root}/leftImg8bit/{split}"
        gt_dir = f"{root}/gtFine/{split}"
        panoptic_json = f"{root}/gtFine/cityscapes_panoptic_{split}.json"
        panoptic_root = f"{root}/gtFine/cityscapes_panoptic"

        DatasetCatalog.register(
            name,
            lambda split=split: load_cityscapes_panoptic(image_dir, panoptic_json, panoptic_root, gt_dir)
        )

        MetadataCatalog.get(name).set(
            stuff_classes=[
                "road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "rider", "car", "truck", "bus", "train",
                "motorcycle", "bicycle"
            ],
            ignore_label=255,
            evaluator_type="cityscapes_panoptic_seg",
            image_dir=image_dir,
            panoptic_root=panoptic_root,
            panoptic_json=panoptic_json,
            gt_dir=gt_dir
        )
