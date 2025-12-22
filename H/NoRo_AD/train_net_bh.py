# train_mask2former_carla.py
# Mask2Former semantic segmentation training script (CARLA -> Cityscapes-19)
from detectron2.engine import hooks  # ← 추가

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except Exception:
    pass

import copy, itertools, logging, os, random
from collections import OrderedDict
from typing import Any, Dict, List, Set

import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, COCOEvaluator, COCOPanopticEvaluator,
    DatasetEvaluators, LVISEvaluator, SemSegEvaluator, verify_results
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# Mask2Former
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper, COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator, MaskFormerInstanceDatasetMapper, MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper, SemanticSegmentorWithTTA, add_maskformer2_config,
)

# ⬇️ 네가 만든 데이터 등록 유틸
from my_mapping import register_carla_semantic


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(DefaultTrainer):
    """Mask2Former용 Trainer 확장 (semantic 전용 mapper 지원)"""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in ["coco_panoptic_seg", "ade20k_panoptic_seg", "cityscapes_panoptic_seg",
                              "mapillary_vistas_panoptic_seg"]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "cityscapes_instance":
            assert torch.cuda.device_count() > comm.get_rank()
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert torch.cuda.device_count() > comm.get_rank()
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output=output_folder)

        if len(evaluator_list) == 0:
            raise NotImplementedError(f"no Evaluator for dataset {dataset_name} type={evaluator_type}")
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            return build_detection_train_loader(cfg, mapper=None)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hyper = copy.copy(defaults)
                if "backbone" in module_name:
                    hyper["lr"] = hyper["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if ("relative_position_bias_table" in module_param_name) or ("absolute_pos_embed" in module_param_name):
                    hyper["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyper["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyper["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyper})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = cfg.SOLVER.CLIP_GRADIENTS.ENABLED and \
                     cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and clip_norm_val > 0.0

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)
            return FullModelGradientClippingOptimizer if enable else optim

        if cfg.SOLVER.OPTIMIZER == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif cfg.SOLVER.OPTIMIZER == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {cfg.SOLVER.OPTIMIZER}")
        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})


def setup(args):
    """Create configs, register datasets, and perform basic setups."""
    set_seed(42)

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # === (1) Dataset registration: CARLA _output3 ===
    # Expecting: <root>/TownXX/<SCENARIO>/{leftImg8bit, gtFine}/frame_***_{leftImg8bit|gtFine_labelIds}.png
    carla_root = "/media/vip-dell/HC/_output3"
    try:
        register_carla_semantic("carla_sem_seg_train", carla_root, split="train")
        register_carla_semantic("carla_sem_seg_val",   carla_root, split="val")
    except TypeError:
        # fallback if your helper doesn't take 'split'
        register_carla_semantic("carla_sem_seg_train", carla_root)
        register_carla_semantic("carla_sem_seg_val",   carla_root)

    # === (2) Load base config (your Cityscapes/Mask2Former yaml) ===
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # === (3) Force semantic-only pipeline ===
    cfg.DATASETS.TRAIN = ("carla_sem_seg_train",)
    cfg.DATASETS.TEST  = ("carla_sem_seg_val",)
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    # evaluation: semantic only
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    # classes: Cityscapes-19
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19

    # sensible defaults for single GPU (override with --opts if you want)
    cfg.DATALOADER.NUM_WORKERS = max(2, os.cpu_count() // 4)
    cfg.SOLVER.IMS_PER_BATCH = cfg.get("SOLVER", {}).get("IMS_PER_BATCH", 2)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.TEST.EVAL_PERIOD = 2000

    # output dir
    if not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = os.path.join(os.getcwd(), "output_mask2former_carla19")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
