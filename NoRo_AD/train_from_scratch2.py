import gc, torch
import torch.backends.cudnn as cudnn
gc.collect()
if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    cudnn.benchmark = True

import sys, os, json, logging, copy, itertools
from collections import OrderedDict
from typing import Any, Dict, List, Set

# --- 프로젝트 경로 ---
sys.path.append("/home/vip-dell/NoRo_AD")
# import resister  # [!!!] 더 이상 import하지 않음

from tqdm import tqdm
import numpy as np
import cv2 # [!!!] cv2 import 추가 (resister 코드용)

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    MetadataCatalog,
    DatasetCatalog, # [!!!] DatasetCatalog 임포트
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,    
    DatasetEvaluator,       # [!!!] DatasetEvaluator 임포트 (SavePanopticPredictions용)
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

# Mask2Former
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,    # LSJ(학습용)
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

# ===== 평가용 세이프 매퍼 (이미지 리사이즈만; H/W 보전) =====
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data.transforms import ResizeShortestEdge

class COCOPanopticEvalSafeMapper(DatasetMapper):
    @classmethod
    def from_config(cls, cfg, is_train=False):
        augs = [ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                   max_size=cfg.INPUT.MAX_SIZE_TEST)]
        return {
            "is_train": False,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": False,
            "use_keypoint": False,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": False,
        }
    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        aug_input = T.AugInput(image)
        _ = self.augmentations(aug_input)
        image = aug_input.image
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        dataset_dict["image"] = image
        dataset_dict.pop("annotations", None)
        return dataset_dict

# ===== 평가시 경로/카테고리만 로깅 (변환 오버라이드 X) =====
class _LogPanopticEvaluator(COCOPanopticEvaluator):
    def __init__(self, dataset_name, output_folder=None):
        super().__init__(dataset_name, output_folder)
        jp = getattr(self._metadata, "panoptic_json", None)
        pr = getattr(self._metadata, "panoptic_root", None)
        try:
            with open(jp, "r") as f:
                cats = json.load(f).get("categories", [])
            print(f"[EVAL-LOG] dataset={dataset_name}")
            print(f"[EVAL-LOG] panoptic_json={jp}")
            print(f"[EVAL-LOG] panoptic_root={pr}")
            print(f"[EVAL-LOG] #categories={len(cats)} (expect 19)")

            # 메타에 적재된 매핑 현황만 로깅(변환 오버라이드 금지)
            thing_d2c = getattr(self._metadata, "thing_dataset_id_to_contiguous_id", {})
            stuff_d2c = getattr(self._metadata, "stuff_dataset_id_to_contiguous_id", {})
            print(f"[EVAL-LOG] thing d2c: {thing_d2c}")
            print(f"[EVAL-LOG] stuff d2c: {stuff_d2c}")
        except Exception as e:
            print("[EVAL-LOG] failed to read json:", e)

# ===== GT JSON이 정말 19인지 강제 확인 =====
def _assert_19cats(dataset_name: str):
    m = MetadataCatalog.get(dataset_name)
    jp = getattr(m, "panoptic_json", None)
    pr = getattr(m, "panoptic_root", None)
    assert jp and os.path.isfile(jp), f"[FATAL] {dataset_name}: panoptic_json 없음: {jp}"
    with open(jp, "r") as f:
        jj = json.load(f)
    cats = jj.get("categories", [])
    print(f"[CHECK] {dataset_name} panoptic_json: {jp}")
    print(f"[CHECK] {dataset_name} panoptic_root: {pr}")
    print(f"[CHECK] {dataset_name} categories = {len(cats)}  (expected 19)")
    print(f"[CHECK] {dataset_name} first_ids = {[c['id'] for c in cats[:10]]}")
    assert len(cats) == 19, f"[FATAL] {dataset_name}: categories={len(cats)} != 19"

# ===== Debug Guard (세그 라벨 범위 즉시 검사; CUDA assert 방지) =====
class _DebugPanopticGuard:
    def __init__(self, mapper, num_classes: int, ignore_val: int = 255):
        self.m = mapper; self.K = int(num_classes); self.IGNORE = int(ignore_val)
    def __call__(self, dataset_dict):
        out = self.m(dataset_dict)
        img_id = dataset_dict.get("image_id", dataset_dict.get("file_name", "unknown"))
        seg_infos = out.get("segments_info", None)
        if isinstance(seg_infos, list):
            for si in seg_infos:
                cid = int(si.get("category_id", -1))
                if not (0 <= cid < self.K):
                    raise ValueError(f"[DEBUG] segments_info.category_id={cid} out of [0,{self.K-1}] @ {img_id}")
        return out

# ===== 추가: panoptic 예측을 파일로 저장하는 evaluator =====
# [!!!] 'id2rgb' NameError 해결
from panopticapi.utils import id2rgb
# import cv2 (이미 위에서 import 함)

class SavePanopticPredictions(DatasetEvaluator):
    """
    Detectron2가 생성한 outputs의 panoptic_seg을 컬러 PNG(+원본 segments_info JSON)로 저장.
    - PNG 경로: <OUTPUT_DIR>/inference_<dataset>/panoptic_pred/<image_id>.png
    - JSON 경로: <...>/panoptic_pred/<image_id>.json (segments_info만 저장)
    
    [ROAD FIX]: 모델 추론 결과 Panoptic ID 0 ('road')을 99999로 재매핑하여 시각화 누락을 방지합니다.
    """
    def __init__(self, dataset_name: str, output_root: str, verbose_first_n: int = 5):
        self.dataset_name = dataset_name
        self.save_dir = os.path.join(output_root, f"inference_{dataset_name}", "panoptic_pred")
        os.makedirs(self.save_dir, exist_ok=True)
        self._n = 0
        self._verbose_first_n = int(verbose_first_n)

    def reset(self):
        pass

    # 🌟 [추가] 추론 ID를 GT 형식으로 재매핑하는 함수
    def _remap_prediction_ids(self, pan_seg_tensor: torch.Tensor, sem_seg_tensor: torch.Tensor) -> np.ndarray:
        # GT 데이터셋을 생성할 때 사용한 Road ID (99999)와 동일하게 맞춥니다.
        ROAD_INFERENCE_ID = 26
        
        # 텐서를 복사하여 numpy 배열로 변환
        pan_remapped = pan_seg_tensor.clone().to(torch.uint32).cpu().numpy()
        sem_seg_np = sem_seg_tensor.cpu().numpy()

        # 모델이 추론한 Panoptic ID가 0 이면서 (pan_remapped == 0), 
        # 모델의 Semantic ID가 0인 영역 (sem_seg_np == 0)은 'road'입니다.
        road_mask = (pan_remapped == 0) & (sem_seg_np == 0)

        # 'road' 영역의 Panoptic ID를 99999로 재매핑합니다.
        pan_remapped[road_mask] = ROAD_INFERENCE_ID
        
        return pan_remapped

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            image_id = inp.get("image_id", None)
            if image_id is None:
                base = os.path.splitext(os.path.basename(inp["file_name"]))[0]
                image_id = base

            panoptic = out.get("panoptic_seg", None)
            # 🌟 [추가] semantic_seg 출력 확보 (재매핑에 필요)
            sem_seg_tensor = out.get("semantic_seg", None) 
            
            if panoptic is None or sem_seg_tensor is None:
                if self._n < self._verbose_first_n:
                    print(f"[SAVE] skip(no panoptic_seg or semantic_seg) image_id={image_id}")
                continue

            try:
                pan_seg, segments_info = panoptic
                
                # 🌟 [수정] 재매핑 함수 호출
                pan_remapped = self._remap_prediction_ids(pan_seg, sem_seg_tensor)
                
                # 🌟 [수정] 컬러맵 생성에 재매핑된 pan_remapped 사용
                color = id2rgb(pan_remapped)  # (H,W,3) uint8 RGB

                png_path = os.path.join(self.save_dir, f"{image_id}.png")
                json_path = os.path.join(self.save_dir, f"{image_id}.json")

                ok = cv2.imwrite(png_path, color[..., ::-1]) # BGR로 저장
                
                # JSON은 segments_info를 그대로 저장합니다 (평가 스크립트가 처리하도록).
                with open(json_path, "w") as f:
                    json.dump({"segments_info": segments_info}, f, ensure_ascii=False)

                self._n += 1
                if self._n <= self._verbose_first_n:
                    print(f"[SAVE] wrote PNG (Road Remapped): {png_path}")
                    print(f"[SAVE] wrote JSON: {json_path}")
            except Exception as e:
                print(f"[SAVE][ERR] image_id={image_id} -> {e}")

    def evaluate(self):
        print(f"[SAVE] panoptic predictions -> {self.save_dir}")
        return {}

# ===== Trainer =====
def infer_task_name(cfg) -> str:
    sem = bool(cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON)
    ins = bool(cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON)
    pan = bool(cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON)
    mapper = str(cfg.INPUT.DATASET_MAPPER_NAME)
    if pan or "panoptic" in mapper: return "panoptic"
    if ins or "instance" in mapper: return "instance"
    if sem or "semantic" in mapper: return "semantic"
    return "unknown"

def maybe_get_epoch_hint(cfg) -> float:
    try:
        ims_per_batch = int(cfg.SOLVER.IMS_PER_BATCH)
        data_len = int(getattr(cfg, "_DATA_LEN_HINT", 0))
        if ims_per_batch > 0 and data_len > 0:
            iters_per_epoch = max(1, data_len // ims_per_batch)
            return float(iters_per_epoch)
    except Exception:
        pass
    return 0.0

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # evaluator 출력 폴더 통일
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)

        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        if evaluator_type in [
            "coco_panoptic_seg","ade20k_panoptic_seg",
            "cityscapes_panoptic_seg","mapillary_vistas_panoptic_seg"
        ]:
            # 1) 기본 panoptic evaluator (metric 계산 + pred json 생성)
            evaluator_list.append(_LogPanopticEvaluator(dataset_name, output_folder))
            # 2) 예측 저장기 (항상 파일로 남김 + 경로 로그)
            evaluator_list.append(SavePanopticPredictions(dataset_name, cfg.OUTPUT_DIR, verbose_first_n=10))

        if evaluator_type == "cityscapes_instance":
            assert torch.cuda.device_count() > comm.get_rank()
            return CityscapesInstanceEvaluator(dataset_name)

        if evaluator_type == "cityscapes_sem_seg":
            assert torch.cuda.device_count() > comm.get_rank()
            return CityscapesSemSegEvaluator(dataset_name)

        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}"
            )
        return evaluator_list[0] if len(evaluator_list) == 1 else DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
        else:
            mapper = None

        # 라벨 범위 가드 (CUDA assert 방지)
        if mapper is not None:
            mapper = _DebugPanopticGuard(mapper, num_classes=19, ignore_val=255)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = COCOPanopticEvalSafeMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] *= cfg.SOLVER.BACKBONE_MULTIPLIER
                if ("relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED and
                        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and
                        clip_norm_val > 0.0)
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure)
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
            raise NotImplementedError("no optimizer type specified")
        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def train(self):
        logger = logging.getLogger("detectron2")
        logger.info(f"Starting training from iteration {self.start_iter}")
        iters_per_epoch = maybe_get_epoch_hint(self.cfg)
        use_tqdm = comm.is_main_process()
        iterator = range(self.start_iter, self.max_iter)
        if use_tqdm:
            task_name = infer_task_name(self.cfg)
            desc = f"Training [{task_name}]"
            if iters_per_epoch > 0:
                desc += f" ~{int(self.max_iter / iters_per_epoch)} epochs (est)"
            iterator = tqdm(iterator, total=self.max_iter, initial=self.start_iter,
                            desc=desc, dynamic_ncols=True)
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in iterator:
                    self.before_step(); self.run_step(); self.after_step()
                    if use_tqdm:
                        postfix = {}
                        latest = self.storage.latest()
                        for k, v in latest.items():
                            if "loss" in k:
                                try: postfix[k] = float(v[0])
                                except Exception: pass
                        if iters_per_epoch > 0:
                            cur_ep = (self.iter + 1) / iters_per_epoch
                            postfix["epoch(est)"] = f"{cur_ep:.2f}"
                        if postfix: iterator.set_postfix(postfix)
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA", name))
            for name in cfg.DATASETS.TEST
        ]
        with torch.inference_mode():
            res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})

# [!!!] --------- resister.py 코드가 여기로 통합됨 --------- [!!!]
def register_carla_panoptic(cfg):
    """
    setup() 함수 내부에서 호출될 등록 함수.
    import resister가 아닌, 이 함수를 직접 호출합니다.
    """
    # [!!!] 'road' 버그가 해결된 'final_dataset5'을 가리킵니다.
    ROOT = "/media/vip-dell/HC/final_dataset5"

    IMAGE_ROOT = f"{ROOT}/leftImg8bit"
    PAN_TRAIN_ROOT = f"{ROOT}/panoptic_gt_id/train"
    PAN_VAL_ROOT   = f"{ROOT}/panoptic_gt_id/val"
    PAN_TRAIN_JSON = f"{ROOT}/panoptic_json/panoptic_train.json"
    PAN_VAL_JSON   = f"{ROOT}/panoptic_json/panoptic_val.json"
        
    TRAIN_NAME = "carla_final_panoptic_train"   
    VAL_NAME   = "carla_final_panoptic_val"

    LABEL_DIVISOR = 1000
    IGNORE_LABEL   = 255
    
    # --- 로더 함수 (resister.py와 동일) ---
    def _safe_remove(name):
        if hasattr(DatasetCatalog, "remove"):
            try: DatasetCatalog.remove(name)
            except KeyError: pass
        reg = getattr(DatasetCatalog, "_REGISTERED", None) or getattr(DatasetCatalog, "REGISTERED", None)
        if isinstance(reg, dict): reg.pop(name, None)
        meta = getattr(MetadataCatalog, "_NAME_TO_META", None)
        if isinstance(meta, dict): meta.pop(name, None)

    def _to_png_candidates(image_root, rel_or_base, split_hint=None):
        base = os.path.basename(rel_or_base).replace("\\","/")
        base_noext = os.path.splitext(base)[0]
        if not base_noext.endswith("_leftImg8bit"):
            base_noext += "_leftImg8bit"
        png = base_noext + ".png"
        rel = rel_or_base.replace("\\","/")
        cands = [
            os.path.join(ROOT, rel), 
            os.path.join(image_root, rel),
            os.path.join(image_root, split_hint or "", png),
            os.path.join(image_root, "train", png),
            os.path.join(image_root, "val", png),
            os.path.join(image_root, png),
        ]
        out, seen = [], set()
        for c in cands:
            if c and c not in seen:
                out.append(c); seen.add(c)
        return out

    def _pick_panoptic_abs(pan_root, ann_file_name):
        pan_abs = os.path.join(pan_root, ann_file_name)
        if os.path.exists(pan_abs):
            return pan_abs
        base_rel = ann_file_name.replace("\\", "/")
        cands = [os.path.join(pan_root, base_rel)]
        if base_rel.endswith("_panopticGT.png"):
            stem = base_rel[:-len("_panopticGT.png")]
            cands.append(os.path.join(pan_root, stem + "_panopticGT_trainIds19.png"))
            cands.append(os.path.join(pan_root, stem + "_trainIds19.png"))
        for p in cands:
            if os.path.exists(p):
                return p
        return None 

    def _make_loader(pjson, image_root, pan_root, ds_name, split_hint):
        def _loader():
            with open(pjson, "r", encoding="utf-8") as f:
                j = json.load(f)
            # 'road'가 포함된 19개 카테고리 확인
            assert len(j.get("categories",[]))==19, f"{ds_name}: categories != 19. '{ROOT}'의 JSON이 맞는지 확인하세요."
            id2img = {im["id"]: im for im in j["images"]}
            imgid2an = {an["image_id"]: an for an in j["annotations"]}
            dataset, miss_img, miss_pan = [], 0, 0
            cache = {}
            exists = lambda p: cache.setdefault(p, os.path.exists(p))
            for iid, info in id2img.items():
                an = imgid2an.get(iid)
                if an is None: continue
                abs_img = None
                rel = info["file_name"] 
                direct_path = os.path.join(ROOT, rel)
                if exists(direct_path):
                    abs_img = direct_path
                else:
                    for c in _to_png_candidates(image_root, rel, split_hint):
                        if exists(c): abs_img=c; break
                if abs_img is None:
                    miss_img += 1
                    continue
                pan_abs = _pick_panoptic_abs(pan_root, an["file_name"])
                if pan_abs is None:
                    miss_pan += 1
                    continue
                segs = an.get("segments_info", [])
                for s in segs:
                    if "iscrowd" not in s: s["iscrowd"]=0
                H = int(info.get("height",0)); W=int(info.get("width",0))
                dataset.append({
                    "image_id": iid,
                    "file_name": abs_img,
                    "pan_seg_file_name": pan_abs,
                    "segments_info": segs,
                    "height": H, "width": W,
                })
            print(f"[custom-loader] {ds_name}: total={len(dataset)}, missing_image={miss_img}, missing_panoptic_gt_id={miss_pan}")
            return dataset
        return _loader
    
    # --- 클래스 정의 (resister.py와 동일) ---
    thing_classes_final = ["traffic light", "traffic sign", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    stuff_classes_final = ["road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky"]
    
    thing_ids_final = [6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
    stuff_ids_final = [0, 1, 2, 3, 4, 5, 8, 9, 10] # [!!!] 'road'=0 포함
    thing_map_final = {i: i for i in thing_ids_final}
    stuff_map_final = {i: i for i in stuff_ids_final}
    pan_d2c = {**thing_map_final, **stuff_map_final}

    # --- 등록 실행 (resister.py와 동일) ---
    for n in (TRAIN_NAME, VAL_NAME): _safe_remove(n)

    DatasetCatalog.register(TRAIN_NAME, _make_loader(PAN_TRAIN_JSON, IMAGE_ROOT, PAN_TRAIN_ROOT, TRAIN_NAME, "train"))
    mt = MetadataCatalog.get(TRAIN_NAME)
    mt.set(
        image_root=IMAGE_ROOT,
        panoptic_root=PAN_TRAIN_ROOT,
        panoptic_json=PAN_TRAIN_JSON,
        evaluator_type="coco_panoptic_seg",
        ignore_label=IGNORE_LABEL,
        thing_classes=thing_classes_final,
        stuff_classes=stuff_classes_final,
        thing_dataset_id_to_contiguous_id=thing_map_final,
        stuff_dataset_id_to_contiguous_id=stuff_map_final,
        panoptic_dataset_id_to_contiguous_id=pan_d2c,
        panoptic_contiguous_id_to_dataset_id={v:k for k,v in pan_d2c.items()},
        label_divisor=LABEL_DIVISOR,
        panoptic_label_divisor=LABEL_DIVISOR,
    )

    DatasetCatalog.register(VAL_NAME, _make_loader(PAN_VAL_JSON, IMAGE_ROOT, PAN_VAL_ROOT, VAL_NAME, "val"))
    mv = MetadataCatalog.get(VAL_NAME)
    mv.set(
        image_root=IMAGE_ROOT,
        panoptic_root=PAN_VAL_ROOT,
        panoptic_json=PAN_VAL_JSON,
        evaluator_type="coco_panoptic_seg",
        ignore_label=IGNORE_LABEL,
        thing_classes=thing_classes_final,
        stuff_classes=stuff_classes_final,
        thing_dataset_id_to_contiguous_id=thing_map_final,
        stuff_dataset_id_to_contiguous_id=stuff_map_final,
        panoptic_dataset_id_to_contiguous_id=pan_d2c,
        panoptic_contiguous_id_to_dataset_id={v:k for k,v in pan_d2c.items()},
        label_divisor=LABEL_DIVISOR,
        panoptic_label_divisor=LABEL_DIVISOR,
    )

    print(f"[OK] Detectron2에 데이터셋 등록 완료 (ROOT: {ROOT}):")
    print(f"  > Train: {TRAIN_NAME}")
    print(f"  > Val:   {VAL_NAME}")
    
    return TRAIN_NAME, VAL_NAME
# [!!!] --------- 등록 코드 통합 완료 --------- [!!!]


# ===== Setup/Main (수정됨) =====
def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(args.config_file)
    try:
        cfg.merge_from_list(args.opts or [])
    except Exception:
        pass

    # [!!! 수정 !!!] import resister 대신, 등록 함수를 직접 호출
    TRAIN, VAL = register_carla_panoptic(cfg)

    cfg.defrost()
    cfg.DATASETS.TRAIN = (TRAIN,)
    cfg.DATASETS.TEST  = (VAL,)
    cfg.DATALOADER.PIN_MEMORY = True
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_panoptic_lsj"

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 19
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.INPUT.MASK_FORMAT = "bitmask"

    if not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = "/media/vip-dell/HC/train_bh_result/panoptic_unified_v4fixed_run1"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()

    if cfg.MODEL.WEIGHTS and not os.path.isfile(cfg.MODEL.WEIGHTS):
        print(f"[WARN] MODEL.WEIGHTS not found ({cfg.MODEL.WEIGHTS}) → training from scratch.")
        cfg.defrost(); cfg.MODEL.WEIGHTS = ""; cfg.freeze()

    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

    # [!!! 수정 !!!] 검증 로직은 resister.py와 동일하므로 원본 유지
    print("\n[META] final_dataset3 메타데이터 검증 시작...")
    for name in (TRAIN, VAL):
        m = MetadataCatalog.get(name)

        thing_map = getattr(m, "thing_dataset_id_to_contiguous_id", {})
        stuff_map = getattr(m, "stuff_dataset_id_to_contiguous_id", {})
        pan_d2c   = getattr(m, "panoptic_dataset_id_to_contiguous_id", {})

        print(f"[META] {name} panoptic d2c size={len(pan_d2c)} "
              f"(thing_map={len(thing_map)}, stuff_map={len(stuff_map)})")

        # 1) 전체 카테고리 수 19 (resister.py 기준)
        assert len(pan_d2c) == 19, f"[FATAL] {name} d2c size={len(pan_d2c)} != 19"

        # 2) pan_d2c은 0..18 -> 0..18 항등 매핑이어야 함
        expected_ids = set(range(19))
        assert set(pan_d2c.keys()) == expected_ids, \
            f"[FATAL] {name} pan_d2c keys != 0..18 : {sorted(pan_d2c.keys())}"
        assert all(pan_d2c[k] == k for k in expected_ids), \
            f"[FATAL] {name} pan_d2c mapping must be identity 0..18, got: {pan_d2c}"

        # 3) thing_map/stuff_map도 항등 매핑인지 확인 (resister.py 기준)
        expected_thing_ids = {6, 7, 11, 12, 13, 14, 15, 16, 17, 18} # 10개
        expected_stuff_ids = {0, 1, 2, 3, 4, 5, 8, 9, 10} # 9개

        assert set(thing_map.keys()) == expected_thing_ids, \
            f"[FATAL] {name} thing keys != {expected_thing_ids} : {sorted(thing_map.keys())}"
        assert all(thing_map[k] == k for k in expected_thing_ids), \
            f"[FATAL] {name} thing mapping must be identity, got: {thing_map}"

        assert set(stuff_map.keys()) == expected_stuff_ids, \
            f"[FATAL] {name} stuff keys != {expected_stuff_ids} : {sorted(stuff_map.keys())}"
        assert all(stuff_map[k] == k for k in expected_stuff_ids), \
            f"[FATAL] {name} stuff mapping must be identity, got: {stuff_map}"

        # 4) pan_d2c은 thing_map ∪ stuff_map 이어야 함
        merged = dict(thing_map); merged.update(stuff_map)
        assert pan_d2c == merged, f"[FATAL] {name} pan_d2c != (thing_map ∪ stuff_map)"

    print(f"[META] {TRAIN}, {VAL} 메타데이터 검증 완료. (항등 매핑 0-18 확인)\n")

    # GT JSON 카테고리 19 고정 확인 (원본 로직 유지)
    _assert_19cats(TRAIN)
    _assert_19cats(VAL)

    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        with torch.inference_mode():
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
    parser = default_argument_parser()
    import argparse as _argparse
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=_argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )