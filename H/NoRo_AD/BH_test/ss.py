# panoptic GT/Pred json을 읽고 category_id별 등장 여부 체크
import json
from collections import defaultdict

GT_JSON = ""
PRED_JSON = "path/to/panoptic_pred.json"  # evaluator가 만든 pred json

with open(GT_JSON, "r") as f:
    gt = json.load(f)
with open(PRED_JSON, "r") as f:
    pred = json.load(f)

gt_cats = set()
for ann in gt["annotations"]:
    for seg in ann["segments_info"]:
        gt_cats.add(seg["category_id"])

pred_cats = set()
for ann in pred["annotations"]:
    for seg in ann["segments_info"]:
        pred_cats.add(seg["category_id"])

print("GT에 등장한 category:", sorted(gt_cats))
print("Pred에 등장한 category:", sorted(pred_cats))
print("둘 중 하나라도 등장한 category:", sorted(gt_cats | pred_cats))
print("아예 안 나온 category:", [c for c in range(19) if c not in (gt_cats | pred_cats)])
