#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA raw semantic-ID -> Cityscapes trainId19 lookup tables and COCO panoptic
category list, shared by `make_carla_panoptic_dataset.py`.

CARLA changed its semantic tag IDs between versions ("old" vs "new" tag sets);
`map_to_train19(..., strategy="auto")` auto-detects which one a given label
map uses and applies the matching LUT. If the input is already 0-18/255
(trainId19), it is passed through untouched.
"""

import numpy as np

# Cityscapes-19 "thing" trainIds (rest are "stuff")
THING_TRAINIDS = {6, 7, 11, 12, 13, 14, 15, 16, 17, 18}

# CARLA 0.9.15 "new" semantic tag set -> Cityscapes trainId19
CARLA_NEW_TO_TRAIN19 = {
    1: 0, 24: 0,   # Road, RoadLine
    2: 1,          # Sidewalk
    3: 2,          # Building
    4: 3,          # Wall
    5: 4, 28: 4,   # Fence, Guard Rail
    6: 5,          # Pole
    7: 6,          # TrafficLight
    8: 7,          # TrafficSign
    9: 8,          # Vegetation
    10: 9, 25: 9,  # Terrain, Ground
    11: 10,        # Sky
    12: 11,        # Pedestrian
    13: 12,        # Rider
    14: 13,        # Car
    15: 14,        # Truck
    16: 15,        # Bus
    17: 16,        # Train
    18: 17,        # Motorcycle
    19: 18,        # Bicycle
    # everything else (0, 20-23, 26, 27, ...) falls through to 255 (ignore)
}

# CARLA "old" semantic tag set -> Cityscapes trainId19
CARLA_OLD_TO_TRAIN19 = {
    7: 0, 6: 0, 8: 1, 1: 2, 11: 3, 2: 4,
    5: 5, 18: 6, 12: 7, 9: 8, 22: 9, 13: 10,
    4: 11, 10: 13,
}

CATEGORIES = [
    {"id": 0, "name": "road", "isthing": 0},
    {"id": 1, "name": "sidewalk", "isthing": 0},
    {"id": 2, "name": "building", "isthing": 0},
    {"id": 3, "name": "wall", "isthing": 0},
    {"id": 4, "name": "fence", "isthing": 0},
    {"id": 5, "name": "pole", "isthing": 0},
    {"id": 6, "name": "traffic light", "isthing": 1},
    {"id": 7, "name": "traffic sign", "isthing": 1},
    {"id": 8, "name": "vegetation", "isthing": 0},
    {"id": 9, "name": "terrain", "isthing": 0},
    {"id": 10, "name": "sky", "isthing": 0},
    {"id": 11, "name": "person", "isthing": 1},
    {"id": 12, "name": "rider", "isthing": 1},
    {"id": 13, "name": "car", "isthing": 1},
    {"id": 14, "name": "truck", "isthing": 1},
    {"id": 15, "name": "bus", "isthing": 1},
    {"id": 16, "name": "train", "isthing": 1},
    {"id": 17, "name": "motorcycle", "isthing": 1},
    {"id": 18, "name": "bicycle", "isthing": 1},
]

TRAIN19_SET = set(range(19))
VALID_TRAIN19 = TRAIN19_SET | {255}


def is_already_train19(arr: np.ndarray) -> bool:
    uniq = set(np.unique(arr).tolist())
    return uniq.issubset(VALID_TRAIN19)


def build_lut(mapping: dict, default_val: int = 255) -> np.ndarray:
    lut = np.full(65536, default_val, dtype=np.uint16)
    for src, dst in mapping.items():
        lut[int(src)] = int(dst)
    return lut


LUT_NEW = build_lut(CARLA_NEW_TO_TRAIN19, default_val=255)
LUT_OLD = build_lut(CARLA_OLD_TO_TRAIN19, default_val=255)


def map_to_train19(arr: np.ndarray, strategy: str = "auto") -> np.ndarray:
    """arr: raw semantic label map (integer). strategy: 'auto' | 'new' | 'old'."""
    arr = arr.astype(np.uint16, copy=False)
    if strategy == "auto":
        if is_already_train19(arr):
            return arr
        uniq = set(np.unique(arr).tolist())
        # heuristic: these IDs only appear in the "new" tag set
        if 24 in uniq or 19 in uniq or 17 in uniq:
            return LUT_NEW[arr]
        return LUT_OLD[arr]
    elif strategy == "new":
        return LUT_NEW[arr]
    elif strategy == "old":
        return LUT_OLD[arr]
    raise ValueError("strategy must be one of {'auto', 'new', 'old'}")
