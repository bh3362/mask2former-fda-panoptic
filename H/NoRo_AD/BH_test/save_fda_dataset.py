#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import torch


def fda_amplitude_swap(src_img, tgt_img, L=0.05):
    DEVICE = src_img.device
    C, H, W = src_img.shape

    b = int(np.floor(min(H, W) * L))
    if b < 1:
        b = 1

    fft_src = torch.fft.fft2(src_img, dim=(-2,-1))
    fft_tgt = torch.fft.fft2(tgt_img, dim=(-2,-1))

    amp_src = torch.abs(fft_src)
    phase_src = torch.angle(fft_src)
    amp_tgt = torch.abs(fft_tgt)

    amp_src_shift = torch.fft.fftshift(amp_src, dim=(-2,-1))
    amp_tgt_shift = torch.fft.fftshift(amp_tgt, dim=(-2,-1))

    cy, cx = H // 2, W // 2
    mask = torch.zeros((H, W), dtype=torch.bool, device=DEVICE)
    mask[cy-b:cy+b, cx-b:cx+b] = True

    new_amp_shift = amp_src_shift.clone()
    new_amp_shift[:, mask] = amp_tgt_shift[:, mask]

    new_amp = torch.fft.ifftshift(new_amp_shift, dim=(-2,-1))

    real = new_amp * torch.cos(phase_src)
    imag = new_amp * torch.sin(phase_src)
    fft_new = torch.complex(real, imag)

    new_img = torch.fft.ifft2(fft_new, dim=(-2,-1)).real
    new_img = new_img.clamp(0, 255)
    return new_img


CITY_ROOT = "/media/vip-dell/HC/leftImg8bit/train"

def load_cityscapes_images():
    pattern = os.path.join(CITY_ROOT, "*/*.png")
    tgt_paths = sorted(glob.glob(pattern))
    if len(tgt_paths) == 0:
        raise RuntimeError("Cityscapes 이미지가 없습니다.")
    print(f"[CITY] target images: {len(tgt_paths)}")
    return tgt_paths

TGT_IMAGES = load_cityscapes_images()


def process_one(args):
    """
    is_sunny == True  → FDA 적용
    is_sunny == False → 원본 그대로 복사
    """
    src_path, dst_path, L, is_sunny = args

    try:
        src_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if src_bgr is None:
            print(f"[WARN] 읽기 실패: {src_path}")
            return

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # SUNNY 아니면 그냥 복사만
        if not is_sunny:
            cv2.imwrite(dst_path, src_bgr)
            return

        # ---- SUNNY: FDA 적용 ----
        src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = src.shape

        tpath = random.choice(TGT_IMAGES)
        tgt = cv2.imread(tpath, cv2.IMREAD_COLOR)
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
        tgt = cv2.resize(tgt, (W, H))

        src_t = torch.from_numpy(src.transpose(2,0,1)).float()
        tgt_t = torch.from_numpy(tgt.transpose(2,0,1)).float()

        fda_img = fda_amplitude_swap(src_t, tgt_t, L=L)
        fda_np = fda_img.cpu().numpy().transpose(1,2,0).astype(np.uint8)

        fda_bgr = cv2.cvtColor(fda_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, fda_bgr)

    except Exception as e:
        print(f"[ERR] {src_path}: {e}")


def convert_folder(src_root, dst_root, L=0.05, num_workers=8, max_images=None):

    pattern = os.path.join(src_root, "*.png")
    src_paths_all = sorted(glob.glob(pattern))

    args_list = []
    sunny_cnt = 0
    for sp in src_paths_all:
        fname_upper = os.path.basename(sp).upper()
        is_sunny = ("SUNNY" in fname_upper)
        if is_sunny:
            sunny_cnt += 1
        rel = os.path.basename(sp)
        dp = os.path.join(dst_root, rel)
        args_list.append((sp, dp, L, is_sunny))

    # ===== 여기서 최대 개수 제한 =====
    if max_images is not None:
        args_list = args_list[:max_images]
        # SUNNY 개수 다시 계산 (정보 출력용)
        sunny_cnt = sum(1 for _, _, _, is_sunny in args_list if is_sunny)

    print(f"[INFO] total images in {src_root}: {len(src_paths_all)}")
    print(f"[INFO] will process at most {len(args_list)} images")
    print(f"[INFO] SUNNY images (FDA 적용): {sunny_cnt}")
    print(f"[INFO] non-SUNNY images (그대로 복사): {len(args_list) - sunny_cnt}")

    if len(args_list) == 0:
        print(f"[WARN] No images found in {src_root}")
        return

    with mp.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_one, args_list), total=len(args_list)))

    print(f"[DONE] FDA+copy saved to {dst_root}")



if __name__ == "__main__":
    ROOT = "/media/vip-dell/HC/final_dataset5"

    SRC_TRAIN = f"{ROOT}/leftImg8bit/train"
    SRC_VAL   = f"{ROOT}/leftImg8bit/val"

    DST_TRAIN = f"{ROOT}/leftImg8bit_fda222/train"
    DST_VAL   = f"{ROOT}/leftImg8bit_fda222/val"

    # 딱 10장만 테스트 (train/val 각각 10장씩)
    convert_folder(SRC_TRAIN, DST_TRAIN, L=0.05, num_workers=12, max_images=10)
    convert_folder(SRC_VAL,   DST_VAL,   L=0.05, num_workers=12, max_images=10)
