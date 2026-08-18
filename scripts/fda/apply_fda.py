#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Domain Adaptation (FDA) — batch dataset conversion.

RECONSTRUCTED SCRIPT. The core algorithm (`fda_amplitude_swap`) below is kept
verbatim from the original experiment script (`H/NoRo_AD/BH_test/save_fda_dataset.py`)
and correctly implements the paper's Eq. 2.1-2.4 (Yang & Soatto, 2020): source
phase is preserved, and only the low-frequency amplitude within a radius-beta
circle around the zero frequency is swapped for the target (Cityscapes)
amplitude. What did NOT survive is the one-off invocation that ran this over
the full 3,500-image CARLA training split (the original script's `__main__`
block only ever processed a 10-image test batch with a different beta and a
different output folder name than the one the trained models actually used).
This version generalizes that logic into a proper CLI so the full run can be
reproduced: point `--src-root` at a CARLA `leftImg8bit` split, `--cityscapes-root`
at real Cityscapes training images to sample style targets from, and it writes
FDA-styled RGB images to `--dst-root` at whatever `--beta` the thesis run used
(reported as beta=0.002 in the final write-up; beta in {0.01, 0.05} were also
tried per the thesis text). Panoptic/semantic labels are untouched by design —
FDA only transforms pixel color statistics, never the label maps.

Usage:
    python apply_fda.py \
        --src-root /path/to/final_dataset5/leftImg8bit/train \
        --cityscapes-root /path/to/cityscapes/leftImg8bit/train \
        --dst-root /path/to/final_dataset5/leftImg8bit_fda/train \
        --beta 0.002 --num-workers 12
"""

import os
import cv2
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import torch


def fda_amplitude_swap(src_img: torch.Tensor, tgt_img: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    """Swap the low-frequency amplitude of src_img for tgt_img's, keep src_img's phase.

    src_img, tgt_img: (C, H, W) float tensors, pixel range [0, 255].
    beta: fraction of the spectrum's shorter side treated as "low frequency" (paper's beta, Eq. 2.2).
    """
    device = src_img.device
    _, H, W = src_img.shape

    b = int(np.floor(min(H, W) * beta))
    b = max(b, 1)

    fft_src = torch.fft.fft2(src_img, dim=(-2, -1))
    fft_tgt = torch.fft.fft2(tgt_img, dim=(-2, -1))

    amp_src = torch.abs(fft_src)
    phase_src = torch.angle(fft_src)
    amp_tgt = torch.abs(fft_tgt)

    amp_src_shift = torch.fft.fftshift(amp_src, dim=(-2, -1))
    amp_tgt_shift = torch.fft.fftshift(amp_tgt, dim=(-2, -1))

    cy, cx = H // 2, W // 2
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    mask[cy - b:cy + b, cx - b:cx + b] = True

    new_amp_shift = amp_src_shift.clone()
    new_amp_shift[:, mask] = amp_tgt_shift[:, mask]
    new_amp = torch.fft.ifftshift(new_amp_shift, dim=(-2, -1))

    real = new_amp * torch.cos(phase_src)
    imag = new_amp * torch.sin(phase_src)
    fft_new = torch.complex(real, imag)

    new_img = torch.fft.ifft2(fft_new, dim=(-2, -1)).real
    return new_img.clamp(0, 255)


def _load_target_pool(cityscapes_root: str) -> list:
    pattern = os.path.join(cityscapes_root, "*", "*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        # fall back to a flat (non city-subfolder) layout
        paths = sorted(glob.glob(os.path.join(cityscapes_root, "*.png")))
    if not paths:
        raise RuntimeError(f"No Cityscapes target images found under {cityscapes_root}")
    print(f"[FDA] target style pool: {len(paths)} Cityscapes images")
    return paths


_TARGET_POOL = None  # populated per-worker via pool initializer


def _worker_init(cityscapes_root: str):
    global _TARGET_POOL
    _TARGET_POOL = _load_target_pool(cityscapes_root)


def _process_one(args):
    src_path, dst_path, beta = args
    try:
        src_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if src_bgr is None:
            print(f"[WARN] failed to read: {src_path}")
            return
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = src.shape

        tgt_path = random.choice(_TARGET_POOL)
        tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
        tgt = cv2.resize(tgt, (W, H))

        src_t = torch.from_numpy(src.transpose(2, 0, 1)).float()
        tgt_t = torch.from_numpy(tgt.transpose(2, 0, 1)).float()

        fda_img = fda_amplitude_swap(src_t, tgt_t, beta=beta)
        fda_np = fda_img.numpy().transpose(1, 2, 0).astype(np.uint8)

        cv2.imwrite(dst_path, cv2.cvtColor(fda_np, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[ERR] {src_path}: {e}")


def convert_folder(src_root: str, dst_root: str, cityscapes_root: str, beta: float, num_workers: int):
    src_paths = sorted(glob.glob(os.path.join(src_root, "*.png")))
    if not src_paths:
        print(f"[WARN] no PNGs found in {src_root}")
        return

    args_list = [(sp, os.path.join(dst_root, os.path.basename(sp)), beta) for sp in src_paths]
    print(f"[FDA] {src_root}: {len(args_list)} images -> {dst_root} (beta={beta})")

    with mp.Pool(processes=num_workers, initializer=_worker_init, initargs=(cityscapes_root,)) as pool:
        list(tqdm(pool.imap(_process_one, args_list), total=len(args_list)))

    print(f"[DONE] {dst_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", required=True, help="CARLA leftImg8bit split dir (flat *.png files), e.g. .../leftImg8bit/train")
    ap.add_argument("--cityscapes-root", required=True, help="Real Cityscapes leftImg8bit/train dir to sample FDA style targets from")
    ap.add_argument("--dst-root", required=True, help="Output dir for FDA-styled images")
    ap.add_argument("--beta", type=float, default=0.002, help="Low-frequency swap radius fraction (paper's beta; thesis final value 0.002)")
    ap.add_argument("--num-workers", type=int, default=12)
    args = ap.parse_args()

    convert_folder(args.src_root, args.dst_root, args.cityscapes_root, args.beta, args.num_workers)


if __name__ == "__main__":
    main()
