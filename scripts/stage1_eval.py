"""Stage 1 eval: Stage1 z-residual vs blue-450 baseline, fine z grid (both-move).

Evaluates the trained FreqCondCCNNcghModel over a 1mm z grid covering comb peaks
and valleys, 100 samples, lambda=450nm. Writes per-z PSNR JSON. A companion
baseline both-move JSON (eval_baseline.py) at the same z grid is the reference.
"""
import sys
import os

import argparse
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.utils.tools import loadimage
from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel
from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_3expert_minexp import ThreeExpertCCNNcghModelMinExp


def compute_asm_kernels(n, m, pitch, wl, z, device):
    Hf = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                         wavelength=wl, z=z, linear_conv=False, return_H=True).to(device)
    Hb = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                         wavelength=wl, z=-z, linear_conv=False, return_H=True).to(device)
    return Hf, Hb


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    peak = (b.max().item()) ** 2
    return 10 * np.log10(peak / mse) if mse > 0 else 99.0


def sfo_psnr(recon_amp, target_amp):
    # SFO convention: intensity domain (I=A^2) + per-image sum-scale + skimage
    # linear PSNR with data_range=1.0 (== 10*log10(1/mse) on clamped intensities).
    I_t = target_amp ** 2
    I_r = recon_amp ** 2
    scale = I_t.sum() / I_r.sum()
    I_r = (I_r * scale).clamp(0, 1)
    mse = torch.mean((I_t - I_r) ** 2).item()
    return 10 * np.log10(1.0 / mse) if mse > 0 else 99.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--valid-path', default='./data/DIV2K_valid_HR')
    p.add_argument('--n', type=int, default=2160)
    p.add_argument('--m', type=int, default=3840)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--wavelength', type=float, default=0.000450)
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--z-list', type=float, nargs='+', required=True)
    p.add_argument('--n-freqs', type=int, default=8)
    p.add_argument('--cond-dim', type=int, default=64)
    p.add_argument('--base-res', type=int, default=64)
    p.add_argument('--encoder', type=str, default='freq', choices=['freq', 'baseline', 'cond', '3expert'])
    p.add_argument('--expert-boundaries', type=float, nargs=2, default=(0.000495, 0.000585))
    p.add_argument('--expert-soft-width', type=float, default=5e-6)
    p.add_argument('--param-dim', type=int, default=16)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.device_id}')
    if args.encoder == 'freq':
        model = FreqCondCCNNcghModel(n_freqs=args.n_freqs, cond_dim=args.cond_dim,
                                     base_res=args.base_res, radial_len=args.base_res).to(device)
    elif args.encoder == 'baseline':
        model = BaselineCCNNcghModelMinExp().to(device)
    elif args.encoder == 'cond':
        model = ConditionalCCNNcghModelMinExp(param_dim=args.param_dim).to(device)
    elif args.encoder == '3expert':
        model = ThreeExpertCCNNcghModelMinExp(param_dim=args.param_dim,
                                              boundaries=args.expert_boundaries,
                                              soft_width=args.expert_soft_width).to(device)
    else:
        raise ValueError(f"unknown encoder {args.encoder}")
    ck = torch.load(args.weight, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    init_phase = torch.zeros(1, 1, args.n, args.m, device=device)
    crit = nn.MSELoss()
    results = {}
    for z in args.z_list:
        Hf, Hb = compute_asm_kernels(args.n, args.m, args.pitch, args.wavelength, z, device)
        ps, sps, ls = [], [], []
        for idx in tqdm(range(args.num_samples), leave=False, desc=f'z{z}'):
            img = loadimage(args.valid_path, 801 + idx, channel=2, flip=0,
                            m=args.m, n=args.n, convert=True, cuda=False)
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            target_amp = torch.sqrt(img).view(1, 1, args.n, args.m).float().to(device)
            with torch.no_grad():
                holo, _ = model(target_amp, init_phase, z, False, args.pitch, args.wavelength, Hf)
                slm = torch.complex(torch.cos(holo), torch.sin(holo))
                recon = propagation_ASM(u_in=slm, z=-z, linear_conv=False,
                                        feature_size=[args.pitch, args.pitch],
                                        wavelength=args.wavelength, precomped_H=Hb)
                recon_amp = torch.abs(recon)
                ls.append(crit(recon_amp, target_amp).item())
            ps.append(psnr(recon_amp, target_amp))
            sps.append(sfo_psnr(recon_amp, target_amp))
        results[f'z{z:.1f}'] = {'psnr_mean': float(np.mean(ps)), 'psnr_std': float(np.std(ps)),
                                'sfo_psnr_mean': float(np.mean(sps)), 'sfo_psnr_std': float(np.std(sps)),
                                'loss_mean': float(np.mean(ls)), 'z': z, 'wavelength': args.wavelength}
        print(f"z={z}: PSNR {np.mean(ps):.2f} ± {np.std(ps):.2f} dB | "
              f"SFO-PSNR {np.mean(sps):.2f} ± {np.std(sps):.2f} dB")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, 'eval_results.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['z_mm', 'psnr_mean', 'psnr_std', 'sfo_psnr_mean', 'sfo_psnr_std', 'loss_mean'])
        for v in results.values():
            w.writerow([v['z'], v['psnr_mean'], v['psnr_std'], v['sfo_psnr_mean'], v['sfo_psnr_std'], v['loss_mean']])


if __name__ == '__main__':
    main()
