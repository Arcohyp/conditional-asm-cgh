"""Engineering benchmark: per-image inference latency, peak VRAM, param count,
weight size on disk for the three contender models (A1 baseline / A3 cond / freq).

Single 4K image (2160x3840), one (z, wavelength) operating point, GPU.
Reports warm-loop median/mean latency over N reps after warmup, and peak
CUDA memory allocated for one forward+reconstruction pass.
"""
import sys
import os

import argparse
import json
import time
import numpy as np
import torch

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel
from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp


def build(encoder, param_dim, device):
    if encoder == 'freq':
        return FreqCondCCNNcghModel(n_freqs=8, cond_dim=64, base_res=64, radial_len=64).to(device)
    if encoder == 'baseline':
        return BaselineCCNNcghModelMinExp().to(device)
    if encoder == 'cond':
        return ConditionalCCNNcghModelMinExp(param_dim=param_dim).to(device)
    raise ValueError(encoder)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True, choices=['freq', 'baseline', 'cond'])
    p.add_argument('--weight', default=None)
    p.add_argument('--param-dim', type=int, default=16)
    p.add_argument('--n', type=int, default=2160)
    p.add_argument('--m', type=int, default=3840)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--wavelength', type=float, default=0.000450)
    p.add_argument('--z', type=float, default=170.0)
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--reps', type=int, default=30)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.device_id}')
    model = build(args.encoder, args.param_dim, device)
    if args.weight:
        ck = torch.load(args.weight, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    weight_mb = None
    if args.weight and os.path.exists(args.weight):
        weight_mb = os.path.getsize(args.weight) / 1e6

    Hf = propagation_ASM(torch.empty(1, 1, args.n, args.m), feature_size=[args.pitch, args.pitch],
                         wavelength=args.wavelength, z=args.z, linear_conv=False, return_H=True).to(device)
    Hb = propagation_ASM(torch.empty(1, 1, args.n, args.m), feature_size=[args.pitch, args.pitch],
                         wavelength=args.wavelength, z=-args.z, linear_conv=False, return_H=True).to(device)

    target_amp = torch.rand(1, 1, args.n, args.m, device=device)
    init_phase = torch.zeros(1, 1, args.n, args.m, device=device)

    def one_pass():
        with torch.no_grad():
            holo, _ = model(target_amp, init_phase, args.z, False, args.pitch, args.wavelength, Hf)
            slm = torch.complex(torch.cos(holo), torch.sin(holo))
            recon = propagation_ASM(u_in=slm, z=-args.z, linear_conv=False,
                                    feature_size=[args.pitch, args.pitch],
                                    wavelength=args.wavelength, precomped_H=Hb)
            return torch.abs(recon)

    for _ in range(args.warmup):
        one_pass()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.reps):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_pass()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    times = np.array(times)
    out = {
        'encoder': args.encoder,
        'n_params': int(n_params),
        'weight_mb_fp32_disk': weight_mb,
        'latency_ms_median': float(np.median(times)),
        'latency_ms_mean': float(np.mean(times)),
        'latency_ms_std': float(np.std(times)),
        'latency_ms_min': float(np.min(times)),
        'peak_vram_mb': float(peak_mb),
        'fps': float(1000.0 / np.median(times)),
        'resolution': f'{args.n}x{args.m}',
        'z_mm': args.z,
        'wavelength': args.wavelength,
        'reps': args.reps,
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
