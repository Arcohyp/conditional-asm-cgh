"""C1: 8-bit SLM holophase quantization robustness (reviewer-mandatory caveat).

Real phase-only SLMs address phase on a finite grid (commonly 8-bit = 256 levels
over a 2*pi stroke). Our model emits a CONTINUOUS holophase; this script wraps it
to [0, 2*pi) and uniformly quantizes to L = 2^bits levels right before the SLM
complex exponential (slm = exp(i*phi_q)), then propagates with the SAME ASM and
measures the PSNR drop vs the continuous-phase reconstruction.

Uses the existing discrete-lambda NPCC weight (no retrain). Reports, per color and
per z, continuous PSNR and the PSNR at {8,6,4}-bit so we can state "8-bit costs
X dB" honestly. PSNR convention matches stage1_eval.py (peak = target_amp.max()^2).
"""
import sys
import os

import argparse
import json
import math
import numpy as np
import torch
from tqdm import tqdm

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.utils.tools import loadimage
from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel
from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp


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


def quantize_phase(phi, bits):
    """Wrap to [0, 2*pi) then uniformly quantize to 2^bits levels (SLM model)."""
    if bits is None:
        return phi
    L = 2 ** bits
    two_pi = 2.0 * math.pi
    wrapped = torch.remainder(phi, two_pi)
    q = torch.round(wrapped / two_pi * L)
    q = torch.remainder(q, L)            # level L wraps back to 0
    return q / L * two_pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--valid-path', default='./data/DIV2K_valid_HR')
    p.add_argument('--n', type=int, default=2160)
    p.add_argument('--m', type=int, default=3840)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--bits', type=int, nargs='+', default=[8, 6, 4])
    p.add_argument('--n-freqs', type=int, default=16)
    p.add_argument('--cond-dim', type=int, default=64)
    p.add_argument('--base-res', type=int, default=128)
    p.add_argument('--encoder', type=str, default='freq', choices=['freq', 'baseline', 'cond'])
    p.add_argument('--param-dim', type=int, default=16)
    args = p.parse_args()

    # (color, wavelength, [z points: design peak + a valley])
    cases = [
        ('blue', 0.000450, [150.0, 179.0]),
        ('green', 0.000532, [150.0, 165.0]),
        ('red', 0.000638, [150.0, 178.0]),
    ]
    bit_settings = [None] + args.bits   # None = continuous reference

    device = torch.device(f'cuda:{args.device_id}')
    if args.encoder == 'freq':
        model = FreqCondCCNNcghModel(n_freqs=args.n_freqs, cond_dim=args.cond_dim,
                                     base_res=args.base_res, radial_len=args.base_res).to(device)
    elif args.encoder == 'baseline':
        model = BaselineCCNNcghModelMinExp().to(device)
    elif args.encoder == 'cond':
        model = ConditionalCCNNcghModelMinExp(param_dim=args.param_dim).to(device)
    else:
        raise ValueError(f"unknown encoder {args.encoder}")
    ck = torch.load(args.weight, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    init_phase = torch.zeros(1, 1, args.n, args.m, device=device)
    results = {}
    for color, wl, zs in cases:
        for z in zs:
            Hf, Hb = compute_asm_kernels(args.n, args.m, args.pitch, wl, z, device)
            ps = {b: [] for b in bit_settings}
            for idx in tqdm(range(args.num_samples), leave=False, desc=f'{color} z{z}'):
                img = loadimage(args.valid_path, 801 + idx, channel=2, flip=0,
                                m=args.m, n=args.n, convert=True, cuda=False)
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img)
                target_amp = torch.sqrt(img).view(1, 1, args.n, args.m).float().to(device)
                with torch.no_grad():
                    holo, _ = model(target_amp, init_phase, z, False, args.pitch, wl, Hf)
                    for b in bit_settings:
                        phi_q = quantize_phase(holo, b)
                        slm = torch.complex(torch.cos(phi_q), torch.sin(phi_q))
                        recon = propagation_ASM(u_in=slm, z=-z, linear_conv=False,
                                                feature_size=[args.pitch, args.pitch],
                                                wavelength=wl, precomped_H=Hb)
                        ps[b].append(psnr(torch.abs(recon), target_amp))
            key = f'{color}_z{z:.1f}'
            row = {}
            cont = float(np.mean(ps[None]))
            for b in bit_settings:
                label = 'cont' if b is None else f'{b}bit'
                m_ = float(np.mean(ps[b]))
                row[label] = {'psnr_mean': m_, 'psnr_std': float(np.std(ps[b])),
                              'drop_vs_cont': float(cont - m_)}
            results[key] = {'wavelength': wl, 'z': z, **row}
            drops = ', '.join(f"{b}bit {row[f'{b}bit']['psnr_mean']:.2f}(-{row[f'{b}bit']['drop_vs_cont']:.2f})"
                              for b in args.bits)
            print(f"{key}: cont {cont:.2f} | {drops}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'c1_quant_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.output_dir}/c1_quant_results.json")


if __name__ == '__main__':
    main()
