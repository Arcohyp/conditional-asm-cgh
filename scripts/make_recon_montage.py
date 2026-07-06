"""Generate diverse reconstruction assets for the OE paper.

Produces two sets of panels:
  1) Content generalization: one color (green) valley reconstructions across
     several DIV2K validation samples.
  2) Parameter generalization: one fixed sample, three colors, peak and valley.

All reconstructions use the final spatial-conditional model
(step3d_cond_contlambda_peak2) and the same single-point baseline weights
used for the F1 baseline curves.
"""
import sys
import os

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.utils.tools import loadimage
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp

HERE = os.path.dirname(os.path.abspath(__file__))

# color -> (wavelength, peak z, valley z)
CASES = {
    'blue': (0.000450, 150.0, 197.0),
    'green': (0.000532, 150.0, 198.0),
    'red': (0.000638, 150.0, 154.0),
}


def kernels(n, m, pitch, wl, z, device):
    Hf = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                         wavelength=wl, z=z, linear_conv=False, return_H=True).to(device)
    Hb = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                         wavelength=wl, z=-z, linear_conv=False, return_H=True).to(device)
    return Hf, Hb


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    peak = (b.max().item()) ** 2
    return 10 * np.log10(peak / mse) if mse > 0 else 99.0


def reconstruct(model, amp, init_phase, z, pitch, wl, Hf, Hb):
    holo, _ = model(amp, init_phase, z, False, pitch, wl, Hf)
    slm = torch.complex(torch.cos(holo), torch.sin(holo))
    recon = propagation_ASM(u_in=slm, z=-z, linear_conv=False,
                            feature_size=[pitch, pitch], wavelength=wl, precomped_H=Hb)
    return torch.abs(recon)


def to_img(t):
    a = t.squeeze().detach().cpu().numpy()
    return np.clip(a / (a.max() + 1e-8), 0, 1)


def load_target_amp(valid_path, image_index, n, m, device):
    img = loadimage(valid_path, image_index, channel=2, flip=0,
                    m=m, n=n, convert=True, cuda=False)
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    return torch.sqrt(img).view(1, 1, n, m).float().to(device)


def process_one(color, img_index, target_amp, base, cond, init_phase, pitch, device, outdir):
    wl, zp, zv = CASES[color]
    records = {}
    saved = {}
    with torch.no_grad():
        for ztag, z in [('peak_z%g' % zp, zp), ('valley_z%g' % zv, zv)]:
            Hf, Hb = kernels(target_amp.shape[-2], target_amp.shape[-1], pitch, wl, z, device)
            rb = reconstruct(base, target_amp, init_phase, z, pitch, wl, Hf, Hb)
            rc = reconstruct(cond, target_amp, init_phase, z, pitch, wl, Hf, Hb)
            pb = psnr(rb, target_amp)
            pc = psnr(rc, target_amp)
            saved[f'{color}_img{img_index}_baseline_{ztag}'] = to_img(rb)
            saved[f'{color}_img{img_index}_ours_{ztag}'] = to_img(rc)
            records[f'{color}_img{img_index}_baseline_{ztag}'] = pb
            records[f'{color}_img{img_index}_ours_{ztag}'] = pc
    return saved, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-blue', default='',
                   help='single-point specialist weight for blue (450 nm)')
    p.add_argument('--baseline-green', default='',
                   help='single-point specialist weight for green (532 nm)')
    p.add_argument('--baseline-red', default='',
                   help='single-point specialist weight for red (638 nm)')
    p.add_argument('--cond-weight', default=os.path.join(HERE, 'experiments', 'step3d_cond_contlambda_peak2', 'colorcond_best.pth'))
    p.add_argument('--valid-path', default='./data/DIV2K_valid_HR')
    p.add_argument('--content-indices', type=int, nargs='+', default=[801, 810, 820, 830],
                   help='validation image indices for content-generalization montage')
    p.add_argument('--param-index', type=int, default=809,
                   help='validation image index for parameter-generalization montage')
    p.add_argument('--content-color', default='green',
                   help='color used for content-generalization montage')
    p.add_argument('--n', type=int, default=2160)
    p.add_argument('--m', type=int, default=3840)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--device-id', type=int, default=3)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.device_id}')
    cond = ConditionalCCNNcghModelMinExp(param_dim=16).to(device)
    cond.load_state_dict(torch.load(args.cond_weight, map_location=device)['model_state_dict'])
    cond.eval()

    init_phase = torch.zeros(1, 1, args.n, args.m, device=device)
    outdir = os.path.join(HERE, 'paper_assets', 'recon')
    os.makedirs(outdir, exist_ok=True)
    psnr_record = {}

    # Pre-load baseline models
    baseline_paths = {
        'blue': args.baseline_blue,
        'green': args.baseline_green,
        'red': args.baseline_red,
    }
    base_models = {}
    for color, bw in baseline_paths.items():
        if not bw:
            raise ValueError(f'--baseline-{color} is required')
        base = BaselineCCNNcghModelMinExp().to(device)
        base.load_state_dict(torch.load(bw, map_location=device)['model_state_dict'])
        base.eval()
        base_models[color] = base

    # ---- Generate all requested sample x color combinations ----
    print('generating recon assets for indices:', args.content_indices)
    for img_index in args.content_indices:
        target_amp = load_target_amp(args.valid_path, img_index, args.n, args.m, device)
        np.save(os.path.join(outdir, f'img{img_index}_target_amp.npy'), to_img(target_amp))
        for color in CASES.keys():
            saved, records = process_one(color, img_index, target_amp,
                                         base_models[color], cond, init_phase,
                                         args.pitch, device, outdir)
            for k, v in saved.items():
                np.save(os.path.join(outdir, f'{k}.npy'), v)
            psnr_record.update(records)

    # Ensure the parameter-generalization index exists even if not in content-indices
    if args.param_index not in args.content_indices:
        print('parameter generalization index:', args.param_index)
        target_amp = load_target_amp(args.valid_path, args.param_index, args.n, args.m, device)
        np.save(os.path.join(outdir, f'img{args.param_index}_target_amp.npy'), to_img(target_amp))
        for color in CASES.keys():
            saved, records = process_one(color, args.param_index, target_amp,
                                         base_models[color], cond, init_phase,
                                         args.pitch, device, outdir)
            for k, v in saved.items():
                np.save(os.path.join(outdir, f'{k}.npy'), v)
            psnr_record.update(records)

    with open(os.path.join(outdir, 'psnr_record.json'), 'w') as f:
        json.dump(psnr_record, f, indent=2)
    print('psnr_record.json written')


if __name__ == '__main__':
    main()
