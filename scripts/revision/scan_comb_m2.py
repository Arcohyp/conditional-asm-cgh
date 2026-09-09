"""M2 anchor-bug fix, step 1: measure the TRUE comb of the single-point base
(step A, z=100mm, pitch 3.74um) so the retrained step B can anchor on peaks
that actually lie inside [85,115]mm (the hardcoded PEAKS table in
train_colorcond.py is for z=150 / pitch 3.6um and put 40% of M2's anchor
samples at z>=150, i.e. outside the training window).

Scans (20 DIV2K val imgs, amplitude-domain PSNR):
  - z-comb:  z in [85,115]mm @1mm for lambda in {450,520,638}nm
  - lambda-comb at z=100: bands around each native line, 1nm steps
Outputs JSON to ./experiments/rev_m2_combscan_z100/*.json

Usage: python scan_comb_m2.py <gpu-id>
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/hologram_task')
sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp')

from stage1_train import compute_asm_kernels, reconstruct  # noqa: E402
from exp_models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp  # noqa: E402

EXP = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp'
WEIGHT = os.path.join(EXP, 'experiments/rev_m2_base_z100_p374/conditional_ccnncgh_best.pth')
OUT = os.path.join(EXP, 'experiments/rev_m2_combscan_z100')
RES = (2160, 3840)
PITCH = 3.74e-3
DESIGN_Z = 100.0
NUM = 20
Z_SCAN = [float(z) for z in range(85, 116)]
WL_SCAN = {
    0.000450: [float(w) for w in range(440, 461)],
    0.000520: [float(w) for w in range(510, 531)],
    0.000638: [float(w) for w in range(628, 649)],
}


def psnr_amp(rec, tgt):
    peak = (tgt.max() ** 2).item()
    mse = torch.mean((rec - tgt) ** 2).item()
    return 10 * np.log10(peak / max(mse, 1e-12))


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    os.makedirs(OUT, exist_ok=True)
    device = torch.device(f'cuda:{gpu}')
    model = ConditionalCCNNcghModelMinExp(param_dim=16).to(device)
    model.param_encoder.z_center = DESIGN_Z
    sd = torch.load(WEIGHT, map_location=device)
    model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    model.eval()

    valid_dir = '/mnt/sda/datasets/DIV2K/DIV2K_valid_HR'
    files = sorted(os.listdir(valid_dir))[:NUM]
    tfm = T.Compose([T.Resize(RES), T.ToTensor()])
    init_phase = torch.zeros(1, 1, *RES, device=device)

    def run_grid(wl_mm, zs):
        cache = {}
        rows = {}
        for z in zs:
            if z not in cache:
                cache[z] = compute_asm_kernels(RES[0], RES[1], PITCH, wl_mm, z, device)
            Hf, Hb = cache[z]
            vals = []
            for f in files:
                img = Image.open(os.path.join(valid_dir, f)).convert('L')
                t = tfm(img).to(device)
                amp = t.sqrt().view(1, 1, *RES)
                with torch.no_grad():
                    rec = reconstruct(model, amp, init_phase, z, PITCH, wl_mm, Hf, Hb)
                vals.append(psnr_amp(rec, amp))
            rows[str(z)] = float(np.mean(vals))
            print(f'wl={wl_mm*1e6:.0f}nm z={z} PSNR={rows[str(z)]:.2f}', flush=True)
        return rows

    for wl in WL_SCAN:
        rows = run_grid(wl, Z_SCAN)
        tag = f'{wl*1e6:.0f}'
        with open(os.path.join(OUT, f'zscan_{tag}.json'), 'w') as f:
            json.dump(rows, f, indent=1)
        # lambda scan: fixed z=100, vary wavelength
        vals = {}
        for w_nm in WL_SCAN[wl]:
            w_mm = w_nm * 1e-3
            Hf, Hb = compute_asm_kernels(RES[0], RES[1], PITCH, w_mm, DESIGN_Z, device)
            acc = []
            for f in files:
                img = Image.open(os.path.join(valid_dir, f)).convert('L')
                t = tfm(img).to(device)
                amp = t.sqrt().view(1, 1, *RES)
                with torch.no_grad():
                    rec = reconstruct(model, amp, init_phase, DESIGN_Z, PITCH, w_mm, Hf, Hb)
                acc.append(psnr_amp(rec, amp))
            vals[str(w_nm)] = float(np.mean(acc))
            print(f'wl-scan {w_nm}nm @z=100 PSNR={vals[str(w_nm)]:.2f}', flush=True)
        with open(os.path.join(OUT, f'wlscan_{tag}.json'), 'w') as f:
            json.dump(vals, f, indent=1)
    print('ALL DONE')


if __name__ == '__main__':
    main()
