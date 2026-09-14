"""N2: regenerate the dense 2-D (z, lambda) PSNR map of the green single-point
baseline around the design point (z=150 mm, lambda=532 nm), for the redrawn
Fig. 1(d). Protocol mirrors Fig. 1(b,c): amplitude-domain PSNR, DIV2K valid.

Grid: z in [145,155] mm @0.5 mm, lambda in [527,537] nm @0.5 nm (21x21=441 pts),
20 images (0801-0820). Output: experiments/rev_n2_grid2d_green/eval_results.json

Usage: python remap_2d.py <gpu-id>
"""
import json
import os
import sys

sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/hologram_task')
sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp')

import numpy as np
import torch
from tqdm import tqdm

from stage1_eval import compute_asm_kernels  # noqa: E402
from tools import loadimage  # noqa: E402
from exp_models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp  # noqa: E402


def psnr_fixed_peak(pred, target, max_val=1.0):
    """Same convention as the original Fig. 1(b,c) data (eval_baseline.py):
    amplitude domain, fixed peak = 1.0. stage1_eval.psnr uses per-image
    peak=(max target)^2 and reads ~10 dB lower -- do NOT use it here."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

WEIGHT = ('/mnt/sda/chengxirun/hologram_task/hologram_task/experiments/'
          'ccnncgh_3840x2160_532nm_with_linear/ccnncgh/ccnncgh_best.pth')
# ^ The ORIGINAL paper single-point specialist (4K, 532 nm, z=150), same weight
# behind fig1(b,c) (diag_green_comb150, 34.30 dB @ 150/532 with 10 imgs,
# verified 2026-09-09) and fig4's sfoconv_baseline_green. Do NOT substitute the
# baseline_generalization_exp 1080p weights -- they read ~12 dB lower at 4K.
OUT = './experiments/rev_n2_grid2d_green'
RES = (2160, 3840)
PITCH = 0.0036
Z_GRID = np.arange(145.0, 155.01, 0.5)
WL_GRID = 1e-6 * np.arange(527.0, 537.01, 0.5)
NUM_IMG = 20
VALID = '/mnt/sda/datasets/DIV2K/DIV2K_valid_HR'


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    device = torch.device(f'cuda:{gpu}')
    model = BaselineCCNNcghModelMinExp().to(device)
    ck = torch.load(WEIGHT, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    init_phase = torch.zeros(1, 1, *RES, device=device)

    targets = []
    for idx in range(NUM_IMG):
        img = loadimage(VALID, 801 + idx, channel=2, flip=0,
                        m=RES[1], n=RES[0], convert=True, cuda=False)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        targets.append(torch.sqrt(img).view(1, 1, *RES).float().to(device))

    results = {}
    for wl in WL_GRID:
        for z in Z_GRID:
            Hf, Hb = compute_asm_kernels(RES[0], RES[1], PITCH, wl, z, device)
            ps = []
            for target_amp in targets:
                with torch.no_grad():
                    holo, _ = model(target_amp, init_phase, z, False, PITCH, wl, Hf)
                    slm = torch.complex(torch.cos(holo), torch.sin(holo))
                    from propagation_ASM import propagation_ASM
                    recon = propagation_ASM(u_in=slm, z=-z, linear_conv=False,
                                            feature_size=[PITCH, PITCH],
                                            wavelength=wl, precomped_H=Hb)
                    recon_amp = torch.abs(recon)
                ps.append(psnr_fixed_peak(recon_amp, target_amp))
            key = f'z{z:.1f}_wl{wl*1e6:.2f}nm'
            results[key] = {'z': z, 'wavelength': wl, 'psnr_mean': float(np.mean(ps)),
                            'psnr_std': float(np.std(ps))}
        print(f'wl={wl*1e6:.0f}nm done, last z={z}: {np.mean(ps):.2f} dB', flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('saved', OUT)


if __name__ == '__main__':
    main()
