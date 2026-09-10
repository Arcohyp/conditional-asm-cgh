"""Exploratory scan: how do the released SFO weights behave OUTSIDE their
training window z in [85,115] mm?

Reuses the exact pipeline of eval_sfo_psnr_ssim.py (SFO repo, released
weights, SFO physical setup). For each color, scans z from 40 to 200 mm at
5 mm steps, 100 DIV2K validation images per point, feeding the true z both
to the model's distance input and to the ASM propagation.

Metrics per point:
  - skimage intensity-domain PSNR/SSIM with per-image sum scaling
    (SFO's own convention, matches sfo_psnr_ssim_100imgs.json);
  - amplitude-domain PSNR on sqrt-intensity with fixed peak 1.0 and the same
    per-image sum scaling (for qualitative comparison with our model's
    extrapolation scans, which used a fixed-peak amplitude convention).

Usage: python scan_sfo_z.py <gpu-id> <r|g|b>
Output: sfo_zscan_<color>.json
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import torchvision.transforms as transforms

SFO_DIR = '/mnt/sda/chengxirun/hologram_task/eghp_experiment/CGH-SFO-solver'
sys.path.insert(0, SFO_DIR)

from model.FourierNet import Flex_FourierNet  # noqa: E402
from prop import ASM_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Z_LIST = list(range(40, 201, 5))


def main():
    gpu = int(sys.argv[1])
    color = sys.argv[2]
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    res = (2160, 3840)
    pitch = 3.74e-3
    tfm = transforms.Compose([transforms.Resize(res), transforms.ToTensor()])
    valid_dir = '/mnt/sda/datasets/DIV2K/DIV2K_valid_HR'
    files = sorted(os.listdir(valid_dir))[:100]

    color_idx = {'r': 0, 'g': 1, 'b': 2}
    wl_of = {'r': 638, 'g': 520, 'b': 450}
    wl = wl_of[color]

    prop = ASM_split(wavelength=wl, res=res, roi=res, pitch=pitch,
                     apply_constraint=True).to(device).eval()
    model = Flex_FourierNet(wl=wl, center_distance=100).to(device)
    sd = torch.load(os.path.join(SFO_DIR, 'model', color, 'FourierNet_flex_100.pth'),
                    map_location=device)
    model.load_state_dict(sd)

    results = {}
    for dist in Z_LIST:
        psnrs, ssims, apsnrs = [], [], []
        dt = torch.tensor([[float(dist)]], device=device)
        for f in files:
            img = Image.open(os.path.join(valid_dir, f)).convert('RGB')
            t = tfm(img)[color_idx[color]].to(device)
            target = t.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                holo = model(target, dt)
                _, _, final = prop(torch.ones_like(holo), holo, dt)
                scale = target.sum((-2, -1), keepdim=True) / final.sum((-2, -1), keepdim=True)
                final = (final * scale).clamp(0, 1)
            tgt = target.squeeze().cpu().numpy().astype(np.float32)
            rec = final.squeeze().cpu().numpy().astype(np.float32)
            psnrs.append(sk_psnr(tgt, rec, data_range=1.0))
            ssims.append(sk_ssim(tgt, rec, data_range=1.0))
            apsnrs.append(sk_psnr(np.sqrt(tgt), np.sqrt(rec), data_range=1.0))
        key = f'{color}_{wl}nm_z{dist}mm'
        results[key] = {'z': dist, 'wl_nm': wl,
                        'psnr_mean': float(np.mean(psnrs)),
                        'psnr_std': float(np.std(psnrs)),
                        'ssim_mean': float(np.mean(ssims)),
                        'amp_psnr_mean': float(np.mean(apsnrs)),
                        'n': len(files)}
        print(key, {k: round(v, 3) for k, v in results[key].items()
                    if isinstance(v, float)}, flush=True)

    out = os.path.join(HERE, f'sfo_zscan_{color}.json')
    with open(out, 'w') as fp:
        json.dump(results, fp, indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
