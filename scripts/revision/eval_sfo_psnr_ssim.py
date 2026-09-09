"""SFO-side PSNR+SSIM for the M2 fair comparison.

Replicates eghp_experiment/CGH-SFO-solver/validate_100imgs.py exactly
(SFO released weights, SFO physical setup, intensity domain, per-image
sum-scale, skimage metrics) and adds SSIM, saving a JSON summary.
Read-only usage of the SFO repo.

Usage: python eval_sfo_psnr_ssim.py <gpu-id>
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


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = torch.device(f'cuda:{gpu}')
    res = (2160, 3840)
    pitch = 3.74e-3
    tfm = transforms.Compose([transforms.Resize(res), transforms.ToTensor()])
    valid_dir = '/mnt/sda/datasets/DIV2K/DIV2K_valid_HR'
    files = sorted(os.listdir(valid_dir))[:100]

    color_idx = {'r': 0, 'g': 1, 'b': 2}
    wl_of = {'r': 638, 'g': 520, 'b': 450}
    results = {}
    for color in ['r', 'g', 'b']:
        wl = wl_of[color]
        prop = ASM_split(wavelength=wl, res=res, roi=res, pitch=pitch,
                         apply_constraint=True).to(device).eval()
        model = Flex_FourierNet(wl=wl, center_distance=100).to(device)
        sd = torch.load(os.path.join(SFO_DIR, 'model', color, 'FourierNet_flex_100.pth'),
                        map_location=device)
        model.load_state_dict(sd)
        for dist in [85, 100, 115]:
            psnrs, ssims = [], []
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
            key = f'{color}_{wl}nm_z{dist}mm'
            results[key] = {'psnr_mean': float(np.mean(psnrs)),
                            'psnr_std': float(np.std(psnrs)),
                            'ssim_mean': float(np.mean(ssims)),
                            'n': 100}
            print(key, {k: round(v, 3) for k, v in results[key].items() if k != 'n'})
        del model, prop
        torch.cuda.empty_cache()

    with open(os.path.join(HERE, 'sfo_psnr_ssim_100imgs.json'), 'w') as fp:
        json.dump(results, fp, indent=2)
    gm = np.mean([v['psnr_mean'] for v in results.values()])
    print(f'SFO GRAND MEAN PSNR: {gm:.2f} dB')


if __name__ == '__main__':
    main()
