"""N4 (Reviewer figure comment): composited full-color RGB reconstruction at
off-peak distances. Runs a spatial-conditional weight on the three color
channels (450/532/638 nm) and composites. Reports per-channel PSNR in SFO's
intensity-domain convention (per-image sum scaling).

Two configurations are used in the paper:
  main  -- the paper's spatial-conditional model (M3 seed 42, z in [145,205],
           pitch 3.6 um), rendered at z=198 mm (deep off-peak valley).
  m2    -- the SFO-window model (pitch 3.74 um, design z=100), at z=113 mm.

Usage: python make_rgb_composite.py <gpu-id>
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import torchvision.transforms as T

sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/hologram_task')
sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp')

from stage1_train import compute_asm_kernels, reconstruct  # noqa: E402
from exp_models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = (2160, 3840)
IMG = '/mnt/sda/datasets/DIV2K/DIV2K_valid_HR/0801.png'
CHANNELS = [(0.000450, 'blue', 2), (0.000532, 'green', 1), (0.000638, 'red', 0)]

CONFIGS = {
    'main': dict(
        weight='/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp/'
               'experiments/rev_m3_spatial_s42/colorcond_best.pth',
        pitch=0.0036, z_center=150.0, z=198.0),
    'm2': dict(
        weight='/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp/'
               'experiments/rev_m2_spatialcond_sfo_window/colorcond_best.pth',
        pitch=3.74e-3, z_center=100.0, z=113.0),
}


def run(tag, gpu):
    cfg = CONFIGS[tag]
    device = torch.device(f'cuda:{gpu}')
    model = ConditionalCCNNcghModelMinExp(param_dim=16).to(device)
    model.param_encoder.z_center = cfg['z_center']
    sd = torch.load(cfg['weight'], map_location=device)
    model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    model.eval()
    pitch, z = cfg['pitch'], cfg['z']

    img = Image.open(IMG).convert('RGB')
    t = T.Compose([T.Resize(RES), T.ToTensor()])(img).to(device)  # [3,H,W] gamma
    init_phase = torch.zeros(1, 1, *RES, device=device)

    recon_rgb = torch.zeros(3, *RES, device=device)
    psnrs = {}
    with torch.no_grad():
        for wl, name, ch in CHANNELS:
            amp_in = t[ch:ch + 1].sqrt().view(1, 1, *RES)
            Hf, Hb = compute_asm_kernels(RES[0], RES[1], pitch, wl, z, device)
            rec_amp = reconstruct(model, amp_in, init_phase, z, pitch, wl, Hf, Hb)
            I = (rec_amp ** 2)
            scale = t[ch].sum() / I.sum()
            I_scaled = (I.view(-1) * scale).clamp(0, 1).view(RES)
            recon_rgb[ch] = I_scaled
            psnrs[name] = sk_psnr(t[ch].cpu().numpy().astype(np.float32),
                                  I_scaled.cpu().numpy().astype(np.float32),
                                  data_range=1.0)
            print(f'[{tag}] {name} ({wl*1e6:.0f} nm): PSNR {psnrs[name]:.2f} dB',
                  flush=True)

    small = (540, 960)
    comp = Image.fromarray((recon_rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    comp.resize((small[1], small[0]), Image.LANCZOS).save(
        os.path.join(HERE, 'composite', f'rgb_composite_{tag}_z{int(z)}.png'))
    if tag == 'main':
        target = Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        target.resize((small[1], small[0]), Image.LANCZOS).save(
            os.path.join(HERE, 'composite', 'target_rgb.png'))
    torch.save({'recon_rgb': recon_rgb.cpu(), 'target': t.cpu(), 'psnrs': psnrs},
               os.path.join(HERE, 'composite', f'rgb_tensors_{tag}.pt'))
    print(f'[{tag}] saved', flush=True)
    return psnrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gpu', type=int, nargs='?', default=0)
    ap.add_argument('--only', choices=list(CONFIGS), default=None)
    args = ap.parse_args()
    tags = [args.only] if args.only else list(CONFIGS)
    for tag in tags:
        run(tag, args.gpu)


if __name__ == '__main__':
    main()
