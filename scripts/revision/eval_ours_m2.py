"""M2 eval: our spatial-conditional model (retrained on SFO's window) under
BOTH metric conventions, on the same 100 DIV2K val images as validate_100imgs.

Convention A (SFO-matched, primary for the comparison table):
  target intensity t = gamma [0,1] RGB channel (PIL ToTensor, like SFO)
  input amplitude = sqrt(t)  (model-native input)
  recon intensity I = recon_amp**2, per-image sum-scale, clamp[0,1]
  skimage PSNR(data_range=1) + SSIM -- identical pipeline to SFO's.

Convention B (our paper's native): amplitude-domain PSNR,
  peak=(max target_amp)**2, no per-image scaling.

Usage: python eval_ours_m2.py <gpu-id>
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import torchvision.transforms as T

sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/hologram_task')
sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp')

from stage1_train import compute_asm_kernels, reconstruct  # noqa: E402
from exp_models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp  # noqa: E402

OUT = './experiments/rev_m2_spatialcond_sfo_window'
WEIGHT = os.path.join(OUT, 'colorcond_best.pth')
RES = (2160, 3840)
PITCH = 3.74e-3
GRID = [(0.000450, 'b', 2), (0.000520, 'g', 1), (0.000638, 'r', 0)]  # (wl_mm, name, RGB idx)
Z_LIST = [85, 100, 115]
NUM = 100
DESIGN_Z = 100.0


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
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

    results = {}
    with torch.no_grad():
        for wl, cname, ch in GRID:
            kernel_cache = {}
            for z in Z_LIST:
                psnr_a, ssim_a, psnr_b = [], [], []
                for f in files:
                    img = Image.open(os.path.join(valid_dir, f)).convert('RGB')
                    t = tfm(img)[ch].to(device)            # [H,W] gamma intensity
                    amp_in = t.sqrt().view(1, 1, *RES)
                    if z not in kernel_cache:
                        kernel_cache[z] = compute_asm_kernels(RES[0], RES[1], PITCH, wl, float(z), device)
                    Hf, Hb = kernel_cache[z]
                    rec_amp = reconstruct(model, amp_in, init_phase, float(z), PITCH, wl, Hf, Hb)
                    # Convention A (SFO-style, intensity + sum-scale)
                    I = (rec_amp ** 2)
                    scale = t.sum() / I.sum()
                    I_scaled = (I.view(-1) * scale).clamp(0, 1).view(RES)
                    t_np = t.cpu().numpy().astype(np.float32)
                    i_np = I_scaled.cpu().numpy().astype(np.float32)
                    psnr_a.append(sk_psnr(t_np, i_np, data_range=1.0))
                    ssim_a.append(sk_ssim(t_np, i_np, data_range=1.0))
                    # Convention B (amplitude domain, paper style)
                    peak = amp_in.max() ** 2
                    mse = ((rec_amp - amp_in) ** 2).mean().item()
                    psnr_b.append(10 * np.log10(peak.item() / max(mse, 1e-12)))
                key = f'{cname}_{int(wl*1e6)}nm_z{z}mm'
                results[key] = {
                    'psnr_sfo_mean': float(np.mean(psnr_a)),
                    'psnr_sfo_std': float(np.std(psnr_a)),
                    'ssim_sfo_mean': float(np.mean(ssim_a)),
                    'psnr_amp_mean': float(np.mean(psnr_b)),
                    'n': NUM,
                }
                print(key, {k: round(v, 3) for k, v in results[key].items() if k != 'n'})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'm2_ours_eval_both_conventions.json'), 'w') as fp:
        json.dump(results, fp, indent=2)
    gm = np.mean([v['psnr_sfo_mean'] for v in results.values()])
    print(f"GRAND MEAN (SFO convention): {gm:.2f} dB")


if __name__ == '__main__':
    main()
