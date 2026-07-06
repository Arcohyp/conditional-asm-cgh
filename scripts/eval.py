import sys
import os

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import csv
from tqdm import tqdm
from PIL import Image

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.utils.tools import loadimage
from torchvision import transforms

from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp as ConditionalCCNNcghModel
from conditional_asm_cgh.models.ccnn_cgh_physenc_minexp import PhysCondCCNNcghModel
from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel
from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp


from conditional_asm_cgh.models.ccnn_cgh_3expert_minexp import ThreeExpertCCNNcghModelMinExp


def compute_asm_kernels(n, m, pitch, wavelength, z, device):
    H_forward = propagation_ASM(
        torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
        wavelength=wavelength, z=z, linear_conv=False, return_H=True
    ).to(device)
    H_backward = propagation_ASM(
        torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
        wavelength=wavelength, z=-z, linear_conv=False, return_H=True
    ).to(device)
    return H_forward, H_backward


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def ssim(pred, target, window_size=11):
    import torch.nn.functional as F
    sigma = 1.5
    channel = pred.size(1)
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.expand(channel, 1, window_size, window_size).contiguous().to(pred.device)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def evaluate(args):
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    if args.encoder == 'phys':
        model = PhysCondCCNNcghModel(param_dim=args.param_dim, n_freqs=args.n_freqs).to(device)
    elif args.encoder == 'freq':
        model = FreqCondCCNNcghModel(n_freqs=args.n_freqs, cond_dim=args.cond_dim,
                                     base_res=args.base_res, radial_len=args.base_res).to(device)
    elif args.encoder == 'baseline':
        model = BaselineCCNNcghModelMinExp().to(device)
    elif args.encoder == 'cond':
        model = ConditionalCCNNcghModel(param_dim=args.param_dim).to(device)
    elif args.encoder == '3expert':
        model = ThreeExpertCCNNcghModelMinExp(param_dim=args.param_dim,
                                              boundaries=args.expert_boundaries,
                                              soft_width=args.expert_soft_width).to(device)
    else:
        model = ConditionalCCNNcghModel(param_dim=args.param_dim).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    init_phase = torch.zeros(1, 1, args.n, args.m, device=device)
    criterion = nn.MSELoss()

    configs = []
    for z in args.z_list:
        for wl in args.wl_list:
            configs.append((z, wl))

    results = {}
    for z, wl in configs:
        print(f"Evaluating z={z} mm, wavelength={wl*1e6:.1f} nm ...")
        H_forward, H_backward = compute_asm_kernels(args.n, args.m, args.pitch, wl, z, device)

        psnrs, ssims, losses = [], [], []
        for idx in tqdm(range(args.num_samples), leave=False):
            image_index = 801 + idx
            img = loadimage(args.valid_path, image_index, channel=2, flip=0,
                            m=args.m, n=args.n, convert=True, cuda=False)
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            target_amp = torch.sqrt(img).view(1, 1, args.n, args.m).float().to(device)

            with torch.no_grad():
                holo_phase, _ = model(target_amp, init_phase, z, False, args.pitch, wl, H_forward)
                slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
                recon_complex = propagation_ASM(
                    u_in=slm_complex, z=-z, linear_conv=False,
                    feature_size=[args.pitch, args.pitch], wavelength=wl,
                    precomped_H=H_backward
                )
                recon_amp = torch.abs(recon_complex)
                loss = criterion(recon_amp, target_amp).item()

            psnrs.append(psnr(recon_amp, target_amp))
            ssims.append(ssim(recon_amp, target_amp))
            losses.append(loss)

        results[f'z{z:.1f}_wl{wl*1e6:.0f}nm'] = {
            'psnr_mean': float(np.mean(psnrs)),
            'psnr_std': float(np.std(psnrs)),
            'ssim_mean': float(np.mean(ssims)),
            'loss_mean': float(np.mean(losses)),
            'z': z,
            'wavelength': wl,
        }
        print(f"  PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB, SSIM: {np.mean(ssims):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args.output_dir, 'eval_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z_mm', 'wavelength_nm', 'psnr_mean', 'psnr_std', 'ssim_mean', 'loss_mean'])
        for key, val in results.items():
            writer.writerow([val['z'], val['wavelength']*1e6, val['psnr_mean'], val['psnr_std'],
                             val['ssim_mean'], val['loss_mean']])

    print(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./conditional_asm_exp/eval_results')
    parser.add_argument('--valid-path', type=str, default='./data/DIV2K_valid_HR')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--device-id', type=int, default=0)

    parser.add_argument('--n', type=int, default=1080)
    parser.add_argument('--m', type=int, default=1920)
    parser.add_argument('--pitch', type=float, default=0.0036)
    parser.add_argument('--z-list', type=float, nargs='+', default=[145.0, 150.0, 155.0])
    parser.add_argument('--wl-list', type=float, nargs='+', default=[0.000638])
    parser.add_argument('--param-dim', type=int, default=16)
    parser.add_argument('--encoder', type=str, default='plain', choices=['plain', 'phys', 'freq', 'baseline', 'cond', '3expert'])
    parser.add_argument('--expert-boundaries', type=float, nargs=2, default=(0.000495, 0.000585))
    parser.add_argument('--expert-soft-width', type=float, default=5e-6)
    parser.add_argument('--n-freqs', type=int, default=8)
    parser.add_argument('--cond-dim', type=int, default=64)
    parser.add_argument('--base-res', type=int, default=64)
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
