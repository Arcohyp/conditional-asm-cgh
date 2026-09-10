"""Fill the amplitude-convention (Convention B) cells for Table 5, for BOTH
models, including amplitude-domain SSIM which neither prior eval produced.

Convention B (our paper's native): amplitude domain, PSNR with
peak=(max target_amp)^2, NO per-image scaling; SSIM on amplitudes clamped
to [0,1] with data_range=1.0.

The 'ours' branch reuses the exact pipeline of eval_ours_m2.py (same model
construction, compute_asm_kernels/reconstruct, weight file); the 'sfo'
branch reuses the exact pipeline of eval_sfo_psnr_ssim.py plus the
Convention B metrics on the same reconstructions.

Usage: python eval_table5_amp.py <gpu-id> <sfo|ours> <r|g|b>
Output: experiments/m2_sfo/table5_amp_<mode>_<color>.json
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))

RES = (2160, 3840)
PITCH = 3.74e-3
Z_LIST = [85, 100, 115]
NUM = 100
DESIGN_Z = 100.0
OUR_MODEL_ROOT = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp'
WEIGHT = os.path.join(OUR_MODEL_ROOT, 'experiments',
                      'rev_m2_spatialcond_sfo_window', 'colorcond_best.pth')
SFO_DIR = '/mnt/sda/chengxirun/hologram_task/eghp_experiment/CGH-SFO-solver'

valid_dir = '/mnt/sda/datasets/DIV2K/DIV2K_valid_HR'
files = sorted(os.listdir(valid_dir))[:NUM]
tfm = T.Compose([T.Resize(RES), T.ToTensor()])
color_idx = {'r': 0, 'g': 1, 'b': 2}
wl_of = {'r': 638, 'g': 520, 'b': 450}


def amp_psnr_ssim(rec_amp, amp_target):
    """Convention B: PSNR with peak=(max target amp)^2, no scaling; SSIM on
    amplitudes clamped to [0,1], data_range=1.0."""
    peak = amp_target.max() ** 2
    mse = ((rec_amp - amp_target) ** 2).mean().item()
    psnr = 10 * np.log10(peak.item() / max(mse, 1e-12))
    ra = rec_amp.clamp(0, 1).squeeze().cpu().numpy().astype(np.float32)
    ta = amp_target.clamp(0, 1).squeeze().cpu().numpy().astype(np.float32)
    return float(psnr), float(sk_ssim(ta, ra, data_range=1.0))


def main():
    gpu = int(sys.argv[1])
    mode = sys.argv[2]
    color = sys.argv[3]
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    wl_nm = wl_of[color]

    results = {}
    if mode == 'sfo':
        sys.path.insert(0, SFO_DIR)
        from model.FourierNet import Flex_FourierNet
        from prop import ASM_split
        prop = ASM_split(wavelength=wl_nm, res=RES, roi=RES, pitch=PITCH,
                         apply_constraint=True).to(device).eval()
        model = Flex_FourierNet(wl=wl_nm, center_distance=100).to(device)
        sd = torch.load(os.path.join(SFO_DIR, 'model', color,
                                     'FourierNet_flex_100.pth'),
                        map_location=device)
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            for dist in Z_LIST:
                psnrs, ssims = [], []
                dt = torch.tensor([[float(dist)]], device=device)
                for f in files:
                    img = Image.open(os.path.join(valid_dir, f)).convert('RGB')
                    t = tfm(img)[color_idx[color]].to(device)
                    target = t.unsqueeze(0).unsqueeze(0)
                    holo = model(target, dt)
                    _, _, final = prop(torch.ones_like(holo), holo, dt)
                    p, s = amp_psnr_ssim(final.sqrt(), target.sqrt())
                    psnrs.append(p)
                    ssims.append(s)
                key = f'{color}_{wl_nm}nm_z{dist}mm'
                results[key] = {'psnr_amp_mean': float(np.mean(psnrs)),
                                'psnr_amp_std': float(np.std(psnrs)),
                                'ssim_amp_mean': float(np.mean(ssims)),
                                'n': NUM}
                print(key, {k: round(v, 4) for k, v in results[key].items()
                            if k != 'n'}, flush=True)
    else:
        sys.path.insert(0, '/mnt/sda/chengxirun/hologram_task/hologram_task')
        sys.path.insert(0, OUR_MODEL_ROOT)
        from stage1_train import compute_asm_kernels, reconstruct
        from exp_models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
        model = ConditionalCCNNcghModelMinExp(param_dim=16).to(device)
        model.param_encoder.z_center = DESIGN_Z
        ck = torch.load(WEIGHT, map_location=device)
        model.load_state_dict(ck['model_state_dict']
                              if 'model_state_dict' in ck else ck)
        model.eval()
        wl_mm = wl_nm * 1e-6
        init_phase = torch.zeros(1, 1, *RES, device=device)
        with torch.no_grad():
            for dist in Z_LIST:
                Hf, Hb = compute_asm_kernels(RES[0], RES[1], PITCH, wl_mm,
                                             float(dist), device)
                psnrs, ssims = [], []
                for f in files:
                    img = Image.open(os.path.join(valid_dir, f)).convert('RGB')
                    t = tfm(img)[color_idx[color]].to(device)
                    amp_in = t.sqrt().view(1, 1, *RES)
                    rec_amp = reconstruct(model, amp_in, init_phase,
                                          float(dist), PITCH, wl_mm, Hf, Hb)
                    p, s = amp_psnr_ssim(rec_amp, amp_in)
                    psnrs.append(p)
                    ssims.append(s)
                key = f'{color}_{wl_nm}nm_z{dist}mm'
                results[key] = {'psnr_amp_mean': float(np.mean(psnrs)),
                                'psnr_amp_std': float(np.std(psnrs)),
                                'ssim_amp_mean': float(np.mean(ssims)),
                                'n': NUM}
                print(key, {k: round(v, 4) for k, v in results[key].items()
                            if k != 'n'}, flush=True)

    out = os.path.join(HERE, f'table5_amp_{mode}_{color}.json')
    with open(out, 'w') as fp:
        json.dump(results, fp, indent=2)
    gm = np.mean([v['psnr_amp_mean'] for v in results.values()])
    print(f'{mode} {color} GRAND MEAN amp PSNR: {gm:.2f} dB; '
          f'amp SSIM: {np.mean([v["ssim_amp_mean"] for v in results.values()]):.4f}')


if __name__ == '__main__':
    main()
