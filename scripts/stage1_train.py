"""Stage 1: blue-450 z-residual training (residual_zcond_plan stage 1, comb-valley target).

Mechanism (per plan + 2026-06-25 comb finding):
  - Reuse FreqCondCCNNcghModel: CCNN1/CCNN2 warm-started from the blue-450 baseline
    and FROZEN; only the zero-init unit-modulus frequency-domain phase residual
    (freq_mult + cond_proj) is trained.
  - At init the residual = identity (M=1), so the model == frozen baseline.
  - lambda is fixed at 450nm, pitch fixed -> physics_embed varies only with z,
    i.e. this is z-only conditioning. Per-iter random z, recompute H each step.
  - Target: lift the self-imaging comb VALLEYS (e.g. blue z=170/179 ~19dB) toward
    the peaks, WITHOUT hurting the design peak z=150.

Self-supervised MSE, same pipeline/optics as train_baseline.py (both-move).
"""
import sys
import os

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.utils.tools import loadimage
from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel


class HolographyDataset(Dataset):
    def __init__(self, data_path, start_index, num_samples, n, m, augment=True, use_linear_convert=True):
        self.data_path = data_path
        self.start_index = start_index
        self.num_samples = num_samples
        self.n = n
        self.m = m
        self.augment = augment
        self.use_linear_convert = use_linear_convert

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_index = self.start_index + idx
        flip = np.random.randint(low=0, high=100) if self.augment else 0
        img = loadimage(path=self.data_path, image_index=image_index, channel=2, flip=flip,
                        m=self.m, n=self.n, convert=self.use_linear_convert, cuda=False)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        return torch.sqrt(img).view(1, self.n, self.m).float()


def compute_asm_kernels(n, m, pitch, wavelength, z, device):
    H_forward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=z, linear_conv=False, return_H=True).to(device)
    H_backward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                 wavelength=wavelength, z=-z, linear_conv=False, return_H=True).to(device)
    return H_forward, H_backward


def warm_start_and_freeze(model, ckpt_path, device, freeze=True):
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    base_keys = [k for k in sd if k.startswith('ccnn1.') or k.startswith('ccnn2.')]
    sub = {k: v for k, v in sd.items() if k in base_keys}

    # Support nested experts (e.g. ThreeExpertCCNNcghModelMinExp) by finding every
    # submodule that owns ccnn1/ccnn2 and loading the base backbone into it.
    expert_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'ccnn1') and hasattr(module, 'ccnn2'):
            expert_modules.append((name, module))
    if not expert_modules:
        # original flat model path
        missing, unexpected = model.load_state_dict(sub, strict=False)
        model_base = [n for n, _ in model.named_parameters()
                      if n.startswith('ccnn1.') or n.startswith('ccnn2.')]
        not_loaded = [n for n in model_base if n not in sub]
        assert not not_loaded, f"warm-start incomplete, missing base keys: {not_loaded[:5]}"
    else:
        loaded_per_expert = 0
        for name, module in expert_modules:
            # module is a submodule; load_state_dict expects keys relative to it,
            # not prefixed by the parent path.
            missing, unexpected = module.load_state_dict(sub, strict=False)
            # all ccnn1/ccnn2 in this submodule must be loaded
            mod_base = [n for n, _ in module.named_parameters()
                        if n.startswith('ccnn1.') or n.startswith('ccnn2.')]
            mod_not_loaded = [n for n in mod_base if n not in sub]
            assert not mod_not_loaded, f"warm-start incomplete for {name}: {mod_not_loaded[:5]}"
            loaded_per_expert = len(sub)
        sub = {f'<{len(expert_modules)} experts>': None}
        print(f"warm-start ok [nested experts]: loaded {loaded_per_expert} base tensors into each of {len(expert_modules)} expert(s)")

    if freeze:
        for n, p in model.named_parameters():
            if '.ccnn1.' in n or '.ccnn2.' in n or n.startswith('ccnn1.') or n.startswith('ccnn2.'):
                p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    mode = 'frozen-backbone' if freeze else 'end-to-end (backbone unfrozen)'
    print(f"warm-start ok [{mode}]: loaded {len(sub)} base tensors; "
          f"frozen={n_frozen:,} trainable={n_trainable:,}")
    return model


def reconstruct(model, target_amp, init_phase, z, pitch, wl, Hf, Hb):
    holo_phase, _ = model(target_amp, init_phase, z, False, pitch, wl, Hf)
    slm = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
    recon = propagation_ASM(u_in=slm, z=-z, linear_conv=False,
                            feature_size=[pitch, pitch], wavelength=wl, precomped_H=Hb)
    return torch.abs(recon)


def train(args):
    device = torch.device(f'cuda:{args.device_id}')
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    model = FreqCondCCNNcghModel(n_freqs=args.n_freqs, cond_dim=args.cond_dim,
                                 base_res=args.base_res, radial_len=args.base_res).to(device)
    model = warm_start_and_freeze(model, args.base_weight, device,
                                  freeze=not args.no_freeze)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    train_ds = HolographyDataset(args.train_path, 100, args.num_train, args.n, args.m, augment=True)
    valid_ds = HolographyDataset(args.valid_path, 801, args.num_valid, args.n, args.m, augment=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    init_phase = torch.zeros(1, 1, args.n, args.m, device=device)
    pitch, wl = args.pitch, args.wavelength
    # bounded LRU H cache: peak-anchored z repeat (worth caching) but uniform z
    # draws are mostly one-shot. An unbounded cache accumulates ~132MB per
    # distinct z (two 4K complex kernels) and OOMs mid-epoch on a shared GPU.
    from collections import OrderedDict
    H_cache = OrderedDict()
    H_CACHE_MAX = 16

    def get_H(zq):
        key = round(zq, 1)
        if key in H_cache:
            H_cache.move_to_end(key)
            return H_cache[key]
        val = compute_asm_kernels(args.n, args.m, pitch, wl, key, device)
        H_cache[key] = val
        if len(H_cache) > H_CACHE_MAX:
            H_cache.popitem(last=False)
        return val

    os.makedirs(args.output_dir, exist_ok=True)
    best = float('inf')
    history = {'train_loss': [], 'valid_loss': []}

    # identity sanity at init: at z=150 the residual must == baseline (M=1)
    model.eval()
    with torch.no_grad():
        ex = valid_ds[0].unsqueeze(0).to(device)
        Hf, Hb = get_H(150.0)
        r = reconstruct(model, ex, init_phase, 150.0, pitch, wl, Hf, Hb)
        print(f"init identity check z=150 MSE={criterion(r, ex).item():.6f}")

    for epoch in trange(args.epochs, desc='Epochs'):
        model.train()
        # keep frozen base in eval mode (no BN/dropout here, but be safe)
        loss_sum = 0.0
        for target_amp in tqdm(train_loader, desc=f'Train {epoch}', leave=False):
            target_amp = target_amp.to(device)
            # peak-anchored sampling: with prob peak_prob draw a known comb peak z
            # (loss upweighted) so the residual stays ~identity there; else draw a
            # valley-region z uniformly (where the residual must do the lifting).
            if args.peak_z and random.random() < args.peak_prob:
                z = random.choice(args.peak_z)
                w = args.peak_weight
            else:
                z = random.uniform(args.z_lo, args.z_hi)
                w = 1.0
            Hf, Hb = get_H(z)
            optimizer.zero_grad()
            recon_amp = reconstruct(model, target_amp, init_phase, round(z, 1), pitch, wl, Hf, Hb)
            loss = w * criterion(recon_amp, target_amp)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        avg_train = loss_sum / len(train_loader)
        history['train_loss'].append(avg_train)

        # validation: sweep a few representative z (peaks+valleys), avg
        model.eval()
        vsum, vcount = 0.0, 0
        with torch.no_grad():
            for target_amp in valid_loader:
                target_amp = target_amp.to(device)
                for zv in args.val_z:
                    Hf, Hb = get_H(float(zv))
                    recon_amp = reconstruct(model, target_amp, init_phase, float(zv), pitch, wl, Hf, Hb)
                    vsum += criterion(recon_amp, target_amp).item(); vcount += 1
        avg_valid = vsum / vcount
        history['valid_loss'].append(avg_valid)
        print(f"Epoch {epoch}: train={avg_train:.6f} valid(multi-z)={avg_valid:.6f}")

        if avg_valid < best:
            best = avg_valid
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'train_loss': avg_train, 'valid_loss': avg_valid},
                       os.path.join(args.output_dir, 'stage1_best.pth'))
            print("  -> new best saved")

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"done. best multi-z valid loss={best:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='./experiments/stage1_blue_zresidual')
    p.add_argument('--base-weight', required=True)
    p.add_argument('--train-path', default='./data/DIV2K_train_HR')
    p.add_argument('--valid-path', default='./data/DIV2K_valid_HR')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--num-train', type=int, default=500)
    p.add_argument('--num-valid', type=int, default=20)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n', type=int, default=2160)
    p.add_argument('--m', type=int, default=3840)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--wavelength', type=float, default=0.000450)
    p.add_argument('--z-lo', type=float, default=145.0)
    p.add_argument('--z-hi', type=float, default=205.0)
    p.add_argument('--val-z', type=float, nargs='+', default=[150.0, 159.0, 170.0, 179.0])
    p.add_argument('--peak-z', type=float, nargs='+', default=[],
                   help='known comb-peak z to anchor (residual stays ~identity there)')
    p.add_argument('--peak-prob', type=float, default=0.0,
                   help='prob per iter of drawing a peak z instead of uniform')
    p.add_argument('--peak-weight', type=float, default=1.0,
                   help='loss multiplier on peak-anchored samples')
    p.add_argument('--n-freqs', type=int, default=8)
    p.add_argument('--cond-dim', type=int, default=64)
    p.add_argument('--base-res', type=int, default=64)
    p.add_argument('--no-freeze', action='store_true',
                   help='warm-start the baseline backbone but fine-tune it '
                        'end-to-end (do NOT freeze ccnn1/ccnn2)')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
