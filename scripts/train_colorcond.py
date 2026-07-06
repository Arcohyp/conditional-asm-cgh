"""Step3-b: color-parametric (z, lambda)-conditioned comb-fill training.

Mechanism (extends the D1 unfreeze result to all three colors in ONE model):
  - Same FreqCondCCNNcghModel, which already conditions on (z, lambda) via
    physics_embed -> cond_proj -> ConditionalFreqMultiplier. So NO model change;
    we only feed all three wavelengths during training.
  - warm-start the CCNN1/CCNN2 backbone from ONE baseline (blue 450) for a stable
    init, then UNFREEZE (fine-tune end-to-end) -- the D1 finding that unfreezing
    is what breaks the comb-valley ceiling. The backbone co-adapts into a
    color-conditional operator.
  - Each step: draw a color (wavelength) uniformly, then peak-anchored z for THAT
    color (its own self-imaging comb peaks), recompute H(z, lambda), self-sup MSE.

Goal: a single conditional model that holds every color's comb peaks AND fills
its valleys to usable (>=25dB), across z=145-205 at all of {450,532,638}nm.
Judged by comb_metrics.py per color vs that color's both-move baseline.
"""
import sys
import os

import argparse
import json
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from conditional_asm_cgh.scripts.stage1_train import HolographyDataset, compute_asm_kernels, reconstruct, warm_start_and_freeze
from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel
from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_3expert_minexp import ThreeExpertCCNNcghModelMinExp


# self-imaging z-comb peaks per color (mm), from the 1mm 100-sample comb maps
# (blue ~9mm period, green ~10mm, red ~5mm; all anchored at the z=150 design pt)
PEAKS = {
    0.000450: [150, 159, 168, 177, 186, 195, 204],
    0.000532: [150, 160, 170, 180, 190, 200],
    0.000638: [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 204],
}

# lambda peaks at z=150 extracted from baseline both-move lambda-comb scans (1 nm grid)
# used for the optional hard lambda-peak anchor ablation.
LAMBDA_PEAKS = {
    450: [441, 450, 459],
    532: [523, 532, 541],
    638: [627, 638, 649],
}


def npcc_loss(recon, target):
    """Negative Pearson correlation coefficient (SFO's structural loss term).
    Scale/shift-invariant, complements MSE by rewarding structural agreement.
    batch_size==1 here, so compute over all pixels of the single image."""
    a = recon - recon.mean()
    b = target - target.mean()
    num = (a * b).sum()
    den = torch.sqrt((a * a).sum() * (b * b).sum()) + 1e-8
    return -num / den


def lambda_valley_weight(z_mm, wl_mm, pitch_mm, n_freqs=8, k=2.0):
    """Compute a scalar valley-ness weight for a given (z, lambda) sample.

    The ASM transfer function phase for a propagating spatial frequency f is
    phi(f) = (2*pi/wl) * z * sqrt(1 - wl**2 * f**2).
    A frequency is at a self-imaging peak when phi is a multiple of 2*pi,
    and at a valley when phi is an odd multiple of pi.
    We aggregate valley-ness over a representative set of radial frequencies
    and map it to a loss multiplier: peak -> 1.0, valley -> 1 + lambda_valley_weight.

    Args:
        z_mm, wl_mm, pitch_mm: physical parameters in mm.
        n_freqs: number of radial frequencies to sample.
        k: steepness of the valley-sensitivity (larger = sharper discrimination).
    Returns:
        A float in [1.0, 1 + lambda_valley_weight_max].
    """
    wl = wl_mm
    z = z_mm
    # Nyquist-like max radial frequency for the SLM pixel grid (1/mm).
    f_max = 1.0 / pitch_mm
    # Sample radial frequencies on a log-ish grid from low to near Nyquist.
    fs = np.logspace(-2, 0, n_freqs) * f_max
    # Propagating frequencies only: wl * f < 1.
    valid = wl * fs < 0.999
    if not valid.any():
        return 1.0
    fs = fs[valid]
    phase = (2.0 * np.pi / wl) * z * np.sqrt(1.0 - (wl * fs) ** 2)
    # Valley-ness: 0 at peaks, 1 at valleys (sin^2(phi/2)).
    valley = np.mean(np.sin(phase * 0.5) ** 2)
    return 1.0 + valley


def train(args):
    device = torch.device(f'cuda:{args.device_id}')
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    wavelengths = args.wavelengths
    if args.encoder == 'freq':
        model = FreqCondCCNNcghModel(n_freqs=args.n_freqs, cond_dim=args.cond_dim,
                                     base_res=args.base_res, radial_len=args.base_res).to(device)
    elif args.encoder == 'baseline':
        # A1 ablation: NO conditioning at all. Backbone can only adapt globally;
        # z/lambda enter solely through the ASM kernel H. Tests whether unfreeze
        # alone (no freq multiplier) can fill the comb valleys.
        model = BaselineCCNNcghModelMinExp().to(device)
    elif args.encoder == 'cond':
        # A3 ablation: spatial/global (param-vector) conditioning instead of the
        # frequency-domain radial multiplier. Same idea of conditioning on (z,lambda)
        # but in the WRONG domain -- tests whether freq-domain specifically matters.
        model = ConditionalCCNNcghModelMinExp(param_dim=args.param_dim).to(device)
    elif args.encoder == '3expert':
        # M1e: three independent (z,lambda)-conditioned experts blended by lambda.
        # Tests whether the bottleneck is requiring one shared backbone to fit all
        # three colours, rather than the capacity of the individual CCNN U-Net.
        model = ThreeExpertCCNNcghModelMinExp(param_dim=args.param_dim,
                                              boundaries=args.expert_boundaries,
                                              soft_width=args.expert_soft_width).to(device)
    else:
        raise ValueError(f"unknown encoder {args.encoder}")
    # warm-start blue backbone but DO NOT freeze (D1: end-to-end is what fills valleys)
    model = warm_start_and_freeze(model, args.base_weight, device, freeze=False)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    train_ds = HolographyDataset(args.train_path, 100, args.num_train, args.n, args.m, augment=True)
    valid_ds = HolographyDataset(args.valid_path, 801, args.num_valid, args.n, args.m, augment=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    init_phase = torch.zeros(1, 1, args.n, args.m, device=device)
    pitch = args.pitch

    # bounded LRU H cache keyed by (wavelength, z) -- unbounded caching of 4K
    # complex kernels OOMs mid-epoch on a shared GPU (see D1 notes).
    H_cache = OrderedDict()
    H_CACHE_MAX = 24

    def get_H(wl, zq):
        key = (round(wl, 9), round(zq, 1))
        if key in H_cache:
            H_cache.move_to_end(key)
            return H_cache[key]
        val = compute_asm_kernels(args.n, args.m, pitch, key[0], key[1], device)
        H_cache[key] = val
        if len(H_cache) > H_CACHE_MAX:
            H_cache.popitem(last=False)
        return val

    os.makedirs(args.output_dir, exist_ok=True)
    best = float('inf')
    history = {'train_loss': [], 'valid_loss': []}

    # identity sanity at init: at z=150, each color must == its warm-started base
    model.eval()
    with torch.no_grad():
        ex = valid_ds[0].unsqueeze(0).to(device)
        for wl in wavelengths:
            Hf, Hb = get_H(wl, 150.0)
            r = reconstruct(model, ex, init_phase, 150.0, pitch, wl, Hf, Hb)
            print(f"init identity z=150 wl={wl*1e6:.0f}nm MSE={criterion(r, ex).item():.6f}")

    for epoch in trange(args.epochs, desc='Epochs'):
        model.train()
        loss_sum = 0.0
        for target_amp in tqdm(train_loader, desc=f'Train {epoch}', leave=False):
            target_amp = target_amp.to(device)
            if args.lambda_continuous:
                # continuous-lambda training: draw wavelength uniformly across the
                # working band so the conditional multiplier must interpolate the
                # per-lambda residual correction, not just memorise 3 discrete nm.
                wl = random.uniform(args.lambda_lo, args.lambda_hi)
            else:
                wl = random.choice(wavelengths)
            # peak-anchored z for THIS color: with prob, draw a known comb peak
            # (loss upweighted) so the model stays ~identity at peaks; else uniform.
            if random.random() < args.peak_prob:
                if args.lambda_continuous:
                    # z=150 (design point) is a peak for EVERY lambda; the per-color
                    # revival peaks in PEAKS are only defined at the 3 native nm.
                    z = 150.0
                else:
                    z = float(random.choice(PEAKS[round(wl, 9)]))
                w = args.peak_weight
            else:
                z = random.uniform(args.z_lo, args.z_hi)
                w = 1.0

            # Optional lambda-axis peak / valley aware reweighting.
            if args.lambda_anchor_mode == 'valley':
                # upweight samples that sit in lambda-valleys according to the
                # aggregated ASM phase across representative spatial frequencies.
                w = w * (1.0 + args.lambda_valley_gain *
                         (lambda_valley_weight(z, wl, pitch, n_freqs=args.lambda_valley_nfreqs) - 1.0))
            elif args.lambda_anchor_mode == 'peak':
                # hard anchor on known lambda-peaks near the native wavelengths.
                # In continuous-lambda mode the "current colour" is not well-defined,
                # so we first draw a native centre (450/532/638) and then optionally
                # perturb it slightly to stay inside the continuous band while keeping
                # the sample anchored around a known self-imaging peak.
                if random.random() < args.lambda_peak_prob:
                    native_wl = random.choice(wavelengths)
                    if args.lambda_continuous:
                        # small +/- 2 nm perturbation around native peak to avoid
                        # memorising exactly 3 wavelengths.
                        wl = native_wl + random.uniform(-2e-6, 2e-6)
                        wl = max(args.lambda_lo, min(args.lambda_hi, wl))
                    else:
                        wl = native_wl
                    w = w * args.lambda_peak_weight
            Hf, Hb = get_H(wl, z)
            optimizer.zero_grad()
            recon_amp = reconstruct(model, target_amp, init_phase, round(z, 1), pitch, wl, Hf, Hb)
            loss = w * (args.mse_weight * criterion(recon_amp, target_amp)
                        + args.npcc_weight * npcc_loss(recon_amp, target_amp))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        avg_train = loss_sum / len(train_loader)
        history['train_loss'].append(avg_train)

        # validation: per color, sweep a few representative z (peaks+valleys)
        model.eval()
        vsum, vcount = 0.0, 0
        with torch.no_grad():
            for target_amp in valid_loader:
                target_amp = target_amp.to(device)
                for wl in wavelengths:
                    for zv in args.val_z:
                        Hf, Hb = get_H(wl, float(zv))
                        recon_amp = reconstruct(model, target_amp, init_phase, float(zv), pitch, wl, Hf, Hb)
                        vsum += criterion(recon_amp, target_amp).item(); vcount += 1
        avg_valid = vsum / vcount
        history['valid_loss'].append(avg_valid)
        print(f"Epoch {epoch}: train={avg_train:.6f} valid(multi-color,multi-z)={avg_valid:.6f}")

        if avg_valid < best:
            best = avg_valid
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'train_loss': avg_train, 'valid_loss': avg_valid},
                       os.path.join(args.output_dir, 'colorcond_best.pth'))
            print("  -> new best saved")

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"done. best multi-color valid loss={best:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='./experiments/step3b_colorcond')
    p.add_argument('--base-weight', required=True, help='warm-start backbone (blue 450 baseline)')
    p.add_argument('--train-path', default='./data/DIV2K_train_HR')
    p.add_argument('--valid-path', default='./data/DIV2K_valid_HR')
    p.add_argument('--wavelengths', type=float, nargs='+',
                   default=[0.000450, 0.000532, 0.000638])
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--num-train', type=int, default=600)
    p.add_argument('--num-valid', type=int, default=15)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n', type=int, default=2160)
    p.add_argument('--m', type=int, default=3840)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--z-lo', type=float, default=145.0)
    p.add_argument('--z-hi', type=float, default=205.0)
    p.add_argument('--val-z', type=float, nargs='+', default=[150.0, 165.0, 179.0])
    p.add_argument('--peak-prob', type=float, default=0.4)
    p.add_argument('--peak-weight', type=float, default=4.0)
    p.add_argument('--mse-weight', type=float, default=2.0,
                   help='weight on MSE term (SFO uses 2)')
    p.add_argument('--npcc-weight', type=float, default=1.0,
                   help='weight on NPCC term (SFO uses 1); 0 disables = pure MSE')
    p.add_argument('--lambda-continuous', action='store_true',
                   help='draw wavelength uniformly from [lambda-lo, lambda-hi] each '
                        'step instead of from the 3 discrete --wavelengths; forces '
                        'z=150 (design-pt peak, valid for every lambda) at peak draws')
    p.add_argument('--lambda-lo', type=float, default=0.000440)
    p.add_argument('--lambda-hi', type=float, default=0.000650)
    p.add_argument('--n-freqs', type=int, default=16)
    p.add_argument('--cond-dim', type=int, default=64)
    p.add_argument('--base-res', type=int, default=128)
    p.add_argument('--encoder', type=str, default='freq', choices=['freq', 'baseline', 'cond', '3expert'],
                   help='freq=conditional freq multiplier (main); baseline=no conditioning '
                        '(A1 ablation); cond=spatial/global param conditioning (A3 ablation); '
                        '3expert=three lambda-routed experts (M1e)')
    p.add_argument('--param-dim', type=int, default=16,
                   help='param-vector dim for the spatial/global cond ablation (encoder=cond) and 3expert')
    p.add_argument('--expert-boundaries', type=float, nargs=2, default=[0.000495, 0.000585],
                   help='lambda boundaries (mm) between blue/green/red experts for encoder=3expert')
    p.add_argument('--expert-soft-width', type=float, default=5e-6,
                   help='blending width (mm) at expert boundaries for encoder=3expert')
    p.add_argument('--lambda-anchor-mode', type=str, default='none',
                   choices=['none', 'valley', 'peak'],
                   help='lambda-axis comb-aware reweighting: none=uniform (baseline); '
                        'valley=upweight lambda-valleys via ASM phase; '
                        'peak=hard anchor on known lambda-peaks near z=150')
    p.add_argument('--lambda-valley-gain', type=float, default=1.0,
                   help='multiplier on the computed lambda-valley-ness (mode=valley); '
                        'loss weight becomes 1 + gain * valley-ness')
    p.add_argument('--lambda-valley-nfreqs', type=int, default=8,
                   help='number of radial spatial frequencies used for valley-ness estimate')
    p.add_argument('--lambda-peak-prob', type=float, default=0.4,
                   help='prob of replacing the sampled lambda with a known lambda-peak (mode=peak)')
    p.add_argument('--lambda-peak-weight', type=float, default=4.0,
                   help='loss multiplier for hard lambda-peak anchors (mode=peak)')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
