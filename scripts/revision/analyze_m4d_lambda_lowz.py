"""Merge and analyse M4d: the lambda-axis cross-check of the lambda*z collapse law.

Reads the per-GPU eval_results.json written by run_m4d_lambda_lowz.sh, merges them
into one JSON, and locates the collapse crossing on each fixed-z wavelength profile.

Convention match: the z-axis crossing is defined the same way here -- the first
sample at/above 25 dB when scanning toward larger lambda*z (i.e. upward in lambda at
fixed z). The law predicts that crossing sits at a constant lambda*z regardless of
which axis the scan runs along:

    lambda*z_cross = 0.88 x (lambda*z band minimum) = 0.88 x (440nm x 145mm) = 56.2k nm.mm

so at z=120mm the crossing should be at 468nm and at z=110mm at 511nm. Both are
inside the training band [440,650]nm, so a wavelength-extrapolation failure cannot
masquerade as a lambda*z collapse.

Peak-anchor training leaves +/-10nm ripple at the three native lines, so the crossing
is reported both raw and on a 5-point rolling median (2nm step -> 10nm window).

Usage: python analyze_m4d_lambda_lowz.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp/experiments'
ROOT = os.path.join(EXP, 'rev_m4_lambda_lowz')
OUT = os.path.join(ROOT, 'merged_lowz_lambda.json')

BAND_MIN_LZ = 440.0 * 145.0          # nm.mm, training band lambda*z minimum
LAW_RATIO = 0.885                    # measured crossings / band minimum, both weights
THRESH_DB = 25.0


def load_points():
    by_z = {}
    for p in sorted(glob.glob(os.path.join(ROOT, 'z*', 'gpu*', 'eval_results.json'))):
        with open(p) as f:
            d = json.load(f)
        for v in d.values():
            z = float(v['z'])
            wl_nm = float(v['wavelength']) * 1e6
            by_z.setdefault(z, {})[wl_nm] = {
                'wavelength_nm': wl_nm,
                'psnr_mean': float(v['psnr_mean']),
                'psnr_std': float(v['psnr_std']),
                'ssim_mean': float(v['ssim_mean']),
            }
    return by_z


def rolling_median(x, w=5):
    pad = w // 2
    xp = np.pad(x, pad, mode='edge')
    return np.array([np.median(xp[i:i + w]) for i in range(len(x))])


def crossing(wl, psnr):
    """First index at/above the threshold, searching from the low-lambda end."""
    above = np.nonzero(psnr >= THRESH_DB)[0]
    return int(above[0]) if len(above) else None


def main():
    by_z = load_points()
    if not by_z:
        print('no M4d results yet under', ROOT)
        return

    merged = {
        'lowz_lambda_scan': {
            'description': (
                'Single spatial-conditional weight (rev_m3_spatial_s42, windows z in '
                '[145,205]mm and lambda in [440,650]nm) scanned in lambda at fixed low z '
                'below the distance window, pitch 3.6um, 100 DIV2K validation images per '
                'point, 2nm step (2026-09-11, run_m4d_lambda_lowz.sh). Purpose: cross-check '
                'the lambda*z collapse boundary on the wavelength axis. '
                'Fixed-peak amplitude convention, same as the M4 z scans.'
            ),
            'per_z': {},
        }
    }

    for z in sorted(by_z):
        pts = sorted(by_z[z].values(), key=lambda v: v['wavelength_nm'])
        wl = np.array([p['wavelength_nm'] for p in pts])
        ps = np.array([p['psnr_mean'] for p in pts])
        sm = rolling_median(ps)
        ci_raw, ci_sm = crossing(wl, ps), crossing(wl, sm)

        print(f'\n=== z = {z:.0f} mm  ({len(pts)} points, {wl.min():.0f}-{wl.max():.0f} nm) ===')
        step = 10
        print('  lambda(nm): ' + ' '.join(f'{int(w):>6d}' for w in wl[::step]))
        print('  PSNR  (dB): ' + ' '.join(f'{p:6.1f}' for p in ps[::step]))
        print('  rollmed   : ' + ' '.join(f'{p:6.1f}' for p in sm[::step]))

        pred = BAND_MIN_LZ * LAW_RATIO / z
        for name, idx in (('raw', ci_raw), ('rollmed', ci_sm)):
            if idx is None:
                print(f'  {name:8s}: no point reaches {THRESH_DB:.0f} dB')
                continue
            lz = wl[idx] * z / 1000.0
            print(f'  {name:8s}: crossing at {wl[idx]:.0f} nm '
                  f'-> lambda*z = {lz:.2f}k nm.mm '
                  f'({lz / (BAND_MIN_LZ / 1000.0):.3f} x band min, '
                  f'{100 * (wl[idx] - pred) / pred:+.1f}% vs predicted {pred:.0f} nm)')

        merged['lowz_lambda_scan']['per_z'][f'z{int(z)}mm'] = {
            'z': z,
            'predicted_crossing_nm': round(BAND_MIN_LZ * LAW_RATIO / z, 1),
            'points': pts,
        }

    os.makedirs(ROOT, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(merged, f, indent=2)
    print('\nsaved', OUT)


if __name__ == '__main__':
    main()
