"""M3 stats: aggregate the 3-seed retrain evals into mean+-std numbers for
Tables 3 and 4 of the revised manuscript.

For each model x seed x color:
  z-scan (145-205mm @1mm, 100 imgs): window mean, worst valley, % points >= 25 dB
For each model x seed (lambda finescan @z=150, 63 pts, 100 imgs):
  per-color lambda-ripple = peak-to-valley swing within +-10nm of native center
Prints mean+-std across seeds. Usage: python compute_m3_stats.py
"""
import json
import os
import sys
from statistics import mean, stdev

EXP = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp/experiments'
MODELS = ['spatial', 'frequency', 'unconditional']
SEEDS = [42, 43, 44]
COLORS = {'blue': 450, 'green': 532, 'red': 638}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def z_stats(d):
    vals = sorted(d.values(), key=lambda v: v['z'])
    ps = [v['psnr_mean'] for v in vals]
    return {
        'window_mean': mean(ps),
        'worst_valley': min(ps),
        'pct_above25': 100.0 * sum(p >= 25.0 for p in ps) / len(ps),
    }


def ripple(d, center_nm, half_width=10):
    vals = [v for v in d.values()
            if abs(v['wavelength'] * 1e6 - center_nm) <= half_width]
    if not vals:
        return None
    ps = [v['psnr_mean'] for v in vals]
    return max(ps) - min(ps)


def ms(vals, fmt='{:.2f}'):
    m, s = mean(vals), stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(m)}$\\pm${fmt.format(s)}"


def main():
    rows = {}
    for model in MODELS:
        rows[model] = {'z': {c: [] for c in COLORS}, 'ripple': {c: [] for c in COLORS}}
        for seed in SEEDS:
            for c, wl in COLORS.items():
                zp = f'{EXP}/rev_m3_{model}_s{seed}_eval_{c}/eval_results.json'
                rows[model]['z'][c].append(z_stats(load_json(zp)))
            lp = f'{EXP}/rev_m3_{model}_s{seed}_lambda_finescan_z150/eval_results.json'
            ld = load_json(lp)
            for c, wl in COLORS.items():
                rows[model]['ripple'][c].append(ripple(ld, wl))

    print('=== Table 3 (ablation, mean+-std over 3 seeds) ===')
    hdr = f"{'model':<14}" + ''.join(f'{c:>28}' for c in COLORS)
    print(hdr + '   (z-valley dB | lambda-ripple dB)')
    for model in MODELS:
        cells = []
        for c in COLORS:
            zv = [r['worst_valley'] for r in rows[model]['z'][c]]
            rp = rows[model]['ripple'][c]
            cells.append(f"{ms(zv):>12} | {ms(rp):>12}")
        print(f'{model:<14}' + '  '.join(cells))

    print()
    print('=== Table 4 (spatial-conditional full recipe, mean+-std over 3 seeds) ===')
    for c in COLORS:
        wm = [r['window_mean'] for r in rows['spatial']['z'][c]]
        wv = [r['worst_valley'] for r in rows['spatial']['z'][c]]
        pa = [r['pct_above25'] for r in rows['spatial']['z'][c]]
        print(f'{c:<6} window_mean {ms(wm)} dB | worst_valley {ms(wv)} dB | '
              f'>=25dB {ms(pa, "{:.1f}")} %')

    # overall across colors (for text statements)
    print()
    all_wm = [r['window_mean'] for c in COLORS for r in rows['spatial']['z'][c]]
    all_wv = [r['worst_valley'] for c in COLORS for r in rows['spatial']['z'][c]]
    print(f'spatial overall: window mean {ms(all_wm)} dB, worst valley {ms(all_wv)} dB')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'm3_stats.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    print('saved m3_stats.json')


if __name__ == '__main__':
    main()
