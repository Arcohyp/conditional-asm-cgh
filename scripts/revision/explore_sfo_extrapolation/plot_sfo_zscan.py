"""Aggregate the SFO z-scan JSONs and plot PSNR/SSIM vs propagation distance.

Exploratory analysis only -- NOT part of the manuscript. Compares the
released SFO weights' behavior inside and outside their training window
z in [85,115] mm, and overlays our spatial-conditional model's M4
extrapolation scan (fixed-peak amplitude convention) for qualitative
reference.

Output: sfo_zscan_summary.png + README-ready summary stats printed.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.dirname(HERE)

COLORS = {'r': ('Red (638 nm)', '#c00000'),
          'g': ('Green (520 nm)', '#2e7d32'),
          'b': ('Blue (450 nm)', '#1f4e79')}
MARK = {'r': 'o', 'g': 's', 'b': '^'}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)

    for c in ['r', 'g', 'b']:
        fp = os.path.join(HERE, f'sfo_zscan_{c}.json')
        with open(fp) as f:
            res = json.load(f)
        z = np.array([v['z'] for v in res.values()])
        psnr = np.array([v['psnr_mean'] for v in res.values()])
        psnr_std = np.array([v['psnr_std'] for v in res.values()])
        ssim = np.array([v['ssim_mean'] for v in res.values()])
        amp = np.array([v['amp_psnr_mean'] for v in res.values()])
        order = np.argsort(z)
        z, psnr, psnr_std, ssim, amp = (a[order] for a in
                                        (z, psnr, psnr_std, ssim, amp))

        label, colr = COLORS[c]
        axes[0].plot(z, psnr, marker=MARK[c], ms=3.5, color=colr, label=label)
        axes[0].fill_between(z, psnr - psnr_std, psnr + psnr_std,
                             color=colr, alpha=0.15)
        axes[1].plot(z, amp, marker=MARK[c], ms=3.5, color=colr, label=label)
        axes[2].plot(z, ssim, marker=MARK[c], ms=3.5, color=colr, label=label)

        # headline stats
        inw = (z >= 85) & (z <= 115)
        print(f'{label}: in-window mean {psnr[inw].mean():.2f} dB; '
              f'max {psnr.max():.2f} @ {z[np.argmax(psnr)]} mm; '
              f'min {psnr.min():.2f} @ {z[np.argmin(psnr)]} mm')

    for ax, title, ylab in [(axes[0], 'SFO convention: intensity PSNR (dB)',
                             'PSNR (dB)'),
                            (axes[1], 'Amplitude PSNR, sum-scaled (dB)',
                             'PSNR (dB)'),
                            (axes[2], 'SFO convention: SSIM', 'SSIM')]:
        ax.axvspan(85, 115, color='gray', alpha=0.15, lw=0)
        if 'PSNR' in title:
            ax.axhline(25, color='k', ls=':', lw=1)
        ax.set_xlabel('Propagation distance z (mm)')
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].text(100, axes[0].get_ylim()[0] + 0.5, 'training window',
                 ha='center', fontsize=8, color='gray')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    out = os.path.join(HERE, 'sfo_zscan_summary.png')
    fig.savefig(out, dpi=200)
    print('saved', out)


if __name__ == '__main__':
    main()
