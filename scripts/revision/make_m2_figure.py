"""M2 figure: per-cell PSNR comparison on the shared SFO window.
Bars: SFO (per-color 109M specialists) vs ours (single 44,221-param weight),
SFO intensity-domain convention, 100 DIV2K val images, z in {85,100,115}.
Output: fig10_m2_sfo_quality.png next to this script.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = HERE
EXP = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp'

NAVY = '#1f4e79'
AMBER = '#c55a11'
SFO_GRAY = '#b3b3b3'
GRAY = '#595959'
BLACK = '#000000'

for face in ['Regular', 'Bold', 'Italic', 'Bold_Italic']:
    ttf = f'/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_{face}.ttf'
    if os.path.exists(ttf):
        fm.fontManager.addfont(ttf)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'text.color': BLACK,
    'axes.labelcolor': BLACK,
    'xtick.color': BLACK,
    'ytick.color': BLACK,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': GRAY,
    'grid.alpha': 0.3,
})

CELLS = [('b', 'blue', 450), ('g', 'green', 520), ('r', 'red', 638)]
Z_LIST = [85, 100, 115]


def main():
    sfo = json.load(open(os.path.join(M2, 'sfo_psnr_ssim_100imgs.json')))
    ours_p = os.path.join(HERE, 'm2_ours_eval_stepD.json')
    ours = json.load(open(ours_p))

    labels, sfo_v, ours_v = [], [], []
    for c, name, wl in CELLS:
        for z in Z_LIST:
            labels.append(f'{name[0].upper()}{z}')
            sfo_v.append(sfo[f'{c}_{wl}nm_z{z}mm']['psnr_mean'])
            ours_v.append(ours[f'{c}_{wl}nm_z{z}mm']['psnr_sfo_mean'])

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    b1 = ax.bar(x - w / 2, sfo_v, w, color=SFO_GRAY, label='SFO (109M/color)')
    b2 = ax.bar(x + w / 2, ours_v, w, color=AMBER, label='ours (44k, one weight)')
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.4,
                    f'{r.get_height():.1f}', ha='center', va='bottom',
                    fontsize=6, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('channel & distance (mm)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_ylim(0, 40)
    ax.grid(True, axis='y')
    ax.legend(frameon=False, loc='upper right', ncol=2)
    # group separators + color band labels
    for i in (3, 6):
        ax.axvline(i - 0.5, color=GRAY, linewidth=0.6, alpha=0.5)
    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(HERE, '..', 'figs_n2n3n5', 'fig10_m2_sfo_quality.png'))
    plt.close(fig)
    print('saved fig10_m2_sfo_quality.png')
    print('SFO mean', np.mean(sfo_v), 'ours mean', np.mean(ours_v),
          'delta', np.mean(sfo_v) - np.mean(ours_v))


if __name__ == '__main__':
    main()
