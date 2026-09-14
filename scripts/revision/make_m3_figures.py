"""Regenerate fig4_zcomb and fig5_lambdacomb from the unified M3 reruns
(3 seeds, full recipe), plotting the seed mean. Single-point baseline curves
in fig4 keep the original specialist data (unchanged by M3).
N5 colors: uncond/single-point=gray, spatial=orange (highlight), freq=navy.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp/experiments'

AMBER = '#c55a11'
NAVY = '#1f4e79'
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
    'axes.titlesize': 11,
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

SEEDS = [42, 43, 44]
MODELS = {'spatial': AMBER, 'frequency': NAVY, 'unconditional': GRAY}
NAMES = {'spatial': 'spatial-cond.', 'frequency': 'freq-cond.', 'unconditional': 'uncond.'}
STYLES = {'spatial': '-s', 'frequency': '-.^', 'unconditional': '--o'}
COLORS = {'blue': 450, 'green': 532, 'red': 638}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def z_scan(model, seed, color):
    d = load_json(f'{EXP}/rev_m3_{model}_s{seed}_eval_{color}/eval_results.json')
    items = sorted(d.values(), key=lambda v: v['z'])
    return np.array([v['z'] for v in items]), np.array([v['psnr_mean'] for v in items])


def wl_scan(model, seed):
    d = load_json(f'{EXP}/rev_m3_{model}_s{seed}_lambda_finescan_z150/eval_results.json')
    items = sorted(d.values(), key=lambda v: v['wavelength'])
    return (1e6 * np.array([v['wavelength'] for v in items]),
            np.array([v['psnr_mean'] for v in items]))


def fig4_zcomb_v3():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, (c, wl) in zip(axes, COLORS.items()):
        z_b, p_b = z_scan('spatial', 42, c)  # placeholder, replaced below
        d = load_json(f'{EXP}/sfoconv_baseline_{c}/eval_results.json')
        items = sorted(d.values(), key=lambda v: v['z'])
        z_b = np.array([v['z'] for v in items])
        p_b = np.array([v['psnr_mean'] for v in items])
        zs = np.stack([z_scan('spatial', s, c)[1] for s in SEEDS])
        z_x = z_scan('spatial', 42, c)[0]
        ax.plot(z_b, p_b, '--o', markersize=3, linewidth=1.2,
                label='single-point', color=GRAY)
        ax.plot(z_x, zs.mean(axis=0), '-s', markersize=3, linewidth=1.5,
                label='spatial-cond.', color=AMBER)
        ax.fill_between(z_x, zs.mean(axis=0) - zs.std(axis=0),
                        zs.mean(axis=0) + zs.std(axis=0), color=AMBER, alpha=0.12, linewidth=0)
        ax.axhline(25, color=GRAY, linestyle=':', linewidth=1, alpha=0.7)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{c.capitalize()} ({wl} nm)')
        ax.set_xlim(145, 205)
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(os.path.join(HERE, 'fig4_zcomb_v3.png'))
    plt.close(fig)
    print('saved fig4_zcomb_v3.png')


def fig5_lambdacomb_v3():
    scans = {m: [wl_scan(m, s) for s in SEEDS] for m in MODELS}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, (c, ctr) in zip(axes, COLORS.items()):
        for m in ['unconditional', 'spatial', 'frequency']:
            xs = scans[m][0][0]
            mask = (xs >= ctr - 10) & (xs <= ctr + 10)
            curves = np.stack([p[mask] for _, p in scans[m]])
            ax.plot(xs[mask], curves.mean(axis=0), STYLES[m], markersize=3,
                    linewidth=1.2, label=NAMES[m], color=MODELS[m])
            ax.fill_between(xs[mask], curves.mean(axis=0) - curves.std(axis=0),
                            curves.mean(axis=0) + curves.std(axis=0),
                            color=MODELS[m], alpha=0.10, linewidth=0)
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{c.capitalize()} @ z=150 mm')
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=9)
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(os.path.join(HERE, 'fig5_lambdacomb_v3.png'))
    plt.close(fig)
    print('saved fig5_lambdacomb_v3.png')


if __name__ == '__main__':
    fig4_zcomb_v3()
    fig5_lambdacomb_v3()
