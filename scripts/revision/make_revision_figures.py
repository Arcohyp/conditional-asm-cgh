"""Revision figure redraws for OE #610780.
N3: fig8_engineering -- bar value labels, fixed latency y ticks.
N5: unified variant colors across figures:
      unconditional / single-point = gray (#595959)
      spatial-conditional          = orange (#c55a11, main highlight)
      frequency-conditional        = navy (#1f4e79)
      SFO baseline                 = light gray (#b3b3b3)
Outputs go to the directory containing this script.
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
OUT = HERE

for face in ['Regular', 'Bold', 'Italic', 'Bold_Italic']:
    ttf = f'/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_{face}.ttf'
    if os.path.exists(ttf):
        fm.fontManager.addfont(ttf)

NAVY = '#1f4e79'
AMBER = '#c55a11'
GRAY = '#595959'
SFO_GRAY = '#b3b3b3'
WHITE = '#ffffff'
BLACK = '#000000'

C_UNCOND = GRAY
C_SPATIAL = AMBER
C_FREQ = NAVY
C_SFO = SFO_GRAY

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
    'axes.titlecolor': BLACK,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': WHITE,
    'axes.facecolor': WHITE,
    'figure.facecolor': WHITE,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': BLACK,
    'grid.color': GRAY,
    'grid.alpha': 0.3,
})


def apply_black_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color(BLACK)
    ax.tick_params(colors=BLACK)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    ax.title.set_color(BLACK)


def load_z_scan(path):
    with open(path) as f:
        data = json.load(f)
    items = sorted(data.values(), key=lambda v: v['z'])
    return (np.array([v['z'] for v in items]),
            np.array([v['psnr_mean'] for v in items]))


def load_lambda_scan(path):
    with open(path) as f:
        data = json.load(f)
    items = sorted(data.values(), key=lambda v: v['wavelength'])
    return (1e6 * np.array([v['wavelength'] for v in items]),
            np.array([v['psnr_mean'] for v in items]))


# ---------- N3+N5: fig8_engineering ----------
def fig8_engineering():
    models = ['spatial-cond.', 'uncond.', 'freq-cond.', 'SFO']
    colors = [C_SPATIAL, C_UNCOND, C_FREQ, C_SFO]
    params = np.array([44.2e3, 42.3e3, 85.6e3, 109.15e6])
    weights = np.array([0.20, 0.19, 0.37, 436.7])
    latency = np.array([72.5, 51.3, 56.1, 384.6])
    vram = np.array([1.47, 0.92, 0.93, 11.4])
    labels = [['44.2k', '42.3k', '85.6k', '109.2M'],
              ['0.20', '0.19', '0.37', '436.7'],
              ['72.5', '51.3', '56.1', '384.6'],
              ['1.47', '0.92', '0.93', '11.4']]
    titles = ['Parameters', 'Weight size', 'Latency', 'Peak VRAM']
    units = ['', 'MB', 'ms', 'GB']
    scales = ['log', 'log', 'linear', 'log']
    yticks = [[1e5, 1e6, 1e7, 1e8], [0.2, 2, 20, 200], None, [1, 2, 4, 8]]
    yticklabels = [['0.1M', '1M', '10M', '100M'], ['0.2', '2', '20', '200'], None, ['1', '2', '4', '8']]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7))
    x = np.arange(len(models))
    for ax, vals, labs, title, unit, sc, yt, ytl in zip(
            axes, [params, weights, latency, vram], labels, titles, units, scales, yticks, yticklabels):
        bars = ax.bar(x, vals, color=colors, edgecolor=BLACK, linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_yscale(sc)
        if sc == 'log':
            ax.set_yticks(yt)
            ax.set_yticklabels(ytl)
            ax.set_ylim(vals.min() * 0.85, vals.max() * 1.9)
        else:
            ax.set_yticks([0, 100, 200, 300, 400])
            ax.set_ylim(0, vals.max() * 1.18)
        ax.set_title(title)
        ax.set_ylabel(f'{unit}' if unit else title)
        ax.grid(True, axis='y')
        if sc == 'log':
            ymin, ymax = ax.get_ylim()
            for bar, v, lab in zip(bars, vals, labs):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15, lab,
                        ha='center', va='bottom', fontsize=7.5, color=BLACK)
        else:
            for bar, v, lab in zip(bars, vals, labs):
                ax.text(bar.get_x() + bar.get_width() / 2, v + vals.max() * 0.02, lab,
                        ha='center', va='bottom', fontsize=7.5, color=BLACK)
        apply_black_axes(ax)
    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, 'fig8_engineering_v3.png'))
    plt.close(fig)
    print('saved fig8_engineering_v3.png')


# ---------- N5: fig4_zcomb ----------
def fig4_zcomb():
    colors = ['blue', 'green', 'red']
    native_wl = {'blue': 450, 'green': 532, 'red': 638}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, c in zip(axes, colors):
        z_b, p_b = load_z_scan(f'{EXP}/sfoconv_baseline_{c}/eval_results.json')
        z_a, p_a = load_z_scan(f'{EXP}/ablation_A3_spatialcond_eval_{c}/eval_results.json')
        ax.plot(z_b, p_b, '--o', markersize=3, linewidth=1.2,
                label='single-point', color=C_UNCOND)
        ax.plot(z_a, p_a, '-s', markersize=3, linewidth=1.5,
                label='spatial-cond.', color=C_SPATIAL)
        ax.axhline(25, color=C_UNCOND, linestyle=':', linewidth=1, alpha=0.7)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{c.capitalize()} ({native_wl[c]} nm)')
        ax.set_xlim(145, 205)
        ax.grid(True)
        apply_black_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(os.path.join(OUT, 'fig4_zcomb_v2.png'))
    plt.close(fig)
    print('saved fig4_zcomb_v2.png')


# ---------- N5: fig5_lambdacomb ----------
def fig5_lambdacomb():
    colors = ['blue', 'green', 'red']
    centers = {'blue': 450, 'green': 532, 'red': 638}
    palette = {'uncond.': C_UNCOND, 'spatial-cond.': C_SPATIAL, 'freq-cond.': C_FREQ}
    styles = {'uncond.': '--o', 'spatial-cond.': '-s', 'freq-cond.': '-.^'}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    wl1, p1 = load_lambda_scan(f'{EXP}/ablation_A1_lambda_finescan_z150/eval_results.json')
    wl3, p3 = load_lambda_scan(f'{EXP}/step3d_cond_contlambda_peak2_lambda_finescan_z150/eval_results.json')
    wl_f, pf = load_lambda_scan(f'{EXP}/npcc_lambda_finescan_z150/eval_results.json')
    for ax, c in zip(axes, colors):
        ctr = centers[c]
        for wl, p, name in [(wl1, p1, 'uncond.'), (wl3, p3, 'spatial-cond.'),
                            (wl_f, pf, 'freq-cond.')]:
            mask = (wl >= ctr - 10) & (wl <= ctr + 10)
            ax.plot(wl[mask], p[mask], styles[name], markersize=3, linewidth=1.2,
                    label=name, color=palette[name])
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{c.capitalize()} @ z=150 mm')
        ax.grid(True)
        apply_black_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=9)
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(os.path.join(OUT, 'fig5_lambdacomb_v2.png'))
    plt.close(fig)
    print('saved fig5_lambdacomb_v2.png')


if __name__ == '__main__':
    fig8_engineering()
    fig4_zcomb()
    fig5_lambdacomb()
