"""N2: rebuild Fig. 1 (fig1_motivation) with larger fonts in panels (b)(c)(d) and
explicit valley markers. Reuses the original top strip (panel (a)) and redraws
the bottom row from data:
  (b) z scan (green single-point baseline, DIV2K-valid mean) + z-shift recon
  (c) lambda scan + lambda-shift recon
  (d) dense 2-D (z, lambda) PSNR map (from remap_2d.py)
Valley points below 25 dB are NOT marked here -- the user adds the markers
manually during figure assembly.

Colors follow the ORIGINAL fig1 scheme: scan curves in navy #1f4e79
(sampled from paper_oe/figures/fig1_panels/fig1e_zscan.png), 25 dB and
design-point reference lines in dotted gray.

Outputs (next to this script):
  fig1_motivation_v2.png   full composed figure (for reference)
  fig1_bottom_v2.png       bottom row only (for reference)
  fig1b_zscan_v2.png       standalone panel (b) plot  <-- for manual assembly
  fig1c_wlscan_v2.png      standalone panel (c) plot  <-- for manual assembly
  fig1d_map_v2.png         standalone panel (d) 2D map with colorbar
The two recon insets are the UNCHANGED original assets
(paper_oe/figures/fig1_panels/fig1d_z_shift.png, fig1c_lambda_shift.png),
only title-cropped; reuse them as-is.

Usage: python make_fig1_v2.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp'
ORIG_FIG1 = os.path.join(REPO, 'paper_oe/figures/fig1_motivation.png')
PANELS = os.path.join(REPO, 'paper_oe/figures/fig1_panels')
Z_JSON = ('/mnt/sda/chengxirun/hologram_task/expanded_work/baseline_generalization_exp/'
          'eval_results/diag_green_comb150/eval_results.json')
WL_JSON = ('/mnt/sda/chengxirun/hologram_task/expanded_work/baseline_generalization_exp/'
           'eval_results/wlcomb_green_100samp/eval_results.json')
MAP_JSON = os.path.join(REPO, 'experiments/rev_n2_grid2d_green/eval_results.json')

NAVY = '#1f4e79'   # original fig1 scan-curve color (sampled from fig1e_zscan.png)
GRAY = '#595959'
RED = '#d62728'
BLACK = '#000000'

for face in ['Regular', 'Bold', 'Italic', 'Bold_Italic']:
    ttf = f'/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_{face}.ttf'
    if os.path.exists(ttf):
        fm.fontManager.addfont(ttf)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_z(path):
    d = json.load(open(path))
    items = sorted(d.values(), key=lambda v: v['z'])
    return np.array([v['z'] for v in items]), np.array([v['psnr_mean'] for v in items])


def load_wl(path):
    d = json.load(open(path))
    items = sorted(d.values(), key=lambda v: v['wavelength'])
    return 1e6 * np.array([v['wavelength'] for v in items]), np.array([v['psnr_mean'] for v in items])


def circle_valleys(ax, x, y, thr=25.0):
    """Circle local minima below thr in red. (Unused: user marks valleys
    manually during assembly; kept for reference.)"""
    for i in range(1, len(y) - 1):
        if y[i] < thr and y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            ax.plot(x[i], y[i], 'o', ms=9, mfc='none', mec=RED, mew=1.6, zorder=5)


def plot_zscan(ax):
    z, pz = load_z(Z_JSON)
    ax.plot(z, pz, '-', linewidth=1.3, color=NAVY)
    ax.axhline(25, color=GRAY, linestyle=':', linewidth=1.1)
    ax.axvline(150, color=GRAY, linestyle=':', linewidth=1.0)
    ax.set_xlim(145, 205)
    ax.set_ylim(14, 36)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('z scan, green', fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_wlscan(ax):
    wl, pw = load_wl(WL_JSON)
    ax.plot(wl, pw, '-', linewidth=1.3, color=NAVY)
    ax.axhline(25, color=GRAY, linestyle=':', linewidth=1.1)
    ax.axvline(532, color=GRAY, linestyle=':', linewidth=1.0)
    ax.set_xlim(520, 544)
    ax.set_ylim(14, 36)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(r'$\lambda$ scan, green', fontsize=11)
    ax.grid(True, alpha=0.3)


def load_map():
    d = json.load(open(MAP_JSON))
    zs = sorted({v['z'] for v in d.values()})
    wls = sorted({v['wavelength'] for v in d.values()})
    P = np.full((len(wls), len(zs)), np.nan)
    for v in d.values():
        P[wls.index(v['wavelength']), zs.index(v['z'])] = v['psnr_mean']
    return np.array(zs), 1e6 * np.array(wls), P


def plot_map(ax, with_colorbar=True):
    zs, wls, P = load_map()
    im = ax.imshow(P, origin='lower', cmap='viridis', aspect='auto',
                   extent=[min(zs), max(zs), min(wls), max(wls)],
                   vmin=15, vmax=35, interpolation='nearest')
    ax.axvline(150, color='white', linestyle=':', linewidth=1.0)
    ax.axhline(532, color='white', linestyle=':', linewidth=1.0)
    ax.set_xlabel('z (mm)')
    ax.set_title('single-point baseline', fontsize=11)
    cb = None
    if with_colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.055)
        cb.set_label('PSNR (dB)', fontsize=10.5)
        cb.ax.tick_params(labelsize=9.5)
    return im


def load_panel(fname):
    im = Image.open(os.path.join(PANELS, fname)).convert('L')
    w, h = im.size
    im = im.crop((0, int(h * 0.25), w, int(h * 0.97)))  # strip baked-in title/labels
    return np.asarray(im, dtype=float) / 255.0


def standalone_panels():
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    plot_zscan(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(HERE, 'fig1b_zscan_v2.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    plot_wlscan(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(HERE, 'fig1c_wlscan_v2.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    plot_map(ax, with_colorbar=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(HERE, 'fig1d_map_v2.png'))
    plt.close(fig)
    print('saved standalone panels fig1b/fig1c/fig1d _v2.png')


def composed():
    img_z = load_panel('fig1d_z_shift.png')
    img_wl = load_panel('fig1c_lambda_shift.png')

    fig = plt.figure(figsize=(7.4, 2.35))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.15, 0.85, 1.15, 0.85, 1.45],
                          wspace=0.30, left=0.065, right=0.965, top=0.86, bottom=0.20)

    ax = fig.add_subplot(gs[0])
    plot_zscan(ax)

    ax = fig.add_subplot(gs[1])
    ax.imshow(img_z, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax.set_title('z shift', fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[2])
    plot_wlscan(ax)

    ax = fig.add_subplot(gs[3])
    ax.imshow(img_wl, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax.set_title(r'$\lambda$ shift', fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[4])
    plot_map(ax, with_colorbar=True)

    fig.savefig(os.path.join(HERE, 'fig1_bottom_v2.png'))
    plt.close(fig)

    orig = Image.open(ORIG_FIG1).convert('RGB')
    W, H = orig.size
    a = np.array(orig.convert('L'))
    dark = a < 100
    rowfrac = dark.mean(axis=1)
    sep = int(np.argmax(rowfrac[: int(H * 0.7)] > 0.25))  # dashed divider row
    top = orig.crop((0, 0, W, sep - 6))

    bottom = Image.open(os.path.join(HERE, 'fig1_bottom_v2.png')).convert('RGB')
    bw = W
    bh = int(bottom.height * bw / bottom.width)
    bottom = bottom.resize((bw, bh), Image.LANCZOS)

    canvas = Image.new('RGB', (W, top.height + 18 + bh), 'white')
    canvas.paste(top, (0, 0))
    dr = ImageDraw.Draw(canvas)
    for x in range(0, W, 28):
        dr.line([(x, top.height + 8), (x + 16, top.height + 8)], fill='black', width=4)
    canvas.paste(bottom, (0, top.height + 18))

    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf', 64)
    dr = ImageDraw.Draw(canvas)
    y0 = top.height + 18
    dr.text((int(W * 0.015), y0 + int(bh * 0.04)), '(b)', fill='black', font=font)
    dr.text((int(W * 0.36), y0 + int(bh * 0.04)), '(c)', fill='black', font=font)
    dr.text((int(W * 0.70), y0 + int(bh * 0.04)), '(d)', fill='black', font=font)

    canvas.save(os.path.join(HERE, 'fig1_motivation_v2.png'))
    print('saved fig1_motivation_v2.png', canvas.size)


if __name__ == '__main__':
    standalone_panels()
    composed()
