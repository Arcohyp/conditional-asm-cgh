"""Generate publication-quality figures for the A3-anchored OE paper.
Color/font scheme: academic twilight-discrete, Times New Roman.
"""
import sys, os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from skimage.metrics import structural_similarity as compute_ssim

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)
PANEL_DIR = os.path.join(FIG_DIR, 'fig1_panels')
os.makedirs(PANEL_DIR, exist_ok=True)

# Make system Times New Roman available inside this conda env.
import matplotlib.font_manager as fm
for face in ['Regular', 'Bold', 'Italic', 'Bold_Italic']:
    ttf = f'/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_{face}.ttf'
    if os.path.exists(ttf):
        fm.fontManager.addfont(ttf)

# ---------- color / font scheme ----------
NAVY = '#1f4e79'
AMBER = '#c55a11'
GRAY = '#595959'
WHITE = '#ffffff'
BLACK = '#000000'


def set_default_style(axis_color=BLACK):
    """Apply the default figure style. Axis/label text uses axis_color."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'mathtext.cal': 'Times New Roman',
        'mathtext.sf': 'Times New Roman',
        'mathtext.tt': 'Times New Roman',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.color': BLACK,
        'axes.labelcolor': axis_color,
        'xtick.color': axis_color,
        'ytick.color': axis_color,
        'axes.titlecolor': BLACK,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': WHITE,
        'axes.facecolor': WHITE,
        'figure.facecolor': WHITE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': axis_color,
        'grid.color': GRAY,
        'grid.alpha': 0.3,
    })


set_default_style(axis_color=BLACK)


def twilight_discrete(n, shift=0.0):
    """Sample n uniformly spaced colors from the twilight cyclic colormap.
    A phase offset avoids landing on the dark wrap boundaries at 0/0.5.
    """
    cmap = matplotlib.colormaps['twilight']
    offset = 0.15
    return [to_hex(cmap(((i + 0.5) / n + offset + shift) % 1.0)) for i in range(n)]


# Semantic palette for small-N line/bar charts (readable on screen/print).
# Green is intentionally excluded because the twilight colormap does not contain it.
C_PRIMARY = NAVY
C_SECONDARY = AMBER
C_TERTIARY = twilight_discrete(4)[0]   # blue-purple for a third category
C_NEUTRAL = GRAY


def load_z_scan(path):
    with open(path) as f:
        data = json.load(f)
    items = sorted(data.values(), key=lambda v: v['z'])
    zs = np.array([v['z'] for v in items])
    ps = np.array([v['psnr_mean'] for v in items])
    return zs, ps


def load_lambda_scan(path):
    with open(path) as f:
        data = json.load(f)
    items = sorted(data.values(), key=lambda v: v['wavelength'])
    wls = np.array([v['wavelength'] for v in items])
    ps = np.array([v['psnr_mean'] for v in items])
    return 1e6 * wls, ps


def load_quant(path):
    with open(path) as f:
        return json.load(f)


def overlay_metric(ax, psnr, ssim_val, x=0.97, y=0.03, fontsize=7):
    """Overlay PSNR + SSIM on the image with a translucent black background."""
    text = f'PSNR {psnr:.1f} dB\nSSIM {ssim_val:.3f}'
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            color='white', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='black',
                      alpha=0.55, edgecolor='none'))


# ---------- Fig 1: teaser panels saved individually ----------
def apply_black_axes(ax):
    """Force all axis elements on ax to black."""
    for spine in ax.spines.values():
        spine.set_color(BLACK)
    ax.tick_params(axis='both', colors=BLACK)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    ax.title.set_color(BLACK)


def _make_panel_figure(title, im, fname):
    """Save a single Fig.1 panel with black title text."""
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(im, cmap='gray')
    ax.set_title(title, color=BLACK)
    ax.axis('off')
    fig.savefig(os.path.join(PANEL_DIR, fname))
    plt.close(fig)
    print(f'saved {fname}')


def fig1_teaser():
    recon_dir = '../paper_assets/recon'
    target = np.load(os.path.join(recon_dir, 'blue_target.npy'))
    std = np.load(os.path.join(recon_dir, 'blue_baseline_peak_z150.npy'))
    z_shift = np.load(os.path.join(recon_dir, 'blue_baseline_valley_z197.npy'))
    wl_shift = np.load(os.path.join(recon_dir, 'green_baseline_valley_z198.npy'))

    z_path = '../../baseline_generalization_exp/eval_results/diag_green_comb150/eval_results.json'
    wl_path = '../../baseline_generalization_exp/eval_results/wlcomb_green_100samp/eval_results.json'
    z_data = load_z_scan(z_path)
    wl_data = load_lambda_scan(wl_path)

    # Four amplitude images
    images = [
        ('fig1a_target.png', 'Target', target),
        ('fig1b_standard.png', 'Standard point', std),
        ('fig1c_lambda_shift.png', '$\\lambda$ shift', wl_shift),
        ('fig1d_z_shift.png', '$z$ shift', z_shift),
    ]
    for fname, title, im in images:
        _make_panel_figure(title, im, fname)

    # z-scan line plot
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(z_data[0], z_data[1], '-', linewidth=1.2, color=C_PRIMARY)
    ax.axhline(25, color=C_NEUTRAL, linestyle=':', linewidth=1)
    ax.axvline(150, color=AMBER, linestyle=':', linewidth=0.9)
    ax.set_xlim(145, 205)
    ax.set_ylim(15, 36)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Distance scan (1 mm step), green')
    ax.grid(True)
    apply_black_axes(ax)
    fig.savefig(os.path.join(PANEL_DIR, 'fig1e_zscan.png'))
    plt.close(fig)
    print('saved fig1e_zscan.png')

    # wavelength-scan line plot
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    GREEN = '#2e7d32'
    ax.plot(wl_data[0], wl_data[1], '-', linewidth=1.2, color=GREEN)
    ax.axhline(25, color=C_NEUTRAL, linestyle=':', linewidth=1)
    ax.axvline(532, color=GREEN, linestyle=':', linewidth=0.9)
    ax.set_xlim(500, 560)
    ax.set_ylim(15, 36)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Wavelength scan (1 nm step), green')
    ax.grid(True)
    apply_black_axes(ax)
    fig.savefig(os.path.join(PANEL_DIR, 'fig1f_wlscan.png'))
    plt.close(fig)
    print('saved fig1f_wlscan.png')


# ---------- Fig 2: method diagram ----------
def fig2_method():
    fig = plt.figure(figsize=(7.4, 2.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis('off')

    box_colors = twilight_discrete(5)
    c_input, c_enc, c_fusion, c_back, c_out = box_colors

    def box(x, y, w, h, text, color='white', fs=8):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=NAVY,
                             linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, color=NAVY, zorder=3)

    box(0.2, 3.0, 1.6, 0.7, '$z, \\lambda, \\mathrm{pitch}$', c_input, fs=9)
    box(0.2, 1.9, 1.6, 0.9, 'ParamEncoder\n3$\\to$32$\\to$16', c_enc, fs=8)
    box(2.4, 3.0, 1.4, 0.7, 'Target amp\n$1\\times H\\times W$', c_input, fs=8)
    box(2.4, 1.9, 1.4, 0.9, 'broadcast\n$16\\times H\\times W$', c_enc, fs=8)
    box(4.2, 1.9, 1.6, 1.0, 'input\\_fusion\nConv(17$\\to$8$\\to$1)', c_fusion, fs=8)
    box(6.1, 2.8, 1.3, 0.8, 'CCNN1', c_back, fs=9)
    box(6.1, 1.3, 1.3, 0.8, 'ASM', c_back, fs=9)
    box(8.0, 1.3, 1.3, 0.8, 'CCNN2', c_back, fs=9)
    box(8.0, 2.8, 1.3, 0.8, 'Holo\nphase', c_out, fs=9)

    arrows = [
        (1.0, 3.0, 1.0, 2.8), (1.0, 1.9, 1.0, 1.55), (1.0, 1.55, 3.1, 1.55),
        (3.1, 3.0, 3.1, 2.8), (3.1, 1.9, 3.1, 1.55), (4.0, 1.55, 4.2, 1.7),
        (5.8, 2.45, 6.1, 3.0), (6.75, 2.8, 6.75, 2.15), (6.75, 1.3, 6.75, 0.9),
        (7.45, 0.9, 8.65, 0.9), (8.65, 0.9, 8.65, 1.3), (7.45, 2.8, 8.0, 3.1)
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=NAVY, lw=1, zorder=1))

    ax.text(5.0, 0.35, 'Total trainable parameters: 44,221',
            ha='center', fontsize=10, style='italic', color=BLACK)
    fig.savefig(os.path.join(FIG_DIR, 'fig2_method.png'))
    plt.close(fig)
    print('saved fig2_method.png')


# ---------- Fig 3: z comb fill ----------
def fig3_zcomb():
    colors = ['blue', 'green', 'red']
    native_wl = {'blue': 450, 'green': 532, 'red': 638}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, c in zip(axes, colors):
        z_b, p_b = load_z_scan(f'../experiments/sfoconv_baseline_{c}/eval_results.json')
        z_a, p_a = load_z_scan(f'../experiments/ablation_A3_spatialcond_eval_{c}/eval_results.json')
        ax.plot(z_b, p_b, '--o', markersize=3, linewidth=1.2,
                label='single-point', color=C_PRIMARY)
        ax.plot(z_a, p_a, '-s', markersize=3, linewidth=1.5,
                label='spatial-cond.', color=C_SECONDARY)
        ax.axhline(25, color=C_NEUTRAL, linestyle=':', linewidth=1)
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
    fig.savefig(os.path.join(FIG_DIR, 'fig3_zcomb.png'))
    plt.close(fig)
    print('saved fig3_zcomb.png')


# ---------- Fig 4: lambda comb ablation ----------
def fig4_lambdacomb():
    colors = ['blue', 'green', 'red']
    centers = {'blue': 450, 'green': 532, 'red': 638}
    names = ['uncond.', 'spatial-cond.', 'freq-cond.']
    palette = {'uncond.': C_PRIMARY, 'spatial-cond.': C_SECONDARY, 'freq-cond.': C_TERTIARY}
    styles = {'uncond.': '--o', 'spatial-cond.': '-s', 'freq-cond.': '-.^'}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, c in zip(axes, colors):
        wl1, p1 = load_lambda_scan('../experiments/ablation_A1_lambda_finescan_z150/eval_results.json')
        wl3, p3 = load_lambda_scan('../experiments/step3d_cond_contlambda_peak2_lambda_finescan_z150/eval_results.json')
        wl_f, pf = load_lambda_scan('../experiments/npcc_lambda_finescan_z150/eval_results.json')
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
    apply_black_axes(ax)
    fig.savefig(os.path.join(FIG_DIR, 'fig4_lambdacomb.png'))
    plt.close(fig)
    print('saved fig4_lambdacomb.png')


# ---------- Fig 5: quantization ----------
def fig5_quant():
    data = load_quant('../experiments/c1_quant_A3/c1_quant_results.json')
    colors = ['blue', 'green', 'red']
    bits = [8, 6, 4]
    bit_labels = ['8-bit', '6-bit', '4-bit']
    bar_colors = [C_NEUTRAL, AMBER, C_PRIMARY]  # 8-bit gray, 6-bit amber, 4-bit navy
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    for ax, c in zip(axes, colors):
        key = f'{c}_z150.0'
        cont = data[key]['cont']['psnr_mean']
        drops = [data[key][f'{b}bit']['psnr_mean'] - cont for b in bits]
        bars = ax.bar(bit_labels, drops, color=bar_colors, edgecolor=NAVY, linewidth=0.6)
        ax.axhline(0, color=NAVY, linewidth=0.8)
        ax.set_ylabel('PSNR drop (dB)')
        ax.set_title(f'{c.capitalize()} @ z=150 mm')
        ax.set_ylim(-6, 0.7)
        ax.grid(True, axis='y')
        apply_black_axes(ax)
        for bar, d in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width() / 2, max(d, 0) + 0.12, f'{d:.2f}',
                    ha='center', va='bottom', fontsize=8, color=BLACK)
    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG_DIR, 'fig5_quant.png'))
    plt.close(fig)
    print('saved fig5_quant.png')


# ---------- Fig 6: quantization ----------
# (kept in source order before reconstruction figures)


# ---------- Fig 7: content generalization (green valley, multiple samples) ----------
def fig7_content_recon():
    recon_dir = '../paper_assets/recon'
    psnr_record = load_quant(os.path.join(recon_dir, 'psnr_record.json'))

    color = 'green'
    wl_nm = 532
    zv = 198
    indices = [801, 810, 820, 830]
    col_titles = ['Target amplitude', 'single-point', 'spatial-cond.']

    fig, axes = plt.subplots(len(indices), 3, figsize=(6.4, 4.4))
    fig.subplots_adjust(hspace=0.04, wspace=0.06)
    for i, img_idx in enumerate(indices):
        target = np.load(os.path.join(recon_dir, f'img{img_idx}_target_amp.npy'))
        base_valley = np.load(os.path.join(recon_dir, f'{color}_img{img_idx}_baseline_valley_z{zv}.npy'))
        a3_valley = np.load(os.path.join(recon_dir, f'{color}_img{img_idx}_ours_valley_z{zv}.npy'))

        pb = psnr_record[f'{color}_img{img_idx}_baseline_valley_z{zv}']
        pa = psnr_record[f'{color}_img{img_idx}_ours_valley_z{zv}']
        sb = compute_ssim(target, base_valley, data_range=1.0)
        sa = compute_ssim(target, a3_valley, data_range=1.0)

        imgs = [target, base_valley, a3_valley]
        psnrs = [None, pb, pa]
        ssims = [None, sb, sa]
        for j, (im, pval, sval) in enumerate(zip(imgs, psnrs, ssims)):
            ax = axes[i, j]
            ax.imshow(im, cmap='gray', vmin=0.0, vmax=1.0)
            ax.axis('off')
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9, pad=4)
            if pval is not None and sval is not None:
                overlay_metric(ax, pval, sval)

    for ax in axes.flat:
        apply_black_axes(ax)
    fig.savefig(os.path.join(FIG_DIR, 'fig7_content_recon.png'))
    plt.close(fig)
    print('saved fig7_content_recon.png')


# ---------- Fig 8: parameter generalization (one sample, three colors, peak+valley) ----------
def fig8_param_recon():
    recon_dir = '../paper_assets/recon'
    psnr_record = load_quant(os.path.join(recon_dir, 'psnr_record.json'))

    img_idx = 809
    rows = [
        ('blue', 450, 197),
        ('green', 532, 198),
        ('red', 638, 154),
    ]
    col_titles = ['Target', 'single-point\npeak', 'spatial-cond.\npeak', 'spatial-cond.\nvalley']

    fig, axes = plt.subplots(3, 4, figsize=(7.4, 3.4))
    fig.subplots_adjust(hspace=0.04, wspace=0.08, top=0.86)
    for i, (color, wl_nm, zv) in enumerate(rows):
        target = np.load(os.path.join(recon_dir, f'img{img_idx}_target_amp.npy'))
        base_peak = np.load(os.path.join(recon_dir, f'{color}_img{img_idx}_baseline_peak_z150.npy'))
        a3_peak = np.load(os.path.join(recon_dir, f'{color}_img{img_idx}_ours_peak_z150.npy'))
        a3_valley = np.load(os.path.join(recon_dir, f'{color}_img{img_idx}_ours_valley_z{zv}.npy'))

        pp_sp = psnr_record[f'{color}_img{img_idx}_baseline_peak_z150']
        pp_ours = psnr_record[f'{color}_img{img_idx}_ours_peak_z150']
        pv_ours = psnr_record[f'{color}_img{img_idx}_ours_valley_z{zv}']

        sp_ssim = compute_ssim(target, base_peak, data_range=1.0)
        op_ssim = compute_ssim(target, a3_peak, data_range=1.0)
        ov_ssim = compute_ssim(target, a3_valley, data_range=1.0)

        imgs = [target, base_peak, a3_peak, a3_valley]
        psnrs = [None, pp_sp, pp_ours, pv_ours]
        ssims = [None, sp_ssim, op_ssim, ov_ssim]
        for j, (im, pval, sval) in enumerate(zip(imgs, psnrs, ssims)):
            ax = axes[i, j]
            ax.imshow(im, cmap='gray', vmin=0.0, vmax=1.0)
            ax.axis('off')
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9, pad=4)
            if pval is not None and sval is not None:
                overlay_metric(ax, pval, sval)
            if j == 0:
                ax.text(-0.18, 0.5, f'{color.capitalize()}\n{wl_nm} nm',
                        transform=ax.transAxes, rotation=90, va='center', ha='center',
                        fontsize=9, color=BLACK)

    for ax in axes.flat:
        apply_black_axes(ax)
    fig.savefig(os.path.join(FIG_DIR, 'fig8_param_recon.png'))
    plt.close(fig)
    print('saved fig8_param_recon.png')


# ---------- Fig 7: engineering comparison ----------
def fig7_engineering():
    # SFO numbers measured on the same 4K rig (per-color weight).
    models = ['spatial-cond.', 'uncond.', 'freq-cond.', 'SFO']
    params = np.array([44.2e3, 42.3e3, 85.6e3, 109.15e6])
    weights = np.array([0.20, 0.19, 0.37, 436.7])
    latency = np.array([72.5, 51.3, 56.1, 384.6])
    vram = np.array([1.47, 0.92, 0.93, 11.4])

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6))
    x = np.arange(len(models))
    colors = twilight_discrete(4)

    for ax, vals, title, unit in zip(axes, [params, weights, latency, vram],
                                      ['Parameters', 'Weight size', 'Latency', 'Peak VRAM'],
                                      ['', 'MB', 'ms', 'GB']):
        ax.bar(x, vals, color=colors, edgecolor=NAVY, linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, axis='y')
        apply_black_axes(ax)

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG_DIR, 'fig7_engineering.png'))
    plt.close(fig)
    print('saved fig7_engineering.png')


if __name__ == '__main__':
    fig1_teaser()
    fig2_method()
    fig3_zcomb()
    fig4_lambdacomb()
    fig5_quant()
    fig7_content_recon()
    fig8_param_recon()
    print('all figures done')
