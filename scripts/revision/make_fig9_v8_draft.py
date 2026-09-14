"""Fig.9 v8 draft (2026-09-11): v7 + axis/label fixes.

Two separate z illustrations + lambda panel. (a) ours, (b) SFO alone with three
per-color curves, (c) lambda. Panels (a) and (b) are ILLUSTRATIONS of extrapolation
behaviour, NOT a head-to-head; the quantitative head-to-head stays in Sec. 4.3
(Table 5 + Fig. 6). The M4c "ours retrained on the SFO window" curves are dropped
(their lambda*z shift validation becomes one prose sentence).

  (a) ours, main weight      z 85-265 mm, fixed-peak convention
  (b) SFO released weights   z 40-200 mm, SFO native sum-scaled convention, 3 colors
  (c) ours, full-band lambda 410-680 nm

v8 changes vs v7:
  - Panel (b) series reordered Red/Green/Blue -> Blue/Green/Red so both z panels list
    the legend in the same order as panel (a) (and as the rest of the paper, which
    writes "450/532/638 nm" and whose main figure script iterates blue/green/red).
    Ascending wavelength also matches the direction of panel (c)'s axis.
  - Panel (a) 638 nm line recoloured amber #c55a11 -> red #c00000, matching panel (b)
    so the same wavelength reads as the same colour in both panels. Amber in this
    figure is not the method accent used in the bar/comparison figures, it was acting
    as "638 nm"; this also frees amber to mean only "extrapolated" in panel (c).
  - Panel (a) x ticks anchored at the axis origin: 85,105,...,265, so the low-z
    boundary 85 and the training-window edges 145/205 are all readable. Previously
    the default locator labelled 100,120,...,260 and the origin 85 was unlabelled.
  - Panel (c) xlim tightened 408-682 -> 410-680 (the range quoted in Sec. 4.5),
    with both endpoints ticked.
  - Near-side collapse markers moved to the first z at/above 25 dB -- 89/106/125 mm
    (red/green/blue) -- so the figure matches the onsets quoted in Sec. 4.5. The old
    values (87/105/124) were the last sub-threshold points, and red's 87 mm was the
    depth of its comb valley rather than the crossing, which made the annotated
    lambda*z (55.5k) inconsistent with the 89/106/125 onsets (56.2-56.8k).
    Annotation now reads lambda*z ~= 56.5k nm.mm, inside the [55,57] quoted in text.

v8b label alignment (2026-09-11, user request: 强迫症 about the panel (c) label):
  - Panel (c) train-region label text 'training band' -> 'training window' so all
    three panels name the gray region the same way (and as Sec. 4.5 / Limitations).
  - Panel (c) label position moved from just above the 25 dB gridline (below center,
    y=26.2) to vertically centered inside the gray window at (545, 27.6), matching
    panels (a) and (b), which both center their 'training window'/'SFO window' text
    inside the shaded band near the top. Panel (a) text keeps y=36.3 (inside the
    box), panel (b) keeps y=36.3; (c) is placed at a commensurate relative height.

v8d label placement (2026-09-12, user feedback on v8c): move the labels OFF the 25 dB
dotted line (the text overlapped the dashes) and center panel (c)'s label at ~550 nm.
Final: all three labels at y = 22.5, the middle of the 20--25 band, vertically centered,
below the curves and clear of the 25 dB guide. At y = 22.5 the text box spans
~21.2--23.8 in (c) (16--35.8 range) and ~20.1--24.9 in (a)/(b) (2--38 range), so it sits
between the 25 dB line and the lower window edge without touching either, and clears the
curves (in-window min (a) 27.2, (b) 32.4, (c) 27.5). Panel (c) is centered at x = 545:
the earlier off-center x = 460 was only needed because the label was up near the curve;
dropped to y = 22.5, the window-center column is clear, so the label is centered as in
(a) and (b). Verified by text-bbox/data-point intersection: 0 curve points inside any
label box. (v8c history: labels first placed at 63% relative height, (a)/(b) y=24.7 and
(c) at 28.5, which overlapped (c)'s 27.5 dB valleys; (c) was moved to x=460, y=25.0.)

v8e reference lines (2026-09-12, user: why does panel (c) lack the vertical/horizontal
reference lines of (a) and (b)?): panel (c) now carries the same reference structure as
(a) and (b) -- the 25 dB usability threshold (horizontal dotted line, already present)
plus vertical dotted lines at the training-window edges 440 and 650 nm, matching the
window-boundary verticals of (a) (145/205 mm) and (b) (85/115 mm). Unlike (a)/(b), (c)
has no collapse-onset verticals because the wavelength curve never collapses; the window
edge verticals replace them so all three panels show the window boundary explicitly. The
25 dB line in (c) reads as "the whole curve stays above threshold", reinforcing the
graceful-extrapolation claim rather than marking a collapse that does not occur.

Caption must state that (a) and (b) use different amplitude conventions, so their
levels are not directly comparable, and point to Table 5 for the matched comparison.
Draft only.
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
SFO = '/mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp/ConditionalASMCGH_OE_Revision/experiments/m2_sfo/explore_sfo_extrapolation'
OUT = os.path.join(HERE, 'fig9_v8_draft.png')

NAVY = '#1f4e79'
GREEN = '#2e7d32'
AMBER = '#c55a11'
SFO_RED = '#c00000'
GRAY = '#595959'
BLACK = '#000000'
COLORS_MAIN = {'blue': (NAVY, 450), 'green': (GREEN, 532), 'red': (SFO_RED, 638)}
COLLAPSE_MAIN = {'blue': 125, 'green': 106, 'red': 89}   # first z at/above 25 dB (mm)
SFO3 = {'b': (NAVY, 450, 'Blue'), 'g': (GREEN, 520, 'Green'), 'r': (SFO_RED, 638, 'Red')}

for face in ['Regular', 'Bold', 'Italic', 'Bold_Italic']:
    ttf = f'/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_{face}.ttf'
    if os.path.exists(ttf):
        fm.fontManager.addfont(ttf)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'text.color': BLACK, 'axes.labelcolor': BLACK,
    'xtick.color': BLACK, 'ytick.color': BLACK,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRAY, 'grid.alpha': 0.3,
})


def load(p):
    with open(p) as f:
        return json.load(f)


def zcurve(tag):
    d = load(f'{EXP}/rev_m4_zscan_below_{tag}/eval_results.json')
    d.update(load(f'{EXP}/rev_m4_zscan_above_{tag}/eval_results.json'))
    d.update(load(f'{EXP}/rev_m3_spatial_s42_eval_{tag}/eval_results.json'))
    items = sorted(d.values(), key=lambda v: v['z'])
    return np.array([v['z'] for v in items]), np.array([v['psnr_mean'] for v in items])


def sfo_zcurve(tag):
    """Released SFO weights: coarse 5mm scan merged with the 1mm dense scan."""
    d = load(os.path.join(SFO, f'sfo_zscan_{tag}.json'))
    dense = os.path.join(SFO, f'sfo_zscan_dense_{tag}.json')
    if os.path.exists(dense):
        d.update(load(dense))
    items = sorted(d.values(), key=lambda v: v['z'])
    return (np.array([v['z'] for v in items]),
            np.array([v['psnr_mean'] for v in items]))


YLO, YHI = 2, 38.0
BBOX = dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.85)


def main():
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85], hspace=0.55)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # (a) ours, main weight -- the extrapolation illustration
    for tag, (c, wl) in COLORS_MAIN.items():
        z, p = zcurve(tag)
        ax1.plot(z, p, '-', linewidth=1.2, color=c, label=f'{tag.capitalize()} ({wl} nm)',
                 zorder=3)
    for tag, (c, wl) in COLORS_MAIN.items():
        ax1.axvline(COLLAPSE_MAIN[tag], color=c, linestyle=':', linewidth=1.0, zorder=2)
    ax1.axhline(25, color=GRAY, linestyle=':', linewidth=1, alpha=0.8)
    ax1.axvspan(145, 205, color=GRAY, alpha=0.12, linewidth=0, zorder=0)
    ax1.text(175, 22.5, 'training window', ha='center', va='center', fontsize=8.5, color=GRAY)
    ax1.text(90, 34.4, 'collapse at $\\lambda z$ $\\approx$ 56.5k nm·mm', ha='left',
             fontsize=8, color=GRAY, bbox=BBOX)
    ax1.set_xlabel('z (mm)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_xlim(85, 265)
    ax1.set_xticks(range(85, 266, 20))
    ax1.set_ylim(YLO, YHI)
    ax1.grid(True)
    ax1.legend(frameon=False, loc='lower right', ncol=3, handlelength=1.6,
               columnspacing=1.2)

    # (b) SFO released weights -- top-hat illustration, three per-color curves
    for tag, (c, wl, name) in SFO3.items():
        z, p = sfo_zcurve(tag)
        ax2.plot(z, p, '-', linewidth=1.2, color=c, zorder=3,
                 label=f'{name} ({wl} nm)')
    ax2.axhline(25, color=GRAY, linestyle=':', linewidth=1, alpha=0.8)
    ax2.axvspan(85, 115, color=GRAY, alpha=0.12, linewidth=0, zorder=0)
    for xe in (85, 115):
        ax2.axvline(xe, color=GRAY, linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)
    ax2.text(100, 22.5, 'SFO window', ha='center', va='center', fontsize=8.5, color=GRAY)
    ax2.text(122, 28.0, 'collapses within\n1\u20133 mm outside', ha='left', fontsize=8,
             color=GRAY, bbox=BBOX)
    ax2.set_xlabel('z (mm)')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_xlim(40, 200)
    ax2.set_ylim(YLO, YHI)
    ax2.grid(True)
    ax2.legend(frameon=False, loc='lower right', ncol=3, handlelength=1.6,
               columnspacing=1.2)

    # (c) ours, full-band lambda sweep
    d = load(f'{EXP}/rev_m4_lambda_dense_z150/merged_fullband_z150.json')['fullband_lambda_scan_z150']
    wl = np.array([p['wavelength_um'] for p in d['points']])
    p = np.array([p['psnr_mean'] for p in d['points']])
    extra = (wl < 440) | (wl > 650)
    ax3.plot(wl[~extra], p[~extra], '-', linewidth=1.2, color=NAVY)
    ax3.plot(wl[extra], p[extra], 'o', markersize=3, color=AMBER, label='extrapolated')
    ax3.axvspan(440, 650, color=GRAY, alpha=0.12, linewidth=0)
    ax3.axhline(25, color=GRAY, linestyle=':', linewidth=1, alpha=0.8)
    for xe in (440, 650):
        ax3.axvline(xe, color=GRAY, linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
    ax3.text(545, 22.5, 'training window', ha='center', va='center', fontsize=8.5, color=GRAY)
    ax3.legend(frameon=False, loc='lower right', handletextpad=0.3)
    ax3.set_xlabel('wavelength (nm)')
    ax3.set_ylabel('PSNR (dB)')
    ax3.set_xlim(410, 680)
    ax3.set_xticks([410, 450, 500, 550, 600, 650, 680])
    ax3.set_ylim(16, 35.8)

    fig.savefig(OUT)
    plt.close(fig)
    print('saved', OUT)


if __name__ == '__main__':
    main()
