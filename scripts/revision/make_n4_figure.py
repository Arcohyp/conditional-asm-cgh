"""N4 figure: full-color RGB composites at off-peak distances vs target.
Three-panel figure for the revised manuscript (Sec. Reconstruction quality):
  (a) target RGB (DIV2K 0801)
  (b) paper's spatial-conditional model (M3 seed 42, z in [145,205] mm,
      pitch 3.6 um) at z=198 mm, per-channel 27.5/29.7/24.8 dB (B/G/R)
  (c) SFO-window model (pitch 3.74 um, design z=100) at z=113 mm,
      per-channel 18.0/28.9/29.3 dB
PSNRs are in SFO's intensity-domain convention (per-image sum scaling).
Source panels: composite/rgb_composite_{main_z198,m2_z113}.png + target_rgb.png
Output: ../figs_n2n3n5/fig11_rgb_composite.png (300 dpi, 5.3 in wide).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

BLACK = '#000000'
for face in ['Regular', 'Bold', 'Italic', 'Bold_Italic']:
    ttf = f'/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_{face}.ttf'
    if os.path.exists(ttf):
        fm.fontManager.addfont(ttf)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'text.color': BLACK,
    'axes.labelcolor': BLACK,
    'xtick.color': BLACK,
    'ytick.color': BLACK,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

PANELS = [
    ('target_rgb.png', '(a) Target', None),
    ('rgb_composite_main_z198.png', '(b) Main model, $z=198$ mm', (27.5, 29.7, 24.8)),
    ('rgb_composite_m2_z113.png', '(c) SFO-window model, $z=113$ mm', (18.0, 28.9, 29.3)),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(5.3, 1.75))
    for ax, (fname, tag, psnr) in zip(axes, PANELS):
        im = Image.open(os.path.join(HERE, 'composite', fname))
        ax.imshow(im)
        ax.set_title(tag, fontsize=8, pad=3)
        ax.axis('off')
        if psnr is not None:
            ax.text(0.5, -0.06,
                    'B/G/R: %.1f/%.1f/%.1f dB' % psnr,
                    transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
    plt.tight_layout(pad=0.4)
    out = os.path.join(HERE, '..', 'figs_n2n3n5', 'fig11_rgb_composite.png')
    fig.savefig(out)
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
