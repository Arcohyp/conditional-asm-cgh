"""Regenerate the figures in ``benchmarks/figs`` from the recorded results.

All numbers come from the JSON files in ``benchmarks/results``; re-run the
benchmark scripts first if you want your own hardware. Only matplotlib is
required.

    python -m conditional_asm_cgh.benchmarks.make_figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conditional_asm_cgh.benchmarks.common import REPO_ROOT


RESULTS = os.path.join(REPO_ROOT, 'benchmarks', 'results')
FIGS = os.path.join(REPO_ROOT, 'benchmarks', 'figs')

C_REF = '#9aa0a6'
C_FAST = '#1a73e8'
C_ACC = '#188038'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.edgecolor': '#dadce0',
    'axes.labelcolor': '#202124',
    'axes.titlecolor': '#202124',
    'xtick.color': '#3c4043',
    'ytick.color': '#3c4043',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#eef0f2',
    'grid.linewidth': 0.8,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def fig_latency(gpu, cpu, outdir):
    panels = [(gpu, 'GPU (RTX A6000)'), (cpu, 'CPU (2x Xeon 8352V, 72 threads)')]
    panels = [(d, t) for d, t in panels if d]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.7))
    if len(panels) == 1:
        axes = [axes]

    for ax, (data, title) in zip(axes, panels):
        rows = data['rows']
        labels = [r['tag'] for r in rows]
        ref = [r['ref_ms'] for r in rows]
        fast = [r['fast_ms'] for r in rows]
        x = range(len(rows))

        ax.plot(x, ref, 'o-', color=C_REF, lw=2, ms=5, label='reference')
        ax.plot(x, fast, 'o-', color=C_FAST, lw=2, ms=5, label='fast path')
        ax.set_yscale('log')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel('resolution')
        ax.set_ylabel('latency per frame (ms, log)')
        ax.set_title(title, fontsize=10, pad=8)
        style(ax)

        for i, r in enumerate(rows):
            # geometric midpoint keeps the label between the two curves
            mid = (r['fast_ms'] * r['ref_ms']) ** 0.5
            ax.annotate('%.2fx' % r['speedup'], (i, mid),
                        textcoords='offset points', xytext=(6, 0),
                        ha='left', va='center', fontsize=8, color=C_FAST)

        ax.legend(frameon=False, fontsize=8.5, loc='upper left')

    fig.suptitle('Single-frame latency: model forward + reconstruction, batch 1',
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig1_latency_vs_resolution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig] %s' % path)


def fig_models(outdir):
    """Speed-up at 4K for the three released models."""
    models = [('freq', 'frequency-conditional\n85,588 params'),
              ('spatial', 'spatial-conditional\n44,221 params'),
              ('uncond', 'unconditional\n42,260 params')]
    labels, speedups, shares = [], [], []
    for key, label in models:
        data = load('gpu_%s.json' % key)
        if not data:
            continue
        row = [r for r in data['rows'] if r['tag'] == '4k']
        if not row:
            continue
        labels.append(label)
        speedups.append(row[0]['speedup'])
        shares.append(row[0]['ref_ms'])

    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bars = ax.bar(range(len(labels)), speedups, color=C_FAST, width=0.55)
    for b, s, ms in zip(bars, speedups, shares):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.05, '%.2fx' % s,
                ha='center', fontsize=9, color=C_FAST)
        ax.text(b.get_x() + b.get_width() / 2, 0.08, '%.1f ms' % ms,
                ha='center', fontsize=8, color='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('speed-up at 4K (GPU)')
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.set_title('Speed-up is bounded by the share of time spent in complex convolutions',
                 fontsize=9.5, pad=8)
    style(ax)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig2_speedup_per_model.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig] %s' % path)


def fig_training(data, outdir):
    if not data:
        return
    fwd, step = data['forward_ms'], data['step_ms']
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    x = [0, 1]
    w = 0.32
    d = w / 2 + 0.04
    ax.bar([i - d for i in x], fwd, width=w, color=C_REF, label='reference')
    ax.bar([i + d for i in x], step, width=w, color=C_FAST, label='fast path')
    for i, (a, b) in enumerate(zip(fwd, step)):
        ax.text(i - d, a + 0.6, '%.1f' % a, ha='center', fontsize=8, color='#3c4043')
        ax.text(i + d, b + 0.6, '%.1f' % b, ha='center', fontsize=8, color=C_FAST)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels(['forward', 'forward + backward'])
    ax.set_ylabel('ms per step')
    ax.set_title('Training step, 1080p, batch 1', fontsize=9.5, pad=8)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig3_training_step.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig] %s' % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=FIGS)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fig_latency(load('gpu_freq.json'), load('cpu_freq.json'), args.outdir)
    fig_models(args.outdir)
    fig_training(load('train_equiv.json'), args.outdir)


if __name__ == '__main__':
    main()
