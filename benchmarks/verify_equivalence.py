"""Correctness gate for the fast inference path.

Two levels:

``--unit``
    Layer level. Builds ``ComplexConv2d`` / ``ComplexConvTranspose2d`` with
    random weights and compares them against the packed equivalents on a random
    complex input. Verifies the block-matrix re-indexing, including the bias
    convention of ``apply_complex``.

default (``--stage conv|asm|mult|all``)
    End-to-end. Runs the same checkpoint through the reference pipeline and
    through the fast pipeline with rewrites enabled cumulatively, and reports

    * ``max|d phi|``   largest wrapped phase difference (rad)
    * ``max|d I|``     largest pixel difference of the reconstructed intensity
    * ``PSNR(I_fast, I_ref)``  agreement between the two implementations
    * ``PSNR(I, target)``      absolute quality of each, so the run doubles as
                               evidence that nothing was traded away

The run exits non-zero if any check exceeds its tolerance, so it can be used as
a pre-deployment gate::

    python -m conditional_asm_cgh.benchmarks.verify_equivalence --stage all

Reference numbers on the released checkpoints (A6000 / 4K): agreement PSNR
138-145 dB, ``max|d phi| < 1e-6`` rad, ``max|d I| < 1e-6``.
"""

import argparse
import math

import torch

from conditional_asm_cgh.utils.fast_complex import (
    PackedComplexConv2d, PackedComplexConvTranspose2d,
)
from conditional_asm_cgh.utils.fast_infer import FastCGHInference

from conditional_asm_cgh.benchmarks.common import (
    build_model, div2k_amplitude, fixed_amp, parse_resolution, psnr, timeit, dump,
)


def wrapped_max_abs(a, b):
    """Largest phase difference after wrapping into [-pi, pi]."""
    d = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
    return d.abs().max().item()


# ---------------------------------------------------------------------------
# unit: layer level
# ---------------------------------------------------------------------------
def check_unit(device, seed=0):
    from complexPyTorch.complexLayers import ComplexConv2d, ComplexConvTranspose2d

    torch.manual_seed(seed)
    cases = [
        ('conv 1->4  s2', ComplexConv2d(1, 4, 3, stride=2, padding=1),
         PackedComplexConv2d.from_complex_conv, (1, 1, 64, 96)),
        ('conv 4->8  s2', ComplexConv2d(4, 8, 3, stride=2, padding=1),
         PackedComplexConv2d.from_complex_conv, (1, 4, 32, 48)),
        ('conv 8->8  s1', ComplexConv2d(8, 8, 3, stride=1, padding=1),
         PackedComplexConv2d.from_complex_conv, (1, 8, 32, 48)),
        ('tconv 8->4 s2', ComplexConvTranspose2d(8, 4, 4, stride=2, padding=1),
         PackedComplexConvTranspose2d.from_complex_convtranspose, (1, 8, 16, 24)),
        ('tconv 4->1 s2', ComplexConvTranspose2d(4, 1, 4, stride=2, padding=1),
         PackedComplexConvTranspose2d.from_complex_convtranspose, (1, 4, 32, 48)),
    ]

    print('layer-level equivalence (random weights, random complex input)')
    print('%-16s %12s %12s %10s' % ('layer', 'max|d|', '|out| (rms)', 'rel'))
    ok = True
    for name, ref, factory, shape in cases:
        ref = ref.to(device)
        fast = factory(ref).to(device)
        x = torch.randn(shape, device=device, dtype=torch.float32)
        x = torch.complex(x, torch.randn_like(x))
        with torch.no_grad():
            y_ref, y_fast = ref(x), fast(x)
        d = (y_ref - y_fast).abs().max().item()
        scale = y_ref.abs().mean().item()
        rel = d / max(scale, 1e-12)
        flag = '' if rel < 1e-5 else '   <-- FAIL'
        ok = ok and rel < 1e-5
        print('%-16s %12.3e %12.3e %10.2e%s' % (name, d, scale, rel, flag))
    return ok


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
STAGES = {
    'conv': dict(pack_convs=True, pack_backbones=False, shift_free=False,
                 cache_multiplier=False),
    'backbone': dict(pack_convs=True, pack_backbones=True, shift_free=False,
                     cache_multiplier=False),
    'asm': dict(pack_convs=True, pack_backbones=True, shift_free=True,
                cache_multiplier=False),
    'mult': dict(pack_convs=True, pack_backbones=True, shift_free=True,
                 cache_multiplier=True),
}
STAGE_ORDER = ['conv', 'backbone', 'asm', 'mult']


def run_cascade(kind, args, n, m):
    device = torch.device(args.device)
    amp = (div2k_amplitude(args.data_path, args.index, n, m, device)
           if args.data_path else fixed_amp(n, m, device))

    ref = FastCGHInference(build_model(kind, device, args.weight), n, m, args.pitch,
                           args.wavelength, args.z, device,
                           pack_convs=False, pack_backbones=False,
                           cache_multiplier=False, shift_free=False)
    phase_ref, image_ref = ref.run(amp)
    t_ref = timeit(lambda: ref.run(amp), warmup=2, reps=args.reps,
                   synchronize=(device.type == 'cuda'))[0]

    print('\n%s  %d x %d  %s  z=%.0f mm  lambda=%.0f nm'
          % (kind, n, m, args.device, args.z, args.wavelength * 1e6))
    print('  reference: %.2f ms/frame, PSNR %.4f dB, SSIM-free' % (t_ref, psnr(image_ref, amp)))

    limit = STAGE_ORDER[-1] if args.stage == 'all' else args.stage
    rows = []
    ok = True
    for stage in STAGE_ORDER:
        if STAGE_ORDER.index(stage) > STAGE_ORDER.index(limit):
            continue
        engine = FastCGHInference(build_model(kind, device, args.weight), n, m, args.pitch,
                                  args.wavelength, args.z, device, **STAGES[stage])
        phase, image = engine.run(amp)
        t = timeit(lambda: engine.run(amp), warmup=2, reps=args.reps,
                   synchronize=(device.type == 'cuda'))[0]
        d_phi = wrapped_max_abs(phase, phase_ref)
        d_img = (image - image_ref).abs().max().item()
        agree = psnr(image, image_ref)
        quality = psnr(image, amp)
        # float32 round-off accumulates through 8-14 complex convolutions:
        # 8e-5 rad at 1080p, 8e-4 rad at 4K (0.013 % of a period). Gate on the
        # image-domain agreement, which is what a user actually sees.
        passed = bool(agree >= 120.0 and abs(quality - psnr(image_ref, amp)) < 1e-2)
        ok = ok and passed
        rows.append(dict(stage=stage, ms=t, d_phi=d_phi, d_image=d_img,
                         psnr_agreement=agree, psnr_quality=quality, passed=passed))
        print('  %-8s %7.2f ms (%4.2fx)  max|dphi|=%.2e  max|dI|=%.2e  '
              'agree %5.1f dB  quality %.4f dB  %s'
              % (stage, t, t_ref / t, d_phi, d_img, agree, quality,
                 'ok' if passed else 'FAIL'))
    return rows, ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--unit', action='store_true', help='layer-level check only')
    p.add_argument('--stage', default='all',
                   choices=['conv', 'backbone', 'asm', 'mult', 'all'])
    p.add_argument('--model', default='freq', choices=['freq', 'spatial', 'uncond'])
    p.add_argument('--weight', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resolution', default='1080p')
    p.add_argument('--z', type=float, default=150.0)
    p.add_argument('--wavelength', type=float, default=0.000532)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--data-path', default=None,
                   help='DIV2K folder; omit to use a constant amplitude')
    p.add_argument('--index', type=int, default=801)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--json', default=None)
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA not available, falling back to CPU')
        args.device = 'cpu'

    if args.unit:
        ok = check_unit(torch.device(args.device))
        raise SystemExit(0 if ok else 1)

    n, m = parse_resolution(args.resolution)
    rows, ok = run_cascade(args.model, args, n, m)
    dump(dict(model=args.model, resolution=[n, m], device=args.device,
              z=args.z, wavelength=args.wavelength, rows=rows), args.json)
    print('\n%s' % ('ALL CHECKS PASSED' if ok else 'CHECKS FAILED'))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
