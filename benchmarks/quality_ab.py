"""Quality A/B: does the fast path change the reconstruction?

The fast implementation is an algebraic rearrangement, so the expected answer is
"no, beyond float32 round-off". This script checks that on real data against the
metric the paper reports: PSNR and SSIM of the reconstructed intensity at the
image plane.

For every (image, wavelength, z) triple it runs the reference pipeline and the
fast pipeline from the same checkpoint and prints both scores plus their
difference. A change in the metric would show up as a non-zero delta; the
agreement PSNR between the two outputs is printed separately (it measures how
close the implementations are to each other, not how good they are)::

    python -m conditional_asm_cgh.benchmarks.quality_ab \
        --data-path /path/to/DIV2K_valid_HR --num-images 3 --device cuda
"""

import argparse

import torch

from conditional_asm_cgh.utils.fast_infer import FastCGHInference

from conditional_asm_cgh.benchmarks.common import (
    build_model, div2k_amplitude, parse_resolution, psnr, ssim, dump,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-path', required=True, help='DIV2K_valid_HR folder')
    p.add_argument('--model', default='freq', choices=['freq', 'spatial'])
    p.add_argument('--weight', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resolution', default='1080p')
    p.add_argument('--num-images', type=int, default=3)
    p.add_argument('--start-index', type=int, default=801)
    p.add_argument('--zs', type=float, nargs='+', default=[145.0, 165.0, 179.0])
    p.add_argument('--wavelengths', type=float, nargs='+',
                   default=[0.000450, 0.000532, 0.000638])
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--json', default=None)
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA not available, falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    n, m = parse_resolution(args.resolution)

    # one amplitude per image, reused across every (z, lambda)
    amps = [div2k_amplitude(args.data_path, args.start_index + i, n, m, device)
            for i in range(args.num_images)]

    print('%s  %d x %d  %d images x %d z x %d lambda'
          % (args.device, n, m, len(amps), len(args.zs), len(args.wavelengths)))
    print('%-4s %-7s %-6s | %8s %8s %8s | %8s %8s %8s | %9s'
          % ('img', 'z(mm)', 'lam', 'PSNR ref', 'PSNR fast', 'dPSNR',
             'SSIM ref', 'SSIM fast', 'dSSIM', 'agree(dB)'))

    rows = []
    for z in args.zs:
        for wl in args.wavelengths:
            ref = FastCGHInference(build_model(args.model, device, args.weight), n, m,
                                   args.pitch, wl, z, device,
                                   pack_convs=False, pack_backbones=False,
                                   cache_multiplier=False, shift_free=False)
            fast = FastCGHInference(build_model(args.model, device, args.weight), n, m,
                                    args.pitch, wl, z, device)
            for i, amp in enumerate(amps):
                _, img_r = ref.run(amp)
                _, img_f = fast.run(amp)
                p_r, p_f = psnr(img_r, amp), psnr(img_f, amp)
                s_r, s_f = ssim(img_r, amp), ssim(img_f, amp)
                agree = psnr(img_f, img_r)
                rows.append(dict(image=args.start_index + i, z=z, wavelength=wl,
                                 psnr_ref=float(p_r), psnr_fast=float(p_f),
                                 d_psnr=float(p_f - p_r),
                                 ssim_ref=float(s_r), ssim_fast=float(s_f),
                                 d_ssim=float(s_f - s_r),
                                 agreement_db=float(agree),
                                 max_abs_diff=float((img_f - img_r).abs().max())))
                print('%-4d %-7.0f %-6.0f | %8.4f %8.4f %8.5f | %8.5f %8.5f %8.6f | %9.1f'
                      % (args.start_index + i, z, wl * 1e6, p_r, p_f, p_f - p_r,
                         s_r, s_f, s_f - s_r, agree))

    dp = [r['d_psnr'] for r in rows]
    ds = [r['d_ssim'] for r in rows]
    print('\n%d groups: mean PSNR %.4f dB (ref) / %.4f dB (fast), '
          'max |dPSNR| %.5f dB, max |dSSIM| %.2e'
          % (len(rows), sum(r['psnr_ref'] for r in rows) / len(rows),
             sum(r['psnr_fast'] for r in rows) / len(rows),
             max(abs(v) for v in dp), max(abs(v) for v in ds)))
    print('worst pixel difference between the two pipelines: %.2e'
          % max(r['max_abs_diff'] for r in rows))

    dump(dict(model=args.model, resolution=[n, m], device=args.device,
              zs=args.zs, wavelengths=args.wavelengths, rows=rows), args.json)


if __name__ == '__main__':
    main()
