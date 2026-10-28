"""End-to-end inference latency of the released models, reference vs fast path.

One frame = model forward (SLM phase) + backward propagation (image plane
intensity), batch size 1, median over ``--reps`` after warm-up. Nothing is
cached between frames except the quantities that are constant by construction
(transfer functions, the (z, lambda) multiplier).

Examples
--------
GPU, all four resolutions::

    python -m conditional_asm_cgh.benchmarks.bench_deploy --device cuda \
        --resolutions 720p,1080p,1440p,4k --reps 20

CPU, pin the thread count (see note below)::

    python -m conditional_asm_cgh.benchmarks.bench_deploy --device cpu \
        --resolutions 1080p,4k --threads 36 --reps 3

Note on CPU numbers
-------------------
Complex 4K convolutions are memory-bandwidth bound. On a shared machine the
reference implementation varies by up to 50 % between runs because it streams
~8x the bytes of the packed version through the memory system; the fast path is
stable to about 1 %. Pin the process to one NUMA node (``taskset -c 0-35``) and
report the median of several runs when comparing machines.
"""

import argparse

import torch

from conditional_asm_cgh.utils.fast_infer import FastCGHInference

from conditional_asm_cgh.benchmarks.common import (
    build_model, fixed_amp, parse_resolution, timeit, dump,
)


def measure(kind, n, m, device, args):
    amp = fixed_amp(n, m, device)

    ref = FastCGHInference(build_model(kind, device, args.weight), n, m, args.pitch,
                           args.wavelength, args.z, device,
                           pack_convs=False, pack_backbones=False,
                           cache_multiplier=False, shift_free=False)
    fast = FastCGHInference(build_model(kind, device, args.weight), n, m, args.pitch,
                            args.wavelength, args.z, device,
                            pack_convs=True, pack_backbones=True,
                            cache_multiplier=True, shift_free=True)

    sync = device.type == 'cuda'
    t_ref = timeit(lambda: ref.run(amp), warmup=args.warmup, reps=args.reps, synchronize=sync)
    t_fast = timeit(lambda: fast.run(amp), warmup=args.warmup, reps=args.reps, synchronize=sync)
    return t_ref, t_fast


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='freq', choices=['freq', 'spatial', 'uncond'])
    p.add_argument('--weight', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resolutions', default='720p,1080p,1440p,4k')
    p.add_argument('--z', type=float, default=150.0)
    p.add_argument('--wavelength', type=float, default=0.000532)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--threads', type=int, default=0,
                   help='torch CPU threads (0 = leave the default)')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--json', default=None)
    p.add_argument('--label', default=None, help='tag stored in the json output')
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA not available, falling back to CPU')
        args.device = 'cpu'
    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print('%s  threads=%d' % (name, torch.get_num_threads()))
    print('%-8s %10s %10s %9s %10s %10s' %
          ('res', 'ref (ms)', 'fast (ms)', 'speedup', 'ref (fps)', 'fast (fps)'))

    rows = []
    for spec in args.resolutions.split(','):
        n, m = parse_resolution(spec)
        (med_r, lo_r, hi_r), (med_f, lo_f, hi_f) = measure(args.model, n, m, device, args)
        rows.append(dict(resolution=[n, m], tag=spec, ref_ms=med_r, fast_ms=med_f,
                         speedup=med_r / med_f, ref_fps=1000.0 / med_r,
                         fast_fps=1000.0 / med_f,
                         ref_range=[lo_r, hi_r], fast_range=[lo_f, hi_f]))
        print('%-8s %10.2f %10.2f %8.2fx %10.1f %10.1f'
              % (spec, med_r, med_f, med_r / med_f, 1000.0 / med_r, 1000.0 / med_f))

    dump(dict(model=args.model, device=str(device), hardware=name,
              threads=torch.get_num_threads(), z=args.z,
              wavelength=args.wavelength, label=args.label, rows=rows), args.json)


if __name__ == '__main__':
    main()
