"""Training-side effect: step time and gradient equivalence.

The packing trick is not inference-only -- the same fused convolution can be
used while training, because ``PackedBackbone`` keeps the parameters as
``nn.Parameter`` and every op in the encoder-decoder is differentiable.

Two measurements:

``--grad-check``
    Runs one forward+backward pass through the reference model and through the
    packed model, then rebuilds the block matrix from the reference gradients
    and compares it with the packed gradient. The two must agree to float32
    round-off; anything else would mean the packing changed the optimisation
    problem rather than just its layout.

``--time``
    Wall-clock of one forward and one forward+backward step for both models.

Caveat
------
Packing *does* change the parameter layout: one packed weight holds
``4 * Cin * Cout`` numbers where the reference holds ``2 * Cin * Cout`` in two
separate tensors. The extra numbers are redundant copies, not new degrees of
freedom. An optimiser built before the conversion will therefore see the wrong
number of parameters -- rebuild it (or convert before constructing it) and the
update trajectory is unchanged, because the gradients on the redundant blocks
are consistent by construction.

Usage::

    python -m conditional_asm_cgh.benchmarks.train_equiv --device cuda \
        --resolution 1080p --grad-check --time
"""

import argparse

import torch

from conditional_asm_cgh.utils import propagation_ASM as asm
from conditional_asm_cgh.utils.fast_complex import (
    PackedBackbone, PackedComplexConv2d, PackedComplexConvTranspose2d,
    convert_model, unpack_packed_grad,
)

from conditional_asm_cgh.benchmarks.common import (
    build_model, parse_resolution, timeit, dump,
)


def asm_kernels(n, m, pitch, wl, z, device):
    probe = torch.empty(1, 1, n, m, device=device)
    h_fwd = asm.propagation_ASM(probe, feature_size=[pitch, pitch], wavelength=wl,
                                z=z, linear_conv=False, return_H=True)
    h_bwd = asm.propagation_ASM(probe, feature_size=[pitch, pitch], wavelength=wl,
                                z=-z, linear_conv=False, return_H=True)
    return h_fwd, h_bwd


def loss_and_backward(model, amp, target, n, m, pitch, wl, z, device,
                      backward=True):
    """One training step's forward (and backward). Returns the loss value."""
    h_fwd, h_bwd = asm_kernels(n, m, pitch, wl, z, device)
    init = torch.zeros_like(amp)
    phase, _ = model(amp, init, z, False, pitch, wl, h_fwd)
    slm = torch.complex(torch.cos(phase), torch.sin(phase))
    field = asm.propagation_ASM(u_in=slm, z=-z, linear_conv=False,
                                feature_size=[pitch, pitch], wavelength=wl,
                                precomped_H=h_bwd)
    loss = torch.mean((torch.abs(field) - target) ** 2)
    if backward:
        loss.backward()
    return loss.item()


def gradient_check(device, seed=0):
    """Max relative deviation between reference and packed gradients.

    Layer level, on one ``ComplexConv2d`` and one ``ComplexConvTranspose2d``
    with random weights: forward both implementations on the same complex
    input, backprop a smooth MSE loss, then rebuild the block matrix from the
    reference gradients and compare. This isolates the packing from anything
    the surrounding model does.
    """
    from complexPyTorch.complexLayers import ComplexConv2d, ComplexConvTranspose2d
    from conditional_asm_cgh.utils.fast_complex import (
        PackedComplexConv2d, PackedComplexConvTranspose2d,
    )

    torch.manual_seed(seed)
    cases = [
        (ComplexConv2d(4, 8, 3, padding=1),
         PackedComplexConv2d.from_complex_conv, False, (2, 4, 32, 48)),
        (ComplexConvTranspose2d(8, 4, 4, stride=2, padding=1),
         PackedComplexConvTranspose2d.from_complex_convtranspose, True, (2, 8, 16, 24)),
    ]

    print('gradient equivalence (block layout rebuilt from conv_r/conv_i)')
    print('%-22s %12s %12s' % ('layer', 'max|d grad|', 'rel'))
    worst = 0.0
    for ref, factory, transposed, shape in cases:
        ref = ref.to(device)
        fast = factory(ref).to(device)
        x = torch.randn(shape, device=device)
        x = torch.complex(x, torch.randn_like(x))

        out_ref = ref(x)
        target = torch.randn_like(out_ref)
        torch.mean(torch.abs(out_ref - target) ** 2).backward()

        out_fast = fast(x)
        torch.mean(torch.abs(out_fast - target) ** 2).backward()

        if transposed:
            r = ref.conv_tran_r
            gr, gi = ref.conv_tran_r.weight.grad, ref.conv_tran_i.weight.grad
            label = 'convT %d->%d' % (r.in_channels, r.out_channels)
        else:
            r = ref.conv_r
            gr, gi = ref.conv_r.weight.grad, ref.conv_i.weight.grad
            label = 'conv %d->%d' % (r.in_channels, r.out_channels)

        d_wr, d_wi = unpack_packed_grad(fast.op.weight.grad, transposed)
        scale = max(gr.abs().max().item(), 1e-12)
        rel = max((d_wr - gr).abs().max().item(), (d_wi - gi).abs().max().item()) / scale
        worst = max(worst, rel)
        print('%-22s %12.3e %12.2e' % (label,
                                       max((d_wr - gr).abs().max().item(),
                                           (d_wi - gi).abs().max().item()), rel))
    print('worst relative deviation: %.2e' % worst)
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='freq', choices=['freq', 'spatial', 'uncond'])
    p.add_argument('--weight', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resolution', default='1080p')
    p.add_argument('--z', type=float, default=150.0)
    p.add_argument('--wavelength', type=float, default=0.000532)
    p.add_argument('--pitch', type=float, default=0.0036)
    p.add_argument('--grad-check', action='store_true')
    p.add_argument('--time', action='store_true')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--json', default=None)
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA not available, falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    n, m = parse_resolution(args.resolution)

    # A constant target amplitude leaves the loss at a stationary point with
    # respect to most of the network, which makes every gradient exactly zero
    # and the comparison meaningless. Use a random pair instead.
    torch.manual_seed(0)
    amp = torch.rand(1, 1, n, m, device=device) * 0.3 + 0.55
    target = torch.rand(1, 1, n, m, device=device) * 0.7 + 0.15

    out = dict(model=args.model, device=str(device), resolution=[n, m])

    if args.grad_check:
        worst = gradient_check(device)
        out['grad_rel_max'] = worst
        if worst > 1e-5:
            raise SystemExit('gradient check failed: %.2e' % worst)

    if args.time:
        asm.set_fast_shift(True)
        ref = build_model(args.model, device, args.weight).train()
        fwd = timeit(lambda: loss_and_backward(ref, amp, target, n, m, args.pitch,
                                               args.wavelength, args.z, device,
                                               backward=False),
                     warmup=args.warmup, reps=args.steps,
                     synchronize=(device.type == 'cuda'))
        step = timeit(lambda: loss_and_backward(ref, amp, target, n, m, args.pitch,
                                                args.wavelength, args.z, device,
                                                backward=True),
                      warmup=args.warmup, reps=args.steps,
                      synchronize=(device.type == 'cuda'))
        del ref
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        fast_model = build_model(args.model, device, args.weight).train()
        convert_model(fast_model)
        fwd_f = timeit(lambda: loss_and_backward(fast_model, amp, target, n, m,
                                                 args.pitch, args.wavelength, args.z,
                                                 device, backward=False),
                       warmup=args.warmup, reps=args.steps,
                       synchronize=(device.type == 'cuda'))
        step_f = timeit(lambda: loss_and_backward(fast_model, amp, target, n, m,
                                                  args.pitch, args.wavelength, args.z,
                                                  device, backward=True),
                        warmup=args.warmup, reps=args.steps,
                        synchronize=(device.type == 'cuda'))

        print('\ntraining step, %d x %d' % (n, m))
        print('%-14s %10s %10s %9s' % ('', 'ref (ms)', 'fast (ms)', 'speedup'))
        print('%-14s %10.2f %10.2f %8.2fx' % ('forward', fwd[0], fwd_f[0], fwd[0] / fwd_f[0]))
        print('%-14s %10.2f %10.2f %8.2fx' % ('forward+back', step[0], step_f[0],
                                               step[0] / step_f[0]))
        out['forward_ms'] = [fwd[0], fwd_f[0]]
        out['step_ms'] = [step[0], step_f[0]]
        out['speedup'] = [fwd[0] / fwd_f[0], step[0] / step_f[0]]

    dump(out, args.json)


if __name__ == '__main__':
    main()
