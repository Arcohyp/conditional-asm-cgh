"""Two-channel packing for complexPyTorch convolutional layers.

`complexPyTorch.ComplexConv2d` implements a complex convolution as

.. code-block:: python

    (conv_r(x.real) - conv_i(x.imag)) + 1j * (conv_r(x.imag) + conv_i(x.real))

which is four real convolutions, two full-size casts to ``complex64`` and two
non-contiguous ``.real`` / ``.imag`` views per layer. On modern accelerators
that costs 7 kernel launches and 2 extra full-resolution temporaries per layer.

The same operation is a single real convolution if the real and imaginary parts
are stacked along the channel axis. Writing a complex weight ``w = wr + i wi``
as the block matrix

    [[ wr, -wi],
     [ wi,  wr]]

and applying it to a stacked input ``[x.real; x.imag]`` reproduces
``w * x`` exactly. Nothing is approximated, quantised or pruned: the packed
weight is a pure re-indexing of the original parameters, and the module
therefore produces bit-comparable output for bit-identical input.

Usage
-----
Load the checkpoint into the original model first, then convert in place::

    from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
    from conditional_asm_cgh.utils.fast_complex import convert_model

    model = ConditionalCCNNcghModelMinExp()
    model.load_state_dict(torch.load('weights/spatial_conditional.pth')['model_state_dict'])
    n = convert_model(model)          # 8 layers replaced, state dict untouched
    model.cuda().eval()

``convert_model`` walks the module tree and replaces every ``ComplexConv2d`` /
``ComplexConvTranspose2d`` by a :class:`PackedComplexConv2d` /
:class:`PackedComplexConvTranspose2d` that holds the same parameters in packed
form. Because the replacement happens after ``load_state_dict``, no checkpoint
key changes, and ``convert_model`` is idempotent.

Note on the bias
----------------
``apply_complex`` reuses ``conv_r`` and ``conv_i`` for both output parts, so the
leaked bias of the reference implementation is ``(br - bi) + 1j (br + bi)``
rather than ``br + 1j bi``. The packed bias reproduces that behaviour so that
the two implementations agree bit for bit; see ``benchmarks/verify_equivalence.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexPyTorch.complexLayers import ComplexConv2d, ComplexConvTranspose2d


__all__ = [
    'complex_to_2ch',
    'ch2_to_complex',
    'pack_conv_weight',
    'pack_convtranspose_weight',
    'packed_bias',
    'unpack_packed_grad',
    'PackedComplexConv2d',
    'PackedComplexConvTranspose2d',
    'PackedBackbone',
    'convert_model',
    'count_complex_layers',
]


# ---------------------------------------------------------------------------
# Layout helpers.  The channel order is [all real channels, all imag channels].
# ---------------------------------------------------------------------------
def complex_to_2ch(x):
    """``(B, C, H, W) complex`` -> ``(B, 2C, H, W) real``."""
    b, c, h, w = x.shape
    return torch.view_as_real(x).permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def ch2_to_complex(y):
    """``(B, 2C, H, W) real`` -> ``(B, C, H, W) complex``."""
    c = y.shape[1] // 2
    return torch.complex(y[:, :c], y[:, c:])


# ---------------------------------------------------------------------------
# Weight packing.
# ---------------------------------------------------------------------------
def pack_conv_weight(wr, wi):
    """Pack ``(out, in, kh, kw)`` real/imag kernels into ``(2*out, 2*in, kh, kw)``."""
    cout, cin = wr.shape[0], wr.shape[1]
    w = wr.new_zeros(2 * cout, 2 * cin, *wr.shape[2:])
    w[:cout, :cin] = wr
    w[:cout, cin:] = -wi
    w[cout:, :cin] = wi
    w[cout:, cin:] = wr
    return w


def pack_convtranspose_weight(wr, wi):
    """Pack ``(in, out, kh, kw)`` transpose kernels into ``(2*in, 2*out, kh, kw)``."""
    cin, cout = wr.shape[0], wr.shape[1]
    w = wr.new_zeros(2 * cin, 2 * cout, *wr.shape[2:])
    w[:cin, :cout] = wr
    w[:cin, cout:] = wi
    w[cin:, :cout] = -wi
    w[cin:, cout:] = wr
    return w


def packed_bias(br, bi):
    """Reproduce ``apply_complex``'s leaked bias ``(br - bi, br + bi)``."""
    return torch.cat([br - bi, br + bi])


def unpack_packed_grad(grad, transposed=False):
    """Sum a packed gradient back into ``(dL/dwr, dL/dwi)``.

    The reference module shares one ``wr`` between the real and the imaginary
    output, so its gradient is the sum of both contributions; in the packed
    layout those contributions land in two different blocks. For a convolution
    the blocks are ``(0,0)`` and ``(1,1)`` for ``wr`` and ``(0,1)``/``(1,0)``
    with opposite signs for ``wi``; the transposed layer swaps the roles of the
    channel axes.
    """
    if transposed:
        cin, cout = grad.shape[0] // 2, grad.shape[1] // 2
        d_wr = grad[:cin, :cout] + grad[cin:, cout:]
        d_wi = grad[:cin, cout:] - grad[cin:, :cout]
    else:
        cout, cin = grad.shape[0] // 2, grad.shape[1] // 2
        d_wr = grad[:cout, :cin] + grad[cout:, cin:]
        d_wi = grad[cout:, :cin] - grad[:cout, cin:]
    return d_wr, d_wi


# ---------------------------------------------------------------------------
# Drop-in modules.
# ---------------------------------------------------------------------------
class PackedComplexConv2d(nn.Module):
    """``ComplexConv2d`` with the four real convolutions fused into one.

    Input and output are complex tensors of shape ``(B, C, H, W)``, exactly as
    for ``ComplexConv2d``. Internally a single ``nn.Conv2d`` runs on a stacked
    ``(B, 2C, H, W)`` real tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = nn.Conv2d(2 * in_channels, 2 * out_channels, kernel_size,
                            stride=stride, padding=padding, dilation=dilation,
                            bias=bias, padding_mode=padding_mode)

    @classmethod
    def from_complex_conv(cls, m):
        """Build from a ``complexPyTorch.ComplexConv2d`` instance."""
        r = m.conv_r
        if r.groups != 1:
            raise NotImplementedError('grouped complex convolution is not supported')
        new = cls(r.in_channels, r.out_channels, r.kernel_size, stride=r.stride,
                  padding=r.padding, dilation=r.dilation,
                  bias=r.bias is not None, padding_mode=r.padding_mode)
        with torch.no_grad():
            new.op.weight.copy_(pack_conv_weight(r.weight, m.conv_i.weight))
            if r.bias is not None:
                new.op.bias.copy_(packed_bias(r.bias, m.conv_i.bias))
        return new.to(device=r.weight.device, dtype=r.weight.dtype)

    def forward(self, x):
        return ch2_to_complex(self.op(complex_to_2ch(x)))

    def extra_repr(self):
        return 'in_channels=%d, out_channels=%d' % (self.in_channels, self.out_channels)


class PackedComplexConvTranspose2d(nn.Module):
    """``ComplexConvTranspose2d`` with the four transposed convolutions fused."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = nn.ConvTranspose2d(2 * in_channels, 2 * out_channels, kernel_size,
                                     stride=stride, padding=padding,
                                     output_padding=output_padding, dilation=dilation,
                                     bias=bias, padding_mode=padding_mode)

    @classmethod
    def from_complex_convtranspose(cls, m):
        r = m.conv_tran_r
        if r.groups != 1:
            raise NotImplementedError('grouped complex convolution is not supported')
        new = cls(r.in_channels, r.out_channels, r.kernel_size, stride=r.stride,
                  padding=r.padding, output_padding=r.output_padding,
                  dilation=r.dilation, bias=r.bias is not None,
                  padding_mode=r.padding_mode)
        with torch.no_grad():
            new.op.weight.copy_(pack_convtranspose_weight(r.weight, m.conv_tran_i.weight))
            if r.bias is not None:
                new.op.bias.copy_(packed_bias(r.bias, m.conv_tran_i.bias))
        return new.to(device=r.weight.device, dtype=r.weight.dtype)

    def forward(self, x):
        return ch2_to_complex(self.op(complex_to_2ch(x)))

    def extra_repr(self):
        return 'in_channels=%d, out_channels=%d' % (self.in_channels, self.out_channels)


# ---------------------------------------------------------------------------
# Whole-backbone execution in the two-channel layout.
# ---------------------------------------------------------------------------
class PackedBackbone(nn.Module):
    """Run a ``CCNN1`` / ``CCNN2`` encoder-decoder without leaving the 2-channel layout.

    Replacing the individual convolutions is enough for correctness, but each
    replacement still has to convert the tensor at the layer boundary
    (``complex -> 2ch`` before the convolution, back afterwards). For this
    U-Net that is 6-8 conversions per backbone, and on an A6000 the conversion
    traffic costs about as much as the convolutions saved.

    ``PackedBackbone`` keeps the activations in ``(B, 2C, H, W)`` for the whole
    encoder-decoder instead. This is sound because every non-convolution step of
    ``CCNN1`` / ``CCNN2`` acts on the real and imaginary parts independently:

    * ``complex_relu``   -> ``F.relu`` on the stacked tensor
    * ``_match_complex_size`` -> ``F.interpolate`` on the stacked tensor
    * the residual adds  -> plain ``+``
    * the final ``atan2``-> ``atan2(imag_half, real_half)``

    The module takes a complex tensor and returns a phase, exactly like
    ``CCNN1.forward`` / ``CCNN2.forward``, so it is a drop-in replacement and
    the surrounding model code (conditioning MLP, frequency multiplier, ASM)
    does not change.
    """

    def __init__(self, ccnn):
        super().__init__()
        n = 0
        while hasattr(ccnn, 'netdown%d' % (n + 1)):
            n += 1
        if n == 0 or not hasattr(ccnn, 'netup1'):
            raise ValueError('module does not look like a CCNN encoder-decoder')
        self.n_stages = n
        self.name = getattr(ccnn, 'name', 'packed_backbone')

        # Keep the registration order identical to the original tree
        # (netdown1..n, then netupn..netup1) so that a walk over .modules()
        # visits the convolutions in the same order as the reference model.
        downs = [getattr(ccnn, 'netdown%d' % (i + 1)).COV1[0].op for i in range(n)]
        ups = [getattr(ccnn, 'netup%d' % i).COV1[0].op for i in range(n, 0, -1)]
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

    @staticmethod
    def _match(target, source):
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(source, size=target.shape[-2:], mode='bilinear',
                             align_corners=False)

    def forward(self, x):
        """``(B, 1, H, W) complex`` -> ``(B, 1, H, W) real`` phase."""
        target_hw = x.shape[-2:]
        h = complex_to_2ch(x)

        feats = []
        for down in self.downs:
            h = F.relu(down(h))
            feats.append(h)

        n = self.n_stages
        for k in range(n - 1):
            h = F.relu(self.ups[k](h))          # netup{n-k}
            h = h + self._match(h, feats[n - 2 - k])
        h = self.ups[n - 1](h)                  # netup1: no activation

        if h.shape[-2:] != target_hw:
            h = F.interpolate(h, size=target_hw, mode='bilinear', align_corners=False)
        c = h.shape[1] // 2
        return torch.atan2(h[:, c:], h[:, :c])


# ---------------------------------------------------------------------------
# In-place model conversion.
# ---------------------------------------------------------------------------
def _iter_with_parent(root):
    for name, mod in root.named_modules():
        if not name:
            continue
        parent_name, _, leaf = name.rpartition('.')
        parent = root.get_submodule(parent_name) if parent_name else root
        yield parent, leaf, mod


def count_complex_layers(model):
    """Number of ``complexPyTorch`` convolution layers still in the tree."""
    return sum(1 for _, mod in model.named_modules()
               if isinstance(mod, (ComplexConv2d, ComplexConvTranspose2d)))


def _looks_like_backbone(mod):
    return (not isinstance(mod, PackedBackbone)
            and hasattr(mod, 'netdown1') and hasattr(mod, 'netup1'))


def convert_model(model, verbose=False, pack_backbones=True):
    """Replace every complex convolution in ``model`` in place.

    Call this **after** ``load_state_dict``: the packed modules expose a
    different parameter layout, and the conversion reads the original weights.

    With ``pack_backbones=True`` (default) each ``CCNN1`` / ``CCNN2``
    encoder-decoder is additionally wrapped in :class:`PackedBackbone`, which
    removes the per-layer layout conversions. Set it to ``False`` to keep the
    original module graph and only fuse the convolutions.

    Returns the number of layers replaced. The operation is idempotent and
    leaves every other module (conditioning MLPs, ``_radial_to_2d``, ASM calls)
    untouched.
    """
    targets = [(p, leaf, m) for p, leaf, m in _iter_with_parent(model)
               if isinstance(m, (ComplexConv2d, ComplexConvTranspose2d))]
    for parent, leaf, mod in targets:
        if isinstance(mod, ComplexConv2d):
            new = PackedComplexConv2d.from_complex_conv(mod)
        else:
            new = PackedComplexConvTranspose2d.from_complex_convtranspose(mod)
        setattr(parent, leaf, new)
        if verbose:
            print('  packed %s.%s' % (type(parent).__name__, leaf))

    if pack_backbones:
        backbones = [(p, leaf, m) for p, leaf, m in _iter_with_parent(model)
                     if _looks_like_backbone(m)]
        for parent, leaf, mod in backbones:
            packed = PackedBackbone(mod)
            setattr(parent, leaf, packed)
            if verbose:
                print('  wrapped backbone %s.%s (%d stages)'
                      % (type(mod).__name__, leaf, packed.n_stages))

    return len(targets)
