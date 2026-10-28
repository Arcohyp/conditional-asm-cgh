"""Single-call fast inference path for the CGH models in this repository.

This module bundles the three inference-only rewrites that are described in
``benchmarks/README.md``:

1. **Packed complex convolutions** -- every ``ComplexConv2d`` /
   ``ComplexConvTranspose2d`` is replaced by a single real convolution on a
   stacked ``[real; imag]`` channel tensor (:mod:`fast_complex`).
2. **Cached (z, lambda) frequency multiplier** -- ``ConditionalFreqMultiplier``
   rebuilds a full-resolution ``meshgrid -> sqrt -> grid_sample`` map on every
   call (116 ms for a 4K frame). The map depends only on the conditioning
   vector, so it is computed once per distinct (z, wavelength, pitch) and
   reused. The model object itself is not modified: the memoised wrapper is
   attached to the instance.
3. **Shift-free ASM** -- the ``ifftshift`` / ``fftshift`` pair around the
   transfer-function multiply cancels exactly (see
   :func:`propagation_ASM.set_fast_shift`), which removes two 4K array rolls
   per propagation.

None of the three changes a weight, a parameter count, a checkpoint format or a
numerical result beyond float round-off. Everything here is inference-only; for
training see ``benchmarks/train_equiv.py``.

Typical use::

    import torch
    from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
    from conditional_asm_cgh.utils.fast_infer import FastCGHInference

    model = ConditionalCCNNcghModelMinExp()
    model.load_state_dict(torch.load('weights/spatial_conditional.pth')['model_state_dict'])

    engine = FastCGHInference(model, n=2160, m=3840, pitch=0.0036,
                              wavelength=0.000532, z=150.0, device='cuda')
    phase, image = engine.run(amp)          # amp: (1, 1, n, m) target amplitude
"""

import collections

import torch

from conditional_asm_cgh.utils import propagation_ASM as _asm
from conditional_asm_cgh.utils.fast_complex import convert_model, count_complex_layers


__all__ = ['FastCGHInference']


class FastCGHInference(object):
    """Wraps a trained CGH model with the fast inference path.

    Parameters
    ----------
    model : nn.Module
        One of the models in ``models/``, with its weights already loaded.
        The module is modified in place (complex convolutions are packed).
    n, m : int
        Spatial resolution of the SLM plane (rows, columns).
    pitch : float
        Pixel pitch in mm.
    wavelength : float
        Wavelength in mm.
    z : float
        Propagation distance in mm.
    device : str or torch.device
        ``'cuda'`` or ``'cpu'``.
    pack_convs, cache_multiplier, shift_free : bool
        Individual switches, all ``True`` by default. Set any of them to
        ``False`` to isolate one rewrite (``benchmarks/verify_equivalence.py``
        does exactly that).
    """

    def __init__(self, model, n, m, pitch, wavelength, z, device='cuda',
                 pack_convs=True, pack_backbones=True, cache_multiplier=True,
                 shift_free=True):
        self.device = torch.device(device)
        self.n, self.m = int(n), int(m)
        self.pitch = float(pitch)
        self.wavelength = float(wavelength)
        self.z = float(z)
        self.shift_free = bool(shift_free)

        self.model = model.to(self.device).eval()

        if pack_convs:
            self.n_packed = convert_model(self.model, pack_backbones=pack_backbones)
        else:
            self.n_packed = 0
        self.backbones_packed = bool(pack_convs and pack_backbones)
        self.multiplier_cached = (self._install_multiplier_cache()
                                  if cache_multiplier else False)

        probe = torch.empty(1, 1, self.n, self.m, device=self.device)
        self.H = _asm.propagation_ASM(probe, feature_size=[self.pitch, self.pitch],
                                      wavelength=self.wavelength, z=self.z,
                                      linear_conv=False, return_H=True)
        self.H_back = _asm.propagation_ASM(probe, feature_size=[self.pitch, self.pitch],
                                           wavelength=self.wavelength, z=-self.z,
                                           linear_conv=False, return_H=True)
        del probe

    # ------------------------------------------------------------------
    # rewrite 2: memoise the (z, lambda) multiplier geometry
    # ------------------------------------------------------------------
    def _install_multiplier_cache(self):
        fm = getattr(self.model, 'freq_mult', None)
        if fm is None or not hasattr(fm, '_radial_to_2d'):
            return False

        original = fm._radial_to_2d
        cache = collections.OrderedDict()
        maxsize = 2

        def cached_radial_to_2d(radial, h, w, device):
            key = (int(h), int(w), str(device),
                   radial.detach().to('cpu', torch.float32).contiguous().numpy().tobytes())
            hit = cache.get(key)
            if hit is None:
                hit = original(radial, h, w, device)
                cache[key] = hit
                while len(cache) > maxsize:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(key)
            return hit

        fm._radial_to_2d = cached_radial_to_2d
        return True

    # ------------------------------------------------------------------
    # rewrite 3: global ASM switch
    # ------------------------------------------------------------------
    def use_fast_asm(self, enabled):
        _asm.set_fast_shift(bool(enabled) and self.shift_free)
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def holophase(self, amp, init_phase=None):
        """SLM phase for a target amplitude ``(B, 1, n, m)``."""
        if init_phase is None:
            init_phase = torch.zeros_like(amp)
        self.use_fast_asm(True)
        with torch.no_grad():
            phase, _ = self.model(amp, init_phase, self.z, False,
                                  self.pitch, self.wavelength, self.H)
        return phase

    def reconstruct(self, phase):
        """Intensity at the image plane for a phase-only SLM."""
        self.use_fast_asm(True)
        with torch.no_grad():
            slm = torch.complex(torch.cos(phase), torch.sin(phase))
            field = _asm.propagation_ASM(u_in=slm, z=-self.z, linear_conv=False,
                                         feature_size=[self.pitch, self.pitch],
                                         wavelength=self.wavelength,
                                         precomped_H=self.H_back)
            return torch.abs(field)

    def run(self, amp, init_phase=None):
        """Convenience wrapper: target amplitude -> (phase, reconstructed image)."""
        phase = self.holophase(amp, init_phase)
        return phase, self.reconstruct(phase)

    # ------------------------------------------------------------------
    def describe(self):
        return {
            'device': str(self.device),
            'resolution': [self.n, self.m],
            'packed_layers': self.n_packed,
            'packed_backbones': self.backbones_packed,
            'multiplier_cached': self.multiplier_cached,
            'shift_free_asm': self.shift_free,
            'complex_layers_left': count_complex_layers(self.model),
        }
