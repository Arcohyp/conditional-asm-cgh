"""
Optimized Angular Spectrum Method (ASM) propagation module.

This implementation preserves the public API (propagation_ASM / propagation_ASM1 /
propagation_ASM2) while addressing the following issues of the original
implementation:

1. Removes the three near-identical copies of the propagation code.
   A single implementation is selected via `shift_mode`.
2. Builds the transfer function H directly on the target device using
   torch.linspace / torch.meshgrid, avoiding the NumPy -> GPU roundtrip.
3. Caches H (and H_exp) so that repeated calls with the same geometry,
   wavelength, and distance reuse the pre-computed kernel.
"""

import math
import torch
import torch.fft

from conditional_asm_cgh.utils._asm_utils import (
    polar_to_rect,
    ifftshift,
    fftshift,
    pad_image,
    crop_image,
)


# ---------------------------------------------------------------------------
# Cache for transfer functions.
#
# In closed-loop evaluation / TTA, the same (resolution, pixel size,
# wavelength, z, dtype, device, shift_mode) combination is evaluated many
# times.  Rebuilding the frequency grid and the band-limited filter every time
# is the dominant cost for small batches.
# ---------------------------------------------------------------------------
_H_CACHE = {}
_H_EXP_CACHE = {}


def _cache_key(u_in, feature_size, wavelength, z, dtype, shift_mode):
    """Return a hashable key describing the transfer function geometry."""
    num_y, num_x = u_in.size()[-2:]
    dy, dx = feature_size
    return (
        num_y, num_x,
        float(dy), float(dx),
        float(wavelength),
        float(z),
        str(dtype),
        str(u_in.device),
        shift_mode,
    )


def _clear_H_cache():
    """Clear the transfer function caches.  Mostly useful in unit tests."""
    _H_CACHE.clear()
    _H_EXP_CACHE.clear()


def _frequency_grids(num_y, num_x, dy, dx, device, dtype):
    """Return (FX, FY, y_size, x_size) frequency grids on `device`.

    This mirrors the sampling used by the original propagation_ASM.py but
    avoids the NumPy -> CPU -> GPU copy.
    """
    y = dy * float(num_y)
    x = dx * float(num_x)

    # Compute the 1-D frequency coordinates in float64.  This reproduces numpy's
    # default linspace behavior exactly.  The grids are returned as float64 so
    # that the phase kernel HH can be computed in float64 and only cast to the
    # target dtype at the end, matching the original numpy -> torch.float32 path.
    fy = torch.linspace(
        -1 / (2 * dy) + 0.5 / (2 * y),
        1 / (2 * dy) - 0.5 / (2 * y),
        num_y, device=device, dtype=torch.float64,
    )
    fx = torch.linspace(
        -1 / (2 * dx) + 0.5 / (2 * x),
        1 / (2 * dx) - 0.5 / (2 * x),
        num_x, device=device, dtype=torch.float64,
    )
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    return FX, FY, x, y


def _build_H_exp(u_in, feature_size, wavelength, dtype):
    """Build the distance-independent phase kernel H_exp on device."""
    num_y, num_x = u_in.size()[-2:]
    dy, dx = feature_size
    FX, FY, x, y = _frequency_grids(num_y, num_x, dy, dx, u_in.device, dtype)

    # Compute in float64 (matching numpy) and cast to the target dtype at the end.
    HH = 2 * math.pi * torch.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))
    H_exp = HH.reshape(1, 1, num_y, num_x).to(dtype)
    return H_exp, (FX, FY, x, y)


def _build_H(u_in, feature_size, wavelength, z, dtype,
             precomped_H=None, precomped_H_exp=None):
    """Build the ASM band-limited transfer function on u_in.device.

    Returns:
        H: complex transfer function ready to multiply in Fourier domain.
        H_exp: phase exponent after multiplication by z (for optional reuse).
        grids: (FX, FY, x, y) frequency grids, or None if H was supplied.
    """
    if precomped_H is not None:
        return precomped_H, None, None

    if precomped_H_exp is None:
        H_exp, grids = _build_H_exp(u_in, feature_size, wavelength, dtype)
        FX, FY, x, y = grids
    else:
        H_exp = precomped_H_exp
        # precomped_H_exp only carries the phase kernel.  We still need the
        # frequency grids to compute the band-limited filter, so rebuild them
        # cheaply on device.
        num_y, num_x = u_in.size()[-2:]
        dy, dx = feature_size
        FX, FY, x, y = _frequency_grids(num_y, num_x, dy, dx,
                                        u_in.device, dtype)
        grids = (FX, FY, x, y)

    # Multiply by propagation distance.
    H_exp = torch.mul(H_exp, z)

    # Band-limited ASM - Matsushima et al. (2009).
    fy_max = 1 / math.sqrt((2 * z * (1 / y)) ** 2 + 1) / wavelength
    fx_max = 1 / math.sqrt((2 * z * (1 / x)) ** 2 + 1) / wavelength
    H_filter = ((torch.abs(FX) < fx_max) &
                (torch.abs(FY) < fy_max)).to(dtype)

    # Convert polar (magnitude / phase) to rectangular (real / imag).
    H_real, H_imag = polar_to_rect(H_filter, H_exp)

    H = torch.stack((H_real, H_imag), dim=-1)
    # Note: the original propagation_ASM applies ifftshift(H) here, while
    # propagation_ASM1 and propagation_ASM2 do not.  We therefore return the
    # stacked rectangular tensor and let the caller decide whether to shift.
    return H, H_exp, grids


def _finalize_H(H_stacked, shift_mode):
    """Apply the fftshift/ifftshift that the original code applies to H."""
    if shift_mode == 'full':
        H_stacked = ifftshift(H_stacked)
    return torch.view_as_complex(H_stacked)


def _propagation_ASM_core(u_in, feature_size, wavelength, z, linear_conv=True,
                          padtype='zero', return_H=False, precomped_H=None,
                          return_H_exp=False, precomped_H_exp=None,
                          dtype=torch.float32, shift_mode='full'):
    """Single implementation of ASM propagation.

    `shift_mode` selects the fftshift/ifftshift placement:
        'full' : ifftshift before fft, fftshift after ifft  (propagation_ASM)
        'half' : fftshift after fft, ifftshift before ifft  (propagation_ASM1)
        'none' : no shifts                                    (propagation_ASM2)
    """
    if shift_mode not in ('full', 'half', 'none'):
        raise ValueError("shift_mode must be one of 'full', 'half', 'none'")

    if linear_conv:
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        else:
            raise ValueError(f"Unknown padtype: {padtype}")
        u_in = pad_image(u_in, conv_size, padval=padval,
                               stacked_complex=False)

    H_stacked, H_exp, _ = _build_H(u_in, feature_size, wavelength, z, dtype,
                                   precomped_H=precomped_H,
                                   precomped_H_exp=precomped_H_exp)

    if return_H_exp:
        return H_exp

    if precomped_H is not None:
        H = precomped_H
    else:
        H = _finalize_H(H_stacked, shift_mode)

    if return_H:
        return H

    if shift_mode == 'full':
        U1 = torch.fft.fftn(ifftshift(u_in), dim=(-2, -1), norm='ortho')
        U2 = H * U1
        u_out = fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))
    elif shift_mode == 'half':
        U1 = fftshift(torch.fft.fftn(u_in, dim=(-2, -1), norm='ortho'))
        U2 = H * U1
        u_out = torch.fft.ifftn(ifftshift(U2), dim=(-2, -1), norm='ortho')
    else:  # 'none'
        U1 = torch.fft.fftn(u_in, dim=(-2, -1), norm='ortho')
        U2 = H * U1
        u_out = torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho')

    if linear_conv:
        return crop_image(u_out, input_resolution, pytorch=True,
                                stacked_complex=False)
    return u_out


def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=True,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32):
    """Backward-compatible wrapper for the 'full' fftshift variant."""
    return _propagation_ASM_core(
        u_in, feature_size, wavelength, z,
        linear_conv=linear_conv, padtype=padtype,
        return_H=return_H, precomped_H=precomped_H,
        return_H_exp=return_H_exp, precomped_H_exp=precomped_H_exp,
        dtype=dtype, shift_mode='full',
    )


def propagation_ASM1(u_in, feature_size, wavelength, z, linear_conv=True,
                     padtype='zero', return_H=False, precomped_H=None,
                     return_H_exp=False, precomped_H_exp=None,
                     dtype=torch.float32):
    """Backward-compatible wrapper for the 'half' fftshift variant."""
    return _propagation_ASM_core(
        u_in, feature_size, wavelength, z,
        linear_conv=linear_conv, padtype=padtype,
        return_H=return_H, precomped_H=precomped_H,
        return_H_exp=return_H_exp, precomped_H_exp=precomped_H_exp,
        dtype=dtype, shift_mode='half',
    )


def propagation_ASM2(u_in, feature_size, wavelength, z, linear_conv=True,
                     padtype='zero', return_H=False, precomped_H=None,
                     return_H_exp=False, precomped_H_exp=None,
                     dtype=torch.float32):
    """Backward-compatible wrapper for the 'none' fftshift variant."""
    return _propagation_ASM_core(
        u_in, feature_size, wavelength, z,
        linear_conv=linear_conv, padtype=padtype,
        return_H=return_H, precomped_H=precomped_H,
        return_H_exp=return_H_exp, precomped_H_exp=precomped_H_exp,
        dtype=dtype, shift_mode='none',
    )


# ---------------------------------------------------------------------------
# Cached variants for the closed-loop / TTA hot path.
#
# These wrap the core implementation with a small in-memory cache keyed by
# the propagation geometry.  Use them in eval loops where z or wavelength
# are perturbed but repeated many times.
# ---------------------------------------------------------------------------
def propagation_ASM_cached(u_in, feature_size, wavelength, z, linear_conv=True,
                           padtype='zero', return_H=False, precomped_H=None,
                           return_H_exp=False, precomped_H_exp=None,
                           dtype=torch.float32):
    """Cached 'full' fftshift variant.  Reuses H/H_exp across repeated calls."""
    key = _cache_key(u_in, feature_size, wavelength, z, dtype, 'full')

    if return_H_exp:
        if key not in _H_EXP_CACHE:
            _H_EXP_CACHE[key] = _build_H_exp(u_in, feature_size, wavelength,
                                              dtype)[0]
        return _H_EXP_CACHE[key]

    if precomped_H is None and key in _H_CACHE:
        precomped_H = _H_CACHE[key]

    result = _propagation_ASM_core(
        u_in, feature_size, wavelength, z,
        linear_conv=linear_conv, padtype=padtype,
        return_H=return_H, precomped_H=precomped_H,
        return_H_exp=return_H_exp, precomped_H_exp=precomped_H_exp,
        dtype=dtype, shift_mode='full',
    )

    if return_H:
        _H_CACHE[key] = result
    return result
