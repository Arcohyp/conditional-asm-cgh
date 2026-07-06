import sys
import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import CCNN1, CCNN2
from conditional_asm_cgh.models.ccnn_cgh_physenc_minexp import physics_embed


class ConditionalFreqMultiplier(nn.Module):
    """Stage B mechanism (SFO Flex_Fourier_filter范式, 物理化+稳定版).

    Conditions a complex field in the frequency domain on (z, lambda) by a
    *unit-modulus phase* multiplier -- mirroring ASM physics where the transfer
    function H is pure phase, so this can never attenuate/annihilate the field:
      M(f) = exp(i * dphi(f; z, lambda))
      X = fft2(field);  X = X * ifftshift(M);  field = ifft2(X)
    dphi = base_phase(f) + cond_phase(f; z, lambda), BOTH zero-initialised so at
    init M = 1 (identity pass-through) and the base CCNN still works; training
    then learns a (z,lambda)-dependent phase correction to the angular spectrum.
    - base_phase: truncated-resolution learnable phase map, bilinearly upsampled
      (mode-truncation -> lightweight).
    - cond_phase: physics embedding -> MLP (last layer zero-init) -> radial phase
      profile -> broadcast onto a 2D grid by radius (ASM circular symmetry).
    """

    def __init__(self, cond_dim, base_res=64, radial_len=64):
        super().__init__()
        self.base_res = base_res
        self.radial_len = radial_len
        # learnable base phase [1, base_res, base_res], zero-init -> identity
        self.base_phase = nn.Parameter(torch.zeros(1, base_res, base_res))
        last = nn.Linear(128, radial_len)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.to_radial = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(inplace=True),
            last,
        )

    def _radial_to_2d(self, radial, h, w, device):
        # radial: [B, L] (real phase profile) -> [B, h, w] sampled by radius
        B = radial.size(0)
        L = self.radial_len
        prof = radial.view(B, 1, 1, L)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij',
        )
        r = torch.sqrt(xx ** 2 + yy ** 2) / math.sqrt(2.0)  # [h,w] in [0,1]
        gx = (r * 2 - 1).unsqueeze(0).expand(B, h, w)
        gy = torch.zeros_like(gx)
        grid = torch.stack([gx, gy], dim=-1)
        sampled = F.grid_sample(prof, grid, mode='bilinear',
                                align_corners=True, padding_mode='border')  # [B,1,h,w]
        return sampled[:, 0]  # [B,h,w]

    def forward(self, field, cond):
        B, C, h, w = field.shape
        device = field.device
        Xf = fft.fft2(field, dim=(2, 3), norm='ortho')

        base = self.base_phase
        if base.shape[1:3] != (h, w):
            base = F.interpolate(base.unsqueeze(1), size=(h, w),
                                 mode='bilinear', align_corners=True).squeeze(1)
        cond_phase = self._radial_to_2d(self.to_radial(cond), h, w, device)  # [B,h,w]
        dphi = base + cond_phase  # [B,h,w]
        M = torch.exp(1j * dphi.to(torch.float32)).unsqueeze(1)  # [B,1,h,w] unit modulus
        M = fft.ifftshift(M, dim=(-2, -1))

        Xf = Xf * M
        return fft.ifft2(Xf, dim=(2, 3), norm='ortho')


class FreqCondCCNNcghModel(nn.Module):
    """Stage B: NO input-end fusion. CCNN1 predicts SLM phase from the target
    complex (as in baseline); a (z,lambda)-conditioned frequency-domain
    multiplier corrects the complex SLM field right before internal ASM #1 --
    the headroom-max location identified in baseline doc section 5.
    """

    def __init__(self, n_freqs=8, cond_dim=64, base_res=64, radial_len=64,
                 name='freqcond_ccnncgh'):
        super().__init__()
        self.name = name
        self.n_freqs = n_freqs
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * n_freqs, cond_dim),
            nn.ReLU(inplace=True),
        )
        self.freq_mult = ConditionalFreqMultiplier(cond_dim, base_res, radial_len)
        self.ccnn1 = CCNN1()
        self.ccnn2 = CCNN2()

    def _cond(self, z, wavelength, pitch, batch_size, device):
        def _to_b(v):
            if isinstance(v, float):
                v = torch.tensor(v, dtype=torch.float32, device=device)
            if v.dim() == 0:
                v = v.unsqueeze(0).expand(batch_size)
            return v
        z, wavelength, pitch = _to_b(z), _to_b(wavelength), _to_b(pitch)
        enc = physics_embed(z, wavelength, pitch, self.n_freqs)
        return self.cond_proj(enc)

    def forward(self, amp, phase, z, pad, pitch, wavelength, H):
        batch_size = amp.size(0)
        device = amp.device

        cond = self._cond(z, wavelength, pitch, batch_size, device)

        target_complex = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
        predict_phase = self.ccnn1(target_complex)
        predict_complex = torch.complex(amp * torch.cos(predict_phase),
                                        amp * torch.sin(predict_phase))

        # (z, lambda)-conditioned frequency-domain correction before ASM #1
        predict_complex = self.freq_mult(predict_complex, cond)

        z_val = z if isinstance(z, (int, float)) else (z[0].item() if torch.is_tensor(z) else z)
        pitch_val = pitch if isinstance(pitch, (int, float)) else pitch
        wl_val = wavelength if isinstance(wavelength, (int, float)) else wavelength

        slmfield = propagation_ASM(
            u_in=predict_complex, z=z_val, linear_conv=pad,
            feature_size=[pitch_val, pitch_val],
            wavelength=wl_val,
            precomped_H=H
        )

        holophase = self.ccnn2(slmfield)
        return holophase, predict_phase
