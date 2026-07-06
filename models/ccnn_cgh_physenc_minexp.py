import sys
import os

import math
import torch
import torch.nn as nn

from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import CCNN1, CCNN2


def physics_embed(z, wavelength, pitch, n_freqs):
    """SFO-style physics-derived positional encoding.

    Frequency bands span the axial-wavenumber range from the angular-spectrum
    dispersion relation, so the encoding cos/sin(z * k_z(lambda, f)) is the
    actual ASM propagation phase. lambda enters analytically via max/min freq,
    making the encoding jointly (z, lambda)-aware.

    z, wavelength, pitch: tensors shape [B], all in mm.
    returns: [B, 2*n_freqs]
    """
    two_pi = 2.0 * math.pi
    max_fre = two_pi / wavelength
    min_fre = two_pi / wavelength * torch.sqrt(
        torch.clamp(1.0 - 2.0 * (wavelength / pitch / 2.0) ** 2, min=1e-8)
    )
    steps = torch.linspace(1, n_freqs, steps=n_freqs, device=z.device, dtype=z.dtype)
    # freq_bands: [B, n_freqs]
    freq_bands = (max_fre - min_fre).unsqueeze(-1) / n_freqs * steps.unsqueeze(0) + min_fre.unsqueeze(-1)
    scaled = z.unsqueeze(-1) * freq_bands
    return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)


class PhysEmbedEncoder(nn.Module):
    """Drop-in replacement for ParamEncoder: physics embedding -> param_dim."""

    def __init__(self, param_dim=16, n_freqs=8):
        super().__init__()
        self.n_freqs = n_freqs
        self.proj = nn.Sequential(
            nn.Linear(2 * n_freqs, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, param_dim),
        )

    def forward(self, z, wavelength, pitch):
        enc = physics_embed(z, wavelength, pitch, self.n_freqs)
        return self.proj(enc)


class PhysCondCCNNcghModel(nn.Module):
    """Stage A1: identical to ConditionalCCNNcghModelMinExp but the parameter
    encoder is the physics-derived embedding. Input-end fusion is unchanged so
    this isolates 'physics encoding vs plain MLP encoding'.
    """

    def __init__(self, param_dim=16, n_freqs=8, name='physcond_ccnncgh'):
        super().__init__()
        self.name = name
        self.param_dim = param_dim
        self.param_encoder = PhysEmbedEncoder(param_dim=param_dim, n_freqs=n_freqs)

        self.input_fusion = nn.Sequential(
            nn.Conv2d(1 + param_dim, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.ccnn1 = CCNN1()
        self.ccnn2 = CCNN2()

    def _encode_params(self, z, wavelength, pitch, batch_size, device):
        if isinstance(z, float):
            z = torch.tensor(z, dtype=torch.float32, device=device)
        if isinstance(wavelength, float):
            wavelength = torch.tensor(wavelength, dtype=torch.float32, device=device)
        if isinstance(pitch, float):
            pitch = torch.tensor(pitch, dtype=torch.float32, device=device)

        if z.dim() == 0:
            z = z.unsqueeze(0).expand(batch_size)
        if wavelength.dim() == 0:
            wavelength = wavelength.unsqueeze(0).expand(batch_size)
        if pitch.dim() == 0:
            pitch = pitch.unsqueeze(0).expand(batch_size)

        return self.param_encoder(z, wavelength, pitch)

    def forward(self, amp, phase, z, pad, pitch, wavelength, H):
        batch_size = amp.size(0)
        device = amp.device
        h, w = amp.shape[-2:]

        param = self._encode_params(z, wavelength, pitch, batch_size, device)
        param_map = param.view(batch_size, self.param_dim, 1, 1).expand(batch_size, self.param_dim, h, w)

        fused = self.input_fusion(torch.cat([amp, param_map], dim=1))

        target_complex = torch.complex(fused * torch.cos(phase), fused * torch.sin(phase))
        predict_phase = self.ccnn1(target_complex)

        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))

        z_val = z if isinstance(z, (int, float)) else z[0].item()
        pitch_val = pitch if isinstance(pitch, (int, float)) else pitch[0].item()
        wl_val = wavelength if isinstance(wavelength, (int, float)) else wavelength[0].item()

        slmfield = propagation_ASM(
            u_in=predict_complex, z=z_val, linear_conv=pad,
            feature_size=[pitch_val, pitch_val],
            wavelength=wl_val,
            precomped_H=H
        )

        holophase = self.ccnn2(slmfield)
        return holophase, predict_phase
