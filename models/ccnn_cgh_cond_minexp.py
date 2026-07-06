import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexConvTranspose2d, ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM


class CDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, x):
        return complex_relu(self.COV1(x))


class CUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))

    def forward(self, x):
        return complex_relu(self.COV1(x))


class CUp2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))

    def forward(self, x):
        return self.COV1(x)


def _match_complex_size(target, source):
    t_h, t_w = target.shape[-2:]
    s_h, s_w = source.shape[-2:]
    if s_h != t_h or s_w != t_w:
        source = torch.complex(
            F.interpolate(source.real, size=(t_h, t_w), mode='bilinear', align_corners=False),
            F.interpolate(source.imag, size=(t_h, t_w), mode='bilinear', align_corners=False)
        )
    return source


class ParamEncoder(nn.Module):
    """Encode physical parameters (z, wavelength, pitch) into a condition vector."""

    def __init__(self, param_dim=16, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, param_dim),
        )

    def forward(self, z, wavelength, pitch):
        # z in mm, wavelength in mm, pitch in mm
        if self.normalize:
            z_norm = (z - 150.0) / 10.0
            wl_norm = (wavelength - 0.000638) / 1e-5
            pitch_norm = (pitch - 0.0036) / 1e-4
            x = torch.stack([z_norm, wl_norm, pitch_norm], dim=-1)
        else:
            x = torch.stack([z, wavelength, pitch], dim=-1)
        return self.mlp(x)


class CCNN1(nn.Module):
    """Original CCNN1 unchanged."""

    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)
        self.netdown4 = CDown(16, 32)

        self.netup4 = CUp(32, 16)
        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        target_h, target_w = x.shape[-2], x.shape[-1]
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)
        out4 = self.netdown4(out3)

        out17 = self.netup4(out4)
        out3_matched = _match_complex_size(out17, out3)
        out18 = self.netup3(out17 + out3_matched)
        out2_matched = _match_complex_size(out18, out2)
        out19 = self.netup2(out18 + out2_matched)
        out1_matched = _match_complex_size(out19, out1)
        out20 = self.netup1(out19 + out1_matched)

        if out20.shape[-2:] != (target_h, target_w):
            out20 = torch.complex(
                F.interpolate(out20.real, size=(target_h, target_w), mode='bilinear', align_corners=False),
                F.interpolate(out20.imag, size=(target_h, target_w), mode='bilinear', align_corners=False)
            )

        predictphase = torch.atan2(out20.imag, out20.real)
        return predictphase


class CCNN2(nn.Module):
    """Original CCNN2 unchanged."""

    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)

        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)

        out18 = self.netup3(out3)
        out2_matched = _match_complex_size(out18, out2)
        out19 = self.netup2(out18 + out2_matched)
        out1_matched = _match_complex_size(out19, out1)
        out20 = self.netup1(out19 + out1_matched)

        holophase = torch.atan2(out20.imag, out20.real)
        return holophase


class ConditionalCCNNcghModelMinExp(nn.Module):
    """CCNN-CGH conditioned on physical parameters z, wavelength, pitch.

    Minimal change: a small real-valued fusion network combines target amplitude
    with broadcasted physical-parameter features, then the original CCNN1/CCNN2
    generate the hologram.
    """

    def __init__(self, param_dim=16, name='conditional_ccnncgh'):
        super().__init__()
        self.name = name
        self.param_dim = param_dim
        self.param_encoder = ParamEncoder(param_dim=param_dim)

        # Fuse [amp, param_broadcast] into a single channel, then feed original CCNN1
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

        # Fuse amplitude and parameter map
        fused = self.input_fusion(torch.cat([amp, param_map], dim=1))

        target_complex = torch.complex(fused * torch.cos(phase), fused * torch.sin(phase))
        predict_phase = self.ccnn1(target_complex)

        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))

        # Use scalar physical parameters for ASM
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
