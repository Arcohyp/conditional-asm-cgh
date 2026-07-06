import sys
import os

import torch
import torch.nn as nn
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp


class LambdaRouter(nn.Module):
    """Soft routing weights over three wavelength experts based on lambda.

    Boundaries are chosen to place the three native wavelengths near the centre
    of each expert's region, with blending zones at the edges so the model is
    continuous in lambda.
    """

    def __init__(self, boundaries=(0.000495, 0.000585), soft_width=5e-6):
        super().__init__()
        self.register_buffer('b1', torch.tensor(boundaries[0]))
        self.register_buffer('b2', torch.tensor(boundaries[1]))
        self.register_buffer('soft_width', torch.tensor(soft_width))

    def forward(self, wavelength):
        # wavelength is a scalar tensor or python float in mm
        if not isinstance(wavelength, torch.Tensor):
            wavelength = torch.tensor(wavelength, dtype=torch.float32)
        x = wavelength
        sw = self.soft_width
        # weights for [blue, green, red]
        w_blue = torch.sigmoid((self.b1 - x) / sw)
        w_red = torch.sigmoid((x - self.b2) / sw)
        w_green = 1.0 - w_blue - w_red
        weights = torch.stack([w_blue, w_green, w_red], dim=-1)  # [B, 3]
        return weights


class ThreeExpertCCNNcghModelMinExp(nn.Module):
    """Three independent (z,lambda)-conditioned experts with lambda-dependent blending.

    This tests whether the single-model multi-wavelength bottleneck is mainly the
    requirement that one shared backbone fit all three colours, rather than the
    capacity of the individual CCNN U-Net.
    """

    def __init__(self, param_dim=16, boundaries=(0.000495, 0.000585), soft_width=5e-6):
        super().__init__()
        self.name = 'three_expert_ccnncgh'
        self.router = LambdaRouter(boundaries=boundaries, soft_width=soft_width)
        self.expert_blue = ConditionalCCNNcghModelMinExp(param_dim=param_dim, name='expert_blue')
        self.expert_green = ConditionalCCNNcghModelMinExp(param_dim=param_dim, name='expert_green')
        self.expert_red = ConditionalCCNNcghModelMinExp(param_dim=param_dim, name='expert_red')

    def forward(self, amp, phase, z, pad, pitch, wavelength, H):
        batch_size = amp.size(0)
        device = amp.device
        route = self.router(wavelength)  # [B, 3]

        # Each expert produces its own hologram phase (and an auxiliary predict phase).
        holo_blue, _ = self.expert_blue(amp, phase, z, pad, pitch, wavelength, H)
        holo_green, _ = self.expert_green(amp, phase, z, pad, pitch, wavelength, H)
        holo_red, _ = self.expert_red(amp, phase, z, pad, pitch, wavelength, H)

        # Blend holophases weighted by lambda routing.
        # Shape [B, 1, H, W].
        w = route.view(batch_size, 3, 1, 1, 1)
        stacked = torch.stack([holo_blue, holo_green, holo_red], dim=1)  # [B, 3, 1, H, W]
        holophase = (stacked * w).sum(dim=1)  # [B, 1, H, W]
        return holophase, None
