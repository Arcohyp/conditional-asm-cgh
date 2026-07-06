"""
Utility helpers for ASM propagation, adapted from Neural Holography utils.

Original credit:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with
Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular."""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def roll_torch(tensor, shift, axis):
    """Implements numpy roll() / Matlab circshift() for tensors."""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def ifftshift(tensor):
    """ifftshift for tensors of shape [..., H, W, 2]."""
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[-3] / 2.0), dim=-3)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[-2] / 2.0), dim=-2)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of shape [..., H, W, 2]."""
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[-3] / 2.0), dim=-3)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[-2] / 2.0), dim=-2)
    return tensor_shifted


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size."""
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                if padval == 0:
                    pad_width = (0, 0, *pad_axes)
                    return F.pad(field, pad_width, mode=mode)
                else:
                    real, imag = field[..., 0], field[..., 1]
                    real = F.pad(real, pad_axes, mode=mode, value=padval)
                    imag = F.pad(imag, pad_axes, mode=mode, value=0)
                    return torch.stack((real, imag), -1)
            else:
                return F.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field to target_shape."""
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field
