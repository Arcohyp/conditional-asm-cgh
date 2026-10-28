"""Shared helpers for the deployment benchmarks.

Everything here is deliberately dependency-free (torch + numpy + the repo's own
utilities) so that a reader can reproduce a number with one command.
"""

import json
import os
import time

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEIGHTS = {
    'spatial': os.path.join(REPO_ROOT, 'weights', 'm3_seeds', 'spatial_s42.pth'),
    'freq': os.path.join(REPO_ROOT, 'weights', 'm3_seeds', 'frequency_s42.pth'),
    'uncond': os.path.join(REPO_ROOT, 'weights', 'm3_seeds', 'unconditional_s42.pth'),
    'spatial_main': os.path.join(REPO_ROOT, 'weights', 'spatial_conditional.pth'),
    'freq_main': os.path.join(REPO_ROOT, 'weights', 'frequency_conditional.pth'),
}

RESOLUTIONS = {
    '720p': (720, 1280),
    '1080p': (1080, 1920),
    '1440p': (1440, 2560),
    '4k': (2160, 3840),
}


def parse_resolution(text):
    """'2160x3840' or '4k' -> (2160, 3840)."""
    key = text.strip().lower()
    if key in RESOLUTIONS:
        return RESOLUTIONS[key]
    n, m = key.split('x')
    return int(n), int(m)


# ---------------------------------------------------------------------------
# metrics (identical to scripts/eval.py so numbers are comparable)
# ---------------------------------------------------------------------------
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def ssim(pred, target, window_size=11):
    import torch.nn.functional as F
    sigma = 1.5
    channel = pred.size(1)
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.expand(channel, 1, window_size, window_size).contiguous().to(pred.device)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def build_model(kind, device='cuda', weight=None):
    """Instantiate one of the released models, optionally loading a checkpoint."""
    from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
    from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel
    from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp

    if kind == 'spatial':
        model = ConditionalCCNNcghModelMinExp(param_dim=16)
    elif kind == 'freq':
        model = FreqCondCCNNcghModel(n_freqs=16, cond_dim=64, base_res=128, radial_len=128)
    elif kind == 'uncond':
        model = BaselineCCNNcghModelMinExp()
    else:
        raise ValueError('unknown model %r' % kind)

    weight = weight or WEIGHTS[kind]
    if weight and os.path.exists(weight):
        ckpt = torch.load(weight, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(
            'checkpoint not found: %s\nPass --weight explicitly.' % weight)
    return model.to(device).eval()


def fixed_amp(n, m, device, value=0.7):
    """Deterministic target amplitude, used for timing runs."""
    return torch.full((1, 1, n, m), value, device=device)


def div2k_amplitude(root, index, n, m, device):
    """DIV2K image -> target amplitude sqrt(linear RGB), as in scripts/eval.py."""
    from conditional_asm_cgh.utils.tools import loadimage
    img = loadimage(root, index, channel=2, flip=0, m=m, n=n, convert=True, cuda=False)
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    return torch.sqrt(img.view(1, 1, n, m).float().to(device))


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def timeit(fn, warmup=3, reps=10, synchronize=False):
    """Median wall-clock time of `fn` in milliseconds."""
    for _ in range(warmup):
        fn()
    if synchronize:
        torch.cuda.synchronize()
    stamps = []
    for _ in range(reps):
        if synchronize:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if synchronize:
            torch.cuda.synchronize()
        stamps.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(stamps)), float(np.min(stamps)), float(np.max(stamps))


def dump(obj, path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    print('[json] %s' % path)
