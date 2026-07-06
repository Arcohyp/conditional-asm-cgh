"""
Minimal sanity check: build each ablation model and run one forward pass.

Run with:
    python -m conditional_asm_cgh.scripts.sanity_check
"""
import torch

from conditional_asm_cgh.models.ccnn_cgh_baseline_minexp import BaselineCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
from conditional_asm_cgh.models.ccnn_cgh_freqcond_minexp import FreqCondCCNNcghModel
from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n, m = 2160, 3840
    pitch = 0.0036
    z = 150.0
    wl = 0.000532

    amp = torch.ones(1, 1, n, m, device=device) * 0.5
    phase = torch.zeros_like(amp)

    # Precompute ASM kernel once.
    H = propagation_ASM(torch.empty(1, 1, n, m, device=device),
                        feature_size=[pitch, pitch], wavelength=wl, z=z,
                        linear_conv=False, return_H=True).to(device)

    models = {
        'unconditional': BaselineCCNNcghModelMinExp(),
        'spatial-conditional': ConditionalCCNNcghModelMinExp(),
        'frequency-conditional': FreqCondCCNNcghModel(),
    }

    for name, model in models.items():
        model = model.to(device).eval()
        with torch.no_grad():
            holophase, _ = model(amp, phase, z, False, pitch, wl, H)
        print(f'{name}: output shape {holophase.shape}, device {holophase.device}')

    print('Sanity check passed.')


if __name__ == '__main__':
    main()
