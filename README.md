# Conditional-ASM-CGH

A lightweight, single-weight complex-valued U-Net CGH model conditioned on propagation distance $z$, wavelength $\lambda$, and pixel pitch. The model covers the ASM self-imaging parameter plane with only **44,221 parameters**, replacing the conventional one-model-per-$(z,\lambda)$ pipeline.

This repository accompanies the manuscript:

> **A lightweight complex-valued U-Net CGH for joint distance–wavelength coverage of the ASM parameter plane**  
> Xirun Cheng, Yong Liu, Dapeng Sun, Quan Wang, Chaofan Zhang, Zhenyu Gao

Code and trained weights are linked to the manuscript.

## Highlights

- **Single shared weight** for RGB (450/532/638 nm) across $z=145$–205 mm.
- **44k parameters**, 0.20 MB checkpoint, 8-bit phase quantization compatible.
- Three ablation models: unconditional, spatial-conditional (proposed), frequency-conditional.
- 4K DIV2K validation; PSNR/SSIM evaluation, engineering benchmark, and figure generation scripts included.

## First revision (September 2026)

The revised manuscript reports re-measured results under a unified protocol. The
supporting checkpoints and scripts are included in this repository:

- **3-seed ablation checkpoints** (`weights/m3_seeds/`): all three ablation models
  retrained from seeds 42/43/44 with the identical full recipe (continuous wavelength
  sampling plus two-axis peak anchoring). Revised Tables 3–4 and Figs. 4–5 report
  mean $\pm$ std over these seeds. Under the unified recipe the frequency-conditional
  and spatial-conditional models are statistically tied; the spatial-conditional form
  reaches the same coverage at roughly half the parameters (44,221 vs 85,588).
- **SFO-window model** (`weights/sfo_window_spatial.pth`): the spatial-conditional
  model trained on the SFO baseline's own operating window (pitch 3.74 µm,
  450/520/638 nm, $z\in[85,115]$ mm). On SFO's intensity-domain evaluation
  convention it scores 27.08 dB PSNR / 0.764 SSIM against the released SFO
  weights' 33.12 dB / 0.883 on the same 100 DIV2K images (Table 5, Fig. 6).
- **Revision experiment scripts** (`scripts/revision/`): seed retraining, full
  evaluation suite, extrapolation scans beyond the training window (safe envelope
  $z\in[125,265]$ mm for the blue channel), the SFO comparison, and the RGB
  composite rendering.

## Installation

```bash
git clone https://github.com/Arcohyp/conditional-asm-cgh.git
cd conditional-asm-cgh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** This code requires `complexPyTorch` for complex-valued convolutions. Install it from the official repository:

```bash
pip install git+https://github.com/wavefrontshaping/complexPyTorch.git
```

## Quick sanity check

After installing `complexPyTorch`, verify that all three ablation models can be instantiated and run a forward pass:

```bash
python -m conditional_asm_cgh.scripts.sanity_check
```

## Repository structure

```
conditional_asm_cgh/
├── models/              # Model definitions (unconditional, spatial-conditional, frequency-conditional)
├── utils/               # ASM propagation and image loading utilities
├── scripts/             # Training, evaluation, figure generation
├── configs/             # Placeholder for training configurations
├── figures/             # Generated figures
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

### 1. Inference with the proposed spatial-conditional model

```python
import torch
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
from conditional_asm_cgh.utils.tools import loadimage
from conditional_asm_cgh.utils.propagation_ASM import propagation_ASM

model = ConditionalCCNNcghModelMinExp()
model.load_state_dict(torch.load('path/to/spatial_conditional.pth')['model_state_dict'])
model.cuda().eval()

amp = loadimage('path/to/DIV2K', image_index=801, m=2160, n=3840, cuda=True)
phase = torch.zeros_like(amp)
with torch.no_grad():
    holophase, _ = model(amp, phase, z=150.0, pad=True, pitch=0.0036,
                         wavelength=0.000532, H=None)
```

### 2. Training

Train the spatial-conditional model from a warm-started blue baseline:

```bash
python -m conditional_asm_cgh.scripts.stage1_train \
  --data-path /path/to/DIV2K_train \
  --outdir ./experiments/stage1_blue \
  --z 150.0 --wavelength 0.000450 --pitch 0.0036 \
  --device-id 0
```

Then train the full spatial-conditional model:

```bash
bash scripts/run_colorcond_npcc.sh
```

### 3. Evaluation

```bash
python -m conditional_asm_cgh.scripts.eval \
  --data-path /path/to/DIV2K_valid \
  --cond-weight /path/to/spatial_conditional.pth \
  --z-start 145 --z-end 205 --z-step 1 \
  --wavelengths 0.000450 0.000532 0.000638 \
  --device-id 0
```

### 4. Reproduce paper figures

```bash
python -m conditional_asm_cgh.scripts.make_paper_figures_a3 \
  --data-path /path/to/DIV2K_valid \
  --cond-weight /path/to/spatial_conditional.pth \
  --baseline-weights /path/to/specialists \
  --outdir ./figures
```

## Reproduction checklist

To reproduce the main results in the paper, run the shell scripts in this order:

1. `scripts/run_colorcond_npcc.sh` — train spatial-conditional model.
2. `scripts/run_ablation_A1_nocond.sh` — train unconditional ablation.
3. `scripts/run_colorcond_contlambda.sh` — continuous-wavelength variant.
4. `scripts/run_baseline_quant.sh` — 8-bit/6-bit/4-bit quantization study.
5. `scripts/bench_engineering.py` — latency and memory benchmark.

See `scripts/` for all training and evaluation entry points.

## Data

The training and validation use the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Place the dataset at any location and pass the path via `--data-path`.

## Trained weights

The main model checkpoints used in the paper are provided under `weights/`:

| Checkpoint | Model | Params | Description |
|------------|-------|--------|-------------|
| `weights/spatial_conditional.pth` | Spatial-conditional (proposed) | 44,221 | Main model; one weight for all $(z, \lambda)$ in the paper range |
| `weights/unconditional.pth` | Unconditional ablation | 42,260 | Same backbone without parameter conditioning |
| `weights/frequency_conditional.pth` | Frequency-conditional ablation | 85,588 | Fourier-domain radial multiplier baseline |
| `weights/m3_seeds/spatial_s{42,43,44}.pth` | Spatial-conditional | 44,221 | 3-seed unified-recipe retraining (revision Tables 3–4, Figs. 4–5) |
| `weights/m3_seeds/frequency_s{42,43,44}.pth` | Frequency-conditional | 85,588 | 3-seed unified-recipe retraining (revision Tables 3–4, Figs. 4–5) |
| `weights/m3_seeds/unconditional_s{42,43,44}.pth` | Unconditional | 42,260 | 3-seed unified-recipe retraining (revision Tables 3–4, Figs. 4–5) |
| `weights/sfo_window_spatial.pth` | Spatial-conditional (SFO window) | 44,221 | Trained on SFO's operating window (pitch 3.74 µm, $z\in[85,115]$ mm) for the revision head-to-head comparison (Table 5, Fig. 6) |

Load them with the matching model class:

```python
from conditional_asm_cgh.models.ccnn_cgh_cond_minexp import ConditionalCCNNcghModelMinExp
import torch

model = ConditionalCCNNcghModelMinExp()
ckpt = torch.load('weights/spatial_conditional.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
```

The checkpoints store both `model_state_dict` and training metadata; only `model_state_dict` is needed for inference.

## Citation

```bibtex
@article{cheng2026conditional,
  title={A lightweight complex-valued U-Net CGH for joint distance--wavelength coverage of the ASM parameter plane},
  author={Cheng, Xirun and Liu, Yong and Sun, Dapeng and Wang, Quan and Zhang, Chaofan and Gao, Zhenyu},
  year={2026},
  note={Manuscript under review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or dataset/weight requests, please contact the corresponding authors listed in the paper.
