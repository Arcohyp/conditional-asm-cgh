# Exploratory: SFO behavior outside its training window (z scan)

**Status: the summarized result (mean/min–max band over the three per-color
weights, under SFO's native convention) appears in the manuscript as the gray
band of Fig. 9(a) and in Sec. 4.5 / the response to Major revision 4; the
per-cell JSONs here allow every point of that band to be reproduced.**
Question posed 2026-09-10: how do the released SFO weights behave at
propagation distances outside their training window z ∈ [85, 115] mm?

## Method

- `scan_sfo_z.py <gpu> <r|g|b>`: released SFO weights
  (`CGH-SFO-solver/model/{r,g,b}/FourierNet_flex_100.pth`), exact SFO
  pipeline (Flex_FourierNet + ASM_split, pitch 3.74 µm, true z fed both to
  the model's distance input and to ASM propagation), 100 DIV2K validation
  images per point, z = 40..200 mm in 5 mm steps (33 points).
- Metrics per point: SFO's own convention (skimage intensity PSNR/SSIM with
  per-image sum scaling) plus sum-scaled amplitude-domain PSNR on
  sqrt-intensities. Shaded band = per-image std.
- Note: SFO's `prop.py` builds frequency grids on the *current* CUDA device;
  `torch.cuda.set_device(gpu)` is required before model/prop construction.

## Results (`sfo_zscan_{r,g,b}.json`, `sfo_zscan_summary.png`)

SFO behaves as a perfect top-hat in z — excellent inside the window,
collapsed immediately outside it:

| z (mm) | Red PSNR (dB) | Green | Blue |
|--------|---------------|-------|------|
| 80     | 20.2          | 24.1  | 21.9 |
| 84     | 20.9          | 27.3  | 14.7 |
| 85     | 32.4          | 33.9  | 32.5 |
| 100    | 33.1          | 34.7  | 32.7 |
| 115    | 32.7          | 33.7  | 32.4 |
| 116    | 21.0          | 27.2  | 14.9 |
| 120    | 20.3          | 24.0  | 22.3 |
| 130    | 18.2          | 20.1  | 20.5 |
| 150    | 17.7          | 18.8  | 19.4 |
| 200    | 17.0          | 16.0  | 17.8 |

- In-window means: 32.83 (r) / 34.25 (g) / 32.59 (b) dB — flat across the
  window, no comb structure (SFO's explicit distance conditioning
  interpolates well within the trained range).
- **Extrapolation is exactly zero**: a dense 1 mm re-scan of the transition
  regions (`scan_sfo_z_dense.py`, z = 76–134 mm, `sfo_zscan_dense_*.json`)
  shows the drop completes within 1 mm of the window boundary —
  e.g. red falls 32.7 → 21.0 dB between z = 115 and 116 mm, and blue shows
  a transient dip to 14.9 dB at z = 116 mm. The transition is a step
  discontinuity at the training boundary, not a slope.
- Usable envelope ≈ the training window itself, ±0 mm tolerance.

## Contrast with our conditioned model

Our single spatial-conditional weight (trained on the same [85, 115] mm
window for the M2 comparison, 44,221 parameters shared across colors) shows
the opposite asymmetry: far-edge extrapolation is graceful — all three
colors stay above 25 dB out to z = 265 mm (worst 26.5 dB over
206–265 mm, manuscript Sec. 4.5 / Fig. 9), i.e. +150 mm beyond the window,
while the near edge degrades below 25 dB at z ≈ 89/106/125 mm
(r/g/b). SFO's per-color 109 M-parameter specialists cover distance
**interpolation only**, within ±5 mm of the window.

Interpretation: SFO's distance input is a network conditioning signal that
the model learns to trust only where supervised; nothing in its training
ties the output phase to the ASM physics of unseen distances. Our
conditioning is likewise input-side, but the training range and the
single-weight (z, λ) coverage still leave the far-edge ASM transfer
extrapolable — consistent with the manuscript's Limitations point that
short-distance extrapolation is the fragile direction for both approaches.

## Related calibration observation (supports manuscript Sec. 4.3)

SFO's propagated output is not absolutely amplitude-calibrated: at z = 100 mm
(red, in-window) the per-image sum-recovery scale target.sum/final.sum
averages ≈ 2.14× (peak amplitude ratio final/target ≈ 0.49), and the
attenuation grows with 1/λ (r < g ≈ b ordering), which is why SFO scores
13.9/7.3/7.3 dB under the fixed-peak amplitude convention vs our 27.78 dB.
Measured by `eval_table5_amp.py` / `verify_sfo_scale.py` (in `../`).
