# Revision experiments (first revision, September 2026)

Scripts behind the new experiments in the revised manuscript (Manuscript ID 610780).
They are kept close to the form in which they were run on the authors' cluster;
absolute paths, GPU IDs, and the conda environment refer to that setup and need
to be adapted for other machines.

| Script | Content | Manuscript location |
|--------|---------|---------------------|
| `run_m3_seeds.sh` | Re-train all three ablation models (unconditional / spatial-conditional / frequency-conditional) from 3 seeds (42/43/44) under one unified recipe (continuous wavelength sampling + two-axis peak anchoring) | Tables 3, 4; Figs. 4, 5 |
| `run_m3_evals.sh` | Full evaluation suite of the M3 models: 61-point distance scan per color + 63-point wavelength fine scan, 100 DIV2K images per point; per-point JSONs in `data_m3_evals/` | Tables 2, 3, 4; Figs. 4, 5 |
| `compute_m3_stats.py` | Aggregate the M3 eval JSONs into mean $\pm$ std statistics | Tables 3, 4 |
| `make_m3_figures.py` | Draw Figs. 4 and 5 (3-seed mean $\pm$ std bands) from the M3 eval JSONs | Figs. 4, 5 |
| `run_m4.sh` | Extrapolation scans with a single trained weight: distance axis 85--265 mm, wavelength axis beyond the 440--650 nm training band; per-point JSONs in `data_fig9_zscan/` | Sec. 4.6, Fig. 9(a) |
| `run_m4b_lambda_dense.sh` | Dense full-band wavelength scan at z = 150 mm (410--680 nm, 2 nm steps, 3 GPUs by wavelength range) that fills the gaps of the M3 finescan; merged per-point results in `data_lambda_dense/` | Fig. 9(c), Sec. 4.6 |
| `run_m4c_envelope_shift.sh`, `run_m4c_envelope_far.sh` | Scan the SFO-window weight (released as `weights/sfo_window_spatial.pth`) over z = 45--230 mm and z = 231--300 mm, 3 colors, to test whether the near-side collapse boundary follows the training window's $\lambda z$ band; per-point JSONs in `data_m4c/` | Sec. 4.6 (second-weight validation of the $\lambda z$ boundary) |
| `run_m4d_lambda_lowz.sh`, `analyze_m4d_lambda_lowz.py` | Wavelength-axis cross-check of the $\lambda z$ boundary at two fixed distances below the training window (z = 110/120 mm); merged results in `data_m4d/` | Sec. 4.6 (468 nm at z = 120 mm, 512 nm at z = 110 mm) |
| `make_fig9_v8_draft.py` | Generate the final three-panel Fig. 9 (our distance scan; released SFO weights over z = 40--200 mm; full-band wavelength scan) | Fig. 9 |
| `run_m2.sh`, `run_m2b.sh`, `run_m2c.sh`, `run_m2d.sh` | Progressive training of the spatial-conditional model on the SFO operating window (pitch 3.74 um, 450/520/638 nm, z in [85,115] mm) for the head-to-head comparison | Sec. 4.3, Table 5, Fig. 6 |
| `eval_ours_m2.py` | Evaluate our SFO-window model per cell in both evaluation conventions (SFO intensity-domain skimage and our amplitude-domain) | Table 5, Fig. 6 |
| `eval_sfo_psnr_ssim.py` | Re-evaluate the released SFO weights with SFO's own convention on the same 100 DIV2K images | Table 5, Fig. 6 |
| `eval_table5_amp.py` | Amplitude-domain fixed-peak PSNR/SSIM (no per-image scaling) for both models on the shared window; per-color results in `data_table5/` | Table 5 (amplitude-convention column) |
| `verify_sfo_scale.py` | Measure SFO's per-image sum-recovery scale at z = 100 mm (evidence that the released SFO field is not absolutely calibrated) | Sec. 4.3 |
| `verify_sfo_global_cal.py` | Least-squares single global amplitude constant for SFO; restores its fixed-peak score from 13.8 to 31.3 dB (calibration offset carries no pattern-quality information) | Sec. 4.3, Table 5 note |
| `explore_sfo_extrapolation/` | Scan the released SFO weights over z = 40--200 mm (5 mm steps + 1 mm dense across the window transitions); scripts, per-cell JSONs, and summary plot. The three per-color curves are panel (b) of Fig. 9 | Fig. 9(b), Sec. 4.6 |
| `scan_comb_m2.py` | Per-image comb scans used for the per-image statistics | Sec. 4.3 |
| `make_m2_figure.py` | Bar-chart figure of the per-cell SFO comparison | Fig. 6 |
| `make_rgb_composite.py` | Render the per-channel reconstructions and composite them into a full-color RGB image at off-peak distances (main model @ z = 198 mm; SFO-window model @ z = 113 mm) | Sec. 4.7, Fig. 12 |
| `make_n4_figure.py` | Three-panel RGB composite figure | Fig. 12 |
| `remap_2d.py` | Dense 2D ($z$, $\lambda$) remapping grid around the green design point (0.5 mm $\times$ 0.5 nm, 21$\times$21 points, 20 DIV2K images) behind the new Fig. 1(d); JSONs in `data_fig1_2d/` | Fig. 1(d), Sec. 1 |
| `make_fig1_v2.py` | Redraw Fig. 1 (b)(c)(d) with enlarged fonts and the 2D remapping heatmap for panel (d) | Fig. 1 |

Checkpoints produced by `run_m3_seeds.sh` are released under `weights/m3_seeds/`;
the final SFO-window model (step D) is `weights/sfo_window_spatial.pth`.
