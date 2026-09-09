# Revision experiments (first revision, September 2026)

Scripts behind the new experiments in the revised manuscript (Manuscript ID 610780).
They are kept close to the form in which they were run on the authors' cluster;
absolute paths, GPU IDs, and the conda environment refer to that setup and need
to be adapted for other machines.

| Script | Content | Manuscript location |
|--------|---------|---------------------|
| `run_m3_seeds.sh` | Re-train all three ablation models (unconditional / spatial-conditional / frequency-conditional) from 3 seeds (42/43/44) under one unified recipe (continuous wavelength sampling + two-axis peak anchoring) | Tables 3, 4; Figs. 4, 5 |
| `run_m3_evals.sh` | Full evaluation suite of the M3 models: 61-point distance scan per color + 63-point wavelength fine scan, 100 DIV2K images per point | Tables 3, 4; Figs. 4, 5 |
| `compute_m3_stats.py` | Aggregate the M3 eval JSONs into mean $\pm$ std statistics | Tables 3, 4 |
| `run_m4.sh` | Extrapolation scans with a single trained weight: distance axis 85--265 mm, wavelength axis beyond the 440--650 nm training band | Sec. 4.5, Fig. 9 |
| `run_m2.sh`, `run_m2b.sh`, `run_m2c.sh`, `run_m2d.sh` | Progressive training of the spatial-conditional model on the SFO operating window (pitch 3.74 um, 450/520/638 nm, z in [85,115] mm) for the head-to-head comparison | Sec. 4.3, Table 5, Fig. 6 |
| `eval_ours_m2.py` | Evaluate our SFO-window model per cell in both evaluation conventions (SFO intensity-domain skimage and our amplitude-domain) | Table 5, Fig. 6 |
| `eval_sfo_psnr_ssim.py` | Re-evaluate the released SFO weights with SFO's own convention on the same 100 DIV2K images | Table 5, Fig. 6 |
| `scan_comb_m2.py` | Per-image comb scans used for the per-image statistics | Sec. 4.3 |
| `make_rgb_composite.py` | Render the per-channel reconstructions and composite them into a full-color RGB image at off-peak distances (main model @ z = 198 mm; SFO-window model @ z = 113 mm) | Sec. 4.6, Fig. 12 |
| `make_m2_figure.py` | Bar-chart figure of the per-cell SFO comparison | Fig. 6 |
| `make_n4_figure.py` | Three-panel RGB composite figure | Fig. 12 |

Checkpoints produced by `run_m3_seeds.sh` are released under `weights/m3_seeds/`;
the final SFO-window model (step D) is `weights/sfo_window_spatial.pth`.
