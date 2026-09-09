#!/usr/bin/env bash
# M2 (Reviewer Major #2): fair reconstruction-quality comparison vs SFO.
# SFO native settings (from CGH-SFO-solver code): res 2160x3840, pitch 3.74um,
# wavelengths {450,520,638}nm (green is 520, not 532), distance 85-115mm,
# one 109M weight per color, intensity-domain skimage PSNR/SSIM with per-image
# sum-scale. SFO's honest 100-img DIV2K numbers already exist
# (eghp_experiment/CGH-SFO-solver/validate_100imgs.log, grand mean 33.12 dB).
#
# Our side: retrain the spatial-conditional model on SFO's window/settings:
#   step A: fresh single-point base at (z=100mm, 450nm, pitch 3.74um)  [train.py]
#   step B: spatial-conditional, z in [85,115], continuous lambda, full recipe
# Requires a FREE GPU (run after M3 trainings finish). GPU passed as $1.
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m2_sfo
mkdir -p "$REV/logs"
GPU=${1:?usage: run_m2.sh <free-gpu-id>}
BASE_OUT=./experiments/rev_m2_base_z100_p374
OUT=./experiments/rev_m2_spatialcond_sfo_window

echo "[$(date)] M2 step A: base @ z=100mm, 450nm, pitch 3.74um, SFO-matched gamma blue (GPU$GPU)" | tee -a $REV/logs/m2.log
$PY $REV/m2_base_driver.py $GPU > $REV/logs/train_base_z100.log 2>&1
echo "[$(date)] step A rc=$?" | tee -a $REV/logs/m2.log

echo "[$(date)] M2 step B: spatial-conditional on SFO window, SFO-matched RGB content (GPU$GPU)" | tee -a $REV/logs/m2.log
$PY train_colorcond.py \
  --base-weight $BASE_OUT/ccnncgh_best.pth \
  --n 2160 --m 3840 --epochs 12 --num-train 600 --num-valid 15 \
  --pitch 0.00374 --design-z 100.0 --rgb-channels \
  --wavelengths 0.000450 0.000520 0.000638 \
  --lambda-continuous --lambda-lo 0.000440 --lambda-hi 0.000650 \
  --z-lo 85 --z-hi 115 --val-z 90 100 110 \
  --peak-prob 0.4 --peak-weight 4.0 --mse-weight 2.0 --npcc-weight 1.0 \
  --lambda-anchor-mode peak --lambda-peak-prob 0.4 --lambda-peak-weight 4.0 \
  --encoder cond --param-dim 16 --seed 42 --device-id $GPU \
  --output-dir $OUT > $REV/logs/train_m2_cond.log 2>&1
echo "[$(date)] step B rc=$?" | tee -a $REV/logs/m2.log
echo "[$(date)] M2 training DONE -> $OUT" | tee -a $REV/logs/m2.log

echo "[$(date)] M2 eval: ours (both conventions) + SFO (PSNR+SSIM) (GPU$GPU)" | tee -a $REV/logs/m2.log
$PY $REV/eval_ours_m2.py $GPU > $REV/logs/eval_ours_m2.log 2>&1
echo "[$(date)] eval ours rc=$?" | tee -a $REV/logs/m2.log
$PY $REV/eval_sfo_psnr_ssim.py $GPU > $REV/logs/eval_sfo.log 2>&1
echo "[$(date)] eval sfo rc=$?" | tee -a $REV/logs/m2.log
echo "[$(date)] N4: composited RGB reconstruction (off-design point)" | tee -a $REV/logs/m2.log
$PY $REV/make_rgb_composite.py $GPU > $REV/logs/rgb_composite.log 2>&1
echo "[$(date)] rgb composite rc=$?" | tee -a $REV/logs/m2.log
echo "[$(date)] M2 ALL DONE" | tee -a $REV/logs/m2.log
