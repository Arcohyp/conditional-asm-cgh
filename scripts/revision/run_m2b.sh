#!/usr/bin/env bash
# M2 resume: step A (base @ z=100mm) already done -> ./experiments/rev_m2_base_z100_p374/
# (checkpoint is train.py's conditional_ccnncgh_best.pth; z/wl ranges were 0 so it is
# the single-point warm-start base). This script runs step B + evals + N4.
# GPU passed as $1. NOTE: eval_sfo runs under CUDA_VISIBLE_DEVICES so that SFO's
# hardcoded cuda:0 tensors land on the right physical GPU.
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m2_sfo
mkdir -p "$REV/logs"
GPU=${1:?usage: run_m2b.sh <gpu-id>}
BASE_OUT=./experiments/rev_m2_base_z100_p374
OUT=./experiments/rev_m2_spatialcond_sfo_window

echo "[$(date)] M2 step B: spatial-conditional on SFO window, SFO-matched RGB content (GPU$GPU)" | tee -a $REV/logs/m2.log
$PY train_colorcond.py \
  --base-weight $BASE_OUT/conditional_ccnncgh_best.pth \
  --n 2160 --m 3840 --epochs 12 --num-train 600 --num-valid 15 \
  --pitch 0.00374 --design-z 100.0 --rgb-channels \
  --wavelengths 0.000450 0.000520 0.000638 \
  --lambda-continuous --lambda-lo 0.000440 --lambda-hi 0.000650 \
  --z-lo 85 --z-hi 115 --val-z 90 100 110 \
  --peak-prob 0.4 --peak-weight 4.0 --mse-weight 2.0 --npcc-weight 1.0 \
  --lambda-anchor-mode peak --lambda-peak-prob 0.4 --lambda-peak-weight 4.0 \
  --encoder cond --param-dim 16 --seed 42 --device-id $GPU \
  --output-dir $OUT > $REV/logs/train_m2_cond.log 2>&1
rc=$?
echo "[$(date)] step B rc=$rc" | tee -a $REV/logs/m2.log
[ $rc -ne 0 ] && exit $rc

echo "[$(date)] M2 eval: ours (both conventions) (GPU$GPU)" | tee -a $REV/logs/m2.log
$PY $REV/eval_ours_m2.py $GPU > $REV/logs/eval_ours_m2.log 2>&1
rc=$?
echo "[$(date)] eval ours rc=$rc" | tee -a $REV/logs/m2.log

echo "[$(date)] M2 eval: SFO PSNR+SSIM (GPU$GPU via CUDA_VISIBLE_DEVICES)" | tee -a $REV/logs/m2.log
CUDA_VISIBLE_DEVICES=$GPU $PY $REV/eval_sfo_psnr_ssim.py 0 > $REV/logs/eval_sfo.log 2>&1
rc=$?
echo "[$(date)] eval sfo rc=$rc" | tee -a $REV/logs/m2.log

echo "[$(date)] N4: composited RGB reconstruction (off-design point)" | tee -a $REV/logs/m2.log
$PY $REV/make_rgb_composite.py $GPU > $REV/logs/rgb_composite.log 2>&1
rc=$?
echo "[$(date)] rgb composite rc=$rc" | tee -a $REV/logs/m2.log
echo "[$(date)] M2 RESUME ALL DONE" | tee -a $REV/logs/m2.log
