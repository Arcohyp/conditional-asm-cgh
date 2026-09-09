#!/usr/bin/env bash
# M2 step C: continue training the spatial-conditional SFO-window model for
# 24 more epochs (warm-start from the step-B best) to lift the comb-valley
# cells; then re-run our dual-convention eval + N4 RGB composite.
# Rationale: pitch 3.74um @ z~100mm has ~3x denser self-imaging comb than the
# paper's 3.6um @ 150mm setting, so the standard 12-epoch schedule underfits
# the window edges (eval showed 14-17 dB edge cells vs 23-27 dB interior).
# GPU passed as $1.
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m2_sfo
mkdir -p "$REV/logs"
GPU=${1:?usage: run_m2c.sh <gpu-id>}
OUT=./experiments/rev_m2_spatialcond_sfo_window

echo "[$(date)] M2 step C: continue training 24 epochs from step-B best (GPU$GPU)" | tee -a $REV/logs/m2.log
$PY train_colorcond.py \
  --base-weight $OUT/colorcond_best.pth --full-warm-start \
  --n 2160 --m 3840 --epochs 24 --num-train 600 --num-valid 15 \
  --pitch 0.00374 --design-z 100.0 --rgb-channels \
  --wavelengths 0.000450 0.000520 0.000638 \
  --lambda-continuous --lambda-lo 0.000440 --lambda-hi 0.000650 \
  --z-lo 85 --z-hi 115 --val-z 85 100 115 \
  --peak-prob 0.4 --peak-weight 4.0 --mse-weight 2.0 --npcc-weight 1.0 \
  --lambda-anchor-mode peak --lambda-peak-prob 0.4 --lambda-peak-weight 4.0 \
  --encoder cond --param-dim 16 --seed 43 --device-id $GPU \
  --output-dir $OUT > $REV/logs/train_m2_cond_c.log 2>&1
rc=$?
echo "[$(date)] step C rc=$rc" | tee -a $REV/logs/m2.log
[ $rc -ne 0 ] && exit $rc

echo "[$(date)] M2 step C eval: ours (both conventions) (GPU$GPU)" | tee -a $REV/logs/m2.log
$PY $REV/eval_ours_m2.py $GPU > $REV/logs/eval_ours_m2_c.log 2>&1
rc=$?
echo "[$(date)] eval ours rc=$rc" | tee -a $REV/logs/m2.log

echo "[$(date)] N4 re-run: composited RGB reconstruction" | tee -a $REV/logs/m2.log
$PY $REV/make_rgb_composite.py $GPU > $REV/logs/rgb_composite_c.log 2>&1
rc=$?
echo "[$(date)] rgb composite rc=$rc" | tee -a $REV/logs/m2.log
echo "[$(date)] M2 STEP C ALL DONE" | tee -a $REV/logs/m2.log
