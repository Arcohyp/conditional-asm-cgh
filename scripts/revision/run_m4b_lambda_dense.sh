#!/usr/bin/env bash
# M4b (2026-09-10): dense full-band wavelength scan to fill the unmeasured
# gaps of Fig. 9(b). Single spatial-conditional weight (rev_m3_spatial_s42,
# z in [145,205]mm, lambda band 440-650nm, pitch 3.6um) at z=150mm,
# lambda = 410-680nm @2nm, 100 DIV2K imgs per point.
# Split across 3 GPUs by wavelength range: GPU passed as $1, range as $2:$3 (nm).
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m4_extrapolation
mkdir -p "$REV/logs"
GPU=${1:?usage: run_m4b_lambda_dense.sh <gpu-id> <wl-start-nm> <wl-end-nm>}
W1=${2:?wl start (nm)}
W2=${3:?wl end (nm)}
W=./experiments/rev_m3_spatial_s42/colorcond_best.pth

WL=$(seq "$W1" 2 "$W2" | awk '{printf "0.000%03d ", $1}')
echo "[$(date)] lambda dense $W1-$W2 nm start (GPU$GPU)" >> $REV/logs/m4b.log
$PY eval.py --model-path $W --encoder cond --param-dim 16 \
  --output-dir ./experiments/rev_m4_lambda_dense_z150/gpu$GPU \
  --n 2160 --m 3840 --pitch 0.0036 --z-list 150.0 \
  --wl-list $WL --num-samples 100 --device-id $GPU \
  > $REV/logs/lambda_dense_gpu$GPU.log 2>&1
rc=$?
echo "[$(date)] lambda dense $W1-$W2 nm rc=$rc (GPU$GPU)" >> $REV/logs/m4b.log
