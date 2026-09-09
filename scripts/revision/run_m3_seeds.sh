#!/usr/bin/env bash
# M3 (Reviewer Major #3): seed-variance retraining.
# Unified "full recipe" for all three ablation models:
#   continuous lambda + lambda-peak anchor (mode=peak, p=0.4, w=4.0)
#   + z peak anchor (p=0.4, w=4.0) + 2*MSE+1*NPCC, z in [145,205] mm,
#   wavelengths {450,532,638} nm, 12 epochs, 600 train / 15 valid, 4K.
# Seeds 42/43/44 x {baseline(unconditional), cond(spatial), freq(frequency)}.
# This replaces paper Table 3/4 with a consistent recipe across models
# (the submitted Table 3 mixed A1/A3 z-scans with a discrete-lambda freq run).
#
# Usage: bash run_m3_seeds.sh        (launches 3 GPU queues in background)
# GPU0: cond s42->s43->s44 | GPU2: freq s42->s43->s44 | GPU3: uncond s42->s43->s44
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BW=/mnt/sda/chengxirun/hologram_task/hologram_task/experiments/ccnncgh_3840x2160_450nm_with_linear/ccnncgh/ccnncgh_best.pth
REV=ConditionalASMCGH_OE_Revision/experiments/m3_seeds
mkdir -p "$REV/logs"

COMMON="--base-weight $BW --n 2160 --m 3840 --epochs 12 --num-train 600 --num-valid 15 \
  --wavelengths 0.000450 0.000532 0.000638 \
  --lambda-continuous --lambda-lo 0.000440 --lambda-hi 0.000650 \
  --z-lo 145 --z-hi 205 --val-z 150 165 179 \
  --peak-prob 0.4 --peak-weight 4.0 --mse-weight 2.0 --npcc-weight 1.0 \
  --lambda-anchor-mode peak --lambda-peak-prob 0.4 --lambda-peak-weight 4.0"

queue() { # $1=gpu  $2=encoder-args  $3=name
  local gpu=$1; shift
  local enc=$1; shift
  local name=$1; shift
  for seed in 42 43 44; do
    local out=./experiments/rev_m3_${name}_s${seed}
    echo "[$(date)] $name seed=$seed GPU$gpu start -> $out" >> "$REV/logs/queue_${name}.log"
    $PY train_colorcond.py $COMMON $enc --seed $seed --device-id $gpu \
      --output-dir $out > "$REV/logs/train_${name}_s${seed}.log" 2>&1
    echo "[$(date)] $name seed=$seed rc=$?" >> "$REV/logs/queue_${name}.log"
  done
}

( queue 0 "--encoder cond --param-dim 16" spatial  ) &
( queue 2 "--encoder freq --n-freqs 16 --base-res 128" frequency ) &
( queue 3 "--encoder baseline" unconditional ) &
echo "[$(date)] 3 GPU queues launched (PIDs: $(jobs -p | tr '\n' ' '))"
