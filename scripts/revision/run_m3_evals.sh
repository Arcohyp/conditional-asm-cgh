#!/usr/bin/env bash
# M3 eval queue: for each of the 9 retrained models (3 encoders x 3 seeds),
# run the paper's evaluation grid:
#   - z-scan 145-205mm @1mm, 3 native colors, 100 DIV2K val imgs
#   - lambda fine-scan 440-460/522-542/628-648nm @1nm at z=150, 100 imgs
# Output: experiments/rev_m3_{spatial,frequency,unconditional}_s{42,43,44}_{eval_blue|eval_green|eval_red,_lambda_finescan_z150}/
# Waits for all checkpoints first (launched while training is still running).
# GPU0=spatial | GPU2=frequency | GPU3=unconditional (matches training queues)
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m3_seeds
ZG=$(seq 145 1 205 | tr '\n' ' ')
WL=$(python3 -c "print(' '.join(f'{x*1e-6:.7f}' for x in list(range(440,461))+list(range(522,543))+list(range(628,649))))")

# wait for all 9 training checkpoints
while true; do
  missing=0
  for name in spatial frequency unconditional; do
    for seed in 42 43 44; do
      [ -f ./experiments/rev_m3_${name}_s${seed}/colorcond_best.pth ] || missing=$((missing+1))
    done
  done
  [ "$missing" -eq 0 ] && break
  echo "[$(date)] waiting for $missing checkpoints..." >> $REV/logs/eval_queue.log
  sleep 300
done
echo "[$(date)] all 9 checkpoints ready, eval queues start" >> $REV/logs/eval_queue.log

eval_model() { # $1=name $2=enc-args $3=gpu
  local name=$1; shift
  local enc=$1; shift
  local gpu=$1; shift
  for seed in 42 43 44; do
    local out=./experiments/rev_m3_${name}_s${seed}
    for c in 0.000450:blue 0.000532:green 0.000638:red; do
      local wl=${c%%:*}; local tag=${c##*:}
      echo "[$(date)] $name s$seed zscan $tag (GPU$gpu)" >> $REV/logs/eval_queue.log
      $PY stage1_eval.py --weight $out/colorcond_best.pth \
        --output-dir ${out}_eval_$tag \
        --n 2160 --m 3840 --pitch 0.0036 --wavelength $wl \
        --num-samples 100 --device-id $gpu --z-list $ZG \
        $enc > $REV/logs/eval_${name}_s${seed}_${tag}.log 2>&1
    done
    echo "[$(date)] $name s$seed finescan (GPU$gpu)" >> $REV/logs/eval_queue.log
    $PY eval.py --model-path $out/colorcond_best.pth \
      --output-dir ${out}_lambda_finescan_z150 \
      --n 2160 --m 3840 --pitch 0.0036 --z-list 150.0 \
      --wl-list $WL --num-samples 100 --device-id $gpu \
      $enc > $REV/logs/eval_${name}_s${seed}_finescan.log 2>&1
    echo "[$(date)] $name s$seed ALL EVAL DONE" >> $REV/logs/eval_queue.log
  done
}

( eval_model spatial "--encoder cond --param-dim 16" 0 ) &
( eval_model frequency "--encoder freq --n-freqs 16 --base-res 128" 2 ) &
( eval_model unconditional "--encoder baseline" 3 ) &
wait
echo "[$(date)] M3 ALL EVAL DONE" >> $REV/logs/eval_queue.log
