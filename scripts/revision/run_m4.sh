#!/usr/bin/env bash
# M4 (Reviewer Major #4): bound the coverage of the single global condition by
# extrapolating BEYOND the training window with the revised anchor model
# (rev_m3_spatial_s42, z in [145,205]mm, lambda band 440-650nm, pitch 3.6um):
#   - z-scan 85-144mm and 206-265mm @1mm, 3 native colors, 100 imgs
#   - lambda extrapolation {420,425,430,660,665,670}nm @ z=150, 100 imgs
# Waits for the M3 spatial s42 checkpoint. GPU passed as $1.
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m4_extrapolation
mkdir -p "$REV/logs"
GPU=${1:?usage: run_m4.sh <free-gpu-id>}
W=./experiments/rev_m3_spatial_s42/colorcond_best.pth

while ! grep -q "spatial seed=42 rc=0" $REV/../m3_seeds/logs/queue_spatial.log 2>/dev/null; do
  echo "[$(date)] waiting for M3 spatial s42 training to finish" >> $REV/logs/m4.log
  sleep 300
done
# make sure the final best checkpoint is flushed
sleep 30
echo "[$(date)] anchor checkpoint ready, M4 scan start (GPU$GPU)" >> $REV/logs/m4.log

ZB=$(seq 85 1 144 | tr '\n' ' ')
ZA=$(seq 206 1 265 | tr '\n' ' ')
WLE="0.000420 0.000425 0.000430 0.000660 0.000665 0.000670"
for c in 0.000450:blue 0.000532:green 0.000638:red; do
  wl=${c%%:*}; tag=${c##*:}
  echo "[$(date)] z below-window $tag" >> $REV/logs/m4.log
  $PY stage1_eval.py --weight $W --encoder cond --param-dim 16 \
    --output-dir ./experiments/rev_m4_zscan_below_$tag \
    --n 2160 --m 3840 --pitch 0.0036 --wavelength $wl \
    --num-samples 100 --device-id $GPU --z-list $ZB \
    > $REV/logs/zscan_below_$tag.log 2>&1
  echo "[$(date)] z above-window $tag" >> $REV/logs/m4.log
  $PY stage1_eval.py --weight $W --encoder cond --param-dim 16 \
    --output-dir ./experiments/rev_m4_zscan_above_$tag \
    --n 2160 --m 3840 --pitch 0.0036 --wavelength $wl \
    --num-samples 100 --device-id $GPU --z-list $ZA \
    > $REV/logs/zscan_above_$tag.log 2>&1
done
echo "[$(date)] lambda extrapolation" >> $REV/logs/m4.log
$PY eval.py --model-path $W --encoder cond --param-dim 16 \
  --output-dir ./experiments/rev_m4_lambda_extrap_z150 \
  --n 2160 --m 3840 --pitch 0.0036 --z-list 150.0 \
  --wl-list $WLE --num-samples 100 --device-id $GPU \
  > $REV/logs/lambda_extrap.log 2>&1
echo "[$(date)] M4 ALL DONE" >> $REV/logs/m4.log
