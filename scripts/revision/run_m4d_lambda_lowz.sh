#!/usr/bin/env bash
# M4d (2026-09-11): lambda-axis cross-check of the lambda*z collapse law.
#
# Why: the collapse boundary lambda*z ~= 0.88 x (band minimum) rests so far only on
# the z axis -- six near-side crossings from two independently trained weights
# (main weight, window 145-205mm, band min 63.8k -> crossings 56.25/56.39/56.78k;
# SFO-window weight, window 85-115mm, band min 37.4k -> crossings 32.85/33.80/32.54k;
# ratios 0.885 / 0.884). The existing full-band scan sits at z=150mm, where the
# predicted threshold (56.2k/150 = 375nm) lies BELOW the scanned range, so no
# wavelength-side collapse can appear there and the law is untested on the lambda axis.
#
# This scans lambda at two smaller z, putting the threshold inside both the scan
# range and the training band [440,650]nm, so wavelength extrapolation is not a
# confound:
#   z = 120mm -> predicted threshold 468nm
#   z = 110mm -> predicted threshold 511nm
# A single z gives one crossing; two z test that the crossing moves as 1/z, which
# is the actual content of the invariant.
#
# Same weight and pipeline as M4b (rev_m3_spatial_s42, pitch 3.6um, 100 DIV2K
# imgs/point), lambda at 2nm. Step choice: the z scan shows the transition completes
# within ~1mm ~= ~4nm in lambda at these z, so 2nm resolves it.
#
# Waits for the SFO far scan (scan_sfo_z_far.py) to release GPUs 0/2/3 first.
# Usage: run_m4d_lambda_lowz.sh   (no args; job queue is hard-coded below)
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m4_extrapolation
mkdir -p "$REV/logs"
W=./experiments/rev_m3_spatial_s42/colorcond_best.pth

# NOTE: bracket form so the pattern cannot match this script's own wait-loop
# command line (a plain "scan_sfo_z_far.py" pgrep also matches any monitor shell
# whose cmdline quotes that string, deadlocking the wait forever).
while pgrep -f 'scan_sfo_z_far[.]py' > /dev/null; do
  echo "[$(date)] waiting for SFO far scan to release the GPUs" >> $REV/logs/m4d.log
  sleep 60
done
sleep 30

# "<gpu> <z_mm> <wl_start_nm> <wl_end_nm>"  -- balanced ~47 points per GPU, 142 total
JOBS=(
  "0 120 430 522"
  "2 120 524 570"
  "2 110 460 504"
  "3 110 506 600"
)

run_chunk() {
  local gpu=$1 z=$2 w1=$3 w2=$4
  local out=./experiments/rev_m4_lambda_lowz/z$z/gpu$gpu
  local WL
  WL=$(seq "$w1" 2 "$w2" | awk '{printf "0.000%03d ", $1}')
  local n=$(( (w2 - w1) / 2 + 1 ))
  echo "[$(date)] z=${z}mm lambda ${w1}-${w2}nm (${n} pts) start (GPU$gpu)" >> $REV/logs/m4d.log
  mkdir -p "$(dirname "$out")"
  $PY eval.py --model-path $W --encoder cond --param-dim 16 \
    --output-dir "$out" \
    --n 2160 --m 3840 --pitch 0.0036 --z-list "$z" \
    --wl-list $WL --num-samples 100 --device-id "$gpu" \
    > "$REV/logs/m4d_z${z}_${w1}-${w2}_gpu${gpu}.log" 2>&1
  rc=$?
  echo "[$(date)] z=${z}mm lambda ${w1}-${w2}nm rc=$rc (GPU$gpu)" >> $REV/logs/m4d.log
}

# group the queue by GPU so each device runs its chunks sequentially
for g in 0 2 3; do
  (
    for job in "${JOBS[@]}"; do
      set -- $job
      [ "$1" = "$g" ] && run_chunk "$1" "$2" "$3" "$4"
    done
  ) &
done
wait
echo "[$(date)] M4d ALL DONE" >> $REV/logs/m4d.log
