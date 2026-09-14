#!/usr/bin/env bash
# M4c (2026-09-11): envelope-shift verification of the lambda*z coverage law.
# Scan the SFO-window spatial-conditional weight (rev_m2_spatialcond_sfo_window,
# trained z in [85,115]mm, continuous lambda 440-650nm, pitch 3.74um, seed 42)
# over z = 45-230mm @1mm at the three SFO native wavelengths, 100 DIV2K imgs.
# Predicted safe envelope from the main-weight law (0.87x band min, 1.27x band
# max of lambda*z in um*mm): near z ~ 72/62/51mm (b/g/r), far z ~ 211/183/149mm.
# One color per GPU: run_m4c_envelope_shift.sh <gpu-id> <color: blue|green|red>
set -u
cd /mnt/sda/chengxirun/hologram_task/expanded_work/conditional_asm_exp
PY=/home/chengxirun/.conda/envs/asm_hologram/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=ConditionalASMCGH_OE_Revision/experiments/m4_extrapolation
mkdir -p "$REV/logs"
GPU=${1:?usage: run_m4c_envelope_shift.sh <gpu-id> <blue|green|red>}
TAG=${2:?usage: run_m4c_envelope_shift.sh <gpu-id> <blue|green|red>}
W=./experiments/rev_m2_spatialcond_sfo_window/colorcond_best.pth

case $TAG in
  blue)   WL=0.000450 ;;
  green)  WL=0.000520 ;;
  red)    WL=0.000638 ;;
  *) echo "unknown color $TAG"; exit 2 ;;
esac
ZLIST=$(seq 45 1 230 | tr '\n' ' ')
echo "[$(date)] M4c envelope-shift z-scan $TAG start (GPU$GPU)" >> $REV/logs/m4c.log
$PY stage1_eval.py --weight $W --encoder cond --param-dim 16 \
  --output-dir ./experiments/rev_m4c_envelope_shift_$TAG \
  --n 2160 --m 3840 --pitch 0.00374 --wavelength $WL \
  --num-samples 100 --device-id $GPU --z-list $ZLIST \
  > $REV/logs/m4c_zscan_$TAG.log 2>&1
rc=$?
echo "[$(date)] M4c envelope-shift z-scan $TAG rc=$rc (GPU$GPU)" >> $REV/logs/m4c.log
