#!/bin/bash
# Detector-level comparison under the GPU lock: builds -> ovdet_eval.py on ep0 40 s + ep200.
#   bash eval_linux.sh [smoke]
set -e
T=$(cd "$(dirname "$0")/.." && pwd)
R=$(cd "$T/../../.." && pwd)
M=~/ovdet_models/x86_sm120
PY=~/ovdet_export_venv/bin/python
OUT=~/ovdet_eval; mkdir -p $OUT
bash $T/scripts/build_linux.sh > /dev/null
source $R/src/sim/engine/scripts/gpu_lock.sh
gpu_lock_acquire ovdet "ovdet: detector comparison (6 heads, ep0 40 s + ep200)" 30 10
trap 'gpu_lock_release ovdet' EXIT
if [ "$1" = smoke ]; then
  $PY $T/tools/ovdet_eval.py --episodes 0:10:2:10 --det y11s=$M/yoloe-11s-task.plan \
    --out $OUT/smoke.json 2>&1 | grep -v Warning
  exit
fi
$PY $T/tools/ovdet_eval.py --episodes 0:0:40:5 200:0:0:15 \
  --det y11s=$M/yoloe-11s-task.plan --det y11m=$M/yoloe-11m-task.plan --det y11l=$M/yoloe-11l-task.plan \
  --det y11s_all=$M/yoloe-11s-all.plan --det y11l_all=$M/yoloe-11l-all.plan \
  --det "y11l_allp=$M/yoloe-11l-all.plan,prompt=all" \
  --out $OUT/compare.json 2>&1 | grep -v Warning
