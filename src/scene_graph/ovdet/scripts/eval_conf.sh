#!/bin/bash
# Confidence sweep of the YOLOE heads (GPU lock), ep0 40 s + ep200: bash eval_conf.sh
set -e
T=$(cd "$(dirname "$0")/.." && pwd)
M=~/ovdet_models/x86_sm120
PY=~/ovdet_export_venv/bin/python
source $T/../../sim/engine/scripts/gpu_lock.sh
gpu_lock_acquire ovdet "ovdet: YOLOE confidence sweep (ep0 40 s + ep200)" 25 10
trap 'gpu_lock_release ovdet' EXIT
$PY $T/tools/ovdet_eval.py --episodes 0:0:40:5 200:0:0:15 \
  --det "y11m_c10=$M/yoloe-11m-task.plan,conf_th=0.1" --det "y11l_c10=$M/yoloe-11l-task.plan,conf_th=0.1" \
  --det "y11l_c05=$M/yoloe-11l-task.plan,conf_th=0.05" --det "y11s_c10=$M/yoloe-11s-task.plan,conf_th=0.1" \
  --out ~/ovdet_eval/conf.json 2>&1 | grep -vE "Warning|Broken pipe|trailer"
