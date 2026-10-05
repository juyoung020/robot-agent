#!/bin/bash
# BEHAVIOR 장면을 Isaac Sim 창으로 열고 로봇만 우리 LIMO+OMX 로 바꿔 띄운다(정책은 0 행동 = 가만히).
#   view_scene_with_limo.sh [task=turning_on_radio] [instance_index=0]
set -u
TASK=${1:-turning_on_radio}; IDX=${2:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
OG=$(cd "$(dirname "$0")/../../behavior-2026/BEHAVIOR-1K/OmniGibson" && pwd)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_KEEP_VIEWER_CAMERA=1
cd /tmp && python "$HERE/eval_with_limo.py" --policy local --task-name "$TASK" --mode public_test --instance-indices "$IDX" --num-envs 1 \
  --max-steps 100000 --no-headless --robot-config "$HERE/limo_omx_eval.yaml" --output-dir /tmp/limo_view
