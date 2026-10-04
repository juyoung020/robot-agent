#!/bin/bash
# BEHAVIOR 장면 전부 렌더(sim_render.py, 장면마다 og.lock 을 따로 잡음) → 광선 라벨(label_sim.py).
#   bash render_all.sh [train|eval|all]
# 출력: $FASTSAM_DATA/sim/<scene>/ (기본 ~/datasets/fastsam_obj)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/env_local.sh" ] && source "$HERE/env_local.sh"
OG_LOCK=${OG_LOCK:-/tmp/og.lock}
DATA=${FASTSAM_DATA:-$HOME/datasets/fastsam_obj}
WHICH=${1:-all}
source "$HERE/scenes.sh"
mkdir -p "$DATA/sim" "$DATA/logs"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${OG_CONDA_ENV:-behavior}"
export OMNI_KIT_ACCEPT_EULA=YES
ulimit -c 0
run() {   # scene n seed
  local s=$1 n=$2 seed=$3 out=$DATA/sim/$1
  if [ -f "$out/frames.jsonl" ] && [ "$(wc -l < "$out/frames.jsonl")" -ge "$n" ]; then echo "[all] $s have"; return; fi
  [ "$(df --output=avail -k "$HOME" | tail -1)" -lt $((15 * 1024 * 1024)) ] && { echo "[render] disk free < 15 GB — stop"; exit 3; }
  timeout 1800 flock "$OG_LOCK" python "$HERE/sim_render.py" --scene "$s" --n "$n" --seed "$seed" --out "$out" \
    > "$DATA/logs/render_$s.log" 2>&1
  grep -E "^\[render\]" "$DATA/logs/render_$s.log" | tail -1
  "${FS_PY:-python}" "$HERE/label_sim.py" "$out" >> "$DATA/logs/label.log" 2>&1 &
}
i=0
if [ "$WHICH" != eval ]; then
  for s in "${TRAIN_SCENES[@]}"; do
    i=$((i + 1)); n=200; case $s in house_*|*_int|gates_bedroom) n=400;; esac
    run "$s" $n $((1000 + i))
  done
fi
if [ "$WHICH" != train ]; then
  for s in "${EVAL_SCENES[@]}"; do
    i=$((i + 1)); n=150; [ "$s" = house_double_floor_lower ] && n=300
    run "$s" $n $((5000 + i))
  done
fi
wait
echo "[all] done"
