#!/bin/bash
# 과제 인스턴스 장면(작은 과제 물체가 놓인 *_template.json)으로 더 렌더 → $FASTSAM_DATA/sim/<scene>__<task>/
# 평가: house_double_floor_lower turning_on_radio(sim_detcmp 과 같은 판, 라디오) 등. 학습: 템플릿이 있는 학습 장면.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/env_local.sh" ] && source "$HERE/env_local.sh"
OG_LOCK=${OG_LOCK:-/tmp/og.lock}
DATA=${FASTSAM_DATA:-$RA_TRAIN_DATA/fastsam_obj}
T=$B1K_ROOT/datasets/2026-challenge-task-instances/scenes
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${OG_CONDA_ENV:-behavior}"
export OMNI_KIT_ACCEPT_EULA=YES
ulimit -c 0
run() {   # scene task n seed
  local f=$T/$1/json/$1_task_$2_0_0_template.json out=$DATA/sim/$1__$2
  [ -f "$f" ] || { echo "[tasks] no $f"; return; }
  if [ -f "$out/frames.jsonl" ] && [ "$(wc -l < "$out/frames.jsonl")" -ge "$3" ]; then echo "[tasks] $1 $2 have"; return; fi
  mkdir -p "$out"
  [ "$(df --output=avail -k "$HOME" | tail -1)" -lt $((15 * 1024 * 1024)) ] && { echo "[render] disk free < 15 GB — stop"; exit 3; }
  timeout 1800 flock "$OG_LOCK" python "$HERE/sim_render.py" --scene "$1" --scene-file "$f" --n "$3" --seed "$4" \
    --out "$out" > "$DATA/logs/render_$1__$2.log" 2>&1
  grep -E "^\[render\]" "$DATA/logs/render_$1__$2.log" | tail -1
  rm -f "$out/scene_file.json"
  "${FS_PY:-python}" "$HERE/label_sim.py" "$out" >> "$DATA/logs/label.log" 2>&1
}
pick() { ls "$T/$1/json/" | grep -E '_template\.json$' | sed -E "s/^$1_task_(.*)_0_0_template\.json$/\1/" | sort | awk -v k="$2" 'NR % k == 1'; }
# eval
run house_double_floor_lower turning_on_radio 200 7001
for t in $(pick house_double_floor_upper 5); do run house_double_floor_upper "$t" 60 7100; done
for t in $(pick office_cubicles_right 3); do run office_cubicles_right "$t" 60 7200; done
# train
s=7300
for sc in Rs_int hotel_suite_large house_single_floor restaurant_diner; do
  k=3; [ $sc = house_single_floor ] && k=6
  for t in $(pick $sc $k); do s=$((s + 1)); run $sc "$t" 80 $s; done
done
echo "[tasks] done"
