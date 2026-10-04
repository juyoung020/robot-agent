#!/bin/bash
# 후보 .pt 하나 → 임시 엔진($FASTSAM_DATA/cand/) → 검출 단계 평가(옛 엔진과 짝 비교) sim_eval·coco_val·ade_val.
#   bash cand_eval.sh <run 이름> [best|last] [sets...]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/env_local.sh" ] && source "$HERE/env_local.sh"
DATA=${FASTSAM_DATA:-$HOME/datasets/fastsam_obj}
R=$1; W=${2:-best}; shift 2 || true
SETS=${*:-sim_eval coco_val ade_val}
N=cand_${R}_$W
OUT_PT=$DATA/cand OUT_PLAN=$DATA/cand FORCE=1 bash "$HERE/export.sh" "$DATA/runs/$R/weights/$W.pt" "$N" > "$DATA/logs/export_$N.log" 2>&1
for S in $SETS; do
  "${FS_PY:-python}" "$HERE/eval_det.py" --set "$S" --plans base=$BASE_PLAN \
    new=$DATA/cand/$N.plan --out "$DATA/eval/${N}_$S.json" > "$DATA/logs/eval_${N}_$S.log" 2>&1
  echo "=== $N $S"; grep -v "^\[eval\]" "$DATA/logs/eval_${N}_$S.log"
done
