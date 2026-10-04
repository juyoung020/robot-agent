#!/bin/bash
# 끝에서 끝: sim_detcmp radio r3 스트림(LIMO, house_double_floor_lower)에 엔진 하나로 realbag_run(slam·gt) → 확률 모드 채점기
#   (detcmp_eval 표 + 잘못 합침·문창계단).
#   bash e2e.sh <이름> <엔진 .plan> <plain|objprob>   plain = 옛 규칙(sim_detcmp FastSAM-s-416 판과 같은 설정),
#                                                   objprob = scenemap 확률 모드(+ --label-prior)
# 이름 바꾸기(옛 이름 → objprob) 전후 둘 다 돈다: 플래그·채점기·사전 경로를 있는 쪽으로 고른다.
# 빌드: behavior-2026 main(ca85b1a 이상) realbag → ~/realbag_build_fsobj. 결과: $FASTSAM_DATA/e2e/<이름>_<plain|objprob>/
set -euo pipefail
N=$1; ENG=$2; MODE=$3
DATA=${FASTSAM_DATA:-$HOME/datasets/fastsam_obj}
BIN=${REALBAG_BIN:-$HOME/realbag_build_fsobj}
B26=${B26:-$HOME/b26-wt-fsobj}
S=$HOME/datasets/sim_detcmp/streams/radio_limo_r3
GTPGM=$HOME/behavior-2026/src/sim/explore/gt/house_double_floor_lower__turning_on_radio.pgm
OLD=ap''rime   # 옛 이름(바뀌기 전 빌드·폴더용)
PD=$HOME/datasets/objprob; [ -d "$PD" ] || PD=$HOME/datasets/$OLD
PRIOR=${LABEL_PRIOR:-$PD/fit1/label_prior.json}
FLAG=--objprob; grep -aq -- "--objprob" "$BIN/realbag_run" || FLAG=--$OLD
EV=$B26/src/scene_graph/tools/realbag/objprob_eval.py; [ -f "$EV" ] || EV=$B26/src/scene_graph/tools/realbag/${OLD}_eval.py
R=$DATA/e2e/${N}_$MODE
EXTRA=(--det fastsam --engine "$ENG")
[ "$MODE" = objprob ] && EXTRA+=($FLAG --label-prior "$PRIOR")
rm -rf "$R"; mkdir -p "$R"
COMMON=(--det-every 1 --max-depth 4 --conf 0.25)
"$BIN/realbag_run" "$S" "$R/slam" --pose slam "${COMMON[@]}" "${EXTRA[@]}" --dump "$R/dets.gz" --sg "$R/sg" > "$R/slam.log" 2>&1 &
P=$!; peak=0
while kill -0 $P 2>/dev/null; do
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p=$P '$1==p{print $2}')
  [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
  sleep 0.2
done
wait $P
echo "$peak" > "$R/slam/gpu_mb.txt"
LOADX=(); [ "$MODE" = objprob ] && LOADX=($FLAG --label-prior "$PRIOR")
"$BIN/realbag_run" "$S" "$R/gt" --pose gt "${COMMON[@]}" --load "$R/dets.gz" "${LOADX[@]}" > "$R/gt.log" 2>&1
~/realbag_venv/bin/python "$EV" "$S" "$GTPGM" "$R/slam,$R/gt" --json "$R/eval.json" > "$R/eval.txt" 2>&1
echo "[e2e] $N $MODE peak GPU $peak MiB"; head -40 "$R/eval.txt"
