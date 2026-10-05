#!/bin/bash
# 같은 스트림에 검출기 하나를 slam·gt 자세로 돌린다(검출은 slam 판에서 한 번, gt 판은 캐시 --load — 두 판의 검출이 바이트 같음).
#   detcmp_run.sh <stream dir> <out root> <name> <realbag_run 검출 옵션…>
#   예) detcmp_run.sh data/datasets/sim_detcmp/streams/radio_limo_r3 data/datasets/sim_detcmp A --det fastsam
# 결과: <root>/runs/<name>_{slam,gt}/(metrics.json·objects.csv·walls.csv·memory/), <root>/sg/<name>/ 재생 판(slam), <root>/dets/<name>.gz,
#       <root>/logs/<name>_*.log, <root>/runs/<name>_slam/gpu_mb.txt(검출 판 동안 이 프로세스 GPU 사용 최대, nvidia-smi 0.2 s)
set -euo pipefail
S=$1; R=$2; N=$3; shift 3
. "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/config/paths.env"
BIN=${REALBAG_BIN:-$RA_BUILD/bin}
COMMON=(--det-every "${DET_EVERY:-1}" --max-depth "${MAX_DEPTH:-4}" --conf "${CONF:-0.25}")
mkdir -p "$R/runs" "$R/dets" "$R/sg" "$R/logs"
rm -rf "$R/sg/$N" "$R/runs/${N}_slam" "$R/runs/${N}_gt"
"$BIN/realbag_run" "$S" "$R/runs/${N}_slam" --pose carto "${COMMON[@]}" "$@" --dump "$R/dets/$N.gz" --sg "$R/sg/$N" > "$R/logs/${N}_slam.log" 2>&1 &
P=$!
peak=0
while kill -0 $P 2>/dev/null; do
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p=$P '$1==p{print $2}')
  [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
  sleep 0.2
done
wait $P
echo "$peak" > "$R/runs/${N}_slam/gpu_mb.txt"
"$BIN/realbag_run" "$S" "$R/runs/${N}_gt" --pose gt "${COMMON[@]}" --load "$R/dets/$N.gz" > "$R/logs/${N}_gt.log" 2>&1
echo "[detcmp] $N done: slam $(tail -c 0 /dev/null)$R/runs/${N}_slam gt $R/runs/${N}_gt peak GPU ${peak} MiB"
