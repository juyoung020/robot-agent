#!/bin/bash
# OG 다시 돌리기 줄 일꾼: $RA_TRAINVIEW_WORK/og_queue/*.job(record_bc 가 체크포인트마다 학생·교사 앞 판 몇 개를 넣음)을 오래된 것부터 하나씩
# og_replay.sh 로 돌린다 → <판>_og.sg(재생 탭에 "REAL" 판). 일꾼은 하나만(flock -n), OG 는 og.lock 으로 한 번에 하나. 줄이 비면 끝.
# record_bc 가 줄에 넣은 뒤 `setsid -f` 로 띄운다(이미 돌면 바로 끝남). 손으로: og_queue.sh [줄 폴더]
#   job 에 "force": true 가 있으면 이미 만든 _og.sg 가 있어도 다시 돌린다(오래된 파이프라인·손목 카메라 없는 판 다시 만들기). 이름이 9999… 로 시작하면 맨 뒤(낮은 우선순위).
#   자원(램·디스크·GPU)이 모자라 og_replay.sh 가 75 로 끝나면 일을 줄에 되돌리고 2 분 쉼(학습을 막지 않음).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../../config/paths.env"
Q=${1:-$RA_TRAINVIEW_WORK/og_queue}
mkdir -p "$Q/running" "$Q/done" "$Q/failed"
exec 9>"$Q/.worker.lock"
flock -n 9 || exit 0
log() { echo "[og_queue $(date +%F' '%T)] $*"; }
while :; do
  J=$(ls -1 "$Q"/*.job 2>/dev/null | sort | head -1)
  [ -z "$J" ] && { log "queue empty"; exit 0; }
  B=$(basename "$J")
  mv "$J" "$Q/running/$B" || continue
  TRP=$(python3 -c "import json,sys;print(json.loads(open(sys.argv[1]).readline())['trp'])" "$Q/running/$B" 2>/dev/null)
  if [ -z "$TRP" ] || [ ! -f "$TRP" ]; then log "$B: no trp ($TRP)"; mv "$Q/running/$B" "$Q/failed/"; continue; fi
  FORCE=$(python3 -c "import json,sys;print(1 if json.loads(open(sys.argv[1]).readline()).get('force') else 0)" "$Q/running/$B" 2>/dev/null)
  if [ "${FORCE:-0}" != 1 ] && [ -f "${TRP%.trp}_og.sg/meta.json" ]; then log "$B: already done"; mv "$Q/running/$B" "$Q/done/"; continue; fi
  log "$B: $TRP"
  "$HERE/og_replay.sh" "$TRP" > "$Q/running/$B.log" 2>&1
  rc=$?
  [ $rc -ne 75 ] && n75=0
  if [ $rc -eq 0 ]; then mv "$Q/running/$B" "$Q/done/"; grep -a "^\[og_replay\]" "$Q/running/$B.log" > "$Q/done/$B.log"; rm -f "$Q/running/$B.log"; log "$B: done"
  elif [ $rc -eq 75 ]; then   # 자원 모자람(학습이 GPU 를 씀): 줄 맨 뒤로(새 시각 이름) — 작은 장면 일이 먼저 돌 수 있게. 다 75 면 쉬며 기다림
    N=$(date +%s)_${B#*_}; mv "$Q/running/$B" "$Q/$N"; rm -f "$Q/running/$B.log"; log "$B: resources busy — requeued as $N"
    n75=$(( ${n75:-0} + 1 )); [ $n75 -ge $(ls -1 "$Q"/*.job 2>/dev/null | wc -l) ] && { sleep 300; n75=0; }
  else mv "$Q/running/$B" "$Q/failed/"; tail -c 20000 "$Q/running/$B.log" > "$Q/failed/$B.log"; rm -f "$Q/running/$B.log"; log "$B: failed rc=$rc"; fi
done
