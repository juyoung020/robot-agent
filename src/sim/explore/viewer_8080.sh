#!/bin/bash
# 8080 뷰어(sgview, Rust + three.js)를 이 탐사 판의 memory 로 바꾼다(뷰어는 하나만). 다른 뷰어는 PID 로만 끈다(pkill 안 씀).
#   viewer_8080.sh <run dir>    — memory/view.json 이 생길 때까지 기다린 뒤 파일 모드로 켠다
# 실시간(소켓 → SSE, 60 Hz 이상)은 robot-agent 의 tools/run_explore_live.sh 를 쓴다(판 + 뷰어 한 번에).
# 옛 파이썬 뷰어 sgviz(src/scene_graph/viewer, 실시간 아님)는 쓰지 않는다 — 켜져 있으면 끄기만 한다.
set -u
RUN=$(cd "$1" && pwd); MEM=$RUN/memory
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SV=$REPO/src/scene_graph/sgview
BIN=$SV/target/release/sgview
[ -x "$BIN" ] || (cd "$SV" && cargo build --release) || { echo "[viewer] sgview build failed"; exit 1; }
for i in $(seq 1 240); do [ -f "$MEM/view.json" ] && break; sleep 5; done
[ -f "$MEM/view.json" ] || { echo "[viewer] no view.json in $MEM"; exit 1; }
for pid in $(ps -eo pid,args | awk '(/sgviz\.py/ || /release\/sgview/) && /--port 8080/ && !/awk/ {print $1}'); do kill "$pid"; done
sleep 2
setsid nohup "$BIN" "$MEM" --port 8080 > "$RUN/viewer.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  c=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/ || true)
  [ "$c" = 200 ] && { echo "[viewer] sgview http://localhost:8080 -> $MEM"; exit 0; }
  sleep 1
done
echo "[viewer] not answering"; exit 1
