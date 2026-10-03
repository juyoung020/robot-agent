#!/bin/bash
# 탐사 한 판 + 실시간 뷰어를 한 번에 — 경로·라이브러리·뷰어 폴더를 스크립트가 맞춘다(사람이 틀릴 자리를 없앰).
#   tools/run_explore_live.sh [policy=frontier] [task=turning_on_radio] [tag=live] [--port 8080] [--pose slam|odom|gt]
# 하는 일: ① libsgrt 증분 빌드 + SGRT_STREAM 지원 확인(없으면 중단) ② explore 에이전트 바이너리 확인
#          ③ 시뮬 판을 robot-agent/src/behavior-2026 에서 실행(자세 기본 slam, 지도는 매 갱신 전송)
#          ④ 판의 memory/ 가 생기면 뷰어를 그 폴더로 켠다(예전 뷰어는 PID 로만 끈다)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BH=$ROOT/src/behavior-2026
POL=frontier; TASK=turning_on_radio; TAG=live; VPORT=8080; POSE=slam; pos=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port) VPORT=$2; shift ;;
    --pose) POSE=$2; shift ;;
    *) pos+=("$1") ;;
  esac; shift
done
[ ${#pos[@]} -ge 1 ] && POL=${pos[0]}; [ ${#pos[@]} -ge 2 ] && TASK=${pos[1]}; [ ${#pos[@]} -ge 3 ] && TAG=${pos[2]}

BUILD=${SGRT_BUILD:-$HOME/sgrt_build_explore}
[ -d "$BUILD" ] || { echo "[live] 빌드 폴더 없음: $BUILD (cmake -S $BH/src/scene_graph/runtime -B $BUILD)"; exit 1; }
cmake --build "$BUILD" -j"$(nproc)" --target sgrt >/dev/null
grep -aqF SGRT_STREAM "$BUILD/libsgrt.so" ||  # strings|grep -q 는 pipefail 에서 SIGPIPE(141)로 항상 실패
  { echo "[live] libsgrt 에 SGRT_STREAM 이 없다(옛 빌드): $BUILD"; exit 1; }
AG=$ROOT/src/agent/skills/explore/target/release/explore
[ -x "$AG" ] || { echo "[live] explore 에이전트 없음: (cd $ROOT/src/agent/skills/explore && cargo build --release)"; exit 1; }
SV=$ROOT/src/scene_graph/sgview/target/release/sgview
[ -x "$SV" ] || (cd "$ROOT/src/scene_graph/sgview" && cargo build --release)

export SGRT_LIB=$BUILD/libsgrt.so SGRT_POSE=$POSE SGRT_STREAM=127.0.0.1:9001 SGRT_MAP_EVERY=${SGRT_MAP_EVERY:-1}
export ROBOT_AGENT=$ROOT
cd "$BH"
before=$(ls -d outputs/explore_*_"$TASK"_"$POL"_"$TAG" 2>/dev/null | sort | tail -1 || true)   # 같은 태그의 예전 판을 집지 않게
setsid nohup src/sim/explore/run_explore.sh "$POL" "$TASK" "$TAG" > "/tmp/run_explore_$TAG.log" 2>&1 &
echo "[live] 판 시작(자세 $POSE) — 로그 /tmp/run_explore_$TAG.log"
RUN=""
for i in $(seq 1 120); do
  RUN=$(ls -d outputs/explore_*_"$TASK"_"$POL"_"$TAG" 2>/dev/null | sort | tail -1 || true)
  [ -n "$RUN" ] && [ "$RUN" != "$before" ] && [ -d "$BH/$RUN" ] && break; RUN=""; sleep 1
done
[ -n "$RUN" ] || { echo "[live] 출력 폴더가 안 생김"; tail -5 "/tmp/run_explore_$TAG.log"; exit 1; }
MEM=$BH/$RUN/memory; mkdir -p "$MEM"
for pid in $(ps -eo pid,args | awk -v p="--port $VPORT" '/release\/sgview/ && index($0,p) && !/awk/ {print $1}'); do kill "$pid"; done
sleep 1
setsid nohup "$ROOT/tools/run_sgview.sh" "$MEM" --live --port "$VPORT" > "/tmp/run_sgview_$VPORT.log" 2>&1 &
echo "[live] 뷰어 http://localhost:$VPORT  (메모리 $MEM)"
