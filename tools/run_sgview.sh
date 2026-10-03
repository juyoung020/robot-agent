#!/bin/bash
# 실시간 물체 기억 뷰어(sgview, Rust 서버 + three.js)를 켠다. 로봇 URDF 모델(LIMO + OMX-F)이 함께 나온다.
#   tools/run_sgview.sh <memory_dir>            녹화된 폴더 보기(view.json·map.pgm·objects/)
#   tools/run_sgview.sh <memory_dir> --live     시뮬/로봇이 소켓(127.0.0.1:9001)으로 보내는 실시간 스트림 받기
#                                               (시뮬 쪽: SGRT_STREAM=127.0.0.1:9001 [SGRT_MAP_EVERY=1])
#   옵션: --port 8080
# 처음에는 cargo 로 빌드한다(src/scene_graph/sgview). 브라우저에서 http://localhost:<port>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=${1:?usage: run_sgview.sh <memory_dir> [--live] [--port N]}; shift
PORT=8080; LIVE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --live) LIVE="--ingest 127.0.0.1:9001" ;;
    --port) PORT=$2; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
SV="$ROOT/src/scene_graph/sgview"
BIN="$SV/target/release/sgview"
[ -x "$BIN" ] || (cd "$SV" && cargo build --release)
echo "sgview: http://localhost:$PORT  ($DIR${LIVE:+, live stream on 127.0.0.1:9001})"
exec "$BIN" "$DIR" --port "$PORT" $LIVE
