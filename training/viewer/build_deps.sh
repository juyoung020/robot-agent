#!/bin/bash
# 학습 뷰어·OG 다시 돌리기가 쓰는 장면 그래프 쪽 바이너리를 **robot-agent 의 소스에서** 만든다(복사본·옛 빌드 쓰지 않음).
#   training/viewer/build_deps.sh            # 바뀐 것만 다시 빌드(make/cargo 가 판단), -j4
#   TRAINVIEW_DEPS=<폴더>                      # 결과 링크 폴더(기본 ~/trainview_work/deps)
# 만드는 것(결과는 $DEPS 에 링크 — og_replay.py·og_replay.sh·trainview 가 거기서만 찾는다):
#   libsgrt.so  = src/scene_graph/runtime (ObjectSAM + SigLIP 2 + scenemap objprob. 엔진·objprob 기본값은 그 소스의 것 — 여기서 따로 정하지 않음)
#   sgview      = src/scene_graph/sgview (진짜 뷰어 — 실시간에 쓰는 같은 바이너리)
#   sgs_play    = src/scene_graph/tools/realbag/sgs_play.cpp (재생 보내는 쪽, --ctl)
#   og2sg       = training/viewer/tools/og2sg (underlay 만드는 데만)
# 장면 그래프 소스(src/scene_graph, 원본 하나)를 고친 뒤: 이 스크립트 → 끝.
# 판 폴더(.sg)의 meta.json pipeline 에 만든 소스의 해시가 있어, 지금 소스와 다르면 재생 화면이 "stale pipeline" 으로 표시한다.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
SG26=$REPO/src/scene_graph
DEPS=${TRAINVIEW_DEPS:-$HOME/trainview_work/deps}
BLD=${TRAINVIEW_BUILD:-$HOME/trainview_work/build}
mkdir -p "$DEPS" "$BLD"
J=${JOBS:-4}
echo "[deps] libsgrt"
if [ ! -f "$BLD/sgrt/CMakeCache.txt" ]; then
  # 이미 있는 같은 소스의 빌드 폴더가 있으면 그걸 씀(처음 한 번 CUDA 빌드가 길다)
  if [ -f "$HOME/sgrt_build_explore/CMakeCache.txt" ] && grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$SG26/runtime$" "$HOME/sgrt_build_explore/CMakeCache.txt"; then ln -sfn "$HOME/sgrt_build_explore" "$BLD/sgrt"
  else cmake -S "$SG26/runtime" -B "$BLD/sgrt" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-120}"; fi
fi
cmake --build "$BLD/sgrt" -j"$J" --target sgrt
ln -sfn "$BLD/sgrt/libsgrt.so" "$DEPS/libsgrt.so"
echo "[deps] sgview"
(cd "$REPO/src/scene_graph/sgview" && cargo build --release -j"$J")
ln -sfn "$REPO/src/scene_graph/sgview/target/release/sgview" "$DEPS/sgview"
echo "[deps] sgs_play"
g++ -O2 -std=c++17 -pthread "$SG26/tools/realbag/sgs_play.cpp" -o "$BLD/sgs_play.new" && mv -f "$BLD/sgs_play.new" "$BLD/sgs_play"
ln -sfn "$BLD/sgs_play" "$DEPS/sgs_play"
echo "[deps] og2sg"
if [ ! -f "$BLD/og2sg/CMakeCache.txt" ]; then
  if [ -f "$HOME/ra_og2sg/CMakeCache.txt" ] && grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$HERE/tools/og2sg$" "$HOME/ra_og2sg/CMakeCache.txt"; then ln -sfn "$HOME/ra_og2sg" "$BLD/og2sg"
  else cmake -S "$HERE/tools/og2sg" -B "$BLD/og2sg" -DCMAKE_BUILD_TYPE=Release; fi
fi
cmake --build "$BLD/og2sg" -j"$J"
ln -sfn "$BLD/og2sg/og2sg" "$DEPS/og2sg"
echo "[deps] trainview"
(cd "$HERE" && cargo build --release -j"$J")
echo "[deps] done -> $DEPS"; ls -l "$DEPS"
