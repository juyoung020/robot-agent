#!/bin/bash
# 학습 뷰어·OG 다시 돌리기가 쓰는 장면 그래프 쪽 바이너리를 **robot-agent 의 소스에서** 만든다(복사본·옛 빌드 쓰지 않음).
#   training/viewer/build_deps.sh            # 바뀐 것만 다시 빌드(make/cargo 가 판단), -j4
#   TRAINVIEW_DEPS=<폴더>                      # 결과 링크 폴더(기본 $RA_TRAINVIEW_WORK/deps)
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
. "$REPO/config/paths.env"
DEPS=${TRAINVIEW_DEPS:-$RA_TRAINVIEW_WORK/deps}   # 소비자(og_replay·trainview)가 읽는 곳. 빌드 결과는 $RA_BUILD(tools/build_all.sh), 여기엔 링크만
"$REPO/tools/build_all.sh" sgrt sgview realbag og2sg trainview
if true; then
  mkdir -p "$DEPS"
  for f in libsgrt.so sgview sgs_play og2sg trainview; do ln -sfn "$RA_BUILD/bin/$f" "$DEPS/$f"; done
fi
echo "[deps] done -> $DEPS"; ls -l "$DEPS"
