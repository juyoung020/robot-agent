#!/bin/bash
# 시뮬 테스트 공간(behavior-2026 서브모듈)의 scene_graph 코드를 이 저장소(실제 로봇 쪽) src/scene_graph/ 로 맞춘다.
# 원본은 behavior-2026 — 거기서 고치고 서브모듈을 갱신한 뒤 이 스크립트로 복사한다(반대로 고치면 어긋난다).
#   tools/sync_scene_graph.sh          # 복사
#   tools/sync_scene_graph.sh --check  # 다른 파일이 있으면 목록만 보이고 종료코드 1
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/src/behavior-2026/src/scene_graph"
DST="$ROOT/src/scene_graph"
[ -d "$SRC" ] || { echo "서브모듈이 없다: git submodule update --init src/behavior-2026"; exit 2; }
# sgview/assets/robot (URDF 모델 GLB·robot.json) 은 실제 로봇 쪽(robot-agent)에만 있다 — 동기화가 지우지 않게 제외. behavior-2026 에는 없고, 없으면 뷰어는 상자 모양으로 대체한다
OPTS=(-a --exclude target --exclude build --exclude __pycache__ --exclude '*.pyc' --exclude '/assets/robot')
rc=0
for d in scenemap da spark_dsg sgview runtime ovdet clip; do
  if [ "${1:-}" = "--check" ]; then
    out=$(rsync "${OPTS[@]}" -n -i --delete "$SRC/$d/" "$DST/$d/" | grep -v '^\.' || true)
    [ -z "$out" ] || { echo "== $d"; echo "$out"; rc=1; }
  else
    rsync "${OPTS[@]}" --delete "$SRC/$d/" "$DST/$d/"
  fi
done
exit $rc
