#!/bin/bash
# robot-agent 전체 빌드 진입점 — 모든 결과가 $RA_BUILD(기본 <저장소>/build) 한 곳에 모인다. 증분(바뀐 것만), -j4.
#   tools/build_all.sh                 # 전부
#   tools/build_all.sh sgrt sgview     # 골라서
#   tools/build_all.sh --list          # 대상 목록
#   JOBS=2 tools/build_all.sh map      # 병렬 수 바꿈(기본 4)
# 결과: cmake 폴더 $RA_BUILD/<이름>, cargo 대상 $RA_BUILD/cargo/<이름>, 실행 파일·공유 라이브러리 링크 $RA_BUILD/bin/
# 외부 경로는 config/paths.env 에서만 정한다(내 PC 값은 config/paths.local.env).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/config/paths.env"
J=${JOBS:-4}
B=$RA_BUILD
SG=$ROOT/src/scene_graph
TR=$ROOT/training
mkdir -p "$B/bin" "$B/cargo"

ALL=(scenemap sgrt sgclip realbag sgview og2sg env map map_cmp ppo bc vla record_replay trainview agent ppo_driver bc_driver)
if [ "${1:-}" = "--list" ]; then printf '%s\n' "${ALL[@]}"; exit 0; fi
TARGETS=("$@"); [ ${#TARGETS[@]} -gt 0 ] || TARGETS=("${ALL[@]}")

link() { # link <실제 파일> [이름]
  [ -e "$1" ] || { echo "[build] 결과가 없다: $1" >&2; return 1; }
  ln -sfn "$1" "$B/bin/${2:-$(basename "$1")}"
}
cm() { # cm <이름> <소스 폴더> [cmake 인자...] — 구성(처음 한 번) + 빌드(-j)
  local name=$1 src=$2; shift 2
  [ -f "$B/$name/CMakeCache.txt" ] || cmake -S "$src" -B "$B/$name" -DCMAKE_BUILD_TYPE=Release "$@"
  cmake --build "$B/$name" -j"$J" ${CM_TARGETS:+--target $CM_TARGETS}
}
cg() { # cg <이름> <crate 폴더> <실행 파일...> — cargo release
  local name=$1 dir=$2; shift 2
  (cd "$dir" && CARGO_TARGET_DIR="$B/cargo/$name" cargo build --release -j"$J")
  for bin in "$@"; do link "$B/cargo/$name/release/$bin"; done
}

for t in "${TARGETS[@]}"; do
  echo "== [build] $t"
  case $t in
    scenemap)   cm scenemap "$SG/scenemap" ;;                                  # 시험은 ctest --test-dir $RA_BUILD/scenemap
    sgrt)       CM_TARGETS=sgrt cm sgrt "$SG/runtime" -DCMAKE_CUDA_ARCHITECTURES="$RA_CUDA_ARCH"; link "$B/sgrt/libsgrt.so" ;;
    sgclip)     cm sgclip "$SG/clip"; link "$B/sgclip/libsgclip_c.so" ;;       # 에이전트 search_objects 가 링크(SGCLIP_LIB_DIR=$RA_BUILD/sgclip)
    realbag)    CM_TARGETS="realbag_run sgs_play" cm realbag "$SG/tools/realbag"; link "$B/realbag/realbag_run"; link "$B/realbag/sgs_play" ;;
    sgview)     cg sgview "$SG/sgview" sgview ;;
    og2sg)      cm og2sg "$TR/viewer/tools/og2sg"; link "$B/og2sg/og2sg" ;;
    env)        cm env "$TR/RL/env" ;;
    map)        cm map "$TR/RL/map" ;;
    map_cmp)    cm map_cmp "$TR/RL/map_cmp" ;;
    ppo)        cm ppo "$TR/RL/ppo" ;;
    bc)         cm bc "$TR/BC" ;;
    vla)        cm vla "$TR/vla" ;;
    record_replay) cm record_replay "$TR/viewer/tools/record_replay" ;;
    trainview)  cg trainview "$TR/viewer" trainview ;;
    ppo_driver) cg ppo_driver "$TR/RL/ppo/driver" ;;
    bc_driver)  cg bc_driver "$TR/BC/driver" ;;
    agent)      export SGCLIP_LIB_DIR=$B/sgclip
                [ -f "$B/sgclip/libsgclip_c.so" ] || "$0" sgclip
                cg agent "$ROOT/src/agent/runtime" run-skill ;;
    *) echo "모르는 대상: $t  ($0 --list)" >&2; exit 2 ;;
  esac
done
echo "== [build] 끝 -> $B/bin"
