#!/bin/bash
# 전체 점검 — 빌드·시험·실제 실행이 다 되는지 한 번에. 실패한 것만 끝에 모아 알려 준다.
#   tools/check_all.sh            # 빌드(증분) + 시험 + 실제 bag 재생 + 기록 재생(og2sg) + 뷰어 둘 + 잔재·경로 검사
#   tools/check_all.sh --sim      # 위 + OmniGibson 시뮬 탐사 한 판(짧게, og.lock) — 오래 걸림
#   tools/check_all.sh --no-build # 빌드 건너뜀
# 결과·로그: $RA_BUILD/check/<시각>/
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/config/paths.env"
SIM=0; BUILD=1
for a in "$@"; do case $a in --sim) SIM=1 ;; --no-build) BUILD=0 ;; *) echo "모르는 인자 $a"; exit 2 ;; esac; done
OUT=$RA_BUILD/check/$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
fails=(); oks=()
step() { # step <이름> <명령...> — 로그는 $OUT/<이름>.log
  local name=$1; shift
  local t0=$SECONDS
  if "$@" > "$OUT/$name.log" 2>&1; then oks+=("$name ($((SECONDS - t0)) s)"); echo "  ok    $name"
  else fails+=("$name"); echo "  실패  $name  → $OUT/$name.log"; fi
}
cd "$ROOT"
echo "[check] $OUT"

[ $BUILD = 1 ] && step build tools/build_all.sh
step paths tools/check_paths.sh
step audit bash -c 'python3 tools/audit.py | tee /dev/stderr | grep -q "남은 항목 0"'
step ctest_scenemap ctest --test-dir "$RA_BUILD/scenemap" --output-on-failure
step cargo_move_robot bash -c "cd src/agent/tools/move_robot && CARGO_TARGET_DIR=$RA_BUILD/cargo/test_move_robot cargo test -q --release"
step cargo_runtime bash -c "cd src/agent/runtime && SGCLIP_LIB_DIR=$RA_BUILD/sgclip CARGO_TARGET_DIR=$RA_BUILD/cargo/test_runtime cargo test -q --release"
step cargo_trainview bash -c "cd training/viewer && CARGO_TARGET_DIR=$RA_BUILD/cargo/test_trainview cargo test -q --release"

# 실제 bag 한 판(OpenLORIS office1-1): ObjectSAM + SigLIP 2 + objprob + Cartographer 끝까지
BAG=$RA_DATASETS/realbags/streams/ol_office1-1
if [ -d "$BAG" ]; then
  step realbag bash -c "build/bin/realbag_run '$BAG' '$OUT/realbag' && test -s '$OUT/realbag/objects.csv' && test -s '$OUT/realbag/memory/scene.json'"
else echo "  건너뜀 realbag (없음: $BAG)"; fi

# 시뮬 기록(라이다 포함) → libsgrt 로 다시 돌려 학습 뷰어 판 만들기
REC=$(ls -t "$RA_DATASETS"/outputs/*/rec.bin "$RA_DATASETS"/limo_rec/*.bin 2>/dev/null | head -1)
if [ -n "$REC" ]; then
  step og2sg bash -c "build/bin/og2sg --rec '$REC' --run '$(dirname "$REC")' --out '$OUT/og2sg/ep.sg' --frames 600 && test -s '$OUT/og2sg/ep.sg/stream.sgs'"
else echo "  건너뜀 og2sg (기록 없음)"; fi

# 뷰어 둘: 띄워서 응답 확인 후 끔
step sgview bash -c "build/bin/sgview '$OUT/realbag/memory' --port 18090 & p=\$!; sleep 3; curl -sf http://127.0.0.1:18090/ >/dev/null; r=\$?; kill \$p; exit \$r"
step trainview bash -c "build/bin/trainview --root '$RA_TRAINVIEW_WORK/behavior_og' --port 18095 & p=\$!; sleep 3; curl -sf http://127.0.0.1:18095/api/runs | grep -q runs; r=\$?; kill \$p; exit \$r"

if [ $SIM = 1 ]; then
  step sim_explore bash -c "MAXSIM=60 MAXCALLS=10 flock '${OG_LOCK:-/tmp/claude-1000/og.lock}' src/sim/explore/run_explore.sh frontier turning_on_radio check && d=\$(ls -td data/outputs/explore_*_check | head -1) && test -s \$d/memory/scene.json"
fi

echo
echo "[check] 통과 ${#oks[@]}, 실패 ${#fails[@]}"
for f in "${fails[@]}"; do echo "  실패: $f ($OUT/$f.log)"; done
[ ${#fails[@]} -eq 0 ]
