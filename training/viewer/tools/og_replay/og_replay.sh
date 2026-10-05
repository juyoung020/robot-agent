#!/bin/bash
# GPU 환경 BEHAVIOR 판 하나 → OmniGibson 다시 돌리기 + 우리 인지 → <판>_og.sg (og_replay.py). og.lock 을 잡고 낮은 우선순위로.
#   og_replay.sh <판.trp> [og_replay.py 인자…]
# 끝나면(성공·실패 모두) OG 텍스처 캐시·임시 폴더를 지운다. 램 15 GB·디스크 15 GB·GPU 메모리 7 GB 가 비어 있지 않으면 기다린다(최대 30 분, 넘으면 종료 코드 75).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
TRP=$1; shift
OG_LOCK=${OG_LOCK:-/tmp/claude-1000/og.lock}
B1K=${B1K_ROOT:-$REPO/src/behavior-2026/BEHAVIOR-1K}
TEXCACHE=$B1K/OmniGibson/appdata/global/cache/texturecache
TMPBASE=$HOME/trainview_work/og_tmp
mkdir -p "$TMPBASE"
OUT=${TRP%.trp}_og.sg
cleanup() {
  rm -rf "$TEXCACHE" "$TMPBASE"/ogrp_* 2>/dev/null
  find /tmp -maxdepth 1 -name 'tmp*' -user "$(id -u)" -mmin +1 -newer "$STAMP" -exec rm -rf {} + 2>/dev/null
  rm -f "$STAMP" "$STAMP.gpu"
}
STAMP=$(mktemp "$TMPBASE/stamp.XXXX")
trap cleanup EXIT
ok_res() {
  local mem disk gpu
  mem=$(free -g | awk '/^Mem:/{print $7}')
  disk=$(df --output=avail -k "$HOME" | tail -1)
  gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${mem:-0}" -ge 15 ] && [ "${disk:-0}" -ge $((15 * 1024 * 1024)) ] && [ "${gpu:-0}" -ge "${OG_GPU_MIN_MB:-5000}" ]
}
for _ in $(seq 180); do ok_res && break; sleep 10; done
ok_res || { echo "[og_replay] resources busy (RAM/disk/GPU) — later"; exit 75; }
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${OG_CONDA_ENV:-behavior}"
export OMNI_KIT_ACCEPT_EULA=YES B1K_ROOT=$B1K
ulimit -c 0
SCENE=$(python3 -c "import json,struct,sys;b=open(sys.argv[1],'rb').read(132000);n=struct.unpack('<I',b[4:8])[0];h=json.loads(open(sys.argv[1],'rb').read(8+n)[8:]);w=h['scene']['world'];print(w['rasc'],w['task'])" "$TRP") || exit 2
set -- "$@"
flock "$OG_LOCK" nice -n 19 timeout 7200 python "$HERE/og_replay.py" --trp "$TRP" --out "$OUT" "$@" &
PY=$!
# 학습을 지킴: GPU 빈 메모리가 OG_GPU_FLOOR_MB(기본 600) 아래로 떨어지면 OG 쪽을 끈다(학습기가 아니라 이 판이 진다)
( while kill -0 $PY 2>/dev/null; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$f" ] && [ "$f" -lt "${OG_GPU_FLOOR_MB:-600}" ]; then echo "[og_replay] GPU free ${f} MB < floor — stopping OG to protect training"; touch "$STAMP.gpu"; pkill -TERM -P $PY; kill -TERM $PY; sleep 5; pkill -KILL -P $PY; fi
    sleep 2
  done ) &
WD=$!
wait $PY
rc=$?
kill $WD 2>/dev/null
if [ -f "$OUT/meta.json" ]; then
  read -r RASC TASK <<<"$SCENE"
  "${OG2SG:-$HOME/ra_og2sg/og2sg}" --underlay "$OUT/underlay.json" --rasc "$RASC" --task "$TASK" || true
  echo "[og_replay] done rc=$rc -> $OUT"
  exit 0
fi
rm -rf "$OUT"
if [ -f "$STAMP.gpu" ]; then rm -f "$STAMP.gpu"; echo "[og_replay] stopped for GPU memory — later ($TRP)"; exit 75; fi
echo "[og_replay] failed rc=$rc ($TRP)"
exit 1
