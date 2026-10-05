#!/bin/bash
# objprob 엔진별 다시 맞추기·비교(README "엔진별 objprob 매개변수"):
#   bash objprob_refit.sh <이름> <엔진 .plan | -> [r3 검출 캐시 dets.gz] [office1-1 캐시] [office1-5 캐시]
# 1) 검출(GPU, 짧게): 캐시를 안 주면 그 엔진으로 radio r3(det-every 1)·OpenLORIS office1-1·1-5(det-every 3)를 한 번씩 돌려 RBD2 캐시
#    (확률 모드 + 내장 기본값 — 캐시는 검출·임베딩뿐이라 매개변수와 상관없음)
# 2) 맞추기(CPU): objprob_fit.py → $OUT/fit/{fit.json,label_prior.json,objprob_params.json}
# 3) 재생(--load, 통째 다시 담기만 GPU): 문턱(same_p/merge_p) 몇 쌍 × 새 매개변수, 그리고 비교 기준 PARAMS_BASE(기본 = 옛 엔진 파일)로
#    r3 slam·gt → objprob_eval.py, OpenLORIS → 기하 대용(천장 띠·벽 같은 판·바닥 조각·같은 이름 0.5 m 쌍)
# 결과: $OUT/summary.txt(판마다 한 줄). 판 이름 = <매개변수>_<same_p>_<merge_p>
# 환경: TH="0.6/0.7 …"(문턱 쌍), EXTRA_KV="ap_bridge_drop=1"(새 매개변수 판에 더할 것), PREFIX(판 이름 앞, 기본 fit), NO_BASE=1(비교 기준 판 안 돎),
#       CONF(검출 conf, 기본 0.25), REALBAG_BIN(빌드 폴더), OUT
set -euo pipefail
N=$1; ENG=$2; R3D=${3:-}; O11=${4:-}; O15=${5:-}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${REALBAG_BIN:-$HOME/realbag_build}/realbag_run
PY=${PY:-$HOME/realbag_venv/bin/python}
OUT=${OUT:-$HOME/datasets/objprob/refit/$N}
S=$HOME/datasets/sim_detcmp/streams/radio_limo_r3
OL=$HOME/datasets/realbags/streams
GTPGM=${GTPGM:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/src/sim/explore/gt/house_double_floor_lower__turning_on_radio.pgm}
PARAMS_BASE=${PARAMS_BASE:-$HERE/objprob_params/FastSAM-s-416.json}
TH=${TH-"0.6/0.7 0.5/0.7 0.7/0.7 0.6/0.8 0.7/0.8 0.5/0.6"}
CONF=${CONF:-0.25}   # 검출 conf(보정한 엔진은 0.25 = 보정 문턱 t. 더 높이면 실효 t 가 오름)
COMMON=(--det-every 1 --max-depth 4 --conf $CONF)
OLX=(--pose slam --max-depth 4 --conf $CONF)
mkdir -p "$OUT/logs"
ENGX=(); [ "$ENG" != - ] && ENGX=(--engine "$ENG")

# 1) 검출 캐시
if [ -z "$R3D" ]; then
  R3D=$OUT/det/r3.gz; mkdir -p "$OUT/det"
  "$BIN" "$S" "$OUT/det/r3_slam" --pose slam "${COMMON[@]}" --det fastsam "${ENGX[@]}" --objprob --objprob-params none --dump "$R3D" > "$OUT/logs/det_r3.log" 2>&1
fi
if [ -z "$O11" ]; then
  for k in 1 5; do
    "$BIN" "$OL/ol_office1-$k" "$OUT/det/ol1${k}_slam" "${OLX[@]}" --det fastsam "${ENGX[@]}" --objprob --objprob-params none \
      --dump "$OUT/det/ol1$k.gz" > "$OUT/logs/det_ol1$k.log" 2>&1
  done
  O11=$OUT/det/ol11.gz; O15=$OUT/det/ol15.gz
fi
WALLRUN=$OUT/det/r3_slam
[ -d "$WALLRUN" ] || { "$BIN" "$S" "$OUT/wallrun" --pose slam "${COMMON[@]}" --load "$R3D" --objprob --objprob-params "$PARAMS_BASE" > "$OUT/logs/wallrun.log" 2>&1; WALLRUN=$OUT/wallrun; }

# 2) 맞추기
[ -f "$OUT/fit/objprob_params.json" ] || "$PY" "$HERE/objprob_fit.py" "$S" "$R3D" "$OUT/fit" --walls "$WALLRUN/walls.csv" --metrics "$WALLRUN/metrics.json" \
  --engine "$(basename "$ENG")" > "$OUT/logs/fit.log" 2>&1

# 3) 재생·채점
olcheck() {   # 기하 대용(realcheck 와 같은 셈)
  "$PY" - "$1" <<'E'
import csv, json, sys
d = sys.argv[1]
m = json.load(open(f'{d}/metrics.json'))
rows = [r for r in csv.DictReader(open(f'{d}/objects.csv')) if r['state'] != 'gone']
f = lambda r, k: float(r[k])
ceil = sum(1 for r in rows if f(r, 'z') + f(r, 'ez') / 2 > 2.3)
plane = sum(1 for r in rows if max(f(r, 'ex'), f(r, 'ey')) > 1.5 and min(f(r, 'ex'), f(r, 'ey')) < 0.12 and f(r, 'ez') > 1.0)
flat = sum(1 for r in rows if f(r, 'ez') < 0.06 and f(r, 'z') < 0.1 and max(f(r, 'ex'), f(r, 'ey')) > 0.8)
o = m['objects']
print(json.dumps(dict(live=o.get('live_all'), dup_pairs=o.get('dup_pairs_same_name_0p5m_all'), ceiling_band=ceil, wall_like=plane, floor_like=flat)))
E
}
: > "$OUT/summary.txt"
run() {   # run <판 이름> <params json> <same_p> <merge_p>
  local tag=$1 pj=$2 sp=$3 mp=$4 X=()
  local env="ap_same_p=$sp,ap_merge_p=$mp${EXTRA_KV:+,$EXTRA_KV}"
  for pose in slam gt; do
    SM_OBJ_PARAMS=$env "$BIN" "$S" "$OUT/runs/${tag}_$pose" --pose $pose "${COMMON[@]}" --load "$R3D" --objprob --objprob-params "$pj" \
      > "$OUT/logs/${tag}_$pose.log" 2>&1
  done
  "$PY" "$HERE/objprob_eval.py" "$S" "$GTPGM" "$OUT/runs/${tag}_slam,$OUT/runs/${tag}_gt" --json "$OUT/runs/${tag}_eval.json" > "$OUT/runs/${tag}_eval.txt" 2>&1
  local ol=""
  for k in 1 5; do
    local cache=$O11; [ $k = 5 ] && cache=$O15
    SM_OBJ_PARAMS=$env "$BIN" "$OL/ol_office1-$k" "$OUT/runs/${tag}_ol1$k" "${OLX[@]}" --load "$cache" --objprob --objprob-params "$pj" \
      > "$OUT/logs/${tag}_ol1$k.log" 2>&1
    ol="$ol ol1-$k $(olcheck "$OUT/runs/${tag}_ol1$k")"
  done
  echo "$tag$ol" >> "$OUT/summary.txt"
  grep -E '^\{"run"|fp_by_structure|wrong_merges|struct_objects' "$OUT/runs/${tag}_eval.txt" | cut -c1-400 >> "$OUT/summary.txt"
}
for t in $TH; do
  run "${PREFIX:-fit}_${t%/*}_${t#*/}" "$OUT/fit/objprob_params.json" "${t%/*}" "${t#*/}"
done
[ -n "${NO_BASE:-}" ] || run base_0.6_0.7 "$PARAMS_BASE" 0.6 0.7
echo "[refit] $N done → $OUT/summary.txt"
