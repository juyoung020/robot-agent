#!/bin/bash
# OpenLORIS office1-1..7: Cartographer(realbag_run 기본 --pose carto) — 같은 검출 캐시(ObjectSAM + SigLIP 2 + objprob)로 판마다 둘
# (gt = 지도 기준, carto) 돌리고, 옛 scenemap slam2d 판(<out>/slam_<k>, 10-06 archive 전에 잰 것)이 있으면 같이 표로 낸다.
#   carto_vs_slam2d.sh [out=$RA_DATASETS/realbags/carto_cmp]      (KS="1 5" 로 판 고르기)
# 사전: 스트림에 scans.bin(bag2stream.py --scan-only), tools/build_all.sh cartographer realbag. 검출 캐시가 없으면 한 번 GPU 로 만든다.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../../config/paths.env"
BIN=$RA_BUILD/realbag/realbag_run
PY=${PY:-python3}   # 표는 표준 라이브러리만
OL=$RA_DATASETS/realbags/streams
O=${1:-$RA_DATASETS/realbags/carto_cmp}
X=(--max-depth 4 --conf 0.25 --objprob)
mkdir -p "$O/det" "$O/logs"
for k in ${KS:-1 2 3 4 5 6 7}; do
  d=$OL/ol_office1-$k
  [ -f "$O/det/ol1$k.gz" ] || "$BIN" "$d" "$O/det/ol1${k}_dump" "${X[@]}" --dump "$O/det/ol1$k.gz" > "$O/logs/dump_$k.log" 2>&1
  "$BIN" "$d" "$O/gt_$k" --pose gt "${X[@]}" --load "$O/det/ol1$k.gz" > "$O/logs/gt_$k.log" 2>&1
  "$BIN" "$d" "$O/carto_$k" "${X[@]}" --load "$O/det/ol1$k.gz" --ref-map "$O/gt_$k/memory" > "$O/logs/carto_$k.log" 2>&1 || echo "FAIL carto $k"
  echo "[cmp] office1-$k done"
done
"$PY" "$HERE/carto_vs_slam2d_table.py" "$O" | tee "$O/table.md"
