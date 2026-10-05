#!/bin/bash
# 실제 bag(OpenLORIS office1-1·1-5) 비교: 같은 스트림에 엔진 하나로 realbag_run(slam, det-every 3, max-depth 4 — sim_detcmp
# openloris_fullcount 와 같은 설정) → 살아 있는 노드 수·같은 이름 0.5 m 쌍·구조물처럼 보이는 노드(정답 없음 → 기하 대용).
#   bash realcheck.sh <이름> <엔진 .plan> [plain|objprob]
set -euo pipefail
. "$(dirname "$0")/env_local.sh"
N=$1; ENG=$2; MODE=${3:-plain}
DATA=${FASTSAM_DATA:-$RA_DATASETS/fastsam_obj}
BIN=${REALBAG_BIN:-$RA_BUILD/realbag}
OLD=ap''rime
PD=$RA_DATASETS/objprob; [ -d "$PD" ] || PD=$RA_DATASETS/$OLD
FLAG=--objprob; grep -aq -- "--objprob" "$BIN/realbag_run" || FLAG=--$OLD
X=(); [ "$MODE" = objprob ] && X=($FLAG --label-prior "$PD/fit1/label_prior.json")
for s in ol_office1-1 ol_office1-5; do
  R=$DATA/realcheck/${N}_${MODE}_$s
  rm -rf "$R"; mkdir -p "$DATA/realcheck"
  "$BIN/realbag_run" "$RA_DATASETS/realbags/streams/$s" "$R" --pose slam --max-depth 4 --det fastsam --engine "$ENG" "${X[@]}" \
    > "$R.log" 2>&1
done
$FS_PY - "$DATA/realcheck" "$N" "$MODE" <<'E'
import csv, json, sys
root, n, mode = sys.argv[1:]
for s in ('ol_office1-1', 'ol_office1-5'):
    d = f'{root}/{n}_{mode}_{s}'
    m = json.load(open(f'{d}/metrics.json'))
    rows = [r for r in csv.DictReader(open(f'{d}/objects.csv')) if r['state'] != 'gone']
    f = lambda r, k: float(r[k])
    ceil = sum(1 for r in rows if f(r, 'z') + f(r, 'ez') / 2 > 2.3)                       # 천장 띠
    plane = sum(1 for r in rows if max(f(r, 'ex'), f(r, 'ey')) > 1.5 and min(f(r, 'ex'), f(r, 'ey')) < 0.12
                and f(r, 'ez') > 1.0)                                                       # 벽 같은 큰 세운 판
    flat = sum(1 for r in rows if f(r, 'ez') < 0.06 and f(r, 'z') < 0.1 and max(f(r, 'ex'), f(r, 'ey')) > 0.8)  # 바닥 조각
    o = m['objects']
    out = dict(stream=s, live_all=o.get('live_all'), live_small=o.get('live'), dup_pairs_all=o.get('dup_pairs_same_name_0p5m_all'),
               ceiling_band=ceil, wall_like=plane, floor_like=flat, dets_per_frame=round(m['dets_per_frame'], 2),
               det_ms=round(m['det_ms'], 3), det_gpu_mb=m.get('det_gpu_mb'), wall_segments=m['map'].get('wall_segments'))
    json.dump({'summary': out, 'by_label_all': o.get('by_label_all')}, open(f'{d}/check.json', 'w'), indent=1)
    print(json.dumps(out))
E
