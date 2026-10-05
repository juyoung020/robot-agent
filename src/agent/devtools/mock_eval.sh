#!/bin/bash
# 가짜 집(정답 바닥 + 카메라 흉내)에서 LLM 과 기준선을 같이 돌려 비교. Isaac Sim 없이 몇 분.
#   mock_eval.sh <tag> [run-skill 인자...]    → $RA_ROOT/data/explore_mock_<tag>/<집>_<정책>[_s<k>]/
# 실행은 에이전트 런타임(src/agent/runtime, run-skill --skill explore). 기준선 frontier = skill.json baseline(move_robot 모드 explore).
# 출발: 과제 instance 시작 자세(s0)와 같은 자리에서 방향만 120°, 240° 돌린 것(s1, s2). STARTS=1 이면 s0 만.
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../config/paths.env"
set -a; [ -f "${KAU_ENV:-$HOME/.config/robot-agent/kau.env}" ] && . "${KAU_ENV:-$HOME/.config/robot-agent/kau.env}"; set +a   # LLM 키(frontier 만 돌리면 없어도 됨)
TAG=$1; shift
BIN=$RA_BUILD/bin/run-skill
[ -x "$BIN" ] || { echo "런타임 없음: tools/build_all.sh agent"; exit 1; }
GT=$HERE/mock_gt                 # 정답 바닥 지도(gt_trav.py 가 만든 것, 작아서 저장소에 둠)
O=$RA_ROOT/data/explore_mock_$TAG
STARTS=${STARTS:-3}; POLS=${POLS:-"llm frontier"}
mkdir -p $O
for H in house_single_floor__bringing_water house_double_floor_lower__turning_on_radio; do
  for k in $(seq 0 $((STARTS-1))); do
    ST=$(python3 -c "import json,math;d=json.load(open('$GT/$H.json'))['start_pose_world'];print('%f,%f,%f'%(d[0],d[1],math.degrees(d[2])+120*$k))")
    for P in $POLS; do
      while [ $(jobs -r | wc -l) -ge ${PAR:-2} ]; do wait -n; done
      D=$O/${H%%__*}_${P}_s$k; rm -rf $D
      $BIN --skill explore --policy $P --mock $GT/$H.pgm --gt $GT/$H.json --out $D --task ${H##*__} --max-calls 60 --start $ST "$@" > $D.log 2>&1 &
    done
  done
done
wait
python3 - "$O" <<'P'
import json,sys,glob,os
rows=[]
for d in sorted(glob.glob(sys.argv[1]+'/*/summary.json')):
    s=json.load(open(d)); m={c['cov']:c for c in s['coverage_marks']}
    n=os.path.basename(os.path.dirname(d))
    print('%-38s cov %.3f contacts %3s calls %2d llm %2d tok %6d path %5.1f sim %4.0f wall %4.0f p@90 %s end %s'%(n,s['gt_cov'] or 0,int(s['contacts'] or 0),s['calls'],s['llm_calls'],s['prompt_tokens']+s['completion_tokens'],s['path_m'] or 0,s['sim_s'] or 0,s['wall_s'],m.get(0.9,{}).get('path_m'),s['end']))
P
