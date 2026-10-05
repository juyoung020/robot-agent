#!/bin/bash
# 스킬 explore 한 판(시뮬): 평가기(run_explore.py, 지도·move_robot) + 에이전트(robot-agent 런타임 run-skill --skill explore, LLM 또는 기준선 frontier).
#   run_explore.sh <policy llm|frontier> <task> [tag] [extra agent args...]
# 결과: outputs/explore_<ts>_<task>_<policy>[_tag]/ (memory/ 지도, decisions.jsonl, timeline.jsonl, summary.json, sim.log)
# 사전 조건: VRAM 여유 ≥ 9 GB, RAM 여유 ≥ 16 GB (아니면 기다린다). 키는 환경변수로만(kau.env).
# 로봇: 우리 LIMO + OMX-F 하나뿐(R1 Pro 는 10-06 에 뺌; BEHAVIOR 과제 틀의 R1 은 장면·시작 자세만 읽고 로봇은 --robot-config 로 바꿔 끼움, robot_poses 키는 eval_with_limo.py 가 맞춤). 평가기를 $ROBOT_AGENT/src/robot/og/eval_with_limo.py 로
#   띄우고(--robot-config limo_omx_eval.yaml), move_robot 은 libmove_robot 이 LIMO + OMX-F 를 직접 말한다(base·arm·gripper, 탐사는 base 만 씀; move_robot_limo.py 는 행동 8 을 시뮬 행동 9 칸에 놓기만),
#   정답 자세·물체 기록(SGRT_GT_LOG=<out>/gt_poses.csv, .objects.json)·poses.csv·pose_diag.json 기본 켬. 가까운 자르기는
#   robot-agent 391c04b 부터 eval_with_limo.py 가 0.05 m 로 둔다(옛 자산이면 LIMO_NEAR_CLIP, 기본 0.05 까지만 올림).
set -u
POL=$1; TASK=$2; TAG=${3:-}; shift 3 2>/dev/null || shift $#
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)            # robot-agent
. "$REPO/config/paths.env"
SUPER=$REPO
AGENT_BIN=src/agent/runtime/target/release/run-skill   # 에이전트 런타임(robot-agent 10-06: 옛 src/agent/skills/explore/target/release/explore)
TS=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/data/outputs/explore_${TS}_${TASK}_${POL}${TAG:+_$TAG}
ROBOT=${SGRT_ROBOT:-limo_omx}
case "$ROBOT" in limo_omx) ;; *) echo "[run] SGRT_ROBOT=$ROBOT 모름 — 우리 로봇은 limo_omx 하나"; exit 1;; esac
export MOVE_ROBOT_LIB=${MOVE_ROBOT_LIB:-$RA_BUILD/bin/libmove_robot.so}   # 서브모듈 밖(클론)에서도
GT=$HERE/gt   # gt/ 는 git 밖(gt_trav.py 로 만듦)
ROBOT_ARGS=(); EVAL_ROBOT=()
if [ "$ROBOT" = limo_omx ]; then
  OGDIR=${LIMO_OG_DIR:-$SUPER/src/robot/og}
  ROBOT_ARGS=(--robot limo_omx --limo-shim "${LIMO_SHIM:-$OGDIR/eval_with_limo.py}")
  EVAL_ROBOT=(--robot-config "$OGDIR/limo_omx_eval.yaml")
  export SGRT_ROBOT=limo_omx SGRT_GT_LOG=${SGRT_GT_LOG:-$OUT/gt_poses.csv}
fi
PORT=${PORT:-8771}
MAXSTEPS=${MAXSTEPS:-27000}
mkdir -p "$OUT"
while true; do
  fv=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}')
  fr=$(free -g | awk '/Mem:/{print $7}')
  [ "$fv" -ge 9000 ] && [ "$fr" -ge 16 ] && break
  echo "[run] waiting: VRAM free $fv MiB, RAM avail $fr GB"; sleep 30
done
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES
export SGRT_POSE=${SGRT_POSE:-carto}   # 실제 로봇과 같게 Cartographer(시뮬 2D 라이다 + 바퀴 오도메트리). 정답 자세 확인용은 gt
export SGRT_LIB=${SGRT_LIB:-$RA_BUILD/bin/libsgrt.so}
# 검출 = ObjectSAM(YOLO26n 학생) + SigLIP 2 + objprob(libsgrt 가 분할 엔진이면 켬), 살펴본 정도 켬. 다른 엔진은 SGRT_ENGINE(sgrt_glue.py)
export SGRT_ENGINE=${SGRT_ENGINE:-$OVDET_MODELS/x86_sm120/yolo26n-seg-obj-416.plan} SGRT_INSPECT=${SGRT_INSPECT:-1}
if [ "${SGRT_OBJPROB:-}" != 0 ] && ! grep -aqF sgrt_objprob_enabled "$SGRT_LIB"; then echo "[run] $SGRT_LIB 에 objprob 앞단(sgrt_objprob_enabled)이 없다(옛 빌드) — 다시 빌드할 것"; exit 1; fi
if [ -n "${SGRT_STREAM:-}" ] && ! strings "$SGRT_LIB" | grep -q SGRT_STREAM; then echo "[run] $SGRT_LIB 에 SGRT_STREAM 이 없다(옛 빌드) — 다시 빌드할 것"; exit 1; fi   # 뷰어가 조용히 비는 실수 방지
if [ "$ROBOT" = limo_omx ] && ! grep -aqF sgrt_set_robot "$SGRT_LIB"; then echo "[run] $SGRT_LIB 에 로봇 고르기(sgrt_set_robot)가 없다(옛 빌드) — 다시 빌드할 것"; exit 1; fi
cd "$OUT"
python "$HERE/run_explore.py" --listen 127.0.0.1:$PORT --out "$OUT" --gt-dir "$GT" "${ROBOT_ARGS[@]}" -- --task-name "$TASK" --mode public_test \
  --instance-indices 0 --num-envs 1 --max-steps $MAXSTEPS --headless "${EVAL_ROBOT[@]}" \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper > "$OUT/sim.log" 2>&1 &
SIM=$!
echo "[run] sim pid $SIM out $OUT"
# 평가기가 첫 관측을 받을 때까지(지도가 생김)
for i in $(seq 1 120); do
  grep -q "reference set\|Traceback" "$OUT/sim.log" 2>/dev/null && break
  kill -0 $SIM 2>/dev/null || break
  sleep 5
done
grep -q "Traceback" "$OUT/sim.log" && { echo "[run] sim failed"; tail -30 "$OUT/sim.log"; kill $SIM 2>/dev/null; exit 1; }
set -a; . ~/.config/robot-agent/kau.env; set +a
"$SUPER/$AGENT_BIN" --skill explore --policy "$POL" --addr 127.0.0.1:$PORT --out "$OUT" --task "$TASK" \
  --max-calls ${MAXCALLS:-80} --max-sim-s ${MAXSIM:-880} --max-wall-s ${MAXWALL:-3600} "$@" > "$OUT/agent.log" 2>&1
echo "[run] agent done: $(tail -c 300 $OUT/agent.log | tr '\n' ' ')"
kill -INT $SIM 2>/dev/null; sleep 20; kill $SIM 2>/dev/null; wait $SIM 2>/dev/null
echo "[run] finished $OUT"
