#!/bin/bash
# training/fastsam → 공개 저장소 ObjectSAM 작업 트리(tools/)로 복사하면서 이 PC 기본 경로를 저장소 상대 경로로 바꾼다.
#   bash publish_objectsam.sh <ObjectSAM 작업 트리>
# 이 PC 전용 파일(env_local.sh, e2e.sh, realcheck.sh, README.md)은 넣지 않는다. 라이선스·README 는 저장소 쪽에서 관리.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$1/tools
mkdir -p "$DST"
FILES=(common.py evalsets.py realsets.py select_real.py build_data.py train.py calibrate.py eval_det.py sweep_t.py ref_detect.py
       make_table.py bench_latency.py sim_render.py label_sim.py scenes.sh render_all.sh render_tasks.sh export.sh
       build_engine.py cand_eval.sh heldout_lvis.txt)
for f in "${FILES[@]}"; do
  sed -e 's|~/fastsam_venv/bin/python |python |g' \
      -e "s|f'{HOME}/ovdet_models/x86_sm120/FastSAM-s-416.plan'|'models/FastSAM-s-416.onnx'|g" \
      -e "s|f'{c.HOME}/ovdet_models/pt/FastSAM-s.pt'|'models/FastSAM-s.pt'|g" \
      -e "s|f'{HOME}/ovdet_build/libovdet.so'|'libovdet.so'|g" \
      -e 's|~/ovdet_models/x86_sm120/FastSAM-s-416.plan|models/FastSAM-s-416.onnx|g' \
      -e 's|~/ovdet_models/x86_sm120/FastSAM-s-416-obj.plan|models/FastSAM-s-416-obj.onnx|g' \
      -e 's|\$HOME/ovdet_models/pt|out|g' -e 's|\$HOME/ovdet_models/x86_sm120|out|g' \
      -e 's|"$HOME/ovdet_models/x86_sm120/FastSAM-s-416.plan.names.txt"|/dev/null|g' \
      -e 's|(sim_detcmp 과 같은 판, 라디오)|(라디오)|g' \
      "$HERE/$f" > "$DST/$f"
done
chmod +x "$DST"/*.sh
if grep -rn -i -E "juyoung|claude|/home/|robot-agent|behavior-2026|ovdet_models|fastsam_venv" "$DST"; then
  echo "publish: local names left (above)"; exit 1
fi
echo "publish: ${#FILES[@]} files -> $DST"
