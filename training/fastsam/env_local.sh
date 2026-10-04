# 이 PC(robot-agent) 설정 — 공개 저장소에는 넣지 않는다. 스크립트가 있으면 읽는다.
export B1K_ROOT=${B1K_ROOT:-$HOME/robot-agent/src/behavior-2026/BEHAVIOR-1K}
export OG_LOCK=${OG_LOCK:-/tmp/claude-1000/og.lock}
export OG_CONDA_ENV=${OG_CONDA_ENV:-behavior}
export FASTSAM_DATA=${FASTSAM_DATA:-$HOME/datasets/fastsam_obj}
export OVDET_LIB=${OVDET_LIB:-$HOME/ovdet_build/libovdet.so}
export BASE_PLAN=${BASE_PLAN:-$HOME/ovdet_models/x86_sm120/FastSAM-s-416.plan}
export FS_PY=${FS_PY:-$HOME/fastsam_venv/bin/python}
export TRT_PY=${TRT_PY:-$HOME/ovdet_venv/bin/python}
