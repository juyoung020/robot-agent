# ObjectSAM 학습 스크립트 공통 설정 — 경로는 config/paths.env 에서 받는다(이 PC 전용 값은 config/paths.local.env).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config/paths.env"
export B1K_ROOT OG_CONDA_ENV FS_PY TRT_PY
export OG_LOCK=${OG_LOCK:-/tmp/claude-1000/og.lock}
export FASTSAM_DATA=${FASTSAM_DATA:-$RA_TRAIN_DATA/fastsam_obj}
export OVDET_LIB=${OVDET_LIB:-$RA_BUILD/sgrt/ovdet/libovdet.so}
export BASE_PLAN=${BASE_PLAN:-$OVDET_MODELS/x86_sm120/FastSAM-s-416.plan}
