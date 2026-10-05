#!/bin/bash
# 외부 경로·도구가 다 있는지 본다(없는 것만 말해 줌). 경로는 config/paths.env(+ paths.local.env).
ROOT=$(cd "$(dirname "$0")/.." && pwd); . "$ROOT/config/paths.env"
miss=0
chk() { # chk <종류> <경로> <없으면 어떻게>
  if [ -e "$2" ]; then printf '  ok    %-14s %s\n' "$1" "$2"; else printf '  없음  %-14s %s  -> %s\n' "$1" "$2" "$3"; miss=$((miss+1)); fi
}
echo "[경로] config/paths.env"
chk ovdet      "$OVDET_MODELS/x86_sm120/yolo26n-seg-obj-416.plan" "ObjectSAM 엔진(OVDET_MODELS)"
chk labels     "$RA_LABELS/table.jsonl"             "라벨 표 objects-v1(RA_LABELS)"
chk records    "$RA_DATASETS"                       "실행·평가 기록 폴더(RA_DATASETS)"
chk train-data "$RA_B1K_SCENES"                     "학습 장면 묶음(RA_B1K_SCENES)"
chk BEHAVIOR-1K "$B1K_ROOT/datasets"                "시뮬 장면 원천. B1K_ROOT 를 맞추거나 받을 것"
chk og-python  "$OG_PYTHON"                         "conda env $OG_CONDA_ENV (OmniGibson)"
echo "[도구]"
for c in cmake cargo g++ nvidia-smi; do command -v $c >/dev/null && printf '  ok    %s\n' $c || { printf '  없음  %s\n' $c; miss=$((miss+1)); }; done
chk nvcc       "$RA_CUDA_ROOT/bin/nvcc"             "RA_CUDA_ROOT"
echo "[빌드 결과] $RA_BUILD/bin (tools/build_all.sh)"
for b in libsgrt.so sgview realbag_run sgs_play og2sg trainview run-skill; do chk build "$RA_BUILD/bin/$b" "tools/build_all.sh"; done
[ $miss -eq 0 ] && echo "모두 있음" || echo "빠진 것 $miss 개"
exit $((miss>0))
