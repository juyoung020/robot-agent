#!/bin/bash
# 학습 결과 .pt → ONNX(입력 크기 IMGSZ, 기본 416, opset 13, end2end 없음 — 옛 FastSAM-s-416.onnx 와 같은 모양
# 1×37×A + 1×32×(S/4)×(S/4)) → TensorRT FP16 엔진(build_engine.py, ovdet 엔진과 같은 설정).
#   bash export.sh <best.pt> <이름(예: FastSAM-s-416-obj)>
# 결과: $OUT_PT/<이름>.pt·.onnx, $OUT_PLAN/<이름>.plan(+ .names.txt = 'object')
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/env_local.sh" ] && source "$HERE/env_local.sh"
PT=$1; NAME=$2; IMGSZ=${IMGSZ:-416}
P=${OUT_PT:-$HOME/ovdet_models/pt}; E=${OUT_PLAN:-$HOME/ovdet_models/x86_sm120}; mkdir -p "$P" "$E"
FS_PY=${FS_PY:-python}; TRT_PY=${TRT_PY:-python}
[ -e "$E/$NAME.plan" ] && [ "${FORCE:-0}" != 1 ] && { echo "$E/$NAME.plan exists (FORCE=1 to overwrite)"; exit 1; }
cp "$PT" "$P/$NAME.pt"
T=$(mktemp -d); cp "$PT" "$T/$NAME.pt"
( cd "$T" && "$FS_PY" -c "
from ultralytics import YOLO
YOLO('$NAME.pt').export(format='onnx', imgsz=$IMGSZ, opset=13, simplify=True, dynamic=False, end2end=False)" )
cp "$T/$NAME.onnx" "$P/$NAME.onnx"; rm -rf "$T"
"$FS_PY" -c "
import onnx; m = onnx.load('$P/$NAME.onnx')
print([(t.name, [d.dim_value for d in t.type.tensor_type.shape.dim]) for t in list(m.graph.input) + list(m.graph.output)])"
if "$TRT_PY" -c "import tensorrt" 2>/dev/null; then
  "$TRT_PY" "$HERE/build_engine.py" "$P/$NAME.onnx" --out "$E" --workspace-gb 2
  ls -la "$E/$NAME.plan" "$E/$NAME.plan.names.txt"
else
  echo "tensorrt not available: ONNX only ($P/$NAME.onnx)"
fi
