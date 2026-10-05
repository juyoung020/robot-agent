#!/bin/bash
# SigLIP 2 B/32 efficiency study (docs/clip_candidates.md 8): export -> TRT build -> accuracy on the eval set -> GPU time.
#   tools/study.sh NAME "EXPORT_ARGS" "BUILD_ARGS"
# e.g.  tools/study.sh tome4 "--tome 4@6" "--half-input --pin norm,mlp/act"
# Outputs (not in git): models/ovdet/x86_sm120/siglip2_b32/study/NAME.{onnx,plan}, data/clip_bench/study.jsonl
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
. "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/config/paths.env"
D=${SIGLIP_DIR:-$OVDET_MODELS/x86_sm120/siglip2_b32}/study
mkdir -p "$D"
NAME=$1; EXP=$2; BLD=$3
[ -f "$D/$NAME.onnx" ] || python "$HERE/export_siglip2.py" --out "$D/$NAME.onnx" $EXP 2>&1 | grep -E "exported|cosine"
python "$HERE/build_engine.py" "$D/$NAME.onnx" "$D/$NAME.plan" $BLD 2>&1 | grep -E "pinned|wrote|failed|Error"
cd "$HERE"
python eval_variants.py --out data/clip_bench/study.jsonl "trt:$D/$NAME.plan" 2>&1 | grep variant | cut -c1-400
python -c "
import json, sys
from trt_run import Engine
e = Engine('$D/$NAME.plan'); r = {}
for n in sorted(set(e.bucket(k) for k in (1, 8))):
    r['b%d' % n] = e.bench(n)[0]
r['name'] = '$NAME'; r['bench'] = True
print(json.dumps(r)); open('$RA_ROOT/data/clip_bench/study.jsonl', 'a').write(json.dumps(r) + '\n')"
