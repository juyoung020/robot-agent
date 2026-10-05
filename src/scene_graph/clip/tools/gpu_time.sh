#!/bin/bash
# GPU 실행 시간(커널 합 / 추론 수) — GPU 를 다른 일과 나눠 쓸 때 CUDA event 시간은 시간 조각 기다림까지 들어가므로
# nsys 로 커널마다 실제 실행 시간을 더한다.   tools/gpu_time.sh PLAN BATCH  → "ms_per_inference top_kernel"
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
N=60
/usr/local/cuda-12.8/bin/nsys profile -o "$OUT/p" -f true -t cuda python -c "
import sys; sys.path.insert(0, '$HERE')
from trt_run import Engine
e = Engine('$1'); b = e.bucket($2); e.bench(b, iters=$N, graph=False, warm=0)" > /dev/null 2>&1
/usr/local/cuda-12.8/bin/nsys stats -r cuda_gpu_kern_sum -f csv "$OUT/p.nsys-rep" 2>/dev/null | python3 -c "
import sys, csv
rows = [r for r in csv.reader(sys.stdin) if len(r) > 8 and r[0][:1].isdigit()]
tot = sum(int(r[1]) for r in rows); n = 5 * $N + 1   # bench: 5 rounds x N + 1 (profile switch / first)
top = max(rows, key=lambda r: int(r[1]))
print('%.4f ms/inference (b=$2), top kernel %.1f%% %s' % (tot / n / 1e6, float(top[0]), top[8][:60]))"
rm -rf "$OUT"
