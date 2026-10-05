#!/bin/bash
# Build libovdet.so + ovdet_smoke (Linux / WSL: CUDA 12.8, TensorRT 10). Out-of-tree in build/ovdet.
set -e
SRC=$(cd "$(dirname "$0")/.." && pwd)
B=${OVDET_BUILD:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/build/ovdet}
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$B" -j"$(nproc)" 2>&1 | grep -vE "^\[ *[0-9]+%\]" || true
ls -la "$B"/libovdet.so "$B"/ovdet_smoke
