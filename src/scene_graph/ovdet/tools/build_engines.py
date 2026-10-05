#!/usr/bin/env python3
"""YOLOE / YOLO-seg ONNX -> TensorRT FP16 plan, <onnx>.names.txt (or --names) copied next to it as <plan>.names.txt.
YOLO11/YOLO26-seg (closed vocabulary, COCO-80): export with end2end=False (classic 1 x 116 x A head; ovdet does the NMS),
then --names config/coco80.txt:
  python -c "from ultralytics import YOLO; YOLO('yolo26s-seg.pt').export(format='onnx', imgsz=416, opset=13, end2end=False)"
  python build_engines.py yolo26s-seg-416.onnx --names ../config/coco80.txt
GPU: run under the shared GPU lock (src/sim/engine/scripts/gpu_lock.sh).

  python build_engines.py models/ovdet/onnx/yoloe-11l-all.onnx ... --out models/ovdet/x86_sm120
"""
import argparse
import shutil
import time
from pathlib import Path

import tensorrt as trt


def build(onnx_path, out_path, workspace_gb=4):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise SystemExit('ONNX parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    for t in [network.get_input(i) for i in range(network.num_inputs)] +              [network.get_output(i) for i in range(network.num_outputs)]:
        print(f'  {t.name} {tuple(t.shape)}')
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit(f'engine build failed: {onnx_path}')
    out_path.write_bytes(bytes(blob))
    print(f'  -> {out_path} ({out_path.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx', nargs='+')
    ap.add_argument('--out', default=str(Path.home() / 'ovdet_models' / 'x86_sm120'))
    ap.add_argument('--names', help='class names file for every plan (closed-vocabulary engines: config/coco80.txt)')
    ap.add_argument('--workspace-gb', type=float, default=4)
    a = ap.parse_args()
    for o in map(Path, a.onnx):
        dst = Path(a.out) / (o.stem + '.plan')
        print(f'{o} -> {dst}', flush=True)
        build(o, dst, a.workspace_gb)
        names = Path(a.names) if a.names else Path(str(o) + '.names.txt')
        if names.exists():
            shutil.copy(names, str(dst) + '.names.txt')


if __name__ == '__main__':
    main()
