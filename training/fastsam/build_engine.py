#!/usr/bin/env python3
"""ONNX -> TensorRT FP16 engine (.plan) + <plan>.names.txt ('object'). Same builder settings as the ovdet engines.

  python build_engine.py model.onnx --out engines/ [--workspace-gb 2]
TensorRT plans are GPU/TensorRT-version specific: build on the target machine.
"""
import argparse
import time
from pathlib import Path

import tensorrt as trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    ap.add_argument('--out', default='.')
    ap.add_argument('--workspace-gb', type=float, default=2)
    a = ap.parse_args()
    src = Path(a.onnx)
    dst = Path(a.out) / (src.stem + '.plan')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(str(src)):
        raise SystemExit('ONNX parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    t0 = time.time()
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise SystemExit('engine build failed')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(blob))
    Path(str(dst) + '.names.txt').write_text('object\n')
    print(f'{src} -> {dst} ({dst.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
