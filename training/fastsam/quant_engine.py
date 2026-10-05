#!/usr/bin/env python3
"""TensorRT engines at several precisions for one ObjectSAM ONNX (416, output0 1x37xA + output1 1x32xhxw).

  # calibration tensor (letterboxed 416 RGB /255, same as the ovdet runtime), sim + real mix:
  python quant_engine.py calib --out calib512.npy --n 512 [--list train_list.txt]
  # engines:
  python quant_engine.py build model.onnx out.plan --mode fp16
  python quant_engine.py build model.onnx out.plan --mode fp16 --pin 'model.23/proto'          # layers pinned FP32
  python quant_engine.py build model.onnx out.plan --mode int8 --calib calib512.npy --algo entropy|minmax|percentile
  python quant_engine.py build model.onnx out.plan --mode int8 --calib calib512.npy --keep-fp16 'model.23/,model.0/'
  python quant_engine.py build model.onnx out.plan --mode int8 --cache out.plan.calib            # cache only (no images)
  # explicit INT8 (Q/DQ ONNX via onnxruntime; the path that works on TRT 10.16 for this model):
  python quant_engine.py qdq model.onnx model_qdq.onnx --calib calib512.npy --algo entropy [--exclude 'model.23/proto']
  python quant_engine.py build model_qdq.onnx out.plan --mode int8

INT8 here is TensorRT implicit post-training quantization (IInt8*Calibrator). The calibration cache (<plan>.calib, text,
one scale per tensor) is what a Jetson needs: copy the ONNX + cache, build there with --cache (no images on the robot).
Targets: x86 / Orin (sm_87, JetPack 5-6, TRT 8.5-10.x): FP16 or INT8. Jetson Nano (sm_53, JetPack 4.6, TRT 8.2): FP16
only — Maxwell has no fast INT8 path, an INT8 build there falls back to FP16/FP32 kernels.
Handles TRT 8.2 (explicit-batch flag, max_workspace_size, STRICT_TYPES) and TRT 10 (memory pools, OBEY_PRECISION_CONSTRAINTS).
"""
import argparse
import os
import re
import sys

import numpy as np

S = 416


def letterbox(rgb, s=S):
    """Ultralytics LetterBox as ovdet k_letterbox: cv2 INTER_LINEAR resize, centred, 114 pad -> 3 x s x s float /255."""
    import cv2
    h, w = rgb.shape[:2]
    r = min(s / h, s / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    top, left = int(round((s - nh) / 2 - 0.1)), int(round((s - nw) / 2 - 0.1))
    out = np.full((s, s, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out.transpose(2, 0, 1).astype(np.float32) / 255.


def cmd_calib(a):
    from PIL import Image
    files = [l for l in open(a.list).read().split() if l]
    files = sorted(set(files))
    sim = [f for f in files if not os.path.basename(f).startswith(('coco_', 'ade_'))]
    real = [f for f in files if os.path.basename(f).startswith(('coco_', 'ade_'))]
    rng = np.random.default_rng(0)
    k = int(round(a.n * a.sim_frac))
    pick = list(rng.choice(sim, k, replace=False)) + list(rng.choice(real, a.n - k, replace=False))
    x = np.stack([letterbox(np.asarray(Image.open(f).convert('RGB'))) for f in pick]).astype(np.float16)
    np.save(a.out, x)
    open(a.out + '.txt', 'w').write('\n'.join(os.path.basename(f) for f in pick) + '\n')
    print(f'calib {x.shape} (sim {k}, real {a.n - k}) -> {a.out}')


def cmd_qdq(a):
    """Explicit INT8 (Q/DQ ONNX, onnxruntime static quantization): symmetric INT8 activations, per-channel symmetric
    weights, calibration entropy | minmax | percentile over the calibration tensor. --exclude: regexes on ONNX node names
    left unquantized (they run FP16 in TensorRT = mixed precision). TensorRT then builds it with INT8 + FP16 flags.
    TensorRT 10.16 cannot build this model with implicit (calibrator) INT8 — Myelin fails on the fused graph — so explicit
    Q/DQ is the path here; Q/DQ ONNX also builds on TRT 8.2+ (Jetson)."""
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    x = np.load(a.calib)[:a.n]

    class R(CalibrationDataReader):   # strided (CalibStridedMinMax): ORT keeps every activation of a chunk in RAM
        def __init__(self):
            self.i, self.end = 0, len(x)

        def __len__(self):
            return len(x)

        def set_range(self, start_index, end_index):
            self.i, self.end = start_index, end_index

        def get_next(self):
            if self.i >= self.end:
                return None
            self.i += 1
            return {'images': x[self.i - 1:self.i].astype(np.float32)}

    pre = a.out + '.pre.onnx'
    quant_pre_process(a.onnx, pre, skip_optimization=False)
    m = onnx.load(pre)
    ex = [n.name for n in m.graph.node if any(re.search(k, n.name) for k in a.exclude.split(',') if k)]
    meth = {'entropy': CalibrationMethod.Entropy, 'minmax': CalibrationMethod.MinMax,
            'percentile': CalibrationMethod.Percentile}[a.algo]
    quantize_static(pre, a.out, R(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True, calibrate_method=meth, nodes_to_exclude=ex,
                    op_types_to_quantize=a.ops.split(','),
                    extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True, 'CalibStridedMinMax': 16,
                                   'CalibPercentile': a.percentile, 'QuantizeBias': False, 'DedicatedQDQPair': False})
    os.remove(pre)
    print(f'qdq {a.algo} n={len(x)} excluded {len(ex)} nodes -> {a.out}')


def cmd_build(a):
    import tensorrt as trt
    major = int(trt.__version__.split('.')[0])
    lg = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(lg)
    net = b.create_network(0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    p = trt.OnnxParser(net, lg)
    if not p.parse_from_file(a.onnx):
        sys.exit('\n'.join(str(p.get_error(i)) for i in range(p.num_errors)))
    cfg = b.create_builder_config()
    if hasattr(cfg, 'set_memory_pool_limit'):
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    else:
        cfg.max_workspace_size = int(a.workspace_gb * (1 << 30))
    if a.opt_level is not None and hasattr(cfg, 'builder_optimization_level'):
        cfg.builder_optimization_level = a.opt_level
    if a.mode in ('fp16', 'int8'):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if a.mode == 'int8':
        cfg.set_flag(trt.BuilderFlag.INT8)   # with a Q/DQ ONNX (qdq) this is explicit quantization: no calibrator
    pins = [x for x in a.pin.split(',') if x]
    keep16 = [x for x in a.keep_fp16.split(',') if x]
    if pins or keep16:
        cfg.set_flag(getattr(trt.BuilderFlag, 'OBEY_PRECISION_CONSTRAINTS', None) or trt.BuilderFlag.STRICT_TYPES)
    skip = tuple(getattr(trt.LayerType, k) for k in ('CONSTANT', 'SHAPE', 'SHUFFLE', 'CAST', 'GATHER', 'SLICE', 'CONCATENATION')
                 if hasattr(trt.LayerType, k))
    n32 = n16 = 0
    for i in range(net.num_layers):
        L = net.get_layer(i)
        if L.type in skip or any(L.get_output(j).dtype not in (trt.float32, trt.float16) for j in range(L.num_outputs)):
            continue
        if any(re.search(k, L.name) for k in pins):
            L.precision = trt.float32
            for j in range(L.num_outputs):
                L.set_output_type(j, trt.float32)
            n32 += 1
        elif a.mode == 'int8' and any(re.search(k, L.name) for k in keep16):
            L.precision = trt.float16
            n16 += 1
    if a.mode == 'int8' and (a.calib or a.cache):
        cache = a.cache or a.plan + '.calib'
        base = {'entropy': trt.IInt8EntropyCalibrator2, 'minmax': trt.IInt8MinMaxCalibrator,
                'percentile': trt.IInt8LegacyCalibrator}[a.algo]

        class Calib(base):
            def __init__(self):
                base.__init__(self)
                self.x = np.load(a.calib) if a.calib else None
                self.i = 0
                self.ptr = None
                if self.x is not None:
                    import ctypes
                    self.rt = ctypes.CDLL('libcudart.so')
                    self.rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
                    self.rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                    self.ptr = ctypes.c_void_p()
                    self.rt.cudaMalloc(ctypes.byref(self.ptr), a.batch * 3 * S * S * 4)

            def get_batch_size(self):
                return a.batch

            def get_batch(self, names):
                if self.x is None or self.i + a.batch > len(self.x):
                    return None
                c = np.ascontiguousarray(self.x[self.i:self.i + a.batch].astype(np.float32))
                self.rt.cudaMemcpy(self.ptr, c.ctypes.data, c.nbytes, 1)
                self.i += a.batch
                return [int(self.ptr.value)]

            def read_calibration_cache(self):
                if (a.cache or not a.calib) and os.path.exists(cache):
                    return open(cache, 'rb').read()
                return None

            def write_calibration_cache(self, c):
                open(cache, 'wb').write(c)

            # legacy (percentile) calibrator knobs
            def get_quantile(self):
                return a.quantile

            def get_regression_cutoff(self):
                return 1.0

            def read_histogram_cache(self, length):
                return None

            def write_histogram_cache(self, ptr, length):
                return None

        if a.batch != 1:
            sys.exit('calibration uses the static batch-1 ONNX: --batch 1')
        cfg.int8_calibrator = Calib()
    print(f'TRT {trt.__version__} mode {a.mode}{" " + a.algo if a.mode == "int8" else ""}: pinned FP32 {n32}, kept FP16 {n16}')
    blob = b.build_serialized_network(net, cfg)
    if blob is None:
        sys.exit('build failed')
    blob = bytes(blob)
    open(a.plan, 'wb').write(blob)
    open(a.plan + '.names.txt', 'w').write('object\n')
    print(f'wrote {a.plan} {len(blob) / 2**20:.1f} MB')
    if a.layer_info:
        eng = trt.Runtime(lg).deserialize_cuda_engine(blob)
        open(a.layer_info, 'w').write(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('calib')
    c.add_argument('--out', required=True)
    c.add_argument('--n', type=int, default=512)
    c.add_argument('--sim-frac', type=float, default=0.5)
    c.add_argument('--list', required=True, help='training list (one image path per line)')
    b = sp.add_parser('build')
    b.add_argument('onnx')
    b.add_argument('plan')
    b.add_argument('--mode', choices=['fp32', 'fp16', 'int8'], default='fp16')
    b.add_argument('--calib', help='calibration .npy (N x 3 x 416 x 416)')
    b.add_argument('--cache', help='existing calibration cache (build without images)')
    b.add_argument('--algo', choices=['entropy', 'minmax', 'percentile'], default='entropy')
    b.add_argument('--quantile', type=float, default=0.9999)
    b.add_argument('--batch', type=int, default=1)
    b.add_argument('--pin', default='', help='regexes on layer names pinned FP32')
    b.add_argument('--keep-fp16', default='', help='INT8: regexes on layer names kept FP16')
    b.add_argument('--workspace-gb', type=float, default=2)
    b.add_argument('--opt-level', type=int)
    b.add_argument('--layer-info', default='')
    q = sp.add_parser('qdq')
    q.add_argument('onnx')
    q.add_argument('out')
    q.add_argument('--calib', required=True)
    q.add_argument('--n', type=int, default=512)
    q.add_argument('--algo', choices=['entropy', 'minmax', 'percentile'], default='entropy')
    q.add_argument('--percentile', type=float, default=99.99)
    q.add_argument('--exclude', default='', help='regexes on ONNX node names kept unquantized (FP16 in TRT)')
    q.add_argument('--ops', default='Conv', help='op types to quantize')
    a = ap.parse_args()
    {'calib': cmd_calib, 'build': cmd_build, 'qdq': cmd_qdq}[a.cmd](a)


if __name__ == '__main__':
    main()
