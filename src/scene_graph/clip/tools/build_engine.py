"""TensorRT engine for the SigLIP 2 mask-pooled image tower (export_siglip2.py ONNX) or the text tower
(export_siglip2_text.py ONNX, input tok_emb N x 64 x 768 — same FP32 pins).

FP16 with the numerically sensitive layers pinned to FP32 (docs/clip_candidates.md 2.4: SigLIP activations overflow FP16
inside the opset-13 decomposed LayerNorm -> cosine 0.64-0.74 without this). Pinned: every layer whose ONNX name contains
'norm' (LayerNorm ReduceMean/Sub/Pow/Sqrt/Div/Mul/Add and the final L2 normalisation), optionally softmax.

    ~/ovdet_venv/bin/python build_engine.py ONNX PLAN [--profiles 1-8 | 1,2,4,8] [--pin norm,softmax] [--int8 CALIB.npz]
                                            [--half-input]

--profiles 1-8     one dynamic profile min 1 / opt 8 / max 8 (default)
--profiles 1,2,4,8 one static profile per batch bucket (runtime picks the smallest bucket >= n, pads)
--half-input       'images' tensor FP16 (the runtime crop kernel writes FP16, halves the staging ring); wpatch stays FP32
--int8 CALIB.npz   implicit INT8 PTQ (entropy calibration over images/wpatch arrays in the npz); pinned layers stay FP32
                   and --keep-fp16 name substrings stay FP16. INT8 is a PC / memory option only: Maxwell (Nano) has no fast INT8.

TensorRT 8.2 (JetPack 4.6) differences handled here: set_memory_pool_limit vs max_workspace_size, network creation flag
EXPLICIT_BATCH, OBEY_PRECISION_CONSTRAINTS vs STRICT_TYPES, build_serialized_network (8.0+).
"""
import argparse
import re
import sys

import numpy as np
import tensorrt as trt

TRT_MAJOR = int(trt.__version__.split(".")[0])


class Calib(trt.IInt8EntropyCalibrator2):
    def __init__(self, npz, batch, cache):
        super().__init__()
        import ctypes
        self.z = np.load(npz)
        self.im, self.w = self.z["images"].astype(np.float32), self.z["wpatch"].astype(np.float32)
        self.b, self.i, self.cache = batch, 0, cache
        self.cudart = ctypes.CDLL("libcudart.so")
        self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.p = {}
        for k, a in (("images", self.im), ("wpatch", self.w)):
            ptr = ctypes.c_void_p()
            self.cudart.cudaMalloc(ctypes.byref(ptr), a[:batch].nbytes)
            self.p[k] = ptr

    def get_batch_size(self):
        return self.b

    def get_batch(self, names):
        if self.i + self.b > len(self.im):
            return None
        for k, a in (("images", self.im), ("wpatch", self.w)):
            c = np.ascontiguousarray(a[self.i:self.i + self.b])
            self.cudart.cudaMemcpy(self.p[k], c.ctypes.data, c.nbytes, 1)
        self.i += self.b
        return [int(self.p[n].value) for n in names]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, c):
        open(self.cache, "wb").write(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("plan")
    ap.add_argument("--profiles", default="1-8")
    ap.add_argument("--pin", default="norm", help="comma list of regexes on lower-case layer names pinned FP32 (norm, attn_pool, blocks.11 ...), softmax = every softmax, all-fp32, none")
    ap.add_argument("--half-input", action="store_true")
    ap.add_argument("--int8", default="")
    ap.add_argument("--keep-fp16", default="", help="INT8: comma list of layer-name substrings kept FP16")
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--layer-info", default="", help="write engine inspector JSON here")
    a = ap.parse_args()

    lg = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(lg)
    flags = 0 if TRT_MAJOR >= 10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    net = b.create_network(flags)
    p = trt.OnnxParser(net, lg)
    if not p.parse_from_file(a.onnx):
        sys.exit("\n".join(str(p.get_error(i)) for i in range(p.num_errors)))
    cfg = b.create_builder_config()
    if TRT_MAJOR >= 10 or hasattr(cfg, "set_memory_pool_limit"):
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_mb << 20)
    else:
        cfg.max_workspace_size = a.workspace_mb << 20
    pins = set(x for x in a.pin.split(",") if x and x != "none")
    if "all-fp32" not in pins:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    else:
        cfg.set_flag(trt.BuilderFlag.STRICT_TYPES)
    if a.int8:
        cfg.set_flag(trt.BuilderFlag.INT8)

    if a.half_input:
        for i in range(net.num_inputs):
            t = net.get_input(i)
            if t.name == "images":
                t.dtype = trt.float16
    # every input has a dynamic batch axis 0; the rest is static (image tower: images N x 3 x S x S + wpatch N x G²,
    # text tower export_siglip2_text.py: tok_emb N x 64 x 768)
    shapes = {net.get_input(i).name: tuple(net.get_input(i).shape)[1:] for i in range(net.num_inputs)}
    if "-" in a.profiles:
        lo, hi = (int(x) for x in a.profiles.split("-"))
        buckets = [(lo, hi, hi)]
    else:
        buckets = [(n, n, n) for n in (int(x) for x in a.profiles.split(","))]
    for lo, opt, hi in buckets:
        pr = b.create_optimization_profile()
        for nm, rest in shapes.items():
            pr.set_shape(nm, (lo,) + rest, (opt,) + rest, (hi,) + rest)
        cfg.add_optimization_profile(pr)
        if a.int8 and hasattr(cfg, "set_calibration_profile") and (lo, opt, hi) == buckets[-1]:
            cfg.set_calibration_profile(pr)

    skip = tuple(getattr(trt.LayerType, k) for k in ("CONSTANT", "SHAPE", "SHUFFLE", "CAST", "GATHER", "SLICE") if hasattr(trt.LayerType, k))
    keep16 = [x for x in a.keep_fp16.split(",") if x]
    n32 = n16 = 0
    for i in range(net.num_layers):
        L = net.get_layer(i)
        nm = L.name.lower()
        if L.type in skip or any(L.get_output(j).dtype not in (trt.float32, trt.float16) for j in range(L.num_outputs)):
            continue   # shape / index arithmetic
        pin = any(re.search(k, nm) for k in pins if k not in ("softmax", "all-fp32")) or ("softmax" in pins and L.type == trt.LayerType.SOFTMAX)
        if pin:
            L.precision = trt.float32
            for j in range(L.num_outputs):
                L.set_output_type(j, trt.float32)
            n32 += 1
        elif a.int8 and any(k in nm for k in keep16):
            L.precision = trt.float16
            n16 += 1
    if a.int8:
        cfg.int8_calibrator = Calib(a.int8, buckets[-1][2], a.plan + ".calib")
    print(f"TRT {trt.__version__}: {net.num_layers} layers, pinned FP32 {n32}, kept FP16 {n16}, profiles {buckets}, inputs {shapes}")
    blob = b.build_serialized_network(net, cfg)
    if blob is None:
        sys.exit("build failed")
    blob = bytes(blob)
    open(a.plan, "wb").write(blob)
    print("wrote", a.plan, len(blob) >> 20, "MB")
    if a.layer_info:
        rt = trt.Runtime(lg)
        eng = rt.deserialize_cuda_engine(blob)
        ins = eng.create_engine_inspector()
        open(a.layer_info, "w").write(ins.get_engine_information(trt.LayerInformationFormat.JSON))


if __name__ == "__main__":
    main()
