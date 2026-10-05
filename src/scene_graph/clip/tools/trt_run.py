"""Minimal TensorRT 10 runner for the SigLIP 2 mask engine from Python (offline checks / study only — the robot runs C++).

Works from the torch Python env (torch CPU) by borrowing the TensorRT bindings of the TensorRT Python env (same Python 3.11) and the CUDA 12.8
runtime through ctypes — no pycuda / cuda-python needed.

    from trt_run import Engine
    e = Engine(plan); emb = e.run(images_f32_nchw, wpatch_f32)        # numpy in, numpy out (N x 768)
    ms = e.bench(n, iters=200, graph=True)                             # GPU time per call (CUDA events), ms
"""
import ctypes
import glob
import os
import sys

import numpy as np

_site = glob.glob(os.path.expanduser("the TensorRT Python env/lib/python3*/site-packages"))
if _site and _site[0] not in sys.path:
    sys.path.append(_site[0])
import tensorrt as trt  # noqa: E402

_cu = ctypes.CDLL("/usr/local/cuda-12.8/lib64/libcudart.so.12")
_vp = ctypes.c_void_p
for fn, args in (("cudaMalloc", [ctypes.POINTER(_vp), ctypes.c_size_t]), ("cudaFree", [_vp]),
                 ("cudaMemcpy", [_vp, _vp, ctypes.c_size_t, ctypes.c_int]),
                 ("cudaMemcpyAsync", [_vp, _vp, ctypes.c_size_t, ctypes.c_int, _vp]),
                 ("cudaStreamCreate", [ctypes.POINTER(_vp)]), ("cudaStreamSynchronize", [_vp]),
                 ("cudaEventCreate", [ctypes.POINTER(_vp)]), ("cudaEventRecord", [_vp, _vp]),
                 ("cudaEventSynchronize", [_vp]), ("cudaEventElapsedTime", [ctypes.POINTER(ctypes.c_float), _vp, _vp]),
                 ("cudaStreamBeginCapture", [_vp, ctypes.c_int]), ("cudaStreamEndCapture", [_vp, ctypes.POINTER(_vp)]),
                 ("cudaGraphInstantiate", [ctypes.POINTER(_vp), _vp, ctypes.c_ulonglong]), ("cudaGraphLaunch", [_vp, _vp]),
                 ("cudaMemGetInfo", [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)])):
    getattr(_cu, fn).argtypes = args


def _ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what}: cuda error {r}")


def gpu_free_mb():
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    _ck(_cu.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)), "meminfo")
    return f.value >> 20, t.value >> 20


_NP = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32}


class Engine:
    def __init__(self, plan, max_batch=8):
        self.lg = trt.Logger(trt.Logger.ERROR)
        self.rt = trt.Runtime(self.lg)
        self.eng = self.rt.deserialize_cuda_engine(open(plan, "rb").read())
        self.ctx = self.eng.create_execution_context()
        self.nprof = self.eng.num_optimization_profiles
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.dt = {n: _NP[self.eng.get_tensor_dtype(n)] for n in self.names}
        self.S = self.eng.get_tensor_shape("images")[-1]
        self.G = self.eng.get_tensor_shape("wpatch")[-1]
        self.stream = _vp()
        _ck(_cu.cudaStreamCreate(ctypes.byref(self.stream)), "stream")
        self.mb = max_batch
        shp = {"images": (3, self.S, self.S), "wpatch": (self.G,), "emb": (768,)}
        self.buf = {}
        for n in self.names:
            p = _vp()
            _ck(_cu.cudaMalloc(ctypes.byref(p), int(np.prod(shp[n])) * max_batch * np.dtype(self.dt[n]).itemsize), "malloc")
            self.buf[n] = (p, shp[n])
        self.prof = -1
        # profile buckets: (min, max) batch per profile
        self.ranges = []
        for k in range(self.nprof):
            mn, _, mx = self.eng.get_tensor_profile_shape("images", k)
            self.ranges.append((mn[0], mx[0]))

    def _set(self, n):
        k = next(i for i, (lo, hi) in enumerate(self.ranges) if lo <= n <= hi)
        if k != self.prof:
            self.ctx.set_optimization_profile_async(k, self.stream.value)
            self.prof = k
        self.ctx.set_input_shape("images", (n, 3, self.S, self.S))
        self.ctx.set_input_shape("wpatch", (n, self.G))
        for name, (p, _) in self.buf.items():
            self.ctx.set_tensor_address(name, p.value)

    def bucket(self, n):
        """smallest batch the engine accepts that is >= n (static bucket engines pad)."""
        return min(hi if lo == hi else n for lo, hi in self.ranges if hi >= n and (lo == hi or lo <= n))

    def run(self, images, wpatch):
        out = []
        for i in range(0, len(images), self.mb):
            x, w = images[i:i + self.mb], wpatch[i:i + self.mb]
            n = len(x)
            nb = self.bucket(n)
            if nb > n:   # pad to the static bucket
                x = np.concatenate([x, np.repeat(x[-1:], nb - n, 0)])
                w = np.concatenate([w, np.repeat(w[-1:], nb - n, 0)])
            self._set(nb)
            for name, a in (("images", x), ("wpatch", w)):
                a = np.ascontiguousarray(a.astype(self.dt[name]))
                _ck(_cu.cudaMemcpy(self.buf[name][0], a.ctypes.data, a.nbytes, 1), "h2d")
            assert self.ctx.execute_async_v3(self.stream.value)
            _ck(_cu.cudaStreamSynchronize(self.stream), "sync")
            o = np.empty((nb, 768), self.dt["emb"])
            _ck(_cu.cudaMemcpy(o.ctypes.data, self.buf["emb"][0], o.nbytes, 2), "d2h")
            out.append(o[:n].astype(np.float32))
        return np.concatenate(out)

    def bench(self, n, iters=200, graph=True, warm=20):
        """median-free mean GPU time per inference over `iters` launches (CUDA events), ms."""
        self._set(n)
        for _ in range(warm):
            self.ctx.execute_async_v3(self.stream.value)
        _cu.cudaStreamSynchronize(self.stream)
        g = _vp()
        if graph:
            gr = _vp()
            _ck(_cu.cudaStreamBeginCapture(self.stream, 0), "capture")
            self.ctx.execute_async_v3(self.stream.value)
            _ck(_cu.cudaStreamEndCapture(self.stream, ctypes.byref(gr)), "endcapture")
            _ck(_cu.cudaGraphInstantiate(ctypes.byref(g), gr, 0), "instantiate")
        e0, e1 = _vp(), _vp()
        _cu.cudaEventCreate(ctypes.byref(e0)); _cu.cudaEventCreate(ctypes.byref(e1))
        ts = []
        for _ in range(5):   # 5 rounds, keep the best (the GPU is shared with other jobs)
            _cu.cudaEventRecord(e0, self.stream)
            for _ in range(iters):
                if graph:
                    _cu.cudaGraphLaunch(g, self.stream)
                else:
                    self.ctx.execute_async_v3(self.stream.value)
            _cu.cudaEventRecord(e1, self.stream)
            _cu.cudaEventSynchronize(e1)
            ms = ctypes.c_float()
            _cu.cudaEventElapsedTime(ctypes.byref(ms), e0, e1)
            ts.append(ms.value / iters)
        return min(ts), float(np.median(ts))
