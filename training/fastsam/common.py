"""공용: 경로, ovdet(TensorRT) 검출기 ctypes, 구조물 범주, 마스크 도우미.

검출은 실행 때와 똑같이 ovdet C API(libovdet.so + .plan, conf 0.25·NMS 0.7·면적·마스크 중복 제거 기본값)로 한다.
"""
import ctypes as C
import os

import numpy as np

HOME = os.path.expanduser('~')
DATA = os.environ.get('FASTSAM_DATA', f'{HOME}/datasets/fastsam_obj')
LIB = os.environ.get('OVDET_LIB', f'{HOME}/ovdet_build/libovdet.so')     # optional TensorRT path (ovdet C API)
BASE_PLAN = os.environ.get('BASE_PLAN', f'{HOME}/ovdet_models/x86_sm120/FastSAM-s-416.plan')   # original engine (or .onnx)
B1K = os.environ.get('B1K_ROOT', f'{HOME}/BEHAVIOR-1K')                    # BEHAVIOR-1K checkout (datasets/ inside)

# 배경(구조물) = 벽·천장·바닥만. 문·창·계단은 물체로 남긴다(지도의 방·문 토큰, 계단 위험).
SIM_STRUCT = {'walls', 'floors', 'ceilings', 'roof', 'baseboard'}
SIM_DWS = {'door', 'sliding_door', 'garage_door', 'elevator_door', 'fixed_window', 'openable_window', 'stairs',
           'railing'}


class Det(C.Structure):
    _fields_ = [('stamp', C.c_double), ('cam', C.c_int), ('img_w', C.c_int), ('img_h', C.c_int), ('n', C.c_int),
                ('cls', C.POINTER(C.c_int32)), ('score', C.POINTER(C.c_float)), ('box', C.POINTER(C.c_float)),
                ('mask_w', C.c_int), ('mask_h', C.c_int), ('mask_sx', C.c_float), ('mask_sy', C.c_float),
                ('mask_ox', C.c_float), ('mask_oy', C.c_float), ('mask_bits', C.POINTER(C.c_uint32))]


class Cfg(C.Structure):
    _fields_ = [('seg_engine', C.c_char_p), ('names', C.c_char_p), ('device', C.c_int32), ('conf_th', C.c_float),
                ('nms_iou', C.c_float), ('mask_iou', C.c_float), ('area_min', C.c_int32), ('small_area', C.c_int32),
                ('small_conf', C.c_float), ('max_det', C.c_int32), ('class_agnostic', C.c_int32)]


class Img(C.Structure):
    _fields_ = [('stamp', C.c_double), ('cam', C.c_int32), ('data', C.c_void_p), ('h', C.c_int32), ('w', C.c_int32),
                ('row_stride', C.c_int64), ('pix_stride', C.c_int32), ('bgr', C.c_int32), ('on_device', C.c_int32)]


class Timing(C.Structure):
    _fields_ = [('upload_ms', C.c_float), ('net_ms', C.c_float), ('post_ms', C.c_float), ('out_ms', C.c_float),
                ('total_ms', C.c_float), ('candidates', C.c_int32)]


_L = None


def lib():
    global _L
    if _L is None:
        L = C.CDLL(LIB)
        L.ovd_default_config.argtypes = [C.POINTER(Cfg)]
        L.ovd_create.argtypes = [C.POINTER(Cfg), C.c_char_p, C.c_size_t]
        L.ovd_create.restype = C.c_void_p
        L.ovd_destroy.argtypes = [C.c_void_p]
        L.ovd_device_bytes.argtypes = [C.c_void_p]
        L.ovd_device_bytes.restype = C.c_int64
        L.ovd_set_prompt.argtypes = [C.c_void_p, C.POINTER(C.c_char_p), C.c_int32, C.c_char_p, C.c_size_t]
        L.ovd_detect.argtypes = [C.c_void_p, C.POINTER(Img), C.POINTER(Timing)]
        L.ovd_detect.restype = C.POINTER(Det)
        L.ovd_last_error.argtypes = [C.c_void_p]
        L.ovd_last_error.restype = C.c_char_p
        _L = L
    return _L


class Detector:
    """FastSAM 계열 엔진 하나(클래스 'object' 하나). detect(rgb HxWx3 u8) → dict(score, box, masks(full-res bool))."""

    def __init__(self, plan, conf_th=0.25):
        L = lib()
        cfg = Cfg()
        L.ovd_default_config(C.byref(cfg))
        self._keep = [plan.encode(), (plan + '.names.txt').encode()]
        cfg.seg_engine, cfg.names = self._keep
        cfg.conf_th = conf_th
        err = C.create_string_buffer(512)
        self.h = L.ovd_create(C.byref(cfg), err, 512)
        if not self.h:
            raise RuntimeError(err.value.decode())
        arr = (C.c_char_p * 1)(b'object')
        L.ovd_set_prompt(self.h, arr, 1, None, 0)
        self.mem_mb = L.ovd_device_bytes(self.h) / 2**20
        self.ms = []

    def detect(self, rgb, full=True):
        L = lib()
        rgb = np.ascontiguousarray(rgb[..., :3], np.uint8)
        im = Img(0.0, 0, rgb.ctypes.data, rgb.shape[0], rgb.shape[1], rgb.strides[0], 3, 0, 0)
        t = Timing()
        r = L.ovd_detect(self.h, C.byref(im), C.byref(t))
        if not r:
            raise RuntimeError(L.ovd_last_error(self.h).decode())
        d = r.contents
        self.ms.append(t.total_ms)
        n = d.n
        if n == 0:
            H, W = rgb.shape[:2]
            return dict(score=np.zeros(0, np.float32), box=np.zeros((0, 4), np.float32),
                        grid=np.zeros((0, d.mask_h, d.mask_w), bool), masks=np.zeros((0, H, W), bool))
        words = (d.mask_w * d.mask_h + 31) // 32
        bits = np.ctypeslib.as_array(d.mask_bits, (max(n, 1) * words,))[:n * words].reshape(n, words).copy()
        g = np.unpackbits(bits.view(np.uint8), axis=1, bitorder='little')[:, :d.mask_w * d.mask_h]
        g = g.reshape(n, d.mask_h, d.mask_w).astype(bool)
        score = np.ctypeslib.as_array(d.score, (max(n, 1),))[:n].copy()
        box = np.ctypeslib.as_array(d.box, (max(n, 1) * 4,))[:n * 4].reshape(n, 4).copy()
        out = dict(score=score, box=box, grid=g)
        if full:
            H, W = rgb.shape[:2]
            fi = np.clip(((np.arange(W) + 0.5 - d.mask_ox) / d.mask_sx).astype(int), 0, d.mask_w - 1)
            fj = np.clip(((np.arange(H) + 0.5 - d.mask_oy) / d.mask_sy).astype(int), 0, d.mask_h - 1)
            out['masks'] = g[:, fj][:, :, fi] if n else np.zeros((0, H, W), bool)
        return out

    def close(self):
        if self.h:
            lib().ovd_destroy(self.h)
            self.h = None


def make_detector(path, conf_th=0.25):
    """.onnx -> ONNX Runtime reference (ref_detect.py, same post-processing); .plan -> ovdet TensorRT (libovdet)."""
    path = os.path.expanduser(path)
    if path.endswith('.onnx'):
        from ref_detect import OrtDetector
        return OrtDetector(path, conf_th)
    return Detector(path, conf_th)


def mask_to_polys(m, eps=1.0, min_area=6):
    """bool 마스크 → 정규화 다각형 하나(ultralytics YOLO 분할 글 형식). 조각이 여럿이면 merge_multi_segment 로 잇는다."""
    import cv2
    from ultralytics.data.converter import merge_multi_segment
    H, W = m.shape
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [cv2.approxPolyDP(c, eps, True).reshape(-1, 2) for c in cs if cv2.contourArea(c) >= min_area]
    cs = [c for c in cs if len(c) >= 3]
    if not cs:
        return None
    if len(cs) == 1:
        s = cs[0]
    else:
        cs = sorted(cs, key=lambda c: -cv2.contourArea(c))[:8]
        s = np.concatenate(merge_multi_segment([c.reshape(-1).tolist() for c in cs]), 0)
    s = s.astype(np.float64)
    s[:, 0] /= W
    s[:, 1] /= H
    return np.clip(s, 0, 1)
