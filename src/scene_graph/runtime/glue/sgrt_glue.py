"""Thin evaluator-process glue for libsgrt.so (src/scene_graph/runtime/include/sgrt.h): hands the observation tensors'
pointers to the C++/CUDA runtime (YOLOE detection -> scenemap object map -> Spark-DSG save). No computation here.

    mem = SceneMemory(task_name, out_dir)        # once per process
    mem.step(obs)                                # every evaluator step, before the policy acts

The head RGB stays on the GPU (ovdet reads device memory); depth is copied to host only on keyframe steps.

Pose source (env SGRT_POSE, read by libsgrt): slam (default: base_qvel odometry + depth scan matching), odom, gt.
The glue pushes the simulator's ground-truth robot base pose (robot.get_position_orientation(), world frame) every step
whenever it can find the robot (SGRT_GT_POSE=0 turns that off): in gt mode it is the map pose, otherwise it is only
used for the drift diagnostic. GT is for sim debugging/visualisation only -- the challenge rules forbid it at
submission time. SGRT_GT_LOG=<csv> also logs per-keyframe GT base/head-camera poses; on close the GT object poses are
written next to it (<csv>.objects.json) for scoring object positions.
"""
import ctypes
import json
import math
import os
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # repo root (src/scene_graph/runtime/glue -> ../../../..)
LIB = os.environ.get("SGRT_LIB", str(pathlib.Path.home() / "sgrt_build/libsgrt.so"))
ENGINE = os.environ.get("SGRT_ENGINE", str(pathlib.Path.home() / "ovdet_models/x86_sm120/yoloe-11l-all.plan"))
PROMPTS = ROOT / "src/scene_graph/ovdet/config/task_prompts.txt"
HEAD_K = (306.0, 306.0, 360.0, 360.0)  # omnigibson.eval.utils.eval_utils.CAMERA_INTRINSICS["R1Pro"]["head"] (720x720)


class _Cfg(ctypes.Structure):
    _fields_ = [("engine", ctypes.c_char_p), ("names", ctypes.c_char_p), ("out_dir", ctypes.c_char_p),
                ("kf_every", ctypes.c_int32), ("save_s", ctypes.c_double), ("conf_th", ctypes.c_float)]


class _Diag(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int32), ("stamp", ctypes.c_double), ("last_xy", ctypes.c_double), ("last_yaw", ctypes.c_double),
                ("max_xy", ctypes.c_double), ("max_yaw", ctypes.c_double), ("rms_xy", ctypes.c_double), ("rms_yaw", ctypes.c_double),
                ("est", ctypes.c_double * 3), ("ref", ctypes.c_double * 3)]


class _Stage(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("n", ctypes.c_int64), ("mean_us", ctypes.c_double), ("p50_us", ctypes.c_double),
                ("p99_us", ctypes.c_double), ("max_us", ctypes.c_double), ("last_us", ctypes.c_double), ("total_us", ctypes.c_double)]


def _find_robot():
    """The evaluator's robot (single env): og.sim.scenes[0].robots[0]. None if OmniGibson isn't loaded."""
    try:
        import omnigibson as og
        for sc in og.sim.scenes:
            if sc.robots:
                return sc.robots[0]
    except Exception:
        pass
    return None


def _yaw(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def task_prompt_names(task: str) -> list[str]:
    """Task BDDL objects + scene structures, the ovdet prompt table (config/task_prompts.txt)."""
    lines = dict(l.split(":", 1) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if ":" in l and not l.startswith("#"))
    names = [n.strip() for n in lines.get(task, "").split(",") if n.strip()]
    return names + [n.strip() for n in lines["_scene"].split(",") if n.strip() and n.strip() not in names]


def _ptr(t):
    """(pointer, on_device, row_stride_bytes, pix_stride_bytes) of an HxWxC uint8 tensor/array."""
    if hasattr(t, "data_ptr"):
        es = t.element_size()
        return t.data_ptr(), int(t.is_cuda), t.stride(0) * es, t.stride(1) * es
    a = np.ascontiguousarray(t)
    return a.ctypes.data, 0, a.strides[0], a.strides[1]


class SceneMemory:
    def __init__(self, task: str, out_dir: str, kf_every: int = 6, save_s: float = 1.0, robot: str = "robot"):
        self.L = ctypes.CDLL(LIB)
        L = self.L
        L.sgrt_create.restype = ctypes.c_void_p
        L.sgrt_create.argtypes = [ctypes.POINTER(_Cfg), ctypes.c_char_p, ctypes.c_size_t]
        L.sgrt_begin.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t]
        L.sgrt_want_image.argtypes = [ctypes.c_void_p]
        L.sgrt_step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
                                ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
                                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        L.sgrt_save.argtypes = [ctypes.c_void_p]
        L.sgrt_stats.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
        L.sgrt_destroy.argtypes = [ctypes.c_void_p]
        # newer ABI (pose source, timing): an older libsgrt.so without these still works (no GT push, no timing)
        self.has_pose = hasattr(L, "sgrt_push_pose")
        self.has_timing = hasattr(L, "sgrt_get_stage_timing")
        if self.has_pose:
            L.sgrt_push_pose.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
            L.sgrt_get_pose_diag.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Diag)]
        if self.has_timing:
            L.sgrt_get_stage_timing.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stage), ctypes.c_int32]
        cfg = _Cfg(ENGINE.encode(), (ENGINE + ".names.txt").encode(), out_dir.encode(), kf_every, save_s, 0.25)
        err = ctypes.create_string_buffer(512)
        self.h = L.sgrt_create(ctypes.byref(cfg), err, 512)
        if not self.h:
            raise RuntimeError(f"sgrt_create: {err.value.decode()}")
        names = task_prompt_names(task)
        arr = (ctypes.c_char_p * len(names))(*[n.encode() for n in names])
        L.sgrt_begin(self.h, arr, len(names), err, 512)
        if err.value:
            print(f"[sgrt] prompt: {err.value.decode()}", flush=True)
        self.keys = (f"{robot}::proprio", f"{robot}::{robot}:zed_link:Camera:0::rgb", f"{robot}::{robot}:zed_link:Camera:0::depth_linear")
        self.t = 0
        self.robot = None
        self.use_gt = os.environ.get("SGRT_GT_POSE", "1") != "0" and self.has_pose
        self.gt_every = os.environ.get("SGRT_GT_EVERY", "0") == "1"
        self.kf_every = kf_every
        self.gt_log = None
        if os.environ.get("SGRT_GT_LOG"):
            self.gt_log_path = os.environ["SGRT_GT_LOG"]
            self.gt_log = open(self.gt_log_path, "w")
            self.gt_log.write("step,stamp,x,y,z,yaw,cam_x,cam_y,cam_z,cam_qx,cam_qy,cam_qz,cam_qw\n")
        self.py_ns = {"gt": 0, "prep": 0, "call": 0}
        self.py_n = 0
        print(f"[sgrt] {task}: {len(names)} prompt names, out {out_dir}", flush=True)

    def step(self, obs: dict):
        kp, kr, kd = self.keys
        if kp not in obs:  # weights' robot name differs from the evaluator's (e.g. robot_r1): find by suffix
            kp = next(k for k in obs if k.endswith("::proprio"))
            r = kp.split("::")[0]
            kr, kd = f"{r}::{r}:zed_link:Camera:0::rgb", f"{r}::{r}:zed_link:Camera:0::depth_linear"
            self.keys = (kp, kr, kd)
        t0 = time.perf_counter_ns()
        stamp = self.t / 30.0
        if self.use_gt:
            if self.robot is None and self.t % 30 == 0:
                self.robot = _find_robot()
                if self.robot is not None and self.gt_log is not None:
                    self.dump_gt_objects(self.gt_log_path + ".objects.json")  # now: the evaluator may os._exit before close()
            # reading the sim pose costs ~0.6 ms (python/torch); scenemap needs it only at image stamps (keyframe step - lag)
            # and for the keyframe step itself -> read on those steps unless SGRT_GT_EVERY=1
            kf, lag = self.kf_every, int(os.environ.get("SGRT_IMAGE_LAG", "1"))
            need = self.gt_every or kf <= 1 or (self.t % kf) in {0, (-lag) % kf} or self.t < 2
            if self.robot is not None and need:
                pos, q = self.robot.get_position_orientation()
                yaw = _yaw(q)
                self.L.sgrt_push_pose(self.h, stamp, float(pos[0]), float(pos[1]), yaw)
                if self.gt_log is not None and self.L.sgrt_want_image(self.h):
                    cam = self._head_cam()
                    cp, cq = cam.get_position_orientation() if cam is not None else ([math.nan] * 3, [math.nan] * 4)
                    self.gt_log.write(f"{self.t},{stamp:.4f},{float(pos[0]):.5f},{float(pos[1]):.5f},{float(pos[2]):.5f},{yaw:.6f},"
                                      + ",".join(f"{float(v):.6f}" for v in list(cp) + list(cq)) + "\n")
        t1 = time.perf_counter_ns()
        p = obs[kp]
        p = p[0] if p.ndim == 2 else p
        prop = np.ascontiguousarray((p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)), np.float32)
        rgb = depth = None
        rp, dev, rs, ps, w, h = None, 0, 0, 0, 0, 0
        want = self.L.sgrt_want_image(self.h)
        if want and not (kr in obs and kd in obs) and not getattr(self, "_warned", False):
            self._warned = True
            print(f"[sgrt] no head RGB-D in obs ({kr}, {kd}); keys: {sorted(obs)} — use RGBDFullResWrapper", flush=True)
        if want and kr in obs and kd in obs:
            rgb, depth = obs[kr], obs[kd]
            rgb = rgb[0] if rgb.ndim == 4 else rgb
            depth = depth[0] if depth.ndim == 3 else depth
            if hasattr(rgb, "contiguous"):
                rgb = rgb.contiguous()
            rp, dev, rs, ps = _ptr(rgb)
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            if hasattr(depth, "is_cuda") and depth.is_cuda:
                # one reused pinned host buffer: no per-keyframe allocation, faster device->host copy
                buf = getattr(self, "_dbuf", None)
                if buf is None or tuple(buf.shape) != tuple(depth.shape):
                    import torch
                    self._dbuf = buf = torch.empty(tuple(depth.shape), dtype=torch.float32, pin_memory=True)
                buf.copy_(depth)
                depth = buf.numpy()
            else:
                depth = np.ascontiguousarray((depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)), np.float32)
        t2 = time.perf_counter_ns()
        self.L.sgrt_step(self.h, stamp, prop.ctypes.data, prop.size, rp, dev, rs, ps, w, h,
                         depth.ctypes.data if depth is not None else None, *HEAD_K)
        t3 = time.perf_counter_ns()
        self.py_ns["gt"] += t1 - t0
        self.py_ns["prep"] += t2 - t1
        self.py_ns["call"] += t3 - t2
        self.py_n += 1
        self.t += 1
        if self.t % 900 == 0:  # the evaluator shuts the app down hard at the end -> report as we go
            if self.gt_log is not None:
                self.gt_log.flush()
            print(f"[sgrt] t={self.t} pose diag {self.pose_diag()}", flush=True)
            cs = self.clip_stats()
            if cs:
                print(f"[sgrt] t={self.t} clip {cs}", flush=True)
            for k, v in self.timing().items():
                print(f"[sgrt] timing {k}: " + " ".join(f"{a}={b:.1f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()),
                      flush=True)

    def _head_cam(self):
        if not hasattr(self, "_cam"):
            self._cam = None
            for name, sen in getattr(self.robot, "sensors", {}).items():
                if "zed_link" in name:
                    self._cam = sen
                    break
        return self._cam

    def pose_diag(self):
        if not self.has_pose:
            return {}
        d = _Diag()
        self.L.sgrt_get_pose_diag(self.h, ctypes.byref(d))
        return dict(n=d.n, last_xy=d.last_xy, last_yaw_deg=math.degrees(d.last_yaw), max_xy=d.max_xy,
                    max_yaw_deg=math.degrees(d.max_yaw), rms_xy=d.rms_xy, rms_yaw_deg=math.degrees(d.rms_yaw))

    def timing(self):
        if not self.has_timing:
            return {}
        arr = (_Stage * 64)()
        n = self.L.sgrt_get_stage_timing(self.h, arr, 64)
        out = {arr[i].name.decode(): dict(n=arr[i].n, mean=arr[i].mean_us, p50=arr[i].p50_us, p99=arr[i].p99_us, max=arr[i].max_us)
               for i in range(min(n, 64)) if arr[i].n}
        if self.py_n:
            for k, v in self.py_ns.items():
                out[f"py_{k}"] = dict(n=self.py_n, mean=v / self.py_n / 1e3)
        return out

    def dump_gt_objects(self, path):
        """GT object poses (category, name, world position, AABB extent) for scoring object positions."""
        try:
            scene = self.robot.scene
            objs = []
            for o in scene.objects:
                try:
                    pos, _ = o.get_position_orientation()
                    lo, hi = o.aabb
                    objs.append(dict(name=o.name, category=o.category, pos=[float(v) for v in pos],
                                     lo=[float(v) for v in lo], hi=[float(v) for v in hi]))
                except Exception:
                    pass
            pathlib.Path(path).write_text(json.dumps(objs))
            print(f"[sgrt] GT objects: {len(objs)} -> {path}", flush=True)
        except Exception as e:
            print(f"[sgrt] GT objects dump failed: {e}", flush=True)

    def stats(self):
        v = [ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32(), ctypes.c_float(), ctypes.c_float()]
        self.L.sgrt_stats(self.h, *[ctypes.byref(x) for x in v])
        return dict(keyframes=v[0].value, last_dets=v[1].value, objects=v[2].value, det_ms=v[3].value, save_ms=v[4].value)

    def clip_stats(self):
        """SGRT_CLIP counters and last-batch times (sgrt_get_clip_stats), None when off / old library."""
        L = self.L
        if not hasattr(L, "sgrt_get_clip_stats") or not L.sgrt_clip_enabled(ctypes.c_void_p(self.h)):
            return None
        f = [(n, ctypes.c_int32) for n in ("enabled", "n_objects", "n_named", "n_submitted", "n_done", "n_dropped", "last_batch")] + \
            [(n, ctypes.c_float) for n in ("crop_ms", "net_ms", "submit_us", "names_us", "save_ms")]
        st = type("_St", (ctypes.Structure,), {"_fields_": f})()
        L.sgrt_get_clip_stats(ctypes.c_void_p(self.h), ctypes.byref(st))
        return {n: (round(getattr(st, n), 3) if t is ctypes.c_float else getattr(st, n)) for n, t in f}

    def clip_report(self, queries=("radio", "라디오", "chair", "의자", "sofa", "소파")):
        """SGRT_CLIP: per-object names (en/ko) and label-table queries through the C ABI (sgrt.h, clip section)."""
        L = self.L
        if not hasattr(L, "sgrt_clip_enabled") or not L.sgrt_clip_enabled(ctypes.c_void_p(self.h)):
            return None

        class _St(ctypes.Structure):
            _fields_ = [(n, ctypes.c_int32) for n in ("enabled", "n_objects", "n_named", "n_submitted", "n_done", "n_dropped", "last_batch")] + \
                       [(n, ctypes.c_float) for n in ("crop_ms", "net_ms", "submit_us", "names_us", "save_ms")]

        class _Nm(ctypes.Structure):
            _fields_ = [("en", ctypes.c_char_p), ("ko", ctypes.c_char_p), ("score", ctypes.c_float)]
        L.sgrt_get_clip_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_St)]
        L.sgrt_object_names.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_Nm), ctypes.c_int32,
                                        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int32)]
        L.sgrt_query_label.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32),
                                       ctypes.POINTER(ctypes.c_float)]
        L.sgrt_save(self.h)   # names come from the save (label lookup + cache)
        st = _St()
        L.sgrt_get_clip_stats(self.h, ctypes.byref(st))
        out = {"stats": {n: getattr(st, n) for n, _ in _St._fields_}, "objects": {}, "queries": {}}
        for oid in range(1, 400):
            nm = (_Nm * 3)()
            le, lk, sv = ctypes.c_char_p(), ctypes.c_char_p(), ctypes.c_int32()
            n = L.sgrt_object_names(self.h, oid, nm, 3, ctypes.byref(le), ctypes.byref(lk), ctypes.byref(sv))
            if n > 0:
                out["objects"][oid] = (le.value.decode(), lk.value.decode(), round(nm[0].score, 3), [x.en.decode() for x in nm[:n]], bool(sv.value))
        for q in queries:
            ids, sc = (ctypes.c_uint32 * 5)(), (ctypes.c_float * 5)()
            n = L.sgrt_query_label(self.h, q.encode(), 5, ids, sc)
            out["queries"][q] = [(ids[i], round(sc[i], 3)) for i in range(max(n, 0))] if n >= 0 else n
        print(f"[sgrt] clip: {out}", flush=True)
        return out

    def close(self):
        if self.h:
            try:
                self.clip_report()
            except Exception as e:  # report only
                print(f"[sgrt] clip report failed: {e}", flush=True)
            print(f"[sgrt] pose diag (slam vs GT): {self.pose_diag()}", flush=True)
            for k, v in self.timing().items():
                print(f"[sgrt] timing {k}: " + " ".join(f"{a}={b:.1f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()),
                      flush=True)
            if self.gt_log is not None:
                self.gt_log.close()
                if self.robot is not None:
                    self.dump_gt_objects(self.gt_log_path + ".objects.json")
            self.L.sgrt_save(self.h)
            self.L.sgrt_destroy(self.h)
            self.h = None
