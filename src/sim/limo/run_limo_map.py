"""LIMO + OMX-F through the scene-graph runtime in OmniGibson (map check, no agent, no move_robot):
the official evaluator with our robot (robot-agent src/robot/og/eval_with_limo.py, --robot-config limo_omx_eval.yaml) and
--policy local, whose LocalPolicy gets LimoMapPolicy:

    every step : SceneMemory.step(obs)   (libsgrt robot limo_omx: 12-dim LIMO proprio packed by sgrt_glue, body camera
                                          robot_limo:eyes:Camera:0 = scenemap cam 0 with the sensor's intrinsics)
    keyframe   : sgrt_map(view) -> simple wander from the front scan (spin 360 deg, then forward while the corridor ahead
                 is clear, else turn), GT base pose (sim) logged next to the map pose
    at --steps : sgrt_save, map vs the GT floor map (src/sim/explore/gt/<scene>__<task>.pgm, world frame) and pose error
                 (slam vs GT, sgrt_get_pose_diag) -> <out>/summary.json, <out>/overlay.png; body camera extrinsics vs
                 scenemap's LIMO forward kinematics (sm_robot_fk) -> summary.json "cam0_fk"

    python run_limo_map.py --out <dir> --steps 900 [--gt-dir DIR] -- --task-name turning_on_radio --mode public_test \
        --instance-indices 0 --num-envs 1 --max-steps 960 --headless \
        --robot-config <robot-agent>/src/robot/og/limo_omx_eval.yaml --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper
(run_limo_map.sh sets all of this.) GT is used only for logging/scoring, never by the map (SGRT_POSE=slam by default).
"""
import argparse
import ctypes
import json
import math
import os
import pathlib
import runpy
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src/scene_graph/runtime/glue"))
sys.path.insert(0, str(REPO / "src/sim/explore"))
sys.path.insert(0, str(REPO / "src/sim/move_robot"))
SHIM = pathlib.Path(os.environ.get("LIMO_SHIM", pathlib.Path.home() / "robot-agent/src/robot/og/eval_with_limo.py"))

VMAX, WMAX = 0.5, 0.8727          # limo_omx_eval.yaml base command_output_limits (input [-1, 1])


class SmBodyFk(ctypes.Structure):
    """scenemap.h sm_body_fk"""
    _fields_ = [("n_cams", ctypes.c_int32), ("n_hands", ctypes.c_int32), ("cam_valid", ctypes.c_int32 * 3),
                ("T_cam", (ctypes.c_double * 12) * 3), ("eef_valid", ctypes.c_int32 * 2), ("T_eef", (ctypes.c_double * 12) * 2),
                ("grip", ctypes.c_float * 2)]


def quat_R(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LimoMapPolicy:
    def __init__(self, task, out: pathlib.Path, steps: int, gt_dir: pathlib.Path):
        from sgrt_glue import SceneMemory, _find_robot
        from run_explore import SgrtMapView, load_gt

        self.out, self.steps, self.task = out, steps, task
        (out / "memory").mkdir(parents=True, exist_ok=True)
        self.mem = SceneMemory(task, str(out / "memory"), kf_every=6, robot_model=os.environ.get("SGRT_ROBOT") or "limo_omx")
        assert self.mem.robot_id == 1, "libsgrt did not select limo_omx"
        S = self.mem.L
        S.sgrt_map.restype = ctypes.c_int
        S.sgrt_map.argtypes = [ctypes.c_void_p, ctypes.POINTER(SgrtMapView)]
        S.sm_robot_fk.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(SmBodyFk)]
        self.view = SgrtMapView()
        self.robot = _find_robot()
        self.gt_meta, self.gt_floor = load_gt(gt_dir, task)
        self.i = 0
        self.mode, self.turn_left = "spin", 0.0
        self.log = open(out / "poses.csv", "w")
        self.log.write("step,gt_x,gt_y,gt_yaw,map_x,map_y,map_yaw,mode,front_m,gt_z,tilt_deg\n")
        self.first = None      # (gt world pose, map pose) at the first keyframe
        self.done = False
        self.act_dim = None
        self.cam_checks = []

    # ---------- actions
    def _action(self, vx, wz, batched):
        import torch as th
        r = self.robot
        if self.act_dim is None:
            self.act_dim = int(r.action_dim)
            self.idx = {k: v.tolist() if hasattr(v, "tolist") else list(v) for k, v in r.controller_action_idx.items()}
            self.arm_hold = r.reset_joint_pos[r.arm_control_idx["0"]].clone()
            print(f"[limo] action dim {self.act_dim}, idx {self.idx}", flush=True)
        a = th.zeros(self.act_dim, dtype=th.float32)
        b = self.idx["base"]
        a[b[0]] = max(-1.0, min(1.0, vx / VMAX))
        a[b[2]] = max(-1.0, min(1.0, wz / WMAX))
        a[self.idx["arm_0"]] = self.arm_hold.float()
        a[self.idx["gripper_0"]] = -1.0   # omx_gripper_joint_1 lower limit 0 = closed (home)
        return a[None] if batched else a

    def _front_clear(self):
        """nearest scan hit in the corridor ahead (|y| < 0.22 m, base frame of the last keyframe), m"""
        v = self.view
        if v.n_hit <= 0:
            return 9.0
        hx = np.ctypeslib.as_array(ctypes.cast(v.hit_x, ctypes.POINTER(ctypes.c_float)), (v.n_hit,))
        hy = np.ctypeslib.as_array(ctypes.cast(v.hit_y, ctypes.POINTER(ctypes.c_float)), (v.n_hit,))
        m = (hx > 0) & (np.abs(hy) < 0.22)
        return float(hx[m].min()) if m.any() else 9.0

    def _near_clip(self):
        """Body camera near plane: robot-agent 391c04b moved robot_limo:eyes to the lens and eval_with_limo.py sets near clip
        0.05 m. Older assets had the camera ~1 cm inside the shell (depth ~0.0096 m everywhere). Shared raise-only safety net
        (src/sim/move_robot/move_robot_limo.apply_near_clip, LIMO_NEAR_CLIP default 0.05, 0 = leave)."""
        from move_robot_limo import apply_near_clip
        apply_near_clip(self.robot)

    # ---------- checks
    def _cam_fk_check(self, prop12):
        """eyes sensor pose (sim) relative to base_footprint vs scenemap sm_robot_fk(limo_omx) cam 0 (depth_camera_lens_optical_frame)."""
        r = self.robot
        sen = next((s for n, s in r.sensors.items() if ":eyes:" in n), None)
        if sen is None:
            return
        bp, bq = r.get_position_orientation()
        cp, cq = sen.get_position_orientation()
        Rb, Rc = quat_R(bq), quat_R(cq) @ np.diag([1.0, -1.0, -1.0])   # OG camera (-z fwd, y up) -> optical (z fwd, y down)
        pb = np.array([float(v) for v in bp]) - Rb @ np.array([0, 0, 0.15])   # base_link -> base_footprint (URDF base_joint)
        rel_R = Rb.T @ Rc
        rel_p = Rb.T @ (np.array([float(v) for v in cp]) - pb)
        fk = SmBodyFk()
        p = np.ascontiguousarray(prop12, np.float32)
        self.mem.L.sm_robot_fk(1, p.ctypes.data, p.size, ctypes.byref(fk))
        T = np.array(fk.T_cam[0][:]).reshape(3, 4)
        dp = float(np.linalg.norm(T[:, 3] - rel_p))
        dR = float(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(T[:, :3].T @ rel_R) - 1) / 2)))))
        self.cam_checks.append({"step": self.i, "sim_p": rel_p.round(4).tolist(), "fk_p": T[:, 3].round(4).tolist(), "dp_m": dp, "drot_deg": dR})

    def _score(self):
        """map (estimated frame) -> world via the first keyframe's GT/map poses; occupied/free cells vs the GT floor map"""
        v = self.view
        cells = np.ctypeslib.as_array(ctypes.cast(v.cells, ctypes.POINTER(ctypes.c_int8)), (v.h, v.w)).copy()
        out = {"map_w": v.w, "map_h": v.h, "res": v.res, "known_m2": float((cells >= 0).sum() * v.res ** 2),
               "occ_cells": int((cells >= 65).sum()), "free_cells": int(((cells >= 0) & (cells <= 25)).sum())}
        if self.gt_floor is None or self.first is None:
            return out, cells
        (wx0, wy0, wyaw0), (mx0, my0, myaw0) = self.first
        d = wyaw0 - myaw0
        c, s = math.cos(d), math.sin(d)
        ys, xs = np.mgrid[0:v.h, 0:v.w]
        mx, my = v.origin[0] + (xs + 0.5) * v.res - mx0, v.origin[1] + (ys + 0.5) * v.res - my0
        wx, wy = c * mx - s * my + wx0, s * mx + c * my + wy0
        meta, floor = self.gt_meta, self.gt_floor
        gx = np.floor((wx - meta["origin"][0]) / meta["res"]).astype(int)
        gy = np.floor((wy - meta["origin"][1]) / meta["res"]).astype(int)
        inside = (gx >= 0) & (gy >= 0) & (gx < floor.shape[1]) & (gy < floor.shape[0])
        fl = np.zeros_like(inside)
        fl[inside] = floor[gy[inside], gx[inside]]

        def dil(a, r):   # binary dilation by r cells (square)
            o = a.copy()
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    o |= np.roll(np.roll(a, dy, 0), dx, 1)
            return o
        nonfloor = ~floor
        occ, free = cells >= 65, (cells >= 0) & (cells <= 25)
        for tol in (0, 1, 2):   # cells of 0.05 m
            nf = dil(nonfloor, tol) if tol else nonfloor
            fd = dil(floor, tol) if tol else floor
            nf_m = np.ones_like(inside)
            nf_m[inside] = nf[gy[inside], gx[inside]]
            fd_m = np.zeros_like(inside)
            fd_m[inside] = fd[gy[inside], gx[inside]]
            out[f"occ_on_nonfloor_tol{tol * 5}cm"] = float(nf_m[occ].mean()) if occ.any() else None
            out[f"free_on_floor_tol{tol * 5}cm"] = float(fd_m[free].mean()) if free.any() else None
        # GT walls seen: floor-boundary cells (floor next to non-floor) inside the known map area that the map marks occupied (±10 cm)
        bnd = floor & dil(nonfloor, 1)
        known_w = np.zeros_like(floor)
        occ_w = np.zeros_like(floor)
        known_w[gy[inside & (cells >= 0)], gx[inside & (cells >= 0)]] = True
        occ_w[gy[inside & occ], gx[inside & occ]] = True
        seen = bnd & known_w
        out["gt_boundary_cells_in_known"] = int(seen.sum())
        out["gt_boundary_hit_tol10cm"] = float((dil(occ_w, 2)[seen]).mean()) if seen.any() else None
        out["floor_explored_m2"] = float((fl & free).sum() * v.res ** 2)
        self._overlay(floor, meta, occ, free, wx, wy)
        return out, cells

    def _overlay(self, floor, meta, occ, free, wx, wy):
        try:
            from PIL import Image
        except Exception:
            return
        img = np.where(floor[..., None], np.uint8([200, 200, 200]), np.uint8([60, 60, 60])).astype(np.uint8)
        for m, col in ((free, (150, 230, 150)), (occ, (230, 30, 30))):
            gx = np.floor((wx[m] - meta["origin"][0]) / meta["res"]).astype(int)
            gy = np.floor((wy[m] - meta["origin"][1]) / meta["res"]).astype(int)
            ok = (gx >= 0) & (gy >= 0) & (gx < floor.shape[1]) & (gy < floor.shape[0])
            img[gy[ok], gx[ok]] = col
        for row in open(self.out / "poses.csv").read().splitlines()[1:]:
            f = row.split(",")
            gx, gy = int((float(f[1]) - meta["origin"][0]) / meta["res"]), int((float(f[2]) - meta["origin"][1]) / meta["res"])
            if 0 <= gx < floor.shape[1] and 0 <= gy < floor.shape[0]:
                img[gy, gx] = (30, 60, 230)
        Image.fromarray(img[::-1]).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(self.out / "overlay.png")

    def _finish(self):
        self.done = True
        self.log.flush()
        self.mem.L.sgrt_save(self.mem.h)
        self.mem.L.sgrt_map(self.mem.h, ctypes.byref(self.view))
        sc, cells = self._score()
        diag = self.mem.pose_diag()
        rows = [r.split(",") for r in open(self.out / "poses.csv").read().splitlines()[1:]]
        path = sum(math.hypot(float(b[1]) - float(a[1]), float(b[2]) - float(a[2])) for a, b in zip(rows, rows[1:]))
        rot = sum(abs(wrap(float(b[3]) - float(a[3]))) for a, b in zip(rows, rows[1:]))
        summ = {"steps": self.i, "sim_s": self.i / 30.0, "pose_mode": os.environ.get("SGRT_POSE", "slam"), "gt_path_m": path,
                "gt_rot_deg": math.degrees(rot), "pose_diag": diag, "map": sc, "objects": self.mem.stats(),
                "cam0_fk": self.cam_checks, "timing": {k: {a: round(b, 1) if isinstance(b, float) else b for a, b in v.items()}
                                                       for k, v in self.mem.timing().items()}}
        (self.out / "summary.json").write_text(json.dumps(summ, indent=1))
        print(f"[limo] summary: {json.dumps(summ)}", flush=True)

    # ---------- evaluator hooks
    def act(self, obs):
        kp = next(k for k in obs if k.endswith("::proprio"))
        batched = obs[kp].ndim == 2
        self.mem.step(obs)
        if self.robot is None:
            from sgrt_glue import _find_robot
            self.robot = _find_robot()
        if self.i == 0:
            self._near_clip()
        if self.done:
            return self._action(0.0, 0.0, batched)
        vx = wz = 0.0
        if self.i % 6 == 0:
            self.mem.L.sgrt_map(self.mem.h, ctypes.byref(self.view))
            pos, q = self.robot.get_position_orientation()
            g = (float(pos[0]), float(pos[1]), yaw_of(q))
            m = (self.view.pose[0], self.view.pose[1], self.view.pose[2])
            if self.first is None and self.view.w > 0:
                self.first = (g, m)
            self.front = self._front_clear()
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, quat_R(q)[2, 2]))))
            self.log.write(f"{self.i},{g[0]:.4f},{g[1]:.4f},{g[2]:.5f},{m[0]:.4f},{m[1]:.4f},{m[2]:.5f},{self.mode},{self.front:.3f},"
                           f"{float(pos[2]):.4f},{tilt:.2f}\n")
            if self.i % 300 == 0:
                self._cam_fk_check(self.mem._lp.copy())
        # wander: spin once, then forward while the corridor ahead is clear (> 0.7 m), else turn ~100 deg left
        if self.mode == "spin":
            wz = 0.6
            if self.i >= int(2 * math.pi / 0.6 * 30) + 6:
                self.mode = "fwd"
        elif self.mode == "fwd":
            if getattr(self, "front", 9.0) > 0.7:
                vx = 0.25
            else:
                self.mode, self.turn_left = "turn", math.radians(100)
        elif self.mode == "turn":
            wz = 0.6
            self.turn_left -= 0.6 / 30
            if self.turn_left <= 0:
                self.mode = "fwd"
        self.i += 1
        if self.i >= self.steps:
            self._finish()
        return self._action(vx, wz, batched)

    def reset(self):
        pass

    def close(self):
        if not self.done:
            self._finish()
        self.mem.close()


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--gt-dir", default=str(REPO / "src/sim/explore/gt"))
    args = ap.parse_args(argv[:split])
    eval_args = argv[split + 1:]
    if "--policy" in eval_args:
        i = eval_args.index("--policy")
        del eval_args[i:i + 2]
    task = eval_args[eval_args.index("--task-name") + 1]
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SGRT_ROBOT", "limo_omx")

    import omnigibson.eval.policies as P
    orig_init = P.LocalPolicy.__init__
    holder = {}

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.policy = LimoMapPolicy(task, out, args.steps, pathlib.Path(args.gt_dir))
        holder["p"] = self.policy

    P.LocalPolicy.__init__ = init
    sys.argv = [str(SHIM), *eval_args, "--policy", "local", "--output-dir", str(out)]
    try:
        runpy.run_path(str(SHIM), run_name="__main__")
    finally:
        if "p" in holder:
            holder["p"].close()


if __name__ == "__main__":
    main()
