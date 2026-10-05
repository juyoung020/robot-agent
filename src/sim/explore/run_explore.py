"""Launcher for the explore skill (robot-agent src/agent/skills/explore — prompts + skill.json; the loop is the agent runtime src/agent/runtime): the official evaluator (omnigibson.eval.eval,
unmodified) with --policy local, whose LocalPolicy gets ExplorePolicy:

    every step : SceneMemory.step(obs)  (libsgrt: ObjectSAM + SigLIP 2 + scenemap objprob, Cartographer pose + mapper2d grid, periodic save to <out>/memory)
                 keyframe -> sgrt_map(view) -> mr_set_map(robot, &view)   (pointer hand-off, Python touches no cells)
                 MoveRobotPolicy.act(obs)   (libmove_robot closed loop: go_to / probe / delta / joints)
    tool calls : TCP line JSON from the agent (`run-skill --skill explore --addr ...`), result line back (with "map" summary + "_m" metrics)
    executor   : {"executor":"vla","skill":"...","objects":[ids],"max_s":20}
                 LIMO: libmove_robot VLA executor (robot-agent docs/map_vla/POLICY.md 1.3): policy (scripted / replay stand-in
                 until a trained VLA exists, --vla-policy) -> move_robot safety filter -> end signal + verify + budget ->
                 {"status":"done|failed|timeout|handback","reason","evidence","steps","min_clear_m","contacts"}; objects come from
                 the scenemap snapshot (sm_snap_objects, ids "O<id>") every keyframe, contacts split body / arm.
    metrics    : GT floor reference (gt_trav.py) in the map frame -> mr_set_reference; robot-link contacts with anything
                 but the floor (RigidContactAPI) -> mr_set_contacts; robot footprint (AABB) logged once
    viewer     : <out>/memory/explore.json (trail, planned path, goal, frontier ids) every second

    python run_explore.py --listen 127.0.0.1:8771 --out <dir> -- --task-name bringing_water --mode public_test \
        --instance-indices 0 --num-envs 1 --max-steps 27000 --headless --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper

LIMO + OMX-F is the only robot (--robot limo_omx, default; R1 removed 10-06): the evaluator is started through robot-agent's
src/robot/og/eval_with_limo.py (--limo-shim, LIMO_SHIM; presampled-pose alias, agent_metric fix) and the eval args must carry
--robot-config <robot-agent>/src/robot/og/limo_omx_eval.yaml; move_robot runs through move_robot_limo.LimoMoveRobotPolicy
(base only, arm held at home, gripper closed), libsgrt is told limo_omx, and the body camera gets the LIMO_NEAR_CLIP
safety net (raise-only, robot-agent 391c04b already sets 0.05 m) on the first step; GT/slam poses -> poses.csv, pose_diag.json.
The BEHAVIOR task template still names R1 (robot_poses key, robot config); it is only read for scene/task/start pose -- the evaluator robot is replaced by --robot-config limo_omx_eval.yaml and eval_with_limo.py aliases robot_poses to "robot".
"""
import argparse
import ctypes
import json
import os
import pathlib
import runpy
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "move_robot"))
sys.path.insert(0, str(REPO / "src/scene_graph/runtime/glue"))


class SgrtMapView(ctypes.Structure):
    """sgrt.h sgrt_map_view (same layout as move_robot src/ffi.rs SgrtMapView)"""
    _fields_ = [("stamp", ctypes.c_double), ("pose", ctypes.c_double * 3), ("res", ctypes.c_double),
                ("origin", ctypes.c_double * 2), ("w", ctypes.c_int32), ("h", ctypes.c_int32),
                ("cells", ctypes.c_void_p), ("room_res", ctypes.c_double), ("room_origin", ctypes.c_double * 2),
                ("room_w", ctypes.c_int32), ("room_h", ctypes.c_int32), ("room_ids", ctypes.c_void_p),
                ("n_rooms", ctypes.c_int32), ("scan_pose", ctypes.c_double * 3), ("scan_origin", ctypes.c_float * 2),
                ("n_hit", ctypes.c_int32), ("hit_x", ctypes.c_void_p), ("hit_y", ctypes.c_void_p),
                ("n_free", ctypes.c_int32), ("free_x", ctypes.c_void_p), ("free_y", ctypes.c_void_p),
                ("dirty", ctypes.c_int32), ("dirty_box", ctypes.c_int32 * 4), ("map_version", ctypes.c_uint64),
                ("n_movable", ctypes.c_int32), ("movable_xyr", ctypes.c_void_p)]


def yaw_of(q):
    import math
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_gt(gt_dir: pathlib.Path, task: str):
    import numpy as np
    for j in gt_dir.glob(f"*__{task}.json"):
        meta = json.loads(j.read_text())
        # 기본 분모 = 닿을 수 있는 바닥(.reach.pgm, 닫힌 문 뒤 방 뺌). EXPLORE_GT_REF=all 이면 방 안 바닥 전부
        pg = j.with_suffix(".reach.pgm")
        if not pg.exists() or os.environ.get("EXPLORE_GT_REF") == "all":
            pg = j.with_suffix(".pgm")
        print(f"[explore] reference {pg.name}", flush=True)
        b = pg.read_bytes()
        parts = b.split(b"\n", 3)
        w, h = map(int, parts[1].split())
        a = np.frombuffer(parts[3], np.uint8)[: w * h].reshape(h, w) > 127
        return meta, a
    return None, None


def reference_in_map(meta, floor, T):
    """GT floor (world grid) -> map-frame grid (res = meta res). T = (cos, sin, tx, ty): map = R * world + t"""
    import numpy as np
    res = meta["res"]
    ox, oy = meta["origin"]
    ys, xs = np.nonzero(floor)
    wx, wy = ox + (xs + 0.5) * res, oy + (ys + 0.5) * res
    c, s, tx, ty = T
    mx, my = c * wx - s * wy + tx, s * wx + c * wy + ty
    x0, y0 = mx.min() - 1.0, my.min() - 1.0
    W, H = int((mx.max() + 1.0 - x0) / res) + 1, int((my.max() + 1.0 - y0) / res) + 1
    gx, gy = np.meshgrid(x0 + (np.arange(W) + 0.5) * res, y0 + (np.arange(H) + 0.5) * res)
    # inverse: world = R^T (map - t)
    qx, qy = gx - tx, gy - ty
    wxx, wyy = c * qx + s * qy, -s * qx + c * qy
    ix, iy = np.floor((wxx - ox) / res).astype(int), np.floor((wyy - oy) / res).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < floor.shape[1]) & (iy < floor.shape[0])
    out = np.zeros((H, W), np.uint8)
    out[ok] = floor[iy[ok], ix[ok]]
    return out, res, x0, y0


class VlaSource:
    """Wraps the tool-call source: {"executor": "vla", ...} calls are taken by ExplorePolicy, others go to move_robot."""

    def __init__(self, inner, owner):
        self.inner, self.owner = inner, owner
        self.pending = None

    def poll(self):
        c = self.inner.poll()
        if isinstance(c, dict) and c.get("executor") == "vla":
            self.owner.start_vla(c)
            return None
        return c

    def reply(self, r):
        self.inner.reply(r)

    def close(self):
        self.inner.close()


class ExplorePolicy:
    def __init__(self, src, task, out_dir: pathlib.Path, lib_path=None, gt_dir=None, kf_every=6, robot="limo_omx", vla_policy=None):
        from move_robot_limo import LimoMoveRobotPolicy
        from sgrt_glue import SceneMemory
        if robot != "limo_omx":
            raise ValueError("explore runs only our robot (LIMO + OMX-F): robot=%r" % robot)

        self.out = out_dir
        (out_dir / "memory").mkdir(parents=True, exist_ok=True)
        self.src = VlaSource(src, self)
        self.limo = True   # LIMO + OMX-F is the only robot (the template's R1 is replaced via --robot-config, robot-agent src/robot/og/eval_with_limo.py)
        self.mr = LimoMoveRobotPolicy(self.src, lib_path=lib_path, log_path=str(out_dir / "move_robot.jsonl"))
        L = self.mr.lib.L
        L.mr_set_map.restype = ctypes.c_int
        L.mr_set_map.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.mr_set_reference.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        L.mr_set_contacts.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        L.mr_overlay_json.restype = ctypes.c_ssize_t
        L.mr_overlay_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.L = L
        self.mem = SceneMemory(task, str(out_dir / "memory"), kf_every=kf_every, robot_model="limo_omx")
        S = self.mem.L
        S.sgrt_map.restype = ctypes.c_int
        S.sgrt_map.argtypes = [ctypes.c_void_p, ctypes.POINTER(SgrtMapView)]
        self.view = SgrtMapView()
        self.kf_every = kf_every
        self.task = task
        self.gt_meta, self.gt_floor = load_gt(pathlib.Path(gt_dir) if gt_dir else HERE / "gt", task)
        self.step_i = 0
        self.robot = None
        self.contact_n = 0
        self.contact_now = set()
        self.contact_log = open(out_dir / "contacts.jsonl", "a")
        self.timing = {"mem": 0.0, "map": 0.0, "mr": 0.0, "contacts": 0.0, "between": 0.0, "n": 0, "map_n": 0}
        self.t_last = None
        self.tim_log = open(out_dir / "timing.jsonl", "a")
        self.obuf = ctypes.create_string_buffer(1 << 20)
        self.vla_policy = vla_policy          # LIMO: default policy for calls without "policy" (MR_VLA_POLICY, else scripted)
        self.limo_vla = False                 # a LIMO VLA step is running in libmove_robot
        self.vla_log = open(out_dir / "vla.jsonl", "a")
        if self.limo:
            S.sgrt_map_snapshot.restype = ctypes.c_void_p
            S.sgrt_map_snapshot.argtypes = [ctypes.c_void_p]
            S.sm_snap_objects.restype = ctypes.c_int
            S.sm_snap_objects.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        print(f"[explore] task {task}, out {out_dir}, gt {'yes' if self.gt_floor is not None else 'no'}"
              + (", robot limo_omx" if self.limo else ""), flush=True)

    # ---------- VLA executor switch
    def start_vla(self, call):
        return self._start_limo_vla(call)

    def _start_limo_vla(self, call):
        call = dict(call)
        if self.vla_policy and "policy" not in call:
            call["policy"] = self.vla_policy
        self._push_objects()
        self.vla_log.write(json.dumps({"step": self.step_i, "call": call}) + "\n")
        self.vla_t0 = time.perf_counter()
        res = self.mr.vla_start(call)
        if res is not None:
            self._vla_reply(res)
        else:
            self.limo_vla = True
            print(f"[explore] VLA step started: {json.dumps(call)}", flush=True)

    def _vla_reply(self, res):
        res = dict(res)
        res["wall_s"] = round(time.perf_counter() - getattr(self, "vla_t0", time.perf_counter()), 2)
        self.vla_log.write(json.dumps({"step": self.step_i, "result": res}) + "\n")
        self.vla_log.flush()
        print(f"[explore] VLA result: {json.dumps(res)}", flush=True)
        self.src.reply(res)

    def _push_objects(self):
        """scenemap objects of the last sgrt_map snapshot -> libmove_robot (pointer hand-off, ids O<id>)"""
        if not (self.limo and self.mr.lib.has_vla):
            return
        snap = self.mem.L.sgrt_map_snapshot(self.mem.h)
        if not snap:
            return
        p = ctypes.c_void_p()
        n = self.mem.L.sm_snap_objects(snap, ctypes.byref(p))
        if n >= 0:
            self.L.mr_vla_set_objects(self.mr.lib.h, p, n, self.step_i / 30.0)

    # ---------- sim-side measurement
    def _robot(self):
        if self.robot is None:
            import omnigibson as og
            self.robot = og.sim.scenes[0].robots[0]
            try:
                lo, hi = self.robot.aabb
                ext = [round(float(hi[i] - lo[i]), 3) for i in range(3)]
                bl = self.robot.links.get("base_link")
                bext = None
                if bl is not None:
                    blo, bhi = bl.aabb
                    bext = [round(float(bhi[i] - blo[i]), 3) for i in range(3)]
                print(f"[explore] robot aabb extent {ext}, base_link {bext}", flush=True)
                (self.out / "robot_footprint.json").write_text(json.dumps({"robot": "limo_omx",
                                                                     "robot_aabb_extent": ext, "base_link_aabb_extent": bext}))
            except Exception as e:  # noqa: BLE001
                print(f"[explore] aabb: {e}", flush=True)
            self.robot_paths = {l.prim_path for l in self.robot.links.values()}
        return self.robot

    def _set_reference(self):
        import numpy as np
        if self.gt_floor is None:
            return
        r = self._robot()
        pos, q = r.get_position_orientation()
        wyaw = yaw_of(q)
        mx, my, myaw = self.view.pose[0], self.view.pose[1], self.view.pose[2]
        import math
        d = myaw - wyaw
        c, s = math.cos(d), math.sin(d)
        tx = mx - (c * float(pos[0]) - s * float(pos[1]))
        ty = my - (s * float(pos[0]) + c * float(pos[1]))
        self.T = (c, s, tx, ty)
        ref, res, x0, y0 = reference_in_map(self.gt_meta, self.gt_floor, self.T)
        self._ref = np.ascontiguousarray(ref)
        self.L.mr_set_reference(self.mr.lib.h, self._ref.ctypes.data, ref.shape[1], ref.shape[0], res, x0, y0)
        (self.out / "frame.json").write_text(json.dumps({"world_start": [float(pos[0]), float(pos[1]), wyaw],
                                                          "map_start": [mx, my, myaw], "map_from_world": self.T}))
        print(f"[explore] reference set: {int(ref.sum())} cells ({ref.sum() * res * res:.1f} m2)", flush=True)

    def _push_gt_pose_to_move_robot(self):
        """SGRT_POSE=gt: the robot's pose, trajectory (path_log) and path length follow the simulator's GT pose, not base_qvel integration."""
        if os.environ.get("SGRT_POSE", "").lower() != "gt" or not self.mr.lib.has_gt_pose:
            return
        import math
        pos, q = self._robot().get_position_orientation()
        wyaw = yaw_of(q)
        c, s, tx, ty = getattr(self, "T", (1.0, 0.0, 0.0, 0.0))  # map_from_world; identity in GT mode (map = world)
        x, y = float(pos[0]), float(pos[1])
        self.mr.lib.set_gt_pose(c * x - s * y + tx, s * x + c * y + ty, wyaw + math.atan2(s, c))

    def _limo_pose_log(self):
        """LIMO runs: per map keyframe the GT base pose (world and map frame, via map_from_world) next to the map (slam) pose
        -> <out>/poses.csv; libsgrt's slam-vs-GT diagnostics (sgrt_get_pose_diag) -> <out>/pose_diag.json about every second
        (the evaluator may exit before close()). GT is logged only, never fed to the map in slam mode."""
        import math
        if not hasattr(self, "pose_log"):
            self.pose_log = open(self.out / "poses.csv", "w")
            self.pose_log.write("step,gt_x,gt_y,gt_yaw,gt_map_x,gt_map_y,gt_map_yaw,map_x,map_y,map_yaw\n")
        pos, q = self._robot().get_position_orientation()
        x, y, yaw = float(pos[0]), float(pos[1]), yaw_of(q)
        c, s, tx, ty = getattr(self, "T", (1.0, 0.0, 0.0, 0.0))
        gm = (c * x - s * y + tx, s * x + c * y + ty, math.atan2(math.sin(yaw + math.atan2(s, c)), math.cos(yaw + math.atan2(s, c))))
        v = self.view.pose
        self.pose_log.write(f"{self.step_i},{x:.4f},{y:.4f},{yaw:.5f},{gm[0]:.4f},{gm[1]:.4f},{gm[2]:.5f},{v[0]:.4f},{v[1]:.4f},{v[2]:.5f}\n")
        if self.step_i % 30 == 0:
            self.pose_log.flush()
            d = self.mem.pose_diag()
            d["step"] = self.step_i
            (self.out / "pose_diag.json").write_text(json.dumps(d))

    def _contacts(self):
        try:
            from omnigibson.utils.usd_utils import RigidContactAPI
            pairs = RigidContactAPI.get_contact_pairs(0, list(self.robot_paths), None, current_only=False)
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_cwarn", False):
                print(f"[explore] contacts unavailable: {e}", flush=True)
                self._cwarn = True
            return
        now = set()
        links = {}
        for a, b in pairs:
            mine, other = (a, b) if a in self.robot_paths else (b, a)
            if other in self.robot_paths:
                continue
            low = other.lower()
            if "floor" in low or "collisionplane" in low:
                continue
            now.add(other)
            links.setdefault(other, set()).add(mine.rsplit("/", 1)[-1])
        new = now - self.contact_now
        if new:
            self.contact_n += len(new)
            self.contact_log.write(json.dumps({"step": self.step_i, "new": sorted(new), "robot_links": {k: sorted(links[k]) for k in new}}) + "\n")
            self.contact_log.flush()
            self.L.mr_set_contacts(self.mr.lib.h, self.contact_n)
            if self.limo and self.mr.lib.has_vla:
                # body (chassis / wheels) vs arm + gripper (omx_*): a VLA step fails as unsafe on new BODY contacts only
                arm = sum(1 for k in new if all(l.startswith("omx_") for l in links[k]))
                self.contact_arm = getattr(self, "contact_arm", 0) + arm
                self.L.mr_vla_contacts(self.mr.lib.h, self.contact_n - self.contact_arm, self.contact_arm)
        self.contact_now = now

    def act(self, obs):
        t0 = time.perf_counter()
        if self.t_last is not None:
            self.timing["between"] += t0 - self.t_last
        if self.limo and self.step_i == 0:
            from move_robot_limo import apply_near_clip
            apply_near_clip(self._robot())
        self.mem.step(obs)
        t1 = time.perf_counter()
        did_map = False
        if self.step_i % self.kf_every == 0:
            if self.mem.L.sgrt_map(self.mem.h, ctypes.byref(self.view)) == 0 and self.view.w > 0:
                self.L.mr_set_map(self.mr.lib.h, ctypes.byref(self.view))
                did_map = True
                if self.step_i == 0 or not hasattr(self, "T"):
                    self._set_reference()
                if self.limo:
                    self._limo_pose_log()
                    self._push_objects()
        t2 = time.perf_counter()
        if self.step_i % 3 == 0:
            self._robot()
            self._contacts()
        t3 = time.perf_counter()
        self._push_gt_pose_to_move_robot()
        if self.limo_vla:
            # LIMO VLA step: libmove_robot runs policy + safety filter + end conditions; move_robot's own tick is not called
            # (the same Robot integrates odometry inside mr_vla_tick)
            a, res = self.mr.vla_act(obs)
            if res is not None:
                self.limo_vla = False
                self._vla_reply(res)
        else:
            a = self.mr.act(obs)
        t4 = time.perf_counter()
        T = self.timing
        T["mem"] += t1 - t0
        T["map"] += t2 - t1
        T["contacts"] += t3 - t2
        T["mr"] += t4 - t3
        T["n"] += 1
        T["map_n"] += did_map
        if self.step_i % 30 == 0:
            n = self.L.mr_overlay_json(self.mr.lib.h, self.obuf, len(self.obuf))
            if n > 0:
                p = self.out / "memory" / "explore.json"
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(self.obuf.value)
                os.replace(tmp, p)
        if T["n"] >= 300:
            row = {k: round(1e3 * T[k] / T["n"], 3) for k in ("mem", "map", "contacts", "mr", "between")}
            row.update(step=self.step_i, map_per_call_ms=round(1e3 * T["map"] / max(1, T["map_n"]), 3))
            self.tim_log.write(json.dumps(row) + "\n")
            self.tim_log.flush()
            for k in T:
                T[k] = 0
        self.step_i += 1
        self.t_last = time.perf_counter()
        return a

    def reset(self):
        self.mr.reset()

    def close(self):
        try:
            print(f"[sgrt] {self.mem.stats()}", flush=True)
            self.mem.close()
        finally:
            self.mr.close()


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:8771")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--vla-policy", default=os.environ.get("MR_VLA_POLICY"), help="LIMO VLA stand-in: scripted | replay:<jsonl> (default scripted)")
    ap.add_argument("--robot", default=os.environ.get("SGRT_ROBOT") or "limo_omx", choices=["limo_omx"])
    ap.add_argument("--limo-shim", default=os.environ.get("LIMO_SHIM") or str(pathlib.Path(__file__).resolve().parents[3] / "src/robot/og/eval_with_limo.py"))
    args = ap.parse_args(argv[:split])
    eval_args = argv[split + 1:]
    if "--policy" in eval_args:
        i = eval_args.index("--policy")
        del eval_args[i:i + 2]
    task = eval_args[eval_args.index("--task-name") + 1]
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from move_robot_sim import TcpSource
    import omnigibson.eval.policies as P

    host, port = args.listen.rsplit(":", 1)
    src = TcpSource(host, int(port))
    orig_init = P.LocalPolicy.__init__
    holder = {}

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.policy = ExplorePolicy(src, task, out, lib_path=args.lib, gt_dir=args.gt_dir, robot="limo_omx",
                                    vla_policy=args.vla_policy)
        holder["p"] = self.policy

    P.LocalPolicy.__init__ = init
    if "--robot-config" not in eval_args:
        raise SystemExit("[explore] needs --robot-config <robot-agent>/src/robot/og/limo_omx_eval.yaml in the eval args")
    os.environ["SGRT_ROBOT"] = "limo_omx"
    print(f"[explore] robot limo_omx via {args.limo_shim}", flush=True)
    sys.argv = [args.limo_shim, *eval_args, "--policy", "local", "--output-dir", str(out)]
    try:
        runpy.run_path(args.limo_shim, run_name="__main__")
    finally:
        if "p" in holder:
            holder["p"].close()


if __name__ == "__main__":
    main()
