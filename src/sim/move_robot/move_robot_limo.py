"""move_robot for our LIMO + OMX-F (OmniGibson model "limo_omx", robot-agent src/robot/og/limo_omx_eval.yaml).

The closed-loop executor (libmove_robot, robot-agent src/agent/tools/move_robot) speaks R1 Pro only: proprio 61 in,
action 23 out. Its base part (go_to / probe / delta, odometry from base_qvel, map, safety stop) is robot independent,
so for LIMO this adapter runs it unchanged and translates bytes on both sides:

    proprio : LIMO evaluator proprio (base_qvel 3, arm_0_qpos 5, ..., 24) -> R1 61 with only base_qvel [0:3] filled
              (torso / arms / grippers 0; move_robot holds whatever it saw first and never moves them here)
    action  : R1 23 base [0:3] (normalized by BASE_OUT 0.75, 0.75, 1.0) -> m/s, rad/s -> LIMO 9:
              base [0:3] = holonomic vx, vy, wz normalized by limo_omx_eval.yaml command_output_limits (0.5 m/s, 0.8727 rad/s),
              arm_0 [3:8] = reset_joint_pos (home [0, -1.6, 1.45, 0.15, 0], absolute position control),
              gripper_0 [8] = -1 (closed = home)
    calls   : only part "base" reaches move_robot; any other part gets an error (the LIMO arm is not driven by explore)

Odometry check (10-04, headless probe, turning_on_radio): LIMO base_qvel (base frame, vx/vy/wz) integrated at 1/30 s matches the
GT base motion within 1 % (3.00 m vs 2.98 m, 172 vs 171 deg over forward / spin / arc / reverse), so this adapter passes it
unchanged. The ~16 % short path_m of earlier runs was libmove_robot dropping the distance of the step after every map update
(1 of 6 keyframe steps), fixed in robot-agent move_robot (NavState::integrate).

VLA executor (robot-agent docs/map_vla/POLICY.md 1.3, libmove_robot mr_vla_*): {"executor":"vla","skill","objects":[ids],"max_s"}
calls run through vla_start / vla_act: LIMO proprio (24, raw) -> mr_vla_tick (policy + safety filter + end / verify / budget in
Rust) -> action 8 [vx, wz, j1..j5, gripper 0..1] -> LIMO 9 (vy 0). After a VLA step the arm and gripper hold its last command
(so a picked object stays lifted) instead of snapping back home.

Slots come from the sim robot (controller_action_idx, _proprio_obs), not hard-coded, with the yaml order as fallback.
The R1 path (move_robot_sim.MoveRobotPolicy) is not touched.
"""
import json
import os

from move_robot_sim import ACTION_DIM, PROPRIO_DIM, MoveRobotPolicy

R1_BASE_OUT = (0.75, 0.75, 1.0)          # libmove_robot act::BASE_OUT (action [-1, 1] -> m/s, m/s, rad/s)
LIMO_BASE_OUT = (0.5, 0.5, 0.8727)       # limo_omx_eval.yaml base command_output_limits
LIMO_HOME_ARM = (0.0, -1.6, 1.45, 0.15, 0.0)
LIMO_BASE_LINK_Z = 0.15                  # base_footprint -> base_link (limo_four_diff.xacro base_joint)
LIMO_PROPRIO_OBS = [("base_qvel", 3), ("arm_0_qpos", 5), ("arm_0_qvel", 5), ("eef_0_pos", 3), ("eef_0_quat", 4),
                    ("gripper_0_qpos", 2), ("gripper_0_qvel", 2)]
LIMO_ACTION_IDX = {"base": [0, 1, 2], "arm_0": [3, 4, 5, 6, 7], "gripper_0": [8]}


def find_robot():
    try:
        import omnigibson as og
        for sc in og.sim.scenes:
            if sc.robots:
                return sc.robots[0]
    except Exception:
        pass
    return None


def apply_near_clip(robot):
    """Safety net for older robot assets: before robot-agent 391c04b the imported limo_omx put robot_limo:eyes ~1 cm behind
    the chassis shell's front face, so unclipped the body camera saw only its own shell (depth ~0.0096 m). Since 391c04b the
    camera sits at the lens and eval_with_limo.py sets near clip 0.05 m itself, so this is a no-op there: it only RAISES the
    near plane to LIMO_NEAR_CLIP m (default 0.05, 0 = never touch), never lowers it. Returns [(sensor, old, new)]."""
    near = float(os.environ.get("LIMO_NEAR_CLIP", "0.05"))
    done = []
    if near <= 0 or robot is None:
        return done
    for n, sen in robot.sensors.items():
        if ":eyes:" in n and hasattr(sen, "clipping_range"):
            old = [round(float(v), 4) for v in sen.clipping_range]
            if old[0] >= near:
                print(f"[limo] {n} clipping range {old} (already >= {near}, left as is)", flush=True)
                continue
            sen.clipping_range = (near, max(near * 10, float(old[1])))
            new = [round(float(v), 4) for v in sen.clipping_range]
            print(f"[limo] {n} clipping range {old} -> {new}", flush=True)
            done.append((n, old, new))
    return done


class BaseOnlySource:
    """Passes base calls through; answers other parts itself (LIMO arm/gripper are held at home in explore)."""

    def __init__(self, inner):
        self.inner = inner

    def poll(self):
        c = self.inner.poll()
        if isinstance(c, dict) and c.get("part", "base") != "base":
            self.inner.reply({"status": "error", "part": c.get("part"),
                              "message": "LIMO explore: only part 'base' is driven (arm and gripper are held at home)",
                              "hint": "use part base (go_to / probe / delta)"})
            return None
        return c

    def reply(self, r):
        self.inner.reply(r)

    def close(self):
        self.inner.close()

    @property
    def done(self):
        return getattr(self.inner, "done", False)


class LimoMoveRobotPolicy(MoveRobotPolicy):
    """MoveRobotPolicy for limo_omx: same tool calls / results / map hand-off, LIMO proprio in, LIMO action out."""

    def __init__(self, source, lib_path=None, log_path=None, hz=30.0):
        super().__init__(BaseOnlySource(source), lib_path=lib_path, log_path=log_path, hz=hz)
        self.robot = None
        self.qvel = None          # slice of base_qvel in the evaluator proprio
        self.idx = None           # controller_action_idx
        self.act_dim = None
        self.arm_hold = None
        self.grip_hold = -1.0     # gripper action [-1, 1] held while move_robot drives the base
        self.r1 = None
        self.row = None           # last raw LIMO proprio (VLA executor input)

    def _setup(self, n_prop):
        import numpy as np
        r = self.robot = self.robot or find_robot()
        lay, i = {}, 0
        keys = None
        if r is not None:
            try:
                d = r._get_proprioception_dict()
                keys = [(k, int(d[k].numel())) for k in r._proprio_obs]
            except Exception:
                keys = None
        for k, n in keys or LIMO_PROPRIO_OBS:
            lay[k] = (i, i + n)
            i += n
        if i != n_prop or "base_qvel" not in lay:
            raise RuntimeError(f"[limo] proprio layout {lay} ({i}) != observation {n_prop}")
        self.qvel = lay["base_qvel"]
        self.lay = lay
        if r is not None:
            self.idx = {k: [int(x) for x in (v.tolist() if hasattr(v, "tolist") else v)] for k, v in r.controller_action_idx.items()}
            self.act_dim = int(r.action_dim)
            self.arm_hold = [float(x) for x in r.reset_joint_pos[r.arm_control_idx["0"]]]
        else:
            self.idx, self.act_dim, self.arm_hold = LIMO_ACTION_IDX, 9, list(LIMO_HOME_ARM)
        self.r1 = np.zeros(PROPRIO_DIM, np.float32)
        print(f"[limo] move_robot adapter: proprio {n_prop} (base_qvel {self.qvel}), action {self.act_dim} {self.idx}, "
              f"arm hold {[round(x, 3) for x in self.arm_hold]}", flush=True)

    def _proprio(self, obs):
        """LIMO proprio -> R1 61 (base_qvel only) for libmove_robot."""
        row, batched = super()._proprio(obs)
        if self.r1 is None:
            self._setup(len(row))
        self.row = row
        a, b = self.qvel
        v = row[a:b]
        self.r1[0], self.r1[1], self.r1[2] = float(v[0]), float(v[1]), float(v[-1])   # vx, vy, wz (base frame)
        return self.r1, batched

    def limo_action(self, r1_action):
        """R1 23 (list / tensor / batched) -> LIMO action of the robot's dim, same batching."""
        import torch
        t = r1_action if isinstance(r1_action, torch.Tensor) else torch.tensor(r1_action, dtype=torch.float32)
        batched = t.ndim == 2
        r1 = t[0] if batched else t
        out = torch.zeros(self.act_dim, dtype=torch.float32)
        for k in range(3):
            phys = float(r1[k]) * R1_BASE_OUT[k]
            out[self.idx["base"][k]] = max(-1.0, min(1.0, phys / LIMO_BASE_OUT[k]))
        out[self.idx["arm_0"]] = torch.tensor(self.arm_hold, dtype=torch.float32)
        out[self.idx["gripper_0"]] = self.grip_hold
        return out[None] if batched else out

    def act(self, obs):
        return self.limo_action(super().act(obs))

    # ---------- VLA executor (LIMO + OMX-F)
    def vla_start(self, call):
        """Start a VLA step in libmove_robot. Returns the result dict if it ended at once (error / handback), else None."""
        if not self.lib.has_vla:
            return {"status": "unavailable", "executor": "vla", "message": "libmove_robot has no VLA executor (old build)"}
        if self.lib.vla_start(call) == 1:
            return self.lib.take_result()
        return None

    def vla_act(self, obs):
        """One step of the running VLA step: (LIMO action, result or None). The action already passed the safety filter."""
        import torch
        row, batched = self._proprio(obs)
        if self.r1 is None:
            self._setup(len(row))
        row = [float(x) for x in self.row]
        if "eef_0_pos" in self.lay:
            # OmniGibson eef_0_pos is relative to base_footprint_link_name = base_link (limo_omx.yaml); libmove_robot wants
            # base_footprint (floor), as the object memory: base_joint z 0.15 (limo_four_diff.xacro)
            row[self.lay["eef_0_pos"][0] + 2] += LIMO_BASE_LINK_Z
        rc, a8 = self.lib.vla_tick(row)
        out = torch.zeros(self.act_dim, dtype=torch.float32)
        out[self.idx["base"][0]] = max(-1.0, min(1.0, a8[0] / LIMO_BASE_OUT[0]))
        out[self.idx["base"][2]] = max(-1.0, min(1.0, a8[1] / LIMO_BASE_OUT[2]))
        out[self.idx["arm_0"]] = torch.tensor(a8[2:7], dtype=torch.float32)
        g = max(-1.0, min(1.0, 2.0 * a8[7] - 1.0))
        out[self.idx["gripper_0"]] = g
        res = None
        if rc == 2:
            res = self.lib.take_result()
            self.arm_hold = [float(x) for x in a8[2:7]]
            self.grip_hold = g
        return (out[None] if batched else out), res


def _selftest():
    """No sim: fake obs through the adapter with the real libmove_robot (MOVE_ROBOT_LIB), base delta forward."""
    import numpy as np
    import torch
    from move_robot_sim import ScriptSource
    src = ScriptSource([{"part": "left_arm", "mode": "delta", "values": [10, 0, 0, 0, 0, 0, 0]},
                        {"part": "base", "mode": "delta", "values": [0.3, 0, 0]}])
    p = LimoMoveRobotPolicy(src)
    prop = np.zeros(24, np.float32)
    vx = 0.0
    for i in range(200):
        prop[0] = vx
        a = p.act({"robot_limo::proprio": torch.tensor(prop)[None]})
        assert a.shape == (1, 9), a.shape
        vx = float(a[0, 0]) * LIMO_BASE_OUT[0]   # perfect velocity tracking
        assert abs(float(a[0, 1])) < 1e-6 and torch.allclose(a[0, 3:8], torch.tensor(LIMO_HOME_ARM)) and float(a[0, 8]) == -1.0
        if src.done and not p.lib.busy() and len(src.results) == 2:
            break
    print(json.dumps(src.results))
    assert src.results[0]["status"] == "error" and src.results[1]["status"] == "reached", src.results
    # VLA executor: an object within reach, scripted stand-in policy, approach -> done (end signal + verify)
    if p.lib.has_vla:
        p.lib.L.mr_vla_set_objects_json(p.lib.h, json.dumps([{"id": "O7", "pos": [0.30, 0.0, 0.1], "extent": [0.06, 0.06, 0.1],
                                                               "last_seen": 0.0}]).encode(), 0.0)
        assert p.vla_start({"executor": "vla", "skill": "move to table", "objects": ["O7"]})["route"] == "move_robot"
        assert p.vla_start({"executor": "vla", "skill": "approach cup", "objects": ["O7"], "max_s": 10}) is None
        prop[:] = 0
        prop[3:8] = LIMO_HOME_ARM
        res = None
        for i in range(400):
            a, res = p.vla_act({"robot_limo::proprio": torch.tensor(prop)[None]})
            assert a.shape == (1, 9), a.shape
            prop[0] = float(a[0, 0]) * LIMO_BASE_OUT[0]
            prop[2] = float(a[0, 2]) * LIMO_BASE_OUT[2]
            prop[3:8] = a[0, 3:8].numpy()
            if res is not None:
                break
        print(json.dumps(res))
        for k in ("status", "reason", "evidence", "steps", "min_clear_m", "contacts"):
            assert k in res, k
        assert res["status"] == "done", res
    p.close()
    print("ok")


if __name__ == "__main__":
    _selftest()
