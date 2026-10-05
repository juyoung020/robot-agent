"""move_robot for our LIMO + OMX-F (OmniGibson model "limo_omx", robot-agent src/robot/og/limo_omx_eval.yaml).

The closed-loop executor (libmove_robot, robot-agent src/agent/tools/move_robot) speaks LIMO + OMX-F directly: proprio 24 in
(base_qvel 3, arm_0_qpos 5, arm_0_qvel 5, eef_0_pos 3, eef_0_quat 4, gripper_0_qpos 2, gripper_0_qvel 2), action 8 out
([vx m/s, wz rad/s, omx_joint1..5 rad, gripper 0..1]). This adapter only places that vector into the sim robot's own action vector
(limo_omx_eval.yaml controller_action_idx, 9 slots: base [0:3] = vx, vy, wz, arm_0 [3:8], gripper_0 [8]):

    base     : vx, wz normalized by command_output_limits (0.5 m/s, 0.8727 rad/s); the lateral slot vy is always 0 (differential drive)
    arm_0    : joint targets as they are (absolute position control)
    gripper_0: opening 0..1 -> [-1, 1] (-1 closed, +1 open 100 deg)

Odometry check (10-04, headless probe, turning_on_radio): LIMO base_qvel (base frame, vx/wz) integrated at 1/30 s matches the
GT base motion within 1 % (3.00 m vs 2.98 m, 172 vs 171 deg over forward / spin / arc / reverse).

VLA executor (robot-agent docs/map_vla/POLICY.md 1.3, libmove_robot mr_vla_*): {"executor":"vla","skill","objects":[ids],"max_s"}
calls run through vla_start / vla_act: LIMO proprio (24, raw) -> mr_vla_tick (policy + safety filter + end / verify / budget in
Rust) -> action 8. After a VLA step libmove_robot itself holds the arm and gripper at its last command (a picked object stays lifted).

Slots come from the sim robot (controller_action_idx, _proprio_obs), not hard-coded, with the yaml order as fallback.
"""
import json
import os

from move_robot_sim import ACTION_DIM, PROPRIO_DIM, MoveRobotPolicy

LIMO_BASE_OUT = (0.5, 0.5, 0.8727)       # limo_omx_eval.yaml base command_output_limits (vx, vy, wz)
LIMO_BASE_LINK_Z = 0.15                  # base_footprint -> base_link (limo_four_diff.xacro base_joint)
LIMO_PROPRIO_OBS = [("base_qvel", 3), ("arm_0_qpos", 5), ("arm_0_qvel", 5), ("eef_0_pos", 3), ("eef_0_quat", 4),
                    ("gripper_0_qpos", 2), ("gripper_0_qvel", 2)]
LIMO_ACTION_IDX = {"base": [0, 1, 2], "arm_0": [3, 4, 5, 6, 7], "gripper_0": [8]}
LIMO_ACT_DIM = 9
LIMO_HOME_ARM = (0.0, -1.6, 1.45, 0.15, 0.0)   # limo_omx_eval.yaml reset_joint_pos


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


class LimoMoveRobotPolicy(MoveRobotPolicy):
    """MoveRobotPolicy for limo_omx: libmove_robot's 8-vector action goes into the sim robot's action vector."""

    def __init__(self, source, lib_path=None, log_path=None, hz=30.0):
        super().__init__(source, lib_path=lib_path, log_path=log_path, hz=hz)
        self.robot = None
        self.idx = None           # controller_action_idx
        self.act_dim = None
        self.lay = None           # proprio layout {name: (start, end)}

    def _setup(self, n_prop):
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
        if i != n_prop or n_prop != PROPRIO_DIM or lay.get("base_qvel") != (0, 3):
            raise RuntimeError(f"[limo] proprio layout {lay} ({i}) != observation {n_prop} (libmove_robot wants {PROPRIO_DIM}: limo.rs)")
        self.lay = lay
        if r is not None:
            self.idx = {k: [int(x) for x in (v.tolist() if hasattr(v, "tolist") else v)] for k, v in r.controller_action_idx.items()}
            self.act_dim = int(r.action_dim)
        else:
            self.idx, self.act_dim = LIMO_ACTION_IDX, LIMO_ACT_DIM
        print(f"[limo] move_robot adapter: proprio {n_prop}, action {self.act_dim} {self.idx}", flush=True)

    def _proprio(self, obs):
        row, batched = super()._proprio(obs)
        if self.lay is None:
            self._setup(len(row))
        return row, batched

    def sim_action(self, a8):
        """libmove_robot action 8 [vx, wz, j1..j5, gripper 0..1] (list / tensor / batched) -> the sim robot's action, same batching."""
        import torch
        t = a8 if isinstance(a8, torch.Tensor) else torch.tensor(a8, dtype=torch.float32)
        batched = t.ndim == 2
        a = (t[0] if batched else t).float()
        out = torch.zeros(self.act_dim, dtype=torch.float32)
        out[self.idx["base"][0]] = max(-1.0, min(1.0, float(a[0]) / LIMO_BASE_OUT[0]))
        out[self.idx["base"][2]] = max(-1.0, min(1.0, float(a[1]) / LIMO_BASE_OUT[2]))
        out[self.idx["arm_0"]] = a[2:7]
        out[self.idx["gripper_0"]] = max(-1.0, min(1.0, 2.0 * float(a[7]) - 1.0))
        return out[None] if batched else out

    def act(self, obs):
        return self.sim_action(super().act(obs))

    # ---------- VLA executor (LIMO + OMX-F)
    def vla_start(self, call):
        """Start a VLA step in libmove_robot. Returns the result dict if it ended at once (error / handback), else None."""
        if not self.lib.has_vla:
            return {"status": "unavailable", "executor": "vla", "message": "libmove_robot has no VLA executor (old build)"}
        if self.lib.vla_start(call) == 1:
            return self.lib.take_result()
        return None

    def vla_act(self, obs):
        """One step of the running VLA step: (sim action, result or None). The action already passed the safety filter."""
        row, batched = self._proprio(obs)
        row = [float(x) for x in row]
        if "eef_0_pos" in self.lay:
            # OmniGibson eef_0_pos is relative to base_footprint_link_name = base_link (limo_omx.yaml); libmove_robot wants
            # base_footprint (floor), as the object memory: base_joint z 0.15 (limo_four_diff.xacro)
            row[self.lay["eef_0_pos"][0] + 2] += LIMO_BASE_LINK_Z
        rc, a8 = self.lib.vla_tick(row)
        res = self.lib.take_result() if rc == 2 else None
        out = self.sim_action(a8)
        return (out[None] if batched else out), res


def _selftest():
    """No sim: fake obs through the adapter with the real libmove_robot (MOVE_ROBOT_LIB): arm delta, base delta forward, VLA approach."""
    import numpy as np
    import torch
    from move_robot_sim import ScriptSource
    src = ScriptSource([{"part": "arm", "mode": "delta", "values": [10, 0, 0, 0, 0]},
                        {"part": "base", "mode": "delta", "values": [0.3, 0]},
                        {"part": "torso", "mode": "delta", "values": [1]}])
    p = LimoMoveRobotPolicy(src)
    prop = np.zeros(PROPRIO_DIM, np.float32)
    prop[3:8] = (0.0, -1.6, 1.45, 0.15, 0.0)          # LIMO home
    vx = 0.0
    for i in range(400):
        prop[0] = vx
        a = p.act({"robot_limo::proprio": torch.tensor(prop)[None]})
        assert a.shape == (1, 9), a.shape
        vx = float(a[0, 0]) * LIMO_BASE_OUT[0]   # perfect velocity tracking
        prop[3:8] = a[0, 3:8].numpy()              # perfect arm tracking
        assert abs(float(a[0, 1])) < 1e-6, "differential drive: no lateral velocity"
        if src.done and not p.lib.busy() and len(src.results) == 3:
            break
    print(json.dumps(src.results))
    assert [r["status"] for r in src.results] == ["reached", "reached", "error"], src.results
    assert abs(float(a[0, 3]) - 10 * 3.14159265 / 180) < 0.02, "arm joint 1 moved +10 deg and is held"
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
