"""평가기를 우리 로봇(limo_omx)으로 띄우는 얇은 진입점(보기용). OmniGibson 소스는 건드리지 않는다.
시작 위치 파일(robot_poses)은 R1 Pro 용 키만 있어서, 모델 이름이 없으면 첫 항목을 일반 키 "robot" 으로 대신 쓴다.
  python eval_with_limo.py <omnigibson.eval.eval 인자들>
"""
import runpy
import sys

import omnigibson.scenes.scene_base as sb

_orig = sb.Scene.get_task_metadata


def _patched(self, key, *a, **k):
    r = _orig(self, key, *a, **k)
    if key == "robot_poses" and isinstance(r, dict) and "robot" not in {x.lower() for x in r}:
        r = dict(r)
        r["robot"] = next(iter(r.values()))
    return r


sb.Scene.get_task_metadata = _patched

# 로컬 정책(--policy local)은 행동을 모두 0 으로 낸다. 팔은 절대 위치 제어라 0 이면 팔이 관절 0(앞으로 뻗은 L 자세)으로
# 가 버리므로, 보기용으로 팔 행동만 리셋 자세(reset_joint_pos = 접은 자세)로 채워 그대로 들고 있게 한다.
import omnigibson.eval.evaluator as _ev

_orig_load_policy = _ev.BatchedEvaluator.load_policy


def _load_policy(self):
    pol = _orig_load_policy(self)
    if self.cfg.policy_name == "local" and getattr(pol, "policy", None) is None:
        robot = self.instance_eval_states[0].env_accessor.robot
        holds = []
        for arm in robot.arm_names:
            key = "arm_" + arm
            if key in robot.controller_action_idx:
                holds.append((robot.controller_action_idx[key], robot.reset_joint_pos[robot.arm_control_idx[arm]].clone()))
        fwd = pol.forward

        def forward(obs, *a, **k):
            out = fwd(obs, *a, **k)
            for idx, q in holds:
                out[..., idx] = q.to(out.dtype)
            return out

        pol.forward = forward
    return pol


_ev.BatchedEvaluator.load_policy = _load_policy
sys.argv = ["omnigibson.eval.eval"] + sys.argv[1:]
runpy.run_module("omnigibson.eval.eval", run_name="__main__")
