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

# 평가기는 머리 카메라(eval.camera_sensor_names.head = eyes)의 sensor_config 를 링크 키("eyes:Camera:0")로 해상도만 넣어
# 덮어쓰므로 limo_omx_eval.yaml 의 VisionSensor.sensor_kwargs.clipping_range 가 eyes 에는 안 들어간다(손목에는 들어감).
# 게다가 머리 카메라 조리개를 R1 값(40)으로 넓혀 화각이 커져 앞 범퍼가 화면 아래에 찍힌다 → 로드 뒤 우리 로봇의 모든
# 카메라에 yaml 의 가까운 자르기를 다시 넣는다.
_orig_apply_settings = _ev.BatchedEvaluator._apply_robot_eval_settings


def _apply_robot_eval_settings(self):
    _orig_apply_settings(self)
    from omegaconf import OmegaConf
    from omnigibson.sensors.vision_sensor import VisionSensor

    clip = OmegaConf.select(self.cfg, "robot.sensor_config.VisionSensor.sensor_kwargs.clipping_range")
    if clip is None:
        return
    for st in self.instance_eval_states:
        robot = st.env_accessor.robot
        if robot.model != "limo_omx":
            continue
        for sen in robot.sensors.values():
            if isinstance(sen, VisionSensor):
                sen.clipping_range = tuple(float(c) for c in clip)


_ev.BatchedEvaluator._apply_robot_eval_settings = _apply_robot_eval_settings

# 사람 시연 통계(human_stats)는 R1 의 두 팔(left/right) 키만 있어 우리 팔 이름 "0" 으로 정규화하면 에피소드 끝에서
# KeyError '0' 이 난다. 사람 기록이 없는 팔은 정규화 거리(normalized_agent_distance)에서만 빼고, 원 거리(agent_distance)는 그대로 남긴다.
import omnigibson.metrics.agent_metric as _am

_orig_episode_metrics = _am.AgentMetric._compute_episode_metrics


def _episode_metrics(self, env, episode_info):
    hs = self.human_stats
    if hs is None:
        all_d = episode_info.get("delta_agent_distance", self.delta_agent_distance)
        return {"agent_distance": {k: sum(v) for k, v in all_d.items()}}
    all_d = episode_info.get("delta_agent_distance", self.delta_agent_distance)
    missing = [k for k in all_d if k not in hs]
    if not missing:
        return _orig_episode_metrics(self, env, episode_info)
    kept = {k: v for k, v in all_d.items() if k in hs}
    r = _orig_episode_metrics(self, env, {**episode_info, "delta_agent_distance": kept})
    r["agent_distance"] = {k: sum(v) for k, v in all_d.items()}
    r["normalized_agent_distance_skipped"] = missing   # 사람 통계에 없는 팔(예: "0")
    return r


_am.AgentMetric._compute_episode_metrics = _episode_metrics
sys.argv = ["omnigibson.eval.eval"] + sys.argv[1:]
runpy.run_module("omnigibson.eval.eval", run_name="__main__")
