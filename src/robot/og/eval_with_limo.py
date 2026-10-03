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
sys.argv = ["omnigibson.eval.eval"] + sys.argv[1:]
runpy.run_module("omnigibson.eval.eval", run_name="__main__")
