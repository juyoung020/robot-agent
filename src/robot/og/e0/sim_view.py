"""E0 카메라 보기: 몸통 카메라(eyes, 평가기처럼 H-FOV 67.9° 정사각)·손목 카메라(wrist_eye)에 낮은 가구 위 물체가 보이는가.
탁자(윗면 0.6 x 0.6 m, 높이 h)를 로봇 앞 범퍼에서 D 만큼 떨어뜨려 두고, 윗면 가장자리에서 0.10 m 안쪽에 0.05 m 빨간 상자를 둔다.
빨간 화소 수 > 30 이면 보임. python sim_view.py <out.json> [png 폴더]"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import HOME_ARM, make_env  # noqa: E402
import kin  # noqa: E402

OUT = sys.argv[1]
PNG = sys.argv[2] if len(sys.argv) > 2 else None
HS = (0.30, 0.45, 0.60, 0.75)
objs = [dict(type="PrimitiveObject", name="t%d" % i, primitive_type="Cube", size=1.0, scale=[0.6, 0.6, h],
             fixed_base=True, rgba=[0.6, 0.5, 0.35, 1.0], position=[5.0 + 2 * i, 5.0, h / 2]) for i, h in enumerate(HS)]
objs.append(dict(type="PrimitiveObject", name="obj", primitive_type="Cube", size=1.0, scale=[0.05, 0.05, 0.05],
                 rgba=[1.0, 0.0, 0.0, 1.0], position=[5.0, -5.0, 0.025]))
og, env, r = make_env(objects=objs, obs=("rgb",))
import torch as th  # noqa: E402

env.reset()
# eval_with_limo.set_eyes_hfov 와 같은 식(그 모듈은 평가기를 import 해서 실행 뒤엔 못 부른다): H-FOV 67.9°(Dabai 깊이)
for n, s in r.sensors.items():
    if ":eyes:" in n:
        s.horizontal_aperture = 2.0 * float(s.focal_length) * math.tan(math.radians(67.9) / 2.0)
IA = r.controller_action_idx
ARM = r.arm_control_idx["0"]
tables = [env.scene.object_registry("name", "t%d" % i) for i in range(len(HS))]
obj = env.scene.object_registry("name", "obj")
cams = {n.split(":")[1]: s for n, s in r.sensors.items() if "Camera" in n}
log = open(OUT + ".log", "w")


def P(*a):
    print(*a, file=log, flush=True)


def act(arm):
    a = th.zeros(r.action_dim)
    a[IA["arm_0"]] = th.tensor(arm, dtype=th.float32)
    a[IA["gripper_0"]] = -1.0
    return a


def go(arm, n=40):
    r.set_joint_positions(th.tensor(arm, dtype=th.float32), indices=ARM)
    for _ in range(n):
        env.step(act(arm))


# 보는 자세: 손목 카메라를 높이 들어 앞을 내려다봄(잡는 점 높이 0.5 m, 수평 도구 축, 앞으로)
q_look = None
for z in (0.50, 0.48, 0.45):
    sols = kin.ik((kin.J1_AXIS[0] + kin.body_edge(0.0, True) + 0.02, z + kin.FLOOR), 0.0)
    if sols:
        q_look = [0.0, *sols[0], 0.0]
        break
P("cams", list(cams), "q_look", q_look)
go(HOME_ARM, 60)
bp = r.links["base_link"].get_position_orientation()[0].numpy()
front = float(bp[0]) + kin.FRONT_X
res = []
for i, h in enumerate(HS):
    for D in (0.05, 0.30, 0.70, 1.20):
        x_edge = front + D
        tables[i].set_position_orientation(position=th.tensor([x_edge + 0.3, float(bp[1]), h / 2]))
        obj.set_position_orientation(position=th.tensor([x_edge + 0.10 + 0.025, float(bp[1]), h + 0.025]),
                                     orientation=th.tensor([0, 0, 0, 1.0]))
        for pose_name, arm in (("home", HOME_ARM), ("look", q_look)):
            if arm is None:
                continue
            go(arm, 30)
            for _ in range(3):
                og.sim.render()
            row = dict(h=h, D=D, pose=pose_name)
            for cn, cam in cams.items():
                o = cam.get_obs()[0]
                rgb = o["rgb"].cpu().numpy()[..., :3].astype(int)
                red = int(((rgb[..., 0] > 150) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)).sum())
                row[cn + "_red_px"] = red
                row[cn + "_visible"] = red > 30
                if PNG and D in (0.05, 0.70):
                    from PIL import Image
                    Image.fromarray(rgb.astype(np.uint8)).save(f"{PNG}/{cn}_{pose_name}_h{h:.2f}_D{D:.2f}.png")
            cp = {cn: [round(float(v), 3) for v in cam.get_position_orientation()[0]] for cn, cam in cams.items()}
            row["cam_pos"] = cp
            res.append(row)
            P(row)
        tables[i].set_position_orientation(position=th.tensor([5.0 + 2 * i, 5.0, h / 2]))
json.dump(res, open(OUT, "w"), indent=1)
P("done")
log.close()
og.shutdown()
