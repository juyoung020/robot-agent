"""E0 시뮬 측정(헤드리스 OmniGibson, 빈 장면): 잡는 점 위치 / 그리퍼 폭 / 집는 높이 / 하중 / 문턱.
  flock /tmp/claude-1000/og.lock python sim_grasp.py <out.json> [묶음: verify point width height payload quick step]
  (손 링크 질량을 바꿔 하중을 싣는 방법은 안 된다: OmniGibson 은 로봇 링크 중력을 끈다 → 하중은 진짜 상자로)
  GRIP_KP=1e6 처럼 그리퍼 강성을 바꿔 볼 수 있다(results/ 는 1e6 기준, grip_kp1e4_baseline 만 1e4).
각 시험: 가는 기둥(0.02 m 각, 고정) 위 상자 → 예비 자세(도구 축 뒤로 0.06 m, 열림) → 다가감 → 닫음 → 0.08 m 들기 → 2 s 버팀.
성공 = 상자가 0.06 m 이상 올라가고 버틴 뒤에도 손가락 사이(잡는 점에서 0.03 m 안)."""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import HOME_ARM, make_env  # noqa: E402
import kin  # noqa: E402

OUT = sys.argv[1]
GROUPS = sys.argv[2:] or ["point", "width", "height", "payload"]

# 크기별 상자를 미리 싣는다(실행 중 scale 을 바꾸면 PhysX 질량 뷰가 깨진다)
SIZES = sorted({(s, s, s) for s in (0.02, 0.03, 0.04, 0.045)} | {(w, w, 0.03) for w in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)}
               | {(0.03, 0.03, 0.05)})
PARK = {sz: [3.0 + 0.3 * i, 2.0, 0.1] for i, sz in enumerate(SIZES)}
objs = [dict(type="PrimitiveObject", name="box_%d" % i, primitive_type="Cube", size=1.0, scale=list(sz),
             rgba=[0.9, 0.2, 0.2, 1.0], position=PARK[sz]) for i, sz in enumerate(SIZES)] + [
        dict(type="PrimitiveObject", name="stepbox", primitive_type="Cube", size=1.0, scale=[0.10, 1.0, 1.0],
             fixed_base=True, rgba=[0.5, 0.5, 0.5, 1.0], position=[-3.0, 3.0, -0.6]),
        dict(type="PrimitiveObject", name="pillar", primitive_type="Cube", size=1.0, scale=[0.02, 0.02, 1.0],
             fixed_base=True, rgba=[0.3, 0.3, 0.8, 1.0], position=[2.0, 0, -0.4])]
og, env, r = make_env(objects=objs)
import torch as th  # noqa: E402

env.reset()
scene = env.scene
BOX = {sz: scene.object_registry("name", "box_%d" % i) for i, sz in enumerate(SIZES)}
pillar = scene.object_registry("name", "pillar")
stepbox = scene.object_registry("name", "stepbox")
IA = r.controller_action_idx
ARM = r.arm_control_idx["0"]
GRIP = r.gripper_control_idx["0"]
results = {"meta": {}}
log = open(OUT + ".log", "w")


def P(*a):
    print(*a, file=log, flush=True)


def act(arm, grip):
    a = th.zeros(r.action_dim)
    a[IA["arm_0"]] = th.tensor(arm, dtype=th.float32)
    a[IA["gripper_0"]] = float(grip)
    return a


def step(arm, grip, n=1):
    for _ in range(n):
        env.step(act(arm, grip))


def teleport(arm):
    r.set_joint_positions(th.tensor(arm, dtype=th.float32), indices=ARM)
    r.set_joint_velocities(th.zeros(len(ARM)), indices=ARM)


# 그리퍼 드라이브 바꿔 보기(환경 변수): GRIP_KP(강성), GRIP_MAXF(최대 힘, N·m)
import os  # noqa: E402
GJ = [r.joints[n] for n in ("omx_gripper_joint_1", "omx_gripper_joint_2")]
if os.environ.get("GRIP_KP"):
    GJ[0].stiffness = float(os.environ["GRIP_KP"])
if os.environ.get("GRIP_MAXF"):
    GJ[0].max_effort = float(os.environ["GRIP_MAXF"])
results["grip_drive"] = {"kp": float(GJ[0].stiffness), "kd": float(GJ[0].damping), "max_effort": float(GJ[0].max_effort),
                         "kp2": float(GJ[1].stiffness), "max_effort2": float(GJ[1].max_effort)}
# 자리 잡기
step(HOME_ARM, -1, 60)
bp, bq = r.links["base_link"].get_position_orientation()
bp = bp.numpy().astype(float)
yaw = math.atan2(2 * (bq[3] * bq[2] + bq[0] * bq[1]), 1 - 2 * (bq[1] ** 2 + bq[2] ** 2))
floor_z = float(bp[2]) + kin.FLOOR
results["meta"] = {"grip_drive": results.get("grip_drive"), "base_link": bp.tolist(), "yaw": float(yaw), "floor_z_est": floor_z,
                   "wheel_z": [float(r.links[w].get_position_orientation()[0][2]) for w in
                               ("front_left_wheel_link", "rear_left_wheel_link")]}
P("meta", results["meta"])


def to_world(s, z, psi):
    c, sn = math.cos(yaw), math.sin(yaw)
    lx = kin.J1_AXIS[0] + s * math.cos(psi)
    ly = kin.J1_AXIS[1] + s * math.sin(psi)
    return np.array([bp[0] + c * lx - sn * ly, bp[1] + sn * lx + c * ly, bp[2] + z])


def arm_q(s, z, psi, phi, g):
    sols = kin.ik((kin.J1_AXIS[0] + s, z), phi, g)
    for q in sols:
        if not kin.collides(kin.fk(*q, g=g)[0], psi=psi, hull=True):
            return [psi, q[0], q[1], q[2], 0.0]
    return None


def link5_frame():
    p, q = r.links["omx_link5"].get_position_orientation()
    x, y, z, w = (float(v) for v in q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return p.numpy().astype(float), R


def trial(name, s, zc, psi, phi, size=(0.03, 0.03, 0.03), mass=0.05, g_place=kin.G_DEFAULT, hold_s=2.0, lift=0.08):
    """상자 중심을 (joint1 축에서 거리 s, base_link 기준 높이 zc, 방향 psi)에, 그 점이 link5 x = g_place 에 오게 잡는다"""
    rec = dict(name=name, s=s, zc_floor=zc - kin.FLOOR, psi_deg=math.degrees(psi), phi_deg=math.degrees(phi),
               size=list(size), mass=mass, g_place=g_place)
    q_grasp = arm_q(s, zc, psi, phi, g_place)
    # 예비 자세: 도구 축(평면 (cos φ, −sin φ))을 따라 0.06 m 뒤
    q_pre = arm_q(s - 0.06 * math.cos(phi), zc + 0.06 * math.sin(phi), psi, phi, g_place)
    q_lift = arm_q(s, zc + lift, psi, phi, g_place)
    if q_grasp is None or q_pre is None:
        rec["ok"] = False
        rec["why"] = "IK 없음"
        P(rec)
        return rec
    if q_lift is None:
        q_lift = [q_grasp[0], q_grasp[1] - 0.3, q_grasp[2], q_grasp[3] + 0.3, 0.0]
    # 물체 놓기
    c = to_world(s, zc, psi)
    size = tuple(size)
    cube = BOX[size]
    cube.root_link.mass = mass
    step(HOME_ARM, 1, 1)
    teleport(q_pre)
    pillar.set_position_orientation(position=th.tensor([c[0], c[1], c[2] - size[2] / 2 - 0.5], dtype=th.float32))
    cube.set_position_orientation(position=th.tensor(c, dtype=th.float32), orientation=th.tensor([0, 0, 0, 1.0]))
    cube.set_linear_velocity(th.zeros(3))
    cube.set_angular_velocity(th.zeros(3))
    step(q_pre, 1, 30)
    for t in np.linspace(0, 1, 30):
        step([a + (b - a) * t for a, b in zip(q_pre, q_grasp)], 1, 1)
    step(q_grasp, 1, 15)
    p5, R5 = link5_frame()
    cp = cube.get_position_orientation()[0].numpy().astype(float)
    rec["cube_in_link5_before_close"] = (R5.T @ (cp - p5)).round(4).tolist()
    rec["mass_set"] = float(cube.root_link.mass)
    rec["scale_set"] = [round(float(v), 4) for v in cube.scale]
    rec["link5_err_m"] = float(np.linalg.norm(p5 - to_world(s, zc, psi) + R5[:, 0] * g_place))
    step(q_grasp, -1, 30)
    gq = r.get_joint_positions()[GRIP].numpy().astype(float)
    rec["grip_q_closed"] = gq.round(4).tolist()
    z0 = float(cube.get_position_orientation()[0][2])
    for t in np.linspace(0, 1, 40):
        step([a + (b - a) * t for a, b in zip(q_grasp, q_lift)], -1, 1)
    n_hold = int(hold_s * 30)
    effs, geffs = [], []
    for _ in range(n_hold):
        step(q_lift, -1, 1)
        e = r.get_joint_efforts()
        effs.append(e[ARM].numpy().astype(float))
        geffs.append(e[GRIP].numpy().astype(float))
    p5, R5 = link5_frame()
    cp = cube.get_position_orientation()[0].numpy().astype(float)
    rel = R5.T @ (cp - p5)
    rec["cube_in_link5_after"] = rel.round(4).tolist()
    rec["dz"] = float(cp[2] - z0)
    rec["in_hand"] = bool(abs(rel[1] + 0.00165) < 0.03 and 0.03 < rel[0] < 0.11 and rec["dz"] > 0.01)   # 들렸고 손가락 사이
    rec["ok"] = bool(rec["in_hand"] and rec["dz"] > lift * 0.75)                                      # 0.08 m 다 들림
    E = np.array(effs[n_hold // 2:])
    rec["arm_effort_mean"] = E.mean(0).round(4).tolist()      # N·m, joint1..5
    rec["arm_effort_absmax"] = np.abs(E).max(0).round(4).tolist()
    rec["grip_effort_mean"] = np.array(geffs[n_hold // 2:]).mean(0).round(4).tolist()
    rec["q_lift"] = [round(v, 4) for v in q_lift]
    P(rec)
    # 놓고 돌아가기
    step(q_lift, 1, 15)
    cube.set_position_orientation(position=th.tensor(PARK[size]), orientation=th.tensor([0, 0, 0, 1.0]))
    teleport(HOME_ARM)
    step(HOME_ARM, -1, 10)
    return rec


SIDE = math.pi / 2
edge_side = kin.body_edge(SIDE, hull=True)
edge_front = kin.body_edge(0.0, hull=True)
results["meta"]["edge_side_from_j1"] = edge_side
results["meta"]["edge_front_from_j1"] = edge_front
DOWN, HORIZ = math.pi / 2, 0.0
zf = lambda h: h + kin.FLOOR   # 바닥 기준 높이 → base_link 기준

if "verify" in GROUPS:
    # 고친 eef_link 가 link5 x 0.080 에 있는가 + 그 점에 물체 중심을 두고 잡기
    p5, R5 = link5_frame()
    pe = r.links["eef_link"].get_position_orientation()[0].numpy().astype(float)
    pee = r.links["omx_end_effector_link"].get_position_orientation()[0].numpy().astype(float)
    results["eef_in_link5"] = (R5.T @ (pe - p5)).round(5).tolist()
    results["ee_link_in_link5"] = (R5.T @ (pee - p5)).round(5).tolist()
    results["get_eef_in_link5"] = (R5.T @ (r.get_eef_position().numpy().astype(float) - p5)).round(5).tolist()
    P("eef_in_link5", results["eef_in_link5"], "ee_link", results["ee_link_in_link5"], "get_eef", results["get_eef_in_link5"])
    rows = []
    for w in (0.01, 0.02, 0.03, 0.04):
        rows.append(trial(f"verify top w={w}", edge_side + 0.08, zf(0.15), SIDE, DOWN, size=(w, w, 0.03), g_place=results["eef_in_link5"][0]))
    for m in (0.1, 0.25, 0.5):
        rows.append(trial(f"verify horiz m={m}", edge_side + 0.1, zf(0.30), SIDE, HORIZ, size=(0.03, 0.03, 0.05), mass=m,
                          g_place=results["eef_in_link5"][0]))
    results["verify"] = rows
if "quick" in GROUPS:
    results["quick"] = [trial(f"quick m={m}", edge_side + 0.08, zf(0.15), SIDE, DOWN, size=(0.03, 0.03, 0.03), mass=m)
                        for m in (0.05, 0.25)]
if "point" in GROUPS:
    rows = []
    for g in (0.055, 0.065, 0.075, 0.085, 0.0919):
        for sz in (0.02, 0.03, 0.045):
            if os.environ.get('POINT_FAST') and sz == 0.045 and g in (0.055, 0.065):
                continue
            rows.append(trial(f"point g={g} w={sz}", edge_side + 0.08, zf(0.15), SIDE, DOWN, size=(sz, sz, sz), g_place=g))
    results["point"] = rows
if "width" in GROUPS:
    rows = []
    for w in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08):
        rows.append(trial(f"width {w}", edge_side + 0.08, zf(0.15), SIDE, DOWN, size=(w, w, 0.03)))   # 손가락 사이 방향 = 옆
    results["width"] = rows
if "height" in GROUPS:
    rows = []
    # 위에서: 바닥 근처 ~ 최고, 옆으로 / 앞으로
    for psi, edge, lab in ((SIDE, edge_side, "side"), (0.0, edge_front, "front")):
        for d in (0.03, 0.08):
            for h in (0.03, 0.15, 0.22, 0.26):
                rows.append(trial(f"top {lab} d={d} h={h}", edge + d, zf(h), psi, DOWN))
            for h in (0.30, 0.40, 0.45, 0.48, 0.51):
                rows.append(trial(f"horiz {lab} d={d} h={h}", edge + d, zf(h), psi, HORIZ, size=(0.03, 0.03, 0.05)))
    results["height"] = rows
if "payload" in GROUPS:
    rows = []
    for m in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        rows.append(trial(f"payload near m={m}", edge_side + 0.05, zf(0.15), SIDE, DOWN, size=(0.03, 0.03, 0.03), mass=m))
        rows.append(trial(f"payload far m={m}", edge_side + 0.25, zf(0.25), SIDE, HORIZ, size=(0.03, 0.03, 0.05), mass=m))
    results["payload"] = rows
if "step" in GROUPS:
    # 문턱(마지막에, 로봇이 움직이므로): 로봇 앞 0.4 m 에 높이 h 판(가로 1 m, 깊이 0.1 m)을 두고 0.3 m/s 로 8 s 전진 → 넘었는지(base x 이동, 기울기)
    rows = []
    pillar.set_position_orientation(position=th.tensor([2.0, 5.0, -0.4]))
    for h in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        teleport(HOME_ARM)
        step(HOME_ARM, -1, 15)
        p0 = r.links["base_link"].get_position_orientation()[0]
        x0, y0 = float(p0[0]), float(p0[1])
        stepbox.set_position_orientation(position=th.tensor([x0 + 0.40, y0, h - 0.5]))
        maxtilt = 0.0
        for _ in range(240):
            a = act(HOME_ARM, -1)
            a[IA["base"][0]] = 0.3
            env.step(a)
            qq = r.links["base_link"].get_position_orientation()[1]
            maxtilt = max(maxtilt, math.degrees(2 * math.asin(min(1.0, math.hypot(float(qq[0]), float(qq[1]))))))
        p1 = r.links["base_link"].get_position_orientation()[0]
        rows.append(dict(h=h, dx=float(p1[0]) - x0, z_end=float(p1[2]), crossed=bool(float(p1[0]) > x0 + 0.40 + 0.05 + 0.15),
                         max_tilt_deg=maxtilt))
        P(rows[-1])
        stepbox.set_position_orientation(position=th.tensor([-3.0, 3.0, -0.6]))
        step(HOME_ARM, -1, 30)
    results["step"] = rows
json.dump(results, open(OUT, "w"), indent=1, ensure_ascii=False)
P("done")
log.close()
og.shutdown()
