"""OMX-F 평면 순기구학(관절 1 = 0, URDF 값)과 LIMO 위 도달 범위 표본. 시뮬 없이 돈다(numpy).
좌표: base_link 원점 기준 x 앞, z 위. 바닥은 z = -0.15(URDF base_joint). 잡는 점 = omx_link5 + (G, 0, 0)."""
import json
import os
import subprocess
import math
import pathlib

import numpy as np

LIM = json.load(open(pathlib.Path(__file__).resolve().parents[2] / "real_limits.json"))
J2, J3, J4 = (LIM["omx_joint%d" % i] for i in (2, 3, 4))
FLOOR = -0.15
SHOULDER = np.array([-0.04 - 0.01125, 0.034 + 0.0635])    # omx_link2 원점 (x, z), base_link 기준
G_DEFAULT = 0.080                                         # 잡는 점(손끝 패드 사이), omx_link5 x
FRONT_X = 0.165                                           # 앞 범퍼(limo_base 메시 x 최대 ≈ 0.163)
J1_AXIS = np.array([-0.04 - 0.01125, 0.0])                # omx_joint1 축 (x, y), base_link 기준
_ROBOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")   # src/robot
BASE_MESH = os.path.join(_ROBOT, "limo_description", "meshes", "limo_base.stl")
_PROFILE = {}


def body_profile(psi=0.0, hull=False, half_w=0.02, step=0.005):
    """joint1 축에서 방향 psi(0 앞, +π/2 왼쪽) 반직선을 따라 몸통 윗면 높이(base_link 기준) 표: (s 배열, z 배열).
    hull=False: 실제 메시(limo_base, 바퀴 포함), hull=True: 시뮬 충돌 모양(메시의 볼록 껍질)."""
    key = (round(psi, 4), hull)
    if key in _PROFILE:
        return _PROFILE[key]
    import trimesh
    m = trimesh.load(BASE_MESH)
    m.apply_scale(0.001)
    if hull:
        m = m.convex_hull
        pts, _ = trimesh.sample.sample_surface(m, 400000, seed=0)
        pts = np.vstack([pts, m.vertices])
    else:
        pts, _ = trimesh.sample.sample_surface(m, 400000, seed=0)
        pts = np.vstack([pts, m.vertices])
    v = np.c_[-pts[:, 1], pts[:, 0], pts[:, 2] - 0.15]          # 시각 원점 rpy (0,0,1.57), z -0.15
    # 바퀴(반지름 0.045, 폭 0.045, 중심 (±0.1, ±0.065~0.11, -0.1))
    wh = []
    for sx in (0.1, -0.1):
        for sy in (1, -1):
            for a in np.linspace(0, 2 * np.pi, 72):
                for yy in np.linspace(0.065, 0.11, 6):
                    wh.append((sx + 0.045 * np.cos(a), sy * yy, -0.1 + 0.045 * np.sin(a)))
    v = np.vstack([v, np.array(wh)])
    d = v[:, :2] - J1_AXIS
    u = np.array([np.cos(psi), np.sin(psi)])
    s = d @ u
    lat = np.abs(d @ np.array([-u[1], u[0]]))
    sel = lat < half_w
    bins = np.arange(-0.3, 0.4, step)
    zs = np.full(len(bins), -np.inf)
    idx = np.digitize(s[sel], bins) - 1
    for i, z in zip(idx, v[sel, 2]):
        if 0 <= i < len(bins) and z > zs[i]:
            zs[i] = z
    _PROFILE[key] = (bins, zs)
    return bins, zs


def body_edge(psi=0.0, hull=False):
    """반직선을 따라 몸통이 끝나는 거리(joint1 축에서)"""
    b, z = body_profile(psi, hull)
    return float(b[np.isfinite(z)].max() + 0.005)


def rot(a, v):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c])


def fk(j2, j3, j4, g=G_DEFAULT):
    """관절 2–4 → [어깨, 팔꿈치(link3), 손목(link4), link5, 잡는 점, 손끝 끝], 도구 방향 각 φ = j2+j3+j4 (0 수평 앞, +π/2 아래)"""
    p2 = SHOULDER
    p3 = p2 + rot(j2, (0.0415, 0.11315))
    p4 = p3 + rot(j2 + j3, (0.162, 0.0))
    p5 = p4 + rot(j2 + j3 + j4, (0.0287, 0.0))
    pg = p5 + rot(j2 + j3 + j4, (g, 0.0))
    pt = p5 + rot(j2 + j3 + j4, (0.0945, 0.0))
    return [p2, p3, p4, p5, pg, pt], j2 + j3 + j4


def collides(pts, margin=0.015, psi=0.0, hull=False):
    """팔 선분(점 표본, 평면 좌표 x = joint1 축에서 거리 + J1_AXIS[0])이 바닥·몸통 윗면에 닿는가(팔 반두께 margin)"""
    bins, zs = body_profile(psi, hull)
    pts = [np.array(p) for p in pts]
    for k, (a, b) in enumerate(zip(pts[:-2], pts[1:-1])):
        for t in np.linspace(0, 1, 16):
            p = a + (b - a) * t
            if p[1] < FLOOR + 0.005:
                return True
            if k == 0 and t < 0.5:        # 어깨 받침(link2 아래쪽)은 몸통 위에 얹혀 있다 — 시뮬은 base_link–link1 충돌을 끈다
                continue
            s = p[0] - J1_AXIS[0]
            near = np.abs(bins + 0.0025 - s) < margin
            if near.any() and p[1] - margin < zs[near].max():
                return True
    p = pts[-1]
    if p[1] < FLOOR + 0.005:
        return True
    return False


def ik(target, phi, g=G_DEFAULT):
    """잡는 점 target(x, z) + 도구 각 phi → (j2, j3, j4) 해들(팔꿈치 위/아래), 범위 안만"""
    w = np.array(target) - rot(phi, (0.0287 + g, 0.0))   # link4 원점
    d = w - SHOULDER
    a1 = math.hypot(0.0415, 0.11315)
    off = math.atan2(0.0415, 0.11315)          # link3 오프셋 각(위팔 수직에서 앞으로)
    a2 = 0.162
    r = math.hypot(*d)
    if r > a1 + a2 or r < abs(a1 - a2):
        return []
    out = []
    for s in (1, -1):
        c = (r * r - a1 * a1 - a2 * a2) / (2 * a1 * a2)
        c = max(-1, min(1, c))
        el = s * math.acos(c)
        # 평면 각(위 0, 앞으로 +) 으로 풀기: 위팔 방향 θ1(수직에서), 아래팔 θ1+el
        base_ang = math.atan2(d[0], d[1])       # 수직에서 앞으로
        k = math.atan2(a2 * math.sin(el), a1 + a2 * math.cos(el))
        th1 = base_ang - k
        # 위팔 벡터 = rot(j2, (0.0415, 0.11315)) → 수직에서 앞으로 th1 = j2 + off
        j2 = th1 - off
        # 아래팔 방향 수평에서 아래로 φ3 = j2 + j3 ; 수직에서 앞으로 = π/2 + φ3 = th1 + el
        phi3 = th1 + el - math.pi / 2
        j3 = phi3 - j2
        j4 = phi - phi3
        q = (j2, j3, j4)
        if J2[0] <= j2 <= J2[1] and J3[0] <= j3 <= J3[1] and J4[0] <= j4 <= J4[1]:
            pts, _ = fk(*q, g=g)
            if np.allclose(pts[4], target, atol=1e-6):
                out.append(q)
    return out


def reach_table(phi, ds, g=G_DEFAULT, psi=0.0, hull=False):
    """몸통 가장자리에서 반직선 방향으로 ds 만큼 앞 지점들에서 잡는 점 높이 범위(바닥 기준)"""
    edge = body_edge(psi, hull)
    rows = []
    for dd in ds:
        x = J1_AXIS[0] + edge + dd
        zs = []
        for z in np.arange(FLOOR, 0.6, 0.0025):
            for q in ik((x, z), phi, g):
                pts, _ = fk(*q, g=g)
                if not collides(pts, psi=psi, hull=hull):
                    zs.append(z)
                    break
        rows.append((dd, (min(zs) - FLOOR, max(zs) - FLOOR) if zs else None))
    return rows


def max_out(psi=0.0, hull=False, g=G_DEFAULT):
    """몸통 가장자리 밖 최대 수평 거리(아무 높이·도구 각, 충돌 없이) → (거리, 높이, φ)"""
    edge = body_edge(psi, hull)
    best = (-1, None, None)
    for phi in np.radians(np.arange(-90, 91, 5)):
        for x in np.arange(0.40, 0.0, -0.0025):
            ok = False
            for z in np.arange(FLOOR, 0.6, 0.01):
                for q in ik((x, z), phi, g):
                    if not collides(fk(*q, g=g)[0], psi=psi, hull=hull):
                        ok = True
                        if x - J1_AXIS[0] - edge > best[0]:
                            best = (x - J1_AXIS[0] - edge, z - FLOOR, math.degrees(phi))
                        break
                if ok:
                    break
            if ok:
                break
    return best


if __name__ == "__main__":
    q = (-1.6, 1.45, 0.15)
    pts, phi = fk(*q)
    print("home link5 (base_link 기준)", pts[3].round(4), "→ 시뮬 omx_link5 (0.0236, 0.1598) 와 비교")
    ds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    res = {}
    for hull in (False, True):
        for psi, pl in ((0.0, "앞"), (math.pi / 2, "옆")):
            print("#### 몸통", "시뮬 볼록 껍질" if hull else "실제 메시", "방향", pl, "몸통 가장자리(joint1 축에서) %.3f" % body_edge(psi, hull))
            for phi, lab in ((math.pi / 2, "위에서(수직 아래)"), (math.pi / 4, "45° 아래"), (0.0, "옆에서(수평)")):
                rows = reach_table(phi, ds, psi=psi, hull=hull)
                res[f"{'sim' if hull else 'real'}_{pl}_{lab}"] = rows
                print("  ==", lab, "  ".join(f"+{d:.2f}:{'-' if r is None else '%.3f..%.3f' % r}" for d, r in rows))
            print("  최대 수평 도달(몸통 가장자리 밖, 거리 m, 높이 m, φ°):", tuple(round(v, 3) for v in max_out(psi, hull)))


# ---------- 하중: 정적 관절 힘(URDF 링크 질량·무게 중심, 손가락은 닫힘 기준) ----------
URDF = os.environ.get("MAP_VLA_URDF") or subprocess.check_output([os.path.join(_ROBOT, "tools", "build_urdf.sh"), "/tmp/map_vla_kin.urdf"], text=True).strip()   # 저장소 xacro 에서
STALL = {"omx_joint2": 1.5, "omx_joint3": 1.5, "omx_joint4": 0.52}   # N·m: XL430-W250-T @12 V, XL330-M288-T @5 V(e-Manual)


def _inertials():
    import re
    t = open(URDF).read()
    out = {}
    for m in re.finditer(r'<link name="(omx_link[2-7])">(.*?)</link>', t, re.S):
        im = re.search(r'<inertial>\s*<origin xyz="([^"]+)"[^>]*/>\s*<mass value="([^"]+)"', m.group(2))
        out[m.group(1)] = (np.array([float(v) for v in im.group(1).split()]), float(im.group(2)))
    return out


def static_torque(j2, j3, j4, payload=0.0, g=G_DEFAULT):
    """평면 정적 중력 관절 힘(N·m) {joint2, joint3, joint4}. 양수 = 팔을 내리려는 쪽(크기만 본다)"""
    I = _inertials()
    pts, _ = fk(j2, j3, j4, g=g)
    p2, p3, p4, p5 = pts[0], pts[1], pts[2], pts[3]
    a2, a3, a4 = j2, j2 + j3, j2 + j3 + j4
    coms = []
    for name, frame_p, ang, extra in (("omx_link2", p2, a2, (0, 0)), ("omx_link3", p3, a3, (0, 0)), ("omx_link4", p4, a4, (0, 0)),
                                      ("omx_link5", p5, a4, (0, 0)), ("omx_link6", p5, a4, (0.0295, 0)), ("omx_link7", p5, a4, (0.0295, 0))):
        c, m = I[name]
        off = np.array(extra) + np.array([c[0], c[2]])
        coms.append((frame_p + rot(ang, off), m))
    coms.append((pts[4], payload))
    tau = {}
    for jn, axis, first in (("omx_joint2", p2, 0), ("omx_joint3", p3, 1), ("omx_joint4", p4, 2)):
        tau[jn] = sum(9.81 * m * (c[0] - axis[0]) for c, m in coms[first:])
    return tau


def payload_limit(q, frac=1.0, g=G_DEFAULT):
    """자세 q 에서 관절 힘이 frac × 정지 토크를 넘지 않는 최대 하중(kg)과 그때 막는 관절"""
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        t = static_torque(*q, payload=mid, g=g)
        if all(abs(t[j]) <= frac * STALL[j] for j in STALL):
            lo = mid
        else:
            hi = mid
    t = static_torque(*q, payload=lo + 1e-3, g=g)
    worst = max(STALL, key=lambda j: abs(t[j]) / STALL[j])
    return lo, worst


def payload_report():
    rows = {}
    # 완전히 뻗음(수평, 위팔·아래팔·손목 일직선 앞으로): j2 = -off, j3 = +off(아래팔 수평), j4 = 0
    off = math.atan2(0.0415, 0.11315)
    full = (math.pi / 2 - off, -(math.pi / 2 - off), 0.0)
    rows["full_reach_horizontal"] = full
    # 시뮬 시험 자세들
    rows["side_top_down_h0.15_d0.08"] = (0.1779, 0.4704, 0.9225)
    rows["side_horiz_h0.45_d0.05"] = (-0.9558, 0.2642, 0.6916)
    rows["side_horiz_h0.25_d0.25"] = (0.7014, -0.3408, -0.3606)
    out = {}
    for k, q in rows.items():
        pts, _ = fk(*q)
        reach = float(pts[4][0] - SHOULDER[0])
        t0 = static_torque(*q)
        out[k] = dict(q=[round(v, 4) for v in q], reach_from_j2_axis=round(reach, 3),
                      tau_noload={j: round(v, 3) for j, v in t0.items()},
                      limit_stall=[round(v, 3) if isinstance(v, float) else v for v in payload_limit(q, 1.0)],
                      limit_half_stall=[round(v, 3) if isinstance(v, float) else v for v in payload_limit(q, 0.5)])
    return out
