"""OmniGibson 시뮬 LIMO 2D 라이다 — 실제 로봇의 EAI X2L(프로: T-mini Pro)을 흉내 낸 PhysX 광선 쏘기.

장착: URDF laser_link = base_link (0.103, 0, -0.034), 회전 없음(src/robot/limo_description/urdf/limo_four_diff.xacro).
      OmniGibson limo_omx 의 기준 링크는 base_link(src/robot/og/limo_omx.yaml base_footprint_link_name) — 시뮬 USD 에 laser_link 가
      있으면 그 자세를, 없으면 base_link 자세 ∘ 위 평행 이동을 쓴다.
자료표(EAI/YDLIDAR X2L): 거리 0.12–8 m, 360°, 회전 5–8 Hz(권장 6 Hz), 측정 3000 Hz → 6 Hz 에서 0.72°(500 점/바퀴),
      상대 오차 ≤ 1 m 3 %, 1–6 m 3.5 %.
가정(자료표에 없는 것):
  - 오차 3 %·3.5 % 를 약 2σ 로 보고 가우스 σ = 1.5 %·1.75 % × 거리. 광선마다 독립.
  - 광선 빠짐 1 %(검은 면·비스듬한 면은 따로 흉내 안 냄).
  - 한 스캔 = 한 순간(광선을 동시에 쏨) → time_inc = 0. 실제는 한 바퀴 1/6 s 동안 돌며 찍지만, 시뮬에서는 스캔 시각에 모두 쏜다.
  - 로봇 자기 몸(바퀴·차체·팔)에 맞은 광선은 돌려주지 않음(실제 드라이버의 각도 자르기와 같은 효과). 투명·유리도 그냥 맞음.
  - 높이 0.116 m(바닥 위) 수평 한 면만 — 그 아래 낮은 장애물은 안 보인다(실제와 같음).
쓰는 곳: sgrt 글루(src/scene_graph/runtime/glue/sgrt_glue.py, SGRT_LIDAR) — 스캔마다 sgrt_push_scan → libsgrt Cartographer.
"""
import math

import numpy as np

X2L = dict(hz=6.0, fov_deg=360.0, res_deg=0.72, rmin=0.12, rmax=8.0, sigma_near=0.015, sigma_far=0.0175, dropout=0.01)
LASER_IN_BASE = (0.103, 0.0, -0.034)


def _quat_yaw(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class LimoLidar:
    def __init__(self, robot, spec=None, seed=0):
        self.robot = robot
        self.s = dict(X2L, **(spec or {}))
        self.n = int(round(self.s["fov_deg"] / self.s["res_deg"]))
        self.angle_min = -math.radians(self.s["fov_deg"]) / 2
        self.angle_inc = math.radians(self.s["res_deg"])
        self.period = 1.0 / self.s["hz"]
        self.next_t = 0.0
        self.rng = np.random.default_rng(seed)
        links = getattr(robot, "links", {}) or {}
        self.link = links.get("laser_link")
        self.base = links.get("base_link") or getattr(robot, "root_link", None)
        self.robot_path = str(robot.prim_path)
        a = self.angle_min + self.angle_inc * np.arange(self.n)
        self.ca, self.sa = np.cos(a), np.sin(a)
        self.n_scans = 0

    def due(self, t):
        return t + 1e-9 >= self.next_t

    def _pose(self):
        """world 의 라이다 원점(x, y, z)과 yaw"""
        if self.link is not None:
            p, q = self.link.get_position_orientation()
            return [float(v) for v in p], _quat_yaw(q)
        p, q = (self.base or self.robot).get_position_orientation()
        yaw = _quat_yaw(q)
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly, lz = LASER_IN_BASE
        return [float(p[0]) + c * lx - s * ly, float(p[1]) + s * lx + c * ly, float(p[2]) + lz], yaw

    def scan(self, t):
        """스캔 하나: (ranges f32[n](빠짐 = inf), angle_min, angle_inc, time_inc, range_min, range_max). 다음 스캔 시각을 정한다"""
        import omnigibson as og
        self.next_t = max(self.next_t + self.period, t + 0.5 * self.period)
        (ox, oy, oz), yaw = self._pose()
        c, s = math.cos(yaw), math.sin(yaw)
        psqi = og.sim.psqi
        rmax = self.s["rmax"]
        out = np.full(self.n, np.inf, np.float32)
        best = [rmax + 1.0]
        rp = self.robot_path

        def cb(hit):
            if not str(hit.rigid_body).startswith(rp) and hit.distance < best[0]:
                best[0] = hit.distance
            return True

        for i in range(self.n):
            dx, dy = c * self.ca[i] - s * self.sa[i], s * self.ca[i] + c * self.sa[i]
            best[0] = rmax + 1.0
            psqi.raycast_all(origin=[ox, oy, oz], dir=[dx, dy, 0.0], distance=rmax, reportFn=cb)
            d = best[0]
            if d <= rmax:
                out[i] = d
        ok = np.isfinite(out)
        sig = np.where(out <= 1.0, self.s["sigma_near"], self.s["sigma_far"]) * np.where(ok, out, 0)
        out[ok] = out[ok] + (self.rng.standard_normal(int(ok.sum())) * sig[ok]).astype(np.float32)
        out[self.rng.random(self.n) < self.s["dropout"]] = np.inf
        out[(out < self.s["rmin"]) | (out > rmax)] = np.inf
        self.n_scans += 1
        return out, self.angle_min, self.angle_inc, 0.0, self.s["rmin"], rmax
