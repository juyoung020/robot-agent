#!/usr/bin/env python3
"""Cartographer 자세 오차 맞춤 json(src/scene_graph/slam_carto/calib/carto_drift.json) → GPU 지도 떠밀림 흉내 헤더(training/RL/map/include/drift_params.h).

  python3 training/RL/map_calib/tools/carto_drift_header.py [--set all|sim|openloris] [--json PATH] [--out PATH]

GPU_MAP_PORT.md 0.3: 걸음 오차(로봇 앞·옆·yaw)의 분산 = c0 + c_d·Δd + c_r·|Δθ| 와 치우침(keyframe 0.2 s 마다 맞춘 값 — 분산이 Δd·Δθ 에 선형이라 제어 스텝에
그대로 써도 합이 같다. c0·치우침·되돌림은 움직임 양 u 에 비례). 전역 최적화(되돌아옴)가 오차를 묶어 두는 것은 AR(1) 되돌림 ρ 로: 걸음 분산 q(평균 걸음에서)와
판 오차가 다다르는 크기(growth 의 긴 거리 칸 중앙값 m)로 정상 분산 s² = (m / 1.1774)²(2 차원 가우스 크기의 중앙값 = s·√(2 ln 2)), ρ = √(1 − q / s²).
yaw 도 같은 식(growth med_yaw_deg). 값과 출처(파일·세트·json 해시)를 헤더에 적는다.
"""
import argparse
import hashlib
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
ap = argparse.ArgumentParser()
ap.add_argument('--set', default='all')
ap.add_argument('--json', default=os.path.join(ROOT, 'src/scene_graph/slam_carto/calib/carto_drift.json'))
ap.add_argument('--out', default=os.path.join(ROOT, 'training/RL/map/include/drift_params.h'))
ap.add_argument('--plateau-from-m', type=float, default=4.0, help='growth 칸 중 이 거리 이상의 중앙값들(무게 n)로 정상 크기')
a = ap.parse_args()

raw = open(a.json, 'rb').read()
J = json.loads(raw)
S = J[a.set]
M = S['step_model']

def plateau(key):
    num = den = 0.0
    for b in S['growth']:
        if b['lo_m'] >= a.plateau_from_m:
            num += b[key] * b['n']
            den += b['n']
    return num / den

# 평균 걸음(keyframe) 분산 → 제어 스텝(걸음 반)
step_d = M['mean_step_m']
step_r = math.radians(M['mean_step_turn_deg'])
q_xy_kf = (M['var_long_m2']['c0'] + M['var_long_m2']['c_d'] * step_d + M['var_long_m2']['c_r'] * step_r +
           M['var_lat_m2']['c0'] + M['var_lat_m2']['c_d'] * step_d + M['var_lat_m2']['c_r'] * step_r) / 2.0   # 축 하나
q_yaw_kf = M['var_yaw_rad2']['c0'] + M['var_yaw_rad2']['c_d'] * step_d + M['var_yaw_rad2']['c_r'] * step_r
med_xy = plateau('med_xy')
med_yaw = math.radians(plateau('med_yaw_deg'))
s2_xy = (med_xy / 1.1774) ** 2
s2_yaw = (med_yaw / 0.6745) ** 2
rho_xy_kf = math.sqrt(max(0.0, 1.0 - q_xy_kf / s2_xy))
rho_yaw_kf = math.sqrt(max(0.0, 1.0 - q_yaw_kf / s2_yaw))

h = hashlib.sha256(raw).hexdigest()[:12]
rel = os.path.relpath(a.json, ROOT)
out = f"""// 생성 파일 — training/RL/map_calib/tools/carto_drift_header.py --set {a.set} (손으로 고치지 말 것). GPU_MAP_PORT.md 0.3
// 원천: {rel} (sha256 {h}), 세트 "{a.set}": {S.get('source', '')}
// Cartographer 자세 오차 흉내(map.h phase_begin, 제어 스텝마다): 로봇 기준 앞·옆·yaw 걸음 오차 분산 = c0·u + c_d·Δd + c_r·|Δθ|, 치우침·u,
// 그 뒤 믿는 자세 오차를 ρ^u 배로 되돌림(전역 최적화가 오차를 묶음 — growth {a.plateau_from_m:g} m 넘는 칸 중앙값 xy {med_xy:.4f} m·yaw {math.degrees(med_yaw):.3f}° 에 맞춘 AR(1)).
// u = 움직임 양(keyframe 평균 걸음 단위) = max(Δd / step_d, |Δθ| / step_r) — 서 있으면 오차가 그대로(늘지도 줄지도 않음).
#pragma once
namespace gmap {{
struct CartoDrift {{
  static constexpr float long_c0 = {M['var_long_m2']['c0']:.9g}f, long_cd = {M['var_long_m2']['c_d']:.9g}f, long_cr = {M['var_long_m2']['c_r']:.9g}f;
  static constexpr float lat_c0 = {M['var_lat_m2']['c0']:.9g}f, lat_cd = {M['var_lat_m2']['c_d']:.9g}f, lat_cr = {M['var_lat_m2']['c_r']:.9g}f;
  static constexpr float yaw_c0 = {M['var_yaw_rad2']['c0']:.9g}f, yaw_cd = {M['var_yaw_rad2']['c_d']:.9g}f, yaw_cr = {M['var_yaw_rad2']['c_r']:.9g}f;
  static constexpr float bias_long = {M['bias_long_m']:.9g}f, bias_lat = {M['bias_lat_m']:.9g}f, bias_yaw = {M['bias_yaw_rad']:.9g}f;   // keyframe 평균 걸음마다
  static constexpr float step_d = {step_d:.9g}f, step_r = {step_r:.9g}f;   // keyframe 평균 걸음(m, rad)
  static constexpr float ln_rho_xy = {math.log(rho_xy_kf):.9g}f, ln_rho_yaw = {math.log(rho_yaw_kf):.9g}f;   // keyframe 걸음마다 되돌림 ln ρ (ρ {rho_xy_kf:.6f} · {rho_yaw_kf:.6f})
}};
}}  // namespace gmap
"""
open(a.out, 'w').write(out)
print(out)
