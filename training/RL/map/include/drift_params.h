// 생성 파일 — training/RL/map_calib/tools/carto_drift_header.py --set all (손으로 고치지 말 것). GPU_MAP_PORT.md 0.3
// 원천: src/scene_graph/slam_carto/calib/carto_drift.json (sha256 10f528d7cdb5), 세트 "all": Cartographer 2D (src/scene_graph/slam_carto), 실시간 자세
// Cartographer 자세 오차 흉내(map.h phase_begin, 제어 스텝마다): 로봇 기준 앞·옆·yaw 걸음 오차 분산 = c0·u + c_d·Δd + c_r·|Δθ|, 치우침·u,
// 그 뒤 믿는 자세 오차를 ρ^u 배로 되돌림(전역 최적화가 오차를 묶음 — growth 4 m 넘는 칸 중앙값 xy 0.0336 m·yaw 0.324° 에 맞춘 AR(1)).
// u = 움직임 양(keyframe 평균 걸음 단위) = max(Δd / step_d, |Δθ| / step_r) — 서 있으면 오차가 그대로(늘지도 줄지도 않음).
#pragma once
namespace gmap {
struct CartoDrift {
  static constexpr float long_c0 = 7.17629834e-06f, long_cd = 0.000151592076f, long_cr = 5.13356847e-05f;
  static constexpr float lat_c0 = 5.14394001e-06f, lat_cd = 0.000108210908f, lat_cr = 6.85012491e-05f;
  static constexpr float yaw_c0 = 5.4510185e-06f, yaw_cd = 0.000109280009f, yaw_cr = 0.000126293376f;
  static constexpr float bias_long = 4.01912729e-06f, bias_lat = 0.000346275872f, bias_yaw = -1.29895222e-05f;   // keyframe 평균 걸음마다
  static constexpr float step_d = 0.0375285054f, step_r = 0.0449206082f;   // keyframe 평균 걸음(m, rad)
  static constexpr float ln_rho_xy = -0.00849904155f, ln_rho_yaw = -0.121951731f;   // keyframe 걸음마다 되돌림 ln ρ (ρ 0.991537 · 0.885191)
};
}  // namespace gmap
