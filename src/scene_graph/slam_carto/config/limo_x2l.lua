-- LIMO + EAI X2L(프로: T-mini Pro) 2D 라이다 — 실제 로봇과 OmniGibson 시뮬 라이다(src/sim/lidar) 공용.
-- X2L 자료표(EAI/YDLIDAR X2L): 거리 0.12–8 m, 360°, 회전 5–8 Hz(권장 6 Hz), 측정 3000 Hz → 6 Hz 에서 0.72°(500 점).
-- 리모 앞 아래 장착(URDF laser_link: base_link (0.103, 0, -0.034))이라 뒤쪽은 몸에 가린다 — 가린 광선은 범위 밖으로 버림.
include "carto_2d.lua"

TB.min_range = 0.15                 -- 0.12 + 몸 가장자리 여유
TB.max_range = 8.0
TB.missing_data_ray_length = 3.0
TB.adaptive_voxel_filter.max_range = 8.0
TB.loop_closure_adaptive_voxel_filter.max_range = 8.0

return { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER }
