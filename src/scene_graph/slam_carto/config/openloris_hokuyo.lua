-- OpenLORIS-Scene office(Segway 배달 로봇) Hokuyo UTM-30LX 2D 라이다: 270°, 40 Hz, 0.1–30 m.
-- 사무실 크기(약 10 m)에 맞춰 max_range 를 줄인다. 스캔은 하나씩 넣고 노드 수는 움직임 거르기(motion_filter)가 줄인다.
include "carto_2d.lua"

TB.min_range = 0.2
TB.max_range = 12.0
TB.missing_data_ray_length = 5.0
TB.num_accumulated_range_data = 1
TB.adaptive_voxel_filter.max_range = 12.0
TB.loop_closure_adaptive_voxel_filter.max_range = 12.0

return { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER }
