-- slam_carto 공통 2D 설정 — Cartographer 코어(third_party/cartographer)의 기본 lua 위에 우리 값만 덮어쓴다.
-- 센서별 파일(limo_x2l.lua, openloris_hokuyo.lua)이 이 파일을 include 하고 범위·스캔 수만 바꾼다.
-- ROS 쪽 옵션(tracking_frame 등)은 없다: 스캔 점은 slam_carto 가 베이스(base_link 바닥 투영) 기준으로 바꿔 넣는다.

include "map_builder.lua"
include "trajectory_builder.lua"

MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.num_background_threads = 2          -- 리모(Jetson) 를 생각해 작게

TB = TRAJECTORY_BUILDER_2D
TB.use_imu_data = false                         -- IMU 는 있으면 slam_carto 가 켠다(sc_config use_imu)
TB.min_z = -0.5
TB.max_z = 1.0
TB.num_accumulated_range_data = 1
TB.voxel_filter_size = 0.025
TB.use_online_correlative_scan_matching = true  -- 값싼 라이다·바퀴 미끄럼: 상관 맞추기로 먼저 잡고 Ceres 로 다듬음
TB.real_time_correlative_scan_matcher.linear_search_window = 0.15
TB.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.)
TB.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
TB.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1
TB.ceres_scan_matcher.translation_weight = 10.
TB.ceres_scan_matcher.rotation_weight = 40.
TB.motion_filter.max_time_seconds = 5.
TB.motion_filter.max_distance_meters = 0.05     -- OpenLORIS 7 판 맞춤(0.2 m·1° 보다 ATE 낮음)
TB.motion_filter.max_angle_radians = math.rad(0.2)
TB.submaps.num_range_data = 45
TB.submaps.grid_options_2d.resolution = 0.05

PG = MAP_BUILDER.pose_graph
PG.optimize_every_n_nodes = 30
PG.constraint_builder.sampling_ratio = 0.3
PG.constraint_builder.min_score = 0.62
PG.constraint_builder.global_localization_min_score = 0.66
PG.constraint_builder.max_constraint_distance = 10.
PG.constraint_builder.log_matches = false
PG.optimization_problem.huber_scale = 1e1
PG.optimization_problem.odometry_translation_weight = 1e5
PG.optimization_problem.odometry_rotation_weight = 1e5
PG.optimization_problem.ceres_solver_options.num_threads = 2
PG.log_residual_histograms = false
PG.global_sampling_ratio = 0.003
