// 자동 생성: tools/gen_limo_fk_table.cpp ← map_vla.urdf(LIMO + OMX-F). 손으로 고치지 말 것.
// 다시 만들기: gen_limo_fk_table map_vla.urdf(robot-agent src/robot/tools/build_urdf.sh 가 저장소 xacro 에서 펼침) include/scenemap/limo_omx_fk_table.hpp
// 사슬 시작 = base_footprint(바닥, scenemap 의 '베이스' 프레임). q = LIMO proprio 번호(scenemap.h SM_LIMO_*), -1 = 고정.
#pragma once

namespace scenemap::limo {

struct JointDef { int kind; double xyz[3], rpy[3], axis[3]; int q; const char* name; };
struct ChainDef { int n; const JointDef* j; double cam_xyz[3], cam_xyzw[4]; };

// base_footprint → depth_camera_lens_optical_frame: 몸통 카메라 렌즈 광학(cam 0, depth_camera_link +x 0.010 m, z 앞·x 오른쪽·y 아래)
inline constexpr JointDef k_depth_cam[] = {
  {0, {0, 0, 0.14999999999999999}, {0, 0, 0}, {0, 0, 0}, -1, "base_joint"},
  {0, {0.084000000000000005, 0, 0.029999999999999999}, {0, 0, 0}, {0, 0, 0}, -1, "depth_camera_joint"},
  {0, {0.01, 0, 0}, {0, 0, 0}, {0, 0, 0}, -1, "depth_camera_lens_joint"},
  {0, {0, 0, 0}, {-1.5707963, 0, -1.5707963}, {0, 0, 0}, -1, "depth_camera_lens_optical_joint"},
};
inline constexpr ChainDef k_depth_cam_chain = {4, k_depth_cam, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};

// base_footprint → wrist_cam_optical_frame: 손목 카메라 광학(cam 1, 깊이 없음)
inline constexpr JointDef k_wrist_cam[] = {
  {0, {0, 0, 0.14999999999999999}, {0, 0, 0}, {0, 0, 0}, -1, "base_joint"},
  {0, {-0.040000000000000001, 0, 0}, {0, 0, 0}, {0, 0, 0}, -1, "omx_mount_joint"},
  {1, {-0.01125, 0, 0.034000000000000002}, {0, 0, 0}, {0, 0, 1}, 6, "omx_joint1"},
  {1, {0, 0, 0.063500000000000001}, {0, 0, 0}, {0, 1, 0}, 7, "omx_joint2"},
  {1, {0.041500000000000002, 0, 0.11315}, {0, 0, 0}, {0, 1, 0}, 8, "omx_joint3"},
  {1, {0.16200000000000001, 0, 0}, {0, 0, 0}, {0, 1, 0}, 9, "omx_joint4"},
  {1, {0.0287, 0, 0}, {0, 0, 0}, {1, 0, 0}, 10, "omx_joint5"},
  {0, {0.049799999999999997, 0, 0.035099999999999999}, {0, 0.66212000000000004, 0}, {0, 0, 0}, -1, "wrist_cam_joint"},
  {0, {0, 0, 0}, {-1.5707963, 0, -1.5707963}, {0, 0, 0}, -1, "wrist_cam_optical_joint"},
};
inline constexpr ChainDef k_wrist_cam_chain = {9, k_wrist_cam, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};

// base_footprint → grasp_point: 잡기 점(E0 실측 omx_link5 x 0.08003 = OmniGibson get_eef_position) — 잡기 규칙·T_eef
inline constexpr JointDef k_eef[] = {
  {0, {0, 0, 0.14999999999999999}, {0, 0, 0}, {0, 0, 0}, -1, "base_joint"},
  {0, {-0.040000000000000001, 0, 0}, {0, 0, 0}, {0, 0, 0}, -1, "omx_mount_joint"},
  {1, {-0.01125, 0, 0.034000000000000002}, {0, 0, 0}, {0, 0, 1}, 6, "omx_joint1"},
  {1, {0, 0, 0.063500000000000001}, {0, 0, 0}, {0, 1, 0}, 7, "omx_joint2"},
  {1, {0.041500000000000002, 0, 0.11315}, {0, 0, 0}, {0, 1, 0}, 8, "omx_joint3"},
  {1, {0.16200000000000001, 0, 0}, {0, 0, 0}, {0, 1, 0}, 9, "omx_joint4"},
  {1, {0.0287, 0, 0}, {0, 0, 0}, {1, 0, 0}, 10, "omx_joint5"},
  {0, {0.080030000000000004, -0.0016000000000000001, 0}, {0, 0, 0}, {0, 0, 0}, -1, "grasp_point_joint"},
};
inline constexpr ChainDef k_eef_chain = {8, k_eef, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};

// base_footprint → omx_end_effector_link: 팔 끝 — 관절 원점 + 이 점이 팔 뼈대(몸 가리기)
inline constexpr JointDef k_tip[] = {
  {0, {0, 0, 0.14999999999999999}, {0, 0, 0}, {0, 0, 0}, -1, "base_joint"},
  {0, {-0.040000000000000001, 0, 0}, {0, 0, 0}, {0, 0, 0}, -1, "omx_mount_joint"},
  {1, {-0.01125, 0, 0.034000000000000002}, {0, 0, 0}, {0, 0, 1}, 6, "omx_joint1"},
  {1, {0, 0, 0.063500000000000001}, {0, 0, 0}, {0, 1, 0}, 7, "omx_joint2"},
  {1, {0.041500000000000002, 0, 0.11315}, {0, 0, 0}, {0, 1, 0}, 8, "omx_joint3"},
  {1, {0.16200000000000001, 0, 0}, {0, 0, 0}, {0, 1, 0}, 9, "omx_joint4"},
  {1, {0.0287, 0, 0}, {0, 0, 0}, {1, 0, 0}, 10, "omx_joint5"},
  {0, {0.091929999999999998, -0.0016000000000000001, 0}, {0, 0, 0}, {0, 0, 0}, -1, "omx_end_effector_joint"},
};
inline constexpr ChainDef k_tip_chain = {8, k_tip, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};

}  // namespace scenemap::limo
