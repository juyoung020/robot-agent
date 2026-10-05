// scenemap — 순기구학(proprio 관절값 + URDF 표). 평가 규칙상 카메라 자세·팔 자세는 이것으로만 만든다.
//
// 로봇: LIMO + OMX-F 하나(scenemap.h SM_ROBOT_LIMO_OMX).
//   표 include/scenemap/limo_omx_fk_table.hpp(tools/gen_limo_fk_table.cpp ← map_vla.urdf(robot-agent src/robot/tools/build_urdf.sh 가 저장소 xacro 에서 펼침)).
//   베이스 프레임 = base_footprint(바닥). proprio 12(scenemap.h SM_LIMO_*).
#pragma once

namespace scenemap {

enum RobotKind { kRobotLimoOmx = 0 };

constexpr int kLimoProprioDim = 12;   // LIMO + OMX-F (뒤에 바퀴 각 4 개는 선택, 뷰어용)
constexpr int kMaxProprioDim = 16;    // 내부 저장 크기(12 + 바퀴 각 4)
inline int proprioDim(int) { return kLimoProprioDim; }

struct BodyFk {
  float T_head[12];            // 베이스 ← 머리(LIMO: 깊이) 카메라 광학 프레임(행 우선 3×4) — 스캔에 쓰는 T_bc
  float cam_rel[3][7];         // 카메라 prim 자세(xyz + xyzw, 평가기 cam_rel_poses 와 같은 뜻: −z 앞, y 위)
  // 팔 뼈대: omx_link0, 관절 1..5 원점, omx_end_effector_link(T_tip, 나머지는 마지막 점 되풀이)
  static constexpr int kArmPts = 11;
  float arm[2][kArmPts][3];
  float torso[6][3];           // 안 씀(몸통 관절 없음)
  int n_arms;                  // 쓰는 팔 수(1) — limoBodyFk 가 채움
  int n_torso;                 // 쓰는 torso 점 수(0 — 몸은 스캔의 self_r 원으로 뺌)
};

// LIMO + OMX-F(배정밀도). 모든 자세 = base_footprint ← 그 프레임, 행 우선 3×4
struct LimoFk {
  double T_depth[12];          // 몸통 카메라 렌즈 광학(depth_camera_lens_optical_frame)
  double T_wrist[12];          // 손목 카메라 광학(wrist_cam_optical_frame)
  double T_eef[12];            // 잡기 점(grasp_point = omx_link5 x 0.08003, E0 실측 — OmniGibson get_eef_position). 잡기 규칙이 봄
  double T_tip[12];            // 팔 끝(omx_end_effector_link, 잡기 점 + 손가락 축 0.0119 m)
  static constexpr int kPts = 7;
  double pts[kPts][3];         // 팔 뼈대: omx_link0, 관절 1..5 원점, 팔 끝(T_tip)
};
void computeLimoFk(const float* proprio, LimoFk* out);
// LimoFk → BodyFk(스캔 몸 가리기·T_head) + 팔 끝(베이스 기준 T_tip — 몸 가리기 구, 두 칸 다 같은 값)
void limoBodyFk(const LimoFk& f, BodyFk* out, float eef[2][3]);

}  // namespace scenemap
