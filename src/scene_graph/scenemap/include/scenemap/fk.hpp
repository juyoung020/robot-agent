// scenemap — 순기구학(proprio 관절값 + URDF 표). 평가 규칙상 카메라 자세·팔 자세는 이것으로만 만든다.
//
// 로봇 둘(scenemap.h SM_ROBOT_*):
//   R1 Pro(기본)  — 표 include/scenemap/r1pro_fk_table.hpp(tools/gen_fk_table.py ← src/sim/integ/fk/r1pro_cam_fk.json).
//                   src/agent/planner/src/fk.rs 와 같은 계산. proprio 61.
//   LIMO + OMX-F — 표 include/scenemap/limo_omx_fk_table.hpp(tools/gen_limo_fk_table.cpp ← map_vla.urdf(robot-agent src/robot/tools/build_urdf.sh 가 저장소 xacro 에서 펼침)).
//                   베이스 프레임 = base_footprint(바닥). proprio 12(scenemap.h SM_LIMO_*).
#pragma once

namespace scenemap {

enum RobotKind { kRobotR1Pro = 0, kRobotLimoOmx = 1 };

constexpr int kProprioDim = 61;       // R1 Pro
constexpr int kLimoProprioDim = 12;   // LIMO + OMX-F (뒤에 바퀴 각 4 개는 선택, 뷰어용)
constexpr int kMaxProprioDim = 61;    // 내부 저장 크기(로봇 중 가장 긴 것)
inline int proprioDim(int robot) { return robot == kRobotLimoOmx ? kLimoProprioDim : kProprioDim; }

struct BodyFk {
  float T_head[12];            // 베이스 ← 머리(LIMO: 깊이) 카메라 광학 프레임(행 우선 3×4) — 스캔에 쓰는 T_bc
  float cam_rel[3][7];         // 카메라 prim 자세(xyz + xyzw, 평가기 cam_rel_poses 와 같은 뜻: −z 앞, y 위)
  // 팔 뼈대: R1 = 팔 받침, 관절 1..7 원점, 그리퍼, 손가락 끝(그리퍼에서 마지막 링크 방향으로 0.15 m).
  //          LIMO = omx_link0, 관절 1..5 원점, omx_end_effector_link(T_tip, 나머지는 마지막 점 되풀이)
  static constexpr int kArmPts = 11;
  float arm[2][kArmPts][3];
  float torso[6][3];           // R1: 베이스 원점 위, 몸통 관절 1..4 원점, 머리 카메라. LIMO: 안 씀
  int n_arms;                  // 쓰는 팔 수(R1 2, LIMO 1) — computeBodyFk·limoBodyFk 가 채움
  int n_torso;                 // 쓰는 torso 점 수(R1 6, LIMO 0 — 몸은 스캔의 self_r 원으로 뺌)
};

void computeBodyFk(const float* proprio, BodyFk* out);   // R1 Pro

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
