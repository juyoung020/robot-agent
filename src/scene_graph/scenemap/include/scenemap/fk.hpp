// scenemap — R1Pro 순기구학(proprio 관절값 + URDF 표). 평가 규칙상 카메라 자세·팔 자세는 이것으로만 만든다.
//
// 표: include/scenemap/r1pro_fk_table.hpp(tools/gen_fk_table.py ← src/sim/integ/fk/r1pro_cam_fk.json). src/agent/planner/src/fk.rs 와 같은 계산.
#pragma once

namespace scenemap {

constexpr int kProprioDim = 61;

struct BodyFk {
  float T_head[12];            // 베이스 ← 머리 카메라 광학 프레임(행 우선 3×4) — 스캔에 쓰는 T_bc
  float cam_rel[3][7];         // 카메라 셋 prim 자세(xyz + xyzw, 평가기 cam_rel_poses 와 같은 뜻)
  // 팔 뼈대: 팔 받침, 관절 1..7 원점, 그리퍼, 손가락 끝(그리퍼에서 마지막 링크 방향으로 0.15 m)
  static constexpr int kArmPts = 11;
  float arm[2][kArmPts][3];
  float torso[6][3];           // 베이스 원점 위, 몸통 관절 1..4 원점, 머리 카메라
};

void computeBodyFk(const float* proprio, BodyFk* out);

}  // namespace scenemap
