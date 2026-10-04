// LIMO + OMX-F 순기구학(limo_omx_fk_table.hpp + fk.cpp)이 URDF 에서 따로 계산한 기준값(tests/limo_fk_ref.hpp ←
// tests/gen_limo_fk_ref.py: xml.etree + numpy 4×4)과 맞는지 — 깊이 카메라·손목 카메라 광학, 팔 끝. 위치 1e-5 m, 회전 원소 1e-6.
// 같은 것을 C ABI(sm_robot_fk)로도 확인하고, R1 의 sm_robot_fk 머리 카메라가 computeBodyFk 의 T_head 와 같은지도 본다.
#include <cmath>
#include <cstdio>

#include "scenemap.h"
#include "scenemap/fk.hpp"
#include "limo_fk_ref.hpp"

using namespace scenemap;

int main() {
  int fail = 0;
  double worst_p = 0, worst_r = 0;
  const int n = int(sizeof(kLimoFkRef) / sizeof(kLimoFkRef[0]));
  for (int c = 0; c < n; ++c) {
    const LimoFkRef& r = kLimoFkRef[c];
    LimoFk f;
    computeLimoFk(r.q, &f);
    sm_body_fk b;
    if (sm_robot_fk(SM_ROBOT_LIMO_OMX, r.q, 12, &b) != 0) { std::printf("%s: sm_robot_fk failed\n", r.name); return 1; }
    const double* mine[3] = {f.T_depth, f.T_wrist, f.T_eef};
    const double* api[3] = {b.T_cam[0], b.T_cam[1], b.T_eef[0]};
    double cp = 0, cr = 0;
    for (int t = 0; t < 3; ++t)
      for (int k = 0; k < 12; ++k) {
        const double e = std::fabs(mine[t][k] - r.T[t][k]);
        if (k % 4 == 3) cp = std::fmax(cp, e); else cr = std::fmax(cr, e);
        if (api[t][k] != mine[t][k]) { std::printf("%s: C ABI differs from internal FK (target %d, %d)\n", r.name, t, k); ++fail; }
      }
    // 그리퍼 값·카메라 유효
    if (b.grip[0] != r.q[11] || b.n_cams != 2 || !b.cam_valid[0] || !b.cam_valid[1] || b.cam_valid[2] || b.n_hands != 1) {
      std::printf("%s: sm_body_fk fields wrong\n", r.name);
      ++fail;
    }
    const bool ok = cp < 1e-5 && cr < 1e-6;
    std::printf("%-6s pos %.2e m  rot %.2e  eef (%.4f, %.4f, %.4f) %s\n", r.name, cp, cr, f.T_eef[3], f.T_eef[7], f.T_eef[11], ok ? "ok" : "FAIL");
    worst_p = std::fmax(worst_p, cp);
    worst_r = std::fmax(worst_r, cr);
    fail += !ok;
  }
  std::printf("LIMO FK vs independent URDF FK: %d configs, worst position %.2e m, worst rotation element %.2e\n", n, worst_p, worst_r);

  // 몸통 카메라 렌즈(depth_camera_link 0.084 + 렌즈 0.010): base_footprint 위 0.18 m, 앞 0.094 m, 광학 z = 베이스 x(앞)
  {
    LimoFk f;
    computeLimoFk(kLimoFkRef[0].q, &f);
    const bool ok = std::fabs(f.T_depth[3] - 0.094) < 1e-9 && std::fabs(f.T_depth[11] - 0.18) < 1e-9 && f.T_depth[2] > 0.999999 &&
                    f.T_depth[4] < -0.999999 && f.T_depth[9] < -0.999999;
    std::printf("body camera lens (0.094, 0, 0.18), optical z = base x, x = −base y, y = −base z: %s\n", ok ? "ok" : "FAIL");
    fail += !ok;
  }
  // 몸 뼈대: 팔 한 개, 몸통 없음, 마지막 점 = 팔 끝
  {
    LimoFk f;
    computeLimoFk(kLimoFkRef[1].q, &f);
    BodyFk b;
    float eef[2][3];
    limoBodyFk(f, &b, eef);
    bool ok = b.n_arms == 1 && b.n_torso == 0;
    for (int i = 0; i < 3; ++i) ok = ok && std::fabs(b.arm[0][BodyFk::kArmPts - 1][i] - f.T_eef[i * 4 + 3]) < 1e-6 && eef[0][i] == eef[1][i];
    std::printf("body skeleton (1 arm, no torso, tip = eef): %s\n", ok ? "ok" : "FAIL");
    fail += !ok;
  }
  // 적은 proprio 는 거절
  {
    sm_body_fk b;
    const bool ok = sm_robot_fk(SM_ROBOT_LIMO_OMX, kLimoFkRef[0].q, 11, &b) < 0 && sm_robot_fk(7, kLimoFkRef[0].q, 12, &b) < 0 &&
                    sm_proprio_dim(SM_ROBOT_LIMO_OMX) == 12 && sm_proprio_dim(SM_ROBOT_R1PRO) == 61;
    std::printf("dims / bad input rejected: %s\n", ok ? "ok" : "FAIL");
    fail += !ok;
  }
  // R1: sm_robot_fk 머리 광학 = computeBodyFk T_head(float 반올림 안)
  {
    float q[61] = {0};
    q[53] = 0.3f; q[54] = -0.5f; q[55] = 0.2f; q[56] = 0.1f;
    BodyFk fk;
    computeBodyFk(q, &fk);
    sm_body_fk b;
    sm_robot_fk(SM_ROBOT_R1PRO, q, 61, &b);
    double e = 0;
    for (int k = 0; k < 12; ++k) e = std::fmax(e, std::fabs(b.T_cam[0][k] - fk.T_head[k]));
    const bool ok = e < 1e-6 && b.n_cams == 3 && b.n_hands == 2;
    std::printf("R1 sm_robot_fk head = T_head: %.1e %s\n", e, ok ? "ok" : "FAIL");
    fail += !ok;
  }
  return fail ? 1 : 0;
}
