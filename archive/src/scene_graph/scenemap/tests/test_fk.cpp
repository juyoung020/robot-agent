// 순기구학이 시연 robot2cam 을 재현하는지(src/agent/planner/src/fk.rs 테스트와 같은 표본: 200번 판 3000 프레임).
#include <cmath>
#include <cstdio>

#include "scenemap/fk.hpp"

using namespace scenemap;

static const float STATE[61] = {
    -0.0003644f, 0.0004747f, 0.0005906f, -0.5135943f, 0.1073003f, -0.0477751f, -1.0079254f, 0.3504786f, 0.6210044f,
    0.5144721f, -0.0518736f, -0.0281572f, -0.0266854f, -0.2630615f, 0.6464736f, -0.2789042f, 0.010686f, 0.6501044f,
    0.3716516f, 0.5129846f, -0.1484585f, 0.941596f, 0.1572233f, 0.2581432f, 0.0075572f, 0.0003005f, 0.0209931f,
    -0.0171822f, -0.4822987f, 0.1745f, 0.7338722f, -1.5772073f, -0.0225522f, 1.0406232f, 0.0773677f, -0.0002653f,
    0.0003976f, 0.0011009f, 0.0045551f, 0.0169314f, 0.0107928f, 0.0021622f, 0.5823845f, 0.0940711f, 0.6938694f,
    -0.3479472f, 0.8557385f, 0.1888288f, 0.3331488f, 0.05f, 0.0244954f, 0.008631f, 0.0002444f, 1.2696129f, -1.896482f,
    -0.9405322f, -0.0004273f, -0.0026646f, -0.0017128f, 0.0187404f, -0.0033622f};
static const double R2C[3][7] = {
    {0.3747753, -2.79e-05, 1.2254213, 0.3100148, -0.3102268, -0.6355818, 0.6353628},
    {0.6619903, 0.3600194, 0.5937633, 0.1189633, 0.0504662, -0.5638065, 0.8157347},
    {0.5787872, 0.0836327, 0.7755998, 0.1755402, -0.0232762, -0.3736257, 0.9105206},
};

int main() {
  BodyFk fk;
  computeBodyFk(STATE, &fk);
  int fail = 0;
  for (int c = 0; c < 3; ++c) {
    double dp = 0, dot = 0, dm = 0, ds = 0;
    for (int i = 0; i < 3; ++i) dp += std::pow(fk.cam_rel[c][i] - R2C[c][i], 2);
    for (int i = 0; i < 4; ++i) dot += fk.cam_rel[c][3 + i] * R2C[c][3 + i];
    const double sg = dot < 0 ? -1 : 1;
    for (int i = 0; i < 4; ++i) {
      dm += std::pow(fk.cam_rel[c][3 + i] - sg * R2C[c][3 + i], 2);
      ds += std::pow(fk.cam_rel[c][3 + i] + sg * R2C[c][3 + i], 2);
    }
    dp = std::sqrt(dp);
    const double ang = 4 * std::atan2(std::sqrt(dm), std::sqrt(ds)) * 180 / M_PI;   // acos 는 1 근처에서 f32 반올림에 약함
    const bool ok = dp < 1e-4 && ang < 0.01;
    std::printf("cam %d: 위치 %.2e m, 각 %.4f° %s\n", c, dp, ang, ok ? "ok" : "FAIL");
    fail += !ok;
  }
  // 팔 뼈대: 그리퍼 점이 proprio 팔 끝(17:20, 42:45)과 가까워야 한다(같은 링크 근처)
  for (int s = 0; s < 2; ++s) {
    const float* e = STATE + (s == 0 ? 17 : 42);
    double best = 1e9;
    for (int k = 0; k < BodyFk::kArmPts; ++k) {
      double d = 0;
      for (int i = 0; i < 3; ++i) d += std::pow(fk.arm[s][k][i] - e[i], 2);
      best = std::fmin(best, std::sqrt(d));
    }
    const bool ok = best < 0.15;
    std::printf("arm %d: 팔 끝까지 가장 가까운 뼈대 점 %.3f m %s\n", s, best, ok ? "ok" : "FAIL");
    fail += !ok;
  }
  return fail ? 1 : 0;
}
