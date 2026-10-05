// 잡기 확인 규칙 단위 시험(CPU, map.h hands_step — GPU 와 같은 소스). scenemap 의 limo_held 시험과 같은 경우:
//   컵 4 cm 를 0.41 rad 로 쥐고 멈춤 → 듦, 열면 놓음 / 든 뒤 끝까지 닫힘 → 놓침 / 빈손으로 끝까지 닫힘 → 아님 / 8 cm(가운데 변 > 6 cm) → 아님 /
//   큰 탁자(고정 종류)를 팔 끝에 두고 닫음 → 아님 / 닫히는 중(멈추지 않음) → 아직 아님 / 틈과 폭이 안 맞음(1 cm 물체를 4 cm 틈) → 아님.
// 실패 수를 내고, 0 이 아니면 종료 코드 1.
#include <cstdio>
#include <vector>

#include "map.h"

using namespace gmap;

static int fails = 0;
static void check(bool ok, const char* what) {
  std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
  fails += ok ? 0 : 1;
}

struct Rig {
  MapCore m;
  std::vector<Slot> ob = std::vector<Slot>(NOBJ);   // 물체 저장소
  EnvView e{};
  Rig() {
    init_core(m, 1, 0);
    e.rhx = 3.f; e.rhy = 3.f; e.ep = 0;
    for (int k = 0; k < env::N_Q; ++k) e.q[k] = 0.f;
    e.q[1] = 0.6f; e.q[2] = 0.3f;   // 팔을 앞으로 숙임(잡는 점이 몸통 앞)
    e.q[5] = 1.2f;                  // 열림
    reset_core(m, e);
    hands_step(m, e, ob.data());    // 잡는 점 계산
  }
  void put(int b, int cls, float wx, float wy, float wz) {
    Slot& S = ob[b];
    m.objv[b >> 5] |= 1u << (b & 31);
    S.valid = 1; S.confirmed = 1; S.cls = cls; S.state = S_SEEN; S.held = 0; S.id = b + 1;
    for (int a = 0; a < 3; ++a) S.pos[a] = m.gp_m[a];
    S.ext[0] = wx; S.ext[1] = wy; S.ext[2] = wz;
    ap_init(S);
  }
  void step(float grip, int n) {
    for (int k = 0; k < n; ++k) { m.t += 1; e.q[5] = grip; hands_step(m, e, ob.data()); }
  }
};

int main() {
  std::printf("map_grasp_test: grasp point %.4f m behind omx_end_effector_link, closed < %.2f rad, max mid side %.3f m\n", -MP::grasp_off, MP::grip_closed,
              MP::grasp_max_w);
  {
    Rig r;
    r.put(0, C_CUP, 0.04f, 0.04f, 0.10f);
    r.step(0.41f, 1);
    check(r.m.held_slot < 0, "cup 4 cm: closing (not settled yet) -> not held");
    r.step(0.41f, 2);
    check(r.m.held_slot == 0, "cup 4 cm: closed at 0.41 rad and settled -> held");
    const float p0 = r.ob[0].pos[0];
    r.e.q[1] = 0.4f;   // 팔을 들면 물체가 잡는 점을 따라감
    r.step(0.41f, 1);
    check(r.ob[0].pos[0] != p0 && r.m.held_slot == 0, "held object follows the grasp point");
    r.step(1.2f, 1);
    check(r.m.held_slot < 0 && r.ob[0].held == 0, "opened -> released");
  }
  {
    Rig r;
    r.put(0, C_CUP, 0.04f, 0.04f, 0.10f);
    r.step(0.41f, 3);
    r.step(0.0f, 3);
    check(r.m.held_slot < 0, "held, then fully closed (gap < 5 mm) -> released (lost)");
  }
  {
    Rig r;
    r.step(0.0f, 4);
    check(r.m.held_slot < 0 && r.m.n_grasp_total == 0, "empty hand fully closed -> nothing");
    r.put(0, C_CUP, 0.04f, 0.04f, 0.10f);
    r.step(0.0f, 3);
    check(r.m.held_slot < 0, "already closed empty, object appears -> no new pick (one try per close)");
  }
  {
    Rig r;
    r.put(0, C_CUP, 0.08f, 0.08f, 0.10f);
    r.step(0.5f, 3);
    check(r.m.held_slot < 0, "8 cm wide object (mid side > 6 cm) -> not held");
  }
  {
    Rig r;
    r.put(0, C_TABLE, 1.2f, 0.6f, 0.75f);
    r.step(0.41f, 3);
    check(r.m.held_slot < 0, "table (static kind, big) at the gripper -> not held");
  }
  {
    Rig r;
    r.put(0, C_CUP, 0.01f, 0.01f, 0.01f);
    r.step(0.41f, 3);
    check(r.m.held_slot < 0, "1 cm object with a 4 cm gap (gap > widest + 2.5 cm) -> not held");
  }
  {
    Rig r;
    r.put(0, C_CUP, 0.04f, 0.04f, 0.10f);
    for (int k = 0; k < 6; ++k) r.step(0.55f - 0.02f * (float)k, 1);   // 0.02 rad/스텝으로 계속 닫힘(멈추지 않음)
    check(r.m.held_slot < 0, "gripper still closing (not settled) -> not held");
  }
  {
    Rig r;
    r.put(0, C_CUP, 0.04f, 0.04f, 0.10f);
    r.ob[0].pos[0] += 0.2f;   // 잡는 점에서 0.2 m(grasp_r 0.12 밖)
    r.step(0.41f, 3);
    check(r.m.held_slot < 0, "object 0.2 m from the grasp point (> grasp_r) -> not held");
  }
  {   // 팔 캡슐 가림: 카메라 → 팔 뼈대 가운데 점 선분은 가림, 바닥 먼 점은 안 가림(팔이 위에 있음)
    Rig r;
    float mid[3];
    for (int a = 0; a < 3; ++a) mid[a] = 0.5f * (r.m.arm[4][a] + r.m.arm[5][a]);
    const bool front = mid[0] > env::K::cam_x + 0.05f;
    if (front) {
      const float far[3] = {env::K::cam_x + 2.f * (mid[0] - env::K::cam_x), 2.f * mid[1], env::K::cam_z + 2.f * (mid[2] - env::K::cam_z)};
      check(arm_blocks(r.m, far), "camera ray through the arm (point behind the arm) -> blocked");
    }
    r.e.q[1] = 0.f; r.e.q[2] = 0.f;   // 홈 자세(팔이 위로 접힘)
    r.step(1.2f, 1);
    const float side_pt[3] = {env::K::cam_x + 0.3f, 1.5f, env::K::cam_z};
    check(!arm_blocks(r.m, side_pt), "camera ray to the left side (away from the arm) -> not blocked");
  }
  std::printf("map_grasp_test: %d failures\n", fails);
  return fails ? 1 : 0;
}
