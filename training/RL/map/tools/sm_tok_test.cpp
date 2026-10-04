// sm_tok.h 단위 시험(MAP_STATE_PLAN.md 4단계 통과 기준: 같은 지도에 로봇 자세를 바꿔 넣으면 상대 좌표가 맞게 변함).
//   1. 강체 불변: 지도(물체·로봇·문)를 같은 SE(2)(회전 + 평행 이동)로 옮기면 토큰이 같다(FP16 한 칸 안). 단 회전에서는 T_EEF_S 제외
//      (map 축 상자 — GPU 형식 그대로라 map 을 돌리면 상자가 바뀜). 평행 이동만이면 모든 값이 같다
//   2. 로봇만 θ 돌리면 T_POS 가 −θ 로 돌고, 방위 −θ, 거리·크기·상태는 그대로
//   3. 로봇만 d 옮기면 T_POS = Rᵀ(p − (x + d))
//   4. 칸 순서(목표 맨 앞, 나머지 가까운 순)·빈 칸(0, 번호 −1)·n_slot·KSLOT 자르기, 상태 one-hot, 문 상대 좌표
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "scenemap.h"   // 스냅숏 변환(sm_tok_from_snapshot)도 같이 컴파일되게(부르지는 않음 — 링크 불필요)
#include "sm_tok.h"

using namespace gmap;

static int g_fail = 0;
#define CHECK(c, ...)                                   \
  do {                                                  \
    if (!(c)) {                                         \
      ++g_fail;                                         \
      std::printf("FAIL %s:%d ", __FILE__, __LINE__);   \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
    }                                                   \
  } while (0)

static float h2f(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  uint32_t x;
  if (e == 0) {
    if (!m) x = sign;
    else { int ee = -1; uint32_t mm = m; do { ++ee; mm <<= 1; } while (!(mm & 0x400)); x = sign | uint32_t(127 - 15 - ee) << 23 | (mm & 0x3ff) << 13; }
  } else if (e == 31) x = sign | 0x7f800000u | m << 13;
  else x = sign | (e + 112) << 23 | m << 13;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
static float tol(float v) { return std::fmax(2e-3f, std::fabs(v) * 2e-3f); }   // FP16 반올림 두 번 + 입력 float 오차

static SmTokIn scene(std::mt19937& rng, int n) {
  std::uniform_real_distribution<float> U(-6.f, 6.f), E(0.05f, 1.0f), Z(0.f, 1.5f), S(0.3f, 1.f);
  SmTokIn in;
  in.x = U(rng); in.y = U(rng); in.yaw = U(rng) * 0.5f;
  in.eef_b[0] = 0.25f; in.eef_b[1] = 0.02f; in.eef_b[2] = 0.1f;
  in.now = 30.0; in.last_image = 29.5;
  in.have_door = true; in.door_xy[0] = U(rng); in.door_xy[1] = U(rng);
  in.room_type = R_OFFICE;
  for (int i = 0; i < n; ++i) {
    SmTokObj o;
    o.id = uint32_t(i + 1);
    o.name_id = i % 5;
    for (int k = 0; k < 2; ++k) { o.pos[k] = U(rng); o.first_pos[k] = o.pos[k] + 0.3f * float(k + 1) * float(i % 2); }
    o.pos[2] = o.first_pos[2] = Z(rng);
    for (int k = 0; k < 3; ++k) o.ext[k] = E(rng);
    o.state = i % 4;
    o.score = S(rng);
    o.n_obs = float(3 + i);
    o.last_seen = i % 3 ? 29.5 : 20.0;
    in.objs.push_back(o);
  }
  return in;
}

// map 전체(로봇·물체·처음 자리·문)를 (θ, t) 로 옮김
static SmTokIn moved(const SmTokIn& a, float th, float tx, float ty) {
  SmTokIn b = a;
  const float c = std::cos(th), s = std::sin(th);
  auto tr = [&](float* p) { const float x = p[0], y = p[1]; p[0] = c * x - s * y + tx; p[1] = s * x + c * y + ty; };
  float r[2] = {a.x, a.y};
  tr(r);
  b.x = r[0]; b.y = r[1]; b.yaw = a.yaw + th;
  tr(b.door_xy);
  for (auto& o : b.objs) { tr(o.pos); tr(o.first_pos); }
  return b;
}

// skip_box: T_EEF_S(팔 끝 ↔ 상자 겉면)는 map 축 상자(scenemap extent, GPU make_tokens 와 같음)로 재므로 map 을 돌리면 달라진다 — 회전 시험에서만 뺌
static void same(const MapTok& A, const MapTok& B, const char* what, bool skip_box = false) {
  CHECK(A.n_slot == B.n_slot, "%s n_slot %d %d", what, A.n_slot, B.n_slot);
  for (int r = 0; r < A.n_slot; ++r) {
    CHECK(A.name_id[r] == B.name_id[r], "%s slot %d name %d %d", what, r, A.name_id[r], B.name_id[r]);
    for (int q = 0; q < TOK_SLOT_VALS; ++q) {
      if (skip_box && q == T_EEF_S) continue;
      const float a = h2f(A.slot[r][q]), b = h2f(B.slot[r][q]);
      const float t = q == T_BEAR ? 4e-3f : tol(a);
      CHECK(std::fabs(a - b) <= t, "%s slot %d val %d: %f vs %f", what, r, q, a, b);
    }
  }
  for (int q = 0; q < N_ROOMTOK; ++q) CHECK(std::fabs(h2f(A.room[q]) - h2f(B.room[q])) <= tol(h2f(A.room[q])), "%s room %d", what, q);
}

int main() {
  std::mt19937 rng(7);
  int n_cases = 0;
  // 1. 강체 불변(로봇이 물체에 너무 가까우면 방위가 불안정하므로 거리 0.2 m 넘는 장면만)
  for (int it = 0; it < 200; ++it) {
    SmTokIn a = scene(rng, 12);
    bool ok = true;
    for (auto& o : a.objs) ok = ok && std::hypot(o.pos[0] - a.x, o.pos[1] - a.y) > 0.2f;
    if (!ok) continue;
    std::uniform_real_distribution<float> T(-20.f, 20.f), R(-3.1f, 3.1f);
    const SmTokIn b = moved(a, R(rng), T(rng), T(rng));
    MapTok A, B;
    make_sm_tokens(a, &A);
    make_sm_tokens(b, &B);
    same(A, B, "rigid", true);
    const SmTokIn t = moved(a, 0.f, T(rng), T(rng));   // 평행 이동만: 모든 값이 같아야 함
    MapTok C;
    make_sm_tokens(t, &C);
    same(A, C, "translate");
    n_cases += 2;
  }
  // 2. 로봇만 θ 회전: 칸 순서(거리)는 그대로, T_POS 는 Rz(−θ)
  {
    SmTokIn a = scene(rng, 6);
    MapTok A;
    make_sm_tokens(a, &A);
    for (float th : {0.5f, 1.5707963f, -2.f, 3.1f}) {
      SmTokIn b = a;
      b.yaw = a.yaw + th;
      MapTok B;
      make_sm_tokens(b, &B);
      const float c = std::cos(th), s = std::sin(th);
      for (int r = 0; r < A.n_slot; ++r) {
        const float ax = h2f(A.slot[r][T_POS]), ay = h2f(A.slot[r][T_POS + 1]);
        const float ex = c * ax + s * ay, ey = -s * ax + c * ay;
        CHECK(std::fabs(h2f(B.slot[r][T_POS]) - ex) <= tol(ex) * 2, "rot x %f %f", h2f(B.slot[r][T_POS]), ex);
        CHECK(std::fabs(h2f(B.slot[r][T_POS + 1]) - ey) <= tol(ey) * 2, "rot y %f %f", h2f(B.slot[r][T_POS + 1]), ey);
        CHECK(A.slot[r][T_POS + 2] == B.slot[r][T_POS + 2] && A.slot[r][T_DIST] == B.slot[r][T_DIST], "rot z/dist changed");
        for (int q = 0; q < 3; ++q) CHECK(A.slot[r][T_EXT + q] == B.slot[r][T_EXT + q], "rot ext");
        for (int q = 0; q < 4; ++q) CHECK(A.slot[r][T_STATE + q] == B.slot[r][T_STATE + q], "rot state");
        float db = h2f(B.slot[r][T_BEAR]) - (h2f(A.slot[r][T_BEAR]) - th);
        db = std::remainder(db, 6.2831853f);
        CHECK(std::fabs(db) < 4e-3f, "rot bearing %f", db);
        ++n_cases;
      }
    }
  }
  // 3. 로봇만 평행 이동: T_POS = Rᵀ(p − x)
  {
    SmTokIn a = scene(rng, 6);
    for (float d : {0.5f, -1.3f, 2.f}) {
      SmTokIn b = a;
      b.x += d; b.y -= 0.5f * d;
      MapTok B;
      make_sm_tokens(b, &B);
      const float c = std::cos(b.yaw), s = std::sin(b.yaw);
      // 칸 순서는 바뀔 수 있으므로 이름·n_obs 로 짝(n_obs 가 물체마다 다름)
      for (int r = 0; r < B.n_slot; ++r) {
        const float nobs = h2f(B.slot[r][T_NOBS]);
        for (auto& o : b.objs) {
          if (std::fabs(o.n_obs - nobs) > 0.25f) continue;
          const float dx = o.pos[0] - b.x, dy = o.pos[1] - b.y;
          const float ex = c * dx + s * dy, ey = -s * dx + c * dy;
          CHECK(std::fabs(h2f(B.slot[r][T_POS]) - ex) <= tol(ex), "trans x %f %f", h2f(B.slot[r][T_POS]), ex);
          CHECK(std::fabs(h2f(B.slot[r][T_POS + 1]) - ey) <= tol(ey), "trans y");
          CHECK(std::fabs(h2f(B.slot[r][T_POS + 2]) - (o.pos[2] - MP::base_z)) <= tol(o.pos[2]), "trans z");
          ++n_cases;
        }
      }
    }
  }
  // 4. 순서·빈 칸·자르기·one-hot·목표·문
  {
    SmTokIn a = scene(rng, 20);
    a.target_name = 3;
    MapTok A;
    make_sm_tokens(a, &A);
    CHECK(A.n_slot == KSLOT, "n_slot %d", A.n_slot);
    CHECK(A.name_id[0] == 3 && h2f(A.slot[0][T_TARGET]) == 1.f, "target first: name %d", A.name_id[0]);
    for (int r = 2; r < A.n_slot; ++r) CHECK(h2f(A.slot[r][T_DIST]) >= h2f(A.slot[r - 1][T_DIST]), "distance order at %d", r);
    for (int r = 0; r < A.n_slot; ++r) {
      float sum = 0;
      for (int q = 0; q < 4; ++q) sum += h2f(A.slot[r][T_STATE + q]);
      CHECK(sum == 1.f, "one-hot slot %d", r);
      CHECK(A.app_id[r] == NCLS, "app_id default");
    }
    SmTokIn e = scene(rng, 3);
    MapTok E;
    make_sm_tokens(e, &E);
    CHECK(E.n_slot == 3, "n_slot 3");
    for (int r = 3; r < KSLOT; ++r) {
      CHECK(E.name_id[r] == -1 && E.app_id[r] == -1, "empty ids");
      for (int q = 0; q < TOK_SLOT_VALS; ++q) CHECK(E.slot[r][q] == 0, "empty slot value");
    }
    const float c = std::cos(e.yaw), s = std::sin(e.yaw);
    const float dx = e.door_xy[0] - e.x, dy = e.door_xy[1] - e.y;
    CHECK(std::fabs(h2f(E.room[6]) - (c * dx + s * dy)) <= tol(c * dx + s * dy) && h2f(E.room[9]) == 1.f && h2f(E.room[R_OFFICE]) == 1.f, "door/room");
    // 물체 속도: 0.1 s 뒤 x 로 0.05 m → 0.5 m/s(로봇 축)
    SmTokPrev prev;
    SmTokIn v0 = scene(rng, 1);
    v0.yaw = 0.f;
    MapTok V;
    make_sm_tokens(v0, &V, &prev);
    v0.now += 0.1; v0.objs[0].pos[0] += 0.05f;
    make_sm_tokens(v0, &V, &prev);
    CHECK(std::fabs(h2f(V.slot[0][T_VEL]) - 0.5f) < 2e-3f, "velocity %f", h2f(V.slot[0][T_VEL]));
    n_cases += 4;
  }
  std::printf("sm_tok_test: %d cases, %d failures\n", n_cases, g_fail);
  return g_fail ? 1 : 0;
}
