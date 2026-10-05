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
  // 5. 목표 칸(VLA_INPUT 2.1): 강체 불변(지도·점을 같이 옮김), 값 = Rᵀ(p − x), sin·cos, 손끝 기준, 사라짐(GONE)·스냅숏에 없음(마지막 자리) → GV_LOST, 모름, 점으로 가기 flags
  for (int it = 0; it < 50; ++it) {
    std::uniform_real_distribution<float> R(-3.1f, 3.1f), T(-4.f, 4.f);
    SmTokIn a = scene(rng, 6);
    a.goal[GE_PICK].kind = 1; a.goal[GE_PICK].id = a.objs[2].id;
    a.goal[GE_PLACE].kind = 2; a.goal[GE_PLACE].pt[0] = T(rng); a.goal[GE_PLACE].pt[1] = T(rng); a.goal[GE_PLACE].pt[2] = 0.45f;
    MapTok A;
    make_sm_tokens(a, &A);
    const float c = std::cos(a.yaw), s = std::sin(a.yaw);
    const float* P = a.goal[GE_PLACE].pt;
    const float x = c * (P[0] - a.x) + s * (P[1] - a.y), y = -s * (P[0] - a.x) + c * (P[1] - a.y), z = P[2] - MP::base_z, d = std::sqrt(x * x + y * y);
    CHECK(h2f(A.goal[1][GV_PRESENT]) == 1.f && h2f(A.goal[1][GV_KPT]) == 1.f && h2f(A.goal[1][GV_KOBJ]) == 0.f && h2f(A.goal[1][GV_KNOWN]) == 1.f, "point flags");
    CHECK(std::fabs(h2f(A.goal[1][GV_POS]) - x) <= tol(x) && std::fabs(h2f(A.goal[1][GV_POS + 1]) - y) <= tol(y) && std::fabs(h2f(A.goal[1][GV_POS + 2]) - z) <= tol(z), "point pos");
    CHECK(std::fabs(h2f(A.goal[1][GV_DIST]) - d) <= tol(d) && std::fabs(h2f(A.goal[1][GV_SIN]) - y / d) <= 2e-3f && std::fabs(h2f(A.goal[1][GV_COS]) - x / d) <= 2e-3f, "point dist/sincos");
    CHECK(std::fabs(h2f(A.goal[1][GV_EEF]) - (x - a.eef_b[0])) <= tol(x - a.eef_b[0]) && std::fabs(h2f(A.goal[1][GV_EEF + 2]) - (z - a.eef_b[2])) <= tol(z - a.eef_b[2]), "point eef");
    CHECK(h2f(A.goal[0][GV_KOBJ]) == 1.f && h2f(A.goal[0][GV_LOST]) == 0.f && h2f(A.goal[0][GV_KNOWN]) == 1.f, "pick flags");
    for (int q = GV_EEF + 3; q < N_GV; ++q) CHECK(A.goal[0][q] == 0 && A.goal[1][q] == 0, "reserved zero");
    // 강체 불변(점도 같이)
    SmTokIn b = moved(a, R(rng), T(rng), T(rng));
    {
      const float th = b.yaw - a.yaw, cc = std::cos(th), ss = std::sin(th);
      const float tx = b.x - (cc * a.x - ss * a.y), ty = b.y - (ss * a.x + cc * a.y);
      b.goal[GE_PLACE].pt[0] = cc * P[0] - ss * P[1] + tx; b.goal[GE_PLACE].pt[1] = ss * P[0] + cc * P[1] + ty;
    }
    MapTok B;
    make_sm_tokens(b, &B);
    for (int e = 0; e < N_GENT; ++e)
      for (int q = 0; q < N_GV; ++q) CHECK(std::fabs(h2f(A.goal[e][q]) - h2f(B.goal[e][q])) <= std::fmax(4e-3f, std::fabs(h2f(A.goal[e][q])) * 4e-3f), "rigid goal e%d q%d %f %f", e, q, h2f(A.goal[e][q]), h2f(B.goal[e][q]));
    // 사라짐 → 잃음(같은 자리), 스냅숏에 없음 → 호출자 마지막 자리, 없음 → 모름
    SmTokIn g = a;
    g.objs[2].state = 1;
    MapTok G;
    make_sm_tokens(g, &G);
    CHECK(h2f(G.goal[0][GV_LOST]) == 1.f && G.goal[0][GV_POS] == A.goal[0][GV_POS], "gone -> lost at last position");
    g = a;
    g.goal[GE_PICK].have_last = true;
    for (int k = 0; k < 3; ++k) g.goal[GE_PICK].last[k] = a.objs[2].pos[k];
    g.objs.erase(g.objs.begin() + 2);
    make_sm_tokens(g, &G);
    CHECK(h2f(G.goal[0][GV_LOST]) == 1.f && h2f(G.goal[0][GV_KNOWN]) == 1.f && G.goal[0][GV_POS] == A.goal[0][GV_POS], "missing -> caller's last position");
    g.goal[GE_PICK].have_last = false;
    g.goto_point = true;
    make_sm_tokens(g, &G);
    CHECK(h2f(G.goal[0][GV_PRESENT]) == 1.f && h2f(G.goal[0][GV_KNOWN]) == 0.f && G.goal[0][GV_POS] == 0 && (G.flags & 2), "unknown object / goto flag");
    n_cases += 1;
  }
  // 6. 격자(topview.h): 로봇 가운데·앞이 위로 읽은 점 분류(점유·빈칸·모름·물체 상자) + 덩이 격자 = 같은 표본
  {
    std::vector<int8_t> cells(200 * 200, -1);
    for (int y = 0; y < 200; ++y) for (int x = 0; x < 200; ++x) if (x >= 50 && x < 150 && y >= 50 && y < 150) cells[(size_t)y * 200 + x] = 0;   // 5 m 네모 빈칸(0.05 m 칸)
    for (int y = 100; y < 108; ++y) for (int x = 60; x < 140; ++x) cells[(size_t)y * 200 + x] = 100;   // 벽: map y 5.0–5.4, x 3–7(표본 간격 0.4 m 보다 두껍게)
    SmTokIn in;
    in.grid.cells = cells.data(); in.grid.w = 200; in.grid.h = 200; in.grid.res = 0.05; in.grid.ox = 0; in.grid.oy = 0;
    in.x = 5.f; in.y = 4.f; in.yaw = 1.5707963f;   // map +y 를 봄 → 벽은 앞 1.0–1.4 m
    SmTokObj cup; cup.id = 7; cup.pos[0] = 4.f; cup.pos[1] = 4.f; cup.ext[0] = cup.ext[1] = 0.2f; cup.ext[2] = 0.1f; in.objs.push_back(cup);   // 로봇 왼쪽 1 m
    TvIn t;
    sm_tv_input(in, t);
    const SmGridCell cell{&in.grid};
    auto at = [&](float fx, float ly) { return tv_class(t, cell, (int)std::floor(TV_HALF - ly / TV_RES), (int)std::floor(TV_HALF - fx / TV_RES)); };   // 로봇 좌표 → 분류
    CHECK(at(1.10f, 0.f) == TV_OBST && at(0.6f, 0.f) == TV_FREE && at(4.0f, 4.0f) == TV_UNEXP, "wall ahead = obstacle, free near, outside = unexplored");
    CHECK(at(0.f, 1.0f) == TV_OBST, "object box on the left = obstacle");
    MapTok T;
    make_sm_tokens(in, &T);
    int tot = 0;
    for (int by = 0; by < TV_B; ++by) for (int bx = 0; bx < TV_B; ++bx) { CHECK(T.tv[0][by][bx] + T.tv[1][by][bx] <= TV_BS * TV_BS, "grid range"); tot += T.tv[0][by][bx]; }
    CHECK(tot > 0 && T.tv[0][7][8] + T.tv[0][7][7] + T.tv[0][6][7] + T.tv[0][6][8] > 0, "grid sees the wall ahead (blocks above the centre)");
    n_cases += 1;
  }
  std::printf("sm_tok_test: %d cases, %d failures\n", n_cases, g_fail);
  return g_fail ? 1 : 0;
}
