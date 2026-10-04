// BEHAVIOR 집 장면 판(계획서 CURRICULUM_BEHAVIOR2026.md 5절 E2, 커리큘럼 B1–B3). env_soa.h 끝에서 include 된다.
// 상자 방(A0–A2 = B0) 코드는 그대로 두고, 같은 공용 조각(act_prepare·substeps·obs_body)에 장면 충돌·광선·성공 판정만 바꿔 붙인다.
//   - 판 리셋: 장치 커리큘럼 값(bsc::BCurr)으로 단계(B1/B2/B3)·장면·split 을 고르고, 호스트가 미리 검사한 시작 조건 표(bsc::Entry)에서 하나를 뽑는다.
//   - 충돌: 몸통 직사각형 대 정적 회전 상자(BK_COLL, 1 m 묶음으로 찾음) + 과제 물체 상자(움직이는 상자, 바닥 0.35 m 아래).
//   - 벽 광선 16: 같은 상자들(4 m). 보임: 카메라 → 목표 가운데 선분이 정적 상자(모든 높이)·다른 과제 물체에 가리지 않음.
//   - 다가가기 거리(보상·관측 74): **정책이 아는 지도**(지도 단계가 앞 스텝에 만든 거리장, bsc::NavFb)로. 지도가 안 붙었으면 직선 거리.
//     진행 보상 = 같은 거리장으로 잰 (지난 자리 거리 − 지금 자리 거리) — 거리장이 바뀌어도 한 스텝 안에서는 같은 퍼텐셜.
//   - 성공(단계마다, 아래 K B): B1 목표 방 안 + 목표 점 0.5 m + 멈춤 1 s, B2 보임 + 지도 확정 0.3 s, B3 잡는 점 작업 공간에 물체 상자 + 에임 10° + 멈춤 1 s.
// 관측 80 의 배치는 상자 방과 같다(목표 = B1 목표 점 / B2·B3 목표 물체 가운데).
#pragma once
#include "bscene.h"
#include "omx_workspace_grasp.h"

namespace env {

// BEHAVIOR 판 상수(가정 표시는 env.h K 와 같은 뜻)
struct KB {
  static constexpr float b1_r = 0.5f;          // B1: 목표 점 0.5 m 안(CURRICULUM_BEHAVIOR2026 3절)
  static constexpr float b1_tz = 0.05f;        // B1 목표 점 높이(보임 판정용, 가정)
  static constexpr int b2_ticks = 3;           // B2: 보임 + 확정 0.3 s (가정)
  static constexpr float seed_b1 = 0.15f;      // 거리장 씨앗 반경(지도 쪽과 같은 값): B1 점 둘레
  static constexpr float seed_obj = 0.45f;     // B2·B3 물체 둘레(팔 닿는 원, 가정: 옆 0.27 m + 몸통 반 폭 0.11 m 남짓)
  static constexpr float slack_b1 = 10.f, slack_b2 = 30.f, slack_b3 = 10.f;   // 시간 예산 = 참 최단 경로 / 0.3 m/s + 여유 s (가정: B2 는 찾느라 더)
};
DEV float seed_r_of(int kind) { return kind == bsc::EK_B1 ? KB::seed_b1 : KB::seed_obj; }

struct BState { float wx, wy, px, py, tz, ex[3], dist; int kind, scene, ent, room, fset, instr; };
DEV void load_b(const Soa& s, int i, BState& b) {
  const int N = s.N;
  b.wx = s.f[F_B_WX * N + i]; b.wy = s.f[F_B_WY * N + i]; b.px = s.f[F_B_PX * N + i]; b.py = s.f[F_B_PY * N + i];
  b.tz = s.f[F_B_TZ * N + i];
  for (int a = 0; a < 3; ++a) b.ex[a] = s.f[(F_B_EX0 + a) * N + i];
  b.dist = s.f[F_B_DIST * N + i];
  b.kind = s.iv[I_B_KIND * N + i]; b.scene = s.iv[I_B_SCENE * N + i]; b.ent = s.iv[I_B_ENT * N + i]; b.room = s.iv[I_B_ROOM * N + i];
  b.fset = s.iv[I_B_FSET * N + i]; b.instr = s.iv[I_B_INSTR * N + i];
}
DEV void store_b(const Soa& s, int i, const BState& b) {
  const int N = s.N;
  s.f[F_B_WX * N + i] = b.wx; s.f[F_B_WY * N + i] = b.wy; s.f[F_B_PX * N + i] = b.px; s.f[F_B_PY * N + i] = b.py;
  s.f[F_B_TZ * N + i] = b.tz;
  for (int a = 0; a < 3; ++a) s.f[(F_B_EX0 + a) * N + i] = b.ex[a];
  s.f[F_B_DIST * N + i] = b.dist;
  s.iv[I_B_KIND * N + i] = b.kind; s.iv[I_B_SCENE * N + i] = b.scene; s.iv[I_B_ENT * N + i] = b.ent; s.iv[I_B_ROOM * N + i] = b.room;
  s.iv[I_B_FSET * N + i] = b.fset; s.iv[I_B_INSTR * N + i] = b.instr;
}

// 판 리셋 B1: (장면, split) → 방 표에서 시작 조건 하나(로봇 시작 = 인스턴스 값)
DEV int pick_entry(const bsc::SceneSet& ss, const bsc::BCurr& cu, int list, uint64_t& rng) {
  int cand = 0;
  for (int sc = 0; sc < ss.nsc; ++sc) {
    if (!((cu.scene_mask >> sc) & 1u)) continue;
    for (int sp = 0; sp < 2; ++sp) {
      if (cu.split != 2 && cu.split != sp) continue;
      cand += ss.lcnt[sc][list][sp] > 0;
    }
  }
  if (cand == 0) return -1;
  int k = (int)(rand01(rng) * (float)cand);
  k = k >= cand ? cand - 1 : k;
  for (int sc = 0; sc < ss.nsc; ++sc) {
    if (!((cu.scene_mask >> sc) & 1u)) continue;
    for (int sp = 0; sp < 2; ++sp) {
      if (cu.split != 2 && cu.split != sp) continue;
      const int n = ss.lcnt[sc][list][sp];
      if (n <= 0) continue;
      if (k-- == 0) {
        int j = (int)(rand01(rng) * (float)n);
        j = j >= n ? n - 1 : j;
        return ss.loff[sc][list][sp] + j;
      }
    }
  }
  return -1;
}
DEV int rand_below(uint64_t& rng, int n) { int k = (int)(rand01(rng) * (float)n); return k >= n ? n - 1 : k; }
// 판 리셋 B2–B5(집기·놓기, 사용자 규칙): 모든 인스턴스(쓰는 장면·split) 중 거르개를 지나는 짝이 있는 것 하나(고르게) → 그 인스턴스의 집을 물체 하나
// (고르게) → 그 물체의 놓을 곳 하나(고르게). 엄격이면 엄격을 지나는 것만(표에서 앞쪽)
DEV int pick_pnp(const bsc::SceneSet& ss, const bsc::BCurr& cu, uint64_t& rng) {
  int tot = 0;
  for (int sc = 0; sc < ss.nsc; ++sc) {
    if (!((cu.scene_mask >> sc) & 1u)) continue;
    for (int sp = 0; sp < 2; ++sp) if (cu.split == 2 || cu.split == sp) tot += cu.strict ? ss.icnt_in[sc][sp] : ss.icnt[sc][sp];
  }
  if (tot == 0) return -1;
  int k = rand_below(rng, tot), inst = -1;
  for (int sc = 0; sc < ss.nsc && inst < 0; ++sc) {
    if (!((cu.scene_mask >> sc) & 1u)) continue;
    for (int sp = 0; sp < 2 && inst < 0; ++sp) {
      if (cu.split != 2 && cu.split != sp) continue;
      const int n = cu.strict ? ss.icnt_in[sc][sp] : ss.icnt[sc][sp];
      if (k < n) inst = ss.ioff[sc][sp] + k; else k -= n;
    }
  }
  const bsc::PnpInst& I = ss.pinst[inst];
  const int pk = I.pick_off + rand_below(rng, cu.strict ? I.npick_in : I.npick);
  const bsc::PnpPick& P = ss.ppick[pk];
  return P.ent_off + rand_below(rng, cu.strict ? P.n_in : P.n);
}
// 시작 자리 거르개(집기·놓기): 창 가장자리 0.8 m 안쪽, 로봇 중심 칸이 **창 안에서** 잡는 자세 칸에 닿음(이웃 바닥 높이 차 ≤ 문턱, 엄격이면 엄격 문턱 —
// 호스트 BFS 비트), 집을 물체에서 ≥ spawn_min_d, 그 yaw 로 몸통이 정적 상자·과제 물체에 안 닿음
struct KP {
  static constexpr float win_margin = 0.8f, spawn_min_d = 1.0f;   // (가정)
  static constexpr int spawn_tries = 64;
};
DEV int pnp_cell(const bsc::SceneDev& d, const bsc::Entry& E, float x, float y) {   // 창 좌표 → 장면 칸 번호(밖 −1)
  const int c0 = (int)floorf((E.wx - bsc::WIN_HALF - d.ox) * bsc::INV_CELL + 0.5f), r0 = (int)floorf((E.wy - bsc::WIN_HALF - d.oy) * bsc::INV_CELL + 0.5f);
  const int c = c0 + (int)floorf(x * bsc::INV_CELL) + bsc::WIN / 2, r = r0 + (int)floorf(y * bsc::INV_CELL) + bsc::WIN / 2;
  return (c < 0 || r < 0 || c >= d.W || r >= d.H) ? -1 : r * d.W + c;
}
DEV bool body_free_beh(const bsc::SceneSet& ss, const bsc::Entry& E, float x, float y, float yaw) {
  float s, co;
  sincosf_d(yaw, &s, &co);
  if (bsc::body_hits_scene(ss.sc[E.scene], x + E.wx, y + E.wy, s, co, K::half_len, K::half_wid)) return false;
  for (int p = 0; p < E.nprim; ++p) {
    const bsc::BPrim& P = E.prim[p];
    if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
    if (bsc::rect_hits_aabb(x, y, s, co, K::half_len, K::half_wid, P.lo, P.hi)) return false;
  }
  return true;
}
DEV bool spawn_ok(const bsc::SceneSet& ss, const bsc::Entry& E, float x, float y, float yaw, int strict, int nofilter) {
  if (!(absf(x) <= bsc::WIN_HALF - KP::win_margin && absf(y) <= bsc::WIN_HALF - KP::win_margin)) return false;
  const bsc::SceneDev& d = ss.sc[E.scene];
  const int ci = pnp_cell(d, E, x, y);
  if (ci < 0 || d.comp[ci] == 0) return false;
  if (!(nofilter & bsc::NF_SPAWN_REACH)) {   // 창 안에서 잡는 자세 칸에 닿는 칸(호스트가 BFS 로 만든 비트)
    const int off = strict ? E.rb_in : E.rb;
    if (off < 0) return false;
    const int wc = ((int)floorf(y * bsc::INV_CELL) + bsc::WIN / 2) * bsc::WIN + (int)floorf(x * bsc::INV_CELL) + bsc::WIN / 2;
    if (!((ss.rbits[off + (wc >> 5)] >> (wc & 31)) & 1u)) return false;
  }
  const float dx = x - E.gx, dy = y - E.gy;
  if (dx * dx + dy * dy < KP::spawn_min_d * KP::spawn_min_d) return false;
  return (nofilter & bsc::NF_SPAWN_FREE) || body_free_beh(ss, E, x, y, yaw);
}

DEV void reset_beh(Core& c, BState& b, const bsc::SceneSet& ss, const bsc::BCurr& cu) {
  const float u = rand01(c.rng);
  int kind = u < cu.p1 ? bsc::EK_B1 : (u < cu.p1 + cu.p2 ? bsc::EK_B2 : bsc::EK_B3);
  int e = kind == bsc::EK_B1 ? pick_entry(ss, cu, bsc::L_ROOM, c.rng) : pick_pnp(ss, cu, c.rng);
  if (e < 0) {   // 그 목록이 비면 다른 목록
    kind = kind == bsc::EK_B1 ? bsc::EK_B3 : bsc::EK_B1;
    e = kind == bsc::EK_B1 ? pick_entry(ss, cu, bsc::L_ROOM, c.rng) : pick_pnp(ss, cu, c.rng);
  }
  if (e < 0) { e = 0; kind = ss.ent[0].list == bsc::L_ROOM ? bsc::EK_B1 : bsc::EK_B3; }   // 설정이 아무것도 못 고름(호스트가 미리 막음) — 첫 판
  const bsc::Entry& E = ss.ent[e];
  c.tx = E.gx; c.ty = E.gy;
  if (kind == bsc::EK_B1) {   // B1: 인스턴스 로봇 시작
    c.x = E.sx; c.y = E.sy;
    c.yaw = cu.yaw_jit > 0.f ? wrap_pi(E.syaw + rand_range(c.rng, -cu.yaw_jit, cu.yaw_jit)) : E.syaw;
    b.fset = -1;
    b.instr = -1;
  } else {    // 집기·놓기: 무작위 시작(거르개를 지나는 칸·무작위 yaw), 못 찾으면 표의 대신 쓸 시작
    bool ok = false;
    float x = 0.f, y = 0.f, yaw = 0.f;
    const float R = bsc::WIN_HALF - KP::win_margin;
    for (int t = 0; t < KP::spawn_tries && !ok; ++t) {
      x = rand_range(c.rng, -R, R);
      y = rand_range(c.rng, -R, R);
      yaw = rand_range(c.rng, -kPi, kPi);
      ok = spawn_ok(ss, E, x, y, yaw, cu.strict, cu.nofilter);
    }
    if (!ok) { x = E.sx; y = E.sy; yaw = E.syaw; }
    c.x = x; c.y = y; c.yaw = yaw;
    b.fset = cu.strict ? 1 : 0;
    const int nt = cu.eval_instr ? ss.ntpl - ss.ntpl_train : ss.ntpl_train;
    const int t = rand_below(c.rng, nt > 0 ? nt : 1);
    b.instr = (E.combo >= 0 && nt > 0) ? E.combo * ss.ntpl + (cu.eval_instr ? ss.ntpl_train : 0) + t : -1;
  }
  c.rhx = bsc::WIN_HALF; c.rhy = bsc::WIN_HALF;   // 창 반 변(상자 방 값 자리 — 지도 완성도 계산만 씀)
  c.nf = 0;
  b.wx = E.wx; b.wy = E.wy;
  b.px = c.x; b.py = c.y;
  b.tz = kind == bsc::EK_B1 ? KB::b1_tz : E.gz;
  for (int a = 0; a < 3; ++a) b.ex[a] = E.ext[a];
  b.kind = kind; b.scene = E.scene; b.ent = e; b.room = E.groom;
  c.v = 0.f; c.w = 0.f; c.wl = 0.f; c.wr = 0.f;
  home_q(c.q);
  for (int i = 0; i < N_Q; ++i) c.qd[i] = 0.f;
  for (int i = 0; i < N_ACT; ++i) c.last_act[i] = 0.f;
  const float ddx = c.tx - c.x, ddy = c.ty - c.y;
  b.dist = sqrtf(ddx * ddx + ddy * ddy);
  c.prev_dist = b.dist;
  c.prev_aim = absf(wrap_pi(atan2f_d(ddy, ddx) - c.yaw));
  c.step = 0;
  c.ok_ticks = 0;
  c.seen = 0;
  const float slack = kind == bsc::EK_B1 ? KB::slack_b1 : kind == bsc::EK_B2 ? KB::slack_b2 : KB::slack_b3;
  // 시간 예산: B1 = 참 최단 경로 / 0.3 m/s + 여유, 집기·놓기 = 직선 × 1.5 / 0.3 m/s + 여유(무작위 시작이라 경로를 미리 모름 — 가정)
  const float plen = kind == bsc::EK_B1 ? E.path : 1.5f * b.dist;
  c.max_steps = (int)ceilf((plen / 0.3f + slack) * 10.f);
  c.ep += 1;
}

// 충돌: 정적 상자(세계) + 과제 물체 상자(창 좌표, 바닥 H_COLL 아래)
DEV bool collides_beh(const Core& c, const BState& b, const bsc::SceneSet& ss) {
  float s, co;
  sincosf_d(c.yaw, &s, &co);
  if (bsc::body_hits_scene(ss.sc[b.scene], c.x + b.wx, c.y + b.wy, s, co, K::half_len, K::half_wid)) return true;
  const bsc::Entry& E = ss.ent[b.ent];
  for (int p = 0; p < E.nprim; ++p) {
    const bsc::BPrim& P = E.prim[p];
    if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
    if (bsc::rect_hits_aabb(c.x, c.y, s, co, K::half_len, K::half_wid, P.lo, P.hi)) return true;
  }
  return false;
}
// 같은 판정을 스텝 시작에 모은 후보로(결과 같음): 서브스텝 10 번 동안 몸통 가운데는 v_max·dt·10 = 0.05 m 안에서만 움직이고, 몸통과 겹치는 상자는
// 가운데에서 외접원(0.194 m) 안에 있으므로 시작 자리에서 0.27 m 안의 상자만 볼 수 있다. 후보가 넘치면 모든 묶음 검사
constexpr float COLL_R = 0.27f;
DEV bool collides_beh_c(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::CollCand& cc) {
  if (cc.overflow) return collides_beh(c, b, ss);
  float s, co;
  sincosf_d(c.yaw, &s, &co);
  const bsc::SceneDev& d = ss.sc[b.scene];
  const float wx = c.x + b.wx, wy = c.y + b.wy;
  for (int q = 0; q < cc.n; ++q)
    if (bsc::rect_hits_obb(wx, wy, s, co, K::half_len, K::half_wid, d.box[cc.idx[q]])) return true;
  const bsc::Entry& E = ss.ent[b.ent];
  for (int p = 0; p < E.nprim; ++p) {
    const bsc::BPrim& P = E.prim[p];
    if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
    if (bsc::rect_hits_aabb(c.x, c.y, s, co, K::half_len, K::half_wid, P.lo, P.hi)) return true;
  }
  return false;
}
static_assert(K::v_max * K::dt * K::sub + 0.1942f + 0.02f <= COLL_R, "collision candidate radius covers one control step");

// 벽 광선 16(상자 방 wall_rays 와 같은 배치·/4 m): 정적 BK_COLL 상자 + 과제 물체 상자
DEV void wall_rays_beh(const Core& c, const BState& b, const bsc::SceneSet& ss, float out[N_RAYS]) {
  const bsc::SceneDev& d = ss.sc[b.scene];
  const bsc::Entry& E = ss.ent[b.ent];
  const float wx = c.x + b.wx, wy = c.y + b.wy;
  for (int i = 0; i < N_RAYS; ++i) {
    float s, co;
    sincosf_d(c.yaw + kTwoPi * (float)i * (1.0f / (float)N_RAYS), &s, &co);
    float t = 4.f;
    bsc::bin_walk(d, wx, wy, co, s,
                  [&](int j, float) { if (d.bkind[j] & bsc::BK_COLL) t = minf(t, bsc::ray_obb2(wx, wy, co, s, d.box[j], t)); },
                  [&]() { return t; });
    for (int p = 0; p < E.nprim; ++p) {
      const bsc::BPrim& P = E.prim[p];
      if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
      const float bb[4] = {P.lo[0], P.lo[1], P.hi[0], P.hi[1]};
      t = minf(t, ray_box2(c.x, c.y, co, s, bb, t));
    }
    out[i] = maxf(t, 0.f) * 0.25f;
  }
}
// 카메라(창 좌표 cx, cy, 높이 cam_z) → 목표 가운데(tx, ty, tz) 선분이 가리나: 정적 상자(모든 높이) + 다른 과제 물체 상자
DEV bool occluded_beh(const Core& c, const BState& b, const bsc::SceneSet& ss, float cx, float cy) {
  const float p[3] = {cx + b.wx, cy + b.wy, K::cam_z}, q[3] = {c.tx + b.wx, c.ty + b.wy, b.tz};
  const bsc::Entry& E = ss.ent[b.ent];
  const int except = b.kind == bsc::EK_B1 ? -1 : E.prim[0].sbox;   // 목표가 정적 상자면 자기 상자는 빼고
  if (bsc::seg_blocked_scene(ss.sc[b.scene], p, q, except)) return true;
  const float pl[3] = {cx, cy, K::cam_z}, v[3] = {c.tx - cx, c.ty - cy, b.tz - K::cam_z};
  for (int k = (b.kind == bsc::EK_B1 ? 0 : 1); k < E.nprim; ++k) {
    const bsc::BPrim& P = E.prim[k];
    if (P.sbox >= 0) continue;
    if (bsc::seg_hits_aabb3(pl, v, P.lo, P.hi)) return true;
  }
  return false;
}
// 잡는 점 작업 공간(omx_workspace_grasp.h, E0 잡는 점)에 물체 상자(가운데 ctr, 크기 ext — 창 좌표)의 (r, z) 범위가 겹치나.
// 지도 토큰의 omx_reach_box(map_tok.h)와 같은 식에 표만 잡는 점 표
DEV bool grasp_reach_box(const float ctr[3], const float ext[3], float px, float py, float c, float s) {
  const float axw = px + (c * omxwsg::AX - s * omxwsg::AY), ayw = py + (s * omxwsg::AX + c * omxwsg::AY);
  const float dx = ctr[0] - axw, dy = ctr[1] - ayw, d = sqrtf(dx * dx + dy * dy);
  const float rho = 0.5f * sqrtf(ext[0] * ext[0] + ext[1] * ext[1]);
  const float rmin = maxf(0.f, d - rho), rmax = d + rho;
  const float base_z = 0.15f;   // base_footprint → base_link (URDF base_joint) = gmap::MP::base_z
  const float zlo = ctr[2] - 0.5f * ext[2] - base_z, zhi = ctr[2] + 0.5f * ext[2] - base_z;
  int r0 = (int)floorf(rmin / omxwsg::RES), r1 = (int)floorf(rmax / omxwsg::RES);
  int z0 = (int)floorf((zlo - omxwsg::Z_LO) / omxwsg::RES), z1 = (int)floorf((zhi - omxwsg::Z_LO) / omxwsg::RES);
  if (r0 >= omxwsg::NR || z1 < 0 || z0 >= omxwsg::NZ) return false;
  r1 = r1 >= omxwsg::NR ? omxwsg::NR - 1 : r1;
  z0 = z0 < 0 ? 0 : z0;
  z1 = z1 >= omxwsg::NZ ? omxwsg::NZ - 1 : z1;
  const uint64_t hiw = r1 >= 63 ? ~0ull : ((1ull << (r1 + 1)) - 1ull), mask = hiw & ~((1ull << r0) - 1ull);
  for (int z = z0; z <= z1; ++z)
    if (omxwsg::row(z) & mask) return true;
  return false;
}
// 거리(보상·관측): 지도 거리장이 이 판 것이면 그것 + 씨앗 반경, 아니면 직선. 거리장에서 못 찾으면 −1
DEV float beh_dist(const Core& c, const BState& b, const uint8_t* lev, int org, float x, float y) {
  if (lev) {
    const float f = bsc::field_dist(lev, org, x, y);
    return f < 0.f ? -1.f : f + seed_r_of(b.kind);
  }
  const float dx = c.tx - x, dy = c.ty - y;
  return sqrtf(dx * dx + dy * dy);
}

// 한 제어 스텝(BEHAVIOR 판). lev: 이 판의 지도 거리장(앞 스텝 지도, 판 번호가 맞을 때만, 아니면 nullptr), conf: 목표 확정(−1 = 지도 없음 → B2 는 보임만)
template <class Hook>
DEV void step_core_beh(Core& c, BState& b, const bsc::SceneSet& ss, const uint8_t* lev, int org, int conf, const float act_in[N_ACT], StepOut& o, bool arm_free,
                       const Hook& hook) {
  float act[N_ACT], jerk, v_cmd, w_cmd, q_cmd[N_Q];
  act_prepare(c, act_in, act, jerk, v_cmd, w_cmd, q_cmd, arm_free);
  bsc::CollCand cand;
  bsc::coll_gather(ss.sc[b.scene], c.x + b.wx, c.y + b.wy, COLL_R, cand);
  const bool hit = substeps(c, v_cmd, w_cmd, q_cmd, hook, [&](const Core& cc) { return collides_beh_c(cc, b, ss, cand); });

  float tb[3], sn, cs;
  int n = obs_body(c, act, b.tz, o.obs, tb, sn, cs);
  const float cam_wx = c.x + cs * K::cam_x, cam_wy = c.y + sn * K::cam_x;
  const float cdx = c.tx - cam_wx, cdy = c.ty - cam_wy;
  const float cdist = sqrtf(cdx * cdx + cdy * cdy);
  const float aim_ang = wrap_pi(atan2f_d(cdy, cdx) - c.yaw);
  const float aim = absf(aim_ang);
  float surf = cdist;   // B1: 점까지. B2·B3: 목표 상자 바닥 자국까지 수평 거리
  if (b.kind != bsc::EK_B1) {
    const float gx = maxf(absf(cdx) - 0.5f * b.ex[0], 0.f), gy = maxf(absf(cdy) - 0.5f * b.ex[1], 0.f);
    surf = sqrtf(gx * gx + gy * gy);
  }
  const bool visible = aim < 0.5f * K::cam_hfov && !occluded_beh(c, b, ss, cam_wx, cam_wy);
  // 거리: 같은 거리장(또는 직선)으로 지난 자리·지금 자리
  const float dn = beh_dist(c, b, lev, org, c.x, c.y), dp = beh_dist(c, b, lev, org, b.px, b.py);
  const float prog = (dn >= 0.f && dp >= 0.f) ? dp - dn : 0.f;
  const float dist = dn >= 0.f ? dn : b.dist;

  float rays[N_RAYS];
  wall_rays_beh(c, b, ss, rays);
  for (int i = 0; i < N_RAYS; ++i) o.obs[n++] = rays[i];
  o.obs[n++] = tb[0]; o.obs[n++] = tb[1];
  o.obs[n++] = dist; o.obs[n++] = aim_ang;
  o.obs[n++] = visible ? 1.f : 0.f;
  o.obs[n++] = surf;
  o.obs[n++] = cdist;
  o.obs[n++] = 1.f;

  float r = 0.f;
  r = r + K::r_prog * prog;
  if (dist < 1.5f) r = r + K::r_aim * (c.prev_aim - aim);
  if (visible && !c.seen) { r = r + K::r_seen; c.seen = 1; }
  float wmin = 4.f;
  for (int i = 0; i < N_RAYS; ++i) wmin = minf(wmin, rays[i] * 4.f);
  const float clear = wmin - K::half_wid;
  if (clear < 0.15f) r = r + K::r_near * (0.15f - maxf(clear, 0.f)) * absf(c.v);
  r = r + K::r_jerk * jerk + K::r_time;

  const bool still = absf(c.v) <= K::succ_v && absf(c.w) <= K::succ_w;
  bool ok;
  int need = K::succ_ticks;
  if (b.kind == bsc::EK_B1) {
    const float gdx = c.tx - c.x, gdy = c.ty - c.y;
    ok = still && gdx * gdx + gdy * gdy <= KB::b1_r * KB::b1_r && bsc::room_at(ss.sc[b.scene], c.x + b.wx, c.y + b.wy) == b.room;
  } else if (b.kind == bsc::EK_B2) {
    ok = visible && conf != 0;
    need = KB::b2_ticks;
  } else {
    const float ctr[3] = {c.tx, c.ty, b.tz};
    ok = still && aim <= K::succ_aim && grasp_reach_box(ctr, b.ex, c.x, c.y, cs, sn);
  }
  c.ok_ticks = ok ? c.ok_ticks + 1 : 0;
  int done = kRunning;
  if (hit) { done = kCollision; r = r + K::r_coll; }
  else if (c.ok_ticks >= need) { done = kSuccess; r = r + K::r_succ + K::r_succ_aim * cosf_d(aim); }
  else if (c.step + 1 >= c.max_steps) done = kTimeout;
  c.prev_dist = dist;
  c.prev_aim = aim;
  b.px = c.x; b.py = c.y;
  b.dist = dist;
  for (int i = 0; i < N_ACT; ++i) c.last_act[i] = act[i];
  c.step += 1;
  o.reward = r;
  o.done = done;
}

// 판 하나(SoA): 처음 / 한 스텝(+ 끝났으면 새 판)
DEV void init_env_beh(const Soa& s, int i, uint64_t seed, const bsc::SceneSet& ss, const bsc::BCurr& cu) {
  Core c{};
  c.fu = furn_view(s, i);
  c.rng = seed * 0x9E3779B97F4A7C15ull + (uint64_t)i * 0xD1B54A32D192ED03ull + 12345ull;
  c.ep = 0;
  BState b{};
  reset_beh(c, b, ss, cu);
  store(s, i, c);
  store_b(s, i, b);
}
DEV void step_env_beh(const Soa& s, int i, const float* act, float* obs, float* rew, int* done, bool arm_free, int bug, const bsc::SceneSet& ss,
                      const bsc::BCurr* cu, const bsc::NavFb& fb) {
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  float a[N_ACT];
  for (int k = 0; k < N_ACT; ++k) a[k] = act[k * s.N + i];
  if (bug == 1) a[1] = -a[1];   // 음성 대조
  const bool fresh = fb.lev != nullptr && fb.tag[i] == c.ep;
  const uint8_t* lev = fresh ? fb.lev + (size_t)i * (bsc::NAV_P * bsc::NAV_P) : nullptr;
  const int org = fresh ? fb.org[i] : 0;
  const int conf = fb.conf == nullptr ? -1 : (fresh ? fb.conf[i] : 0);
  if (bug == 2) b.room = -2;   // 음성 대조(장면): B1 목표 방을 지움
  StepOut o;
  step_core_beh(c, b, ss, lev, org, conf, a, o, arm_free, NoHook{});
  for (int k = 0; k < N_OBS; ++k) obs[k * s.N + i] = o.obs[k];
  rew[i] = o.reward;
  done[i] = o.done;
  if (o.done != kRunning) reset_beh(c, b, ss, *cu);
  store(s, i, c);
  store_b(s, i, b);
}

}  // namespace env
