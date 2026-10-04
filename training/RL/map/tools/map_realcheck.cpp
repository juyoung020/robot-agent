// 진짜 scenemap 코드(읽기 전용, src/behavior-2026/src/scene_graph/scenemap)와 근사판을 같은 입력으로 맞춰 본다(CPU, 학습 경로 아님).
//   map_realcheck [N=64] [steps=600]
// 1) 점유 문턱: 진짜 OccGrid 한 칸을 맞음·빈칸으로 흔들며 export8 ≥ 65(kOccMin) 와 근사의 L ≥ MP::q_occ 가 늘 같은지.
// 2) 격자: CPU 참조판 지도로 G1 판을 몰면서 keyframe 마다 같은 가상 스캔(열 끝)을 같은 믿는 자세로 진짜 OccGrid::insert 에 넣고,
//    칸마다 로그 오즈·본 칸이 같은지.
// 3) 벽: (가) 근사 격자 그대로를 진짜 wallSegments + wallStateVector(sm_snap_wall_state 의 몸통)에 넣어 근사 선분·벽 벡터와 비교 —
//    벽 알고리즘만 본다. (나) 진짜 OccGrid 의 export8 로 같은 비교 — 격자까지 합친 판.
//    무시 영역은 근사판과 같은 규칙(capi.cpp refreshWalls)으로 근사 물체 칸에서 만든다. 근사 벽 벡터는 토큰(FP16)을 풀어 쓴다.
// 4) 들기: 진짜 C ABI(LIMO)로 컵 둘을 확정시키고 팔·그리퍼·베이스를 움직이며, 같은 확정 자리·같은 slam 자세·같은 관절로 근사 hands_step 을
//    돌려 든 물체·상태·자리를 스텝마다 비교. 순기구학 팔 끝(근사 env::fk + base_z 대 sm_robot_fk)도 같이 본다.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"
#include "scenemap.h"
#include "scenemap/grid.hpp"
#include "scenemap/scan.hpp"
#include "scenemap/walls.hpp"

using namespace env;
namespace sm = scenemap;

static int g_fail = 0;
#define CHECK(c, ...) do { const bool ok_ = (c); std::printf("%s: ", ok_ ? "ok  " : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); g_fail += !ok_; } while (0)

// grid.cpp 의 보이는 값 표와 같은 식(근사 L 로 (가) 격자를 만들 때만 씀 — 1) 에서 진짜 OccGrid 로 확인)
static int8_t c8_of(int L) {
  const float l = float(L) / 256.f;
  return int8_t(std::lround(100.f * (1.f / (1.f + std::exp(-l)))));
}

static void check_threshold() {
  sm::GridParams gp;
  gp.res = 0.10f;
  sm::OccGrid g(gp);
  sm::Scan2 hit, miss;
  hit.ox = 0; hit.oy = 0; hit.hx = {1.05f}; hit.hy = {0.05f};   // 칸 (10, 0) 맞음
  miss.ox = 0; miss.oy = 0; miss.fx = {1.55f}; miss.fy = {0.05f};  // 칸 (10, 0) 을 지나는 빈 광선
  uint64_t r = 12345;
  long n = 0, bad = 0, nocc = 0;
  for (int k = 0; k < 20000; ++k) {
    const bool h = dm::rand01(r) < 0.45f;
    g.insert(h ? hit : miss, sm::Pose2{});
    const int Lq = int(std::lround(g.logOdds(10, 0) * 256.f));
    const std::vector<int8_t> c = g.export8();
    const int8_t v = c[size_t(0 - g.y0()) * g.width() + (10 - g.x0())];
    const bool real_occ = v >= sm::kOccMin, ours = Lq >= gmap::MP::q_occ;
    bad += real_occ != ours;
    nocc += real_occ;
    bad += c8_of(Lq) != v;   // 표 식도 같은지
    ++n;
  }
  CHECK(bad == 0, "occupancy threshold: real export8 >= %d vs approx L >= %d over %ld hit/miss updates (%ld occupied): %ld differ", sm::kOccMin,
        gmap::MP::q_occ, n, nocc, bad);
}

struct WallCmp { long bad_idx[3] = {0, 0, 0}; long tie_swap = 0; long n = 0, seg_set_eq = 0, seg_n = 0, seg_n_eq = 0, vec_eq = 0, ray_eq = 0, rays = 0; double ray_max = 0, seg_max = 0; };

// 근사 선분(가로 다음 세로, 세계 m) 목록
static std::vector<sm::WallSeg> approx_segs(const gmap::MapCore& m, const int16_t* segs) {
  std::vector<sm::WallSeg> v;
  for (int k = 0; k < m.nseg_h + m.nseg_v; ++k) {
    const int16_t* q = gmap::seg_at(segs, m, k);
    double w[4];
    for (int a = 0; a < 4; ++a) w[a] = (q[a] * 0.5 + gmap::GX0) * double(gmap::RES);
    v.push_back({w[0], w[1], w[2], w[3]});
  }
  return v;
}

static void compare_walls(const sm::WallGrid& g, const gmap::MapCore& m, const int16_t* segs, const gmap::MapTok& tok, WallCmp& c, bool seg_cmp) {
  std::vector<sm::WallRect> rects;
  for (int b = 0; b < gmap::KSLOT; ++b) {   // capi.cpp refreshWalls 와 같은 규칙(근사 물체 칸에서)
    const gmap::Slot& S = m.slot[b];
    if (!S.valid || !S.confirmed || S.held || S.state == gmap::S_GONE) continue;
    double lo[3], hi[3];
    for (int a = 0; a < 3; ++a) { lo[a] = S.pos[a] - 0.5 * S.ext[a]; hi[a] = S.pos[a] + 0.5 * S.ext[a]; }
    if (lo[2] > 0.4) continue;
    if (hi[0] - lo[0] > 5.0 || hi[1] - lo[1] > 5.0) continue;
    rects.push_back({lo[0] - 0.1, lo[1] - 0.1, hi[0] + 0.1, hi[1] + 0.1});
  }
  const auto real = sm::wallSegments(g, sm::kMinLen, sm::kMaxThick, 0.6, &rects);
  const double pose[3] = {m.ex, m.ey, m.eyaw};
  float vec[sm::kStateLen];
  sm::wallStateVector(g, real, pose, vec);
  ++c.n;
  if (seg_cmp) {
    const auto ap = approx_segs(m, segs);
    c.seg_n += (long)real.size();
    bool eq = ap.size() == real.size();
    for (size_t k = 0; k < std::min(ap.size(), real.size()); ++k) {
      const double d = std::max({std::fabs(ap[k].ax - real[k].ax), std::fabs(ap[k].ay - real[k].ay), std::fabs(ap[k].bx - real[k].bx), std::fabs(ap[k].by - real[k].by)});
      c.seg_max = std::max(c.seg_max, d);
      eq = eq && d < 1e-4;
    }
    c.seg_set_eq += eq;
    c.seg_n_eq += ap.size() == real.size();
  }
  bool veq = true;
  static int shown = 0;
  for (int i = 0; i < sm::kSectors; ++i) {   // 광선 16
    const float a = gmap::h2f(tok.wall[i]);
    const double d = std::fabs(a - vec[i]);
    ++c.rays;
    c.ray_max = std::max(c.ray_max, d);
    c.ray_eq += d < 2e-3;   // FP16 반올림(1 에서 2^-11) 안
    if (d >= 2e-3) {
      veq = false; ++c.bad_idx[0];
      if (shown < 4) { ++shown; std::printf("    ray diff: sector %d approx %.4f real %.4f (x 4 m)\n", i, a, vec[i]); }
    }
  }
  // 가까운 선분 8: 거리가 똑같은 선분끼리는 순서가 정해져 있지 않다(std::partial_sort) → 줄을 집합으로 맞춘다
  std::vector<int> used(sm::kSegments, 0);
  for (int j = 0; j < sm::kSegments; ++j) {
    int hit = -1;
    for (int q = 0; q < sm::kSegments && hit < 0; ++q) {
      if (used[q]) continue;
      bool eq = true;
      for (int k = 0; k < 5; ++k) eq = eq && std::fabs(gmap::h2f(tok.wall[sm::kSectors + 5 * j + k]) - vec[sm::kSectors + 5 * q + k]) < 2e-3;
      if (eq) hit = q;
    }
    if (hit < 0) { veq = false; ++c.bad_idx[1]; } else { used[hit] = 1; c.tie_swap += hit != j; }
  }
  c.vec_eq += veq;
}

static void check_grid_walls(int N, int T) {
  CpuEnv cenv(N, 1, 20261004);
  gmap::CpuMap cmap(N, 99);
  std::vector<float> act((size_t)N_ACT * N), obs, rew;
  std::vector<int> done;
  uint64_t arng = 777;
  std::vector<sm::OccGrid*> real(N, nullptr);
  std::vector<int> ep(N, -1);
  long cells = 0, cell_L_bad = 0, cell_seen_bad = 0, cell_occ_bad = 0, kfs = 0;
  WallCmp same, full;
  for (int t = 0; t < T; ++t) {
    for (size_t i = 0; i < act.size(); ++i) act[i] = dm::rand_range(arng, -1.f, 1.f);
    for (int i = 0; i < N; ++i) { act[0 * N + i] = dm::rand_range(arng, -0.2f, 1.f); act[1 * N + i] = dm::rand_range(arng, -0.6f, 0.6f); }
    if (t > 0) for (int i = 0; i < N; i += 2) { float a[N_ACT]; approach_action(obs.data(), N, i, a); act[0 * N + i] = a[0]; act[1 * N + i] = a[1]; }
    cenv.step(act, obs, rew, done);
    Soa cs{cenv.f.data(), cenv.iv.data(), cenv.rng.data(), N};
    cmap.step(cs);
    for (int i = 0; i < N; ++i) {
      const gmap::MapCore& m = cmap.h.core[i];
      if (m.ep != ep[i]) {   // 새 판: 진짜 격자도 새로
        delete real[i];
        sm::GridParams gp;
        gp.res = gmap::RES;
        real[i] = new sm::OccGrid(gp);
        ep[i] = m.ep;
      }
      if (!m.kf_flag) continue;
      ++kfs;
      // 같은 가상 스캔(참 카메라에서 쏜 열 끝, 베이스 기준) → 진짜 insert, 자세 = 이 keyframe 에 근사가 쓴 믿는 자세
      gmap::Scratch sh;
      const gmap::EnvView e = gmap::read_env(cs, i);
      gmap::phase_cast(m, sh, e, 0, 1, gmap::bctx(nullptr, nullptr));
      sm::Scan2 sc;
      sc.ox = env::K::cam_x; sc.oy = 0.f;
      const gmap::Cam ck = gmap::cam_consts();
      for (int col = 0; col < gmap::NCOL; ++col) {
        const float xn = ((float)(2 * col + 1 - gmap::NCOL) / (float)gmap::NCOL) * ck.tanh;
        const float t = sh.colt_t[col];
        for (int d = 0; d < 2; ++d) {   // 수직면 맞추기 점 → 진짜 Scan2 의 맞추기 점(mx, my)
          const float td = d ? sh.cold1[col] : sh.cold0[col];
          if (td == 0.f || (d && sh.cold1[col] == sh.cold0[col])) continue;
          sc.mx.push_back(env::K::cam_x + td); sc.my.push_back(-xn * td);
        }
        if (sh.colt[col] == 1) { sc.hx.push_back(env::K::cam_x + t); sc.hy.push_back(-xn * t); }
        else if (sh.colt[col] == 2) { sc.fx.push_back(env::K::cam_x + t); sc.fy.push_back(-xn * t); }
      }
      real[i]->insert(sc, sm::Pose2{m.ex, m.ey, m.eyaw});
      // 칸 비교(근사 창 안)
      const sm::OccGrid& g = *real[i];
      const std::vector<int8_t> c8 = g.export8();
      const int16_t* L = cmap.h.L.data() + (size_t)i * gmap::NCELL;
      const uint32_t* seen = cmap.h.seen.data() + (size_t)i * gmap::NWORD;
      const uint32_t* occ = cmap.h.occ.data() + (size_t)i * gmap::NWORD;
      std::vector<int8_t> mine((size_t)gmap::NCELL);
      for (int ly = 0; ly < gmap::GW; ++ly)
        for (int lx = 0; lx < gmap::GW; ++lx) {
          const int idx = ly * gmap::GW + lx, ix = lx + gmap::GX0, iy = ly + gmap::GX0;
          const bool s_me = (seen[idx >> 5] >> (idx & 31)) & 1u, o_me = (occ[idx >> 5] >> (idx & 31)) & 1u;
          mine[idx] = s_me ? c8_of(L[idx]) : int8_t(-1);
          const bool s_re = g.seen(ix, iy);
          const int Lre = int(std::lround(g.logOdds(ix, iy) * 256.f));
          int8_t v = -1;
          if (g.inside(ix, iy)) v = c8[size_t(iy - g.y0()) * g.width() + (ix - g.x0())];
          ++cells;
          cell_seen_bad += s_me != s_re;
          cell_L_bad += (s_me || s_re) && Lre != L[idx];
          cell_occ_bad += o_me != (v >= sm::kOccMin);
        }
      // (가) 근사 격자 그대로
      const sm::WallGrid gw{mine.data(), gmap::GW, gmap::GW, double(gmap::RES), gmap::GX0 * double(gmap::RES), gmap::GX0 * double(gmap::RES)};
      compare_walls(gw, m, cmap.h.segs.data() + (size_t)i * gmap::SEGW, cmap.h.tok[i], same, true);
      // (나) 진짜 격자
      const sm::WallGrid gr{c8.data(), g.width(), g.height(), double(g.res()), g.x0() * double(g.res()), g.y0() * double(g.res())};
      compare_walls(gr, m, cmap.h.segs.data() + (size_t)i * gmap::SEGW, cmap.h.tok[i], full, false);
    }
  }
  for (auto* p : real) delete p;
  std::printf("grid: %ld keyframes, %ld cell checks: log-odds differ %ld (%.4f %%), seen differ %ld (%.4f %%), occupied differ %ld (%.4f %%)\n", kfs, cells,
              cell_L_bad, 100.0 * cell_L_bad / cells, cell_seen_bad, 100.0 * cell_seen_bad / cells, cell_occ_bad, 100.0 * cell_occ_bad / cells);
  std::printf("walls (a) same grid: %ld keyframes, segment lists equal %ld (%.2f %%), segment count equal %.2f %%, max endpoint diff %.2e m, segments/keyframe %.2f\n",
              same.n, same.seg_set_eq, 100.0 * same.seg_set_eq / same.n, 100.0 * same.seg_n_eq / same.n, same.seg_max, (double)same.seg_n / same.n);
  std::printf("    wall vector (56) equal within FP16 (segment rows as a set): %.2f %%; rays within 2e-3: %.3f %% (max |diff| %.4f of 4 m)\n", 100.0 * same.vec_eq / same.n,
              100.0 * same.ray_eq / same.rays, same.ray_max);
  std::printf("    differing: rays %ld of %ld, nearest-8 segment rows without a match %ld (rows equal but in another order at distance ties: %ld)\n",
              same.bad_idx[0], same.rays, same.bad_idx[1], same.tie_swap);
  std::printf("walls (b) real OccGrid: wall vector equal %.2f %%; rays within 2e-3: %.3f %% (max |diff| %.4f)\n", 100.0 * full.vec_eq / full.n,
              100.0 * full.ray_eq / full.rays, full.ray_max);
  CHECK(same.seg_set_eq == same.n, "wall segments on the same grid equal the real wallSegments (%ld / %ld)", same.seg_set_eq, same.n);
}

// ---- 4) 들기 --------------------------------------------------------------------------------------------------------------
namespace grasp {
constexpr int W = 160, H = 120;
const double FX = 80.0 / std::tan(71.0 / 2 * M_PI / 180), FY = FX, CX = 79.5, CY = 59.5;
struct Box { double lo[3], hi[3]; };
std::vector<Box> cups;
struct Pose { double x, y, th; };

std::vector<float> proprio(const Pose& p, const float arm[5], float grip) {
  std::vector<float> q(12, 0.f);
  q[SM_LIMO_ODOM_X] = float(p.x); q[SM_LIMO_ODOM_Y] = float(p.y); q[SM_LIMO_ODOM_YAW] = float(p.th);
  for (int k = 0; k < 5; ++k) q[SM_LIMO_ARM_Q + k] = arm[k];
  q[SM_LIMO_GRIPPER] = grip;
  return q;
}
// 깊이 + 컵마다 마스크(벽 x = 2.5, y = ±1.5, 바닥)
void render(const double T[12], std::vector<float>& depth, std::vector<std::vector<uint8_t>>& mask) {
  depth.assign(size_t(W) * H, 0.f);
  mask.assign(cups.size(), std::vector<uint8_t>(size_t(W) * H, 0));
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const double d[3] = {(u - CX) / FX, (v - CY) / FY, 1.0};
      double o[3], r[3];
      for (int i = 0; i < 3; ++i) { o[i] = T[i * 4 + 3]; r[i] = T[i * 4] * d[0] + T[i * 4 + 1] * d[1] + T[i * 4 + 2] * d[2]; }
      double best = 1e9;
      int who = -1;
      if (r[0] > 1e-9) best = std::fmin(best, (2.5 - o[0]) / r[0]);
      if (r[1] > 1e-9) best = std::fmin(best, (1.5 - o[1]) / r[1]);
      if (r[1] < -1e-9) best = std::fmin(best, (-1.5 - o[1]) / r[1]);
      if (r[2] < -1e-9) best = std::fmin(best, -o[2] / r[2]);
      for (size_t c = 0; c < cups.size(); ++c) {
        double t0 = 0, t1 = 1e9;
        bool hit = true;
        for (int i = 0; i < 3 && hit; ++i) {
          if (std::fabs(r[i]) < 1e-12) { hit = o[i] >= cups[c].lo[i] && o[i] <= cups[c].hi[i]; continue; }
          double a = (cups[c].lo[i] - o[i]) / r[i], b = (cups[c].hi[i] - o[i]) / r[i];
          if (a > b) std::swap(a, b);
          t0 = std::fmax(t0, a); t1 = std::fmin(t1, b);
          hit = t0 <= t1;
        }
        if (hit && t0 > 0 && t0 < best) { best = t0; who = int(c); }
      }
      if (best < 8.0) depth[size_t(v) * W + u] = float(best);
      if (who >= 0) mask[who][size_t(v) * W + u] = 1;
    }
}
}  // namespace grasp

static void check_grasp() {
  using namespace grasp;
  sm_ctx* c = sm_create("{\"robot\": \"limo_omx\"}");
  const char* labels[] = {"cup"};
  sm_set_labels(c, labels, 1);
  const float home[5] = {0.f, 1.3f, -1.9f, 0.7f, 0.f};
  const float reach[5] = {0.f, 0.9f, 0.2f, 0.4f, 0.f};
  // 팔 끝(근사 env::fk + base_z) 대 진짜 sm_robot_fk
  double fk_max = 0;
  for (const float* a : {home, reach}) {
    std::vector<float> q = proprio({0, 0, 0}, a, 0.f);
    sm_body_fk f;
    sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &f);
    float qq[N_Q] = {a[0], a[1], a[2], a[3], a[4], 0.f}, qd[N_Q] = {};
    env::Fk ef;
    env::fk(qq, qd, ef);
    // 잡는 점(scenemap d58c978: T_eef = URDF grasp_point) = 팔 끝 + grasp_off · 링크 x(map.h hands_step 과 같은 식)
    const double me[3] = {ef.ee_p[0] + gmap::MP::grasp_off * ef.ee_R[0], ef.ee_p[1] + gmap::MP::grasp_off * ef.ee_R[3],
                          ef.ee_p[2] + gmap::MP::grasp_off * ef.ee_R[6] + gmap::MP::base_z};
    for (int k = 0; k < 3; ++k) fk_max = std::max(fk_max, std::fabs(me[k] - f.T_eef[0][k * 4 + 3]));
  }
  CHECK(fk_max < 1e-5, "grasp point: approx env::fk + grasp_off + base_z vs real sm_robot_fk T_eef, max diff %.2e m", fk_max);
  // 컵 둘: 앞으로 뻗은 팔 끝 바로 아래(컵 A, 8 cm) 와 옆 7 cm(컵 B) — A 가 더 가까움
  sm_body_fk fr;
  { std::vector<float> q = proprio({0, 0, 0}, reach, 0.f); sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &fr); }
  const double ex = fr.T_eef[0][3], ey = fr.T_eef[0][7];
  // 컵 4 cm(잡기 확인: 가운데 변 ≤ 6 cm, 0.41 rad 로 쥐면 틈 4 cm)
  cups = {{{ex - 0.02, ey - 0.02, 0.0}, {ex + 0.02, ey + 0.02, 0.10}}, {{ex - 0.01, ey + 0.13, 0.0}, {ex + 0.03, ey + 0.17, 0.10}}};
  double t = 0;
  std::vector<float> depth;
  std::vector<std::vector<uint8_t>> mask;
  // 근사 쪽: 진짜 확정 물체 자리를 칸에 넣고 같은 관절·같은 slam 자세로 hands_step
  gmap::MapCore m{};
  m.held_slot = -1;
  m.t = 1;
  std::vector<sm_object> last;
  long steps = 0, bad_state = 0, bad_held = 0, grasps_real = 0, grasps_me = 0;
  double pos_max = 0;
  auto step = [&](const Pose& p, const float arm[5], float grip, bool cmp) {
    t += 0.1;
    std::vector<float> q = proprio(p, arm, grip);
    sm_proprio pp{t, q.data(), 12};
    sm_push_proprio(c, &pp);
    sm_body_fk fk;
    sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &fk);
    double T[12];
    const double cs = std::cos(p.th), sn = std::sin(p.th);
    for (int k = 0; k < 4; ++k) { T[k] = cs * fk.T_cam[0][k] - sn * fk.T_cam[0][4 + k]; T[4 + k] = sn * fk.T_cam[0][k] + cs * fk.T_cam[0][4 + k]; T[8 + k] = fk.T_cam[0][8 + k]; }
    T[3] += p.x; T[7] += p.y;
    render(T, depth, mask);
    std::vector<std::vector<uint32_t>> bits(cups.size(), std::vector<uint32_t>((size_t(W) * H + 31) / 32, 0));
    std::vector<int32_t> cls;
    std::vector<float> score, box;
    std::vector<uint32_t> allbits;
    for (size_t k = 0; k < cups.size(); ++k) {
      int x0 = W, y0 = H, x1 = -1, y1 = -1, n = 0;
      for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u)
          if (mask[k][size_t(v) * W + u]) { const size_t j = size_t(v) * W + u; bits[k][j >> 5] |= 1u << (j & 31); x0 = std::min(x0, u); y0 = std::min(y0, v); x1 = std::max(x1, u); y1 = std::max(y1, v); ++n; }
      if (n) {
        cls.push_back(0); score.push_back(0.9f); box.insert(box.end(), {float(x0), float(y0), float(x1 + 1), float(y1 + 1)});
        allbits.insert(allbits.end(), bits[k].begin(), bits[k].end());   // 마스크 여러 개 = 검출 순서로 이어 붙인 비트
      }
    }
    sm_detections d{};
    d.stamp = t; d.cam = 0; d.img_w = W; d.img_h = H; d.n = int(cls.size());
    d.cls = cls.data(); d.score = score.data(); d.box = box.data();
    d.mask_w = W; d.mask_h = H; d.mask_sx = 1; d.mask_sy = 1; d.mask_ox = 0; d.mask_oy = 0; d.mask_bits = allbits.data();
    sm_image im{t, 0, W, H, nullptr, depth.data(), float(FX), float(FY), float(CX), float(CY)};
    sm_push_image(c, &im, d.n ? &d : nullptr);
    if (!cmp) return;
    sm_snapshot_t* s = nullptr;
    sm_snapshot(c, &s);
    const sm_pose2 P = sm_snap_pose(s);
    const sm_object* o = nullptr;
    const int n = sm_snap_objects(s, &o);
    // 근사: 처음 비교 스텝에 진짜 확정 물체로 칸을 채움
    if (steps == 0) {
      std::printf("grasp: real confirmed objects before the arm moves: %d\n", n);
      for (int k = 0; k < n && k < gmap::KSLOT; ++k) {
        gmap::Slot& S = m.slot[k];
        S.valid = 1; S.confirmed = 1; S.id = int(o[k].id); S.state = o[k].state;
        for (int a = 0; a < 3; ++a) { S.pos[a] = float(o[k].pos[a]); S.ext[a] = float(o[k].extent[a]); }
      }
      m.closed = grip < gmap::MP::grip_closed;
      m.tried = 1;                 // 진짜는 앞 스텝(홈, 닫힘)에서 이미 한 번 골랐음(잡을 것 없음)
      m.gref = grip; m.gref_t = m.t - 10;
    }
    m.t += 1;
    // 들지 않은 물체는 지난 스냅숏 자리·상태로 맞춤(진짜도 이 스텝 들기 규칙은 지난 영상까지의 자리로 판단: integrate → updateHands → 영상)
    for (const auto& q : last) for (int b = 0; b < gmap::KSLOT; ++b) {
      gmap::Slot& S = m.slot[b];
      if (!S.valid || S.held || S.id != int(q.id)) continue;
      for (int a = 0; a < 3; ++a) S.pos[a] = float(q.pos[a]);
      S.state = q.state;
    }
    last.assign(o, o + n);
    m.ex = float(P.x); m.ey = float(P.y); m.eyaw = float(P.yaw);
    gmap::EnvView e{};
    for (int k = 0; k < 5; ++k) e.q[k] = arm[k];
    e.q[5] = grip;
    const int held_before = m.held_slot;
    gmap::hands_step(m, e);
    grasps_me += m.held_slot >= 0 && held_before < 0;
    ++steps;
    int real_held = -1;
    for (int k = 0; k < n; ++k) {
      if (o[k].state == SM_HELD) real_held = int(o[k].id);
      for (int b = 0; b < gmap::KSLOT; ++b) {
        const gmap::Slot& S = m.slot[b];
        if (!S.valid || S.id != int(o[k].id)) continue;
        bad_state += S.state != o[k].state;
        double dd = 0;
        for (int a = 0; a < 3; ++a) dd = std::max(dd, std::fabs(double(S.pos[a]) - o[k].pos[a]));
        if (std::getenv("RC_DEBUG")) std::printf("    step %ld id %u state %d/%d held %d diff %.2e pos %.4f %.4f %.4f real %.4f %.4f %.4f\n", steps, o[k].id, S.state,
                                                 o[k].state, S.held, dd, S.pos[0], S.pos[1], S.pos[2], o[k].pos[0], o[k].pos[1], o[k].pos[2]);
        pos_max = std::max(pos_max, dd);
      }
    }
    const int me_held = m.held_slot >= 0 ? m.slot[m.held_slot].id : -1;
    bad_held += me_held != real_held;
    static int last_real = -1;
    grasps_real += real_held >= 0 && last_real < 0;
    last_real = real_held;
    sm_snapshot_release(s);
  };
  const Pose A{0, 0, 0};
  for (int k = 0; k <= 6; ++k) step({-0.6 + 0.1 * k, 0, 0}, home, 0.f, false);   // 뒤(깊이 0.3 m 밖)에서 다가가며 확정(서로 다른 keyframe)
  step(A, home, 0.f, true);
  for (int k = 0; k < 3; ++k) step(A, reach, 1.2f, true);   // 열기(놓을 것 없음)
  for (int k = 0; k < 4; ++k) step(A, reach, 0.41f, true);   // 닫기(4 cm 쥠) → 멈춘 뒤 가까운 컵 A 를 듦
  for (int k = 1; k <= 10; ++k) {                             // 들고 0.3 m 앞·0.2 rad 돌기(렌더 컵도 손과 함께)
    const Pose p{0.03 * k, 0, 0.02 * k};
    sm_body_fk fk;
    { std::vector<float> q = proprio(p, reach, 0.41f); sm_robot_fk(SM_ROBOT_LIMO_OMX, q.data(), 12, &fk); }
    const double cs = std::cos(p.th), sn = std::sin(p.th);
    const double wx = p.x + cs * fk.T_eef[0][3] - sn * fk.T_eef[0][7], wy = p.y + sn * fk.T_eef[0][3] + cs * fk.T_eef[0][7];
    cups[0] = {{wx - 0.02, wy - 0.02, 0.0}, {wx + 0.02, wy + 0.02, 0.10}};
    step(p, reach, 0.41f, true);
  }
  const Pose C{0.30, 0, 0.20};
  for (int k = 0; k < 2; ++k) step(C, reach, 1.2f, true);   // 놓기 → 옮겨짐
  for (int k = 0; k < 3; ++k) step(C, reach, 0.41f, true);  // 다시 쥠(같은 컵, 잡는 점 바로 아래) → 듦
  for (int k = 0; k < 4; ++k) step(C, reach, 0.f, true);    // 끝까지 닫힘(틈 < 5 mm) → 놓침
  for (int k = 0; k < 2; ++k) step(C, reach, 1.2f, true);   // 열기
  for (int k = 0; k < 4; ++k) step(C, home, 0.f, true);     // 빈손으로 닫힘: 잡는 점 0.12 m 안 물체 없음 → 들지 않음
  sm_destroy(c);
  std::printf("grasp: %ld compared steps, grasps real %ld / approx %ld; held object differs %ld steps, object state differs %ld, max position diff %.2e m\n",
              steps, grasps_real, grasps_me, bad_held, bad_state, pos_max);
  // 자리 차이는 든 물체를 돌며 옮길 때만 생긴다: 진짜는 proprio 적분 시점·keyframe 시점의 자세로 따라가고, 여기는 스냅숏 자세를 넣는다(규칙 차이 아님)
  CHECK(bad_held == 0 && bad_state == 0 && pos_max < 2e-3 && grasps_real > 0, "grasp rule equals real objmap updateHands (LIMO grasp_check: grasp point, settle 0.2 s, holdable, release on full close)");
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 64, T = argc > 2 ? std::atoi(argv[2]) : 600;
  check_threshold();
  check_grid_walls(N, T);
  check_grasp();
  std::printf(g_fail ? "FAILED %d\n" : "all ok\n", g_fail);
  return g_fail ? 1 : 0;
}
