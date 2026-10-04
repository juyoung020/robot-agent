// 집기·놓기 판 고르기 확인(사용자 규칙, 문서 CURRICULUM_BEHAVIOR2026 B3–B5 거르개 표): 판 N 개(기본 10,000)를 환경 리셋으로 뽑고(CPU 참조판,
// 같은 씨앗으로 GPU 도 뽑아 상태가 비트로 같은지), 판마다 거르개를 **원본 RASC 값에서 다시** 잰다(표 만들 때 쓴 값이 아니라).
//   pnp_check [N=10000] [--strict] [--dir RASC] [--negative NAME]
//   NAME = free_area | in_closed | artic | spawn_reach | spawn_free | dst_reach | stance — 그 거르개만 끄고 뽑는다. 그러면 위반이 나와야 한다(종료 코드 0)
// 다시 재는 것(판마다): 집을 물체 — 제외 플래그(벽·바닥·문·창·계단·카펫·로봇·입자·와일드카드)·고정·관절체·상자 있음, 가로 최소 폭 ≤ max_w,
//   종류 평균 질량 ≤ max_mass(없으면 통과), 바닥 높이 ≤ pick_z, 바닥이 topdown_z 위면 옆 잡기: 출발 받침이 면이고 가장자리까지 ≤ edge_dist(엄격),
//   닫힌 곳 안 아님(RASC PK_IN_CLOSED). 놓을 곳 — 면: 윗면 min_top–place_top·작은 변 ≥ min_side, 용기: 윗면 ≤ place_top + inside_margin,
//   닫힌 관절체·선반 종류 아님, 출발 받침과 다름, 빈 넓이 ≥ (가로 + 2·free_margin)(세로 + 2·free_margin).
//   시작 — 창 안 0.8 m, 몸통 안 닿음, 물체에서 ≥ 1 m, 창 안 BFS(로봇 중심 칸, 바닥 높이 차 ≤ 문턱[느슨|엄격])로 잡는 자세 칸(잡는 점 작업 공간 +
//   몸통 안 닿음, 물체에서 stance_r 안)과 놓을 곳 가장자리에 닿는 칸(닿는 거리 낮은 면 reach_low·높은 면 reach_high, 바닥 = 고른 자리)에 닿음.
//   엄격 판 — RASC 엄격 플래그(물체 PK_INNER, 받침 inner).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

#include "bscene_host.h"
#include "env_api.h"
#include "rasc.h"

using namespace bsc;

static const char* kClosed[] = {"cabinet", "refrigerator", "fridge", "microwave", "oven", "washer", "dishwasher", "car", "toolbox", "recycling_bin", "drawer", "hinged_jar", "trunk"};
static const char* kShelf[] = {"bookcase", "shelf", "hall_tree", "rack"};

int main(int argc, char** argv) {
  int N = 10000;
  bool strict = false;
  std::string neg;
  BuildOpt opt;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--strict")) strict = true;
    else if (!std::strcmp(argv[a], "--dir") && a + 1 < argc) opt.dir = argv[++a];
    else if (!std::strcmp(argv[a], "--negative") && a + 1 < argc) neg = argv[++a];
    else N = std::atoi(argv[a]);
  }
  const int nf = neg == "free_area" ? NF_FREE_AREA : neg == "in_closed" ? NF_IN_CLOSED : neg == "artic" ? NF_ARTIC : neg == "spawn_reach" ? NF_SPAWN_REACH
               : neg == "spawn_free" ? NF_SPAWN_FREE : neg == "dst_reach" ? NF_DST_REACH : neg == "stance" ? NF_STANCE : 0;
  if (!neg.empty() && !nf) { std::printf("unknown --negative %s\n", neg.c_str()); return 2; }
  opt.nofilter = nf & ~(NF_SPAWN_REACH | NF_SPAWN_FREE);   // 표 만들기 거르개
  SceneBuild b;
  std::string err;
  if (!build_scenes(opt, b, &err) || !upload(b, &err)) { std::printf("build failed: %s\n", err.c_str()); return 1; }
  if (opt.dir.empty()) opt.dir = default_rasc_dir();
  BCurr cu = kBCurrDefault;
  cu.p1 = 0.f; cu.p2 = 0.f; cu.strict = strict ? 1 : 0;
  cu.split = 2;
  cu.nofilter = nf & (NF_SPAWN_REACH | NF_SPAWN_FREE);       // 시작 거르개
  const PnpFilter& F = b.filt;
  std::printf("pnp_check: N=%d filter set %s, disabled %s; filter table (from RASC LIMITS): loose pick_z %.2f place_top %.2f mass %.2f w %.2f thr %.3f | strict %.2f %.2f %.2f %.2f %.3f |"
              " topdown_z %.2f edge %.2f reach %.2f/%.2f free_margin %.2f stance_r %.2f\n",
              N, strict ? "strict" : "loose", neg.empty() ? "none" : neg.c_str(), F.pick_z[0], F.place_top[0], F.max_mass[0], F.max_w[0], F.threshold[0], F.pick_z[1], F.place_top[1],
              F.max_mass[1], F.max_w[1], F.threshold[1], F.topdown_z, F.edge_dist, F.reach_low, F.reach_high, F.free_margin, F.stance_r);
  const uint64_t seed = 4242;
  env::CpuEnv cpu(N, env::kStageBeh, seed, false, &b.host, cu);
  {   // GPU 로 같은 씨앗 리셋: 상태가 비트로 같은가(장치 난수 고르기)
    env::DeviceEnv gpu(N, env::kStageBeh, seed, false, b.dev, cu);
    std::vector<float> f; std::vector<int> iv; std::vector<uint64_t> rg;
    gpu.download(f, iv, rg);
    const bool same = !std::memcmp(f.data(), cpu.f.data(), 4 * f.size()) && iv == cpu.iv && rg == cpu.rng;
    std::printf("  GPU == CPU sampled episode state (bit-identical): %s\n", same ? "yes" : "NO");
    if (!same) return 1;
  }
  std::vector<rasc::Scene> rs(b.sc.size());
  for (size_t k = 0; k < b.sc.size(); ++k)
    if (!rasc::load(rs[k], (opt.dir + "/" + b.sc[k].name + ".rasc").c_str(), &err)) { std::printf("%s\n", err.c_str()); return 1; }
  enum { V_EXCL, V_ARTIC, V_WIDTH, V_MASS, V_HEIGHT, V_EDGE, V_CLOSED, V_DSTH, V_DSTCAT, V_DSTSRC, V_AREA, V_WIN, V_BODY, V_NEAR, V_REACH_PICK, V_REACH_DST, V_STRICT, V_FSET, NV };
  const char* vname[NV] = {"excluded flags", "articulated/fixed", "width", "mass", "pick height", "side-grasp edge", "inside closed", "place height/size", "place category",
                           "place == source", "free area", "spawn window", "spawn body collides", "spawn too near", "pick stance unreachable", "place edge unreachable",
                           "strict flags", "filter set recorded"};
  long viol[NV] = {}, n_ep = 0, by_scene[MAXSC] = {}, n_floor = 0, n_inside = 0, n_instr = 0, fallback = 0;
  for (int i = 0; i < N; ++i) {
    const int ei = cpu.iv[(size_t)env::I_B_ENT * N + i], kind = cpu.iv[(size_t)env::I_B_KIND * N + i], fs = cpu.iv[(size_t)env::I_B_FSET * N + i];
    if (kind != EK_B3) continue;
    ++n_ep;
    const Entry& e = b.ent[ei];
    ++by_scene[e.scene];
    n_instr += cpu.iv[(size_t)env::I_B_INSTR * N + i] >= 0;
    const SceneBuild::Sc& S = b.sc[e.scene];
    const rasc::Scene& R = rs[e.scene];
    const int fi = fs == 1 ? 1 : 0;
    if (fs != (strict ? 1 : 0)) ++viol[V_FSET];
    // 집을 물체(원본 값)
    const RascPickRec& pk = R.picks[e.pick_rec];
    const RascInstRec& in = R.insts[e.inst];
    const RascTaskRec& tk = R.tasks[in.task];
    const int k = (int)(pk.obj & ~RASC_PNP_TASKOBJ) - (int)tk.obj_off;
    const RascTaskObjRec& o = R.task_objs[tk.obj_off + k];
    const RascPoseRec& ps = R.poses[in.pose_off + k];
    const uint32_t excl = RASC_F_WALL | RASC_F_FLOOR | RASC_F_CEILING | RASC_F_DOOR | RASC_F_WINDOW | RASC_F_STAIRS | RASC_F_CARPET | RASC_F_AGENT | RASC_F_SYSTEM |
                          RASC_F_UNMAPPED | RASC_F_WILDCARD;
    if ((o.flags & excl) || !(o.flags & RASC_F_HAS_BBOX)) ++viol[V_EXCL];
    if (o.flags & (RASC_F_ARTICULATED | RASC_F_FIXED_BASE)) ++viol[V_ARTIC];
    if (2.f * std::min(o.half[0], o.half[1]) > F.max_w[fi] + 1e-6f) ++viol[V_WIDTH];
    if (!std::isnan(o.mass) && o.mass > F.max_mass[fi] + 1e-6f) ++viol[V_MASS];
    float z0;
    {
      const float x = ps.quat[0], y = ps.quat[1], z = ps.quat[2], w = ps.quat[3];
      const float R2[3] = {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
      const float cz = ps.pos[2] + R2[0] * o.offset[0] + R2[1] * o.offset[1] + R2[2] * o.offset[2];
      z0 = cz - (std::fabs(R2[0]) * o.half[0] + std::fabs(R2[1]) * o.half[1] + std::fabs(R2[2]) * o.half[2]);
    }
    if (z0 > F.pick_z[fi] + 1e-4f) ++viol[V_HEIGHT];
    if (z0 > F.topdown_z) {   // 옆 잡기: 출발 받침 가장자리까지
      bool ok = !fi;          // 느슨은 받침이 면이 아니면(용기·모름) 통과
      if (pk.src_place != RASC_NONE32 && R.places[pk.src_place].kind == 1) {
        const RascPlaceRec& sp = R.places[pk.src_place];
        const float dx = pk.center[0] - sp.center[0], dy = pk.center[1] - sp.center[1], c = std::cos(sp.yaw), s = std::sin(sp.yaw);
        const float lx = c * dx + s * dy, ly = -s * dx + c * dy;
        ok = std::max(0.f, std::min(sp.half[0] - std::fabs(lx), sp.half[1] - std::fabs(ly))) <= F.edge_dist + 1e-4f;
      }
      if (!ok) ++viol[V_EDGE];
    }
    if (pk.flags & RASC_PK_IN_CLOSED) ++viol[V_CLOSED];
    // 놓을 곳
    const RascPlaceRec& D = R.places[e.dst_rec];
    if (D.kind == 1 && !(D.top <= F.place_top[fi] + 1e-4f && D.top >= F.min_top - 1e-4f && 2.f * std::min(D.half[0], D.half[1]) >= F.min_side - 1e-4f)) ++viol[V_DSTH];
    if (D.kind == 2 && !(D.top <= F.place_top[fi] + F.inside_margin + 1e-4f)) ++viol[V_DSTH];
    n_floor += D.kind == 3;
    n_inside += D.kind == 2;
    if (D.kind != 3) {
      std::string cat;
      if (D.obj & RASC_PNP_TASKOBJ) { const RascTaskObjRec& d2 = R.task_objs[D.obj & ~RASC_PNP_TASKOBJ]; cat = d2.cat != RASC_NONE16 ? R.str(R.cats[d2.cat].name) : ""; }
      else cat = R.str(R.cats[R.objs[D.obj].cat].name);
      bool bad = false;
      for (auto* w : kClosed) bad = bad || cat.find(w) != std::string::npos;
      for (auto* w : kShelf) bad = bad || cat.find(w) != std::string::npos;
      bad = bad || cat.find("baseboard") != std::string::npos;   // 구조물
      if (bad) ++viol[V_DSTCAT];
    }
    if ((uint32_t)e.dst_rec == pk.src_place) ++viol[V_DSTSRC];
    if (D.kind != 3) {   // 빈 넓이(다시 잼: 받침 사각형 안 가운데·받침 윗면 근처 바닥인 다른 물체 바닥 자국)
      const float c = std::cos(D.yaw), s = std::sin(D.yaw), area = 4.f * D.half[0] * D.half[1];
      float occ = 0.f;
      auto add = [&](const float lo[3], const float hi[3]) {
        const float ox = 0.5f * (lo[0] + hi[0]) - D.center[0], oy = 0.5f * (lo[1] + hi[1]) - D.center[1];
        const float lx = c * ox + s * oy, ly = -s * ox + c * oy;
        if (std::fabs(lx) > D.half[0] || std::fabs(ly) > D.half[1]) return;
        const bool on = D.kind == 1 ? (lo[2] >= D.top - 0.05f && lo[2] <= D.top + 0.05f) : (lo[2] >= D.top - 0.6f && lo[2] < D.top);
        if (on) occ += std::min((hi[0] - lo[0]) * (hi[1] - lo[1]), area);
      };
      for (size_t q = 0; q < tk.n_obj; ++q) {   // 인스턴스 과제 물체(집을 것·놓을 곳 빼고)
        const RascTaskObjRec& t2 = R.task_objs[tk.obj_off + q];
        const RascPoseRec& p2 = R.poses[in.pose_off + q];
        if ((int)q == k || ((D.obj & RASC_PNP_TASKOBJ) && (D.obj & ~RASC_PNP_TASKOBJ) == tk.obj_off + q)) continue;
        if ((t2.flags & (RASC_F_AGENT | RASC_F_SYSTEM | RASC_F_FUTURE | RASC_F_UNMAPPED)) || t2.scene_obj >= 0 || p2.src == 0) continue;
        if (t2.half[0] <= 0.f && t2.half[1] <= 0.f && t2.half[2] <= 0.f) continue;
        const float x = p2.quat[0], y = p2.quat[1], z = p2.quat[2], w = p2.quat[3];
        const float Rm[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                             2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
        float lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
          const float cc = p2.pos[a] + Rm[3 * a] * t2.offset[0] + Rm[3 * a + 1] * t2.offset[1] + Rm[3 * a + 2] * t2.offset[2];
          const float ee = std::fabs(Rm[3 * a]) * t2.half[0] + std::fabs(Rm[3 * a + 1]) * t2.half[1] + std::fabs(Rm[3 * a + 2]) * t2.half[2];
          lo[a] = cc - ee; hi[a] = cc + ee;
        }
        add(lo, hi);
      }
      for (size_t ob = 0; ob < R.objs.n; ++ob) {
        const RascObjRec& O = R.objs[ob];
        if ((O.flags & (RASC_F_WALL | RASC_F_FLOOR | RASC_F_CEILING | RASC_F_DOOR | RASC_F_WINDOW | RASC_F_CARPET)) || (!(D.obj & RASC_PNP_TASKOBJ) && ob == D.obj)) continue;
        add(O.aabb_min, O.aabb_max);
      }
      const float need = (e.ext[0] + 2.f * F.free_margin) * (e.ext[1] + 2.f * F.free_margin);
      if (area - occ < need - 1e-5f) ++viol[V_AREA];
    }
    // 시작(환경 상태)
    const float x = cpu.f[(size_t)env::F_X * N + i], y = cpu.f[(size_t)env::F_Y * N + i], yaw = cpu.f[(size_t)env::F_YAW * N + i];
    fallback += x == e.sx && y == e.sy && yaw == e.syaw;
    if (!(std::fabs(x) <= WIN_HALF - 0.8f && std::fabs(y) <= WIN_HALF - 0.8f)) ++viol[V_WIN];
    if (!body_free_host(S, e, x, y, yaw)) ++viol[V_BODY];
    if (std::hypot(x - e.gx, y - e.gy) < 1.0f - 1e-5f) ++viol[V_NEAR];
    // 창 안 BFS(로봇 중심 칸, 8 이웃, 대각은 양 옆, 바닥 높이 차 ≤ 문턱[fi]) — 성분 표와 무관하게
    {
      const int W = S.d.W, H = S.d.H;
      const int c0 = (int)std::lround((e.wx - WIN_HALF - S.d.ox) / CELL), r0 = (int)std::lround((e.wy - WIN_HALF - S.d.oy) / CELL);
      auto free_at = [&](int i2, int j2) { const int sc = c0 + i2, sr = r0 + j2; return i2 >= 0 && j2 >= 0 && i2 < WIN && j2 < WIN && sc >= 0 && sr >= 0 && sc < W && sr < H && S.freeg[(size_t)sr * W + sc]; };
      auto fz = [&](int i2, int j2) { const int16_t v = S.floor_mm[(size_t)(r0 + j2) * W + c0 + i2]; return v == INT16_MIN ? 0 : (int)v; };
      const int thr = (int)std::lround(F.threshold[fi] * 1000.f);
      std::vector<uint8_t> vis((size_t)WIN * WIN, 0);
      const int si = (int)std::floor(x * 10.f) + WIN / 2, sj = (int)std::floor(y * 10.f) + WIN / 2;
      std::queue<int> q;
      if (free_at(si, sj)) { q.push(sj * WIN + si); vis[sj * WIN + si] = 1; }
      const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1}, dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
      while (!q.empty()) {
        const int u = q.front(); q.pop();
        const int ui = u % WIN, uj = u / WIN;
        for (int k2 = 0; k2 < 8; ++k2) {
          const int ni = ui + dx[k2], nj = uj + dy[k2];
          if (!free_at(ni, nj) || vis[nj * WIN + ni]) continue;
          if (k2 >= 4 && (!free_at(ni, uj) || !free_at(ui, nj))) continue;
          if (std::abs(fz(ni, nj) - fz(ui, uj)) > thr) continue;
          vis[nj * WIN + ni] = 1;
          q.push(nj * WIN + ni);
        }
      }
      bool st_ok = false, d_ok = false;
      const bool dhigh = D.kind != 3 && D.top > F.topdown_z;
      for (int c = 0; c < WIN * WIN && !(st_ok && d_ok); ++c) {
        if (!vis[c]) continue;
        const float cx = ((float)(c % WIN) + 0.5f) * CELL - WIN_HALF, cy = ((float)(c / WIN) + 0.5f) * CELL - WIN_HALF;
        if (!st_ok) {
          const float ex = std::max(std::max(e.prim[0].lo[0] - cx, cx - e.prim[0].hi[0]), 0.f), ey = std::max(std::max(e.prim[0].lo[1] - cy, cy - e.prim[0].hi[1]), 0.f);
          if (std::sqrt(ex * ex + ey * ey) <= F.stance_r && stance_ok(S, e, cx, cy)) st_ok = true;
        }
        if (!d_ok) {
          if (D.kind == 3) d_ok = cx > e.dlo[0] && cx < e.dhi[0] && cy > e.dlo[1] && cy < e.dhi[1];
          else {
            const float wx = cx + e.wx, wy = cy + e.wy, dxw = wx - D.center[0], dyw = wy - D.center[1], c2 = std::cos(D.yaw), s2 = std::sin(D.yaw);
            const float lx = c2 * dxw + s2 * dyw, ly = -s2 * dxw + c2 * dyw;
            const float ex = std::max(std::fabs(lx) - D.half[0], 0.f), ey = std::max(std::fabs(ly) - D.half[1], 0.f), dd = std::sqrt(ex * ex + ey * ey);
            d_ok = dd > 0.f && dd <= (dhigh ? F.reach_high : F.reach_low) + 0.05f;   // 0.05: 장면 물체 바닥 자국(RASC PlaceRec) 대 정적 상자 차 여유
          }
        }
      }
      if (!st_ok) ++viol[V_REACH_PICK];
      if (!d_ok) ++viol[V_REACH_DST];
    }
    if (fi && !((pk.flags & RASC_PK_INNER) && (D.flags & 1))) ++viol[V_STRICT];
  }
  // 놓을 점(목표 점, Entry::ppt — 표 전체, 독립 확인: RASC 놓을 곳 기록에서 다시 잼): 바닥 = 고른 자리 안·z 0·서는 칸(성분 ≠ 0),
  // 면 = 받침 사각형 안(물체 반지름 + free_margin 안쪽)·z = 윗면, 같은 성분의 서는 칸이 점에서 팔 닿는 거리(0.38 / 0.31) 안, 용기 = 점 없음
  long n_pt[3] = {0, 0, 0}, pt_bad = 0;
  for (int ei = 0; ei < b.host.nent; ++ei) {
    const Entry& e = b.ent[ei];
    if (e.list != L_OBJ) continue;
    const rasc::Scene& R = rs[e.scene];
    const RascPlaceRec& D = R.places[e.dst_rec];
    const SceneDev& d = b.sc[e.scene].d;
    if (!e.ppt_ok) { pt_bad += D.kind == 3; continue; }   // 바닥 짝은 늘 점이 있어야
    ++n_pt[D.kind - 1];
    if (D.kind == 2) { ++pt_bad; continue; }
    const float px = e.ppt[0], py = e.ppt[1];
    auto comp_at = [&](float x, float y) {   // 창 좌표 → 장면 칸 성분(느슨)
      const int c = (int)std::floor((x + e.wx - d.ox) / CELL), r = (int)std::floor((y + e.wy - d.oy) / CELL);
      return (c < 0 || r < 0 || c >= d.W || r >= d.H) ? 0 : (int)b.sc[e.scene].comp[(size_t)r * d.W + c];
    };
    if (D.kind == 3) { pt_bad += !(px > e.dlo[0] && px < e.dhi[0] && py > e.dlo[1] && py < e.dhi[1] && e.ppt[2] == 0.f && comp_at(px, py) != 0); continue; }
    const float ro = 0.5f * std::max(e.ext[0], e.ext[1]) + F.free_margin;
    const float c2 = std::cos(D.yaw), s2 = std::sin(D.yaw), dxw = px + e.wx - D.center[0], dyw = py + e.wy - D.center[1];
    const float lx = c2 * dxw + s2 * dyw, ly = -s2 * dxw + c2 * dyw;
    bool ok = std::fabs(lx) <= D.half[0] - ro + 1e-3f && std::fabs(ly) <= D.half[1] - ro + 1e-3f && e.ppt[2] == D.top;
    const float reach = D.top > F.topdown_z ? F.reach_high + F.edge_dist : F.reach_low;
    bool near = false;
    for (int dj = -5; dj <= 5 && !near; ++dj)
      for (int di = -5; di <= 5 && !near; ++di) {
        const float cx = (std::floor((px + WIN_HALF) / CELL) + di + 0.5f) * CELL - WIN_HALF, cy = (std::floor((py + WIN_HALF) / CELL) + dj + 0.5f) * CELL - WIN_HALF;
        near = std::hypot(cx - px, cy - py) <= reach && comp_at(cx, cy) == (int)e.comp;
      }
    pt_bad += !(ok && near);
  }
  std::printf("  place points (all pick-and-place entries): ontop %ld, inside %ld (must be 0), floor %ld; violations %ld\n", n_pt[0], n_pt[1], n_pt[2], pt_bad);
  long tot = pt_bad;
  std::printf("  sampled B3 episodes %ld; per scene:", n_ep);
  for (int s2 = 0; s2 < b.host.nsc; ++s2) std::printf(" %s %ld", b.sc[s2].name.c_str(), by_scene[s2]);
  std::printf("\n  place kind: floor %ld, inside %ld, ontop %ld; with an instruction row %ld; spawn fell back to the table %ld\n", n_floor, n_inside, n_ep - n_floor - n_inside,
              n_instr, fallback);
  for (int v = 0; v < NV; ++v) { std::printf("  %-26s %ld\n", vname[v], viol[v]); tot += viol[v]; }
  if (!neg.empty()) {
    std::printf("negative control (%s disabled): %ld violations (must be > 0)\n", neg.c_str(), tot);
    return tot > 0 ? 0 : 1;
  }
  std::printf(tot ? "FAIL: %ld violations\n" : "OK: every sampled episode passes the filter (%ld violations)\n", tot);
  return tot ? 1 : 0;
}
