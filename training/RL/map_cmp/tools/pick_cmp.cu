// pick_cmp — 집을 물체(B4–B6 목표)가 GPU 지도(../map, 잡음 켬 = 학습 그대로)에 왜 확정되지 않는가 + 진짜 파이프라인 비교용 판 내보내기.
// CURRICULUM_BEHAVIOR2026 5.7.1, TRAINING_DESIGN 6.1(입력 단위 비교). GPU 지도 코드는 고치지 않고 그 판정 함수(det_prefilter·vis_point)를 호스트에서 다시 불러
// keyframe 마다 목표 물체(prim 0)가 어느 규칙에서 떨어졌는지 센다.
//
//   pick_cmp [N=512] [steps=900] [--kind 4|5|6] [--slknown] [--sltol] [--gcand] [--teacher stateful] [--seed S] [--split s]
//            [--find] [--findstart] [--look STEPS]   # 찾을 수 있음 거르개(env findable.h)·2 단계 시작·처음 STEPS 동안 제자리 좌우 보기
//            [--dump K --out DIR]   # 판 K 개(판 i < K 의 첫 판)를 DIR/ep_<i>.jsonl 로 — og_replay/og_cmp.py 가 OmniGibson + 진짜 파이프라인으로 다시 돌림
//
// keyframe 원인(첫 번째로 걸린 규칙, map.h 순서 그대로):
//   range  상자 어느 모서리도 카메라 앞 거리 [ozmin 0.15, ozmax 3.0] 안에 없음(뒤쪽 포함)      — 시야 밖 (a)
//   fov    상자 투영이 화면(가로 67.9°·세로 45.6°)과 안 겹침                                  — 시야 밖 (a)
//   occl   보임 점 5 개가 다 가려짐(가구 상자·다른 물체·팔) 또는 그 깊이가 범위 밖               — 기하 (a′, 진짜도 못 봄)
//   minpx  가장 긴 변 투영 < min_px 6 px                                                        — 규칙 (b)
//   area   실루엣 넓이 어림 < min_points 20 px                                                   — 규칙 (b)
//   areav  넓이 × 보인 점 비율 < min_points                                                     — 규칙 (b)
//   floor  백분위 상자 윗면 < floor_h 0.05 m(바닥 조각 거르기)                                   — 규칙 (b)
//   hand   잡는 점 hand_r 0.10 m 안(손에 든 것 거르기)                                           — 규칙 (b)
//   miss   판정은 다 넘었는데 이 keyframe 에 검출 없음 = 거리별 놓침 p_miss(0.15/0.10/0.79)      — 잡음 (b)
//   det    검출됨(목표 칸이 이 keyframe 에 갱신)
// 판 끝 분류(목표를 찾았나 = map n_task_conf, 아니면 왜): found / (a) noview·occl / (b) small·pmiss·floor·hand·once·name·pos
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "bscene_host.h"
#include "env_api.h"
#include "env_beh.h"
#include "env_pnp.h"
#include "map_api.h"
#include "beh_rec.h"
#include "slot_adapter.h"

using namespace env;
using namespace gmap;

enum Cause { C_RANGE, C_FOV, C_OCCL, C_MINPX, C_AREA, C_AREAV, C_FLOOR, C_HAND, C_MISS, C_DET, C_N };
static const char* kCause[C_N] = {"range", "fov", "occl", "minpx", "area", "areav", "floor", "hand", "miss", "det"};
enum EpCls { E_FOUND, E_NOVIEW, E_OCCL, E_SMALL, E_PMISS, E_FLOOR, E_HAND, E_ONCE, E_NAME, E_POS, E_N };
static const char* kEp[E_N] = {"found", "(a) never in view", "(a) in view, always occluded", "(b) too small (minpx/area)", "(b) detectable, never detected (p_miss)",
                               "(b) floor-piece filter", "(b) hand filter", "(b) detected in < 2 keyframes", "(b) confirmed, wrong name", "(b) confirmed, off position"};

// vis_point(map.h) 와 같은 광선, 떨어진 까닭만 나눔: 1 자기 상자에 안 닿음·깊이 범위 밖(너무 가까움 포함), 2 팔, 3 다른 상자. 0 = 보임
static int vis_why(const MapCore& m, const float o[3], float c, float s, const DetGeo& g, int pi, int q, const BCtx& bx, float* tdep) {
  const float xc = 0.5f * (g.xr[0] + g.xr[1]), yc = 0.5f * (g.yr[0] + g.yr[1]);
  const float hx = 0.4f * (g.xr[1] - g.xr[0]), hy = 0.4f * (g.yr[1] - g.yr[0]);
  const float xn = xc + (q == 1 ? -hx : q == 2 ? hx : 0.f), yn = yc + (q == 3 ? -hy : q == 4 ? hy : 0.f);
  float d[3], inv[3];
  pix_dir(c, s, xn, yn, d);
  ray_inv(d, inv);
  const bsc::SceneDev& sd = *bx.sd;
  const float ow[3] = {o[0] + bx.wx, o[1] + bx.wy, o[2]};
  const int own = bx.bm->sbox[pi];
  const float t = own >= 0 ? ray_obb_inv(ow, d, sd.box[own]) : ray_box_inv(o, d, inv, m.prim[pi]);
  *tdep = t;
  if (!(t >= MP::ozmin && t <= MP::ozmax)) return 1;
  { const float pb[3] = {env::K::cam_x + t, -xn * t, env::K::cam_z - yn * t}; if (arm_blocks(m, pb)) return 2; }
  bool occ = false;
  bsc::bin_walk(sd, ow[0], ow[1], d[0], d[1], [&](int j, float) { if (!occ && j != own && ray_obb_inv(ow, d, sd.box[j]) < t) occ = true; }, [&]() { return occ ? -1.f : t; });
  if (occ) return 3;
  for (int j = 0; j < N_PRIM; ++j) if (j != pi && prim_dyn(bx, j) && ray_box_inv(o, d, inv, m.prim[j]) < t) return 3;
  return 0;
}

struct KfInfo { int cause; float dist, size_px, af; int nv; int why[4]; };

// keyframe 하나에서 목표(prim 0) 원인 — map.h obj_pre·obj_detect 와 같은 함수·같은 순서(참 자세·참 장면)
static KfInfo classify(const MapHost& mh, int ei, const EnvView& e, const BCtx& bx) {
  const MapCore& m = mh.core[ei];
  KfInfo r{C_RANGE, 0.f, 0.f, 0.f, 0, {0, 0, 0, 0}};
  const Cam k = cam_consts();
  float s, c;
  sincosf_d(e.yaw, &s, &c);
  float o[3];
  cam_world(e.x, e.y, c, s, o);
  const Prim& b = m.prim[0];
  float ctr[3];
  for (int a = 0; a < 3; ++a) ctr[a] = 0.5f * (b.lo[a] + b.hi[a]);
  r.dist = std::sqrt((ctr[0] - o[0]) * (ctr[0] - o[0]) + (ctr[1] - o[1]) * (ctr[1] - o[1]) + (ctr[2] - o[2]) * (ctr[2] - o[2]));
  DetGeo g;
  std::memset(&g, 0, sizeof g);
  g.xr[0] = 1.f; g.xr[1] = 0.f;
  const bool ok = det_prefilter(m, o, c, s, k, 0, g);
  float fmin = 1e30f;
  for (int q = 0; q < 4; ++q) {
    const float x = ((q & 1) ? b.hi[0] : b.lo[0]) - o[0], y = ((q & 2) ? b.hi[1] : b.lo[1]) - o[1];
    fmin = std::min(fmin, c * x + s * y);
  }
  const float fe = std::min(std::max(g.fwd, std::max(fmin, MP::ozmin)), MP::ozmax);
  r.size_px = k.fx * max3(g.ext) / fe;
  r.af = g.af;
  if (!ok) {
    if (!(g.inr & 1)) r.cause = C_RANGE;
    else if (!(g.xr[0] < g.xr[1] && g.yr[0] < g.yr[1])) r.cause = C_FOV;
    else r.cause = r.size_px < (float)MP::min_px ? C_MINPX : C_AREA;
    if (r.cause >= C_MINPX) {   // 작아서 떨어졌어도 다 가려진 것이면 가림으로(진짜도 못 봄)
      int nv = 0;
      for (int q = 0; q < NPT; ++q) nv += vis_point(m, o, c, s, g, 0, q, bx);
      if (nv == 0) r.cause = C_OCCL;
    }
    return r;
  }
  int nv = 0;
  for (int q = 0; q < NPT; ++q) nv += vis_point(m, o, c, s, g, 0, q, bx);
  r.nv = nv;
  if (nv == 0) {
    r.cause = C_OCCL;
    for (int q = 0; q < NPT; ++q) { float td; ++r.why[vis_why(m, o, c, s, g, 0, q, bx, &td)]; }
    return r;
  }
  if (g.af * ((float)nv / (float)NPT) < (float)MP::min_points) { r.cause = C_AREAV; return r; }
  if (o[2] + g.bc[2] + 0.5f * g.pe[2] < MP::floor_h) { r.cause = C_FLOOR; return r; }
  {
    const float bxx = env::K::cam_x + g.med[0], byy = g.med[1];
    const float p[3] = {e.x + c * bxx - s * byy, e.y + s * bxx + c * byy, o[2] + g.med[2]};
    if (dist3(p, m.gp_m) < MP::hand_r) { r.cause = C_HAND; return r; }
  }
  r.cause = C_MISS;
  std::vector<SlotRow> rows;
  slot_rows(mh, ei, rows);
  for (const SlotRow& S : rows) if (S.src == 0 && S.last_kf == m.t) r.cause = C_DET;
  return r;
}

struct EpAcc {
  int ep = -1, kind = 0, ent = -1, scene = -1, nkf = 0, found = 0, t_found = -1, t_view = -1, t_det = -1, steps = 0;
  long cause[C_N] = {};
  int ndet = 0, maxobs = 0, conf = 0, cls_conf = -1, cls_true = -1;
  float pos_err = -1.f, d0 = 0.f, dmin_view = 1e9f, rot = 0.f, path = 0.f, lx = 0, ly = 0, lyaw = 0;
  float h_lo = 0.f, size = 0.f, wmin = 0.f;
  int sg = 0;   // 시작 기하 묶음(아래 kSg)
};
// 판 시작의 목표 자리(카메라 기준): 광축 앞 거리 f, 세로 각 = 높이차/f
static const char* kSg[6] = {"f<0.15 m (inside min range)", "f 0.15-0.3, below/above vFOV", "f 0.15-0.3, in vFOV", "f 0.3-0.6", "f>=0.6", "behind/side"};

int main(int argc, char** argv) {
  int N = 512, T = 900, kind = 4, pos = 0, dumpK = 0, sltol = 0, find = 0, look = 0;
  uint64_t seed = 20261006;
  bool stateful = false, gcand = false;
  std::string out;
  bsc::BCurr cu = bsc::kBCurrDefault;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--kind") && a + 1 < argc) kind = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--slknown")) sltol |= 2;
    else if (!std::strcmp(argv[a], "--sltol")) sltol |= 1;
    else if (!std::strcmp(argv[a], "--gcand")) gcand = true;
    else if (!std::strcmp(argv[a], "--teacher") && a + 1 < argc) stateful = !std::strcmp(argv[++a], "stateful");
    else if (!std::strcmp(argv[a], "--seed") && a + 1 < argc) seed = std::strtoull(argv[++a], nullptr, 10);
    else if (!std::strcmp(argv[a], "--split") && a + 1 < argc) cu.split = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--dump") && a + 1 < argc) dumpK = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--out") && a + 1 < argc) out = argv[++a];
    else if (!std::strcmp(argv[a], "--find")) find |= 1;        // 찾을 수 없는 목표 빼기(env findable.h, PF_FIND)
    else if (!std::strcmp(argv[a], "--findstart")) find |= 2;   // B4 시작 = 찾을 수 있는 자세(PF_FINDSTART)
    else if (!std::strcmp(argv[a], "--look") && a + 1 < argc) look = std::atoi(argv[++a]);   // 판 처음 look 스텝: 베이스는 제자리 좌우로 천천히(±0.25 rad/s, 1 s 마다 방향), 팔은 교사
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
    else if (pos == 1) { T = std::atoi(argv[a]); ++pos; }
  }
  if (dumpK > 0 && out.empty()) { std::fprintf(stderr, "--dump needs --out DIR\n"); return 2; }
  cu.p1 = cu.p2 = 0.f;
  cu.p4 = kind == 4 ? 1.f : 0.f; cu.p5 = kind == 5 ? 1.f : 0.f; cu.p6 = kind == 6 ? 1.f : 0.f;
  bsc::SceneBuild sb;
  std::string err;
  if (!bsc::build_scenes(bsc::BuildOpt{}, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
  env::pnp_feasibility(sb);
  if (gcand) env::pnp_stance_cands(sb);
  if (sltol) env::set_sl_tol(sb, sltol);
  if (find) env::pnp_findability(sb);
  if (find & 1) cu.phys |= bsc::PF_FIND;
  if (find & 2) cu.phys |= bsc::PF_FINDSTART;
  cu.phys |= bsc::PF_FEAS;
  DeviceEnv env(N, kStageBeh, seed, true, sb.dev, cu);
  DeviceMap map(N, seed * 7919ull + 3ull, sb.dev);
  MapCurr mc = kCurrEmpty;   // C2: 빈 지도
  cudaMemcpy(map.curr_dev(), &mc, sizeof mc, cudaMemcpyHostToDevice);
  map.set_tokens(false);
  env.set_nav(map.nav_fb());
  if (!stateful) env.enable_teacher_sl();
  rec::BehRec beh(&sb);
  float *act, *obs, *rew; int* done;
  cudaMalloc(&act, sizeof(float) * N_ACT * N); cudaMalloc(&obs, sizeof(float) * N_OBS * N); cudaMalloc(&rew, sizeof(float) * N); cudaMalloc(&done, sizeof(int) * N);
  std::vector<int> dn(N);
  std::vector<float> hf; std::vector<int> hi; std::vector<uint64_t> hr;
  MapHost mh;
  std::vector<EpAcc> acc(N);
  std::vector<FILE*> dumpf(N, nullptr);
  std::vector<int> dumped(N, 0);
  long ecls[bsc::N_EK][E_N] = {}, eres[bsc::N_EK][E_N][2] = {};
  long kfc[C_N] = {}, kfc_d[4][C_N] = {};   // keyframe 원인 전체, 거리 띠별(< 1, 1–1.5, 1.5–2.5, ≥ 2.5 m)
  long occw[4] = {};
  long n_eps = 0, n_succ = 0, sgc[6] = {}, sgo[6][E_N] = {};
  std::vector<float> sgf, sgu, rots, paths;
  long endres[4] = {};
  std::vector<double> t_view, t_det, t_found;
  const bsc::SceneSet* ssh = &sb.host;
  for (int t = 0; t < T; ++t) {
    if (stateful) env.teacher(act); else env.teacher_sl(act);
    if (look > 0) {   // 판 처음 look 스텝: 베이스 행동만 덮어씀(앞뒤 0, 돌기 ±0.25 rad/s — 행동은 K::v_max·w_max 로 정규화)
      std::vector<float> ha((size_t)N_ACT * N);
      cudaMemcpy(ha.data(), act, sizeof(float) * ha.size(), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; ++i) {
        const int ts = std::max(0, hi.empty() ? 0 : hi[(size_t)I_STEP * N + i]);
        if (ts >= look) continue;
        ha[(size_t)0 * N + i] = 0.f;
        ha[(size_t)1 * N + i] = ((ts / 10) % 2 ? -1.f : 1.f) * 0.25f / K::w_max;
      }
      cudaMemcpy(act, ha.data(), sizeof(float) * ha.size(), cudaMemcpyHostToDevice);
    }
    env.step(act, obs, rew, done);
    map.step(env.soa());
    cudaDeviceSynchronize();
    cudaMemcpy(dn.data(), done, sizeof(int) * N, cudaMemcpyDeviceToHost);
    env.download(hf, hi, hr);
    map.download(mh);
    const Soa hs{hf.data(), hi.data(), hr.data(), N};
    for (int i = 0; i < N; ++i) {
      EpAcc& A = acc[i];
      if (dn[i] && A.ep >= 0 && t > 0) {   // 판 끝(환경은 이미 다음 판으로 리셋)
        int cl;
        long inview = 0, small = A.cause[C_MINPX] + A.cause[C_AREA] + A.cause[C_AREAV];
        for (int q = C_OCCL; q < C_N; ++q) inview += A.cause[q];
        if (A.found) cl = E_FOUND;
        else if (inview == 0) cl = E_NOVIEW;
        else if (A.cause[C_DET] == 0 && A.cause[C_MISS] == 0 && A.cause[C_FLOOR] == 0 && A.cause[C_HAND] == 0) cl = small ? E_SMALL : E_OCCL;
        else if (A.cause[C_DET] == 0) cl = A.cause[C_MISS] ? E_PMISS : A.cause[C_FLOOR] ? E_FLOOR : E_HAND;
        else if (!A.conf) cl = E_ONCE;
        else cl = A.cls_conf != A.cls_true ? E_NAME : E_POS;
        const int k = A.kind >= 0 && A.kind < bsc::N_EK ? A.kind : 0;
        ++ecls[k][cl]; ++eres[k][cl][dn[i] == kSuccess]; ++sgo[A.sg][cl];
        rots.push_back(A.rot); paths.push_back(A.path); ++endres[dn[i] & 3];
        ++n_eps; n_succ += dn[i] == kSuccess;
        if (A.t_view >= 0) t_view.push_back(A.t_view * 0.1);
        if (A.t_det >= 0) t_det.push_back(A.t_det * 0.1);
        if (A.t_found >= 0) t_found.push_back(A.t_found * 0.1);
        if (dumpf[i]) {
          std::fprintf(dumpf[i], "{\"end\":1,\"result\":%d,\"class\":\"%s\",\"steps\":%d,\"kf\":%d,\"found\":%d,\"t_found\":%d,\"cause\":{", dn[i], kEp[cl], A.steps, A.nkf, A.found, A.t_found);
          for (int q = 0; q < C_N; ++q) std::fprintf(dumpf[i], "%s\"%s\":%ld", q ? "," : "", kCause[q], A.cause[q]);
          std::fprintf(dumpf[i], "}}\n");
          std::fclose(dumpf[i]);
          dumpf[i] = nullptr;
        }
        A = EpAcc{};
      }
      const int k = hi[(size_t)I_B_KIND * N + i];
      if (k < bsc::EK_B4) continue;
      const MapCore& m = mh.core[i];
      const EnvView e = read_env(hs, i, true);
      if (A.ep != e.ep) {   // 새 판 시작
        A = EpAcc{};
        A.ep = e.ep; A.kind = k; A.ent = e.bent; A.scene = e.bscene;
        const bsc::Entry& E = sb.ent[e.bent];
        A.cls_true = m.prim[0].cls;
        A.h_lo = m.prim[0].lo[2]; A.size = std::max(E.odim[0], std::max(E.odim[1], E.odim[2])); A.wmin = std::min(E.odim[0], E.odim[1]);
        A.d0 = std::hypot(e.bo[0] - e.x, e.bo[1] - e.y);
        A.lx = e.x; A.ly = e.y; A.lyaw = e.yaw;
        {
          float sn, cs; sincosf_d(e.yaw, &sn, &cs);
          float oc[3]; cam_world(e.x, e.y, cs, sn, oc);
          const float rx = e.bo[0] - oc[0], ry = e.bo[1] - oc[1], up = e.bo[2] - oc[2];
          const float f = cs * rx + sn * ry, l = -sn * rx + cs * ry;
          const Cam kk = cam_consts();
          A.sg = f < 0.f || std::fabs(l) > f * kk.tanh + 0.05f ? 5 : f < MP::ozmin ? 0 : f < 0.3f ? (std::fabs(up) > f * kk.tanv ? 1 : 2) : f < 0.6f ? 3 : 4;
          ++sgc[A.sg]; sgf.push_back(f); sgu.push_back(up);
        }
        if (dumpK > 0 && i < dumpK && !dumped[i]) {
          dumped[i] = 1;
          char fn[512];
          std::snprintf(fn, sizeof fn, "%s/ep_%04d.jsonl", out.c_str(), i);
          dumpf[i] = std::fopen(fn, "w");
          env::BState b; env::load_b(hs, i, b);
          env::PState p; env::load_p(hs, i, p);
          std::fprintf(dumpf[i], "{\"head\":1,\"env\":%d,\"seed\":%llu,\"kind\":%d,\"slknown\":%d,\"teacher\":\"%s\",\"dt\":0.1,\"prim0_cls\":%d,\"prim0_name\":\"%s\",\"odim\":[%g,%g,%g],\"scene\":%s}\n",
                       i, (unsigned long long)seed, k, (sltol & 2) ? 1 : 0, stateful ? "stateful" : "stateless", A.cls_true,
                       A.cls_true >= 0 && A.cls_true < (int)sb.name_en.size() ? sb.name_en[A.cls_true].c_str() : "", E.odim[0], E.odim[1], E.odim[2], beh.scene_json(b, p).c_str());
        }
      }
      ++A.steps;
      A.path += std::hypot(e.x - A.lx, e.y - A.ly);
      A.rot += std::fabs(wrap_pi(e.yaw - A.lyaw));
      A.lx = e.x; A.ly = e.y; A.lyaw = e.yaw;
      if (m.n_task_conf && !A.found) { A.found = 1; A.t_found = m.t; }
      KfInfo ki{-1, 0, 0, 0, 0, {0, 0, 0, 0}};
      if (m.kf_flag && m.ep == e.ep) {
        BMapEnv* bm = &mh.bm[i];
        const BCtx bx = bctx(ssh, bm);
        ki = classify(mh, i, e, bx);
        ++A.nkf;
        ++A.cause[ki.cause];
        ++kfc[ki.cause];
        if (ki.cause == C_OCCL) { const int w = ki.why[1] >= ki.why[2] && ki.why[1] >= ki.why[3] ? 1 : ki.why[2] >= ki.why[3] ? 2 : 3; ++occw[w]; }
        ++kfc_d[ki.dist < 1.f ? 0 : ki.dist < 1.5f ? 1 : ki.dist < 2.5f ? 2 : 3][ki.cause];
        if (ki.cause >= C_OCCL) { if (A.t_view < 0) A.t_view = m.t; A.dmin_view = std::min(A.dmin_view, ki.dist); }
        if (ki.cause == C_DET && A.t_det < 0) A.t_det = m.t;
        std::vector<SlotRow> rows;
        slot_rows(mh, i, rows);
        for (const SlotRow& S : rows) {
          if (S.src != 0) continue;
          A.maxobs = std::max(A.maxobs, S.n_obs);
          if (S.conf) { A.conf = 1; A.cls_conf = S.cls; }
        }
      }
      if (dumpf[i]) {   // 스텝 한 줄(창 좌표; 세계 = + wx, wy)
        FILE* f = dumpf[i];
        std::fprintf(f, "{\"t\":%d,\"x\":%.5f,\"y\":%.5f,\"yaw\":%.5f,\"v\":%.4f,\"w\":%.4f,\"q\":[%.4f,%.4f,%.4f,%.4f,%.4f],\"qg\":%.4f,\"o\":[%.5f,%.5f,%.5f,%.5f],\"ost\":%d",
                     m.t, e.x, e.y, e.yaw, e.v, e.w, e.q[0], e.q[1], e.q[2], e.q[3], e.q[4], e.q[5], e.bo[0], e.bo[1], e.bo[2], e.bo[3], hi[(size_t)I_O_ST * N + i]);
        std::fprintf(f, ",\"est\":[%.5f,%.5f,%.5f],\"kf\":%d,\"task_conf\":%d", m.ex, m.ey, m.eyaw, m.kf_flag, m.n_task_conf);
        if (ki.cause >= 0) {
          std::fprintf(f, ",\"cause\":\"%s\",\"dist\":%.3f,\"size_px\":%.1f,\"af\":%.1f,\"nv\":%d,\"slots\":[", kCause[ki.cause], ki.dist, ki.size_px, ki.af, ki.nv);
          int first = 1;
          std::vector<SlotRow> rows;
          slot_rows(mh, i, rows);
          for (const SlotRow& S : rows) {
            std::fprintf(f, "%s{\"id\":%d,\"cls\":%d,\"name\":\"%s\",\"share\":%.3f,\"conf\":%d,\"n_obs\":%d,\"src\":%d,\"state\":%d,\"pos\":[%.4f,%.4f,%.4f],\"ext\":[%.4f,%.4f,%.4f],\"last_kf\":%d}",
                         first ? "" : ",", S.id, S.cls, S.cls >= 0 && S.cls < (int)sb.name_en.size() ? sb.name_en[S.cls].c_str() : "", S.share, S.conf, S.n_obs, S.src,
                         S.state, S.pos[0], S.pos[1], S.pos[2], S.ext[0], S.ext[1], S.ext[2], S.last_kf);
            first = 0;
          }
          std::fprintf(f, "]");
        }
        std::fprintf(f, "}\n");
      }
    }
  }
  for (int i = 0; i < N; ++i) if (dumpf[i]) { std::fprintf(dumpf[i], "{\"end\":1,\"result\":0,\"class\":\"unfinished\"}\n"); std::fclose(dumpf[i]); }
  std::printf("pick_cmp: N %d steps %d kind B%d teacher %s slknown %d sltol %d | episodes %ld success %.3f\n", N, T, kind, stateful ? "stateful" : "stateless", (sltol & 2) ? 1 : 0, sltol & 1,
              n_eps, n_eps ? (double)n_succ / n_eps : 0.0);
  for (int k = bsc::EK_B4; k < bsc::N_EK; ++k) {
    long tot = 0;
    for (int c = 0; c < E_N; ++c) tot += ecls[k][c];
    if (!tot) continue;
    std::printf("  B%d episodes by target outcome (n, share, teacher success in class):\n", k);
    for (int c = 0; c < E_N; ++c)
      if (ecls[k][c]) std::printf("    %-42s %6ld  %.3f  succ %.3f\n", kEp[c], ecls[k][c], (double)ecls[k][c] / tot, (double)eres[k][c][1] / ecls[k][c]);
  }
  long kt = 0;
  for (int q = 0; q < C_N; ++q) kt += kfc[q];
  std::printf("  keyframes by target cause (all %ld):", kt);
  for (int q = 0; q < C_N; ++q) std::printf(" %s %ld (%.3f)", kCause[q], kfc[q], kt ? (double)kfc[q] / kt : 0.0);
  std::printf("\n");
  std::printf("    occl keyframes by main reason: own surface out of depth range [0.15, 3] m %ld, arm %ld, other box (furniture/support) %ld\n", occw[1], occw[2], occw[3]);
  const char* bn[4] = {"<1.0 m ", "1.0-1.5", "1.5-2.5", ">=2.5 m"};
  for (int d = 0; d < 4; ++d) {
    long tt = 0;
    for (int q = 0; q < C_N; ++q) tt += kfc_d[d][q];
    std::printf("    dist %s %7ld:", bn[d], tt);
    for (int q = C_OCCL; q < C_N; ++q) std::printf(" %s %.3f", kCause[q], tt ? (double)kfc_d[d][q] / tt : 0.0);
    std::printf("\n");
  }
  std::printf("  episode start: target relative to the depth camera (f = forward along the optical axis; vFOV half %.1f deg):\n", std::atan(cam_consts().tanv) * 57.29578f);
  for (int g = 0; g < 6; ++g) {
    if (!sgc[g]) continue;
    std::printf("    %-32s starts %6ld | ended:", kSg[g], sgc[g]);
    for (int c = 0; c < E_N; ++c) if (sgo[g][c]) std::printf(" %s %ld;", kEp[c], sgo[g][c]);
    std::printf("\n");
  }
  if (!sgf.empty()) {
    std::vector<float> a = sgf, b = sgu; std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    std::printf("    start f 10/50/90 %% %.2f / %.2f / %.2f m, target height - camera height 10/50/90 %% %.2f / %.2f / %.2f m\n", a[a.size() / 10], a[a.size() / 2], a[a.size() * 9 / 10],
                b[b.size() / 10], b[b.size() / 2], b[b.size() * 9 / 10]);
  }
  if (!rots.empty()) {
    std::vector<float> a = rots, b = paths; std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    long full = 0; for (float r : rots) full += r >= 6.2832f;
    std::printf("  robot motion per episode: turned 10/50/90 %% %.2f / %.2f / %.2f rad (>= one full turn %.3f), driven 50/90 %% %.2f / %.2f m; ended success %ld collision %ld timeout %ld\n",
                a[a.size() / 10], a[a.size() / 2], a[a.size() * 9 / 10], (double)full / rots.size(), b[b.size() / 2], b[b.size() * 9 / 10], endres[kSuccess], endres[kCollision], endres[kTimeout]);
  }
  auto med = [](std::vector<double> v) { if (v.empty()) return -1.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  std::printf("  median time to first in-view %.1f s (n %zu), first detection %.1f s (n %zu), found %.1f s (n %zu)\n", med(t_view), t_view.size(), med(t_det), t_det.size(), med(t_found), t_found.size());
  return 0;
}
