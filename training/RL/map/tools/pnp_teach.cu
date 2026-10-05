// 대본 특권 교사(E6, env teacher.h)의 성공률 — 물리 + 역기구학 + 계획의 상한 확인(CURRICULUM_BEHAVIOR2026 E6). GPU 만(환경 + 지도 + 교사 커널, 판마다 같은 그래프 길).
//   pnp_teach [N=2048] [steps=900] [--kind 4|5|6 (기본 셋 다 1/3 씩)] [--strict] [--curr p0,p1 (처음 지도, 기본 1,0 = 다 앎)] [--fail p_slip,p_occ] [--split s] [--seed S]
//             [--all (PF_FEAS 끔 — 잡기 가능 아닌 짝도)]
// 기본은 잡기 가능 표(env pnp_feasibility)를 만들고 PF_FEAS 로 그 단계가 될 수 있는 짝만 뽑는다. 끝에 집 × 단계, 폭 반, 받침(바닥/면/용기), 포기 까닭(FeasReason).
// 판이 끝날 때마다(교사가 모든 판을 움직임) 단계 × 물체 좁은 가로 폭 반(< 2 · 2–4 · 4–6 · ≥ 6 cm) × 받침 높이(바닥 < 0.1 m / 면)별 성공·충돌·시간초과,
// 교사가 서는 자리를 못 찾고 포기한 판(I_T_TRY > max), 잡기·미끄러짐 수, 판 하나 평균 길이, 잰 ms/스텝(환경+지도+교사, 환경만).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "env_api.h"
#include "map_api.h"
#include "bscene_host.h"

using namespace env;

int main(int argc, char** argv) {
  int N = 2048, T = 900, kind = 0, pos = 0;
  uint64_t seed = 20261005;
  bool all_eps = false, sl = false, agree = false;
  bsc::BCurr cu = bsc::kBCurrDefault;
  gmap::MapCurr mc = gmap::kCurrEmpty;
  mc.p0 = 1.f; mc.p1 = 0.f;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--kind") && a + 1 < argc) kind = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--strict")) cu.strict = 1;
    else if (!std::strcmp(argv[a], "--curr") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &mc.p0, &mc.p1);
    else if (!std::strcmp(argv[a], "--fail") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &cu.p_slip, &cu.p_occ);
    else if (!std::strcmp(argv[a], "--split") && a + 1 < argc) cu.split = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--seed") && a + 1 < argc) seed = std::strtoull(argv[++a], nullptr, 10);
    else if (!std::strcmp(argv[a], "--all")) all_eps = true;
    else if (!std::strcmp(argv[a], "--sl")) sl = true;   // 상태 없는 교사(teacher_sl.h)
    else if (!std::strcmp(argv[a], "--agree")) agree = true;   // 상태 있는 교사가 몰고, 같은 상태에서 상태 없는 교사 라벨을 견줌
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
    else if (pos == 1) { T = std::atoi(argv[a]); ++pos; }
  }
  cu.p1 = 0.f; cu.p2 = 0.f;
  cu.p4 = kind == 0 ? 1.f / 3.f : kind == 4 ? 1.f : 0.f;
  cu.p5 = kind == 0 ? 1.f / 3.f : kind == 5 ? 1.f : 0.f;
  cu.p6 = kind == 0 ? 1.f / 3.f : kind == 6 ? 1.f : 0.f;
  bsc::SceneBuild sb;
  std::string err;
  if (!bsc::build_scenes(bsc::BuildOpt{}, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
  const double feas_s = env::pnp_feasibility(sb);
  if (!all_eps) cu.phys |= bsc::PF_FEAS;
  DeviceEnv env(N, kStageBeh, seed, true, sb.dev, cu);
  gmap::DeviceMap map(N, seed * 7919ull + 3ull, sb.dev);
  cudaMemcpy(map.curr_dev(), &mc, sizeof mc, cudaMemcpyHostToDevice);
  map.set_tokens(false);
  env.set_nav(map.nav_fb());
  if (sl || agree) env.enable_teacher_sl();
  float* act2 = nullptr;
  if (agree) cudaMalloc(&act2, sizeof(float) * N_ACT * N);
  std::vector<float> ha((size_t)N_ACT * N), hb((size_t)N_ACT * N);
  std::vector<SlRec> slag(agree ? N : 0);
  long ag_n[16] = {}, ag_exact[16] = {}, ag_tol[16] = {}, ag_base[16] = {}, ag_arm[16] = {};   // 상태 있는 교사 단계별: 표본, 비트 같음, |Δ| ≤ 0.05, 베이스만 맞음, 팔만 맞음
  long ag_conf[16][SLD_N] = {};   // [상태 있는 단계][상태 없는 단계] — 다를 때
  std::vector<SlRec> slr(sl ? N : 0), slrb(sl ? N : 0);
  long slwhy[bsc::N_EK][SLD_N] = {};
  float *act, *obs, *rew; int* done;
  cudaMalloc(&act, sizeof(float) * N_ACT * N); cudaMalloc(&obs, sizeof(float) * N_OBS * N); cudaMalloc(&rew, sizeof(float) * N); cudaMalloc(&done, sizeof(int) * N);
  std::vector<int> dn(N), ivk(N), ivent(N), ivtry(N), ivfl(N);
  long tab[bsc::N_EK][4][3][4] = {};   // [단계][폭 반][바닥/면/용기 놓기][끝난 수, 성공, 충돌, 포기]
  long house[bsc::MAXSC][bsc::N_EK][2] = {};   // [장면][단계][끝난 수, 성공]
  long why[bsc::N_EK][16] = {};        // 실패 판의 교사 마지막 까닭(FeasReason, 0 = 까닭 없음 — 시간 초과 등)
  long nplan_steps = 0, nplan_env = 0;
  const TBuf tbd = env.tbuf();
  std::vector<int> tiv((size_t)NTI * N), tivb((size_t)NTI * N);
  long grasps = 0, slips = 0, steps_sum = 0;
  std::vector<int> len(N, 0);
  const size_t NI = NUM_I;
  std::vector<int> iv((size_t)NI * N);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  float ms = 0.f;
  for (int t = 0; t < T; ++t) {
    cudaEventRecord(e0);
    if (sl) env.teacher_sl(act); else env.teacher(act);
    if (agree) {
      env.teacher_sl(act2);
      cudaMemcpy(ha.data(), act, sizeof(float) * ha.size(), cudaMemcpyDeviceToHost);
      cudaMemcpy(hb.data(), act2, sizeof(float) * hb.size(), cudaMemcpyDeviceToHost);
      cudaMemcpy(slag.data(), env.slbuf().rec, sizeof(SlRec) * N, cudaMemcpyDeviceToHost);
      std::vector<int> ivn((size_t)NUM_I * N), tvn((size_t)NTI * N);
      cudaMemcpy(ivn.data(), env.soa().iv, sizeof(int) * ivn.size(), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; ++i) {
        if (ivn[(size_t)I_B_KIND * N + i] < bsc::EK_B4) continue;
        const int ph = ivn[(size_t)I_T_PH * N + i] & 15;
        bool ex = true, tl = true, tb = true, ta = true;
        for (int k = 0; k < N_ACT; ++k) {
          const float x = ha[(size_t)k * N + i], y = hb[(size_t)k * N + i];
          ex = ex && x == y;
          const bool ok = std::fabs(x - y) <= 0.05f;
          tl = tl && ok;
          if (k < 2) tb = tb && ok; else ta = ta && ok;
        }
        ++ag_n[ph]; ag_exact[ph] += ex; ag_tol[ph] += tl; ag_base[ph] += tb; ag_arm[ph] += ta;
        if (!tl) { const int md = slag[i].mode; ++ag_conf[ph][md >= 0 && md < SLD_N ? md : 0]; }
      }
    }
    env.step(act, obs, rew, done);
    map.step(env.soa());
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float m1;
    cudaEventElapsedTime(&m1, e0, e1);
    if (t >= 10) ms += m1;
    std::vector<int> ivb = iv;   // 지난 스텝(판 끝 앞의 단계·Entry·교사 시도)
    tivb.swap(tiv);
    cudaMemcpy(dn.data(), done, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(iv.data(), env.soa().iv, sizeof(int) * iv.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(tiv.data(), tbd.iv, sizeof(int) * tiv.size(), cudaMemcpyDeviceToHost);
    if (sl) { slrb.swap(slr); cudaMemcpy(slr.data(), env.slbuf().rec, sizeof(SlRec) * N, cudaMemcpyDeviceToHost); }
    {
      long np = 0;
      for (int i = 0; i < N; ++i) np += tiv[(size_t)TI_NPLAN * N + i] != tivb[(size_t)TI_NPLAN * N + i];
      nplan_env += np;
      nplan_steps += np > 0;
    }
    for (int i = 0; i < N; ++i) {
      ++len[i];
      const int fl = iv[(size_t)I_O_FL * N + i];
      grasps += (fl & OF_GRASP) != 0; slips += (fl & OF_SLIP) != 0;
      if (!dn[i] || t == 0) continue;
      const int k = ivb[(size_t)I_B_KIND * N + i];
      if (k < bsc::EK_B4) continue;
      const bsc::Entry& E = sb.ent[ivb[(size_t)I_B_ENT * N + i]];
      const float w = E.odim[0] < E.odim[1] ? E.odim[0] : E.odim[1];
      const int cl = w < 0.02f ? 0 : w < 0.04f ? 1 : w < 0.06f ? 2 : 3;
      const int hi = k == bsc::EK_B5 ? (E.dkind == bsc::DK_INSIDE ? 2 : E.dkind == bsc::DK_FLOOR ? 0 : 1) : (E.prim[0].lo[2] > 0.1f ? 1 : 0);
      long* q = tab[k][cl][hi];
      ++q[0]; q[1] += dn[i] == kSuccess; q[2] += dn[i] == kCollision; q[3] += ivb[(size_t)I_T_TRY * N + i] > KT::max_try;
      ++house[E.scene][k][0]; house[E.scene][k][1] += dn[i] == kSuccess;
      if (dn[i] != kSuccess && !sl) { const int f = tivb[(size_t)TI_FAIL * N + i]; ++why[k][f >= 0 && f < 16 ? f : 15]; }
      if (dn[i] != kSuccess && sl) { const int md = slr[i].mode; ++slwhy[k][md >= 0 && md < SLD_N ? md : 0]; }
      steps_sum += len[i];
      len[i] = 0;
    }
  }
  // 환경만 잰 값(같은 판 상태에서 교사 행동 그대로 — 지도·교사 빼고)
  float ms_env = 0.f;
  for (int t = 0; t < 50; ++t) {
    cudaEventRecord(e0);
    env.step(act, obs, rew, done);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float m1;
    cudaEventElapsedTime(&m1, e0, e1);
    ms_env += m1;
  }
  std::printf("pnp_teach: N %d steps %d kind %s strict %d map C0/C1 %.2f/%.2f fail %.3f/%.2f | env+map+teacher %.3f ms/step, env only %.3f ms/step\n", N, T,
              kind ? (kind == 4 ? "B4" : kind == 5 ? "B5" : "B6") : "B4+B5+B6", cu.strict, mc.p0, mc.p1, cu.p_slip, cu.p_occ, ms / (T - 10), ms_env / 50);
  const char* cn[4] = {"w<2cm ", "2-4cm ", "4-6cm ", ">=6cm "};
  long all[4] = {0, 0, 0, 0};
  for (int k = bsc::EK_B4; k < bsc::N_EK; ++k) {
    long kk[4] = {0, 0, 0, 0};
    for (int c = 0; c < 4; ++c)
      for (int h = 0; h < 3; ++h) {
        const long* q = tab[k][c][h];
        for (int z = 0; z < 4; ++z) kk[z] += q[z];
        if (q[0]) std::printf("  B%d %s %s: %5ld eps  success %.3f  collision %.3f  teacher gave up %.3f\n", k, cn[c], k == bsc::EK_B5 ? (h == 2 ? "place in " : h ? "place on " : "place flr") : h ? "surface  " : "floor    ", q[0], (double)q[1] / q[0], (double)q[2] / q[0], (double)q[3] / q[0]);
      }
    if (kk[0]) std::printf("  B%d all: %ld eps success %.3f collision %.3f gave up %.3f\n", k, kk[0], (double)kk[1] / kk[0], (double)kk[2] / kk[0], (double)kk[3] / kk[0]);
    for (int z = 0; z < 4; ++z) all[z] += kk[z];
  }
  std::printf("  all: %ld eps success %.3f; grasp events %ld, slips/drops %ld, mean episode length %.1f steps\n", all[0], all[0] ? (double)all[1] / all[0] : 0.0, grasps, slips,
              all[0] ? (double)steps_sum / all[0] : 0.0);
  const char* rn[12] = {"none(timeout etc.)", "wide", "heavy", "thin", "no stance", "no place", "no path", "arm plan at stance", "drops", "stuck", "closed without grasp", "pre-grasp timeout"};
  for (int k = bsc::EK_B4; k < bsc::N_EK; ++k) {
    long tot = 0;
    for (int f = 0; f < 16; ++f) tot += why[k][f];
    if (!tot) continue;
    std::printf("  B%d failures by teacher reason:", k);
    for (int f = 0; f < 12; ++f) if (why[k][f]) std::printf(" %s %ld", rn[f], why[k][f]);
    std::printf("\n");
  }
  if (sl) {
    const char* mn[SLD_N] = {"none", "fail", "nav", "app0", "rot", "armq0", "drive", "fine", "wait", "armg", "close", "reopen", "lift", "fold", "back", "armp", "open", "retreat", "done", "explore", "failarm", "backup"};
    for (int k = bsc::EK_B4; k < bsc::N_EK; ++k) {
      long tot = 0;
      for (int f = 0; f < SLD_N; ++f) tot += slwhy[k][f];
      if (!tot) continue;
      std::printf("  B%d failures by stateless-teacher mode at the end:", k);
      for (int f = 0; f < SLD_N; ++f) if (slwhy[k][f]) std::printf(" %s %ld", mn[f], slwhy[k][f]);
      std::printf("\n");
    }
  }
  if (agree) {
    const char* pn[15] = {"NAV", "APP", "PRE", "DOWN", "CLOSE", "LIFT", "BACK", "FOLD", "NAV2", "APP2", "PREPL", "OPEN", "RETREAT", "DONE", "EXPLORE"};
    const char* mn[SLD_N] = {"none", "fail", "nav", "app0", "rot", "armq0", "drive", "fine", "wait", "armg", "close", "reopen", "lift", "fold", "back", "armp", "open", "retreat", "done", "explore", "failarm", "backup"};
    long n = 0, e = 0, l = 0;
    std::printf("  agreement (stateless label vs stateful action on stateful-teacher states; tol = |da| <= 0.05 on every normalized action):\n");
    for (int ph = 0; ph < 15; ++ph) {
      if (!ag_n[ph]) continue;
      n += ag_n[ph]; e += ag_exact[ph]; l += ag_tol[ph];
      std::printf("    %-8s %8ld samples  exact %.3f  tol %.3f  base-tol %.3f  arm-tol %.3f |", pn[ph], ag_n[ph], (double)ag_exact[ph] / ag_n[ph], (double)ag_tol[ph] / ag_n[ph],
                  (double)ag_base[ph] / ag_n[ph], (double)ag_arm[ph] / ag_n[ph]);
      for (int m = 0; m < SLD_N; ++m) if (ag_conf[ph][m] * 50 > ag_n[ph]) std::printf(" %s %ld", mn[m], ag_conf[ph][m]);
      std::printf("\n");
    }
    if (n) std::printf("    all      %8ld samples  exact %.3f  tol %.3f\n", n, (double)e / n, (double)l / n);
  }
  for (int sc = 0; sc < sb.host.nsc; ++sc) {
    std::printf("  %-26s", sb.sc[sc].name.c_str());
    for (int k = bsc::EK_B4; k < bsc::N_EK; ++k) if (house[sc][k][0]) std::printf("  B%d %4ld eps %.3f", k, house[sc][k][0], (double)house[sc][k][1] / house[sc][k][0]);
    std::printf("\n");
  }
  std::printf("  teacher plans: %ld env-plans over %ld steps with planning (of %d); feasibility table %.1f s, sampling %s\n", nplan_env, nplan_steps, T, feas_s, all_eps ? "all" : "feasible only (PF_FEAS)");
  return 0;
}
