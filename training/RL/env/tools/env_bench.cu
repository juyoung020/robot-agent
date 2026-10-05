// 처리량: 판 수를 늘려 가며 환경 스텝/초(= 판·제어스텝/초, 제어 스텝 하나 = 물리 서브스텝 10)를 잰다.  env_bench [steps=300] [stage=1] [maxN]
// stage 3 = BEHAVIOR 집(B1–B3 섞음, data/b1k_scenes($RA_B1K_SCENES), 지도 되먹임 없음). 잡기 물리(E6): env_bench 300 3 4096 --pnp p4,p5,p6 [--phys BITS] [--teacher] [--mix p1,p2]
//   --teacher: 행동 = 대본 교사(teacher 커널 시간 따로: 앞·계획·행동 커널을 각각 재고, 계획한 판이 있는 스텝/없는 스텝으로 나눔)
//   --feas: 잡기 가능 표(env pnp_feasibility) + PF_FEAS 고르기(교사가 표의 서는 자리를 씀)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "env_api.h"
#include "bscene_host.h"

using namespace env;
#ifdef ENV_PROF
namespace env { void env_prof_read(unsigned long long out[8]); void env_prof_reset(); void tprof_read(unsigned long long out[64][3]); void tseg_read(unsigned long long out[16][2]); void tcause_read(unsigned long long out[2][16]); void tbin_read(unsigned long long out[3][8]); void tact_read(unsigned long long out[16][3]); void tdbg_read(unsigned long long out[32]); void tkc_read(unsigned long long out[3][5]); }
#endif

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 300, stage = argc > 2 ? std::atoi(argv[2]) : 1, maxN = argc > 3 ? std::atoi(argv[3]) : 1048576;
  bsc::BCurr cu = bsc::kBCurrDefault;
  bool teach = false, feas = false, tsl = false, gcand = false;
  int sltol = 0;
  for (int a = 4; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--pnp") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f,%f", &cu.p4, &cu.p5, &cu.p6);
    else if (!std::strcmp(argv[a], "--mix") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &cu.p1, &cu.p2);
    else if (!std::strcmp(argv[a], "--phys") && a + 1 < argc) cu.phys = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--teacher")) teach = true;
    else if (!std::strcmp(argv[a], "--teacher-sl")) { teach = true; tsl = true; }   // 상태 없는 교사(teacher_sl.h): 교사 시간 = 앞 + 계획 + 행동 합
    else if (!std::strcmp(argv[a], "--feas")) feas = true;
    else if (!std::strcmp(argv[a], "--gcand")) gcand = true;
    else if (!std::strcmp(argv[a], "--sltol")) sltol |= 1;   // 상태 없는 교사 배울 수 있는 단계 문턱(teacher_sl.h sl_tol)
    else if (!std::strcmp(argv[a], "--slknown")) sltol |= 2;   // 상태 없는 교사 특권은 지도에 확정된 집을 물체만(B4·B5 도 탐사)
  }
  bsc::SceneBuild sb;
  if (stage >= kStageBeh) {
    std::string err;
    bsc::BuildOpt bo;
    if (!bsc::build_scenes(bo, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
    if (feas) { pnp_feasibility(sb); cu.phys |= bsc::PF_FEAS; if (gcand) pnp_stance_cands(sb); if (sltol) set_sl_tol(sb, sltol); }
  }
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  std::printf("GPU %s (sm_%d%d, %d SMs), stage A%d\n", p.name, p.major, p.minor, p.multiProcessorCount, stage);
  for (int N : {1024, 4096, 16384, 32768, 65536, 262144, 1048576}) {
    if (N > maxN) break;
    const bool pnp = cu.p4 + cu.p5 + cu.p6 > 0.f;
    DeviceEnv e(N, stage, 1, pnp, sb.dev, cu);
    float *act, *obs, *rew; int* done;
    cudaMalloc(&act, sizeof(float) * N_ACT * (size_t)N);
    cudaMalloc(&obs, sizeof(float) * N_OBS * (size_t)N);
    cudaMalloc(&rew, sizeof(float) * N);
    cudaMalloc(&done, sizeof(int) * N);
    std::vector<float> a((size_t)N_ACT * N);
    uint64_t s = 5;
    for (auto& x : a) x = dm::rand_range(s, -1.f, 1.f);
    cudaMemcpy(act, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice);
    if (tsl) e.enable_teacher_sl();
    for (int i = 0; i < 20; ++i) { if (tsl) e.teacher_sl(act); else if (teach) e.teacher(act); e.step(act, obs, rew, done); }
#ifdef ENV_PROF
    env_prof_reset();
#endif
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    float tms = 0.f;
    float tpre = 0.f, tplan_on = 0.f, tplan_off = 0.f, tact = 0.f, tenv = 0.f;
    long n_on = 0, n_plans = 0;
    if (tsl) {
      cudaEvent_t a0, a1, a2;
      cudaEventCreate(&a0); cudaEventCreate(&a1); cudaEventCreate(&a2);
      cudaEventRecord(t0);
      for (int i = 0; i < T; ++i) {
        cudaEventRecord(a0); e.teacher_sl(act); cudaEventRecord(a1); e.step(act, obs, rew, done); cudaEventRecord(a2);
        cudaEventSynchronize(a2);
        float x0, x1;
        cudaEventElapsedTime(&x0, a0, a1); cudaEventElapsedTime(&x1, a1, a2);
        tms += x0; tenv += x1;
      }
    } else if (teach) {   // 교사 커널 셋을 따로 재고, 스텝은 그 행동으로. 계획 목록 수를 읽음(재기용 동기)
      cudaEvent_t a0, a1, a2, a3, a4;
      cudaEventCreate(&a0); cudaEventCreate(&a1); cudaEventCreate(&a2); cudaEventCreate(&a3); cudaEventCreate(&a4);
      cudaEventRecord(t0);
      for (int i = 0; i < T; ++i) {
        cudaEventRecord(a0); e.teacher_pre(bsc::NavFb{nullptr, nullptr, nullptr, nullptr}); cudaEventRecord(a1);
        int nl = 0;
        cudaMemcpy(&nl, e.tbuf().list, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(a1);
        e.teacher_plan(); cudaEventRecord(a2); e.teacher_act(act); cudaEventRecord(a3);
        e.step(act, obs, rew, done); cudaEventRecord(a4);
        cudaEventSynchronize(a4);
        float x0, x1, x2, x3;
        cudaEventElapsedTime(&x0, a0, a1); cudaEventElapsedTime(&x1, a1, a2); cudaEventElapsedTime(&x2, a2, a3); cudaEventElapsedTime(&x3, a3, a4);
        tpre += x0; tact += x2; tenv += x3; tms += x0 + x1 + x2;
        if (nl > 0) { tplan_on += x1; ++n_on; n_plans += nl; } else tplan_off += x1;
      }
    } else
    for (int i = 0; i < T; ++i) e.step(act, obs, rew, done);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    const double sps = (double)N * T / (ms * 1e-3);
    std::printf("N=%8d: %7.3f ms/step  %10.3e env-steps/s  (%.2f us per 1000 envs)%s", N, ms / T, sps, ms / T * 1e3 / (N / 1000.0), teach ? "" : "\n");
    if (tsl) std::printf("  stateless teacher %.3f ms/step | env step %.3f ms\n", tms / T, tenv / T);
    else if (teach)
      std::printf("  teacher %.3f ms/step (pre %.3f + act %.3f per step; plan kernel %.3f ms on the %ld steps with planning (%.1f envs planned per such step), %.3f ms on the %ld steps without) | env step %.3f ms\n",
                  tms / T, tpre / T, tact / T, n_on ? tplan_on / n_on : 0.f, n_on, n_on ? (double)n_plans / n_on : 0.0, (T - n_on) ? tplan_off / (T - n_on) : 0.f, (long)(T - n_on), tenv / T);
#ifdef ENV_PROF
    unsigned long long pr[8];
    env_prof_read(pr);
    std::printf("   reset: %.4f resets/env-step, warp time waiting on resets %.1f %% of step-kernel warp time; pick-place resets %llu, spawn tries %.2f avg, "
                "%.4f fell back to table start; mean reset %.0f cycles\n",
                (double)pr[2] / ((double)N * T), 100.0 * (double)pr[0] / (double)(pr[1] ? pr[1] : 1), pr[3], pr[3] ? (double)pr[4] / pr[3] : 0.0,
                pr[3] ? (double)pr[5] / pr[3] : 0.0, pr[2] ? (double)pr[6] / pr[2] : 0.0);
#endif
#ifdef ENV_PROF
    if (teach) {
      static unsigned long long tp[64][3];
      tprof_read(tp);
      int clk = 0;
      cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
      for (int k = 0; k < 64; ++k)
        if (tp[k][0]) std::printf("   plan need %2d: %llu plans, mean %.3f ms, max %.3f ms (warp clock)\n", k, tp[k][0], (double)tp[k][1] / tp[k][0] / clk, (double)tp[k][2] / clk);
      static unsigned long long sg[16][2];
      tseg_read(sg);
      const char* nm[16] = {"arm at stance (grasp)", "arm at stance (place)", "occupancy + BFS", "stance search (grasp)", "stance search (place)", "table-stance arm plan",
                            "path extract", "", "  occupancy paint", "  BFS x2", "new grasp searches", "new place searches", "searches found", "searches failed", "nopath (lev in slice)", "nopath (after cont)"};
      for (int k = 0; k < 16; ++k)
        if (sg[k][0]) {
          if (k == 12 || k == 13) std::printf("   seg %-24s: %llu, mean slices %.1f\n", nm[k], sg[k][0], (double)sg[k][1] / sg[k][0]);
          else std::printf("   seg %-24s: %llu, mean %.3f ms, total %.1f ms\n", nm[k], sg[k][0], (double)sg[k][1] / sg[k][0] / clk, (double)sg[k][1] / clk);
        }
      static unsigned long long tk[3][5];
      tkc_read(tk);
      for (int r = 0; r < 3; ++r) std::printf("   place slices %s by center: %llu %llu %llu %llu %llu\n", r == 0 ? "none" : r == 1 ? "found" : "cont", tk[r][0], tk[r][1], tk[r][2], tk[r][3], tk[r][4]);
      static unsigned long long td[32];
      tdbg_read(td);
      std::printf("   reject counters:");
      for (int k = 0; k < 32; ++k) if (td[k]) std::printf(" %d=%llu", k, td[k]);
      std::printf("\n");
      static unsigned long long ta[16][3];
      tact_read(ta);
      for (int k = 0; k < 16; ++k)
        if (ta[k][0]) std::printf("   act phase %2d: %llu, mean %.4f ms, max %.3f ms\n", k, ta[k][0], (double)ta[k][1] / ta[k][0] / clk, (double)ta[k][2] / clk);
      static unsigned long long tbn[3][8];
      tbin_read(tbn);
      for (int g = 0; g < 3; ++g) {
        std::printf("   found stance grade %d by cost bin (0..6, direct):", g);
        for (int k = 0; k < 8; ++k) std::printf(" %llu", tbn[g][k]);
        std::printf("\n");
      }
      static unsigned long long tc[2][16];
      tcause_read(tc);
      for (int a = 0; a < 2; ++a) {
        std::printf("   new %s searches by last fail:", a ? "place" : "grasp");
        for (int k = 0; k < 16; ++k) if (tc[a][k]) std::printf(" r%d=%llu", k, tc[a][k]);
        std::printf("\n");
      }
    }
#endif
    cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
  }
  return 0;
}
