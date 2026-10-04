// 처리량: 판 수를 늘려 가며 환경 스텝/초(= 판·제어스텝/초, 제어 스텝 하나 = 물리 서브스텝 10)를 잰다.  env_bench [steps=300] [stage=1] [maxN]
// stage 3 = BEHAVIOR 집(B1–B3 섞음, ~/ra_b1k, 지도 되먹임 없음). 잡기 물리(E6): env_bench 300 3 4096 --pnp p4,p5,p6 [--phys BITS] [--teacher] [--mix p1,p2]
//   --teacher: 행동 = 대본 교사(teacher 커널 시간 따로)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "env_api.h"
#include "bscene_host.h"

using namespace env;
#ifdef ENV_PROF
namespace env { void env_prof_read(unsigned long long out[8]); void env_prof_reset(); }
#endif

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 300, stage = argc > 2 ? std::atoi(argv[2]) : 1, maxN = argc > 3 ? std::atoi(argv[3]) : 1048576;
  bsc::BCurr cu = bsc::kBCurrDefault;
  bool teach = false;
  for (int a = 4; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--pnp") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f,%f", &cu.p4, &cu.p5, &cu.p6);
    else if (!std::strcmp(argv[a], "--mix") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &cu.p1, &cu.p2);
    else if (!std::strcmp(argv[a], "--phys") && a + 1 < argc) cu.phys = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--teacher")) teach = true;
  }
  bsc::SceneBuild sb;
  if (stage >= kStageBeh) {
    std::string err;
    bsc::BuildOpt bo;
    if (!bsc::build_scenes(bo, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
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
    for (int i = 0; i < 20; ++i) { if (teach) e.teacher(act); e.step(act, obs, rew, done); }
#ifdef ENV_PROF
    env_prof_reset();
#endif
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    float tms = 0.f;
    if (teach) {   // 교사 커널만 따로 재고, 스텝은 그 행동으로
      cudaEvent_t a0, a1;
      cudaEventCreate(&a0); cudaEventCreate(&a1);
      cudaEventRecord(t0);
      for (int i = 0; i < T; ++i) {
        cudaEventRecord(a0); e.teacher(act); cudaEventRecord(a1); e.step(act, obs, rew, done);
        cudaEventSynchronize(a1);
        float x; cudaEventElapsedTime(&x, a0, a1); tms += x;
      }
    } else
    for (int i = 0; i < T; ++i) e.step(act, obs, rew, done);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    const double sps = (double)N * T / (ms * 1e-3);
    std::printf("N=%8d: %7.3f ms/step  %10.3e env-steps/s  (%.2f us per 1000 envs)%s", N, ms / T, sps, ms / T * 1e3 / (N / 1000.0), teach ? "" : "\n");
    if (teach) std::printf("  of which teacher kernel %.3f ms/step\n", tms / T);
#ifdef ENV_PROF
    unsigned long long pr[8];
    env_prof_read(pr);
    std::printf("   reset: %.4f resets/env-step, warp time waiting on resets %.1f %% of step-kernel warp time; pick-place resets %llu, spawn tries %.2f avg, "
                "%.4f fell back to table start; mean reset %.0f cycles\n",
                (double)pr[2] / ((double)N * T), 100.0 * (double)pr[0] / (double)(pr[1] ? pr[1] : 1), pr[3], pr[3] ? (double)pr[4] / pr[3] : 0.0,
                pr[3] ? (double)pr[5] / pr[3] : 0.0, pr[2] ? (double)pr[6] / pr[2] : 0.0);
#endif
    cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
  }
  return 0;
}
