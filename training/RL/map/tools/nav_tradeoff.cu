// 다가가기 거리장 주기 K 의 대가(E2 목표 2 "측정한 trade-off"): 같은 씨앗·같은 행동이면 환경 궤적은 거리장과 무관하다(거리장은 보상·관측 74 만 바꿈).
// 그래서 K = 1(매 스텝)을 기준으로 K = 5, 10, 20 판을 같은 행동으로 돌려 스텝 보상·거리 관측·판 보상 합의 차를 잰다(궤적이 같은지도 확인).
//   nav_tradeoff [N=4096] [steps=400] [--curr p0,p1]   (BEHAVIOR 집, ~/ra_b1k, B1–B3 섞음)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bscene_host.h"
#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"

using namespace env;

__global__ void policy_kernel(const float* obs, float* act, int N, int t) {   // map_bench 와 같은 몰기
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float a[N_ACT];
  approach_action(obs, N, i, a);
  if (((i + t / 30) & 3) == 0) { a[0] = 0.6f; a[1] = 0.5f; }
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
}

int main(int argc, char** argv) {
  int N = 4096, T = 400;
  float p0 = 0.f, p1 = 0.f;
  int pos = 0;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--curr") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &p0, &p1);
    else if (pos++ == 0) N = std::atoi(argv[a]);
    else T = std::atoi(argv[a]);
  }
  bsc::SceneBuild sb;
  std::string err;
  bsc::BuildOpt bo;
  if (!bsc::build_scenes(bo, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
  const int Ks[4] = {1, 5, 10, 20};
  std::vector<float> r1((size_t)N * T), d1((size_t)N * T), x1((size_t)N * T);
  std::vector<int> e1((size_t)N * T);
  std::printf("nav_tradeoff: N=%d steps=%d, first map C0 %.2f C1 %.2f (rest C2)\n", N, T, p0, p1);
  for (int ki = 0; ki < 4; ++ki) {
    const int K = Ks[ki];
    DeviceEnv e(N, kStageBeh, 11, false, sb.dev);
    gmap::DeviceMap m(N, 13, sb.dev);
    e.set_nav(m.nav_fb());
    gmap::MapCurr cu = gmap::kCurrEmpty;
    cu.p0 = p0; cu.p1 = p1; cu.nav_k = K;
    cudaMemcpy(m.curr_dev(), &cu, sizeof cu, cudaMemcpyHostToDevice);
    float *act, *obs, *rew; int* done;
    cudaMalloc(&act, sizeof(float) * N_ACT * N); cudaMalloc(&obs, sizeof(float) * N_OBS * N); cudaMalloc(&rew, sizeof(float) * N); cudaMalloc(&done, sizeof(int) * N);
    cudaMemset(obs, 0, sizeof(float) * N_OBS * N);
    std::vector<float> hr(N), hd(N), hx(N);
    std::vector<int> hdone(N);
    double sum_abs_r = 0, sum_abs_d = 0, n_diff = 0, ep_abs = 0, ep_ret1 = 0;
    long n = 0, n_ep = 0, traj_diff = 0, n_end[4] = {0, 0, 0, 0};
    std::vector<double> ret(N, 0.0), ret1(N, 0.0);
    for (int t = 0; t < T; ++t) {
      policy_kernel<<<(N + 127) / 128, 128>>>(obs, act, N, t);
      e.step(act, obs, rew, done);
      m.step(e.soa(), 0, 0, 0);
      cudaMemcpy(hr.data(), rew, 4 * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(hd.data(), obs + (size_t)(N_BODY + N_RAYS + 2) * N, 4 * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(hx.data(), e.soa().f + (size_t)F_X * N, 4 * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(hdone.data(), done, 4 * N, cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; ++i) {
        const size_t k = (size_t)t * N + i;
        if (hdone[i] >= 0 && hdone[i] < 4) ++n_end[hdone[i]];
        if (ki == 0) { r1[k] = hr[i]; d1[k] = hd[i]; x1[k] = hx[i]; e1[k] = hdone[i]; continue; }
        traj_diff += std::memcmp(&hx[i], &x1[k], 4) != 0 || hdone[i] != e1[k];
        sum_abs_r += std::fabs(hr[i] - r1[k]);
        sum_abs_d += std::fabs(hd[i] - d1[k]);
        n_diff += hr[i] != r1[k];
        ++n;
        ret[i] += hr[i]; ret1[i] += r1[k];
        if (hdone[i] != 0) { ep_abs += std::fabs(ret[i] - ret1[i]); ep_ret1 += std::fabs(ret1[i]); ++n_ep; ret[i] = 0; ret1[i] = 0; }
      }
    }
    if (ki == 0) std::printf("  K=1 (reference): episodes ended success %ld collision %ld timeout %ld\n", n_end[1], n_end[2], n_end[3]);
    else
      std::printf("  K=%2d vs K=1: trajectory differs %ld env-steps; per-step |reward diff| mean %.4f, steps with any diff %.3f; |obs path distance diff| mean %.3f m;"
                  " per-episode |return diff| mean %.3f (|return| mean %.3f, %ld episodes)\n",
                  K, traj_diff, sum_abs_r / std::max(1L, n), n_diff / std::max(1L, n), sum_abs_d / std::max(1L, n), ep_abs / std::max(1L, n_ep),
                  ep_ret1 / std::max(1L, n_ep), n_ep);
    cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
  }
  return 0;
}
