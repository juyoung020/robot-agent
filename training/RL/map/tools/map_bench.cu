// 처리량: 판 수를 늘려 가며 지도 단계의 keyframe 갱신/초를 잰다(G1 환경 + 장치 안 접근 제어로 몬다).
//   map_bench [steps=200] [maxN=32768]
// 줄마다: 움직임 거르기 그대로(자연) / 매 스텝 keyframe(force). 지도 커널 시간만 이벤트로 재고, keyframe 수는 장치 카운터로 센다.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"

using namespace env;

__global__ void policy_kernel(const float* obs, float* act, int N, int t) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float a[N_ACT];
  approach_action(obs, N, i, a);
  if (((i + t / 30) & 3) == 0) { a[0] = 0.6f; a[1] = 0.5f; }   // 넷 중 하나는 30 스텝씩 돌며 달림(탐색처럼)
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
}

static long total_kf(const gmap::DeviceMap& m) {
  gmap::MapHost h;
  m.download(h);
  long s = 0;
  for (auto& c : h.core) s += c.n_kf_total;
  return s;
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 200;
  const int maxN = argc > 2 ? std::atoi(argv[2]) : 32768;
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  std::printf("GPU %s (sm_%d%d, %d SMs). map per env: %zu B (grid %d x %d @ %.2f m)\n", p.name, p.major, p.minor, p.multiProcessorCount,
              sizeof(gmap::MapCore) + sizeof(int16_t) * gmap::NCELL + 4 * gmap::NWORD, gmap::GW, gmap::GW, gmap::RES);
  for (int N : {1024, 4096, 16384, 32768, 65536}) {
    if (N > maxN) break;
    for (int force = 0; force < 2; ++force) {
      DeviceEnv e(N, 1, 1);
      gmap::DeviceMap m(N, 7);
      float *act, *obs, *rew; int* done;
      cudaMalloc(&act, sizeof(float) * N_ACT * (size_t)N);
      cudaMalloc(&obs, sizeof(float) * N_OBS * (size_t)N);
      cudaMalloc(&rew, sizeof(float) * N);
      cudaMalloc(&done, sizeof(int) * N);
      cudaMemset(obs, 0, sizeof(float) * N_OBS * (size_t)N);
      auto one = [&](int t, cudaEvent_t a, cudaEvent_t b) {
        policy_kernel<<<(N + 127) / 128, 128>>>(obs, act, N, t);
        e.step(act, obs, rew, done);
        if (a) cudaEventRecord(a);
        m.step(e.soa(), force);
        if (b) cudaEventRecord(b);
      };
      for (int t = 0; t < 20; ++t) one(t, nullptr, nullptr);
      cudaDeviceSynchronize();
      const long kf0 = total_kf(m);
      std::vector<cudaEvent_t> ev(2 * T);
      for (auto& x : ev) cudaEventCreate(&x);
      cudaEvent_t w0, w1;
      cudaEventCreate(&w0); cudaEventCreate(&w1);
      cudaEventRecord(w0);
      for (int t = 0; t < T; ++t) one(20 + t, ev[2 * t], ev[2 * t + 1]);
      cudaEventRecord(w1);
      cudaEventSynchronize(w1);
      float map_ms = 0, wall_ms = 0;
      for (int t = 0; t < T; ++t) { float x; cudaEventElapsedTime(&x, ev[2 * t], ev[2 * t + 1]); map_ms += x; }
      cudaEventElapsedTime(&wall_ms, w0, w1);
      const long kf = total_kf(m) - kf0;
      std::printf("N=%6d %s: keyframes %5.1f %% of steps | map kernel %7.3f ms/step | %.3e keyframe-updates/s | env+policy+map %.3e env-steps/s\n",
                  N, force ? "force  " : "natural", 100.0 * kf / ((double)N * T), map_ms / T, kf / (map_ms * 1e-3), (double)N * T / (wall_ms * 1e-3));
      for (auto& x : ev) cudaEventDestroy(x);
      cudaEventDestroy(w0); cudaEventDestroy(w1);
      cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
    }
  }
  return 0;
}
