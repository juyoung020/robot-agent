// 처리량: 판 수를 늘려 가며 환경 스텝/초(= 판·제어스텝/초, 제어 스텝 하나 = 물리 서브스텝 10)를 잰다.  env_bench [steps=300]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "env_api.h"

using namespace env;

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 300;
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  std::printf("GPU %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  for (int N : {1024, 4096, 16384, 65536, 262144, 1048576}) {
    DeviceEnv e(N, 1, 1);
    float *act, *obs, *rew; int* done;
    cudaMalloc(&act, sizeof(float) * N_ACT * (size_t)N);
    cudaMalloc(&obs, sizeof(float) * N_OBS * (size_t)N);
    cudaMalloc(&rew, sizeof(float) * N);
    cudaMalloc(&done, sizeof(int) * N);
    std::vector<float> a((size_t)N_ACT * N);
    uint64_t s = 5;
    for (auto& x : a) x = dm::rand_range(s, -1.f, 1.f);
    cudaMemcpy(act, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice);
    for (int i = 0; i < 20; ++i) e.step(act, obs, rew, done);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < T; ++i) e.step(act, obs, rew, done);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    const double sps = (double)N * T / (ms * 1e-3);
    std::printf("N=%8d: %7.3f ms/step  %10.3e env-steps/s  (%.2f us per 1000 envs)\n", N, ms / T, sps, ms / T * 1e3 / (N / 1000.0));
    cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
  }
  return 0;
}
