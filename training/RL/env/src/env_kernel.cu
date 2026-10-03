// 환경 커널: 스레드 하나 = 판 하나. 호스트 쪽은 장치 메모리와 실행만(상태는 제자리 갱신 — CUDA 그래프로 잡을 수 있게).
#include <cstdio>
#include <vector>

#include "env_api.h"
#include "env_soa.h"

namespace env {

__global__ void __launch_bounds__(128) init_kernel(Soa s, uint64_t seed, int stage) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) init_env(s, i, seed, stage);
}
__global__ void __launch_bounds__(128) step_kernel(Soa s, const float* act, float* obs, float* rew, int* done, int stage, int arm_free, int bug) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s.N) step_env(s, i, act, obs, rew, done, stage, arm_free != 0, bug);
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while (0)

DeviceEnv::DeviceEnv(int N, int stage, uint64_t seed, bool arm_free) : N_(N), stage_(stage), arm_free_(arm_free) {
  CK(cudaMalloc(&f_, sizeof(float) * NUM_F * (size_t)N));
  CK(cudaMalloc(&iv_, sizeof(int) * NUM_I * (size_t)N));
  CK(cudaMalloc(&rng_, sizeof(uint64_t) * (size_t)N));
  Soa s{f_, iv_, rng_, N};
  init_kernel<<<(N + 127) / 128, 128>>>(s, seed, stage);
  CK(cudaGetLastError());
}
DeviceEnv::~DeviceEnv() { cudaFree(f_); cudaFree(iv_); cudaFree(rng_); }

void DeviceEnv::step(const float* act, float* obs, float* rew, int* done, int bug) {
  Soa s{f_, iv_, rng_, N_};
  step_kernel<<<(N_ + 127) / 128, 128>>>(s, act, obs, rew, done, stage_, arm_free_ ? 1 : 0, bug);
}

void DeviceEnv::download(std::vector<float>& f, std::vector<int>& iv, std::vector<uint64_t>& rng) const {
  f.resize((size_t)NUM_F * N_); iv.resize((size_t)NUM_I * N_); rng.resize(N_);
  CK(cudaMemcpy(f.data(), f_, sizeof(float) * f.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(iv.data(), iv_, sizeof(int) * iv.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rng.data(), rng_, sizeof(uint64_t) * rng.size(), cudaMemcpyDeviceToHost));
}

}  // namespace env
