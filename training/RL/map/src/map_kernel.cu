// 지도 갱신 커널(K2 map_update): 판 하나 = 블록 하나(NT 스레드). 상태는 제자리 갱신(CUDA 그래프로 잡을 수 있게).
#include <cstdio>
#include <cstdlib>

#include "map_api.h"

namespace gmap {

struct BlockSync { __device__ void operator()() const { __syncthreads(); } };

__global__ void __launch_bounds__(NT) map_init_kernel(MapCore* core, int N, uint64_t seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) init_core(core[i], seed, i);
}

__global__ void __launch_bounds__(NT) map_step_kernel(env::Soa s, MapCore* core, int16_t* L, uint32_t* seen, float* met, int force_kf, int bug) {
  __shared__ MapCore m;
  __shared__ Scratch sh;
  const int i = blockIdx.x, tid = threadIdx.x, N = s.N;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(&core[i]);
  uint32_t* dst = reinterpret_cast<uint32_t*>(&m);
  for (int k = tid; k < CORE_WORDS; k += NT) dst[k] = src[k];
  __syncthreads();
  const EnvView e = read_env(s, i);
  map_block(m, sh, e, L + (size_t)i * NCELL, seen + (size_t)i * NWORD, met, N, i, tid, NT, bug, force_kf, BlockSync{});
  __syncthreads();
  uint32_t* out = reinterpret_cast<uint32_t*>(&core[i]);
  for (int k = tid; k < CORE_WORDS; k += NT) out[k] = dst[k];
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

DeviceMap::DeviceMap(int N, uint64_t seed) : N_(N) {
  CK(cudaMalloc(&core_, sizeof(MapCore) * (size_t)N));
  CK(cudaMalloc(&L_, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMalloc(&seen_, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMalloc(&met_, sizeof(float) * N_MET * (size_t)N));
  CK(cudaMemset(L_, 0, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMemset(seen_, 0, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMemset(met_, 0, sizeof(float) * N_MET * (size_t)N));
  map_init_kernel<<<(N + NT - 1) / NT, NT>>>(core_, N, seed);
  CK(cudaGetLastError());
}
DeviceMap::~DeviceMap() { cudaFree(core_); cudaFree(L_); cudaFree(seen_); cudaFree(met_); }

size_t DeviceMap::bytes() const { return (size_t)N_ * (sizeof(MapCore) + sizeof(int16_t) * NCELL + sizeof(uint32_t) * NWORD + sizeof(float) * N_MET); }

void DeviceMap::step(const env::Soa& s, int force_kf, int bug, cudaStream_t st) {
  map_step_kernel<<<N_, NT, 0, st>>>(s, core_, L_, seen_, met_, force_kf, bug);
}

void DeviceMap::download(MapHost& h) const {
  h.core.resize(N_); h.L.resize((size_t)NCELL * N_); h.seen.resize((size_t)NWORD * N_); h.met.resize((size_t)N_MET * N_);
  CK(cudaMemcpy(h.core.data(), core_, sizeof(MapCore) * h.core.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.L.data(), L_, sizeof(int16_t) * h.L.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.seen.data(), seen_, sizeof(uint32_t) * h.seen.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.met.data(), met_, sizeof(float) * h.met.size(), cudaMemcpyDeviceToHost));
}

}  // namespace gmap
