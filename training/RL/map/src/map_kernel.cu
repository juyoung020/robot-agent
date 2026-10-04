// 지도 갱신 커널(K2 map_update). 두 단계: 판마다 한 스레드가 시작(움직임 거르기)을 하고, keyframe 인 판만 블록 하나(NT 스레드)가
// 갱신한다. 상태는 제자리 갱신(CUDA 그래프로 잡을 수 있게), 호스트 동기 없음.
#include <cstdio>
#include <cstdlib>

#include "map_api.h"

namespace gmap {

struct BlockSync { __device__ void operator()() const { __syncthreads(); } };

__global__ void __launch_bounds__(NT) map_init_kernel(MapCore* core, int N, uint64_t seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) init_core(core[i], seed, i);
}

// 1 단계(판마다 한 스레드): 시작(리셋·오도메트리·움직임 거르기). keyframe·리셋인 판은 목록(장치 안, 원자 덧셈)에 넣고,
// 아닌 판은 여기서 완성도만 쓰고 끝난다. 목록 순서는 실행마다 달라도 판끼리 독립이라 결과는 같다.
constexpr int BEGIN_NT = 128;
__global__ void __launch_bounds__(BEGIN_NT) map_begin_kernel(env::Soa s, MapCore* core, float* met, uint32_t* list, int* count, int force_kf) {
  const int i = blockIdx.x * BEGIN_NT + threadIdx.x, N = s.N;
  if (i >= N) return;
  const EnvView e = read_env(s, i);
  MapCore& m = core[i];
  const int f = phase_begin(m, e, force_kf);
#ifdef MAP_PROF
  atomicAdd(&g_prof[P_NSEC + ((f & B_KF) ? 1 : 0)], 1ull);
#endif
  if (f) list[atomicAdd(count, 1)] = (uint32_t)i | ((uint32_t)f << 30);
  else write_metrics(m, e, met, N, i);
}

// 2 단계(목록의 판 하나 = 블록 하나, NT 스레드): keyframe 갱신. 블록 수는 N 으로 고정하고 목록 길이 밖 블록은 바로 끝난다(호스트 동기 없음).
// 블록이 대략 번호 순서로 SM 에 올라가므로, 한 물결(SM 수 × 상주 블록) 뒤 판의 MapCore 를 L2 로 미리 당긴다(값은 안 바뀜).
constexpr int PF_AHEAD = 1024;
__global__ void __launch_bounds__(NT, 10) map_kf_kernel(env::Soa s, MapCore* core, int16_t* L, uint32_t* seen, float* met,
                                                        const uint32_t* list, const int* count, int bug) {
  __shared__ MapCore m;
  __shared__ Scratch sh;
  const int j = blockIdx.x, tid = threadIdx.x, N = s.N;
  const int n = *count;
  if (j >= n) return;
  PROF_START();
  const uint32_t ent = list[j];
  const int i = (int)(ent & 0x3fffffffu), flags = (int)(ent >> 30);
  if (tid < (int)((sizeof(MapCore) + 127) / 128) && j + PF_AHEAD < n)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(&core[list[j + PF_AHEAD] & 0x3fffffffu]) + tid * 128));
  const uint32_t* src = reinterpret_cast<const uint32_t*>(&core[i]);
  uint32_t* dst = reinterpret_cast<uint32_t*>(&m);
  for (int k = tid; k < CORE_WORDS; k += NT) dst[k] = src[k];
  const EnvView e = read_env(s, i);   // 코어 읽기와 겹침
  __syncthreads();
  PROF_MARK(P_LOAD);
  map_rest(m, sh, e, L + (size_t)i * NCELL, seen + (size_t)i * NWORD, met, N, i, tid, NT, bug, flags, BlockSync{});
  __syncthreads();
  uint32_t* out = reinterpret_cast<uint32_t*>(&core[i]);
  for (int k = tid; k < CORE_WORDS; k += NT) out[k] = dst[k];
  PROF_MARK(P_STORE);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

DeviceMap::DeviceMap(int N, uint64_t seed) : N_(N) {
  CK(cudaMalloc(&core_, sizeof(MapCore) * (size_t)N));
  CK(cudaMalloc(&L_, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMalloc(&seen_, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMalloc(&met_, sizeof(float) * N_MET * (size_t)N));
  CK(cudaMalloc(&list_, sizeof(uint32_t) * (size_t)N));
  CK(cudaMalloc(&count_, sizeof(int)));
  CK(cudaMemset(L_, 0, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMemset(seen_, 0, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMemset(met_, 0, sizeof(float) * N_MET * (size_t)N));
  map_init_kernel<<<(N + NT - 1) / NT, NT>>>(core_, N, seed);
  CK(cudaGetLastError());
}
DeviceMap::~DeviceMap() { cudaFree(core_); cudaFree(L_); cudaFree(seen_); cudaFree(met_); cudaFree(list_); cudaFree(count_); }

size_t DeviceMap::bytes() const { return (size_t)N_ * (sizeof(MapCore) + sizeof(int16_t) * NCELL + sizeof(uint32_t) * NWORD + sizeof(float) * N_MET); }

void DeviceMap::step(const env::Soa& s, int force_kf, int bug, cudaStream_t st) {
  // 목록 길이를 0 으로(비동기, 그래프로 잡힘) → 시작 커널(판마다 스레드) → keyframe 커널(목록의 판만 일함)
  CK(cudaMemsetAsync(count_, 0, sizeof(int), st));
  map_begin_kernel<<<(N_ + BEGIN_NT - 1) / BEGIN_NT, BEGIN_NT, 0, st>>>(s, core_, met_, list_, count_, force_kf);
  map_kf_kernel<<<N_, NT, 0, st>>>(s, core_, L_, seen_, met_, list_, count_, bug);
}

void prof_reset() {
#ifdef MAP_PROF
  const unsigned long long z[P_NSEC + 3] = {};
  CK(cudaMemcpyToSymbol(g_prof, z, sizeof z));
#endif
}
void prof_read(unsigned long long out[P_NSEC + 3]) {
  for (int k = 0; k < P_NSEC + 3; ++k) out[k] = 0;
#ifdef MAP_PROF
  CK(cudaMemcpyFromSymbol(out, g_prof, sizeof(unsigned long long) * (P_NSEC + 3)));
#endif
}

void DeviceMap::download(MapHost& h) const {
  h.core.resize(N_); h.L.resize((size_t)NCELL * N_); h.seen.resize((size_t)NWORD * N_); h.met.resize((size_t)N_MET * N_);
  CK(cudaMemcpy(h.core.data(), core_, sizeof(MapCore) * h.core.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.L.data(), L_, sizeof(int16_t) * h.L.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.seen.data(), seen_, sizeof(uint32_t) * h.seen.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.met.data(), met_, sizeof(float) * h.met.size(), cudaMemcpyDeviceToHost));
}

}  // namespace gmap
