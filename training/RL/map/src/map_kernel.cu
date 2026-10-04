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
  if (f) list[atomicAdd(count, 1)] = (uint32_t)i | ((uint32_t)f << LIST_SHIFT);
  else write_metrics(m, e, met, N, i);
}

// 2 단계(목록의 판 하나 = 블록 하나, NT 스레드): keyframe 갱신. 블록 수는 N 으로 고정하고 목록 길이 밖 블록은 바로 끝난다(호스트 동기 없음).
// 블록이 대략 번호 순서로 SM 에 올라가므로, 한 물결(SM 수 × 상주 블록) 뒤 판의 MapCore 를 L2 로 미리 당긴다(값은 안 바뀜).
constexpr int PF_AHEAD = 1024;
// read_env 가 읽는 SoA 값의 자리(미리 당기기용): x, y, yaw, v, w, tx, ty, rhx, rhy, q[N_Q], ep
constexpr int ENV_PF_N = 9 + env::N_Q + 1;
__device__ __forceinline__ const void* env_pf_addr(const env::Soa& s, int i, int k) {
  const int N = s.N;
  constexpr int F[9] = {env::F_X, env::F_Y, env::F_YAW, env::F_V, env::F_W, env::F_TX, env::F_TY, env::F_RHX, env::F_RHY};
  if (k < 9) return s.f + (size_t)F[k] * N + i;
  if (k < 9 + env::N_Q) return s.f + (size_t)(env::F_Q0 + k - 9) * N + i;
  return s.iv + (size_t)env::I_EP * N + i;
}
__global__ void __launch_bounds__(NT, 10) map_kf_kernel(env::Soa s, MapCore* core, int16_t* L, uint32_t* seen, uint32_t* occ,
                                                        int16_t* segs, float* met, const uint32_t* list, const int* count, int bug) {
  __shared__ __align__(16) MapCore m;
  __shared__ KfShared u;
  const int j = blockIdx.x, tid = threadIdx.x, N = s.N;
  const int n = *count;
  if (j >= n) return;
  PROF_START();
  const uint32_t ent = list[j];
  constexpr uint32_t IMASK = (1u << LIST_SHIFT) - 1u;
  const int i = (int)(ent & IMASK), flags = (int)(ent >> LIST_SHIFT);
  if (j + PF_AHEAD < n) {   // 한 물결 뒤 판: MapCore 와 환경 SoA 값(판마다 따로 떨어진 줄)을 L2 로
    const uint32_t ia = list[j + PF_AHEAD] & IMASK;
    if (tid < (int)((sizeof(MapCore) + 127) / 128))
      asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(&core[ia]) + tid * 128));
    else if (tid >= 32 && tid < 32 + ENV_PF_N)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(env_pf_addr(s, (int)ia, tid - 32)));
  }
  if (tid < (int)(sizeof(uint32_t) * NWORD / 128) && (flags & (B_KF | B_WALL)))   // 벽 단계가 읽을 점유 비트 2 KB 를 L2 로 미리
    asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(occ + (size_t)i * NWORD) + tid * 128));
  static_assert(sizeof(MapCore) % 16 == 0, "MapCore copied in 16-byte pieces");
  const uint4* src = reinterpret_cast<const uint4*>(&core[i]);
  uint4* dst = reinterpret_cast<uint4*>(&m);
  for (int k = tid; k < (int)(sizeof(MapCore) / 16); k += NT) dst[k] = src[k];
  const EnvView e = read_env(s, i);   // 코어 읽기와 겹침
  __syncthreads();
  PROF_MARK(P_LOAD);
  const MapGrid g{L + (size_t)i * NCELL, seen + (size_t)i * NWORD, occ + (size_t)i * NWORD, segs + (size_t)i * SEGW};
  map_rest(m, u, e, g, met, N, i, tid, NT, bug, flags, BlockSync{});
  __syncthreads();
  uint4* out = reinterpret_cast<uint4*>(&core[i]);
  for (int k = tid; k < (int)(sizeof(MapCore) / 16); k += NT) out[k] = dst[k];
  PROF_MARK(P_STORE);
}

// 3 단계(판마다 레인 16, 블록 = 판 8): 지도 토큰. 지도 갱신이 끝난 뒤 모든 판. 레인이 광선·선분·칸을 나눠 하고 반 워프 동기
constexpr int TOK_NL = 16, TOK_EPB = 8;
struct HalfSync { unsigned mask; __device__ void operator()() const { __syncwarp(mask); } };
__global__ void __launch_bounds__(TOK_NL * TOK_EPB) map_tok_kernel(int N, const MapCore* core, const uint32_t* occ, const int16_t* segs,
                                                                   TPrev* tprev, MapTok* out) {
  __shared__ TokScratch ts[TOK_EPB];
  const int sub = threadIdx.x / TOK_NL, lane = threadIdx.x % TOK_NL;
  const int i0 = blockIdx.x * TOK_EPB + sub;
  const bool live = i0 < N;
  const int i = live ? i0 : N - 1;   // 남는 레인도 같은 동기를 지나도록 마지막 판을 읽기만 함
  const HalfSync hs{0xffffu << (16 * (sub & 1))};
  PROF_START();
  make_tokens_n<TOK_NL>(core[i], occ + (size_t)i * NWORD, segs + (size_t)i * SEGW, tprev + (size_t)i * KSLOT, ts[sub], lane, TOK_NL, live, hs);
  hs();
  PROF_MARK(TK_ROOM);
  if (!live) return;
  const uint4* src = reinterpret_cast<const uint4*>(&ts[sub].out);
  uint4* dst = reinterpret_cast<uint4*>(out + i);
  for (int k = lane; k < (int)(sizeof(MapTok) / 16); k += TOK_NL) dst[k] = src[k];
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

DeviceMap::DeviceMap(int N, uint64_t seed) : N_(N) {
  CK(cudaMalloc(&core_, sizeof(MapCore) * (size_t)N));
  CK(cudaMalloc(&L_, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMalloc(&seen_, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMalloc(&met_, sizeof(float) * N_MET * (size_t)N));
  CK(cudaMalloc(&occ_, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMalloc(&segs_, sizeof(int16_t) * SEGW * (size_t)N));
  CK(cudaMalloc(&tprev_, sizeof(TPrev) * KSLOT * (size_t)N));
  CK(cudaMalloc(&tok_, sizeof(MapTok) * (size_t)N));
  CK(cudaMemset(occ_, 0, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMemset(segs_, 0, sizeof(int16_t) * SEGW * (size_t)N));
  CK(cudaMemset(tprev_, 0, sizeof(TPrev) * KSLOT * (size_t)N));
  CK(cudaMemset(tok_, 0, sizeof(MapTok) * (size_t)N));
  CK(cudaMalloc(&list_, sizeof(uint32_t) * (size_t)N));
  CK(cudaMalloc(&count_, sizeof(int)));
  CK(cudaMemset(L_, 0, sizeof(int16_t) * NCELL * (size_t)N));
  CK(cudaMemset(seen_, 0, sizeof(uint32_t) * NWORD * (size_t)N));
  CK(cudaMemset(met_, 0, sizeof(float) * N_MET * (size_t)N));
  map_init_kernel<<<(N + NT - 1) / NT, NT>>>(core_, N, seed);
  CK(cudaGetLastError());
}
DeviceMap::~DeviceMap() { cudaFree(core_); cudaFree(L_); cudaFree(seen_); cudaFree(met_); cudaFree(list_); cudaFree(count_);
                         cudaFree(occ_); cudaFree(segs_); cudaFree(tprev_); cudaFree(tok_); }

size_t DeviceMap::bytes() const {
  return (size_t)N_ * (sizeof(MapCore) + sizeof(int16_t) * NCELL + 2 * sizeof(uint32_t) * NWORD + sizeof(float) * N_MET + sizeof(int16_t) * SEGW +
                       sizeof(TPrev) * KSLOT + sizeof(MapTok) + sizeof(uint32_t));
}

void DeviceMap::step(const env::Soa& s, int force_kf, int bug, cudaStream_t st, MapTok* tok) {
  // 목록 길이를 0 으로(비동기, 그래프로 잡힘) → 시작 커널(판마다 스레드) → keyframe 커널(목록의 판만 일함)
  CK(cudaMemsetAsync(count_, 0, sizeof(int), st));
  map_begin_kernel<<<(N_ + BEGIN_NT - 1) / BEGIN_NT, BEGIN_NT, 0, st>>>(s, core_, met_, list_, count_, force_kf);
  map_kf_kernel<<<N_, NT, 0, st>>>(s, core_, L_, seen_, occ_, segs_, met_, list_, count_, bug);
  if (tok_on_) map_tok_kernel<<<(N_ + TOK_EPB - 1) / TOK_EPB, TOK_NL * TOK_EPB, 0, st>>>(N_, core_, occ_, segs_, tprev_, tok ? tok : tok_);
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

void DeviceMap::download(MapHost& h, const MapTok* tok) const {
  h.core.resize(N_); h.L.resize((size_t)NCELL * N_); h.seen.resize((size_t)NWORD * N_); h.met.resize((size_t)N_MET * N_);
  h.occ.resize((size_t)NWORD * N_); h.segs.resize((size_t)SEGW * N_); h.tprev.resize((size_t)KSLOT * N_); h.tok.resize(N_);
  CK(cudaMemcpy(h.occ.data(), occ_, sizeof(uint32_t) * h.occ.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.segs.data(), segs_, sizeof(int16_t) * h.segs.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.tprev.data(), tprev_, sizeof(TPrev) * h.tprev.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.tok.data(), tok ? tok : tok_, sizeof(MapTok) * h.tok.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.core.data(), core_, sizeof(MapCore) * h.core.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.L.data(), L_, sizeof(int16_t) * h.L.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.seen.data(), seen_, sizeof(uint32_t) * h.seen.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.met.data(), met_, sizeof(float) * h.met.size(), cudaMemcpyDeviceToHost));
}

TokenRecorder::TokenRecorder(int N, int T) : N_(N), T_(T) {
  CK(cudaMalloc(&buf_, bytes()));
  CK(cudaMemset(buf_, 0, bytes()));
}
TokenRecorder::~TokenRecorder() { cudaFree(buf_); }
void TokenRecorder::download(std::vector<MapTok>& out) const {
  out.resize((size_t)N_ * T_);
  CK(cudaMemcpy(out.data(), buf_, bytes(), cudaMemcpyDeviceToHost));
}

}  // namespace gmap
