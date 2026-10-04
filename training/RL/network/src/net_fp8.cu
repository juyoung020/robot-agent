// 학습 신경망 FP8 GEMM 실행(G6, 계획서 7.1·7.2·9.1). sm_120a 로만 빌드(이 파일 하나 — gemm_fp8.cuh 의 mma kind::f8f6f4).
// net_kernels.cu 의 GEMM 실행 함수가 층(LayerDesc::fp8)과 켬 표(set_fp8)를 보고 여기로 넘긴다.
//   앞    : E4M3 X × E4M3 W        (AT 0, BT 0)  끝단 그대로(ELU → bf16)
//   dgrad : E5M2 dZ × E4M3 W       (AT 0, BT 1)  W 는 8 비트 ldmatrix 전치로 읽음(전치 사본 없음)
//   wgrad : E5M2 dZᵀ × E4M3 X      (AT 1, BT 1)  split-K 조각 → 기존 dw_reduce
// 배율: 텐서마다, GEMM 바로 앞에서 그 피연산자의 amax 를 장치에서 잼(지금 값 배율, "current scaling") → 2 의 거듭제곱.
//   같은 스트림 차례: memset(칸 0) → amax 커널(정수 원자 최대, 순서 무관 = 결정적) → gemm8_k. 호스트 동기 없음, 그래프에 그대로 잡힌다.
//   칸은 실행마다 돌려 씀(같은 스트림 안 차례라 겹치지 않음).
#include <cstdio>
#include <cstdlib>

#include "gemm_fp8.cuh"
#include "net_ops.h"

namespace net {

#define F8CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

static int g_fp8 = 0;
void set_fp8(int mask) { g_fp8 = mask; }
int fp8_mask() { return g_fp8; }

constexpr int N_SLOT = 64;
__device__ float g_f8_amax[N_SLOT * 4];
static float* slot_base() {
  static float* p = [] {
    void* q = nullptr;
    F8CK(cudaGetSymbolAddress(&q, g_f8_amax));
    return reinterpret_cast<float*>(q);
  }();
  return p;
}

struct AmaxJob { const uint16_t* p; long long rows; int cols; long long ld; float* out; };
struct AmaxJobs { AmaxJob j[4]; };
__global__ void amax4_k(AmaxJobs J) {
  const AmaxJob a = J.j[blockIdx.y];
  float m = 0.f;
  const long long n8 = a.rows * (a.cols / 8);
  for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < n8; q += (long long)gridDim.x * blockDim.x) {
    const long long r = q / (a.cols / 8);
    const int c = (int)(q % (a.cols / 8)) * 8;
    const uint4 v = *reinterpret_cast<const uint4*>(a.p + r * a.ld + c);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      m = fmaxf(m, fabsf(bf2f((uint16_t)(w[e] & 0xffffu))));
      m = fmaxf(m, fabsf(bf2f((uint16_t)(w[e] >> 16))));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0 && m > 0.f) atomicMax(reinterpret_cast<int*>(a.out), __float_as_int(m));
}

template <int FA, int FB, bool AT, bool BT, int EPI, int BM, int BN, int WM, int WN>
static void l8(const f8::G8& g, int np, cudaStream_t st) {
  const GemmP& p = g.p[0];
  f8::gemm8_k<FA, FB, AT, BT, EPI, BM, BN, WM, WN><<<dim3((p.N + BN - 1) / BN, (p.M + BM - 1) / BM, g.zper * np), (BM / WM) * (BN / WN) * 32, 0, st>>>(g);
}
template <int FA, int FB, bool AT, bool BT, int EPI>
static void shape8(const f8::G8& g, int np, cudaStream_t st) {
  if (g.p[0].N >= 128) l8<FA, FB, AT, BT, EPI, 128, 128, 64, 32>(g, np, st);
  else l8<FA, FB, AT, BT, EPI, 128, 64, 32, 32>(g, np, st);
}

// 논리 피연산자(행 R × K)의 저장 영역: T 면 [K][R], 아니면 [R][K]
static AmaxJob job(const uint16_t* p, long long ld, int R, int K, bool T, float* out) {
  return T ? AmaxJob{p, (long long)K, R, ld, out} : AmaxJob{p, (long long)R, K, ld, out};
}

bool fp8_gemm(int role, const GemmP* ps, int np, int gz, int epi, cudaStream_t st) {
  static int rot = 0;
  float* s = slot_base() + 4 * (rot++ % N_SLOT);
  const bool AT = role == 2, BT = role >= 1;
  AmaxJobs J{};
  for (int k = 0; k < np; ++k) {
    const GemmP& p = ps[k];
    if ((AT ? p.M : p.K) % 8 || (BT ? p.N : p.K) % 8) return false;   // 16 B 덩이로 읽는 방향
    J.j[2 * k] = job(p.A, p.lda, p.M, p.K, AT, s + 2 * k);
    J.j[2 * k + 1] = job(p.B, p.ldb, p.N, p.K, BT, s + 2 * k + 1);
  }
  F8CK(cudaMemsetAsync(s, 0, sizeof(float) * 4, st));
  amax4_k<<<dim3(60, 2 * np), 256, 0, st>>>(J);
  f8::G8 g{};
  g.p[0] = ps[0];
  g.p[1] = ps[np - 1];
  g.amA[0] = s; g.amB[0] = s + 1;
  g.amA[1] = s + 2 * (np - 1); g.amB[1] = s + 2 * (np - 1) + 1;
  g.zper = gz;
  using namespace f8;
  if (role == 0) {
    if (epi == EPI_ACT_BF16) shape8<E4M3, E4M3, false, false, EPI_ACT_BF16>(g, np, st);
    else if (epi == EPI_F32) shape8<E4M3, E4M3, false, false, EPI_F32>(g, np, st);
    else return false;
  } else if (role == 1) {
    if (epi == EPI_DACT_BF16) shape8<E5M2, E4M3, false, true, EPI_DACT_BF16>(g, np, st);
    else if (epi == EPI_F32) shape8<E5M2, E4M3, false, true, EPI_F32>(g, np, st);
    else if (epi == EPI_ACC_F32) shape8<E5M2, E4M3, false, true, EPI_ACC_F32>(g, np, st);
    else return false;
  } else {
    if (epi == EPI_SPLIT_F32) shape8<E5M2, E4M3, true, true, EPI_SPLIT_F32>(g, np, st);
    else return false;
  }
  F8CK(cudaGetLastError());
  return true;
}

}  // namespace net
