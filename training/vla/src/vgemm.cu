// 끝단을 붙인 GEMM(RecallVLA 학습 융합). 본 계산은 RL network gemm2_k 와 같은 타일·같은 k 순서(같은 g2_load·ldmatrix·mma)라
// 출력 원소마다 FP32 누산 값이 gemm2_k 와 비트까지 같다. 다른 것은 끝단뿐(tkern.cuh EpiK). 공유 RL 헤더(gemm.cuh)는 고치지 않는다.
#include <cstdio>
#include <cstdlib>

#include "gemm.cuh"
#include "tkern.cuh"

namespace rvla {
namespace tk {

using namespace net;
#define KCK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

__device__ __forceinline__ float vsilu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float vsigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float vdsilu(float x) { const float s = vsigm(x); return s * (1.f + x * (1.f - s)); }
__device__ __forceinline__ float vgelu(float x) {
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + tanhf(u));
}

struct VgP {
  const uint16_t* A; long long lda;
  const uint16_t* B; long long ldb;
  int M, N, K;
  int il;   // > 0: B 의 논리 줄 n → 물리 줄 (n 짝수 ? n/2 : il + n/2) (SwiGLU 짝 끼우기, BT = false 만)
  Epi e;
};

// B 타일 읽기(g2_load 와 같은 자리·같은 0 채움, 줄 번호만 바꿀 수 있음)
template <int R, bool T, int NT>
__device__ __forceinline__ void vg_loadB(uint16_t* S, const uint16_t* P, long long ld, int r0, int rlim, int k0, int klim, int tid, int il) {
  using C = G2Tile<R, T>;
#pragma unroll
  for (int q0 = 0; q0 < C::CHUNKS; q0 += NT) {
    const int q = q0 + tid;
    if (C::CHUNKS % NT != 0 && q >= C::CHUNKS) break;
    if (!T) {
      const int row = q >> 2, c = q & 3, gr = r0 + row, gk = k0 + c * 8;
      const bool ok = gr < rlim && gk < klim;
      const int pr = il > 0 ? ((gr & 1) ? il + (gr >> 1) : (gr >> 1)) : gr;
      cp16(S + row * C::LD + c * 8, ok ? P + (long long)pr * ld + gk : P, ok);
    } else {
      constexpr int MC = R / 8;
      const int mc = q % MC, kr = q / MC, gr = r0 + mc * 8, gk = k0 + kr;
      const bool ok = gr < rlim && gk < klim;
      cp16(S + kr * C::LD + mc * 8, ok ? P + (long long)gk * ld + gr : P, ok);
    }
  }
}

template <int KIND>
__device__ __forceinline__ void vg_epi(const Epi& e, int r, int c, float x0, float x1) {
  if (KIND == EK_RES) {   // C = R + acc (+ b): 예전 복사 → += → bias_add 와 같은 더하기 순서
    const float2 rr = *reinterpret_cast<const float2*>(e.R + (long long)r * e.ldr + c);
    float y0 = rr.x + x0, y1 = rr.y + x1;
    if (e.bias) { y0 = y0 + e.bias[c]; y1 = y1 + e.bias[c + 1]; }
    *reinterpret_cast<float2*>(e.C + (long long)r * e.ldc + c) = make_float2(y0, y1);
  } else if (KIND == EK_BIAS) {   // C = acc + b, Cb = bf16(gelu(C)) (선택)
    float y0 = x0, y1 = x1;
    if (e.bias) { y0 = y0 + e.bias[c]; y1 = y1 + e.bias[c + 1]; }
    if (e.C) *reinterpret_cast<float2*>(e.C + (long long)r * e.ldc + c) = make_float2(y0, y1);
    if (e.Cb) {
      if (e.gelu) { y0 = vgelu(y0); y1 = vgelu(y1); }
      reinterpret_cast<uint32_t*>(e.Cb)[((long long)r * e.ldcb + c) >> 1] = (uint32_t)net::f2bf(y0) | ((uint32_t)net::f2bf(y1) << 16);
    }
  } else if (KIND == EK_SWI) {   // 짝(c, c+1) = (gate_j, up_j), j = c/2 → Hh bf16, GU F32 끼운 꼴(선택)
    if (e.C) *reinterpret_cast<float2*>(e.C + (long long)r * e.ldc + c) = make_float2(x0, x1);
    e.Cb[(long long)r * e.ldcb + (c >> 1)] = net::f2bf(vsilu(x0) * x1);
  } else if (KIND == EK_DSWI) {   // acc = dHh[r][c..c+1], GU 끼운 꼴 → dGU bf16 원래 꼴([gate I | up I])
    const float4 gu = *reinterpret_cast<const float4*>(e.GU + (long long)r * e.ldgu + 2 * c);
    const float xs[2] = {x0, x1}, gs[2] = {gu.x, gu.z}, us[2] = {gu.y, gu.w};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      e.Cb[(long long)r * e.ldcb + c + j] = net::f2bf(xs[j] * us[j] * (e.bug == 1 ? 1.f : vdsilu(gs[j])));
      e.Cb[(long long)r * e.ldcb + e.I + c + j] = net::f2bf(xs[j] * vsilu(gs[j]));
    }
  } else {   // EK_DW: G bf16 = acc 또는 gacc += acc(예전 dwred 의 조각 하나와 같음: 0 + acc)
    const float y0 = 0.f + x0, y1 = 0.f + x1;
    if (e.gacc) {
      float2* d = reinterpret_cast<float2*>(e.gacc + (long long)r * e.ldc + c);
      float2 o = *d;
      o.x = o.x + y0; o.y = o.y + y1;
      *d = o;
    } else reinterpret_cast<uint32_t*>(e.Cb)[((long long)r * e.ldcb + c) >> 1] = (uint32_t)net::f2bf(y0) | ((uint32_t)net::f2bf(y1) << 16);
  }
}

template <bool AT, bool BT, int KIND, int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) vg_k(const __grid_constant__ VgP p) {
  constexpr int NWM = BM / WM, NT = NWM * (BN / WN) * 32, MI = WM / 16, NI = WN / 8;
  using CA = G2Tile<BM, AT>;
  using CB = G2Tile<BN, BT>;
  extern __shared__ __align__(128) uint16_t vgs[];
  uint16_t (*sA)[CA::ELEMS] = reinterpret_cast<uint16_t (*)[CA::ELEMS]>(vgs);
  uint16_t (*sB)[CB::ELEMS] = reinterpret_cast<uint16_t (*)[CB::ELEMS]>(vgs + 2 * CA::ELEMS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kbeg = 0, kend = p.K;
  const int wm = (warp % NWM) * WM, wn = (warp / NWM) * WN;
  float acc[MI][NI][4];
#pragma unroll
  for (int a = 0; a < MI; ++a)
#pragma unroll
    for (int b = 0; b < NI; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  g2_load<BM, AT, NT>(sA[0], p.A, p.lda, m0, p.M, kbeg, kend, tid);
  vg_loadB<BN, BT, NT>(sB[0], p.B, p.ldb, n0, p.N, kbeg, kend, tid, p.il);
  cp_commit();
  int buf = 0;
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  for (int k0 = kbeg; k0 < kend; k0 += G2K) {
    if (k0 + G2K < kend) {
      g2_load<BM, AT, NT>(sA[buf ^ 1], p.A, p.lda, m0, p.M, k0 + G2K, kend, tid);
      vg_loadB<BN, BT, NT>(sB[buf ^ 1], p.B, p.ldb, n0, p.N, k0 + G2K, kend, tid, p.il);
    }
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    const uint16_t* A = sA[buf];
    const uint16_t* B = sB[buf];
#pragma unroll
    for (int kk = 0; kk < G2K; kk += 16) {
      uint32_t af[MI][4], bfr[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        if (!AT) ldsm4(af[mi], A + (wm + mi * 16 + lr + j0 * 8) * CA::LD + kk + j1 * 8);
        else ldsm4t(af[mi], A + (kk + lr + j1 * 8) * CA::LD + wm + mi * 16 + j0 * 8);
      }
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) {
        uint32_t r[4];
        if (!BT) ldsm4(r, B + (wn + nj * 16 + lr + j1 * 8) * CB::LD + kk + j0 * 8);
        else ldsm4t(r, B + (kk + lr + j0 * 8) * CB::LD + wn + nj * 16 + j1 * 8);
        bfr[2 * nj][0] = r[0]; bfr[2 * nj][1] = r[1]; bfr[2 * nj + 1][0] = r[2]; bfr[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], af[mi], bfr[ni]);
    }
    __syncthreads();
    buf ^= 1;
  }
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m0 + wm + mi * 16 + g + h * 8, c = n0 + wn + ni * 8 + 2 * t4;
        if (r >= p.M || c >= p.N) continue;
        vg_epi<KIND>(p.e, r, c, acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);
      }
}

template <bool AT, bool BT, int KIND, int BM, int BN, int WM, int WN>
static void vg2(const VgP& p, cudaStream_t st) {
  constexpr int smem = 2 * 2 * (G2Tile<BM, AT>::ELEMS + G2Tile<BN, BT>::ELEMS);
  if (smem > 48 * 1024) {
    static const bool ok = [] { cudaFuncSetAttribute(vg_k<AT, BT, KIND, BM, BN, WM, WN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); return true; }();
    (void)ok;
  }
  vg_k<AT, BT, KIND, BM, BN, WM, WN><<<dim3((p.N + BN - 1) / BN, (p.M + BM - 1) / BM, 1), (BM / WM) * (BN / WN) * 32, smem, st>>>(p);
}
// 타일 고르기는 qk::gl 과 같음
template <bool AT, bool BT, int KIND>
static void vgl(const VgP& p, cudaStream_t st) {
  if (p.N <= 16) vg2<AT, BT, KIND, 128, 16, 32, 16>(p, st);
  else if (p.M <= 64) vg2<AT, BT, KIND, 64, 64, 32, 32>(p, st);
  else if (p.N >= 128) vg2<AT, BT, KIND, 128, 128, 64, 32>(p, st);
  else vg2<AT, BT, KIND, 128, 64, 32, 32>(p, st);
  KCK();
}
static void chk(bool ok, const char* w) {
  if (!ok) { std::fprintf(stderr, "vgemm: %s\n", w); std::abort(); }
}

void mme(const uint16_t* A, int lda, int M, const uint16_t* W, int N, int K, const Epi& e, cudaStream_t st) {
  if (M == 0) return;
  chk(lda % 8 == 0 && K % 8 == 0 && N % 2 == 0, "mme shape");
  VgP p{};
  p.A = A; p.lda = lda; p.B = W; p.ldb = K; p.M = M; p.N = N; p.K = K; p.e = e;
  switch (e.kind) {
    case EK_RES: vgl<false, false, EK_RES>(p, st); break;
    case EK_BIAS: vgl<false, false, EK_BIAS>(p, st); break;
    case EK_SWI: chk(N == 2 * e.I && e.I % 8 == 0, "swiglu shape"); p.il = e.I; vgl<false, false, EK_SWI>(p, st); break;
    default: chk(false, "mme kind");
  }
}
void mme_dx(const uint16_t* dZ, int ldz, int M, const uint16_t* W, int N, int K, const Epi& e, cudaStream_t st) {
  if (M == 0) return;
  chk(ldz % 8 == 0 && K % 8 == 0 && N % 8 == 0, "mme_dx shape");
  VgP p{};
  p.A = dZ; p.lda = ldz; p.B = W; p.ldb = K; p.M = M; p.N = K; p.K = N; p.e = e;
  switch (e.kind) {
    case EK_DSWI: chk(K == e.I, "dswiglu shape"); vgl<false, true, EK_DSWI>(p, st); break;
    case EK_RES: vgl<false, true, EK_RES>(p, st); break;
    default: chk(false, "mme_dx kind");
  }
}
void mme_dw1(const uint16_t* dZ, int ldz, const uint16_t* X, int ldx, int M, int N, int K, uint16_t* G, float* gacc, cudaStream_t st) {
  VgP p{};
  p.A = dZ; p.lda = ldz; p.B = X; p.ldb = ldx; p.M = N; p.N = K; p.K = M;
  p.e.kind = EK_DW; p.e.Cb = G; p.e.ldcb = K; p.e.gacc = gacc; p.e.ldc = K;
  vgl<true, true, EK_DW>(p, st);
}

}  // namespace tk
}  // namespace rvla
