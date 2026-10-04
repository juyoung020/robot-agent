// FP8 손 GEMM(G6, 계획서 GPU_TRAINING.md 2.2·7.2·7.3): `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.<A>.<B>.f32`, FP32 누산.
// sm_120a 전용 명령이다(PTX 8.7 / CUDA 12.8). 이 헤더를 쓰는 파일은 `-gencode arch=compute_120a,code=sm_120a` 로 빌드한다.
// sm_120(접미사 없음)으로 빌드하면 컴파일은 되지만 FP8 커널은 __trap() 한다(ptxas 가 kind::f8f6f4 를 거부하므로).
//
// 잰 사실(12절): e5m2 × e4m3 섞기는 한 명령으로 된다(아래 mma8<E5M2,E4M3>, V3 에서 수치 확인).
// 8 비트 ldmatrix 전치 `ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` 는 sm_120a 에 있다(SASS LDSM.8.MT1616, sm_120 은 ptxas 거부).
// 조각 배치(잰 값): 메모리 [k][행] 16×16 바이트를 읽으면 레인 (g = lane/4, t = lane%4) 의 r0 = (행 g, k 4t..4t+3), r1 = (행 g+8, k 4t..4t+3)
//   → m16n8k32 의 A 조각 a0/a1(k 0–15), x2 의 둘째 행렬(k 16–31)이 a2/a3. 그래서 M 연속(전치) 피연산자도 전치 사본 없이 읽는다.
// K 연속 피연산자는 8 비트 값 둘을 16 비트 하나로 보면 bf16 판과 같은 ldmatrix(b16)·같은 주소로 조각이 그대로 맞는다(행 g, 바이트 4t..4t+3).
//
// 두 가지 커널
//  (1) tn_k   : A[M][K]·B[N][K]ᵀ, 둘 다 미리 FP8(바이트, K 연속) 또는 16 비트. cp.async 다단 + ldmatrix(b16) — 얼린 인코더 앞 계산(배율은 끝단 함수가 곱함)
//  (2) gemm8_k: 학습 신경망용. 입력은 bf16 그대로(net::GemmP), 공유 메모리에 쓸 때 텐서별 배율로 FP8 로 바꿈("읽으며 양자화").
//      네 배치(AT/BT) 모두. 전치 피연산자는 8 비트 ldmatrix 전치. 끝단은 gemm2_k 와 같은 식에 deq = sA·sB 를 먼저 곱함.
//      배율 = 2 의 거듭제곱(fmax/amax 아래로), amax 는 같은 그래프 안 앞 커널(fp8_amax_k)이 장치 값으로 — 호스트 동기 없음(7.2 의 1).
#pragma once
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "gemm.cuh"

namespace f8 {

enum Fmt : int { E4M3 = 0, E5M2 = 1 };
__host__ __device__ constexpr float fmax_of(int f) { return f == E4M3 ? 448.f : 57344.f; }

template <int F>
__device__ __forceinline__ uint32_t q8(float x) {
  return (uint32_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, F == E4M3 ? __NV_E4M3 : __NV_E5M2);
}
template <int F>
__device__ __forceinline__ float dq8(uint32_t b) {
  const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, F == E4M3 ? __NV_E4M3 : __NV_E5M2);
  return __half2float(__half(h));
}
// 2 의 거듭제곱 배율: inv = 2^floor(log2(fmax / amax)) (amax 0 이면 1). 값·배율 곱이 정확(정규 범위)하고 되돌림 deq = 1/inv 도 정확
__host__ __device__ inline float pow2_inv(float amax, int f) {
  if (!(amax > 0.f)) return 1.f;
  int e;
  (void)frexpf(fmax_of(f) / amax, &e);   // fmax/amax = m·2^e, m ∈ [0.5, 1) → floor(log2) = e − 1
  return ldexpf(1.f, e - 1);
}

#define F8_MMA_ASM(TA, TB)                                                                                                         \
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32." TA "." TB ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                                                    \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
template <int FA, int FB>
__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  if constexpr (FA == E4M3 && FB == E4M3) F8_MMA_ASM("e4m3", "e4m3");
  else if constexpr (FA == E5M2 && FB == E4M3) F8_MMA_ASM("e5m2", "e4m3");
  else if constexpr (FA == E4M3 && FB == E5M2) F8_MMA_ASM("e4m3", "e5m2");
  else F8_MMA_ASM("e5m2", "e5m2");
#else
  (void)d; (void)a; (void)b;
  __trap();
#endif
}
#undef F8_MMA_ASM

// 8 비트 ldmatrix 전치, 16×16 바이트 행렬 둘: 레인 0–15 = 첫 행렬의 줄 주소, 16–31 = 둘째
__device__ __forceinline__ void ldsm8t2(uint32_t (&r)[4], const uint8_t* p) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
  (void)r; (void)p;
  __trap();
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// (1) TN: C = Σ_k A[m][k]·B[n][k] → epi(r, c, acc0, acc1) (열 c, c+1 의 두 값, 배율 곱은 epi 가)
//     KIND: TN_E4E4 / TN_E5E4 (FP8 바이트, 미리 양자화) 또는 TN_F16 / TN_BF16 (16 비트) — 같은 타일·같은 ldmatrix·같은 파이프라인이고 mma 만 다르다
//     (16 비트 낱말 단위로 보면 FP8 은 낱말 하나에 값 둘: kk 16 낱말 = FP8 k32 mma 하나 = 16 비트 k16 mma 하나). 그래서 속도 비교가 공정하다.
//     A16·B16 = 16 비트 낱말 포인터, K16·lda16·ldb16 = 낱말 단위(FP8 이면 바이트/2). K16 은 8 의 배수, 16 B 정렬. 타일 k = 32 낱말, NST 단 cp.async.
// TN_F16H: FP16 피연산자 + **FP16 누산**(GeForce 에서 FP32 누산의 2 배 속도, 백서 175.8 TFLOPS) — 타일(k 32)마다 FP16 부분합을 FP32 누산기로 옮김(잃음을 k 32 안으로 묶음)
enum TnKind : int { TN_E4E4 = 0, TN_E5E4 = 1, TN_F16 = 2, TN_BF16 = 3, TN_F16H = 4 };
template <int BM, int BN>
constexpr int tn_smem(int nst) { return nst * 2 * (net::G2Tile<BM, false>::ELEMS + net::G2Tile<BN, false>::ELEMS); }
template <int KIND>
__device__ __forceinline__ void tn_mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  if constexpr (KIND == TN_E4E4) mma8<E4M3, E4M3>(d, a, b);
  else if constexpr (KIND == TN_E5E4) mma8<E5M2, E4M3>(d, a, b);
  else if constexpr (KIND == TN_F16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else net::mma_bf16(d, a, b);
}

#ifndef TN_HPROMO
#define TN_HPROMO 1
#endif
constexpr int HPROMO = TN_HPROMO;   // TN_F16H 의 FP32 로 옮기는 간격(타일 수, 타일 = k 32)
template <int KIND, int BM, int BN, int WM, int WN, int NST, class Epi>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) tn_k(const uint16_t* __restrict__ A16, int lda16, const uint16_t* __restrict__ B16, int ldb16,
                                                                   int M, int N, int K16, const __grid_constant__ Epi epi) {
  constexpr int NWM = BM / WM, NT = NWM * (BN / WN) * 32, MI = WM / 16, NI = WN / 8;
  using CA = net::G2Tile<BM, false>;
  using CB = net::G2Tile<BN, false>;
  extern __shared__ __align__(128) uint16_t f8s[];
  uint16_t(*sA)[CA::ELEMS] = reinterpret_cast<uint16_t(*)[CA::ELEMS]>(f8s);
  uint16_t(*sB)[CB::ELEMS] = reinterpret_cast<uint16_t(*)[CB::ELEMS]>(f8s + NST * CA::ELEMS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int wm = (warp % NWM) * WM, wn = (warp / NWM) * WN;
  float acc[MI][NI][4];
#pragma unroll
  for (int a = 0; a < MI; ++a)
#pragma unroll
    for (int b = 0; b < NI; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const int nk = (K16 + net::G2K - 1) / net::G2K;
  uint32_t hacc[KIND == TN_F16H ? MI : 1][KIND == TN_F16H ? NI : 1][2];   // TN_F16H: FP16 부분합(HPROMO 타일마다 FP32 로)
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) {
      net::g2_load<BM, false, NT>(sA[s], A16, lda16, m0, M, s * net::G2K, K16, tid);
      net::g2_load<BN, false, NT>(sB[s], B16, ldb16, n0, N, s * net::G2K, K16, tid);
    }
    net::cp_commit();
  }
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  for (int kt = 0; kt < nk; ++kt) {
    const int nx = kt + NST - 1;
    if (nx < nk) {
      net::g2_load<BM, false, NT>(sA[nx % NST], A16, lda16, m0, M, nx * net::G2K, K16, tid);
      net::g2_load<BN, false, NT>(sB[nx % NST], B16, ldb16, n0, N, nx * net::G2K, K16, tid);
    }
    net::cp_commit();
    net::cp_wait<NST - 1>();
    __syncthreads();
    const uint16_t* As = sA[kt % NST];
    const uint16_t* Bs = sB[kt % NST];
    if constexpr (KIND == TN_F16H) if (kt % HPROMO == 0) {
#pragma unroll
      for (int a = 0; a < MI; ++a)
#pragma unroll
        for (int b = 0; b < NI; ++b) hacc[a][b][0] = hacc[a][b][1] = 0u;
    }
#pragma unroll
    for (int kk = 0; kk < net::G2K; kk += 16) {
      uint32_t af[MI][4], bfr[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) net::ldsm4(af[mi], As + (wm + mi * 16 + lr + j0 * 8) * CA::LD + kk + j1 * 8);
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) {
        uint32_t r[4];
        net::ldsm4(r, Bs + (wn + nj * 16 + lr + j1 * 8) * CB::LD + kk + j0 * 8);
        bfr[2 * nj][0] = r[0]; bfr[2 * nj][1] = r[1]; bfr[2 * nj + 1][0] = r[2]; bfr[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          if constexpr (KIND == TN_F16H)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                         : "+r"(hacc[mi][ni][0]), "+r"(hacc[mi][ni][1])
                         : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]), "r"(bfr[ni][0]), "r"(bfr[ni][1]));
          else tn_mma<KIND>(acc[mi][ni], af[mi], bfr[ni]);
        }
    }
    if constexpr (KIND == TN_F16H) if (kt % HPROMO == HPROMO - 1 || kt == nk - 1) {
#pragma unroll
      for (int a = 0; a < MI; ++a)
#pragma unroll
        for (int b = 0; b < NI; ++b)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hacc[a][b][h]));
            acc[a][b][2 * h] += f.x;
            acc[a][b][2 * h + 1] += f.y;
          }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m0 + wm + mi * 16 + g + h * 8, c = n0 + wn + ni * 8 + 2 * t4;
        if (r >= M || c >= N) continue;
        epi(r, c, acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);
      }
}

// ---------------------------------------------------------------------------------------------------------------------
// 학습 신경망 FP8 GEMM 문제(gemm2_k 의 GemmG 에 피연산자 배율 칸을 더함)
struct G8 {
  net::GemmP p[2];
  const float* amA[2];   // A 의 amax(장치)
  const float* amB[2];
  int zper;
  int bug;               // 음성 대조(V3 전용): 1 = 양자화를 0 쪽 자르기로, 2 = B 조각의 k 반쪽 둘을 바꿈, 3 = 되돌림에서 A 배율 빠뜨림
};
constexpr int G8K = 64;   // 타일 k(원소)
template <int R, bool T>
struct G8Tile {
  static constexpr int LD = T ? R + 16 : G8K + 16;   // 줄 간격(바이트): [행][k] 80 B, [k][행] R+16 — ldmatrix 줄이 서로 다른 은행 덩이
  static constexpr int BYTES = (T ? G8K : R) * LD;
  static constexpr int CHUNKS = R * G8K / 8;         // 덩이 = bf16 8 개(16 B 읽기) → FP8 8 바이트 쓰기
};
template <int R, bool T, int NT>
struct G8Ld {
  static constexpr int NJ = (G8Tile<R, T>::CHUNKS + NT - 1) / NT;
  uint4 v[NJ];
  __device__ __forceinline__ void load(const uint16_t* P, long long ld, int r0, int rlim, int k0, int klim, int tid) {
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int q = tid + j * NT;
      v[j] = make_uint4(0, 0, 0, 0);
      if (G8Tile<R, T>::CHUNKS % NT != 0 && q >= G8Tile<R, T>::CHUNKS) continue;
      if (!T) {
        const int row = q >> 3, c = q & 7, gr = r0 + row, gk = k0 + c * 8;
        if (gr < rlim && gk < klim) v[j] = *reinterpret_cast<const uint4*>(P + (long long)gr * ld + gk);
      } else {
        constexpr int MC = R / 8;
        const int mc = q % MC, kr = q / MC, gr = r0 + mc * 8, gk = k0 + kr;
        if (gr < rlim && gk < klim) v[j] = *reinterpret_cast<const uint4*>(P + (long long)gk * ld + gr);
      }
    }
  }
  template <int F>
  __device__ __forceinline__ void store(uint8_t* S, float inv, int tid, int bug) const {
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int q = tid + j * NT;
      if (G8Tile<R, T>::CHUNKS % NT != 0 && q >= G8Tile<R, T>::CHUNKS) continue;
      const uint32_t w[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
      uint32_t o[2] = {0u, 0u};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float x = net::bf2f((uint16_t)(w[e >> 1] >> ((e & 1) * 16))) * inv;
        if (bug == 1) x = __uint_as_float(__float_as_uint(x) & (F == E4M3 ? 0xfff00000u : 0xffe00000u));   // 음성 대조: 0 쪽 자르기
        o[e >> 2] |= q8<F>(x) << ((e & 3) * 8);
      }
      uint8_t* d;
      if (!T) {
        const int row = q >> 3, c = q & 7;
        d = S + row * G8Tile<R, T>::LD + c * 8;
      } else {
        constexpr int MC = R / 8;
        const int mc = q % MC, kr = q / MC;
        d = S + kr * G8Tile<R, T>::LD + mc * 8;
      }
      *reinterpret_cast<uint2*>(d) = make_uint2(o[0], o[1]);
    }
  }
};

template <int FA, int FB, bool AT, bool BT, int EPI, int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) gemm8_k(const __grid_constant__ G8 gg) {
  const int prob = blockIdx.z / gg.zper;
  const net::GemmP& p = gg.p[prob];
  const int zs = blockIdx.z % gg.zper;
  constexpr int NWM = BM / WM, NT = NWM * (BN / WN) * 32, MI = WM / 16, NI = WN / 8;
  static_assert(NI % 2 == 0 && WM % 16 == 0, "warp tile");
  using CA = G8Tile<BM, AT>;
  using CB = G8Tile<BN, BT>;
  __shared__ __align__(128) uint8_t sA[2][CA::BYTES];
  __shared__ __align__(128) uint8_t sB[2][CB::BYTES];
  const float aA = *gg.amA[prob], aB = *gg.amB[prob];
  const float invA = pow2_inv(aA, FA), invB = pow2_inv(aB, FB);
  const float deq = gg.bug == 3 ? 1.f / invB : (1.f / invA) * (1.f / invB);   // bug 3 = 음성 대조(A 배율 빠뜨림)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int kbeg = 0, kend = p.K;
  if (EPI == net::EPI_SPLIT_F32) {
    kbeg = zs * p.kchunk;
    kend = min(p.K, kbeg + p.kchunk);
  }
  const int wm = (warp % NWM) * WM, wn = (warp / NWM) * WN;
  float acc[MI][NI][4];
#pragma unroll
  for (int a = 0; a < MI; ++a)
#pragma unroll
    for (int b = 0; b < NI; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  G8Ld<BM, AT, NT> la;
  G8Ld<BN, BT, NT> lb;
  la.load(p.A, p.lda, m0, p.M, kbeg, kend, tid);
  lb.load(p.B, p.ldb, n0, p.N, kbeg, kend, tid);
  la.template store<FA>(sA[0], invA, tid, gg.bug);
  lb.template store<FB>(sB[0], invB, tid, gg.bug);
  __syncthreads();
  int buf = 0;
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4, l16 = lane & 15, h16 = lane >> 4;
  for (int k0 = kbeg; k0 < kend; k0 += G8K) {
    const bool more = k0 + G8K < kend;
    if (more) {
      la.load(p.A, p.lda, m0, p.M, k0 + G8K, kend, tid);
      lb.load(p.B, p.ldb, n0, p.N, k0 + G8K, kend, tid);
    }
    const uint8_t* As = sA[buf];
    const uint8_t* Bs = sB[buf];
#pragma unroll
    for (int kk = 0; kk < G8K; kk += 32) {
      uint32_t af[MI][4], bfr[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        if (!AT) net::ldsm4(af[mi], reinterpret_cast<const uint16_t*>(As + (wm + mi * 16 + lr + j0 * 8) * CA::LD + kk + j1 * 16));
        else ldsm8t2(af[mi], As + (kk + h16 * 16 + l16) * CA::LD + wm + mi * 16);
      }
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) {
        uint32_t r[4];
        if (!BT) {
          net::ldsm4(r, reinterpret_cast<const uint16_t*>(Bs + (wn + nj * 16 + lr + j1 * 8) * CB::LD + kk + j0 * 16));
          bfr[2 * nj][0] = r[0]; bfr[2 * nj][1] = r[1]; bfr[2 * nj + 1][0] = r[2]; bfr[2 * nj + 1][1] = r[3];
        } else {
          ldsm8t2(r, Bs + (kk + h16 * 16 + l16) * CB::LD + wn + nj * 16);
          bfr[2 * nj][0] = r[0]; bfr[2 * nj][1] = r[2]; bfr[2 * nj + 1][0] = r[1]; bfr[2 * nj + 1][1] = r[3];
        }
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          if (gg.bug == 2) { const uint32_t bb[2] = {bfr[ni][1], bfr[ni][0]}; mma8<FA, FB>(acc[mi][ni], af[mi], bb); }
          else mma8<FA, FB>(acc[mi][ni], af[mi], bfr[ni]);
        }
    }
    if (more) {
      la.template store<FA>(sA[buf ^ 1], invA, tid, gg.bug);
      lb.template store<FB>(sB[buf ^ 1], invB, tid, gg.bug);
    }
    __syncthreads();
    buf ^= 1;
  }
  // 끝단(gemm2_k 와 같은 식, 누산에 deq 를 먼저 곱함)
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m0 + wm + mi * 16 + g + h * 8, c = n0 + wn + ni * 8 + 2 * t4;
        if (r >= p.M || c >= p.N) continue;
        float x0 = acc[mi][ni][2 * h] * deq, x1 = acc[mi][ni][2 * h + 1] * deq;
        if (EPI == net::EPI_ACT_BF16) {
          if (p.act == net::ACT_ELU) { x0 = net::elu(x0); x1 = net::elu(x1); }
          reinterpret_cast<uint32_t*>(p.C)[((long long)r * p.ldc + c) >> 1] = (uint32_t)net::f2bf(x0) | ((uint32_t)net::f2bf(x1) << 16);
        } else if (EPI == net::EPI_F32) {
          *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + (long long)r * p.ldc + c) = make_float2(x0, x1);
        } else if (EPI == net::EPI_ACC_F32) {
          float2* d = reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + (long long)r * p.ldc + c);
          float2 o = *d;
          o.x = o.x + x0;
          o.y = o.y + x1;
          *d = o;
        } else if (EPI == net::EPI_DACT_BF16) {
          const uint32_t yy = reinterpret_cast<const uint32_t*>(p.Y)[((long long)r * p.ldy + c) >> 1];
          if (!p.bug) {
            x0 = x0 * net::elu_grad_from_y(net::bf2f((uint16_t)(yy & 0xffffu)));
            x1 = x1 * net::elu_grad_from_y(net::bf2f((uint16_t)(yy >> 16)));
          }
          reinterpret_cast<uint32_t*>(p.C)[((long long)r * p.ldc + c) >> 1] = (uint32_t)net::f2bf(x0) | ((uint32_t)net::f2bf(x1) << 16);
        } else {   // EPI_SPLIT_F32
          float* C = reinterpret_cast<float*>(p.C) + (long long)zs * p.M * p.N;
          *reinterpret_cast<float2*>(C + (long long)r * p.N + c) = make_float2(x0, x1);
        }
      }
}

// amax 커널 본문(헤더에 둠: 쓰는 번역 단위마다 하나 — inline 이 아니므로 static)
static __global__ void amax_bf16_impl_k(const uint16_t* __restrict__ X, long long rows, int cols, long long ld, float* slot) {
  float m = 0.f;
  const long long n8 = rows * (cols / 8);
  for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < n8; q += (long long)gridDim.x * blockDim.x) {
    const long long r = q / (cols / 8);
    const int c = (int)(q % (cols / 8)) * 8;
    const uint4 v = *reinterpret_cast<const uint4*>(X + r * ld + c);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      m = fmaxf(m, fabsf(net::bf2f((uint16_t)(w[e] & 0xffffu))));
      m = fmaxf(m, fabsf(net::bf2f((uint16_t)(w[e] >> 16))));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0 && m > 0.f) atomicMax(reinterpret_cast<int*>(slot), __float_as_int(m));
}

}  // namespace f8
