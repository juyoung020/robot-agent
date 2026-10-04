// 인코더 GEMM v2(속도판): C[M][N] = A[M][K] · W[N][K]ᵀ (둘 다 K 연속 16 비트) → epi(r, c, acc0, acc1).
// vit.cu 의 예전 커널(f8::tn_k<TN_F16H> 128 × 128 × k32, vgemm_k)과 같은 계산 순서를 지킨다:
//   출력 원소마다 k16 mma 를 k 0 → K 차례로, FP16 누산(HK_F16H)은 HP 타일(k 32 × HP)마다 FP16 부분합을 FP32 누산기로 옮김 → 결과 비트가 예전과 같다.
// 다른 것: 타일 k 64(동기 반으로), 공유 메모리 줄 128 B 에 16 B 덩이 XOR 섞기(덧댐 없음 → 단 수를 늘림), 블록·워프 타일을 모양마다.
// 피연산자 형식(KIND): HK_F16 = FP16·FP32 누산, HK_BF16 = BF16·FP32 누산, HK_F16H = FP16·FP16 누산(옮김 HP).
// 학습(RecallVLA)용 역전파 dX = dY·W 는 W 가 N 연속(전치)이라 이 커널이 아니라 ldmatrix.trans 판이 필요 — 같은 타일 틀에 B 읽기만 바꾸면 된다(남은 일).
#pragma once
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace vit {
namespace hk {

enum Kind : int { HK_F16 = 0, HK_BF16 = 1, HK_F16H = 2 };

__device__ __forceinline__ void cp16(unsigned a, const void* g) {   // a = 공유 메모리 주소(__cvta_generic_to_shared)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// [행][BK] 타일(줄 = BK·2 바이트)에서 (행, 16 B 덩이) 의 바이트 자리. 덩이를 XOR 로 섞어 ldmatrix 8 줄·cp.async 가 서로 다른 은행 묶음(16 B × 8)에 떨어지게:
//   BK 64 (줄 128 B): 덩이 ^= 행 % 8,  BK 32 (줄 64 B, 128 B 에 두 줄): 덩이 ^= (행 / 2) % 4
template <int BK>
__device__ __forceinline__ int swz(int row, int ch) {
  if constexpr (BK == 64) return row * 128 + ((ch ^ (row & 7)) << 4);
  else return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4);
}

template <int R, int BK, int NT>
__device__ __forceinline__ void load_tile(unsigned s, const uint16_t* __restrict__ P, int ld, int r0, int rlim, int k0, int tid) {
  constexpr int CPR = BK / 8, CH = R * CPR;   // 줄당 덩이, 덩이 수
#pragma unroll
  for (int q0 = 0; q0 < CH; q0 += NT) {
    const int q = q0 + tid;
    if (CH % NT != 0 && q >= CH) break;
    const int row = q / CPR, c = q % CPR;
    const int gr = min(r0 + row, rlim - 1);   // M 끝 넘는 행은 마지막 행을 다시 읽음(끝단이 버림)
    cp16(s + swz<BK>(row, c), P + (size_t)gr * ld + k0 + c * 8);
  }
}

template <int KIND>
__device__ __forceinline__ void mma32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  if constexpr (KIND == HK_BF16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(d[0]), "+r"(d[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int BM, int BN, int BK>
constexpr int smem_bytes(int nst) { return nst * (BM + BN) * BK * 2; }

// 요구: N % BN == 0, K % 64 == 0, M ≥ 1(아무 값 — 넘는 행은 끝단에서 버림), 행 시작 16 B 정렬(인코더 모양은 다 맞음 — 호출 쪽이 검사).
// 끝단 Epi: RMW(옛 값을 읽는가), colv(c) = 열 c·c+1 의 더할 값(편향), ld(r, c) = 옛 값(RMW 일 때), st(r, c, x0, x1, old) = 쓰기.
// HP: HK_F16H 옮김 간격(k 32 단위, 1·2·4). 타일 k 64 안에서 k16 번호로 판단하므로 예전 tn_k(타일 k 32) 와 같은 자리에서 옮긴다.
template <int KIND, int HP, int BM, int BN, int WM, int WN, int BK, int NST, int MINB, class Epi>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MINB)
    gemm_k(const uint16_t* __restrict__ A, int lda, const uint16_t* __restrict__ B, int ldb, int M, int K, const __grid_constant__ Epi epi) {
  constexpr int NWM = BM / WM, NT = NWM * (BN / WN) * 32, MI = WM / 16, NI = WN / 8;
  static_assert(NI % 2 == 0 && WM % 16 == 0, "warp tile");
  constexpr int SA = BM * BK * 2, SST = (BM + BN) * BK * 2;   // 단 하나의 바이트(A 다음 B)
  extern __shared__ __align__(128) uint8_t hks[];
  const unsigned s0 = (unsigned)__cvta_generic_to_shared(hks);
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
  uint32_t hacc[KIND == HK_F16H ? MI : 1][KIND == HK_F16H ? NI : 1][2];
  const int nk = K / BK;
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) {
      load_tile<BM, BK, NT>(s0 + s * SST, A, lda, m0, M, s * BK, tid);
      load_tile<BN, BK, NT>(s0 + s * SST + SA, B, ldb, n0, 1 << 30, s * BK, tid);
    }
    commit();
  }
  // ldmatrix 레인 주소: A 는 행렬 j = lane/8 → (행 +8·(j&1), k 덩이 +(j>>1)), B 는 (행 +8·(j>>1), k 덩이 +(j&1))
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  int arow[MI], brow[NI / 2];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) arow[mi] = wm + mi * 16 + lr + j0 * 8;
#pragma unroll
  for (int nj = 0; nj < NI / 2; ++nj) brow[nj] = wn + nj * 16 + lr + j1 * 8;
  for (int kt = 0; kt < nk; ++kt) {
    wait<NST - 2>();
    __syncthreads();
    {   // 다음 단 읽기(지금 읽을 단과 다른 자리 — 위 동기로 앞 단 계산이 끝남)
      const int nx = kt + NST - 1;
      if (nx < nk) {
        const int sl = nx % NST;
        load_tile<BM, BK, NT>(s0 + sl * SST, A, lda, m0, M, nx * BK, tid);
        load_tile<BN, BK, NT>(s0 + sl * SST + SA, B, ldb, n0, 1 << 30, nx * BK, tid);
      }
      commit();
    }
    const unsigned sa = s0 + (kt % NST) * SST, sb = sa + SA;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      const int k16 = kt * (BK / 16) + kk;
      if constexpr (KIND == HK_F16H)
        if (k16 % (2 * HP) == 0) {
#pragma unroll
          for (int a = 0; a < MI; ++a)
#pragma unroll
            for (int b = 0; b < NI; ++b) hacc[a][b][0] = hacc[a][b][1] = 0u;
        }
      uint32_t af[MI][4], bfr[NI / 2][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) ldsm4(af[mi], sa + swz<BK>(arow[mi], kk * 2 + j1));
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) ldsm4(bfr[nj], sb + swz<BK>(brow[nj], kk * 2 + j0));
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const uint32_t b0 = bfr[ni >> 1][(ni & 1) * 2], b1 = bfr[ni >> 1][(ni & 1) * 2 + 1];
          if constexpr (KIND == HK_F16H) mma16(hacc[mi][ni], af[mi], b0, b1);
          else mma32<KIND>(acc[mi][ni], af[mi], b0, b1);
        }
      if constexpr (KIND == HK_F16H)
        if (k16 % (2 * HP) == 2 * HP - 1 || k16 == nk * (BK / 16) - 1) {
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
    }
  }
  wait<0>();
  // 끝단: 열 상수(편향)를 먼저 레지스터로, 행 묶음(mi)마다 옛 값(잔차 X·위치 임베딩)을 모두 먼저 읽고 나서 쓴다
  // (쓰기와 읽기가 겹칠 수 있다고 보는 컴파일러가 원소마다 읽기 왕복을 기다리지 않게 — 잰 값: 잔차 GEMM 1.4 배)
  float2 cv[NI];
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) cv[ni] = epi.colv(n0 + wn + ni * 8 + 2 * t4);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
    float2 old[NI][2];
    if constexpr (Epi::RMW) {
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = m0 + wm + mi * 16 + g + h * 8, c = n0 + wn + ni * 8 + 2 * t4;
          old[ni][h] = r < M ? epi.ld(r, c) : make_float2(0.f, 0.f);
        }
    }
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m0 + wm + mi * 16 + g + h * 8, c = n0 + wn + ni * 8 + 2 * t4;
        if (r < M) epi.st(r, c, acc[mi][ni][2 * h] + cv[ni].x, acc[mi][ni][2 * h + 1] + cv[ni].y, Epi::RMW ? old[ni][h] : make_float2(0.f, 0.f));
      }
  }
}

}  // namespace hk
}  // namespace vit
