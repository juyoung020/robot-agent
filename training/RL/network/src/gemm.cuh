// 손 GEMM: bf16 입력, FP32 누산 (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, 팀 pi05_native/gemm.cuh 와 같은 명령).
//
//   C[M,N] = Σ_k A(m,k) · B(n,k)
//     AT = false: A(m,k) = A[m*lda + k]   (K 연속: 앞 계산의 X, dX 의 dZ)
//     AT = true : A(m,k) = A[k*lda + m]   (M 연속: dW 의 dZᵀ)
//     BT = false: B(n,k) = B[n*ldb + k]   (K 연속: 앞 계산의 W[N][K])
//     BT = true : B(n,k) = B[k*ldb + n]   (N 연속: dX 의 W, dW 의 X)
//   앞:  Y = X·Wᵀ            (AT=0, BT=0)
//   dX: dX = dZ·W            (AT=0, BT=1)  끝단에서 이전 층 ELU' 를 곱해 dZ_prev(bf16) 로
//   dW: dW = dZᵀ·X           (AT=1, BT=1)  K(= 행) 를 나눠(split-K) 조각마다 FP32 부분합 → 고정 순서 합(결정적, 9절 V7)
//
// 공유 메모리 타일은 둘 다 [행][k] (k 연속) 로 둔다. 줄 간격 40 bf16(20 낱말), 16 B 덩이 단위 XOR 섞기(행 >> 3 의 아래 2 비트)로
// 조각 읽기에 은행 충돌이 없다. 전치 피연산자는 k 두 줄을 같이 읽어 32 비트 낱말(k, k+1)로 바꿔 쓴다.
// 요구: M·N·K 모두 8 의 배수, 연속 방향 포인터 16 B 정렬. 타일 128 × 64 × 32, 워프 8 개(4 × 2), 워프마다 32 × 32.
// 이중 버퍼: 다음 타일을 레지스터로 읽어 두고 지금 타일을 계산한 뒤 다른 버퍼에 쓴다(타일마다 동기 한 번).
#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include "net.h"

namespace net {

constexpr int GBM = 128, GBN = 64, GBK = 32, GLW = 20 /* 줄 간격 낱말(40 bf16) */, GNT = 256;

enum Epi : int {
  EPI_ACT_BF16 = 0,   // C(bf16) = act(acc)
  EPI_F32 = 1,        // C(f32) = acc
  EPI_ACC_F32 = 2,    // C(f32) += acc
  EPI_DACT_BF16 = 3,  // C(bf16) = acc · elu'(Y)   (Y = 이전 층 출력 bf16, 같은 자리)
  EPI_SPLIT_F32 = 4,  // C(f32) + z·M·N = acc     (split-K 부분합, 줄 간격 N)
};

struct GemmP {
  const uint16_t* A;
  long long lda;
  const uint16_t* B;
  long long ldb;
  int M, N, K;
  void* C;
  long long ldc;
  const uint16_t* Y;   // EPI_DACT
  long long ldy;
  int act;             // EPI_ACT_BF16: ACT_ELU 또는 ACT_LIN
  int kchunk;          // EPI_SPLIT: 조각당 K
  int bug;             // 음성 대조(검증용): 1 이면 EPI_DACT 에서 elu' 를 빼먹는다
};

__device__ __forceinline__ int gsw(int row, int w) { return w ^ (((row >> 3) & 3) << 2); }

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// R 행 × 32 k 타일 읽기. T = false: 행마다 16 B 덩이 4 개. T = true: (k 두 줄, 8 행) 묶음을 스레드 하나가
template <int R, bool T>
struct TileLd {
  static constexpr int NV = T ? (R / 8) * (GBK / 2) / GNT : R * 4 / GNT;   // 스레드당 할 일 (R=128: 2, R=64: 1 또는 T 이면 0.5 → 아래 처리)
  static constexpr int NJ = NV > 0 ? NV : 1;
  uint4 v[NJ][T ? 2 : 1];
  __device__ __forceinline__ void load(const uint16_t* P, long long ld, int r0, int rlim, int k0, int klim, int tid) {
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int q = tid + j * GNT;
      if (!T) {
        const int row = q >> 2, c = q & 3;
        const int gr = r0 + row, gk = k0 + c * 8;
        v[j][0] = (gr < rlim && gk < klim) ? *reinterpret_cast<const uint4*>(P + (long long)gr * ld + gk) : make_uint4(0, 0, 0, 0);
      } else {
        constexpr int MC = R / 8;
        if (q >= MC * (GBK / 2)) { v[j][0] = v[j][1] = make_uint4(0, 0, 0, 0); continue; }
        const int mc = q % MC, kp = q / MC;
        const int gr = r0 + mc * 8, gk = k0 + 2 * kp;
        const bool ok = gr < rlim;
        v[j][0] = (ok && gk < klim) ? *reinterpret_cast<const uint4*>(P + (long long)gk * ld + gr) : make_uint4(0, 0, 0, 0);
        v[j][1] = (ok && gk + 1 < klim) ? *reinterpret_cast<const uint4*>(P + (long long)(gk + 1) * ld + gr) : make_uint4(0, 0, 0, 0);
      }
    }
  }
  __device__ __forceinline__ void store(uint32_t* S, int tid) const {
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int q = tid + j * GNT;
      if (!T) {
        const int row = q >> 2, c = q & 3;
        *reinterpret_cast<uint4*>(S + row * GLW + gsw(row, c * 4)) = v[j][0];
      } else {
        constexpr int MC = R / 8;
        if (q >= MC * (GBK / 2)) continue;
        const int mc = q % MC, kp = q / MC;
        const uint32_t a[4] = {v[j][0].x, v[j][0].y, v[j][0].z, v[j][0].w}, b[4] = {v[j][1].x, v[j][1].y, v[j][1].z, v[j][1].w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const uint32_t lo = (a[e >> 1] >> ((e & 1) * 16)) & 0xffffu, hi = (b[e >> 1] >> ((e & 1) * 16)) & 0xffffu;
          const int row = mc * 8 + e;
          S[row * GLW + gsw(row, kp)] = lo | (hi << 16);
        }
      }
    }
  }
};

template <bool AT, bool BT, int EPI>
__global__ void __launch_bounds__(GNT) gemm_k(GemmP p) {
  __shared__ __align__(16) uint32_t sA[2][GBM * GLW];
  __shared__ __align__(16) uint32_t sB[2][GBN * GLW];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
  int kbeg = 0, kend = p.K;
  if (EPI == EPI_SPLIT_F32) {
    kbeg = blockIdx.z * p.kchunk;
    kend = min(p.K, kbeg + p.kchunk);
  }
  const int wm = (warp & 3) * 32, wn = (warp >> 2) * 32;
  float acc[2][4][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

  TileLd<GBM, AT> la;
  TileLd<GBN, BT> lb;
  la.load(p.A, p.lda, m0, p.M, kbeg, kend, tid);
  lb.load(p.B, p.ldb, n0, p.N, kbeg, kend, tid);
  la.store(sA[0], tid);
  lb.store(sB[0], tid);
  __syncthreads();
  int buf = 0;
  for (int k0 = kbeg; k0 < kend; k0 += GBK) {
    const bool more = k0 + GBK < kend;
    if (more) {
      la.load(p.A, p.lda, m0, p.M, k0 + GBK, kend, tid);
      lb.load(p.B, p.ldb, n0, p.N, k0 + GBK, kend, tid);
    }
    const uint32_t* A = sA[buf];
    const uint32_t* B = sB[buf];
#pragma unroll
    for (int kk = 0; kk < GBK / 2; kk += 8) {
      uint32_t af[2][4], bfr[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r0 = wm + mi * 16 + g, r1 = r0 + 8;
        af[mi][0] = A[r0 * GLW + gsw(r0, kk + t4)];
        af[mi][1] = A[r1 * GLW + gsw(r1, kk + t4)];
        af[mi][2] = A[r0 * GLW + gsw(r0, kk + 4 + t4)];
        af[mi][3] = A[r1 * GLW + gsw(r1, kk + 4 + t4)];
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int n = wn + ni * 8 + g;
        bfr[ni][0] = B[n * GLW + gsw(n, kk + t4)];
        bfr[ni][1] = B[n * GLW + gsw(n, kk + 4 + t4)];
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) mma_bf16(acc[mi][ni], af[mi], bfr[ni]);
    }
    if (more) {
      la.store(sA[buf ^ 1], tid);
      lb.store(sB[buf ^ 1], tid);
    }
    __syncthreads();
    buf ^= 1;
  }

  // 끝단
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m0 + wm + mi * 16 + g + h * 8, c = n0 + wn + ni * 8 + 2 * t4;
        if (r >= p.M || c >= p.N) continue;
        float x0 = acc[mi][ni][2 * h], x1 = acc[mi][ni][2 * h + 1];
        if (EPI == EPI_ACT_BF16) {
          if (p.act == ACT_ELU) { x0 = elu(x0); x1 = elu(x1); }
          reinterpret_cast<uint32_t*>(p.C)[((long long)r * p.ldc + c) >> 1] = (uint32_t)f2bf(x0) | ((uint32_t)f2bf(x1) << 16);
        } else if (EPI == EPI_F32) {
          *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + (long long)r * p.ldc + c) = make_float2(x0, x1);
        } else if (EPI == EPI_ACC_F32) {
          float2* d = reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + (long long)r * p.ldc + c);
          float2 o = *d;
          o.x = o.x + x0;
          o.y = o.y + x1;
          *d = o;
        } else if (EPI == EPI_DACT_BF16) {
          const uint32_t yy = reinterpret_cast<const uint32_t*>(p.Y)[((long long)r * p.ldy + c) >> 1];
          if (!p.bug) {
            x0 = x0 * elu_grad_from_y(bf2f((uint16_t)(yy & 0xffffu)));
            x1 = x1 * elu_grad_from_y(bf2f((uint16_t)(yy >> 16)));
          }
          reinterpret_cast<uint32_t*>(p.C)[((long long)r * p.ldc + c) >> 1] = (uint32_t)f2bf(x0) | ((uint32_t)f2bf(x1) << 16);
        } else {   // EPI_SPLIT_F32
          float* C = reinterpret_cast<float*>(p.C) + (long long)blockIdx.z * p.M * p.N;
          *reinterpret_cast<float2*>(C + (long long)r * p.N + c) = make_float2(x0, x1);
        }
      }
}

}  // namespace net
