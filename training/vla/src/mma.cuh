// 공유 메모리 bf16 행렬끼리의 텐서 코어 곱(mma.sync m16n8k16, bf16 입력, FP32 누산) — DeltaNet 덩이 꼴·플래시 어텐션 커널이 쓴다.
// 자리 규칙: A(m,k) 는 AT = false 면 sA[m·lda + k](행 우선), AT = true 면 sA[k·lda + m]. B(k,n) 은 BT = false 면 sB[n·ldb + k], BT = true 면 sB[k·ldb + n].
// ld 는 8 의 배수(줄 시작 16 B 정렬, ldmatrix). 줄 간격을 (열 + 8) 로 두면 8 줄이 서로 다른 16 B 덩이에 떨어져 은행 충돌이 없다.
// 결과 조각(레인 l, g = l/4, q = l%4): c0·c1 = (행 g, 열 2q·2q+1), c2·c3 = (행 g+8, 같은 열).
#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace rvla {
namespace mm {

__device__ __forceinline__ uint32_t sad(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldsm2t(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// A 조각(행 m0..m0+15, k k0..k0+15)
template <bool AT>
__device__ __forceinline__ void lda_frag(uint32_t (&a)[4], uint32_t sA, int lda, int m0, int k0) {
  const int l = threadIdx.x & 31;
  if (!AT) ldsm4(a, sA + 2u * (uint32_t)((m0 + (l & 7) + 8 * ((l >> 3) & 1)) * lda + k0 + 8 * (l >> 4)));
  else ldsm4t(a, sA + 2u * (uint32_t)((k0 + (l & 7) + 8 * (l >> 4)) * lda + m0 + 8 * ((l >> 3) & 1)));
}
// B 조각 하나(k k0..k0+15, 열 n0..n0+7)
template <bool BT>
__device__ __forceinline__ void ldb_frag(uint32_t (&b)[2], uint32_t sB, int ldb, int n0, int k0) {
  const int l = threadIdx.x & 15;
  if (!BT) ldsm2(b, sB + 2u * (uint32_t)((n0 + (l & 7)) * ldb + k0 + 8 * (l >> 3)));
  else ldsm2t(b, sB + 2u * (uint32_t)((k0 + (l & 7) + 8 * (l >> 3)) * ldb + n0));
}
// B 조각 둘(열 n0..n0+15): b[0] = n0..n0+7, b[1] = n0+8..n0+15
template <bool BT>
__device__ __forceinline__ void ldb_frag2(uint32_t (&b)[2][2], uint32_t sB, int ldb, int n0, int k0) {
  const int l = threadIdx.x & 31;
  uint32_t r[4];
  if (!BT) {
    ldsm4(r, sB + 2u * (uint32_t)((n0 + (l & 7) + 8 * (l >> 4)) * ldb + k0 + 8 * ((l >> 3) & 1)));
  } else {
    ldsm4t(r, sB + 2u * (uint32_t)((k0 + (l & 7) + 8 * ((l >> 3) & 1)) * ldb + n0 + 8 * (l >> 4)));
  }
  b[0][0] = r[0]; b[0][1] = r[1]; b[1][0] = r[2]; b[1][1] = r[3];
}
// acc(16 × 8 조각) += A[m0.., 0..K) · B[0..K, n0..)
template <int K, bool AT, bool BT>
__device__ __forceinline__ void tile(float (&acc)[4], uint32_t sA, int lda, uint32_t sB, int ldb, int m0, int n0) {
#pragma unroll 4
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4], b[2];
    lda_frag<AT>(a, sA, lda, m0, k0);
    ldb_frag<BT>(b, sB, ldb, n0, k0);
    mma(acc, a, b);
  }
}
// 16 × 16 조각 둘(acc[0] = n0.., acc[1] = n0+8..) — A 조각을 한 번 읽어 두 번 씀
template <int K, bool AT, bool BT>
__device__ __forceinline__ void tile2(float (&acc)[2][4], uint32_t sA, int lda, uint32_t sB, int ldb, int m0, int n0) {
#pragma unroll 4
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4], b[2][2];
    lda_frag<AT>(a, sA, lda, m0, k0);
    ldb_frag2<BT>(b, sB, ldb, n0, k0);
    mma(acc[0], a, b[0]);
    mma(acc[1], a, b[1]);
  }
}

// 블록 단위 나눔: 결과 M × N 을 16 × 16 조각으로(N 이 8 이면 16 × 8), 워프 w 가 조각 w, w + NW, … 를 맡는다.
// 조각 번호 → (m0, n0). 같은 (M, N, NW) 면 늘 같은 배치라 단계 사이에 누산기를 레지스터에 둘 수 있다.
template <int M, int N, int NW>
struct Tiles {
  static constexpr int TN = N >= 16 ? 16 : 8, NT = N / TN, MT = M / 16, T = MT * NT, PW = (T + NW - 1) / NW;
  static_assert(M % 16 == 0 && N % TN == 0, "tile shape");
  __device__ static __forceinline__ bool has(int i) { return (threadIdx.x >> 5) + i * NW < T; }
  __device__ static __forceinline__ int m0(int i) { return (((threadIdx.x >> 5) + i * NW) / NT) * 16; }
  __device__ static __forceinline__ int n0(int i) { return (((threadIdx.x >> 5) + i * NW) % NT) * TN; }
};
// 누산기 묶음: Acc<M,N,NW>::a[i][h][4] (h = 16 × 8 반쪽, TN 8 이면 h = 0 만)
template <int M, int N, int NW>
struct Acc {
  using Tl = Tiles<M, N, NW>;
  static constexpr int H = Tl::TN / 8;
  float a[Tl::PW][2][4];
  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < Tl::PW; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) a[i][h][e] = 0.f;
  }
  // += A·B (K 방향 K)
  template <int K, bool AT, bool BT>
  __device__ __forceinline__ void mac(uint32_t sA, int lda, uint32_t sB, int ldb) {
#pragma unroll
    for (int i = 0; i < Tl::PW; ++i) {
      if (!Tl::has(i)) continue;
      if constexpr (H == 2) tile2<K, AT, BT>(a[i], sA, lda, sB, ldb, Tl::m0(i), Tl::n0(i));
      else tile<K, AT, BT>(a[i][0], sA, lda, sB, ldb, Tl::m0(i), Tl::n0(i));
    }
  }
  // 같은 배치의 누산기 둘을 원소마다 함께: f(행, 열, 이것&, 저것&)
  template <class F>
  __device__ __forceinline__ void each2(Acc& o, F&& f) {
    const int l = threadIdx.x & 31, g = l >> 2, q = l & 3;
#pragma unroll
    for (int i = 0; i < Tl::PW; ++i) {
      if (!Tl::has(i)) continue;
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) f(Tl::m0(i) + g + 8 * (e >> 1), Tl::n0(i) + 8 * h + 2 * q + (e & 1), a[i][h][e], o.a[i][h][e]);
    }
  }
  // 원소마다 f(행, 열, 값&) — 값을 바꿔도 됨
  template <class F>
  __device__ __forceinline__ void each(F&& f) {
    const int l = threadIdx.x & 31, g = l >> 2, q = l & 3;
#pragma unroll
    for (int i = 0; i < Tl::PW; ++i) {
      if (!Tl::has(i)) continue;
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) f(Tl::m0(i) + g + 8 * (e >> 1), Tl::n0(i) + 8 * h + 2 * q + (e & 1), a[i][h][e]);
    }
  }
};

}  // namespace mm
}  // namespace rvla
