// 텐서 코어 어텐션(앞·뒤) — tk::AttP 의 일반 어텐션(구간 1 [B][L1](판마다 유효 len1, 인과면 j ≤ t + qoff) + 구간 2 [B][n2], GQA).
// 블록 = (판, 질의 머리, 질의 QT 행). 점수 행(키 ≤ NKMAX)을 공유 메모리에 통째로 두는 2 단 softmax:
//   S = scale·(bf(Q)·bf(K)ᵀ) (FP32 누산) → 행마다 최대·합(FP32) → lse, P = e^{S − lse}(정규화된 값) → bf(P) → O = bf(P)·bf(V).
// 뒤: D = Σ dO·O(FP32), dP = bf(dO)·bf(V)ᵀ, dS = P(dP − D)(P 는 FP32 로 다시 계산), dQ = scale·bf(dS)·bf(K)(질의 블록),
//     dK = scale·Σ bf(dS)ᵀ·bf(Q), dV = Σ bf(P)ᵀ·bf(dO)(키 블록 = (판, 키 머리, 키 32 개), 같은 무리의 질의 머리·행을 차례로 → 고정 순서, 원자 없음).
// 반올림 자리는 CPU 참조판 EMUL(vref att_f/att_b)과 같다. 키가 NKMAX 를 넘으면 예전 FP32 커널(tkern.cu)로.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "mma.cuh"
#include "net.h"
#include "tkern.cuh"

namespace rvla {
namespace tk {

using net::bf2f;
using net::f2bf;
#define KCK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)
constexpr int FNT = 256, FNW = 8;

__device__ __forceinline__ int s1len(const AttP& p, int b) { return p.len1 ? p.len1[b] : p.n1c; }
__device__ __forceinline__ void put4(uint16_t* d, float4 v) {
  uint2 u;
  u.x = (uint32_t)f2bf(v.x) | ((uint32_t)f2bf(v.y) << 16);
  u.y = (uint32_t)f2bf(v.z) | ((uint32_t)f2bf(v.w) << 16);
  *reinterpret_cast<uint2*>(d) = u;
}
// 키 가상 번호 j(블록 안 [0, n1e) = 구간 1, [n1e, n1e + n2) = 구간 2)의 K·V 행 포인터(머리 kh 의 시작)
__device__ __forceinline__ const float* key_row(const AttP& p, const float* base1, const float* base2, int b, int j, int n1e, int kh) {
  if (j < n1e) return base1 + ((long long)b * p.L1 + j) * p.ldk1 + kh * p.hd;
  return base2 + ((long long)b * p.n2 + (j - n1e)) * p.ldk2 + kh * p.hd;
}

// ---------------------------------------------------------------------------------------------------------------------
template <int HD, int QT, int KT, int NKMAX>
__global__ void __launch_bounds__(FNT) fa_fwd_k(const AttP p) {
  constexpr int LD = HD + 8, LS = NKMAX + 4, LP = 2 * LS;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sQ = (uint16_t*)smem;            // [QT][LD]
  uint16_t* sKV = sQ + QT * LD;              // [KT][LD]
  float* sS = (float*)(sKV + KT * LD);       // [QT][LS], 나중에 bf16 P 가 같은 줄 앞쪽에
  uint16_t* sP = (uint16_t*)sS;
  const int t0 = blockIdx.x * QT, h = blockIdx.y, b = blockIdx.z, kh = h / (p.nq / p.nkv), tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int n1b = s1len(p, b);
  int n1e = n1b;
  if (p.causal) n1e = min(n1b, t0 + QT - 1 + p.qoff + 1);
  n1e = max(n1e, 0);
  const int NK = n1e + p.n2, NKr = (NK + KT - 1) / KT * KT;
  for (int q = tid; q < QT * HD / 4; q += FNT) {
    const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, t = t0 + r;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t < p.n) v = *reinterpret_cast<const float4*>(p.Q + ((long long)b * p.n + t) * p.ldq + h * HD + d);
    put4(sQ + r * LD + d, v);
  }
  for (int k0 = 0; k0 < NKr; k0 += KT) {
    __syncthreads();
    for (int q = tid; q < KT * HD / 4; q += FNT) {
      const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, j = k0 + r;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (j < NK) v = *reinterpret_cast<const float4*>(key_row(p, p.K1, p.K2, b, j, n1e, kh) + d);
      put4(sKV + r * LD + d, v);
    }
    __syncthreads();
    mm::Acc<QT, KT, FNW> s;
    s.zero();
    s.mac<HD, false, false>(mm::sad(sQ), LD, mm::sad(sKV), LD);
    s.each([&](int r, int c, float& v) {
      const int t = t0 + r, j = k0 + c;
      const bool ok = j < n1e ? (j < n1b && (!p.causal || j <= t + p.qoff)) : j < NK;
      sS[r * LS + j] = ok ? v * p.scale : -INFINITY;
    });
  }
  __syncthreads();
  // 행 softmax: 워프 w 가 행 w, w + 8, …
  for (int r = w; r < QT; r += FNW) {
    float mx = -INFINITY;
    for (int j = lane; j < NKr; j += 32) mx = fmaxf(mx, sS[r * LS + j]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
    float pv[NKMAX / 32];
#pragma unroll
    for (int k = 0; k < NKMAX / 32; ++k) {
      const int j = lane + 32 * k;
      pv[k] = (j < NKr && mx > -INFINITY) ? expf(sS[r * LS + j] - mx) : 0.f;
      sum = sum + pv[k];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = sum + __shfl_xor_sync(0xffffffffu, sum, o);
    const float lse = sum > 0.f ? mx + logf(sum) : 0.f;
#pragma unroll
    for (int k = 0; k < NKMAX / 32; ++k) {
      const int j = lane + 32 * k;
      pv[k] = (j < NKr && sum > 0.f) ? expf(sS[r * LS + j] - lse) : 0.f;
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < NKMAX / 32; ++k) {
      const int j = lane + 32 * k;
      if (j < NKr) sP[r * LP + j] = f2bf(pv[k]);
    }
    if (lane == 0 && p.lse && t0 + r < p.n) p.lse[((long long)b * p.n + t0 + r) * p.nq + h] = lse;
  }
  mm::Acc<QT, HD, FNW> o;
  o.zero();
  for (int k0 = 0; k0 < NKr; k0 += KT) {
    __syncthreads();
    for (int q = tid; q < KT * HD / 4; q += FNT) {
      const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, j = k0 + r;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (j < NK) v = *reinterpret_cast<const float4*>(key_row(p, p.V1, p.V2, b, j, n1e, kh) + d);
      put4(sKV + r * LD + d, v);
    }
    __syncthreads();
    o.mac<KT, false, true>(mm::sad(sP + k0), LP, mm::sad(sKV), LD);
  }
  o.each([&](int r, int d, float& v) {
    if (t0 + r < p.n) p.O[((long long)b * p.n + t0 + r) * p.ldo + h * HD + d] = v;
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// 뒤 1: 질의 블록 → D, dQ
template <int HD, int QT, int KT, int NKMAX>
__global__ void __launch_bounds__(FNT) fa_dq_k(const AttP p) {
  constexpr int LD = HD + 8, LS = NKMAX + 8;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sQ = (uint16_t*)smem;
  uint16_t* sdO = sQ + QT * LD;
  uint16_t* sK = sdO + QT * LD;
  uint16_t* sV = sK + KT * LD;
  uint16_t* sdS = sV + KT * LD;              // [QT][LS]
  float* sL = (float*)(sdS + QT * LS);
  float* sD = sL + QT;
  const int t0 = blockIdx.x * QT, h = blockIdx.y, b = blockIdx.z, kh = h / (p.nq / p.nkv), tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int n1b = s1len(p, b);
  int n1e = n1b;
  if (p.causal) n1e = min(n1b, t0 + QT - 1 + p.qoff + 1);
  n1e = max(n1e, 0);
  const int NK = n1e + p.n2, NKr = (NK + KT - 1) / KT * KT;
  for (int r = w; r < QT; r += FNW) {
    const int t = t0 + r;
    float d = 0.f;
    if (t < p.n)
      for (int e = lane; e < HD; e += 32)
        d = d + p.dO[((long long)b * p.n + t) * p.lddo + h * HD + e] * p.O[((long long)b * p.n + t) * p.ldo + h * HD + e];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
    if (lane == 0) {
      sD[r] = d;
      sL[r] = t < p.n ? p.lse[((long long)b * p.n + t) * p.nq + h] : 0.f;
      if (t < p.n) p.Dd[((long long)b * p.n + t) * p.nq + h] = d;
    }
  }
  for (int q = tid; q < QT * HD / 4; q += FNT) {
    const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, t = t0 + r;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), g = a;
    if (t < p.n) {
      a = *reinterpret_cast<const float4*>(p.Q + ((long long)b * p.n + t) * p.ldq + h * HD + d);
      g = *reinterpret_cast<const float4*>(p.dO + ((long long)b * p.n + t) * p.lddo + h * HD + d);
    }
    put4(sQ + r * LD + d, a);
    put4(sdO + r * LD + d, g);
  }
  for (int k0 = 0; k0 < NKr; k0 += KT) {
    __syncthreads();
    for (int q = tid; q < KT * HD / 4; q += FNT) {
      const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, j = k0 + r;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f), v = a;
      if (j < NK) { a = *reinterpret_cast<const float4*>(key_row(p, p.K1, p.K2, b, j, n1e, kh) + d); v = *reinterpret_cast<const float4*>(key_row(p, p.V1, p.V2, b, j, n1e, kh) + d); }
      put4(sK + r * LD + d, a);
      put4(sV + r * LD + d, v);
    }
    __syncthreads();
    mm::Acc<QT, KT, FNW> s, dp;
    s.zero(); dp.zero();
    s.mac<HD, false, false>(mm::sad(sQ), LD, mm::sad(sK), LD);
    dp.mac<HD, false, false>(mm::sad(sdO), LD, mm::sad(sV), LD);
    // 같은 배치라 원소 순서가 같다
    s.each2(dp, [&](int r, int c, float& v, float& dpv) {
      const int t = t0 + r, j = k0 + c;
      const bool ok = t < p.n && (j < n1e ? (j < n1b && (!p.causal || j <= t + p.qoff)) : j < NK);
      const float pr = ok ? expf(v * p.scale - sL[r]) : 0.f;
      sdS[r * LS + j] = f2bf(pr * (dpv - sD[r]));
    });
  }
  mm::Acc<QT, HD, FNW> dq;
  dq.zero();
  for (int k0 = 0; k0 < NKr; k0 += KT) {
    __syncthreads();
    for (int q = tid; q < KT * HD / 4; q += FNT) {
      const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, j = k0 + r;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      if (j < NK) a = *reinterpret_cast<const float4*>(key_row(p, p.K1, p.K2, b, j, n1e, kh) + d);
      put4(sK + r * LD + d, a);
    }
    __syncthreads();
    dq.mac<KT, false, true>(mm::sad(sdS + k0), LS, mm::sad(sK), LD);
  }
  const float sc = p.bug == 1 ? 1.f : p.scale;
  dq.each([&](int r, int d, float& v) {
    if (t0 + r < p.n) p.dQ[((long long)b * p.n + t0 + r) * p.lddq + h * HD + d] = v * sc;
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// 뒤 2: 키 블록(판, 키 머리, 키 KB 개) → dK, dV. blockIdx.x < nt1 이면 구간 1 조각, 아니면 구간 2.
template <int HD, int KB, int QB>
__global__ void __launch_bounds__(FNT) fa_dkv_k(const AttP p, int nt1) {
  constexpr int LD = HD + 8, LT = QB + 8;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sK = (uint16_t*)smem;
  uint16_t* sV = sK + KB * LD;
  uint16_t* sQ = sV + KB * LD;
  uint16_t* sdO = sQ + QB * LD;
  uint16_t* sP = sdO + QB * LD;             // [KB][LT]
  uint16_t* sdS = sP + KB * LT;
  float* sL = (float*)(sdS + KB * LT);
  float* sD = sL + QB;
  const int kh = blockIdx.y, b = blockIdx.z, tid = threadIdx.x;
  const bool seg1 = (int)blockIdx.x < nt1;
  const int j0 = seg1 ? blockIdx.x * KB : (blockIdx.x - nt1) * KB;
  const int Ls = seg1 ? p.L1 : p.n2, lds = seg1 ? p.ldk1 : p.ldk2;
  const float* Kp = seg1 ? p.K1 : p.K2;
  const float* Vp = seg1 ? p.V1 : p.V2;
  float* dKp = seg1 ? p.dK1 : p.dK2;
  float* dVp = seg1 ? p.dV1 : p.dV2;
  const int n1b = s1len(p, b);
  const int nvalid = seg1 ? min(n1b, Ls) : Ls;   // 이 구간에서 유효한 키 수
  for (int q = tid; q < KB * HD / 4; q += FNT) {
    const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, j = j0 + r;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), v = a;
    if (j < nvalid) {
      a = *reinterpret_cast<const float4*>(Kp + ((long long)b * Ls + j) * lds + kh * HD + d);
      v = *reinterpret_cast<const float4*>(Vp + ((long long)b * Ls + j) * lds + kh * HD + d);
    }
    put4(sK + r * LD + d, a);
    put4(sV + r * LD + d, v);
  }
  mm::Acc<KB, HD, FNW> dk, dv;
  dk.zero(); dv.zero();
  const int G = p.nq / p.nkv;
  int tq0 = 0;
  if (seg1 && p.causal) tq0 = max(0, j0 - p.qoff) / QB * QB;
  if (j0 < nvalid)
    for (int hh = 0; hh < G; ++hh) {
      const int h = kh * G + hh;
      for (int q0 = tq0; q0 < p.n; q0 += QB) {
        __syncthreads();
        for (int q = tid; q < QB * HD / 4; q += FNT) {
          const int r = q / (HD / 4), d = (q % (HD / 4)) * 4, t = q0 + r;
          float4 a = make_float4(0.f, 0.f, 0.f, 0.f), g = a;
          if (t < p.n) {
            a = *reinterpret_cast<const float4*>(p.Q + ((long long)b * p.n + t) * p.ldq + h * HD + d);
            g = *reinterpret_cast<const float4*>(p.dO + ((long long)b * p.n + t) * p.lddo + h * HD + d);
          }
          put4(sQ + r * LD + d, a);
          put4(sdO + r * LD + d, g);
        }
        for (int r = tid; r < QB; r += FNT) {
          const int t = q0 + r;
          sL[r] = t < p.n ? p.lse[((long long)b * p.n + t) * p.nq + h] : 0.f;
          sD[r] = t < p.n ? p.Dd[((long long)b * p.n + t) * p.nq + h] : 0.f;
        }
        __syncthreads();
        mm::Acc<KB, QB, FNW> st, dpt;
        st.zero(); dpt.zero();
        st.mac<HD, false, false>(mm::sad(sK), LD, mm::sad(sQ), LD);
        dpt.mac<HD, false, false>(mm::sad(sV), LD, mm::sad(sdO), LD);
        st.each2(dpt, [&](int r, int c, float& v, float& dpv) {
          const int j = j0 + r, t = q0 + c;
          const bool ok = t < p.n && j < nvalid && (!seg1 || !p.causal || j <= t + p.qoff);
          const float pr = ok ? expf(v * p.scale - sL[c]) : 0.f;
          sP[r * LT + c] = f2bf(pr);
          sdS[r * LT + c] = f2bf(pr * (dpv - sD[c]));
        });
        __syncthreads();
        dv.mac<QB, false, true>(mm::sad(sP), LT, mm::sad(sdO), LD);
        dk.mac<QB, false, true>(mm::sad(sdS), LT, mm::sad(sQ), LD);
      }
    }
  const float sc = p.bug == 1 ? 1.f : p.scale;
  dk.each([&](int r, int d, float& v) {
    const int j = j0 + r;
    if (j < Ls) dKp[((long long)b * Ls + j) * lds + kh * HD + d] = j < nvalid ? v * sc : 0.f;
  });
  dv.each([&](int r, int d, float& v) {
    const int j = j0 + r;
    if (j < Ls) dVp[((long long)b * Ls + j) * lds + kh * HD + d] = j < nvalid ? v : 0.f;
  });
}

// ---------------------------------------------------------------------------------------------------------------------
template <class K_>
static void flaunch(K_ kern, dim3 grid, size_t sm, cudaStream_t st, const AttP& a) {
  if (sm > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  kern<<<grid, FNT, sm, st>>>(a);
  KCK();
}
template <int HD, int NKMAX>
static void fa_fwd_t(const AttP& p, cudaStream_t st) {
  constexpr int QT = NKMAX > 256 ? 16 : 32, KT = HD >= 256 ? 32 : 64;
  const size_t sm = 2 * (QT + KT) * (HD + 8) + 4 * QT * (NKMAX + 4);
  flaunch(fa_fwd_k<HD, QT, KT, NKMAX>, dim3((p.n + QT - 1) / QT, p.nq, p.B), sm, st, p);
}
template <int HD, int NKMAX>
static void fa_bwd_t(const AttP& p, cudaStream_t st) {
  constexpr int QT = NKMAX > 256 ? 16 : 32, KT = 32;
  const size_t sm = 2 * (2 * QT + 2 * KT) * (HD + 8) + 2 * QT * (NKMAX + 8) + 4 * 2 * QT;
  flaunch(fa_dq_k<HD, QT, KT, NKMAX>, dim3((p.n + QT - 1) / QT, p.nq, p.B), sm, st, p);
  constexpr int KB = 32, QB = 32;
  const int nt1 = p.dK1 ? (p.L1 + KB - 1) / KB : 0, nt2 = p.dK2 ? (p.n2 + KB - 1) / KB : 0;
  if (nt1 + nt2 == 0) return;
  const size_t sm2 = 2 * (2 * KB + 2 * QB) * (HD + 8) + 2 * 2 * KB * (QB + 8) + 4 * 2 * QB;
  if (sm2 > 48 * 1024) cudaFuncSetAttribute(fa_dkv_k<HD, KB, QB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm2);
  fa_dkv_k<HD, KB, QB><<<dim3(nt1 + nt2, p.nkv, p.B), FNT, sm2, st>>>(p, nt1);
  KCK();
}
static int nk_max(const AttP& p) { return (p.len1 ? p.L1 : std::min(p.L1, p.n1c)) + p.n2; }
bool fa_fwd(const AttP& p, cudaStream_t st) {
  const int nk = nk_max(p);
  if (p.ldq % 4 || p.ldk1 % 4 || (p.n2 && p.ldk2 % 4) || p.ldo % 4) return false;
  if (nk <= 256) {
    switch (p.hd) {
      case 256: fa_fwd_t<256, 256>(p, st); return true;
      case 64: fa_fwd_t<64, 256>(p, st); return true;
      case 32: fa_fwd_t<32, 256>(p, st); return true;
      default: return false;
    }
  }
  if (nk <= 512) {
    switch (p.hd) {
      case 256: fa_fwd_t<256, 512>(p, st); return true;
      case 64: fa_fwd_t<64, 512>(p, st); return true;
      case 32: fa_fwd_t<32, 512>(p, st); return true;
      default: return false;
    }
  }
  return false;
}
bool fa_bwd(const AttP& p, cudaStream_t st) {
  const int nk = nk_max(p);
  if (p.ldq % 4 || p.ldk1 % 4 || (p.n2 && p.ldk2 % 4) || p.ldo % 4 || p.lddo % 4) return false;
  if (nk <= 256) {
    switch (p.hd) {
      case 256: fa_bwd_t<256, 256>(p, st); return true;
      case 64: fa_bwd_t<64, 256>(p, st); return true;
      case 32: fa_bwd_t<32, 256>(p, st); return true;
      default: return false;
    }
  }
  if (nk <= 512) {
    switch (p.hd) {
      case 256: fa_bwd_t<256, 512>(p, st); return true;
      case 64: fa_bwd_t<64, 512>(p, st); return true;
      case 32: fa_bwd_t<32, 512>(p, st); return true;
      default: return false;
    }
  }
  return false;
}

}  // namespace tk
}  // namespace rvla
