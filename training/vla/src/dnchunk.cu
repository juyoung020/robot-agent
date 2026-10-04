// Gated DeltaNet 덩이 꼴(chunkwise parallel, WY 표현 / UT 변환 — Yang 외 Gated DeltaNet, flash-linear-attention chunk_gated_delta_rule 과 같은 식)
// 앞·뒤. 텐서 코어(mma.sync bf16, FP32 누산). 식·반올림 자리는 CPU 참조판 vref::dn_chk_fwd/bwd 와 하나하나 같다(tools/vref.cpp).
//
// 덩이 C 행(마지막 덩이는 0 덧댐: k = v = q = β = g = 0 → 상태에 영향 없음), 판·머리(b, h), 덩이 시작 상태 S [dk][dv]:
//   γ_t = Σ_{s≤t} g_s, Γ = e^γ, M_ts = e^{γt−γs}
//   A_ts = β_t (k_t·k_s) M_ts (s < t), T = (I + A)⁻¹, T1b = bf(T·β_s), T2b = bf(T·β_s Γ_s), U = T1b Vb, Wb = bf(T2b Kb), Pb = bf((Qb Kbᵀ)∘M, s ≤ t)
//   Δ = U − Wb·bf(S), O = Γ_t (Qb bf(S))_t + Pb bf(Δ), S ← Γ_C S + Kbᵀ bf(Δ e^{γC−γ})
// 커널(앞): dn_prep_k(덩이마다 병렬: γ, A, T, Wb, U, Pb, Qb·Kb 사본) → dn_state_k(판·머리·열 조각마다 덩이를 차례로: 상태 S 는 레지스터)
// 커널(뒤): dn_dstate_k(거꾸로 차례: dS, dΔ, ρ) → dn_g1_k(상태 수축: dQ₁·dK_a·dWb·dP) → dn_g2_k(dQ₂·dK·dV·dT) → dn_g3_k(dA·dKK, dγ 역 누적합, dβ)
// 결정적: 부동소수 원자 없음, 모든 합은 고정 차례.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "mma.cuh"
#include "net.h"
#include "tkern.cuh"

namespace rvla {
namespace tk {

using net::bf2f;
using net::f2bf;
#define KCK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

constexpr int NT = 512, NW = 16;
template <int DV>
__host__ __device__ constexpr int jsw() { return DV < 32 ? DV : 32; }   // dn_g1_k 의 dv 조각 폭

// 작업 공간 배치(덩이 번호 ci = (b·lh + h)·nck + c)
struct DnL {
  int B, n, lh, dk, dv, C, nck, NP;
  long long NC;
  uint16_t *Kb, *Qb, *Wb, *Pb, *dWb, *dQKb, *dTb, *Sb, *dSb;   // Sb·dSb = 덩이 시작 상태·덩이 뒤 상태 기울기(bf16 검문점)
  float *U, *T, *gam, *be, *Dl, *dD, *rho, *dgam, *dbt, *dterm;
  long long floats;
};
static int dn_np(int) { return 1; }
static DnL dn_layout(int B, int n, int lh, int dk, int dv, float* ws) {
  DnL L;
  L.B = B; L.n = n; L.lh = lh; L.dk = dk; L.dv = dv; L.C = dn_chunk(dk); L.nck = (n + L.C - 1) / L.C; L.NP = dn_np(dv);
  L.NC = (long long)B * lh * L.nck;
  const long long NC = L.NC, C = L.C;
  float* p = ws;
  long long o = 0;
  auto take = [&](long long fl) { float* r = p ? p + o : nullptr; o += (fl + 63) / 64 * 64; return r; };
  L.Kb = (uint16_t*)take(NC * C * dk / 2); L.Qb = (uint16_t*)take(NC * C * dk / 2); L.Wb = (uint16_t*)take(NC * C * dk / 2);
  L.Pb = (uint16_t*)take(NC * C * C / 2); L.dWb = (uint16_t*)take(NC * C * dk / 2); L.dQKb = (uint16_t*)take(NC * C * C / 2); L.dTb = (uint16_t*)take(NC * C * C / 2);
  L.U = take(NC * C * dv); L.T = take(NC * C * C); L.gam = take(NC * C); L.be = take(NC * C);
  L.Sb = (uint16_t*)take(NC * dk * dv / 2); L.Dl = take(NC * C * dv); L.dSb = (uint16_t*)take(NC * dk * dv / 2); L.dD = take(NC * C * dv);
  L.rho = take(NC * L.NP * C); L.dgam = take(NC * C); L.dbt = take(NC * C); L.dterm = take(NC);
  L.floats = o;
  return L;
}
long long dnc_ws_floats(int B, int n, int lh, int dk, int dv) { return dn_layout(B, n, lh, dk, dv, nullptr).floats; }

struct DnArgs {
  const float *Q, *K, *V, *G, *Bt, *S0, *dO;
  int ldv, n, lh, nodecay, ck, bug;
  float *O, *S1, *dQ, *dK, *dV, *dG, *dB;
  int lddv;
  DnL L;
};

// 블록 단위 도우미
__device__ __forceinline__ void cp_bf(uint16_t* dst, int ldd, const uint16_t* src, int lds, int R, int Cc) {   // bf16 행렬 복사(열 8 의 배수)
  const int cw = Cc / 8;
  for (int q = threadIdx.x; q < R * cw; q += NT) {
    const int r = q / cw, c = (q % cw) * 8;
    *reinterpret_cast<uint4*>(dst + r * ldd + c) = *reinterpret_cast<const uint4*>(src + (long long)r * lds + c);
  }
}
// 블록 합(고정 차례): 스레드 값 → 모두에게 합
__device__ __forceinline__ float bsum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
  for (int w = 0; w < NW; ++w) s += red[w];
  __syncthreads();
  return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// 앞 1: 덩이마다(블록 = 덩이)
template <int DK, int DV, int C>
__global__ void __launch_bounds__(NT) dn_prep_k(const DnArgs p) {
  constexpr int LK = DK + 8, LQ = (DK > DV ? DK : DV) + 8, LC = C + 8, LF = C + 4;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sK = (uint16_t*)smem;
  uint16_t* sQ = sK + C * LK;                     // Q, 나중에 V
  float* sA = (float*)(sQ + C * LQ);              // A [C][LF], 나중에 T1b·T2b(bf16 [C][LC] 둘 = 4·C·LC 바이트)
  float* sT = sA + C * LC;
  float* sg = sT + C * LF + C * 16;   // 앞 C·16 칸 = T 풀기 임시 sX
  float* sGa = sg + C;
  float* sbe = sGa + C;
  uint16_t* sH1 = (uint16_t*)sA;
  uint16_t* sH2 = sH1 + C * LC;
  const DnL& L = p.L;
  const long long ci = blockIdx.x;
  const int c = (int)(ci % L.nck), bh = (int)(ci / L.nck), b = bh / p.lh, h = bh % p.lh, c0 = c * C, Cn = min(C, p.n - c0), tid = threadIdx.x;
  const long long r0 = (long long)b * p.n + c0;
  for (int q = tid; q < C * DK / 4; q += NT) {
    const int t = q / (DK / 4), i = (q % (DK / 4)) * 4;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), k = a;
    if (t < Cn) {
      a = *reinterpret_cast<const float4*>(p.Q + (r0 + t) * p.lh * DK + h * DK + i);
      k = *reinterpret_cast<const float4*>(p.K + (r0 + t) * p.lh * DK + h * DK + i);
    }
    uint16_t* qd = sQ + t * LQ + i;
    uint16_t* kd = sK + t * LK + i;
    qd[0] = f2bf(a.x); qd[1] = f2bf(a.y); qd[2] = f2bf(a.z); qd[3] = f2bf(a.w);
    kd[0] = f2bf(k.x); kd[1] = f2bf(k.y); kd[2] = f2bf(k.z); kd[3] = f2bf(k.w);
  }
  for (int t = tid; t < C; t += NT) {
    sGa[t] = (t < Cn && !p.nodecay) ? p.G[(r0 + t) * p.lh + h] : 0.f;
    sbe[t] = t < Cn ? p.Bt[(r0 + t) * p.lh + h] : 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.f;
    for (int t = 0; t < C; ++t) { acc = acc + sGa[t]; sg[t] = acc; }
  }
  __syncthreads();
  for (int t = tid; t < C; t += NT) sGa[t] = expf(sg[t]);
  __syncthreads();
  for (int t = tid; t < C; t += NT) { L.gam[ci * C + t] = sg[t]; L.be[ci * C + t] = sbe[t]; }
  for (int q = tid; q < C * DK / 8; q += NT) {
    const int t = q / (DK / 8), i = (q % (DK / 8)) * 8;
    *reinterpret_cast<uint4*>(L.Kb + (ci * C + t) * DK + i) = *reinterpret_cast<const uint4*>(sK + t * LK + i);
    *reinterpret_cast<uint4*>(L.Qb + (ci * C + t) * DK + i) = *reinterpret_cast<const uint4*>(sQ + t * LQ + i);
  }
  {   // QK → Pb(전역), KK → A
    mm::Acc<C, C, NW> qk, kk;
    qk.zero(); kk.zero();
    qk.mac<DK, false, false>(mm::sad(sQ), LQ, mm::sad(sK), LK);
    kk.mac<DK, false, false>(mm::sad(sK), LK, mm::sad(sK), LK);
    qk.each([&](int t, int s, float& v) { L.Pb[(ci * C + t) * C + s] = f2bf(s <= t ? v * expf(sg[t] - sg[s]) : 0.f); });
    kk.each([&](int t, int s, float& v) { sA[t * LF + s] = s < t ? sbe[t] * v * expf(sg[t] - sg[s]) : 0.f; });
  }
  __syncthreads();
  // V 를 Q 자리로, T 풀기(열 u 마다 스레드 하나: T_tu = −Σ_{u≤r<t} A_tr T_ru)
  for (int q = tid; q < C * DV / 4; q += NT) {
    const int t = q / (DV / 4), j = (q % (DV / 4)) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t < Cn) v = *reinterpret_cast<const float4*>(p.V + (r0 + t) * p.ldv + h * DV + j);
    uint16_t* d = sQ + t * LQ + j;
    d[0] = f2bf(v.x); d[1] = f2bf(v.y); d[2] = f2bf(v.z); d[3] = f2bf(v.w);
  }
  // T = (I + A)⁻¹ 를 16 × 16 덩어리로: 대각 덩어리는 열마다 앞으로 풀기, 아래 덩어리 (i, j) 는 거리 d = i − j 차례로
  //   T_ij = −T_ii Σ_{k=j}^{i−1} A_ik T_kj   (X = Σ A T 를 sX 에 둔 뒤 T_ii 를 곱함)
  {
    constexpr int NB = C / 16;
    float* sX = sT + C * LF;   // [C][16] 임시
    for (int q = tid; q < C * C; q += NT) { const int t = q / C, u = q % C; sT[t * LF + u] = (t == u) ? 1.f : 0.f; }
    __syncthreads();
    if (tid < C) {
      const int bi = tid / 16, u = tid % 16, o = bi * 16;
      for (int t = u + 1; t < 16; ++t) {
        float a = 0.f;
        for (int r = u; r < t; ++r) a = a + sA[(o + t) * LF + o + r] * sT[(o + r) * LF + o + u];
        sT[(o + t) * LF + o + u] = -a;
      }
    }
    __syncthreads();
    for (int d = 1; d < NB; ++d) {
      for (int q = tid; q < (NB - d) * 256; q += NT) {
        const int j = q / 256, i = j + d, t = (q % 256) / 16, u = q % 16;
        float a = 0.f;
        for (int k = j; k < i; ++k)
          for (int r = 0; r < 16; ++r) a = a + sA[(i * 16 + t) * LF + k * 16 + r] * sT[(k * 16 + r) * LF + j * 16 + u];
        sX[(j * 16 + t) * 16 + u] = a;
      }
      __syncthreads();
      for (int q = tid; q < (NB - d) * 256; q += NT) {
        const int j = q / 256, i = j + d, t = (q % 256) / 16, u = q % 16;
        float a = 0.f;
        for (int r = 0; r <= t; ++r) a = a + sT[(i * 16 + t) * LF + i * 16 + r] * sX[(j * 16 + r) * 16 + u];
        sT[(i * 16 + t) * LF + j * 16 + u] = -a;
      }
      __syncthreads();
    }
  }
  for (int q = tid; q < C * C; q += NT) {
    const int t = q / C, s = q % C;
    const float tv = sT[t * LF + s];
    L.T[(ci * C + t) * C + s] = tv;
    sH1[t * LC + s] = f2bf(tv * sbe[s]);
    sH2[t * LC + s] = f2bf(tv * sbe[s] * sGa[s]);
  }
  __syncthreads();
  {
    mm::Acc<C, DV, NW> u;
    u.zero();
    u.mac<C, false, true>(mm::sad(sH1), LC, mm::sad(sQ), LQ);
    u.each([&](int t, int j, float& v) { L.U[(ci * C + t) * DV + j] = v; });
    mm::Acc<C, DK, NW> w;
    w.zero();
    w.mac<C, false, true>(mm::sad(sH2), LC, mm::sad(sK), LK);
    w.each([&](int t, int i, float& v) { L.Wb[(ci * C + t) * DK + i] = f2bf(v); });
  }
}
template <int DK, int DV, int C>
static size_t prep_smem() { return 2 * C * (DK + 8) + 2 * C * ((DK > DV ? DK : DV) + 8) + 4 * C * (C + 8) + 4 * C * (C + 4) + 4 * 16 * C + 4 * 3 * C; }

// ---------------------------------------------------------------------------------------------------------------------
// 앞 2: 블록 = (판, 머리, 열 조각). 덩이를 차례로; 상태 S [DK][DVP] 는 레지스터(Acc 배치)
template <int NTH>
__device__ __forceinline__ void cp_bfn(uint16_t* dst, int ldd, const uint16_t* src, int lds, int R, int Cc) {
  const int cw = Cc / 8;
  for (int q = threadIdx.x; q < R * cw; q += NTH) {
    const int r = q / cw, c = (q % cw) * 8;
    *reinterpret_cast<uint4*>(dst + r * ldd + c) = *reinterpret_cast<const uint4*>(src + (long long)r * lds + c);
  }
}
// 블록 = (판, 머리), 스레드 512(워프 16). 덩이를 차례로; 상태 S [DK][DV] 와 Δ [C][DV] 는 레지스터(Acc 배치)
constexpr int SNT = 512, SNW = 16;
template <int DK, int DV, int C>
__global__ void __launch_bounds__(SNT) dn_state_k(const DnArgs p) {
  constexpr int LK = DK + 8, LS = DV + 8, LC = C + 8;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sSb = (uint16_t*)smem;          // [DK][LS]
  uint16_t* sK = sSb + DK * LS;             // [C][LK]
  uint16_t* sW = sK + C * LK;               // Wb, 나중에 Qb
  uint16_t* sP = sW + C * LK;               // [C][LC]
  uint16_t* sD = sP + C * LC;               // bf(Δ), 나중에 bf(Δ e^{γC−γ})  [C][LS]
  float* sg = (float*)(sD + C * LS);
  float* sGa = sg + C;
  const DnL& L = p.L;
  const int bh = blockIdx.x, b = bh / p.lh, h = bh % p.lh, tid = threadIdx.x;
  mm::Acc<DK, DV, SNW> S;
  S.zero();
  if (p.S0) S.each([&](int i, int j, float& v) { v = p.S0[((long long)bh * DK + i) * DV + j]; });
  for (int c = 0; c < L.nck; ++c) {
    const long long ci = (long long)bh * L.nck + c;
    const int c0 = c * C, Cn = min(C, p.n - c0);
    cp_bfn<SNT>(sK, LK, L.Kb + ci * C * DK, DK, C, DK);
    cp_bfn<SNT>(sW, LK, L.Wb + ci * C * DK, DK, C, DK);
    cp_bfn<SNT>(sP, LC, L.Pb + ci * C * C, C, C, C);
    for (int t = tid; t < C; t += SNT) { sg[t] = L.gam[ci * C + t]; sGa[t] = expf(sg[t]); }
    S.each([&](int i, int j, float& v) {
      const uint16_t sb = f2bf(v);
      sSb[i * LS + j] = sb;
      if (p.ck) L.Sb[(ci * DK + i) * DV + j] = sb;
    });
    __syncthreads();
    const float gC = sg[C - 1], GC = sGa[C - 1];
    mm::Acc<C, DV, SNW> d;   // Δ = U − Wb·Sb
    d.zero();
    d.mac<DK, false, true>(mm::sad(sW), LK, mm::sad(sSb), LS);
    d.each([&](int t, int j, float& v) {
      v = L.U[(ci * C + t) * DV + j] - v;
      if (p.ck) L.Dl[(ci * C + t) * DV + j] = v;
      sD[t * LS + j] = f2bf(v);
    });
    __syncthreads();
    cp_bfn<SNT>(sW, LK, L.Qb + ci * C * DK, DK, C, DK);
    __syncthreads();
    {   // O = Γ (Qb Sb) + Pb Δb
      mm::Acc<C, DV, SNW> o;
      o.zero();
      o.mac<DK, false, true>(mm::sad(sW), LK, mm::sad(sSb), LS);
      o.each([&](int t, int, float& v) { v = v * sGa[t]; });
      o.mac<C, false, true>(mm::sad(sP), LC, mm::sad(sD), LS);
      o.each([&](int t, int j, float& v) {
        if (t < Cn) p.O[((long long)b * p.n + c0 + t) * p.lh * DV + h * DV + j] = v;
      });
    }
    __syncthreads();
    d.each([&](int t, int j, float& v) { sD[t * LS + j] = f2bf(v * expf(gC - sg[t])); });
    __syncthreads();
    // S ← Γ_C S + Kbᵀ Δ̂b
    S.each([&](int, int, float& v) { v = v * GC; });
    S.mac<C, true, true>(mm::sad(sK), LK, mm::sad(sD), LS);
    __syncthreads();
  }
  if (p.S1) S.each([&](int i, int j, float& v) { p.S1[((long long)bh * DK + i) * DV + j] = v; });
}
template <int DK, int DV, int C>
static size_t state_smem() { return 2 * (DK * (DV + 8) + 2 * C * (DK + 8) + C * (C + 8) + C * (DV + 8)) + 4 * 2 * C; }

// ---------------------------------------------------------------------------------------------------------------------
// 뒤 1: 블록 = (판, 머리), 스레드 512, 덩이를 거꾸로. dS [DK][DV] 는 레지스터.
//   dSb[ci] = bf(dS)(덩이 뒤 상태의 기울기), dterm[ci] = Σ dS·Sb, dΔ̂ = Kb bf(dS), dΔ = e^{γC−γ} dΔ̂ + Pbᵀ bf(dO), ρ_t = Σ_j dΔ̂·Δ e^{γC−γ}
//   dS ← Γ_C dS + Qbᵀ bf(Γ dO) − Wbᵀ bf(dΔ)
template <int DK, int DV, int C>
__global__ void __launch_bounds__(SNT) dn_dstate_k(const DnArgs p) {
  constexpr int LK = DK + 8, LS = DV + 8, LC = C + 8;
  using DA = mm::Acc<C, DV, SNW>;
  constexpr int NSL = DA::Tl::NT;   // 행 하나를 나눠 가진 조각 수
  static_assert(C <= DK, "dΔ 가 sdS 자리에 들어가야 함");
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sdS = (uint16_t*)smem;          // bf(dS) [DK][LS], 나중에 bf(−dΔ) [C][LS]
  uint16_t* sK = sdS + DK * LS;             // Kb, 나중에 Wb
  uint16_t* sQ = sK + C * LK;
  uint16_t* sP = sQ + C * LK;
  uint16_t* sdO = sP + C * LC;              // bf(dO), 나중에 bf(Γ dO)  [C][LS]
  float* sg = (float*)(sdO + C * LS);
  float* sGa = sg + C;
  float* srho = sGa + C;                    // [NSL][C]
  float* red = srho + NSL * C;              // [SNW]
  const DnL& L = p.L;
  const int bh = blockIdx.x, b = bh / p.lh, h = bh % p.lh, tid = threadIdx.x;
  mm::Acc<DK, DV, SNW> dS;
  dS.zero();
  for (int c = L.nck - 1; c >= 0; --c) {
    const long long ci = (long long)bh * L.nck + c;
    const int c0 = c * C, Cn = min(C, p.n - c0);
    float dd = 0.f;
    dS.each([&](int i, int j, float& v) {
      const uint16_t x = f2bf(v);
      L.dSb[(ci * DK + i) * DV + j] = x;
      sdS[i * LS + j] = x;
      dd = dd + v * bf2f(L.Sb[(ci * DK + i) * DV + j]);
    });
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) dd += __shfl_xor_sync(0xffffffffu, dd, o);
    if ((tid & 31) == 0) red[tid >> 5] = dd;
    cp_bfn<SNT>(sK, LK, L.Kb + ci * C * DK, DK, C, DK);
    cp_bfn<SNT>(sQ, LK, L.Qb + ci * C * DK, DK, C, DK);
    cp_bfn<SNT>(sP, LC, L.Pb + ci * C * C, C, C, C);
    for (int t = tid; t < C; t += SNT) { sg[t] = L.gam[ci * C + t]; sGa[t] = expf(sg[t]); }
    for (int q = tid; q < C * DV; q += SNT) {
      const int t = q / DV, j = q % DV;
      sdO[t * LS + j] = f2bf(t < Cn ? p.dO[((long long)b * p.n + c0 + t) * p.lh * DV + h * DV + j] : 0.f);
    }
    __syncthreads();
    if (tid == 0) { float a = 0.f; for (int w = 0; w < SNW; ++w) a = a + red[w]; L.dterm[ci] = a; }
    const float gC = sg[C - 1], GC = sGa[C - 1];
    DA d;
    d.zero();
    d.mac<DK, false, true>(mm::sad(sK), LK, mm::sad(sdS), LS);
    {   // ρ: 행마다(조각 안 열 → 쿼드 나비 → 조각마다 한 칸)
      const int l = threadIdx.x & 31, g = l >> 2, q = l & 3;
#pragma unroll
      for (int ii = 0; ii < DA::Tl::PW; ++ii) {
        if (!DA::Tl::has(ii)) continue;
        const int m0 = DA::Tl::m0(ii), n0 = DA::Tl::n0(ii);
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
          const int t = m0 + g + 8 * hr;
          const float e = expf(gC - sg[t]);
          float sacc = 0.f;
#pragma unroll
          for (int hh = 0; hh < DA::H; ++hh)
#pragma unroll
            for (int e2 = 0; e2 < 2; ++e2) {
              const int j = n0 + 8 * hh + 2 * q + e2;
              sacc = sacc + d.a[ii][hh][2 * hr + e2] * (L.Dl[(ci * C + t) * DV + j] * e);
            }
          sacc = sacc + __shfl_xor_sync(0xffffffffu, sacc, 1);
          sacc = sacc + __shfl_xor_sync(0xffffffffu, sacc, 2);
          if (q == 0) srho[(n0 / DA::Tl::TN) * C + t] = sacc;
        }
      }
    }
    d.each([&](int t, int, float& v) { v = v * expf(gC - sg[t]); });
    d.mac<C, true, true>(mm::sad(sP), LC, mm::sad(sdO), LS);
    d.each([&](int t, int j, float& v) { L.dD[(ci * C + t) * DV + j] = v; });
    __syncthreads();
    for (int t = tid; t < C; t += SNT) {
      float a = 0.f;
      for (int k = 0; k < NSL; ++k) a = a + srho[k * C + t];
      L.rho[ci * C + t] = a;
    }
    d.each([&](int t, int j, float& v) { sdS[t * LS + j] = f2bf(-v); });
    cp_bfn<SNT>(sK, LK, L.Wb + ci * C * DK, DK, C, DK);
    for (int q = tid; q < C * DV; q += SNT) {
      const int t = q / DV, j = q % DV;
      sdO[t * LS + j] = f2bf(t < Cn ? sGa[t] * p.dO[((long long)b * p.n + c0 + t) * p.lh * DV + h * DV + j] : 0.f);
    }
    __syncthreads();
    const float dec = p.bug == 3 ? 1.f : GC;
    dS.each([&](int, int, float& v) { v = v * dec; });
    dS.mac<C, true, true>(mm::sad(sQ), LK, mm::sad(sdO), LS);
    dS.mac<C, true, true>(mm::sad(sK), LK, mm::sad(sdS), LS);
    __syncthreads();
  }
}
template <int DK, int DV, int C>
static size_t dstate_smem() { return 2 * (DK * (DV + 8) + 2 * C * (DK + 8) + C * (C + 8) + C * (DV + 8)) + 4 * (2 * C + mm::Tiles<C, DV, SNW>::NT * C + SNW); }

// ---------------------------------------------------------------------------------------------------------------------
// 뒤 2: 블록 = 덩이. dv 를 JS 조각으로 한 번 돌며
//   dQ₁ = Γ (bf(dO) Sbᵀ) → dQ(=), dK_a = Δ̂b dSbᵀ → dK(=), dWb = bf(−bf(dΔ) Sbᵀ), dP = bf(dO) Δbᵀ
//   (a) dγ_t = Qb_t·dQ₁_t, QK 다시 → (b) dγ, dQKb = bf(dP∘M), (c) ρ, (d) Γ_C·dterm
template <int DK, int DV, int C>
__global__ void __launch_bounds__(NT) dn_g1_k(const DnArgs p) {
  constexpr int JS = jsw<DV>(), LK = DK + 8, LJ = JS + 8, LF = C + 4;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sK = (uint16_t*)smem;
  uint16_t* sQ = sK + C * LK;
  uint16_t* sS = sQ + C * LK;       // [DK][LJ]
  uint16_t* sdS = sS + DK * LJ;     // [DK][LJ]
  uint16_t* sA1 = sdS + DK * LJ;    // [C][LJ]  dO 조각
  uint16_t* sA2 = sA1 + C * LJ;     // Δ̂b
  uint16_t* sA3 = sA2 + C * LJ;     // dΔb
  uint16_t* sA4 = sA3 + C * LJ;     // Δb
  float* sF = (float*)(sA4 + C * LJ);   // [C][LF]
  float* sg = sF + C * LF;
  float* sGa = sg + C;
  float* sdg = sGa + C;
  float* rs = sdg + C;              // [NSL][C]
  const DnL& L = p.L;
  const long long ci = blockIdx.x;
  const int c = (int)(ci % L.nck), bh = (int)(ci / L.nck), b = bh / p.lh, h = bh % p.lh, c0 = c * C, Cn = min(C, p.n - c0), tid = threadIdx.x;
  const long long r0 = (long long)b * p.n + c0;
  cp_bf(sK, LK, L.Kb + ci * C * DK, DK, C, DK);
  cp_bf(sQ, LK, L.Qb + ci * C * DK, DK, C, DK);
  for (int t = tid; t < C; t += NT) { sg[t] = L.gam[ci * C + t]; sGa[t] = expf(sg[t]); sdg[t] = 0.f; }
  __syncthreads();
  const float gC = sg[C - 1], GC = sGa[C - 1];
  using QA = mm::Acc<C, DK, NW>;
  QA aq, ak, aw;
  mm::Acc<C, C, NW> dp;
  aq.zero(); ak.zero(); aw.zero(); dp.zero();
  for (int js = 0; js < DV; js += JS) {
    for (int q = tid; q < DK * JS / 8; q += NT) {
      const int i = q / (JS / 8), j = (q % (JS / 8)) * 8;
      *reinterpret_cast<uint4*>(sS + i * LJ + j) = *reinterpret_cast<const uint4*>(L.Sb + (ci * DK + i) * DV + js + j);
      *reinterpret_cast<uint4*>(sdS + i * LJ + j) = *reinterpret_cast<const uint4*>(L.dSb + (ci * DK + i) * DV + js + j);
    }
    for (int q = tid; q < C * JS; q += NT) {
      const int t = q / JS, j = q % JS;
      const float dl = L.Dl[(ci * C + t) * DV + js + j];
      sA1[t * LJ + j] = f2bf(t < Cn ? p.dO[(r0 + t) * p.lh * DV + h * DV + js + j] : 0.f);
      sA2[t * LJ + j] = f2bf(dl * expf(gC - sg[t]));
      sA3[t * LJ + j] = f2bf(L.dD[(ci * C + t) * DV + js + j]);
      sA4[t * LJ + j] = f2bf(dl);
    }
    __syncthreads();
    aq.mac<JS, false, false>(mm::sad(sA1), LJ, mm::sad(sS), LJ);
    ak.mac<JS, false, false>(mm::sad(sA2), LJ, mm::sad(sdS), LJ);
    aw.mac<JS, false, false>(mm::sad(sA3), LJ, mm::sad(sS), LJ);
    dp.mac<JS, false, false>(mm::sad(sA1), LJ, mm::sad(sA4), LJ);
    __syncthreads();
  }
  aq.each([&](int t, int i, float& v) {
    v = v * sGa[t];
    if (t < Cn) p.dQ[(r0 + t) * p.lh * DK + h * DK + i] = v;
  });
  ak.each([&](int t, int i, float& v) { if (t < Cn) p.dK[(r0 + t) * p.lh * DK + h * DK + i] = v; });
  aw.each([&](int t, int i, float& v) { L.dWb[(ci * C + t) * DK + i] = f2bf(-v); });
  {   // (a) 행 합: 조각 몫 → rs 칸
    constexpr int NSL = QA::Tl::NT;
    const int l = threadIdx.x & 31, g = l >> 2, q = l & 3;
#pragma unroll
    for (int ii = 0; ii < QA::Tl::PW; ++ii) {
      if (!QA::Tl::has(ii)) continue;
      const int m0 = QA::Tl::m0(ii), n0 = QA::Tl::n0(ii);
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const int t = m0 + g + 8 * hr;
        float sacc = 0.f;
#pragma unroll
        for (int hh = 0; hh < QA::H; ++hh)
#pragma unroll
          for (int e2 = 0; e2 < 2; ++e2) sacc = sacc + aq.a[ii][hh][2 * hr + e2] * bf2f(sQ[t * LK + n0 + 8 * hh + 2 * q + e2]);
        sacc = sacc + __shfl_xor_sync(0xffffffffu, sacc, 1);
        sacc = sacc + __shfl_xor_sync(0xffffffffu, sacc, 2);
        if (q == 0) rs[(n0 / QA::Tl::TN) * C + t] = sacc;
      }
    }
    dp.each([&](int t, int s2, float& v) { sF[t * LF + s2] = s2 <= t ? v : 0.f; });
    __syncthreads();
    for (int t = tid; t < C; t += NT) {
      float a = 0.f;
      for (int k = 0; k < NSL; ++k) a = a + rs[k * C + t];
      sdg[t] = sdg[t] + a;
    }
  }
  mm::Acc<C, C, NW> qk;
  qk.zero();
  qk.mac<DK, false, false>(mm::sad(sQ), LK, mm::sad(sK), LK);
  qk.each([&](int t, int s2, float& v) {
    const float M = s2 <= t ? expf(sg[t] - sg[s2]) : 0.f, d = sF[t * LF + s2];
    L.dQKb[(ci * C + t) * C + s2] = f2bf(d * M);
    sF[t * LF + s2] = d * (v * M);
  });
  __syncthreads();
  if (tid < C) {
    float rsum = 0.f, csum = 0.f;
    for (int s2 = 0; s2 < C; ++s2) rsum = rsum + sF[tid * LF + s2];
    for (int t = 0; t < C; ++t) csum = csum + sF[t * LF + tid];
    sdg[tid] = sdg[tid] + rsum - csum;
  }
  __syncthreads();
  if (tid == 0) {   // (c) ρ, (d) Γ_C Σ dS·Sb
    float rt = 0.f;
    for (int t = 0; t < C; ++t) {
      const float r = L.rho[ci * C + t];
      sdg[t] = sdg[t] - r;
      rt = rt + r;
    }
    sdg[C - 1] = sdg[C - 1] + rt + GC * L.dterm[ci];
  }
  __syncthreads();
  for (int t = tid; t < C; t += NT) L.dgam[ci * C + t] = sdg[t];
}
template <int DK, int DV, int C>
static size_t g1_smem() { constexpr int JS = jsw<DV>(); return 2 * (2 * C * (DK + 8) + 2 * DK * (JS + 8) + 4 * C * (JS + 8)) + 4 * (C * (C + 4) + 3 * C + mm::Tiles<C, DK, NW>::NT * C); }

// ---------------------------------------------------------------------------------------------------------------------
// 뒤 3: 블록 = 덩이. dQ += dQKb Kb, dK += dQKbᵀ Qb + T2bᵀ dWb, dV = T1bᵀ dΔb, dT1 = dΔb Vbᵀ, dT2 = dWb Kbᵀ → dβ·dγ 몫, dTb(전역)
template <int DK, int DV, int C>
__global__ void __launch_bounds__(NT) dn_g2_k(const DnArgs p) {
  constexpr int LK = DK + 8, LC = C + 8, LF = C + 4, LX = (DK > DV ? DK : DV) + 8;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sK = (uint16_t*)smem;
  uint16_t* sQ = sK + C * LK;           // Qb → bf(dΔ) → dWb   [C][LX]
  uint16_t* sX = sQ + C * LX;           // dQKb → Vb → T2b     [C][LX]
  uint16_t* sH = sX + C * LX;           // T1b                 [C][LC]
  float* sT = (float*)(sH + C * LC);    // [C][LF]
  float* sF = sT + C * LF;              // [C][LF]
  float* sg = sF + C * LF;
  float* sGa = sg + C;
  float* sbe = sGa + C;
  float* sdg = sbe + C;
  float* sdb = sdg + C;
  const DnL& L = p.L;
  const long long ci = blockIdx.x;
  const int c = (int)(ci % L.nck), bh = (int)(ci / L.nck), b = bh / p.lh, h = bh % p.lh, c0 = c * C, Cn = min(C, p.n - c0), tid = threadIdx.x;
  const long long r0 = (long long)b * p.n + c0;
  cp_bf(sK, LK, L.Kb + ci * C * DK, DK, C, DK);
  cp_bf(sQ, LX, L.Qb + ci * C * DK, DK, C, DK);
  cp_bf(sX, LC, L.dQKb + ci * C * C, C, C, C);
  for (int q = tid; q < C * C; q += NT) sT[(q / C) * LF + q % C] = L.T[ci * C * C + q];
  for (int t = tid; t < C; t += NT) { sg[t] = L.gam[ci * C + t]; sGa[t] = expf(sg[t]); sbe[t] = L.be[ci * C + t]; sdg[t] = 0.f; sdb[t] = 0.f; }
  __syncthreads();
  using QA = mm::Acc<C, DK, NW>;
  QA aq, ak;
  aq.zero(); ak.zero();
  aq.mac<C, false, true>(mm::sad(sX), LC, mm::sad(sK), LK);
  ak.mac<C, true, true>(mm::sad(sX), LC, mm::sad(sQ), LX);
  __syncthreads();
  // dΔb → sQ, Vb → sX, T1b → sH
  for (int q = tid; q < C * DV; q += NT) {
    const int t = q / DV, j = q % DV;
    sQ[t * LX + j] = f2bf(L.dD[(ci * C + t) * DV + j]);
    sX[t * LX + j] = f2bf(t < Cn ? p.V[(r0 + t) * p.ldv + h * DV + j] : 0.f);
  }
  for (int q = tid; q < C * C; q += NT) {
    const int t = q / C, s = q % C;
    sH[t * LC + s] = f2bf(sT[t * LF + s] * sbe[s]);
  }
  __syncthreads();
  {
    mm::Acc<C, DV, NW> dv;
    dv.zero();
    dv.mac<C, true, true>(mm::sad(sH), LC, mm::sad(sQ), LX);
    dv.each([&](int s, int j, float& v) { if (s < Cn) p.dV[(r0 + s) * p.lddv + h * DV + j] = v; });
  }
  mm::Acc<C, C, NW> d1, d2;
  d1.zero(); d2.zero();
  d1.mac<DV, false, false>(mm::sad(sQ), LX, mm::sad(sX), LX);
  __syncthreads();
  // dWb → sQ, T2b → sX
  cp_bf(sQ, LX, L.dWb + ci * C * DK, DK, C, DK);
  for (int q = tid; q < C * C; q += NT) {
    const int t = q / C, s = q % C;
    sX[t * LC + s] = f2bf(sT[t * LF + s] * sbe[s] * sGa[s]);
  }
  __syncthreads();
  ak.mac<C, true, true>(mm::sad(sX), LC, mm::sad(sQ), LX);
  d2.mac<DK, false, false>(mm::sad(sQ), LX, mm::sad(sK), LK);
  // dβ_s += Σ_t (dT1 + dT2 Γ_s) T_ts, dγ_s += Σ_t dT2 T_ts β_s Γ_s, dT = (dT1 + dT2 Γ_s) β_s (s < t)
  d1.each([&](int t, int s, float& v) { sF[t * LF + s] = s <= t ? v * sT[t * LF + s] : 0.f; });
  __syncthreads();
  if (tid < C) { float a = 0.f; for (int t = 0; t < C; ++t) a = a + sF[t * LF + tid]; sdb[tid] = a; }
  __syncthreads();
  d2.each([&](int t, int s, float& v) { sF[t * LF + s] = s <= t ? v * sT[t * LF + s] * sGa[s] : 0.f; });
  __syncthreads();
  if (tid < C) { float a = 0.f; for (int t = 0; t < C; ++t) a = a + sF[t * LF + tid]; sdb[tid] = sdb[tid] + a; sdg[tid] = a * sbe[tid]; }
  __syncthreads();
  d1.each([&](int t, int s, float& v) { sF[t * LF + s] = v; });
  __syncthreads();
  d2.each([&](int t, int s, float& v) {
    const float dt = s < t ? (sF[t * LF + s] + v * sGa[s]) * sbe[s] : 0.f;
    L.dTb[(ci * C + t) * C + s] = f2bf(dt);
  });
  aq.each([&](int t, int i, float& v) { if (t < Cn) p.dQ[(r0 + t) * p.lh * DK + h * DK + i] += v; });
  ak.each([&](int t, int i, float& v) { if (t < Cn) p.dK[(r0 + t) * p.lh * DK + h * DK + i] += v; });
  __syncthreads();
  for (int t = tid; t < C; t += NT) { L.dgam[ci * C + t] += sdg[t]; L.dbt[ci * C + t] = sdb[t]; }
}
template <int DK, int DV, int C>
static size_t g2_smem() { constexpr int LX = (DK > DV ? DK : DV) + 8; return 2 * (C * (DK + 8) + 2 * C * LX + C * (C + 8)) + 4 * (2 * C * (C + 4) + 5 * C); }

// ---------------------------------------------------------------------------------------------------------------------
// 뒤 4: 블록 = 덩이. X = Tbᵀ dTb, dA = −bf(X) Tbᵀ (s < t) → dβ_t·dγ, dKK → dK += bf(dKK + dKKᵀ) Kb; dγ 역 누적합 → dG, dβ
template <int DK, int DV, int C>
__global__ void __launch_bounds__(NT) dn_g3_k(const DnArgs p) {
  constexpr int LK = DK + 8, LC = C + 8, LF = C + 4;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* sK = (uint16_t*)smem;
  uint16_t* sTb = sK + C * LK;
  uint16_t* sdT = sTb + C * LC;         // dTb → dKKs
  uint16_t* sX = sdT + C * LC;
  float* sF = (float*)(sX + C * LC);    // [C][LF]
  float* sE = sF + C * LF;              // [C][LF]
  float* sg = sE + C * LF;
  float* sbe = sg + C;
  float* sdg = sbe + C;
  float* sdb = sdg + C;
  const DnL& L = p.L;
  const long long ci = blockIdx.x;
  const int c = (int)(ci % L.nck), bh = (int)(ci / L.nck), b = bh / p.lh, h = bh % p.lh, c0 = c * C, Cn = min(C, p.n - c0), tid = threadIdx.x;
  const long long r0 = (long long)b * p.n + c0;
  cp_bf(sK, LK, L.Kb + ci * C * DK, DK, C, DK);
  cp_bf(sdT, LC, L.dTb + ci * C * C, C, C, C);
  for (int q = tid; q < C * C; q += NT) sTb[(q / C) * LC + q % C] = f2bf(L.T[ci * C * C + q]);
  for (int t = tid; t < C; t += NT) { sg[t] = L.gam[ci * C + t]; sbe[t] = L.be[ci * C + t]; sdg[t] = L.dgam[ci * C + t]; sdb[t] = L.dbt[ci * C + t]; }
  __syncthreads();
  {
    mm::Acc<C, C, NW> x;
    x.zero();
    x.mac<C, true, true>(mm::sad(sTb), LC, mm::sad(sdT), LC);
    x.each([&](int a, int bb, float& v) { sX[a * LC + bb] = f2bf(v); });
  }
  __syncthreads();
  mm::Acc<C, C, NW> da, kk;
  da.zero(); kk.zero();
  da.mac<C, false, false>(mm::sad(sX), LC, mm::sad(sTb), LC);
  kk.mac<DK, false, false>(mm::sad(sK), LK, mm::sad(sK), LK);
  // 원소: dA = −x (s < t), M, A = β_t KK M; sF = dA·KK·M (dβ_t 행 합), sE = dA·A (dγ 행 +, 열 −)
  da.each([&](int t, int s, float& v) { sF[t * LF + s] = s < t ? -v : 0.f; });
  __syncthreads();
  kk.each([&](int t, int s, float& v) {
    const float dA = sF[t * LF + s], M = s < t ? expf(sg[t] - sg[s]) : 0.f;
    sE[t * LF + s] = dA * (sbe[t] * v * M);
    sF[t * LF + s] = dA * v * M;
  });
  __syncthreads();
  if (tid < C) {
    float rb = 0.f, re = 0.f, ce = 0.f;
    for (int s = 0; s < C; ++s) { rb = rb + sF[tid * LF + s]; re = re + sE[tid * LF + s]; }
    for (int t = 0; t < C; ++t) ce = ce + sE[t * LF + tid];
    sdb[tid] = sdb[tid] + rb;
    sdg[tid] = sdg[tid] + re - ce;
  }
  __syncthreads();
  // dKK_ts = dA β_t M (s < t) → sF, 대칭 합 → sdT(bf16)
  da.each([&](int t, int s, float& v) { sF[t * LF + s] = s < t ? -v * sbe[t] * expf(sg[t] - sg[s]) : 0.f; });
  __syncthreads();
  for (int q = tid; q < C * C; q += NT) {
    const int t = q / C, s = q % C;
    sdT[t * LC + s] = f2bf(sF[t * LF + s] + sF[s * LF + t]);
  }
  __syncthreads();
  mm::Acc<C, DK, NW> ak;
  ak.zero();
  ak.mac<C, false, true>(mm::sad(sdT), LC, mm::sad(sK), LK);
  ak.each([&](int t, int i, float& v) { if (t < Cn) p.dK[(r0 + t) * p.lh * DK + h * DK + i] += v; });
  if (tid == 0) {
    float acc = 0.f;
    for (int t = C - 1; t >= 0; --t) {
      acc = acc + sdg[t];
      if (t < Cn) {
        p.dG[(r0 + t) * p.lh + h] = p.nodecay ? 0.f : acc;
        p.dB[(r0 + t) * p.lh + h] = sdb[t];
      }
    }
  }
}
template <int DK, int DV, int C>
static size_t g3_smem() { return 2 * (C * (DK + 8) + 3 * C * (C + 8)) + 4 * (2 * C * (C + 4) + 4 * C); }

// ---------------------------------------------------------------------------------------------------------------------
template <class K_>
static void launch(K_ kern, unsigned grid, size_t sm, cudaStream_t st, const DnArgs& a) {
  if (sm > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  kern<<<grid, NT, sm, st>>>(a);
  KCK();
}
template <int DK, int DV, int C>
static void fwd_t(const DnArgs& a, cudaStream_t st) {
  launch(dn_prep_k<DK, DV, C>, (unsigned)a.L.NC, prep_smem<DK, DV, C>(), st, a);
  {
    const size_t sm = state_smem<DK, DV, C>();
    if (sm > 48 * 1024) cudaFuncSetAttribute(dn_state_k<DK, DV, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    dn_state_k<DK, DV, C><<<a.L.B * a.lh, SNT, sm, st>>>(a);
    KCK();
  }
}
template <int DK, int DV, int C>
static void bwd_t(const DnArgs& a, cudaStream_t st) {
  {
    const size_t sm = dstate_smem<DK, DV, C>();
    if (sm > 48 * 1024) cudaFuncSetAttribute(dn_dstate_k<DK, DV, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    dn_dstate_k<DK, DV, C><<<a.L.B * a.lh, SNT, sm, st>>>(a);
    KCK();
  }
  launch(dn_g1_k<DK, DV, C>, (unsigned)a.L.NC, g1_smem<DK, DV, C>(), st, a);
  launch(dn_g2_k<DK, DV, C>, (unsigned)a.L.NC, g2_smem<DK, DV, C>(), st, a);
  launch(dn_g3_k<DK, DV, C>, (unsigned)a.L.NC, g3_smem<DK, DV, C>(), st, a);
}

void dnc_fwd(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Beta, int B, int n, int lh, int dk, int dv, const float* S0,
             float* S1, bool nodecay, float* O, float* ws, bool keep, cudaStream_t st) {
  DnArgs a{};
  a.Q = Qn; a.K = Kn; a.V = V; a.G = G; a.Bt = Beta; a.S0 = S0; a.ldv = ldv; a.n = n; a.lh = lh; a.nodecay = nodecay; a.ck = keep; a.O = O; a.S1 = S1;
  a.L = dn_layout(B, n, lh, dk, dv, ws);
  if (dk == 128 && dv == 128) fwd_t<128, 128, 64>(a, st);
  else if (dk == 16 && dv == 16) fwd_t<16, 16, 16>(a, st);
  else { std::fprintf(stderr, "dnc_fwd dk %d dv %d\n", dk, dv); std::abort(); }
}
void dnc_bwd(const float* V, int ldv, const float* dO, int B, int n, int lh, int dk, int dv, bool nodecay, float* ws, float* dQn, float* dKn, float* dV, int lddv,
             float* dG, float* dBeta, int bug, cudaStream_t st) {
  DnArgs a{};
  a.V = V; a.ldv = ldv; a.dO = dO; a.n = n; a.lh = lh; a.nodecay = nodecay; a.bug = bug; a.dQ = dQn; a.dK = dKn; a.dV = dV; a.lddv = lddv; a.dG = dG; a.dB = dBeta;
  a.L = dn_layout(B, n, lh, dk, dv, ws);
  if (dk == 128 && dv == 128) bwd_t<128, 128, 64>(a, st);
  else if (dk == 16 && dv == 16) bwd_t<16, 16, 16>(a, st);
  else { std::fprintf(stderr, "dnc_bwd dk %d dv %d\n", dk, dv); std::abort(); }
}

}  // namespace tk
}  // namespace rvla
