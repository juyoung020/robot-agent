// 칸 MLP 묶음 커널(G4 뒤 갱신 속도, v2 넓은 칸 입력): 앞 = S1 → S2 → 집합, 뒤 = 집합 뒤 → dW S2 · dX S2 → dW S1.
// 계산 순서·끝단 식은 따로 돌던 gemm·pool 커널과 같아 결과 비트가 같다(학습기의 예전 길은 NET_SLOT_OLD=1).
// 칸 입력(SLOT_IN 304 = 숫자 33 + 이름 128 + 생김새 128 + 1 + 0)은 k 를 16 씩 0..288 차례로(gemm 이 0 으로 채운 k 304..319 단계는 0 을 더할 뿐이라 뺌).
// 칸 입력 타일(32 행 × 304)은 Gen 이 공유 메모리에 만든다:
//   SinBuf  — 전역 304 칸 줄(BC 학생: 모으기가 쓴 sin)을 cp.async 로
//   SlotTab — 줄인 칸 줄(net.h SLOT_C 40: 숫자·편향·표 행 번호)과 얼린 이름·생김새 표(bf16, L2 에 머묾)에서 펼침(PPO 학습기).
//             표 행은 칸 줄에서 33 칸(16 B 경계가 아님)에 오므로 16 B 덩이 둘을 16 비트 밀어 붙인다. 펼친 값은 net::slot_col 과 같다.
//             모으기가 304 칸 줄(행마다 608 B)을 쓰고 S1 앞·dW S1 이 두 번 읽던 전역 통행이 행마다 80 B 쓰기·읽기 둘로 준다.
// 지도 토큰에서 바로 만드는 판(sin 을 쓰지·읽지 않음)도 v1 때 해 봤지만, 칸 값 계산(FP16 → 자르기 → bf16)이 두 커널의 지연에 붙어 느렸다(README).
#pragma once
#include "gemm.cuh"

namespace net {

#define SFK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

constexpr int SK_R = 32;                 // 한 번에 다루는 칸 행(= 미니배치 행 2)
constexpr int SK_KI = SLOT_IN;           // 칸 입력 폭
constexpr int SK_LI = SLOT_IN + 8;       // 칸 입력·W1 공유 메모리 줄 간격(bf16) 312: ldmatrix 8 줄이 서로 다른 16 B 은행 묶음
constexpr int SK_NCH = SLOT_IN / 8;      // 행마다 16 B 덩이 38
static_assert(SLOT_IN % 16 == 0 && (SK_LI * 2) % 128 != 0 && (SK_LI * 2) % 16 == 0, "slot input width");

// ---- Gen: 칸 입력 타일 만들기 ----
// 전역 304 칸 줄에서(비동기 복사 — 부르는 쪽이 commit·wait)
struct SinBuf {
  const uint16_t* sin;
  static constexpr bool kStaged = false;
  struct Pre {};
  __device__ __forceinline__ void tile(uint16_t* dst, int r0, int rlim, int tid) const {
    for (int q = tid; q < SK_R * SK_NCH; q += 256) {
      const int r = q / SK_NCH, c = q % SK_NCH, gr = r0 + r;
      const bool ok = gr < rlim;
      cp16(dst + r * SK_LI + c * 8, ok ? sin + (long long)gr * SLOT_IN + c * 8 : sin, ok);
    }
  }
  __device__ __forceinline__ void stage(uint16_t*, int, int, int) const {}
  __device__ __forceinline__ void issue(Pre&, const uint16_t*, int) const {}
  __device__ __forceinline__ void finish(uint16_t*, const Pre&, const uint16_t*, int) const {}
};
// 줄인 칸 줄 + 얼린 표에서. 세 단계: stage(줄인 줄 → 공유 sC, cp.async) → issue(표 덩이를 레지스터로, 기다리지 않음) → finish(밀어 붙여 공유 sIn 에).
// 스레드 하나 = 행 하나(tid >> 3)의 덩이 5 개(tid & 7 → 덩이 5p..5p+4). 덩이 j(5 ≤ j ≤ 36) = 가상 덩이 W[j−4] 의 끝 낱말 + W[j−3] 의 앞 7 낱말,
// W[1..16] = 이름 표 행, W[17..32] = 생김새 표 행, W[33] = (편향, 0 × 7). 덩이 0..3 = 숫자 0..31, 37 = 0.
// v2 assemble 의 경계 덩이(net::slot_col 주석): 덩이 4 = (숫자 32, 0 × 7), 덩이 20 = (이름 127, 이름 다음 행 차원 0..6 — 덩이 X)
struct SlotTab {
  const uint16_t* sc; const uint16_t* name; const uint16_t* app; int n_name;
  static constexpr bool kStaged = true;
  struct Pre { uint4 w[6]; uint4 x; };
  static_assert(SLOT_IN == 304 && SLOT_C == 40 && SLOT_NAME == 33 && SLOT_APP == 161 && SLOT_BIAS == 289 && VEC_D == 128, "SlotTab layout");
  __device__ __forceinline__ void tile(uint16_t*, int, int, int) const {}
  __device__ __forceinline__ void stage(uint16_t* sC, int r0, int rlim, int tid) const {
    if (tid < SK_R * (SLOT_C / 8)) {
      const int r = tid / (SLOT_C / 8), c = tid % (SLOT_C / 8), gr = r0 + r;
      const bool ok = gr < rlim;   // 범위 밖 행은 0(= 죽은 칸 → 펼쳐도 0)
      cp16(sC + r * SLOT_C + c * 8, ok ? sc + (long long)gr * SLOT_C + c * 8 : sc, ok);
    }
  }
  __device__ __forceinline__ void issue(Pre& p, const uint16_t* sC, int tid) const {
    const uint16_t* c = sC + (tid >> 3) * SLOT_C;
    const int part = tid & 7, ni = c[SC_NAME], ai = c[SC_APP];
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      const int m = 5 * part - 4 + k;
      uint4 w = make_uint4(0u, 0u, 0u, 0u);
      if (m >= 1 && m <= 16) { if (ni) w = __ldg(reinterpret_cast<const uint4*>(name + (size_t)(ni - 1) * VEC_D) + (m - 1)); }
      else if (m >= 17 && m <= 32) { if (ai) w = __ldg(reinterpret_cast<const uint4*>(app + (size_t)(ai - 1) * VEC_D) + (m - 17)); }
      else if (m == 33) w.x = c[SC_BIAS];
      p.w[k] = w;
    }
    p.x = make_uint4(0u, 0u, 0u, 0u);
    if (part == 4 && ni && ni < n_name) p.x = __ldg(reinterpret_cast<const uint4*>(name + (size_t)ni * VEC_D));
  }
  __device__ __forceinline__ void finish(uint16_t* dst, const Pre& p, const uint16_t* sC, int tid) const {
    const int r = tid >> 3, part = tid & 7;
    const uint16_t* c = sC + r * SLOT_C;
#pragma unroll
    for (int jj = 0; jj < 5; ++jj) {
      const int j = 5 * part + jj;
      if (j >= SK_NCH) break;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (j < 4) o = *reinterpret_cast<const uint4*>(c + j * 8);
      else if (j == 4) o.x = c[SLOT_VALS - 1];
      else if (j <= 36) {
        const uint4 A = p.w[jj], B = j == 20 ? p.x : p.w[jj + 1];
        o.x = __funnelshift_r(A.w, B.x, 16);
        o.y = __funnelshift_r(B.x, B.y, 16);
        o.z = __funnelshift_r(B.y, B.z, 16);
        o.w = __funnelshift_r(B.z, B.w, 16);
      }
      *reinterpret_cast<uint4*>(dst + r * SK_LI + j * 8) = o;
    }
  }
};

#ifdef SK_PROF   // clock64 구간 재기(측정 빌드만, -DSK_PROF): 블록마다 스레드 0 이 본 구간 사이클 합
__device__ unsigned long long g_sk_prof[2][8];
#define SKP_INIT long long skp_t = clock64(); unsigned long long skp_a[8] = {};
#define SKP(i) { const long long n_ = clock64(); skp_a[i] += (unsigned long long)(n_ - skp_t); skp_t = n_; }
#define SKP_END(k) if (threadIdx.x == 0) for (int i_ = 0; i_ < 8; ++i_) atomicAdd(&g_sk_prof[k][i_], skp_a[i_]);
#else
#define SKP_INIT
#define SKP(i)
#define SKP_END(k)
#endif
template <class F>
static void sk_smem_attr(F* f, int bytes) {   // 48 KB 넘는 동적 공유 메모리(커널마다 한 번 — 그래프 잡는 중에도 됨)
  SFK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

// ---- 칸 MLP 뒤 묶음: 집합 뒤 → dW S2 · dX S2(⊙ELU') → dW S1 을 한 커널로 ------------------------------------------
// 블록 하나 = dW 조각 하나(kchunk 칸 행). 32 행씩 차례로: s2o·s1o·칸 입력을 공유 메모리로(이중 버퍼) → dZ S2 를 그 자리에서(pool_bwd 식) →
// dW S2 += dZ2ᵀ·s1o, dZ1 = (dZ2·W2) ⊙ elu'(s1o) (bf16), dW S1 += dZ1ᵀ·sin. dZ2·dZ1 은 전역에 쓰지 않는다(검증용 dz2_out·dz1_out 만).
// 계산 순서는 따로 돌던 pool_bwd → gemm(dW S2) → gemm(dX S2) → gemm(dW S1) 과 같다: dW 는 조각 시작부터 16 행씩 차례로 mma, dX 는 k 0..63 을
// 16 씩, 끝단 식 같음 → 조각 부분합(ws)·dZ 비트가 같다. dW S1 누산(64 × 304)은 워프 4 × 2: 출력 16 행 × 입력 16 열 묶음 10(뒤 반은 9) — 레지스터 80.
constexpr int SB_L2 = S_H + 8;                 // s2o·dZ2·dZ1·W2 줄 간격(bf16) 72
constexpr int SB_L1 = 104;                     // s1o 줄 간격: 열 0..79 + dW S2 의 n 타일 96 까지 0 + 8
constexpr int SB_NG = SLOT_IN / 16, SB_G0 = (SB_NG + 1) / 2;   // dW S1 열 묶음 19, 워프 반마다 10
static_assert(kLayers[L_S2].K == 80 && kLayers[L_S1].N == S_H && kLayers[L_S2].N == S_H && kLayers[L_S1].K == SLOT_IN, "slot MLP shapes");
constexpr int SB_SMEM = 2 * (S_H * SB_L2 + 2 * SK_R * SB_L2 + 2 * SK_R * SB_L1 + 2 * SK_R * SK_LI + SK_R * SB_L2 + 2 * SK_R * SLOT_C) +
                        4 * 2 * 2 * POOL_W + 2 * 2 * S_H + 4 * 2 * S_H + 16;
template <class Gen>
struct SlotBwdP {
  const float* dpool; const uint16_t* s2o; const uint16_t* s1o; Gen gen; const uint32_t* mask; const uint8_t* amax;
  const uint16_t* W2;   // S2 가중치 bf16 [64][80]
  int M, kchunk;
  float* ws2; float* ws1;   // [조각][64][80], [조각][64][SLOT_IN]
  uint16_t* dz2_out; uint16_t* dz1_out;   // nullptr 이 아니면 dZ S2 [M·16][64], dZ S1 [M·16][64] 도 씀(V4)
};
template <class Gen>
__global__ void __launch_bounds__(256, 1) slot_bwd_k(const __grid_constant__ SlotBwdP<Gen> p) {
  extern __shared__ __align__(128) uint8_t sk_smem[];
  uint16_t* sW2 = reinterpret_cast<uint16_t*>(sk_smem);       // [64][72]
  uint16_t* sS2b = sW2 + S_H * SB_L2;                          // [2][32][72]
  uint16_t* sS1b = sS2b + 2 * SK_R * SB_L2;                    // [2][32][104]
  uint16_t* sInb = sS1b + 2 * SK_R * SB_L1;                    // [2][32][312]
  uint16_t* sD1 = sInb + 2 * SK_R * SK_LI;                     // [32][72]
  uint16_t* sCb = sD1 + SK_R * SB_L2;                          // [2][32][40]
  float* sDpb = reinterpret_cast<float*>(sCb + 2 * SK_R * SLOT_C);   // [2][2][128]
  uint8_t* sAmb = reinterpret_cast<uint8_t*>(sDpb + 2 * 2 * POOL_W);  // [2][2][64]
  float* sDq = reinterpret_cast<float*>(sAmb + 2 * 2 * S_H);           // [2][64] 평균 몫 dpool/n (행마다 한 번 — 칸 16 개가 같은 값)
  uint32_t* sMk = reinterpret_cast<uint32_t*>(sDq + 2 * S_H);          // [2] 채운 칸 비트
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  const int rows = p.M * KSLOT, kbeg = blockIdx.x * p.kchunk, kend = min(rows, kbeg + p.kchunk);
  // 0 칸(늘 0): s1o 열 80..103
  for (int q = tid; q < 2 * SK_R * 3; q += 256) {
    const int b = q / (SK_R * 3), r = (q / 3) % SK_R, c = q % 3;
    *reinterpret_cast<uint4*>(&sS1b[b * SK_R * SB_L1 + r * SB_L1 + 80 + c * 8]) = make_uint4(0, 0, 0, 0);
  }
  for (int q = tid; q < S_H * 8; q += 256) {   // W2[:, 0:64]
    const int k = q >> 3, c = q & 7;
    cp16(&sW2[k * SB_L2 + c * 8], p.W2 + (long long)k * 80 + c * 8, true);
  }
  auto load = [&](int b, int r0) {
    uint16_t* S2 = sS2b + b * SK_R * SB_L2;
    uint16_t* S1 = sS1b + b * SK_R * SB_L1;
    for (int q = tid; q < SK_R * 18; q += 256) {   // 행마다 s2o 8 덩이, s1o 10
      const int r = q / 18, c = q % 18, gr = r0 + r;
      const bool ok = gr < kend;
      if (c < 8) cp16(&S2[r * SB_L2 + c * 8], ok ? p.s2o + (long long)gr * S_H + c * 8 : p.s2o, ok);
      else cp16(&S1[r * SB_L1 + (c - 8) * 8], ok ? p.s1o + (long long)gr * 80 + (c - 8) * 8 : p.s1o, ok);
    }
    if constexpr (!Gen::kStaged) p.gen.tile(sInb + b * SK_R * SK_LI, r0, kend, tid);   // (SlotTab 의 줄인 줄은 한 타일 더 앞서 stage)
    const int R0 = r0 / KSLOT;
    if (tid < 64) {   // dpool 2 행
      const int R = R0 + tid / 32, ok = R < p.M;
      cp16(&sDpb[b * 2 * POOL_W + (tid / 32) * POOL_W + (tid % 32) * 4], ok ? p.dpool + (long long)R * POOL_W + (tid % 32) * 4 : p.dpool, ok);
    } else if (tid < 72) {   // amax 2 행
      const int q = tid - 64, R = R0 + q / 4, ok = R < p.M;
      cp16(&sAmb[b * 2 * S_H + (q / 4) * S_H + (q % 4) * 16], ok ? p.amax + (long long)R * S_H + (q % 4) * 16 : p.amax, ok);
    }
  };
  float a2[6][4], a1[SB_G0][2][4];
#pragma unroll
  for (int ni = 0; ni < 6; ++ni)
#pragma unroll
    for (int c = 0; c < 4; ++c) a2[ni][c] = 0.f;
#pragma unroll
  for (int gi = 0; gi < SB_G0; ++gi)
#pragma unroll
    for (int c = 0; c < 4; ++c) a1[gi][0][c] = a1[gi][1][c] = 0.f;
  typename Gen::Pre pre;
  // 줄인 칸 줄(SlotTab)은 두 타일 앞서 받는다: 타일 k 의 sC 는 sCb[k & 1] — 바퀴 k 맨 앞 기다림(wait<1>)이 타일 k+1 의 sC 까지 보장
  load(0, kbeg);
  if constexpr (Gen::kStaged) {
    p.gen.stage(sCb, kbeg, kend, tid);
    if (kbeg + SK_R < kend) p.gen.stage(sCb + SK_R * SLOT_C, kbeg + SK_R, kend, tid);
  }
  cp_commit();
  if constexpr (Gen::kStaged) {   // 첫 타일 칸 입력 펼치기
    cp_wait<0>();
    __syncthreads();
    p.gen.issue(pre, sCb, tid);
    p.gen.finish(sInb, pre, sCb, tid);
    __syncthreads();   // 바퀴 0 맨 앞이 sCb[0] 에 타일 2 를 받기 전에 finish 가 다 읽음
  }
  const int half = warp >> 2, ng = half ? SB_NG - SB_G0 : SB_G0;
  int buf = 0;
  SKP_INIT
  for (int r0 = kbeg; r0 < kend; r0 += SK_R) {
    const bool nxt = r0 + SK_R < kend;
    if (nxt) load(buf ^ 1, r0 + SK_R);
    if constexpr (Gen::kStaged)   // 타일 k+2 의 줄인 줄 → sCb[k & 1](타일 k 의 것은 앞 바퀴 finish 가 다 씀)
      if (r0 + 2 * SK_R < kend) p.gen.stage(sCb + buf * SK_R * SLOT_C, r0 + 2 * SK_R, kend, tid);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    SKP(0)
    uint16_t* S2 = sS2b + buf * SK_R * SB_L2;
    const uint16_t* S1 = sS1b + buf * SK_R * SB_L1;
    const uint16_t* In = sInb + buf * SK_R * SK_LI;
    const float* sDp = sDpb + buf * 2 * POOL_W;
    const uint8_t* sAm = sAmb + buf * 2 * S_H;
    if constexpr (Gen::kStaged) if (nxt) p.gen.issue(pre, sCb + (buf ^ 1) * SK_R * SLOT_C, tid);   // 다음 타일 표 덩이(기다리지 않음 — 끝의 finish 까지 겹침)
    if (tid < 2 * S_H) {   // 평균 몫 dpool/n 은 행마다 한 번(칸 16 개가 같은 나눗셈 — 같은 값)
      const int lrw = tid / S_H, c = tid % S_H, R = r0 / KSLOT + lrw;
      const uint32_t mk = R < p.M ? p.mask[R] : 0u;
      sDq[tid] = sDp[lrw * POOL_W + c] / (float)__popc(mk);   // n = 0 이면 쓰이지 않음
      if (c == 0) sMk[lrw] = mk;
    }
    __syncthreads();
    {   // 집합 뒤(pool_bwd_k 와 같은 식): 스레드 하나 = 행 하나의 8 칸
      const int r = tid >> 3, c0 = (tid & 7) * 8, gr = r0 + r;
      uint32_t o[4] = {0u, 0u, 0u, 0u};
      if (gr < kend) {
        const int R = gr / KSLOT, s = gr % KSLOT, lrw = R - r0 / KSLOT;
        const uint32_t mk = sMk[lrw];
        if ((mk >> s) & 1u) {
          const uint4 v = *reinterpret_cast<const uint4*>(&S2[r * SB_L2 + c0]);
          const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float d = sDq[lrw * S_H + c0 + e];
            if ((int)sAm[lrw * S_H + c0 + e] == s) d = d + sDp[lrw * POOL_W + S_H + c0 + e];
            d = d * elu_grad_from_y(bf2f((uint16_t)(w[e >> 1] >> ((e & 1) * 16))));
            o[e >> 1] |= (uint32_t)f2bf(d) << ((e & 1) * 16);
          }
        }
        if (p.dz2_out) *reinterpret_cast<uint4*>(p.dz2_out + (long long)gr * S_H + c0) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      __syncwarp();
      *reinterpret_cast<uint4*>(&S2[r * SB_L2 + c0]) = make_uint4(o[0], o[1], o[2], o[3]);
    }
    __syncthreads();
    SKP(1)
    {   // dW S2 += dZ2ᵀ · s1o   (워프 4 × 2: 출력 16 행 × 입력 48 열)
      const int wm = (warp & 3) * 16, wn = (warp >> 2) * 48;
#pragma unroll
      for (int kk = 0; kk < SK_R; kk += 16) {
        uint32_t af[4];
        ldsm4t(af, S2 + (kk + lr + j1 * 8) * SB_L2 + wm + j0 * 8);
#pragma unroll
        for (int nj = 0; nj < 3; ++nj) {
          uint32_t r[4];
          ldsm4t(r, S1 + (kk + lr + j0 * 8) * SB_L1 + wn + nj * 16 + j1 * 8);
          const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
          mma_bf16(a2[2 * nj], af, b0);
          mma_bf16(a2[2 * nj + 1], af, b1);
        }
      }
    }
    SKP(2)
    {   // dZ1 = (dZ2 · W2) ⊙ elu'(s1o)   (워프 2 × 4: 행 16 × 열 16)
      const int wm = (warp & 1) * 16, wn = (warp >> 1) * 16;
      float ax[2][4];
#pragma unroll
      for (int ni = 0; ni < 2; ++ni)
#pragma unroll
        for (int c = 0; c < 4; ++c) ax[ni][c] = 0.f;
#pragma unroll
      for (int kk = 0; kk < S_H; kk += 16) {
        uint32_t af[4], r[4];
        ldsm4(af, S2 + (wm + lr + j0 * 8) * SB_L2 + kk + j1 * 8);
        ldsm4t(r, sW2 + (kk + lr + j0 * 8) * SB_L2 + wn + j1 * 8);
        const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
        mma_bf16(ax[0], af, b0);
        mma_bf16(ax[1], af, b1);
      }
#pragma unroll
      for (int ni = 0; ni < 2; ++ni)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = wm + g + h * 8, c = wn + ni * 8 + 2 * t4;
          const uint32_t yy = *reinterpret_cast<const uint32_t*>(&S1[r * SB_L1 + c]);
          const float x0 = ax[ni][2 * h] * elu_grad_from_y(bf2f((uint16_t)(yy & 0xffffu)));
          const float x1 = ax[ni][2 * h + 1] * elu_grad_from_y(bf2f((uint16_t)(yy >> 16)));
          const uint32_t o = (uint32_t)f2bf(x0) | ((uint32_t)f2bf(x1) << 16);
          *reinterpret_cast<uint32_t*>(&sD1[r * SB_L2 + c]) = o;
          if (p.dz1_out && r0 + r < kend) *reinterpret_cast<uint32_t*>(p.dz1_out + (long long)(r0 + r) * S_H + c) = o;
        }
    }
    SKP(3)
    __syncthreads();
    SKP(4)
    {   // dW S1 += dZ1ᵀ · sin   (워프 4 × 2: 출력 16 행 × 입력 열 묶음 10 / 9)
      const int wm = (warp & 3) * 16;
#pragma unroll
      for (int kk = 0; kk < SK_R; kk += 16) {
        uint32_t af[4];
        ldsm4t(af, sD1 + (kk + lr + j1 * 8) * SB_L2 + wm + j0 * 8);
#pragma unroll
        for (int gi = 0; gi < SB_G0; ++gi) {
          if (gi < ng) {
            uint32_t r[4];
            ldsm4t(r, In + (kk + lr + j0 * 8) * SK_LI + (half * SB_G0 + gi) * 16 + j1 * 8);
            const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
            mma_bf16(a1[gi][0], af, b0);
            mma_bf16(a1[gi][1], af, b1);
          }
        }
      }
    }
    SKP(5)
    if constexpr (Gen::kStaged) if (nxt) p.gen.finish(sInb + (buf ^ 1) * SK_R * SK_LI, pre, sCb + (buf ^ 1) * SK_R * SLOT_C, tid);
    __syncthreads();
    SKP(6)
    buf ^= 1;
  }
  SKP_END(1)
  {   // 조각 부분합(gemm EPI_SPLIT 과 같은 자리: ws + z·64·N, [행][열])
    const int wm = (warp & 3) * 16, wn2 = (warp >> 2) * 48;
    float* W2o = p.ws2 + (long long)blockIdx.x * S_H * 80;
    float* W1o = p.ws1 + (long long)blockIdx.x * S_H * SLOT_IN;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = wm + g + h * 8;
#pragma unroll
      for (int ni = 0; ni < 6; ++ni) {
        const int c = wn2 + ni * 8 + 2 * t4;
        if (c < 80) *reinterpret_cast<float2*>(W2o + r * 80 + c) = make_float2(a2[ni][2 * h], a2[ni][2 * h + 1]);
      }
#pragma unroll
      for (int gi = 0; gi < SB_G0; ++gi)
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
          const int c = (half * SB_G0 + gi) * 16 + ni * 8 + 2 * t4;
          if (gi < ng) *reinterpret_cast<float2*>(W1o + r * SLOT_IN + c) = make_float2(a1[gi][ni][2 * h], a1[gi][ni][2 * h + 1]);
        }
    }
  }
}
template <class Gen>
void slot_bwd_t(const float* dpool, const uint16_t* s2o, const uint16_t* s1o, const Gen& gen, const uint32_t* mask, const uint8_t* amax,
                const uint16_t* W2, int M, int kchunk, float* ws2, float* ws1, uint16_t* dz2_out, uint16_t* dz1_out, cudaStream_t st) {
  static const bool ok = [] { sk_smem_attr(slot_bwd_k<Gen>, SB_SMEM); return true; }();
  (void)ok;
  const SlotBwdP<Gen> p{dpool, s2o, s1o, gen, mask, amax, W2, M, kchunk, ws2, ws1, dz2_out, dz1_out};
  slot_bwd_k<Gen><<<(M * KSLOT + kchunk - 1) / kchunk, 256, SB_SMEM, st>>>(p);
  SFK(cudaGetLastError());
}

// ---- 칸 MLP 앞 묶음: S1 → S2 → 집합을 한 커널로 ---------------------------------------------------------------------
// 32 칸 행(미니배치 행 2)씩, 상주 블록이 타일을 차례로 돎: 칸 입력 타일(공유 메모리) → S1(ELU, bf16) → S2(ELU, bf16) → 평균·최댓값.
// s1o·s2o 는 뒤 계산이 쓰므로 전역에도 쓴다(S1 의 열 0..63, S2 의 열 0..63 — gemm 끝단이 쓰는 자리 그대로). S2 가 s1o 를, 집합이 s2o 를
// 전역에서 다시 읽지 않는다. 계산 순서는 gemm(K 304 → k 0..288 + 0 채운 304, K 80 → 0..64 + 0 채운 80) · pool_fwd_k 와 같다 → 비트 같음.
// 다음 타일 칸 입력은 S1 이 sIn 을 다 읽은 뒤 S2·집합과 겹쳐 만든다(SinBuf: cp.async, SlotTab: 표 덩이를 레지스터로 읽어 두고 끝에 씀).
constexpr int SF_L1 = 88;    // s1·W2 줄 간격(80 + 8). s1 열 64 = 1, 65..79 = 0
constexpr int SF_L2 = 72;
constexpr int SF_SMEM = 2 * (S_H * SK_LI + S_H * SF_L1 + SK_R * SK_LI + SK_R * SF_L1 + SK_R * SF_L2 + 2 * SK_R * SLOT_C);
template <class Gen>
struct SlotFwdP {
  Gen gen; const uint32_t* mask; const uint16_t* W1; const uint16_t* W2;   // W1 [64][304], W2 [64][80]
  int M;
  uint16_t* s1o; uint16_t* s2o; uint16_t* x0; uint8_t* amax;
};
template <class Gen>
__global__ void __launch_bounds__(256, 1) slot_fwd_k(const __grid_constant__ SlotFwdP<Gen> p) {
  extern __shared__ __align__(128) uint8_t sk_smem[];
  uint16_t* sW1 = reinterpret_cast<uint16_t*>(sk_smem);   // [64][312]
  uint16_t* sW2 = sW1 + S_H * SK_LI;                       // [64][88]
  uint16_t* sIn = sW2 + S_H * SF_L1;                       // [32][312]
  uint16_t* sH1 = sIn + SK_R * SK_LI;                      // [32][88]
  uint16_t* sH2 = sH1 + SK_R * SF_L1;                      // [32][72]
  uint16_t* sCb = sH2 + SK_R * SF_L2;                      // [2][32][40]: 이 블록의 k 번째 타일 줄인 줄은 sCb[k & 1]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  const int rows = p.M * KSLOT, ntile = (rows + SK_R - 1) / SK_R, G = gridDim.x;
  // 1·0 칸: s1 열 64 = 1, 65..79 = 0
  for (int q = tid; q < SK_R * 2; q += 256) {
    const int r = q >> 1, c = q & 1;
    *reinterpret_cast<uint4*>(&sH1[r * SF_L1 + S_H + c * 8]) = c == 0 ? make_uint4(0x3f80u, 0, 0, 0) : make_uint4(0, 0, 0, 0);
  }
  for (int q = tid; q < S_H * (SK_NCH + 10); q += 256) {   // W1 38 덩이, W2 10 덩이
    const int k = q / (SK_NCH + 10), c = q % (SK_NCH + 10);
    if (c < SK_NCH) cp16(&sW1[k * SK_LI + c * 8], p.W1 + (long long)k * SLOT_IN + c * 8, true);
    else cp16(&sW2[k * SF_L1 + (c - SK_NCH) * 8], p.W2 + (long long)k * 80 + (c - SK_NCH) * 8, true);
  }
  typename Gen::Pre pre;
  int t = blockIdx.x;
  if constexpr (Gen::kStaged) {
    if (t < ntile) p.gen.stage(sCb, t * SK_R, rows, tid);
    if (t + G < ntile) p.gen.stage(sCb + SK_R * SLOT_C, (t + G) * SK_R, rows, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    if (t < ntile) { p.gen.issue(pre, sCb, tid); p.gen.finish(sIn, pre, sCb, tid); }
  } else {
    if (t < ntile) p.gen.tile(sIn, t * SK_R, rows, tid);
    cp_commit();
  }
  SKP_INIT
  for (int k = 0; t < ntile; t += G, ++k) {
    cp_wait<0>();
    __syncthreads();
    SKP(0)
    const int r0 = t * SK_R, tn = t + G;
    uint16_t* sCn = sCb + ((k + 1) & 1) * SK_R * SLOT_C;   // 다음 타일(tn)의 줄인 줄
    if constexpr (Gen::kStaged)
      if (tn < ntile) p.gen.issue(pre, sCn, tid);   // 다음 타일 표 덩이 읽기(기다리지 않음) — S1·S2·집합과 겹침
    const int prw = r0 / KSLOT + ((tid >> 6) & 1);
    const uint32_t pmk = prw < p.M ? p.mask[prw] : 0u;   // 집합이 쓸 채운 칸 비트(미리 읽음)
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 16;   // 워프 2 × 4: 행 16 × 열 16
    {   // S1: k 0..288
      float ac[2][4] = {};
#pragma unroll
      for (int kk = 0; kk < SLOT_IN; kk += 16) {
        uint32_t af[4], r[4];
        ldsm4(af, sIn + (wm + lr + j0 * 8) * SK_LI + kk + j1 * 8);
        ldsm4(r, sW1 + (wn + lr + j1 * 8) * SK_LI + kk + j0 * 8);
        const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
        mma_bf16(ac[0], af, b0);
        mma_bf16(ac[1], af, b1);
      }
#pragma unroll
      for (int ni = 0; ni < 2; ++ni)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = wm + g + h * 8, c = wn + ni * 8 + 2 * t4;
          const uint32_t o = (uint32_t)f2bf(elu(ac[ni][2 * h])) | ((uint32_t)f2bf(elu(ac[ni][2 * h + 1])) << 16);
          *reinterpret_cast<uint32_t*>(&sH1[r * SF_L1 + c]) = o;
        }
    }
    SKP(1)
    __syncthreads();
    if constexpr (Gen::kStaged) {   // 두 타일 뒤(tn + G)의 줄인 줄 → 이 타일 자리(이 타일 것은 앞 바퀴 issue·finish 가 다 씀). 맨 앞 wait 가 받음
      if (tn + G < ntile) p.gen.stage(sCb + (k & 1) * SK_R * SLOT_C, (tn + G) * SK_R, rows, tid);
      cp_commit();
    } else {
      if (tn < ntile) p.gen.tile(sIn, tn * SK_R, rows, tid);   // 다음 타일 sin(S1 이 다 읽음)
      cp_commit();
    }
    SKP(2)
    {   // s1o 열 0..63 을 행마다 16 B 로(이어 쓰기)
      const int r = tid >> 3, c = (tid & 7) * 8;
      if (r0 + r < rows) *reinterpret_cast<uint4*>(p.s1o + (long long)(r0 + r) * (S_H + 16) + c) = *reinterpret_cast<const uint4*>(&sH1[r * SF_L1 + c]);
    }
    {   // S2: k 0..79 (0 채운 k 80..95 단계는 뺌)
      float ac[2][4] = {};
#pragma unroll
      for (int kk = 0; kk < 80; kk += 16) {
        uint32_t af[4], r[4];
        ldsm4(af, sH1 + (wm + lr + j0 * 8) * SF_L1 + kk + j1 * 8);
        ldsm4(r, sW2 + (wn + lr + j1 * 8) * SF_L1 + kk + j0 * 8);
        const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
        mma_bf16(ac[0], af, b0);
        mma_bf16(ac[1], af, b1);
      }
#pragma unroll
      for (int ni = 0; ni < 2; ++ni)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = wm + g + h * 8, c = wn + ni * 8 + 2 * t4;
          const uint32_t o = (uint32_t)f2bf(elu(ac[ni][2 * h])) | ((uint32_t)f2bf(elu(ac[ni][2 * h + 1])) << 16);
          *reinterpret_cast<uint32_t*>(&sH2[r * SF_L2 + c]) = o;
        }
    }
    SKP(3)
    __syncthreads();
    {   // s2o 를 행마다 16 B 로
      const int r = tid >> 3, c = (tid & 7) * 8;
      if (r0 + r < rows) *reinterpret_cast<uint4*>(p.s2o + (long long)(r0 + r) * S_H + c) = *reinterpret_cast<const uint4*>(&sH2[r * SF_L2 + c]);
    }
    {   // 집합(pool_fwd_k 와 같은 식 — 칸마다 칸 0..15 차례로): 스레드 0..127 = 평균(미니배치 행 2 × 칸 64), 128..255 = 최댓값·그 칸
      const int lrw = (tid >> 6) & 1, c = tid & 63, R = prw;
      if (R < p.M) {
        const uint32_t mk = pmk;
        if (tid < 128) {
          float sum = 0.f;
          int n = 0;
          for (int s = 0; s < KSLOT; ++s) {
            if (!((mk >> s) & 1u)) continue;
            sum = sum + bf2f(sH2[(lrw * KSLOT + s) * SF_L2 + c]);
            ++n;
          }
          p.x0[(long long)R * X0_W + c] = f2bf(n ? sum / (float)n : 0.f);
        } else {
          float mx = 0.f;
          int am = 255;
          for (int s = 0; s < KSLOT; ++s) {
            if (!((mk >> s) & 1u)) continue;
            const float y = bf2f(sH2[(lrw * KSLOT + s) * SF_L2 + c]);
            if (am == 255 || y > mx) { mx = y; am = s; }
          }
          p.x0[(long long)R * X0_W + S_H + c] = f2bf(am != 255 ? mx : 0.f);
          p.amax[(long long)R * S_H + c] = (uint8_t)am;
        }
      }
    }
    SKP(4)
    if constexpr (Gen::kStaged)
      if (tn < ntile) p.gen.finish(sIn, pre, sCn, tid);   // S1 은 앞 장벽 전에 sIn 을 다 읽음
    SKP(5)
  }
  SKP_END(0)
}
template <class Gen>
void slot_fwd_t(const Gen& gen, const uint32_t* mask, const uint16_t* W1, const uint16_t* W2, int M, uint16_t* s1o, uint16_t* s2o, uint16_t* x0,
                uint8_t* amax, cudaStream_t st) {
  static int nblk = 0;   // Gen 마다 한 번: SM 수 × 상주 블록
  if (!nblk) {
    sk_smem_attr(slot_fwd_k<Gen>, SF_SMEM);
    int dev = 0, sms = 0, per = 0;
    SFK(cudaGetDevice(&dev));
    SFK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    SFK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, slot_fwd_k<Gen>, 256, SF_SMEM));
    nblk = sms * (per > 0 ? per : 1);
  }
  const int ntile = (M * KSLOT + SK_R - 1) / SK_R;
  const SlotFwdP<Gen> p{gen, mask, W1, W2, M, s1o, s2o, x0, amax};
  slot_fwd_k<Gen><<<ntile < nblk ? ntile : nblk, 256, SF_SMEM, st>>>(p);
  SFK(cudaGetLastError());
}

#undef SFK
}  // namespace net
