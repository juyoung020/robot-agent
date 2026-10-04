// 칸 MLP 묶음 커널(G4 뒤 갱신 속도): 앞 = S1 → S2 → 집합, 뒤 = 집합 뒤 → dW S2 · dX S2 → dW S1. 계산 순서·끝단 식은 따로 돌던 gemm·pool 커널과
// 같아 결과 비트가 같다(학습기의 예전 길은 NET_SLOT_OLD=1). 칸 입력 줄(sin)은 Gen 이 공유 메모리로 옮긴다 — 지금은 SinBuf(모으기가 쓴 전역 sin).
// 지도 토큰에서 바로 만드는 판(sin 을 쓰지·읽지 않음)도 해 봤지만, 칸 값 계산(FP16 → 자르기 → bf16)이 두 커널의 지연에 붙어 바퀴당 약 21 ms 느렸다(README).
#pragma once
#include "gemm.cuh"

namespace net {

#define SFK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

// sin 타일(32 행 × 48, 줄 간격 ld) 채우기: 전역 버퍼에서(비동기 복사 — 부르는 쪽이 commit·wait)
struct SinBuf {
  const uint16_t* sin;
  __device__ __forceinline__ void tile(uint16_t* dst, int ld, int r0, int rlim, int tid) const {
    if (tid < 32 * (SLOT_IN / 8)) {
      const int r = tid / (SLOT_IN / 8), c = tid % (SLOT_IN / 8), gr = r0 + r;
      const bool ok = gr < rlim;
      cp16(dst + r * ld + c * 8, ok ? sin + (long long)gr * SLOT_IN + c * 8 : sin, ok);
    }
  }
};
// ---- 칸 MLP 뒤 묶음(G4 뒤 속도): 집합 뒤 → dW S2 · dX S2(⊙ELU') → dW S1 을 한 커널로 ------------------------------------------
// 블록 하나 = dW 조각 하나(kchunk 칸 행). 32 행씩 차례로: s2o·s1o·sin 을 공유 메모리로(cp.async 이중 버퍼) → dZ S2 를 그 자리에서(pool_bwd 식) →
// dW S2 += dZ2ᵀ·s1o, dZ1 = (dZ2·W2) ⊙ elu'(s1o) (bf16), dW S1 += dZ1ᵀ·sin. dZ2·dZ1 은 전역에 쓰지 않는다(검증용 dz2_out·dz1_out 만).
// 계산 순서는 따로 돌던 pool_bwd → gemm(dW S2) → gemm(dX S2) → gemm(dW S1) 과 같다: dW 는 조각 시작부터 16 행씩 차례로 mma, dX 는 k 0..63 을
// 16 씩, 끝단 식 같음 → 조각 부분합(ws)·dZ 비트가 같다. 전역 읽기·쓰기만 줄인다(행마다 dZ2·dZ1 쓰기·읽기, s1o 두 번 읽기가 없어짐).
constexpr int SB_R = 32;                       // 한 번에 다루는 칸 행(= 미니배치 행 2)
constexpr int SB_L2 = S_H + 8;                 // s2o·dZ2·dZ1·sin·W2 줄 간격(bf16) 72
constexpr int SB_L1 = 104;                     // s1o 줄 간격: 열 0..79 + dW S2 의 n 타일 96 까지 0 + 8
static_assert(!kSlotFused || SLOT_IN <= S_H, "fused slot kernels need slot input <= 64");
static_assert(kLayers[L_S2].K == 80 && kLayers[L_S1].N == S_H && kLayers[L_S2].N == S_H, "slot MLP shapes");
template <class Gen>
struct SlotBwdP {
  const float* dpool; const uint16_t* s2o; const uint16_t* s1o; Gen gen; const uint32_t* mask; const uint8_t* amax;
  const uint16_t* W2;   // S2 가중치 bf16 [64][80]
  int M, kchunk;
  float* ws2; float* ws1;   // [조각][64][80], [조각][64][48]
  uint16_t* dz2_out; uint16_t* dz1_out;   // nullptr 이 아니면 dZ S2 [M·16][64], dZ S1 [M·16][64] 도 씀(V4)
};
template <class Gen>
__global__ void __launch_bounds__(256) slot_bwd_k(const __grid_constant__ SlotBwdP<Gen> p) {
  __shared__ __align__(128) uint16_t sS2[2][SB_R * SB_L2];
  __shared__ __align__(128) uint16_t sS1[2][SB_R * SB_L1];
  __shared__ __align__(128) uint16_t sIn[2][SB_R * SB_L2];
  __shared__ __align__(128) uint16_t sD1[SB_R * SB_L2];
  __shared__ __align__(128) uint16_t sW2[S_H * SB_L2];
  __shared__ __align__(16) float sDp[2][2 * POOL_W];
  __shared__ __align__(16) uint8_t sAm[2][2 * S_H];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  const int rows = p.M * KSLOT, kbeg = blockIdx.x * p.kchunk, kend = min(rows, kbeg + p.kchunk);
  // 0 칸(늘 0): s1o 열 80..103, sin 열 48..71
  for (int q = tid; q < 2 * SB_R * 3; q += 256) {
    const int b = q / (SB_R * 3), r = (q / 3) % SB_R, c = q % 3;
    *reinterpret_cast<uint4*>(&sS1[b][r * SB_L1 + 80 + c * 8]) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&sIn[b][r * SB_L2 + SLOT_IN + c * 8]) = make_uint4(0, 0, 0, 0);
  }
  for (int q = tid; q < S_H * 8; q += 256) {   // W2[:, 0:64]
    const int k = q >> 3, c = q & 7;
    cp16(&sW2[k * SB_L2 + c * 8], p.W2 + (long long)k * 80 + c * 8, true);
  }
  auto load = [&](int b, int r0) {
    for (int q = tid; q < SB_R * 18; q += 256) {   // 행마다 s2o 8 덩이, s1o 10
      const int r = q / 18, c = q % 18, gr = r0 + r;
      const bool ok = gr < kend;
      if (c < 8) cp16(&sS2[b][r * SB_L2 + c * 8], ok ? p.s2o + (long long)gr * S_H + c * 8 : p.s2o, ok);
      else cp16(&sS1[b][r * SB_L1 + (c - 8) * 8], ok ? p.s1o + (long long)gr * 80 + (c - 8) * 8 : p.s1o, ok);
    }
    p.gen.tile(sIn[b], SB_L2, r0, kend, tid);   // sin(Gen)
    const int R0 = r0 / KSLOT;
    if (tid < 64) {   // dpool 2 행
      const int R = R0 + tid / 32, ok = R < p.M;
      cp16(&sDp[b][(tid / 32) * POOL_W + (tid % 32) * 4], ok ? p.dpool + (long long)R * POOL_W + (tid % 32) * 4 : p.dpool, ok);
    } else if (tid < 72) {   // amax 2 행
      const int q = tid - 64, R = R0 + q / 4, ok = R < p.M;
      cp16(&sAm[b][(q / 4) * S_H + (q % 4) * 16], ok ? p.amax + (long long)R * S_H + (q % 4) * 16 : p.amax, ok);
    }
  };
  float a2[1][6][4], a1[1][4][4];
#pragma unroll
  for (int ni = 0; ni < 6; ++ni)
#pragma unroll
    for (int c = 0; c < 4; ++c) a2[0][ni][c] = 0.f;
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int c = 0; c < 4; ++c) a1[0][ni][c] = 0.f;
  load(0, kbeg);
  cp_commit();
  int buf = 0;
  for (int r0 = kbeg; r0 < kend; r0 += SB_R) {
    if (r0 + SB_R < kend) load(buf ^ 1, r0 + SB_R);
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    uint16_t* S2 = sS2[buf];
    const uint16_t* S1 = sS1[buf];
    const uint16_t* In = sIn[buf];
    {   // 집합 뒤(pool_bwd_k 와 같은 식): 스레드 하나 = 행 하나의 8 칸
      const int r = tid >> 3, c0 = (tid & 7) * 8, gr = r0 + r;
      uint32_t o[4] = {0u, 0u, 0u, 0u};
      if (gr < kend) {
        const int R = gr / KSLOT, s = gr % KSLOT, lrw = R - r0 / KSLOT;
        const uint32_t mk = p.mask[R];
        if ((mk >> s) & 1u) {
          const int n = __popc(mk);
          const uint4 v = *reinterpret_cast<const uint4*>(&S2[r * SB_L2 + c0]);
          const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float d = sDp[buf][lrw * POOL_W + c0 + e] / (float)n;
            if ((int)sAm[buf][lrw * S_H + c0 + e] == s) d = d + sDp[buf][lrw * POOL_W + S_H + c0 + e];
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
    {   // dW S2 += dZ2ᵀ · s1o   (워프 4 × 2: 출력 16 행 × 입력 48 열)
      const int wm = (warp & 3) * 16, wn = (warp >> 2) * 48;
#pragma unroll
      for (int kk = 0; kk < SB_R; kk += 16) {
        uint32_t af[4];
        ldsm4t(af, S2 + (kk + lr + j1 * 8) * SB_L2 + wm + j0 * 8);
#pragma unroll
        for (int nj = 0; nj < 3; ++nj) {
          uint32_t r[4];
          ldsm4t(r, S1 + (kk + lr + j0 * 8) * SB_L1 + wn + nj * 16 + j1 * 8);
          const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
          mma_bf16(a2[0][2 * nj], af, b0);
          mma_bf16(a2[0][2 * nj + 1], af, b1);
        }
      }
    }
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
    __syncthreads();
    {   // dW S1 += dZ1ᵀ · sin   (워프 4 × 2: 출력 16 행 × 입력 32 열)
      const int wm = (warp & 3) * 16, wn = (warp >> 2) * 32;
#pragma unroll
      for (int kk = 0; kk < SB_R; kk += 16) {
        uint32_t af[4];
        ldsm4t(af, sD1 + (kk + lr + j1 * 8) * SB_L2 + wm + j0 * 8);
#pragma unroll
        for (int nj = 0; nj < 2; ++nj) {
          uint32_t r[4];
          ldsm4t(r, In + (kk + lr + j0 * 8) * SB_L2 + wn + nj * 16 + j1 * 8);
          const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
          mma_bf16(a1[0][2 * nj], af, b0);
          mma_bf16(a1[0][2 * nj + 1], af, b1);
        }
      }
    }
    __syncthreads();
    buf ^= 1;
  }
  {   // 조각 부분합(gemm EPI_SPLIT 과 같은 자리: ws + z·64·N, [행][열])
    const int wm = (warp & 3) * 16, wn2 = (warp >> 2) * 48, wn1 = (warp >> 2) * 32;
    float* W2o = p.ws2 + (long long)blockIdx.x * S_H * 80;
    float* W1o = p.ws1 + (long long)blockIdx.x * S_H * SLOT_IN;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = wm + g + h * 8;
#pragma unroll
      for (int ni = 0; ni < 6; ++ni) {
        const int c = wn2 + ni * 8 + 2 * t4;
        if (c < 80) *reinterpret_cast<float2*>(W2o + r * 80 + c) = make_float2(a2[0][ni][2 * h], a2[0][ni][2 * h + 1]);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int c = wn1 + ni * 8 + 2 * t4;
        if (c < SLOT_IN) *reinterpret_cast<float2*>(W1o + r * SLOT_IN + c) = make_float2(a1[0][ni][2 * h], a1[0][ni][2 * h + 1]);
      }
    }
  }
}
template <class Gen>
void slot_bwd_t(const float* dpool, const uint16_t* s2o, const uint16_t* s1o, const Gen& gen, const uint32_t* mask, const uint8_t* amax,
                const uint16_t* W2, int M, int kchunk, float* ws2, float* ws1, uint16_t* dz2_out, uint16_t* dz1_out, cudaStream_t st) {
  const SlotBwdP<Gen> p{dpool, s2o, s1o, gen, mask, amax, W2, M, kchunk, ws2, ws1, dz2_out, dz1_out};
  slot_bwd_k<Gen><<<(M * KSLOT + kchunk - 1) / kchunk, 256, 0, st>>>(p);
  SFK(cudaGetLastError());
}

// ---- 칸 MLP 앞 묶음(G4 뒤 속도): S1 → S2 → 집합을 한 커널로 ---------------------------------------------------------------------
// 32 칸 행(미니배치 행 2)씩: sin 을 공유 메모리로(cp.async 이중 버퍼, 블록이 타일을 차례로 돎) → S1(ELU, bf16) → S2(ELU, bf16) → 평균·최댓값.
// s1o·s2o 는 뒤 계산이 쓰므로 전역에도 쓴다(S1 의 열 0..63, S2 의 열 0..63 — gemm 끝단이 쓰는 자리 그대로). S2 가 s1o 를, 집합이 s2o 를
// 전역에서 다시 읽지 않는다. 계산 순서는 gemm(K 48 → k 0,16,32 + 0 채운 48, K 80 → 0..64 + 0 채운 80) · pool_fwd_k 와 같다 → 비트 같음
constexpr int SF_R = 32;
constexpr int SF_LI = 56;    // sin·W1 줄 간격(48 + 8)
constexpr int SF_L1 = 88;    // s1·W2 줄 간격(80 + 8). s1 열 64 = 1, 65..79 = 0
constexpr int SF_L2 = 72;
template <class Gen>
struct SlotFwdP {
  Gen gen; const uint32_t* mask; const uint16_t* W1; const uint16_t* W2;   // W1 [64][48], W2 [64][80]
  int M;
  uint16_t* s1o; uint16_t* s2o; uint16_t* x0; uint8_t* amax;
};
template <class Gen>
__global__ void __launch_bounds__(256) slot_fwd_k(const __grid_constant__ SlotFwdP<Gen> p) {
  __shared__ __align__(128) uint16_t sW1[S_H * SF_LI];
  __shared__ __align__(128) uint16_t sW2[S_H * SF_L1];
  __shared__ __align__(128) uint16_t sIn[SF_R * SF_LI];   // 한 벌: 다음 타일 sin 은 S1 이 다 읽은 뒤 S2·집합과 겹쳐 받음(공유 메모리 32 KB → SM 당 3 블록)
  __shared__ __align__(128) uint16_t sH1[SF_R * SF_L1];
  __shared__ __align__(128) uint16_t sH2[SF_R * SF_L2];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int lr = lane & 7, j0 = (lane >> 3) & 1, j1 = lane >> 4;
  const int rows = p.M * KSLOT, ntile = (rows + SF_R - 1) / SF_R;
  // 0·1 칸: W1·sin 열 48..63, W2 열 80..95, s1 열 64 = 1, 65..95 = 0
  for (int q = tid; q < SF_R * 2; q += 256) {
    const int r = q >> 1, c = q & 1;
    *reinterpret_cast<uint4*>(&sH1[r * SF_L1 + S_H + c * 8]) = c == 0 ? make_uint4(0x3f80u, 0, 0, 0) : make_uint4(0, 0, 0, 0);
  }
  for (int q = tid; q < S_H * 16; q += 256) {   // W1 6 덩이, W2 10 덩이
    const int k = q >> 4, c = q & 15;
    if (c < 6) cp16(&sW1[k * SF_LI + c * 8], p.W1 + (long long)k * SLOT_IN + c * 8, true);
    else cp16(&sW2[k * SF_L1 + (c - 6) * 8], p.W2 + (long long)k * 80 + (c - 6) * 8, true);
  }
  int t = blockIdx.x;
  if (t < ntile) p.gen.tile(sIn, SF_LI, t * SF_R, rows, tid);
  cp_commit();
  for (; t < ntile; t += gridDim.x) {
    cp_wait<0>();
    __syncthreads();
    const int r0 = t * SF_R;
    const uint16_t* In = sIn;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 16;   // 워프 2 × 4: 행 16 × 열 16
    {   // S1: k 0..47 (gemm 의 0 채운 k 48..63 단계는 0 을 더할 뿐이라 뺌 — 결과 같음, 스냅숏으로 확인)
      float ac[2][4] = {};
#pragma unroll
      for (int kk = 0; kk < SLOT_IN; kk += 16) {
        uint32_t af[4], r[4];
        ldsm4(af, In + (wm + lr + j0 * 8) * SF_LI + kk + j1 * 8);
        ldsm4(r, sW1 + (wn + lr + j1 * 8) * SF_LI + kk + j0 * 8);
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
    __syncthreads();
    if (t + (int)gridDim.x < ntile) p.gen.tile(sIn, SF_LI, (t + gridDim.x) * SF_R, rows, tid);   // 다음 타일 sin(S1 이 다 읽음)
    cp_commit();
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
    __syncthreads();
    {   // s2o 를 행마다 16 B 로
      const int r = tid >> 3, c = (tid & 7) * 8;
      if (r0 + r < rows) *reinterpret_cast<uint4*>(p.s2o + (long long)(r0 + r) * S_H + c) = *reinterpret_cast<const uint4*>(&sH2[r * SF_L2 + c]);
    }
    if (tid < (SF_R / KSLOT) * S_H) {   // 집합(pool_fwd_k 와 같은 식 — 칸마다 칸 0..15 차례로 더함): 스레드 하나 = 미니배치 행 하나의 칸 하나
      const int lrw = tid / S_H, c = tid % S_H, R = r0 / KSLOT + lrw;
      if (R < p.M) {
        const uint32_t mk = p.mask[R];
        float sum = 0.f, mx = 0.f;
        int am = 255, n = 0;
        for (int s = 0; s < KSLOT; ++s) {
          if (!((mk >> s) & 1u)) continue;
          const float y = bf2f(sH2[(lrw * KSLOT + s) * SF_L2 + c]);
          sum = sum + y;
          if (am == 255 || y > mx) { mx = y; am = s; }
          ++n;
        }
        p.x0[(long long)R * X0_W + c] = f2bf(n ? sum / (float)n : 0.f);
        p.x0[(long long)R * X0_W + S_H + c] = f2bf(n ? mx : 0.f);
        p.amax[(long long)R * S_H + c] = (uint8_t)am;
      }
    }
  }
}
template <class Gen>
void slot_fwd_t(const Gen& gen, const uint32_t* mask, const uint16_t* W1, const uint16_t* W2, int M, uint16_t* s1o, uint16_t* s2o, uint16_t* x0,
                uint8_t* amax, cudaStream_t st) {
  static int nblk = 0;   // Gen 마다 한 번: SM 수 × 상주 블록
  if (!nblk) {
    int dev = 0, sms = 0, per = 0;
    SFK(cudaGetDevice(&dev));
    SFK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    SFK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, slot_fwd_k<Gen>, 256, 0));
    nblk = sms * (per > 0 ? per : 1);
  }
  const int ntile = (M * KSLOT + SF_R - 1) / SF_R;
  const SlotFwdP<Gen> p{gen, mask, W1, W2, M, s1o, s2o, x0, amax};
  slot_fwd_k<Gen><<<ntile < nblk ? ntile : nblk, 256, 0, st>>>(p);
  SFK(cudaGetLastError());
}


#undef SFK
}  // namespace net
