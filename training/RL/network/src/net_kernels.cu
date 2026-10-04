// 신경망 커널: GEMM 실행, 집합 앞·뒤, PPO 손실(K5), dW 조각 합, 옵티마이저(K9).
// 결정성(V7): 부동소수 원자 연산을 쓰지 않는다. 모든 합은 블록 안 고정 나무 + 블록 부분합을 고정 순서로 더한다.
#include <cstdio>
#include <cstdlib>

#include "gemm.cuh"
#include "net_ops.h"
#include "slot_fused.cuh"

namespace net {

#define NCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

static inline dim3 ggrid(int M, int N, int z = 1) { return dim3((N + GBN - 1) / GBN, (M + GBM - 1) / GBM, z); }

// GEMM 고르기: gemm2_k(같은 결과, 빠름)가 기본. NET_GEMM_OLD=1 이면 예전 gemm_k(비교·측정용)
static bool gemm_old() {
  static const bool v = [] { const char* e = std::getenv("NET_GEMM_OLD"); return e && std::atoi(e) != 0; }();
  return v;
}
template <bool AT, bool BT, int EPI, int BM, int BN, int WM, int WN>
static void g2(const GemmG& g, int np, cudaStream_t st) {
  const GemmP& p = g.p[0];
  constexpr int smem = 2 * 2 * (G2Tile<BM, AT>::ELEMS + G2Tile<BN, BT>::ELEMS);
  if (smem > 48 * 1024) {   // 48 KB 넘는 동적 공유 메모리는 한 번 허락받음(스트림 일이 아니라 그래프 잡는 중에도 됨)
    static const bool ok = [] {
      NCK(cudaFuncSetAttribute(gemm2_k<AT, BT, EPI, BM, BN, WM, WN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      return true;
    }();
    (void)ok;
  }
  gemm2_k<AT, BT, EPI, BM, BN, WM, WN><<<dim3((p.N + BN - 1) / BN, (p.M + BM - 1) / BM, g.zper * np), (BM / WM) * (BN / WN) * 32, smem, st>>>(g);
}
// 문제 np 개(1 또는 2, 모양 같음), 문제마다 조각 gz 개
template <bool AT, bool BT, int EPI>
static void gemm_launch(const GemmP* ps, int np, int gz, cudaStream_t st) {
  const GemmP& p = ps[0];
  if (gemm_old()) {
    for (int k = 0; k < np; ++k) gemm_k<AT, BT, EPI><<<ggrid(p.M, p.N, gz), GNT, 0, st>>>(ps[k]);
  } else {
    const GemmG g{{ps[0], ps[np - 1]}, gz};
    if (p.N <= 16) g2<AT, BT, EPI, 128, 16, 32, 16>(g, np, st);        // 머리 출력 8
    else if (p.M <= 64) g2<AT, BT, EPI, 64, 64, 32, 32>(g, np, st);   // 칸 층 dW(출력 64 행)
    else if (p.N >= 128) g2<AT, BT, EPI, 128, 128, 64, 32>(g, np, st);   // 몸통 256·128 (128×256·256×128·64×128 타일은 더 느렸음 — README)
    else g2<AT, BT, EPI, 128, 64, 32, 32>(g, np, st);
  }
  NCK(cudaGetLastError());
}

static GemmP fwd_p(const LayerDesc& L, const uint16_t* X, int M, const uint16_t* Wb, void* out) {
  GemmP p{};
  p.A = X; p.lda = L.K; p.B = Wb; p.ldb = L.K; p.M = M; p.N = L.N; p.K = L.K; p.C = out; p.ldc = L.ldo; p.act = L.act;
  return p;
}
static GemmP dx_p(const LayerDesc& L, const uint16_t* dZ, int M, const uint16_t* Wb, const uint16_t* Xin, int Np, uint16_t* dZprev, int bug) {
  GemmP p{};
  p.A = dZ; p.lda = L.N; p.B = Wb; p.ldb = L.K; p.M = M; p.N = Np; p.K = L.N; p.C = dZprev; p.ldc = Np; p.Y = Xin; p.ldy = L.K; p.bug = bug;
  return p;
}
static GemmP dw_p(const LayerDesc& L, const uint16_t* dZ, const uint16_t* X, int M, float* ws, int kchunk) {
  GemmP p{};
  p.A = dZ; p.lda = L.N; p.B = X; p.ldb = L.K; p.M = L.N; p.N = L.K; p.K = M; p.C = ws; p.ldc = L.K; p.kchunk = kchunk;
  return p;
}
static void fwd_launch(const GemmP* ps, int np, int act, cudaStream_t st) {
  if (act == ACT_LIN) gemm_launch<false, false, EPI_F32>(ps, np, 1, st);
  else gemm_launch<false, false, EPI_ACT_BF16>(ps, np, 1, st);
}

void gemm_fwd(const LayerDesc& L, const uint16_t* X, int M, const uint16_t* Wb, void* out, cudaStream_t st) {
  const GemmP p = fwd_p(L, X, M, Wb, out);
  fwd_launch(&p, 1, L.act, st);
}
void gemm_fwd2(const LayerDesc& L, const uint16_t* XA, const uint16_t* XC, int M, const uint16_t* WA, const uint16_t* WC, void* outA, void* outC,
               cudaStream_t st) {
  const GemmP p[2] = {fwd_p(L, XA, M, WA, outA), fwd_p(L, XC, M, WC, outC)};
  fwd_launch(p, 2, L.act, st);
}

void gemm_dx_dact(const LayerDesc& L, const uint16_t* dZ, int M, const uint16_t* Wb, const uint16_t* Xin, int Np, uint16_t* dZprev, int bug,
                  cudaStream_t st) {
  const GemmP p = dx_p(L, dZ, M, Wb, Xin, Np, dZprev, bug);
  gemm_launch<false, true, EPI_DACT_BF16>(&p, 1, 1, st);
}
void gemm_dx_dact2(const LayerDesc& L, const uint16_t* dZA, const uint16_t* dZC, int M, const uint16_t* WA, const uint16_t* WC, const uint16_t* XinA,
                   const uint16_t* XinC, int Np, uint16_t* dZprevA, uint16_t* dZprevC, int bugA, cudaStream_t st) {
  const GemmP p[2] = {dx_p(L, dZA, M, WA, XinA, Np, dZprevA, bugA), dx_p(L, dZC, M, WC, XinC, Np, dZprevC, 0)};
  gemm_launch<false, true, EPI_DACT_BF16>(p, 2, 1, st);
}

void gemm_dx_pool(const LayerDesc& L, const uint16_t* dZ, int M, const uint16_t* Wb, float* dpool, bool accumulate, cudaStream_t st) {
  GemmP p{};
  p.A = dZ; p.lda = L.N; p.B = Wb; p.ldb = L.K; p.M = M; p.N = POOL_W; p.K = L.N; p.C = dpool; p.ldc = POOL_W;
  if (accumulate) gemm_launch<false, true, EPI_ACC_F32>(&p, 1, 1, st);
  else gemm_launch<false, true, EPI_F32>(&p, 1, 1, st);
}

void gemm_dw(const LayerDesc& L, const uint16_t* dZ, const uint16_t* X, int M, float* ws, int kchunk, cudaStream_t st) {
  const GemmP p = dw_p(L, dZ, X, M, ws, kchunk);
  gemm_launch<true, true, EPI_SPLIT_F32>(&p, 1, dw_splits(M, kchunk), st);
}
void gemm_dw2(const LayerDesc& L, const uint16_t* dZA, const uint16_t* dZC, const uint16_t* XA, const uint16_t* XC, int M, float* wsA, float* wsC,
              int kchunk, cudaStream_t st) {
  const GemmP p[2] = {dw_p(L, dZA, XA, M, wsA, kchunk), dw_p(L, dZC, XC, M, wsC, kchunk)};
  gemm_launch<true, true, EPI_SPLIT_F32>(p, 2, dw_splits(M, kchunk), st);
}

struct DwJobs { DwJob j[N_LAYER]; };
__global__ void dw_reduce_k(DwJobs jobs) {
  const DwJob J = jobs.j[blockIdx.y];
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= J.n) return;
  float s = 0.f;
  for (int z = 0; z < J.splits; ++z) s = s + J.ws[(long long)z * J.n + idx];
  J.g[idx] = s;
}
void dw_reduce(const DwJob* jobs, int njobs, cudaStream_t st) {
  DwJobs J{};
  int maxn = 0;
  for (int k = 0; k < njobs; ++k) { J.j[k] = jobs[k]; maxn = maxn > jobs[k].n ? maxn : jobs[k].n; }
  dw_reduce_k<<<dim3((maxn + 255) / 256, njobs), 256, 0, st>>>(J);
  NCK(cudaGetLastError());
}

// ---- 집합 ----
// 스레드 하나 = 행 하나의 칸 8 개(16 B 읽기·쓰기). 칸마다 식·순서는 예전(칸 하나 = 스레드 하나)과 같다
constexpr int PV = 8;
__global__ void pool_fwd_k(const uint16_t* s2o, const uint32_t* mask, int M, uint16_t* x0, uint8_t* amax) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)M * (S_H / PV)) return;
  const int r = (int)(q / (S_H / PV)), c0 = (int)(q % (S_H / PV)) * PV;
  const uint32_t mk = mask[r];
  float sum[PV], mx[PV];
  int am[PV];
#pragma unroll
  for (int e = 0; e < PV; ++e) { sum[e] = 0.f; mx[e] = 0.f; am[e] = 255; }
  int n = 0;
  for (int s = 0; s < KSLOT; ++s) {
    if (!((mk >> s) & 1u)) continue;
    const uint4 v = *reinterpret_cast<const uint4*>(s2o + ((long long)r * KSLOT + s) * S_H + c0);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < PV; ++e) {
      const float y = bf2f((uint16_t)(w[e >> 1] >> ((e & 1) * 16)));
      sum[e] = sum[e] + y;
      if (am[e] == 255 || y > mx[e]) { mx[e] = y; am[e] = s; }
    }
    ++n;
  }
  uint32_t om[4], ox[4], oa[2] = {0u, 0u};
#pragma unroll
  for (int e = 0; e < PV; e += 2) {
    const float m0 = n ? sum[e] / (float)n : 0.f, m1 = n ? sum[e + 1] / (float)n : 0.f;
    om[e >> 1] = (uint32_t)f2bf(m0) | ((uint32_t)f2bf(m1) << 16);
    ox[e >> 1] = (uint32_t)f2bf(n ? mx[e] : 0.f) | ((uint32_t)f2bf(n ? mx[e + 1] : 0.f) << 16);
  }
#pragma unroll
  for (int e = 0; e < PV; ++e) oa[e >> 2] |= (uint32_t)(uint8_t)am[e] << ((e & 3) * 8);
  *reinterpret_cast<uint4*>(x0 + (long long)r * X0_W + c0) = make_uint4(om[0], om[1], om[2], om[3]);
  *reinterpret_cast<uint4*>(x0 + (long long)r * X0_W + S_H + c0) = make_uint4(ox[0], ox[1], ox[2], ox[3]);
  *reinterpret_cast<uint2*>(amax + (long long)r * S_H + c0) = make_uint2(oa[0], oa[1]);
}
void pool_fwd(const uint16_t* s2o, const uint32_t* mask, int M, uint16_t* x0, uint8_t* amax, cudaStream_t st) {
  const long long n = (long long)M * (S_H / PV);
  pool_fwd_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(s2o, mask, M, x0, amax);
  NCK(cudaGetLastError());
}
__global__ void pool_bwd_k(const float* dpool, const uint16_t* s2o, const uint32_t* mask, const uint8_t* amax, int M, uint16_t* dzs2) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)M * KSLOT * (S_H / PV)) return;
  const int c0 = (int)(q % (S_H / PV)) * PV;
  const long long rs = q / (S_H / PV);
  const int r = (int)(rs / KSLOT), s = (int)(rs % KSLOT);
  const uint32_t mk = mask[r];
  uint32_t o[4] = {0u, 0u, 0u, 0u};
  if ((mk >> s) & 1u) {
    const int n = __popc(mk);
    const uint4 v = *reinterpret_cast<const uint4*>(s2o + rs * S_H + c0);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    const uint2 a2 = *reinterpret_cast<const uint2*>(amax + (long long)r * S_H + c0);
    const uint32_t aw[2] = {a2.x, a2.y};
    const float4 m0 = *reinterpret_cast<const float4*>(dpool + (long long)r * POOL_W + c0);
    const float4 m1 = *reinterpret_cast<const float4*>(dpool + (long long)r * POOL_W + c0 + 4);
    const float4 x0 = *reinterpret_cast<const float4*>(dpool + (long long)r * POOL_W + S_H + c0);
    const float4 x1 = *reinterpret_cast<const float4*>(dpool + (long long)r * POOL_W + S_H + c0 + 4);
    const float dm[PV] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w}, dx[PV] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int e = 0; e < PV; ++e) {
      float d = dm[e] / (float)n;
      if ((int)((aw[e >> 2] >> ((e & 3) * 8)) & 0xffu) == s) d = d + dx[e];
      d = d * elu_grad_from_y(bf2f((uint16_t)(w[e >> 1] >> ((e & 1) * 16))));
      o[e >> 1] |= (uint32_t)f2bf(d) << ((e & 1) * 16);
    }
  }
  *reinterpret_cast<uint4*>(dzs2 + rs * S_H + c0) = make_uint4(o[0], o[1], o[2], o[3]);
}
void pool_bwd(const float* dpool, const uint16_t* s2o, const uint32_t* mask, const uint8_t* amax, int M, uint16_t* dzs2, cudaStream_t st) {
  const long long n = (long long)M * KSLOT * (S_H / PV);
  pool_bwd_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(dpool, s2o, mask, amax, M, dzs2);
  NCK(cudaGetLastError());
}

// 칸 MLP 묶음(slot_fused.cuh) — sin 버퍼에서 읽는 판
void slot_fwd(const uint16_t* sin, const uint32_t* mask, const uint16_t* W1, const uint16_t* W2, int M, uint16_t* s1o, uint16_t* s2o, uint16_t* x0,
              uint8_t* amax, cudaStream_t st) {
  slot_fwd_t(SinBuf{sin}, mask, W1, W2, M, s1o, s2o, x0, amax, st);
}
void slot_bwd(const float* dpool, const uint16_t* s2o, const uint16_t* s1o, const uint16_t* sin, const uint32_t* mask, const uint8_t* amax,
              const uint16_t* W2, int M, int kchunk, float* ws2, float* ws1, uint16_t* dz2_out, uint16_t* dz1_out, cudaStream_t st) {
  slot_bwd_t(dpool, s2o, s1o, SinBuf{sin}, mask, amax, W2, M, kchunk, ws2, ws1, dz2_out, dz1_out, st);
}

// ---- 블록 안 고정 나무 합 ----
template <int NQ>
__device__ void block_sum(float (&q)[NQ], float* sh /* [NQ][blockDim] */) {
  const int t = threadIdx.x, nt = blockDim.x;
  for (int k = 0; k < NQ; ++k) sh[k * nt + t] = q[k];
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (t < s)
      for (int k = 0; k < NQ; ++k) sh[k * nt + t] = sh[k * nt + t] + sh[k * nt + t + s];
    __syncthreads();
  }
  for (int k = 0; k < NQ; ++k) q[k] = sh[k * nt];
}

// ---- PPO 손실 ----
__global__ void __launch_bounds__(LOSS_NT) ppo_loss_k(LossIn in, int M, const TrainState* ts, LossHyper h, uint16_t* dzA, uint16_t* dzC,
                                                      float* partial, int bug) {
  __shared__ float sh[LOSS_NQ * LOSS_NT];
  const int r = blockIdx.x * LOSS_NT + threadIdx.x;
  float q[LOSS_NQ];
  for (int k = 0; k < LOSS_NQ; ++k) q[k] = 0.f;
  if (r < M) {
    const float scale = 1.f / (float)M;
    const float A = (in.adv[r] - ts->adv_mean) / (ts->adv_std + 1e-8f);
    float z[N_ACT], inv[N_ACT], logp = 0.f;
    for (int k = 0; k < N_ACT; ++k) {
      const float ls = in.logstd[k];
      inv[k] = expf(-ls);
      z[k] = (in.act[(long long)r * N_ACT + k] - in.mean[(long long)r * N_ACT + k]) * inv[k];
      if (k < h.act_dims) logp = logp + (-0.5f * z[k] * z[k] - ls - 0.5f * kLog2Pi);
    }
    const float lr_ = logp - in.oldlogp[r];
    const float ratio = expf(lr_);
    const float s1 = ratio * A, rc = fminf(fmaxf(ratio, 1.f - h.clip), 1.f + h.clip), s2 = rc * A;
    const float pg = -fminf(s1, s2);
    const float g = (s1 <= s2) ? -ratio * A : 0.f;   // d pg / d logp
    float bl = 0.f;
    for (int k = 0; k < N_ACT; ++k) {
      const bool on = k < h.act_dims;
      // 자르기 밖 평균 벌(bound loss): 환경이 행동을 ±1 로 자르므로 |μ| > 1 이면 표본이 모두 같은 행동이 되어 PPO 기울기가 μ 를 되돌리지 못함
      const float mu = in.mean[(long long)r * N_ACT + k], ex = fabsf(mu) - 1.f;
      const float db = (on && h.bound_coef != 0.f && ex > 0.f) ? h.bound_coef * 2.f * ex * (mu > 0.f ? 1.f : -1.f) : 0.f;
      if (db != 0.f) bl = bl + h.bound_coef * ex * ex;
      const float d0 = scale * g * z[k] * inv[k];
      dzA[(long long)r * N_ACT + k] = on ? f2bf(db != 0.f ? d0 + scale * db : d0) : (uint16_t)0;
      q[k] = on ? scale * g * (z[k] * z[k] - 1.f) : 0.f;
    }
    q[12] = bl;
    const float v = in.val[(long long)r * 8], ov = in.oldv[r], R = in.ret[r];
    float vl, gv;
    if (h.vclip > 0.f) {
      const float dv = v - ov, vc = ov + fminf(fmaxf(dv, -h.vclip), h.vclip);
      const float l1 = (v - R) * (v - R), l2 = (vc - R) * (vc - R);
      vl = 0.5f * fmaxf(l1, l2);
      gv = (l1 >= l2) ? (v - R) : ((dv > -h.vclip && dv < h.vclip) ? (vc - R) : 0.f);
    } else {
      vl = 0.5f * (v - R) * (v - R);
      gv = v - R;
    }
    float dval = scale * h.vf_coef * gv;
    if (bug == 2) dval = -dval;   // 음성 대조
    dzC[(long long)r * 8] = f2bf(dval);
    for (int k = 1; k < 8; ++k) dzC[(long long)r * 8 + k] = 0;
    q[8] = pg;
    q[9] = vl;
    q[10] = (ratio - 1.f) - lr_;
    q[11] = fabsf(ratio - 1.f) > h.clip ? 1.f : 0.f;
  }
  block_sum<LOSS_NQ>(q, sh);
  if (threadIdx.x == 0)
    for (int k = 0; k < LOSS_NQ; ++k) partial[(long long)blockIdx.x * LOSS_NQ + k] = q[k];
}
void ppo_loss(const LossIn& in, int M, const TrainState* ts, const LossHyper& h, uint16_t* dzA, uint16_t* dzC, float* partial, int bug, cudaStream_t st) {
  ppo_loss_k<<<loss_blocks(M), LOSS_NT, 0, st>>>(in, M, ts, h, dzA, dzC, partial, bug);
  NCK(cudaGetLastError());
}

__global__ void __launch_bounds__(256) ppo_loss_reduce_k(const float* partial, int nblk, int M, LossHyper h, float* g_logstd, const float* logstd,
                                                         TrainState* ts) {
  __shared__ float sh[LOSS_NQ * 256];
  float q[LOSS_NQ];
  for (int k = 0; k < LOSS_NQ; ++k) {
    float s = 0.f;
    for (int b = threadIdx.x; b < nblk; b += 256) s = s + partial[(long long)b * LOSS_NQ + k];
    q[k] = s;
  }
  block_sum<LOSS_NQ>(q, sh);
  if (threadIdx.x != 0) return;
  float ent = 0.f;
  for (int k = 0; k < N_ACT; ++k) {
    const bool on = k < h.act_dims;
    g_logstd[k] = on ? q[k] - h.ent_coef : 0.f;   // 엔트로피 = Σ (log σ + ½ + ½ log 2π) → d(−c·H)/d log σ = −c
    if (on) ent = ent + logstd[k] + 0.5f + 0.5f * kLog2Pi;
  }
  const float inv = 1.f / (float)M;
  const float kl = q[10] * inv;
  ts->s_pg = ts->s_pg + q[8] * inv;
  ts->s_vl = ts->s_vl + q[9] * inv;
  ts->s_kl = ts->s_kl + kl;
  ts->s_clip = ts->s_clip + q[11] * inv;
  ts->s_ent = ts->s_ent + ent;
  ts->n_mb = ts->n_mb + 1;
  if (h.adaptive_lr) {   // rl_games 방식: 미니배치 KL 로 학습률 조절(장치 값)
    if (kl > 2.f * h.kl_target) ts->lr = fmaxf(ts->lr / 1.5f, h.lr_min);
    else if (kl < 0.5f * h.kl_target) ts->lr = fminf(ts->lr * 1.5f, h.lr_max);
  }
}
void ppo_loss_reduce(const float* partial, int nblk, int M, const LossHyper& h, float* g_logstd, const float* logstd, TrainState* ts, cudaStream_t st) {
  ppo_loss_reduce_k<<<1, 256, 0, st>>>(partial, nblk, M, h, g_logstd, logstd, ts);
  NCK(cudaGetLastError());
}

// ---- 옵티마이저 ----
__global__ void __launch_bounds__(256) gnorm_partial_k(const float* G, long long n, float* part) {
  __shared__ float sh[256];
  float q[1] = {0.f};
  for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long long)gridDim.x * 256) q[0] = q[0] + G[i] * G[i];
  block_sum<1>(q, sh);
  if (threadIdx.x == 0) part[blockIdx.x] = q[0];
}
__global__ void __launch_bounds__(256) adam_prep_k(const float* part, int nb, TrainState* ts, AdamHyper h) {
  __shared__ float sh[256];
  float q[1] = {0.f};
  for (int b = threadIdx.x; b < nb; b += 256) q[0] = q[0] + part[b];
  block_sum<1>(q, sh);
  if (threadIdx.x != 0) return;
  const float gn = sqrtf(q[0]);
  ts->gnorm = gn;
  ts->clip_coef = h.max_norm > 0.f ? fminf(1.f, h.max_norm / (gn + 1e-6f)) : 1.f;
  ts->adam_t = ts->adam_t + 1;
  ts->bc1 = 1.f - powf(h.b1, (float)ts->adam_t);
  ts->bc2 = 1.f - powf(h.b2, (float)ts->adam_t);
  ts->s_gnorm = ts->s_gnorm + gn;
}
__global__ void __launch_bounds__(256) adam_k(float* P, const float* G, float* m, float* v, uint16_t* Pb, long long n, const TrainState* ts, AdamHyper h) {
  const float c = ts->clip_coef, lr = ts->lr, bc1 = ts->bc1, bc2 = ts->bc2;
  for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long long)gridDim.x * 256) {
    const float g = G[i] * c;
    const float mi = h.b1 * m[i] + (1.f - h.b1) * g;
    const float vi = h.b2 * v[i] + (1.f - h.b2) * (g * g);
    m[i] = mi;
    v[i] = vi;
    const float p = P[i] - lr * (mi / bc1) / (sqrtf(vi / bc2) + h.eps);
    P[i] = p;
    Pb[i] = f2bf(p);
  }
}
void adam_step(float* P, const float* G, float* m, float* v, uint16_t* Pb, long long n, float* gn_partial, TrainState* ts, const AdamHyper& h,
               cudaStream_t st) {
  gnorm_partial_k<<<GN_BLOCKS, 256, 0, st>>>(G, n, gn_partial);
  adam_prep_k<<<1, 256, 0, st>>>(gn_partial, GN_BLOCKS, ts, h);
  adam_k<<<GN_BLOCKS * 2, 256, 0, st>>>(P, G, m, v, Pb, n, ts, h);
  NCK(cudaGetLastError());
}
__global__ void to_bf16_k(const float* P, uint16_t* Pb, long long n) {
  for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long long)gridDim.x * 256) Pb[i] = f2bf(P[i]);
}
void to_bf16(const float* P, uint16_t* Pb, long long n, cudaStream_t st) {
  to_bf16_k<<<GN_BLOCKS, 256, 0, st>>>(P, Pb, n);
  NCK(cudaGetLastError());
}

}  // namespace net
