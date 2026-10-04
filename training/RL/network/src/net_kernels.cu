// 신경망 커널: GEMM 실행, 집합 앞·뒤, PPO 손실(K5), dW 조각 합, 옵티마이저(K9).
// 결정성(V7): 부동소수 원자 연산을 쓰지 않는다. 모든 합은 블록 안 고정 나무 + 블록 부분합을 고정 순서로 더한다.
#include <cstdio>
#include <cstdlib>

#include "gemm.cuh"
#include "net_ops.h"

namespace net {

#define NCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

static inline dim3 ggrid(int M, int N, int z = 1) { return dim3((N + GBN - 1) / GBN, (M + GBM - 1) / GBM, z); }

void gemm_fwd(const LayerDesc& L, const uint16_t* X, int M, const uint16_t* Wb, void* out, cudaStream_t st) {
  GemmP p{};
  p.A = X; p.lda = L.K; p.B = Wb; p.ldb = L.K; p.M = M; p.N = L.N; p.K = L.K; p.C = out; p.ldc = L.ldo; p.act = L.act;
  if (L.act == ACT_LIN) gemm_k<false, false, EPI_F32><<<ggrid(M, L.N), GNT, 0, st>>>(p);
  else gemm_k<false, false, EPI_ACT_BF16><<<ggrid(M, L.N), GNT, 0, st>>>(p);
  NCK(cudaGetLastError());
}

void gemm_dx_dact(const LayerDesc& L, const uint16_t* dZ, int M, const uint16_t* Wb, const uint16_t* Xin, int Np, uint16_t* dZprev, int bug,
                  cudaStream_t st) {
  GemmP p{};
  p.A = dZ; p.lda = L.N; p.B = Wb; p.ldb = L.K; p.M = M; p.N = Np; p.K = L.N; p.C = dZprev; p.ldc = Np; p.Y = Xin; p.ldy = L.K; p.bug = bug;
  gemm_k<false, true, EPI_DACT_BF16><<<ggrid(M, Np), GNT, 0, st>>>(p);
  NCK(cudaGetLastError());
}

void gemm_dx_pool(const LayerDesc& L, const uint16_t* dZ, int M, const uint16_t* Wb, float* dpool, bool accumulate, cudaStream_t st) {
  GemmP p{};
  p.A = dZ; p.lda = L.N; p.B = Wb; p.ldb = L.K; p.M = M; p.N = POOL_W; p.K = L.N; p.C = dpool; p.ldc = POOL_W;
  if (accumulate) gemm_k<false, true, EPI_ACC_F32><<<ggrid(M, POOL_W), GNT, 0, st>>>(p);
  else gemm_k<false, true, EPI_F32><<<ggrid(M, POOL_W), GNT, 0, st>>>(p);
  NCK(cudaGetLastError());
}

void gemm_dw(const LayerDesc& L, const uint16_t* dZ, const uint16_t* X, int M, float* ws, int kchunk, cudaStream_t st) {
  GemmP p{};
  p.A = dZ; p.lda = L.N; p.B = X; p.ldb = L.K; p.M = L.N; p.N = L.K; p.K = M; p.C = ws; p.ldc = L.K; p.kchunk = kchunk;
  gemm_k<true, true, EPI_SPLIT_F32><<<ggrid(L.N, L.K, dw_splits(M, kchunk)), GNT, 0, st>>>(p);
  NCK(cudaGetLastError());
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
__global__ void pool_fwd_k(const uint16_t* s2o, const uint32_t* mask, int M, uint16_t* x0, uint8_t* amax) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)M * S_H) return;
  const int r = (int)(q / S_H), c = (int)(q % S_H);
  const uint32_t mk = mask[r];
  float sum = 0.f, mx = 0.f;
  int am = 255, n = 0;
  for (int s = 0; s < KSLOT; ++s) {
    if (!((mk >> s) & 1u)) continue;
    const float y = bf2f(s2o[((long long)r * KSLOT + s) * S_H + c]);
    sum = sum + y;
    if (am == 255 || y > mx) { mx = y; am = s; }
    ++n;
  }
  const float mean = n ? sum / (float)n : 0.f;
  x0[(long long)r * X0_W + c] = f2bf(mean);
  x0[(long long)r * X0_W + S_H + c] = f2bf(n ? mx : 0.f);
  amax[(long long)r * S_H + c] = (uint8_t)am;
}
void pool_fwd(const uint16_t* s2o, const uint32_t* mask, int M, uint16_t* x0, uint8_t* amax, cudaStream_t st) {
  const long long n = (long long)M * S_H;
  pool_fwd_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(s2o, mask, M, x0, amax);
  NCK(cudaGetLastError());
}
__global__ void pool_bwd_k(const float* dpool, const uint16_t* s2o, const uint32_t* mask, const uint8_t* amax, int M, uint16_t* dzs2) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)M * KSLOT * S_H) return;
  const int c = (int)(q % S_H);
  const long long rs = q / S_H;
  const int r = (int)(rs / KSLOT), s = (int)(rs % KSLOT);
  const uint32_t mk = mask[r];
  float d = 0.f;
  if ((mk >> s) & 1u) {
    const int n = __popc(mk);
    d = dpool[(long long)r * POOL_W + c] / (float)n;
    if (amax[(long long)r * S_H + c] == s) d = d + dpool[(long long)r * POOL_W + S_H + c];
    d = d * elu_grad_from_y(bf2f(s2o[q]));
  }
  dzs2[q] = f2bf(d);
}
void pool_bwd(const float* dpool, const uint16_t* s2o, const uint32_t* mask, const uint8_t* amax, int M, uint16_t* dzs2, cudaStream_t st) {
  const long long n = (long long)M * KSLOT * S_H;
  pool_bwd_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(dpool, s2o, mask, amax, M, dzs2);
  NCK(cudaGetLastError());
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
    for (int k = 0; k < N_ACT; ++k) {
      const bool on = k < h.act_dims;
      dzA[(long long)r * N_ACT + k] = on ? f2bf(scale * g * z[k] * inv[k]) : (uint16_t)0;
      q[k] = on ? scale * g * (z[k] * z[k] - 1.f) : 0.f;
    }
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
