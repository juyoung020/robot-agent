// Qwen3.5 앞 계산 커널. 식은 HF modeling_qwen3_5.py 와 같다(qwen.h 머리말). FP32 계산, GEMM 입력만 bf16.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "gemm.cuh"
#include "qkern.cuh"
#include "qwen.h"
#include "tkern.cuh"

namespace rvla {
namespace qk {

using namespace net;

#define KCK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)
static unsigned nb(long long n, int t = 256) { return (unsigned)((n + t - 1) / t); }

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void embed_k(const int* ids, const uint16_t* E, int R, int H, float* X) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)R * H) return;
  const int r = (int)(q / H), c = (int)(q % H);
  X[q] = bf2f(E[(long long)ids[r] * H + c]);
}
void embed(const int* ids, const uint16_t* E, int R, int H, float* X, cudaStream_t st) {
  embed_k<<<nb((long long)R * H), 256, 0, st>>>(ids, E, R, H, X);
  KCK();
}

// 블록 = 행, 256 스레드. 합: 스레드마다 차례 합 → 워프 나비 합 → 워프 8 개 차례 합(고정 순서)
constexpr int RN = 256;
__device__ __forceinline__ float bsum(float v, float* sh) {
  v = wsum(v);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
  for (int k = 0; k < (int)(blockDim.x >> 5); ++k) s = s + sh[k];
  __syncthreads();
  return s;
}
__global__ void rmsn_k(const float* X, int H, const float* w, int norm, float eps, uint16_t* ob, float* of, float* rs) {
  __shared__ float sh[32];
  const int r = blockIdx.x;
  const float* x = X + (long long)r * H;
  float s = 0.f;
  for (int c = threadIdx.x; c < H; c += RN) s = s + x[c] * x[c];
  s = bsum(s, sh);
  const float rsd = norm ? 1.f / sqrtf(s / (float)H + eps) : 1.f;
  for (int c = threadIdx.x; c < H; c += RN) {
    const float y = norm ? x[c] * rsd * (1.f + w[c]) : x[c];
    if (ob) ob[(long long)r * H + c] = f2bf(y);
    if (of) of[(long long)r * H + c] = y;
  }
  if (threadIdx.x == 0 && rs) rs[r] = rsd;
}
void rmsnorm(const float* X, int R, int H, const float* w, bool norm, float eps, uint16_t* ob, float* of, float* rs, cudaStream_t st) {
  rmsn_k<<<R, RN, 0, st>>>(X, H, w, norm ? 1 : 0, eps, ob, of, rs);
  KCK();
}

// ---- GEMM(RL network gemm2_k, 타일 고르기는 BC tf.cu 와 같음) ----
template <bool AT, bool BT, int EPI, int BM, int BN, int WM, int WN>
static void g2(const GemmG& g, cudaStream_t st) {
  const GemmP& p = g.p[0];
  constexpr int smem = 2 * 2 * (G2Tile<BM, AT>::ELEMS + G2Tile<BN, BT>::ELEMS);
  if (smem > 48 * 1024) {
    static const bool ok = [] { cudaFuncSetAttribute(gemm2_k<AT, BT, EPI, BM, BN, WM, WN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); return true; }();
    (void)ok;
  }
  gemm2_k<AT, BT, EPI, BM, BN, WM, WN><<<dim3((p.N + BN - 1) / BN, (p.M + BM - 1) / BM, g.zper), (BM / WM) * (BN / WN) * 32, smem, st>>>(g);
}
template <bool AT, bool BT, int EPI>
void gl(const GemmP& p, int gz, cudaStream_t st) {
  const GemmG g{{p, p}, gz};
  if (p.N <= 16) g2<AT, BT, EPI, 128, 16, 32, 16>(g, st);
  else if (p.M <= 64) g2<AT, BT, EPI, 64, 64, 32, 32>(g, st);
  else if (p.N >= 128) g2<AT, BT, EPI, 128, 128, 64, 32>(g, st);
  else g2<AT, BT, EPI, 128, 64, 32, 32>(g, st);
  KCK();
}
template void gl<false, false, EPI_F32>(const GemmP&, int, cudaStream_t);
template void gl<false, false, EPI_ACC_F32>(const GemmP&, int, cudaStream_t);
template void gl<false, false, EPI_ACT_BF16>(const GemmP&, int, cudaStream_t);
template void gl<false, true, EPI_F32>(const GemmP&, int, cudaStream_t);
template void gl<false, true, EPI_ACC_F32>(const GemmP&, int, cudaStream_t);
template void gl<false, true, EPI_ACT_BF16>(const GemmP&, int, cudaStream_t);
template void gl<true, true, EPI_SPLIT_F32>(const GemmP&, int, cudaStream_t);
void gemm_f32(const uint16_t* A, int lda, int M, const uint16_t* Wb, const MT& w, float* out, int ldo, bool acc, cudaStream_t st) {
  if (M <= 8) { gemv(A, lda, M, Wb + w.off, w.N, w.K, out, ldo, acc, st); return; }   // 디코딩
  GemmP p{};
  p.A = A; p.lda = lda; p.B = Wb + w.off; p.ldb = w.K; p.M = M; p.N = w.N; p.K = w.K; p.C = out; p.ldc = ldo; p.act = ACT_LIN;
  if (acc) gl<false, false, EPI_ACC_F32>(p, 1, st);
  else gl<false, false, EPI_F32>(p, 1, st);
}

// ---- 풀 어텐션 준비: 워프 = (행, 머리). 머리 0..nq−1 = q, nq..nq+nkv−1 = k(+ v 복사) ----
constexpr int MAXHD = 256;
__global__ void full_prep_k(const float* T0, int ldT, int R, int n, int pos0, int nq, int nkv, int hd, int rot, float theta, float eps, const float* qn,
                            const float* kn, int skipnorm, float* Q, float* Kc, float* Vc, int Lc, const int* poff, const int* dpos) {
  __shared__ float sv[8][MAXHD];
  const int w = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31, wl = threadIdx.x / 32;
  const int nh = nq + nkv;
  if (w >= R * nh) return;
  if (dpos) pos0 = *dpos;
  const int r = w / nh, h = w % nh, b = r / n, t = r % n, pos = (poff ? poff[b] : pos0) + t, kpos = pos0 + t;
  const bool isq = h < nq;
  const int kh = h - nq;
  const float* src = T0 + (long long)r * ldT + (isq ? h * 2 * hd : nq * 2 * hd + kh * hd);
  const float* nw = isq ? qn : kn;
  float s = 0.f;
  for (int d = lane; d < hd; d += 32) { const float x = src[d]; sv[wl][d] = x; s = s + x * x; }
  s = wsum(s);
  const float rsd = skipnorm ? 1.f : 1.f / sqrtf(s / (float)hd + eps);
  for (int d = lane; d < hd; d += 32) sv[wl][d] = skipnorm ? sv[wl][d] : sv[wl][d] * rsd * (1.f + nw[d]);
  __syncwarp();
  const int half = rot / 2;
  float* dst = isq ? Q + (long long)r * nq * hd + h * hd : Kc + ((long long)b * Lc + kpos) * nkv * hd + kh * hd;
  for (int d = lane; d < hd; d += 32) {
    float y = sv[wl][d];
    if (d < rot) {
      const int i = d % half;
      const float inv = 1.f / powf(theta, (float)(2 * i) / (float)rot);
      const float fr = (float)pos * inv;
      const float rh = d < half ? -sv[wl][d + half] : sv[wl][d - half];
      y = y * cosf(fr) + rh * sinf(fr);
    }
    dst[d] = y;
  }
  if (!isq) {
    const float* vs = T0 + (long long)r * ldT + nq * 2 * hd + nkv * hd + kh * hd;
    float* vd = Vc + ((long long)b * Lc + kpos) * nkv * hd + kh * hd;
    for (int d = lane; d < hd; d += 32) vd[d] = vs[d];
  }
}
void full_prep(const float* T0, int ldT, int R, int B, int n, int pos0, int nq, int nkv, int hd, int rot, float theta, float eps, const float* qn,
               const float* kn, bool skipnorm, float* Q, float* Kc, float* Vc, int Lc, cudaStream_t st, const int* poff, const int* dpos) {
  (void)B;
  if (hd > MAXHD) { std::fprintf(stderr, "hd > 256\n"); std::abort(); }
  full_prep_k<<<nb((long long)R * (nq + nkv), 8), 256, 0, st>>>(T0, ldT, R, n, pos0, nq, nkv, hd, rot, theta, eps, qn, kn, skipnorm, Q, Kc, Vc, Lc, poff, dpos);
  KCK();
}

// ---- 인과 어텐션: 워프 = (행, 질의 머리), 레인마다 E = hd/32 차원, 온라인 softmax(FP32) ----
template <int E>
__global__ void attn_k(const float* Q, const float* Kc, const float* Vc, int n, int pos0, int Lc, int nq, int nkv, float scale, int R, float* O, float* lse,
                       const int* dpos) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (w >= R * nq) return;
  constexpr int hd = 32 * E;
  const int h = w % nq, r = w / nq, b = r / n, t = r % n, kh = h / (nq / nkv);
  float qv[E], acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) { qv[e] = Q[(long long)r * nq * hd + h * hd + lane * E + e]; acc[e] = 0.f; }
  const int nk = (dpos ? *dpos : pos0) + t + 1;
  float m = -INFINITY, s = 0.f;
  const long long kst = (long long)nkv * hd;
  const float* kb = Kc + (long long)b * Lc * kst + kh * hd + lane * E;
  const float* vb = Vc + (long long)b * Lc * kst + kh * hd + lane * E;
  for (int j = 0; j < nk; ++j) {
    float d = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) d = d + qv[e] * kb[j * kst + e];
    d = wsum(d) * scale;
    const float mn = fmaxf(m, d), cr = expf(m - mn), p = expf(d - mn);
    s = s * cr + p;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = acc[e] * cr + p * vb[j * kst + e];
    m = mn;
  }
  const float inv = 1.f / s;
#pragma unroll
  for (int e = 0; e < E; ++e) O[(long long)r * nq * hd + h * hd + lane * E + e] = acc[e] * inv;
  if (lane == 0 && lse) lse[(long long)r * nq + h] = m + logf(s);
}
// 디코딩 어텐션(질의 행이 적을 때): 블록 = (행, 머리), 워프 8 개가 키를 나눠(j ≡ w mod 8) 각자 온라인 softmax → 공유 메모리에서 고정 순서로 합침
template <int E>
__global__ void __launch_bounds__(256) attn_dec_k(const float* Q, const float* Kc, const float* Vc, int n, int pos0, int Lc, int nq, int nkv, float scale, float* O,
                                                  float* lse, const int* dpos) {
  constexpr int hd = 32 * E;
  __shared__ float sm_[8], ss_[8], sacc[8][hd];
  const int r = blockIdx.x, h = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, b = r / n, t = r % n, kh = h / (nq / nkv);
  const int nk = (dpos ? *dpos : pos0) + t + 1;
  float qv[E], acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) { qv[e] = Q[(long long)r * nq * hd + h * hd + lane * E + e]; acc[e] = 0.f; }
  float m = -INFINITY, s = 0.f;
  const long long kst = (long long)nkv * hd;
  const float* kb = Kc + (long long)b * Lc * kst + kh * hd + lane * E;
  const float* vb = Vc + (long long)b * Lc * kst + kh * hd + lane * E;
  for (int j = w; j < nk; j += 8) {
    float d = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) d = d + qv[e] * kb[j * kst + e];
    d = wsum(d) * scale;
    const float mn = fmaxf(m, d), cr = expf(m - mn), p = expf(d - mn);
    s = s * cr + p;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = acc[e] * cr + p * vb[j * kst + e];
    m = mn;
  }
  if (lane == 0) { sm_[w] = m; ss_[w] = s; }
#pragma unroll
  for (int e = 0; e < E; ++e) sacc[w][lane * E + e] = acc[e];
  __syncthreads();
  if (w == 0) {
    float M = -INFINITY;
    for (int k = 0; k < 8; ++k) M = fmaxf(M, sm_[k]);
    float S = 0.f, a[E];
#pragma unroll
    for (int e = 0; e < E; ++e) a[e] = 0.f;
    for (int k = 0; k < 8; ++k) {
      const float f = sm_[k] > -INFINITY ? expf(sm_[k] - M) : 0.f;
      S = S + ss_[k] * f;
#pragma unroll
      for (int e = 0; e < E; ++e) a[e] = a[e] + sacc[k][lane * E + e] * f;
    }
    const float inv = 1.f / S;
#pragma unroll
    for (int e = 0; e < E; ++e) O[(long long)r * nq * hd + h * hd + lane * E + e] = a[e] * inv;
    if (lane == 0 && lse) lse[(long long)r * nq + h] = M + logf(S);
  }
}
void attn(const float* Q, const float* Kc, const float* Vc, int B, int n, int pos0, int Lc, int nq, int nkv, int hd, float* O, float* lse, cudaStream_t st,
          const int* dpos) {
  const int R = B * n;
  const float sc = 1.f / sqrtf((float)hd);
  if (n >= 16 && !dpos && !tk::att_old()) {   // prefix: 텐서 코어 판
    tk::AttP a;
    a.Q = Q; a.ldq = nq * hd; a.K1 = Kc; a.V1 = Vc; a.ldk1 = nkv * hd; a.L1 = Lc; a.n1c = pos0 + n; a.causal = 1; a.qoff = pos0;
    a.B = B; a.n = n; a.nq = nq; a.nkv = nkv; a.hd = hd; a.scale = sc; a.O = O; a.ldo = nq * hd; a.lse = lse;
    if (tk::fa_fwd(a, st)) return;
  }
  if (n <= 4 && !tk::att_old()) {   // 디코딩: 키를 워프 8 개로 나눔
    const dim3 g(R, nq);
    switch (hd) {
      case 256: attn_dec_k<8><<<g, 256, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, O, lse, dpos); KCK(); return;
      case 128: attn_dec_k<4><<<g, 256, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, O, lse, dpos); KCK(); return;
      case 64: attn_dec_k<2><<<g, 256, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, O, lse, dpos); KCK(); return;
      case 32: attn_dec_k<1><<<g, 256, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, O, lse, dpos); KCK(); return;
      default: break;
    }
  }
  const unsigned g = nb((long long)R * nq * 32, 128);
  switch (hd) {
    case 256: attn_k<8><<<g, 128, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, R, O, lse, dpos); break;
    case 128: attn_k<4><<<g, 128, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, R, O, lse, dpos); break;
    case 64: attn_k<2><<<g, 128, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, R, O, lse, dpos); break;
    case 32: attn_k<1><<<g, 128, 0, st>>>(Q, Kc, Vc, n, pos0, Lc, nq, nkv, sc, R, O, lse, dpos); break;
    default: std::fprintf(stderr, "attn hd %d\n", hd); std::abort();
  }
  KCK();
}

__global__ void gate2_k(const float* O, const float* T0, int ldT, int R, int nq, int hd, uint16_t* A) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = nq * hd;
  if (q >= (long long)R * W) return;
  const int r = (int)(q / W), c = (int)(q % W), h = c / hd, d = c % hd;
  A[q] = f2bf(O[q] * sigm(T0[(long long)r * ldT + h * 2 * hd + hd + d]));
}
void gate(const float* O, const float* T0, int ldT, int R, int nq, int hd, uint16_t* A, cudaStream_t st) {
  gate2_k<<<nb((long long)R * nq * hd), 256, 0, st>>>(O, T0, ldT, R, nq, hd, A);
  KCK();
}

__global__ void hist_k(const float* T0, int ld, int n, int pos0, int C, float* hist, int Lmax, long long tot, const int* dpos) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= tot) return;
  if (dpos) pos0 = *dpos;
  const int r = (int)(q / C), c = (int)(q % C), b = r / n, t = r % n;
  hist[((long long)b * Lmax + pos0 + t) * C + c] = T0[(long long)r * ld + c];
}
void hist_put(const float* T0, int ld, int B, int n, int pos0, int C, float* hist, int Lmax, cudaStream_t st, const int* dpos) {
  const long long tot = (long long)B * n * C;
  hist_k<<<nb(tot), 256, 0, st>>>(T0, ld, n, pos0, C, hist, Lmax, tot, dpos);
  KCK();
}

__global__ void conv_k(const float* src, int ld, int Ls, int n, int pos0, int C, int K, const float* w, float* out, long long tot, const int* dpos) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= tot) return;
  if (dpos) pos0 = *dpos;
  const int r = (int)(q / C), c = (int)(q % C), b = r / n, t = r % n;
  float s = 0.f;
  for (int k = 0; k < K; ++k) {
    const int p = pos0 + t - (K - 1) + k;
    if (p >= 0) s = s + w[c * K + k] * src[((long long)b * Ls + p) * ld + c];
  }
  out[q] = silu(s);
}
void conv_silu(const float* src, int ld, int Ls, int B, int n, int pos0, int C, int K, const float* w, float* out, cudaStream_t st, const int* dpos) {
  const long long tot = (long long)B * n * C;
  conv_k<<<nb(tot), 256, 0, st>>>(src, ld, Ls, n, pos0, C, K, w, out, tot, dpos);
  KCK();
}

// 워프 = (행, 머리): q·k L2 정규화(ε 1e-6, rsqrt(Σx² + ε)), q /= √dk, β, g
__global__ void lin_prep_k(const float* T1, int ld1, const float* T0, int ld0, int boff, int R, int lh, int dk, const float* alog, const float* dtb,
                           int skipnorm, float* Qn, float* Kn, float* G, float* Beta) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (w >= R * lh) return;
  const int r = w / lh, h = w % lh;
  const float* q = T1 + (long long)r * ld1 + h * dk;
  const float* k = T1 + (long long)r * ld1 + lh * dk + h * dk;
  float sq = 0.f, sk = 0.f;
  for (int d = lane; d < dk; d += 32) { sq = sq + q[d] * q[d]; sk = sk + k[d] * k[d]; }
  sq = wsum(sq); sk = wsum(sk);
  const float iq = skipnorm ? 1.f : 1.f / sqrtf(sq + 1e-6f), ik = skipnorm ? 1.f : 1.f / sqrtf(sk + 1e-6f);
  const float sc = 1.f / sqrtf((float)dk);
  for (int d = lane; d < dk; d += 32) {
    Qn[(long long)r * lh * dk + h * dk + d] = q[d] * iq * sc;
    Kn[(long long)r * lh * dk + h * dk + d] = k[d] * ik;
  }
  if (lane == 0) {
    const float bb = T0[(long long)r * ld0 + boff + h], aa = T0[(long long)r * ld0 + boff + lh + h];
    const float x = aa + dtb[h];
    const float sp = x > 20.f ? x : log1pf(expf(x));
    Beta[(long long)r * lh + h] = sigm(bb);
    G[(long long)r * lh + h] = -expf(alog[h]) * sp;
  }
}
void lin_prep(const float* T1, int ld1, const float* T0, int ld0, int boff, int R, int lh, int dk, const float* alog, const float* dtb, bool skipnorm,
              float* Qn, float* Kn, float* G, float* Beta, cudaStream_t st) {
  lin_prep_k<<<nb((long long)R * lh * 32, 128), 128, 0, st>>>(T1, ld1, T0, ld0, boff, R, lh, dk, alog, dtb, skipnorm, Qn, Kn, G, Beta);
  KCK();
}

// 재귀(블록 = (판, 머리), 스레드 j = 열 j, S[:, j] 는 레지스터)
template <int DK, int DV>
__global__ void __launch_bounds__(DV) dn_fwd_k(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Bt, int n, int lh,
                                               const float* S0, float* S1, int nodecay, float* O, float* ck) {
  __shared__ float sq[DK], sk[DK];
  const int bh = blockIdx.x, b = bh / lh, h = bh % lh, j = threadIdx.x;
  float S[DK];
#pragma unroll
  for (int i = 0; i < DK; ++i) S[i] = S0 ? S0[((long long)bh * DK + i) * DV + j] : 0.f;
  // 검문점(학습 뒤 계산용, tk::deltanet_bwd 의 배치): 16 스텝마다 S_{t−1}, 열 조각 4 개(tk DNS 와 같아야 함)
  constexpr int DVS = DV / 4;
  const int nck = (n + 15) / 16;
  for (int t = 0; t < n; ++t) {
    const long long r = (long long)b * n + t;
    if (ck && t % 16 == 0) {
      float* o = ck + (((long long)(bh * 4 + j / DVS) * (nck + 16) + t / 16) * DK) * DVS + j % DVS;
#pragma unroll
      for (int i = 0; i < DK; ++i) o[(long long)i * DVS] = S[i];
    }
    for (int i = j; i < DK; i += DV) { sq[i] = Qn[r * lh * DK + h * DK + i]; sk[i] = Kn[r * lh * DK + h * DK + i]; }
    __syncthreads();
    const float a = nodecay ? 1.f : expf(G[r * lh + h]), beta = Bt[r * lh + h], v = V[r * ldv + h * DV + j];
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < DK; ++i) { S[i] = S[i] * a; m = m + S[i] * sk[i]; }
    const float d = beta * (v - m);
    float o = 0.f;
#pragma unroll
    for (int i = 0; i < DK; ++i) { S[i] = S[i] + sk[i] * d; o = o + S[i] * sq[i]; }
    O[r * lh * DV + h * DV + j] = o;
    __syncthreads();
  }
  if (S1) {
#pragma unroll
    for (int i = 0; i < DK; ++i) S1[((long long)bh * DK + i) * DV + j] = S[i];
  }
}
void deltanet(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Beta, int B, int n, int lh, int dk, int dv,
              const float* S0, float* S1, bool nodecay, float* O, cudaStream_t st, float* ck) {
  if (dk == 128 && dv == 128) dn_fwd_k<128, 128><<<B * lh, 128, 0, st>>>(Qn, Kn, V, ldv, G, Beta, n, lh, S0, S1, nodecay, O, ck);
  else if (dk == 16 && dv == 16) dn_fwd_k<16, 16><<<B * lh, 16, 0, st>>>(Qn, Kn, V, ldv, G, Beta, n, lh, S0, S1, nodecay, O, ck);
  else { std::fprintf(stderr, "deltanet dk %d dv %d\n", dk, dv); std::abort(); }
  KCK();
}

// 워프 = (행, 머리): RMSN(o)·w ⊙ SiLU(z) → bf16
__global__ void gnorm_k(const float* O, const float* Z, int ldz, int R, int lh, int dv, const float* w, float eps, uint16_t* A) {
  const int wi = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wi >= R * lh) return;
  const int r = wi / lh, h = wi % lh;
  const float* o = O + (long long)r * lh * dv + h * dv;
  float s = 0.f;
  for (int d = lane; d < dv; d += 32) s = s + o[d] * o[d];
  s = wsum(s);
  const float rsd = 1.f / sqrtf(s / (float)dv + eps);
  for (int d = lane; d < dv; d += 32) A[(long long)r * lh * dv + h * dv + d] = f2bf(w[d] * (o[d] * rsd) * silu(Z[(long long)r * ldz + h * dv + d]));
}
void gnorm(const float* O, const float* Z, int ldz, int R, int lh, int dv, const float* w, float eps, uint16_t* A, cudaStream_t st) {
  gnorm_k<<<nb((long long)R * lh * 32, 128), 128, 0, st>>>(O, Z, ldz, R, lh, dv, w, eps, A);
  KCK();
}

__global__ void swiglu_k(const float* T0, int R, int I, uint16_t* A) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)R * I) return;
  const int r = (int)(q / I), i = (int)(q % I);
  const float g = T0[(long long)r * 2 * I + i], u = T0[(long long)r * 2 * I + I + i];
  A[q] = f2bf(silu(g) * u);
}
void swiglu(const float* T0, int R, int I, uint16_t* A, cudaStream_t st) {
  swiglu_k<<<nb((long long)R * I), 256, 0, st>>>(T0, R, I, A);
  KCK();
}

__global__ void f2bf_k(const float* X, long long n, uint16_t* A) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q < n) A[q] = f2bf(X[q]);
}
void f2bf_rows(const float* X, int R, int H, uint16_t* A, cudaStream_t st) {
  f2bf_k<<<nb((long long)R * H), 256, 0, st>>>(X, (long long)R * H, A);
  KCK();
}
__global__ void last_k(const float* hid, int B, int n, int H, float* out) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)B * H) return;
  const int b = (int)(q / H), c = (int)(q % H);
  out[q] = hid[((long long)b * n + n - 1) * H + c];
}
void gather_last(const float* hid, int B, int n, int H, float* out, cudaStream_t st) {
  last_k<<<nb((long long)B * H), 256, 0, st>>>(hid, B, n, H, out);
  KCK();
}
// argmax: 1 단 = 행마다 조각 AMB 개(블록), 2 단 = 조각 최댓값 중(같으면 작은 번호)
constexpr int AMB = 64;
__device__ __forceinline__ void am_better(float& bv, int& bi, float ov, int oi) {
  if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}
__device__ __forceinline__ void am_block(float& bv, int& bi) {
  __shared__ float sv[256];
  __shared__ int si[256];
  sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      float v = sv[threadIdx.x]; int i = si[threadIdx.x];
      am_better(v, i, sv[threadIdx.x + s], si[threadIdx.x + s]);
      sv[threadIdx.x] = v; si[threadIdx.x] = i;
    }
    __syncthreads();
  }
  bv = sv[0]; bi = si[0];
}
__global__ void argmax1_k(const float* L, int V, float* pv, int* pi) {
  const int r = blockIdx.y, part = blockIdx.x, per = (V + AMB - 1) / AMB, v0 = part * per, v1 = min(V, v0 + per);
  const float* l = L + (long long)r * V;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int v = v0 + threadIdx.x; v < v1; v += 256) am_better(bv, bi, l[v], v);
  am_block(bv, bi);
  if (threadIdx.x == 0) { pv[r * AMB + part] = bv; pi[r * AMB + part] = bi; }
}
__global__ void argmax2_k(const float* pv, const int* pi, int* out) {
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  if (threadIdx.x < AMB) { bv = pv[blockIdx.x * AMB + threadIdx.x]; bi = pi[blockIdx.x * AMB + threadIdx.x]; }
  am_block(bv, bi);
  if (threadIdx.x == 0) out[blockIdx.x] = bi;
}
void argmax_rows(const float* L, int R, int V, int* out, cudaStream_t st) {
  static float* pv = nullptr;
  static int* pi = nullptr;
  static int cap = 0;
  if (R > cap) {   // 작업 칸(처음 한 번 — 그래프로 잡기 전에 같은 R 로 한 번 불릴 것)
    if (pv) { cudaFree(pv); cudaFree(pi); }
    cap = std::max(R, 64);
    cudaMalloc(&pv, sizeof(float) * cap * AMB);
    cudaMalloc(&pi, sizeof(int) * cap * AMB);
  }
  argmax1_k<<<dim3(AMB, R), 256, 0, st>>>(L, V, pv, pi);
  argmax2_k<<<R, 256, 0, st>>>(pv, pi, out);
  KCK();
}

// ---- 디코딩 GEMV: 워프 = 출력 열 하나(SW: gate 열 n 과 up 열 I + n 둘). 레인마다 k 8 개 덩이(16 B)를 K 에 걸쳐 차례로, 끝에 워프 나비 합 ----
__device__ __forceinline__ void fma8(float& a, uint4 w, uint4 x) {
  const uint32_t ww[4] = {w.x, w.y, w.z, w.w}, xx[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a = fmaf(__uint_as_float(ww[i] << 16), __uint_as_float(xx[i] << 16), a);
    a = fmaf(__uint_as_float(ww[i] & 0xffff0000u), __uint_as_float(xx[i] & 0xffff0000u), a);
  }
}
template <int M, int SW>
__global__ void __launch_bounds__(256) gemv_k(const uint16_t* __restrict__ A, int lda, const uint16_t* __restrict__ W, int N, int K, float* out, int ldo, int acc,
                                              uint16_t* outb) {
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (n >= N) return;
  float a0[M], a1[M];
#pragma unroll
  for (int m = 0; m < M; ++m) { a0[m] = 0.f; a1[m] = 0.f; }
  const uint16_t* w0 = W + (long long)n * K;
  const uint16_t* w1 = W + (long long)(n + N) * K;   // SW: up 열
#pragma unroll 4
  for (int k = lane * 8; k < K; k += 256) {
    const uint4 wa = __ldg(reinterpret_cast<const uint4*>(w0 + k));
    uint4 wb = make_uint4(0, 0, 0, 0);
    if (SW) wb = __ldg(reinterpret_cast<const uint4*>(w1 + k));
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4 x = *reinterpret_cast<const uint4*>(A + (long long)m * lda + k);
      fma8(a0[m], wa, x);
      if (SW) fma8(a1[m], wb, x);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      a0[m] += __shfl_xor_sync(0xffffffffu, a0[m], o);
      if (SW) a1[m] += __shfl_xor_sync(0xffffffffu, a1[m], o);
    }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if (SW) outb[(long long)m * N + n] = f2bf(silu(a0[m]) * a1[m]);
      else {
        float* o = out + (long long)m * ldo + n;
        *o = acc ? *o + a0[m] : a0[m];
      }
    }
  }
}
template <int SW>
static void gemv_go(const uint16_t* A, int lda, int M, const uint16_t* W, int N, int K, float* out, int ldo, bool acc, uint16_t* outb, cudaStream_t st) {
  if (K % 8 || lda % 8) { std::fprintf(stderr, "gemv: K %d lda %d\n", K, lda); std::abort(); }
  const dim3 g((N + 7) / 8);
  switch (M) {
#define GV(MM) case MM: gemv_k<MM, SW><<<g, 256, 0, st>>>(A, lda, W, N, K, out, ldo, acc ? 1 : 0, outb); break;
    GV(1) GV(2) GV(3) GV(4) GV(5) GV(6) GV(7) GV(8)
#undef GV
    default: std::fprintf(stderr, "gemv M %d\n", M); std::abort();
  }
  KCK();
}
void gemv(const uint16_t* A, int lda, int M, const uint16_t* W, int N, int K, float* out, int ldo, bool acc, cudaStream_t st) {
  gemv_go<0>(A, lda, M, W, N, K, out, ldo, acc, nullptr, st);
}
void gemv_swiglu(const uint16_t* A, int lda, int M, const uint16_t* W, int I, int K, uint16_t* out, cudaStream_t st) {
  gemv_go<1>(A, lda, M, W, I, K, nullptr, 0, false, out, st);
}
__global__ void inc_k(int* p) { if (threadIdx.x == 0) *p += 1; }
void inc(int* p, cudaStream_t st) { inc_k<<<1, 32, 0, st>>>(p); KCK(); }

}  // namespace qk
}  // namespace rvla
