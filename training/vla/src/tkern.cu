// 학습용 커널 — 설명은 tkern.cuh. 식은 앞 계산(qkern.cu)의 미분.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "gemm.cuh"
#include "tkern.cuh"

namespace rvla {
namespace qk {
template <bool AT, bool BT, int EPI>
void gl(const net::GemmP& p, int gz, cudaStream_t st);
}
namespace tk {

using namespace net;
#define KCK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)
static unsigned nb(long long n, int t = 256) { return (unsigned)((n + t - 1) / t); }

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float dsilu(float x) { const float s = sigm(x); return s * (1.f + x * (1.f - s)); }

// ---- colsum ----
constexpr int CS = 256;
__global__ void colpart_k(const float* T, int R, int C, int ld, float* part) {
  const int r0 = blockIdx.y * CS, r1 = min(R, r0 + CS);
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < C; c += gridDim.x * blockDim.x) {
    float s = 0.f;
    for (int r = r0; r < r1; ++r) s = s + T[(long long)r * ld + c];
    part[(long long)blockIdx.y * C + c] = s;
  }
}
__global__ void colred_k(const float* part, int nbk, int C, float* out, int acc) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s = 0.f;
  for (int k = 0; k < nbk; ++k) s = s + part[(long long)k * C + c];
  out[c] = acc ? out[c] + s : s;
}
void colsum(const float* T, int R, int C, int ld, float* part, float* out, bool acc, cudaStream_t st) {
  const int nbk = (R + CS - 1) / CS;
  colpart_k<<<dim3(std::min<unsigned>(nb(C, 128), 64), nbk), 128, 0, st>>>(T, R, C, ld, part);
  colred_k<<<nb(C), 256, 0, st>>>(part, nbk, C, out, acc ? 1 : 0);
  KCK();
}
__global__ void f2bf_k(const float* x, long long n, uint16_t* y) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = net::f2bf(x[i]);
}
void f2bf(const float* x, long long n, uint16_t* y, cudaStream_t st) { f2bf_k<<<nb(n), 256, 0, st>>>(x, n, y); KCK(); }
__global__ void add_k(float* y, const float* x, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] + x[i];
}
void add(float* y, const float* x, long long n, cudaStream_t st) { add_k<<<nb(n), 256, 0, st>>>(y, x, n); KCK(); }
__global__ void copy2d_k(const float* s, int lds, float* d, int ldd, int R, int C, int acc) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)R * C) return;
  const int r = (int)(q / C), c = (int)(q % C);
  const float v = s[(long long)r * lds + c];
  float* o = d + (long long)r * ldd + c;
  *o = acc ? *o + v : v;
}
void copy2d(const float* src, int lds, float* dst, int ldd, int R, int C, bool acc, cudaStream_t st) {
  copy2d_k<<<nb((long long)R * C), 256, 0, st>>>(src, lds, dst, ldd, R, C, acc ? 1 : 0);
  KCK();
}

// ---- GEMM ----
static void chk8(int a, int b, int c, int d) {
  if (a % 8 || b % 8 || c % 8 || d % 8) { std::fprintf(stderr, "tk GEMM: shapes/strides must be multiples of 8 (%d %d %d %d)\n", a, b, c, d); std::abort(); }
}
void mm(const uint16_t* A, int lda, int M, const uint16_t* W, int N, int K, float* out, int ldo, bool acc, cudaStream_t st) {
  if (M == 0) return;
  chk8(lda, N, K, ldo);
  GemmP p{};
  p.A = A; p.lda = lda; p.B = W; p.ldb = K; p.M = M; p.N = N; p.K = K; p.C = out; p.ldc = ldo; p.act = ACT_LIN;
  if (acc) qk::gl<false, false, EPI_ACC_F32>(p, 1, st);
  else qk::gl<false, false, EPI_F32>(p, 1, st);
}
void mm_dx(const uint16_t* dZ, int ldz, int M, const uint16_t* W, int N, int K, float* out, int ldo, bool acc, cudaStream_t st) {
  if (M == 0) return;
  chk8(ldz, N, K, ldo);
  GemmP p{};
  p.A = dZ; p.lda = ldz; p.B = W; p.ldb = K; p.M = M; p.N = K; p.K = N; p.C = out; p.ldc = ldo; p.act = ACT_LIN;
  if (acc) qk::gl<false, true, EPI_ACC_F32>(p, 1, st);
  else qk::gl<false, true, EPI_F32>(p, 1, st);
}
__global__ void dwred_k(const float* ws, int sp, long long n, uint16_t* G, float* gacc) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = 0.f;
  for (int z = 0; z < sp; ++z) s = s + ws[(long long)z * n + i];
  if (gacc) gacc[i] = gacc[i] + s;
  else G[i] = net::f2bf(s);
}
void mm_dw(const uint16_t* dZ, int ldz, const uint16_t* X, int ldx, int M, int N, int K, float* ws, int chunk, uint16_t* G, float* gacc, cudaStream_t st) {
  const long long n = (long long)N * K;
  if (M == 0) {
    if (G) cudaMemsetAsync(G, 0, n * 2, st);
    return;
  }
  chk8(ldz, ldx, N, K);
  GemmP p{};
  // 행(합 방향)을 나누는 수: 결과 타일이 GPU 를 채우면(≥ 280 = SM 70 × 4) 나누지 않음, 모자라면 채울 만큼만(최대 ⌈M/chunk⌉ — 작업 공간 상한)
  const int spmax = (M + chunk - 1) / chunk, tiles = ((N + 127) / 128) * ((K + 127) / 128);
  int sp = std::min(spmax, std::max(1, (280 + tiles - 1) / tiles));
  const int ch = ((M + sp - 1) / sp + 31) / 32 * 32;
  sp = (M + ch - 1) / ch;
  if (sp == 1) { mme_dw1(dZ, ldz, X, ldx, M, N, K, G, gacc, st); return; }   // 조각 하나: 끝단에서 바로 bf16(같은 값)
  p.A = dZ; p.lda = ldz; p.B = X; p.ldb = ldx; p.M = N; p.N = K; p.K = M; p.C = ws; p.ldc = K; p.kchunk = ch;
  qk::gl<true, true, EPI_SPLIT_F32>(p, sp, st);
  dwred_k<<<nb(n), 256, 0, st>>>(ws, sp, n, G, gacc);
  KCK();
}

// ---- 줄마다 더하는 변수 기울기를 커널 안에서(D): 블록 = 벡터 RPB 개(워프 8, 워프마다 RPB/8 개), 레인이 d = lane + 32j 를 레지스터에 더하고
// 블록 끝에 워프 0..7 순서로 합해 조각 part[블록][D] → colred(조각 순서대로). 결정적(원자 없음). wt 행렬을 쓰고 다시 읽던 것을 없앰.
constexpr int RPB = 8, PJ = 32;   // 블록 = 벡터 8 개(워프마다 하나), D ≤ 1024
struct PAcc {
  float a[PJ];
  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int j = 0; j < PJ; ++j) a[j] = 0.f;
  }
};
// 블록 합: 워프 순서 고정
__device__ __forceinline__ void pacc_flush(PAcc& A, int D, float* part, int ldp) {
  __shared__ float sm[8][PJ * 32];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int j = 0; j < PJ; ++j) { const int d = lane + 32 * j; if (d < D) sm[w][d] = A.a[j]; }
  __syncthreads();
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) s = s + sm[k][d];
    part[(long long)blockIdx.x * ldp + d] = s;
  }
}
long long npart_floats(long long rows, int D) { const long long nb_ = (rows + RPB - 1) / RPB; return nb_ * D + ((nb_ + 255) / 256 + 1) * (long long)D; }

// ---- RMSNorm 뒤: 블록 = 벡터 RPB 개, 워프 = 벡터 ----
__global__ void __launch_bounds__(256) rms_bwd_k(const float* dY, int lddy, const float* X, int ldx, int NV, int per, int D, const float* w, int w1, float eps,
                                                 float* dX, int lddx, int acc, uint16_t* dXb, float* part, int bug) {
  const int wl = threadIdx.x >> 5, lane = threadIdx.x & 31;
  PAcc A;
  A.zero();
  for (int vi = wl; vi < RPB; vi += 8) {
    const int v = blockIdx.x * RPB + vi;
    if (v >= NV) break;
    const long long ox = (long long)(v / per) * ldx + (v % per) * D, oy = (long long)(v / per) * lddy + (v % per) * D,
                    od = (long long)(v / per) * lddx + (v % per) * D;
    float s = 0.f;
    for (int d = lane; d < D; d += 32) s = s + X[ox + d] * X[ox + d];
    s = wsum(s);
    const float r = 1.f / sqrtf(s / (float)D + eps);
    float gx = 0.f;
    for (int d = lane; d < D; d += 32) {
      const float g = bug == 1 ? dY[oy + d] : dY[oy + d] * (w1 ? 1.f + w[d] : w[d]);
      gx = gx + g * X[ox + d];
    }
    gx = wsum(gx);
    const float c = r * r * r * gx / (float)D;
#pragma unroll
    for (int j = 0; j < PJ; ++j) {
      const int d = lane + 32 * j;
      if (d >= D) break;
      const float g = bug == 1 ? dY[oy + d] : dY[oy + d] * (w1 ? 1.f + w[d] : w[d]);
      const float dx = r * g - X[ox + d] * c;
      const float o = acc ? dX[od + d] + dx : dx;
      dX[od + d] = o;
      if (dXb) dXb[od + d] = net::f2bf(o);
      A.a[j] = A.a[j] + dY[oy + d] * X[ox + d] * r;
    }
  }
  if (part) pacc_flush(A, D, part, D);
}
void rms_bwd(const float* dY, int lddy, const float* X, int ldx, int NV, int per, int D, const float* w, bool w1, float eps, float* dX, int lddx,
             bool acc, float* npart, float* gw, int bug, cudaStream_t st, uint16_t* dXb) {
  if (D > PJ * 32) { std::fprintf(stderr, "rms_bwd D > 1024\n"); std::abort(); }
  const int nbk = (NV + RPB - 1) / RPB;
  rms_bwd_k<<<nbk, 256, 0, st>>>(dY, lddy, X, ldx, NV, per, D, w, w1 ? 1 : 0, eps, dX, lddx, acc ? 1 : 0, dXb, gw ? npart : nullptr, bug);
  if (gw) colsum(npart, nbk, D, D, npart + (long long)nbk * D, gw, false, st);
  KCK();
}

// ---- LayerNorm ----
__global__ void ln_fwd_k(const float* X, int R, int D, const float* g, const float* b, float eps, uint16_t* out, float* outf) {
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (r >= R) return;
  const float* x = X + (long long)r * D;
  float s = 0.f;
  for (int d = lane; d < D; d += 32) s = s + x[d];
  const float m = wsum(s) / (float)D;
  float q = 0.f;
  for (int d = lane; d < D; d += 32) { const float e = x[d] - m; q = q + e * e; }
  const float rs = 1.f / sqrtf(wsum(q) / (float)D + eps);
  for (int d = lane; d < D; d += 32) {
    const float y = (x[d] - m) * rs * g[d] + b[d];
    if (out) out[(long long)r * D + d] = net::f2bf(y);
    if (outf) outf[(long long)r * D + d] = y;
  }
}
void ln_fwd(const float* X, int R, int D, const float* g, const float* b, float eps, uint16_t* out, float* outf, cudaStream_t st) {
  ln_fwd_k<<<nb((long long)R * 32, 128), 128, 0, st>>>(X, R, D, g, b, eps, out, outf);
  KCK();
}
__global__ void __launch_bounds__(256) ln_bwd_k(const float* dY, const float* X, int R, int D, const float* g, float eps, float* dX, uint16_t* dXb,
                                                float* pg, float* pb) {
  const int wl = threadIdx.x >> 5, lane = threadIdx.x & 31;
  PAcc Ag, Ab;
  Ag.zero(); Ab.zero();
  for (int ri = wl; ri < RPB; ri += 8) {
    const int r = blockIdx.x * RPB + ri;
    if (r >= R) break;
    const float* x = X + (long long)r * D;
    const float* dy = dY + (long long)r * D;
    float s = 0.f;
    for (int d = lane; d < D; d += 32) s = s + x[d];
    const float m = wsum(s) / (float)D;
    float q = 0.f;
    for (int d = lane; d < D; d += 32) { const float e = x[d] - m; q = q + e * e; }
    const float rs = 1.f / sqrtf(wsum(q) / (float)D + eps);
    float s1 = 0.f, s2 = 0.f;
    for (int d = lane; d < D; d += 32) {
      const float xh = (x[d] - m) * rs, gd = dy[d] * g[d];
      s1 = s1 + gd; s2 = s2 + gd * xh;
    }
    s1 = wsum(s1) / (float)D; s2 = wsum(s2) / (float)D;
#pragma unroll
    for (int j = 0; j < PJ; ++j) {
      const int d = lane + 32 * j;
      if (d >= D) break;
      const float xh = (x[d] - m) * rs, gd = dy[d] * g[d];
      const float o = dX[(long long)r * D + d] + rs * (gd - s1 - xh * s2);
      dX[(long long)r * D + d] = o;
      if (dXb) dXb[(long long)r * D + d] = net::f2bf(o);
      Ag.a[j] = Ag.a[j] + dy[d] * xh;
      Ab.a[j] = Ab.a[j] + dy[d];
    }
  }
  pacc_flush(Ag, D, pg, D);
  __syncthreads();
  pacc_flush(Ab, D, pb, D);
}
void ln_bwd(const float* dY, const float* X, int R, int D, const float* g, float eps, float* dX, float* npart, float* gg, float* gb, cudaStream_t st,
            uint16_t* dXb) {
  if (D > PJ * 32) { std::fprintf(stderr, "ln_bwd D > 1024\n"); std::abort(); }
  const int nbk = (R + RPB - 1) / RPB;
  float* pb = npart + (long long)nbk * D;
  float* p2 = pb + (long long)nbk * D;
  ln_bwd_k<<<nbk, 256, 0, st>>>(dY, X, R, D, g, eps, dX, dXb, npart, pb);
  colsum(npart, nbk, D, D, p2, gg, false, st);
  colsum(pb, nbk, D, D, p2, gb, false, st);
  KCK();
}
__global__ void bias_k(float* Y, int R, int N, int ld, const float* b) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)R * N) return;
  const int r = (int)(q / N), c = (int)(q % N);
  Y[(long long)r * ld + c] = Y[(long long)r * ld + c] + b[c];
}
void bias_add(float* Y, int R, int N, int ld, const float* b, cudaStream_t st) {
  bias_k<<<nb((long long)R * N), 256, 0, st>>>(Y, R, N, ld, b);
  KCK();
}
__device__ __forceinline__ float gelu_t(float x) {
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + tanhf(u));
}
__device__ __forceinline__ float dgelu_t(float x) {
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x), t = tanhf(u);
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * 0.7978845608028654f * (1.f + 3.f * 0.044715f * x * x);
}
__global__ void gelu_k(const float* X, long long n, uint16_t* o) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = net::f2bf(gelu_t(X[i]));
}
void gelu_fwd(const float* X, long long n, uint16_t* out, cudaStream_t st) { gelu_k<<<nb(n), 256, 0, st>>>(X, n, out); KCK(); }
__global__ void dgelu_k(const float* dY, const float* X, long long n, uint16_t* o, float* of) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = dY[i] * dgelu_t(X[i]);
  o[i] = net::f2bf(v);
  if (of) of[i] = v;
}
void gelu_bwd(const float* dY, const float* X, long long n, uint16_t* dX, float* dXf, cudaStream_t st) { dgelu_k<<<nb(n), 256, 0, st>>>(dY, X, n, dX, dXf); KCK(); }

// ---- 어텐션 ----
__device__ __forceinline__ int seg1_len(const AttP& p, int b) { return p.len1 ? p.len1[b] : p.n1c; }
template <int E>
__global__ void att_fwd_k(const AttP p) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (w >= p.B * p.n * p.nq) return;
  constexpr int hd = 32 * E;
  const int h = w % p.nq, r = w / p.nq, b = r / p.n, t = r % p.n, kh = h / (p.nq / p.nkv);
  float qv[E], acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) { qv[e] = p.Q[(long long)r * p.ldq + h * hd + lane * E + e]; acc[e] = 0.f; }
  float m = -INFINITY, s = 0.f;
  int n1 = seg1_len(p, b);
  if (p.causal) n1 = min(n1, t + p.qoff + 1);
  const int nk = n1 + p.n2;
  for (int j = 0; j < nk; ++j) {
    const float *kr, *vr;
    if (j < n1) { const long long o = ((long long)b * p.L1 + j) * p.ldk1 + kh * hd + lane * E; kr = p.K1 + o; vr = p.V1 + o; }
    else { const long long o = ((long long)b * p.n2 + (j - n1)) * p.ldk2 + kh * hd + lane * E; kr = p.K2 + o; vr = p.V2 + o; }
    float d = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) d = d + qv[e] * kr[e];
    d = wsum(d) * p.scale;
    const float mn = fmaxf(m, d), cr = expf(m - mn), pr = expf(d - mn);
    s = s * cr + pr;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = acc[e] * cr + pr * vr[e];
    m = mn;
  }
  const float inv = s > 0.f ? 1.f / s : 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) p.O[(long long)r * p.ldo + h * hd + lane * E + e] = acc[e] * inv;
  if (lane == 0) p.lse[(long long)r * p.nq + h] = s > 0.f ? m + logf(s) : 0.f;
}
template <int E>
__global__ void att_dq_k(const AttP p) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (w >= p.B * p.n * p.nq) return;
  constexpr int hd = 32 * E;
  const int h = w % p.nq, r = w / p.nq, b = r / p.n, t = r % p.n, kh = h / (p.nq / p.nkv);
  float qv[E], dov[E], acc[E];
  float D = 0.f;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    qv[e] = p.Q[(long long)r * p.ldq + h * hd + lane * E + e];
    dov[e] = p.dO[(long long)r * p.lddo + h * hd + lane * E + e];
    D = D + dov[e] * p.O[(long long)r * p.ldo + h * hd + lane * E + e];
    acc[e] = 0.f;
  }
  D = wsum(D);
  const float ls = p.lse[(long long)r * p.nq + h];
  int n1 = seg1_len(p, b);
  if (p.causal) n1 = min(n1, t + p.qoff + 1);
  const int nk = n1 + p.n2;
  for (int j = 0; j < nk; ++j) {
    const float *kr, *vr;
    if (j < n1) { const long long o = ((long long)b * p.L1 + j) * p.ldk1 + kh * hd + lane * E; kr = p.K1 + o; vr = p.V1 + o; }
    else { const long long o = ((long long)b * p.n2 + (j - n1)) * p.ldk2 + kh * hd + lane * E; kr = p.K2 + o; vr = p.V2 + o; }
    float d = 0.f, dp = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) { d = d + qv[e] * kr[e]; dp = dp + dov[e] * vr[e]; }
    d = wsum(d); dp = wsum(dp);
    const float pr = expf(d * p.scale - ls), ds = pr * (dp - D);
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = acc[e] + ds * kr[e];
  }
  const float sc = p.bug == 1 ? 1.f : p.scale;
#pragma unroll
  for (int e = 0; e < E; ++e) p.dQ[(long long)r * p.lddq + h * hd + lane * E + e] = acc[e] * sc;
  if (lane == 0) p.Dd[(long long)r * p.nq + h] = D;
}
// 워프 = (판, 키 자리(구간 1 은 0..L1−1, 구간 2 는 L1..L1+n2−1), 키 머리)
template <int E>
__global__ void att_dkv_k(const AttP p) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  const int nkey = p.L1 + p.n2;
  if (w >= p.B * nkey * p.nkv) return;
  constexpr int hd = 32 * E;
  const int kh = w % p.nkv, kk = (w / p.nkv) % nkey, b = w / (p.nkv * nkey);
  const bool s1 = kk < p.L1;
  if (s1 && !p.dK1) return;
  if (!s1 && !p.dK2) return;
  const int j = s1 ? kk : kk - p.L1;
  const long long ko = s1 ? ((long long)b * p.L1 + j) * p.ldk1 + kh * hd : ((long long)b * p.n2 + j) * p.ldk2 + kh * hd;
  const float* K = s1 ? p.K1 : p.K2;
  const float* V = s1 ? p.V1 : p.V2;
  float kv[E], vv[E], dk[E], dv[E];
#pragma unroll
  for (int e = 0; e < E; ++e) { kv[e] = K[ko + lane * E + e]; vv[e] = V[ko + lane * E + e]; dk[e] = 0.f; dv[e] = 0.f; }
  const int n1b = seg1_len(p, b);
  const bool valid = !s1 || j < n1b;
  const int g = p.nq / p.nkv;
  if (valid) {
    for (int t = 0; t < p.n; ++t) {
      if (s1 && p.causal && j > t + p.qoff) continue;
      const long long r = (long long)b * p.n + t;
      for (int hh = 0; hh < g; ++hh) {
        const int h = kh * g + hh;
        const float* q = p.Q + r * p.ldq + h * hd + lane * E;
        const float* dO = p.dO + r * p.lddo + h * hd + lane * E;
        float d = 0.f, dp = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) { d = d + q[e] * kv[e]; dp = dp + dO[e] * vv[e]; }
        d = wsum(d); dp = wsum(dp);
        const float pr = expf(d * p.scale - p.lse[r * p.nq + h]), ds = pr * (dp - p.Dd[r * p.nq + h]);
#pragma unroll
        for (int e = 0; e < E; ++e) { dv[e] = dv[e] + pr * dO[e]; dk[e] = dk[e] + ds * q[e]; }
      }
    }
  }
  const float sc = p.bug == 1 ? 1.f : p.scale;
  float* dK = s1 ? p.dK1 : p.dK2;
  float* dV = s1 ? p.dV1 : p.dV2;
#pragma unroll
  for (int e = 0; e < E; ++e) { dK[ko + lane * E + e] = dk[e] * sc; dV[ko + lane * E + e] = dv[e]; }
}
#define ATT_DISPATCH(KER, NW)                                                       \
  switch (p.hd) {                                                                   \
    case 256: KER<8><<<nb((long long)(NW) * 32, 128), 128, 0, st>>>(p); break;     \
    case 128: KER<4><<<nb((long long)(NW) * 32, 128), 128, 0, st>>>(p); break;     \
    case 64: KER<2><<<nb((long long)(NW) * 32, 128), 128, 0, st>>>(p); break;      \
    case 32: KER<1><<<nb((long long)(NW) * 32, 128), 128, 0, st>>>(p); break;      \
    default: std::fprintf(stderr, "att hd %d\n", p.hd); std::abort();             \
  }
bool att_old() {
  static const bool o = [] { const char* e = getenv("RVLA_ATT_OLD"); return e && e[0] == '1'; }();
  return o;
}

void att_fwd(const AttP& p, cudaStream_t st) {
  if (!att_old() && fa_fwd(p, st)) return;
  ATT_DISPATCH(att_fwd_k, (long long)p.B * p.n * p.nq);
  KCK();
}
void att_bwd(const AttP& p, cudaStream_t st) {
  if (!att_old() && fa_bwd(p, st)) return;
  ATT_DISPATCH(att_dq_k, (long long)p.B * p.n * p.nq);
  ATT_DISPATCH(att_dkv_k, (long long)p.B * (p.L1 + p.n2) * p.nkv);
  KCK();
}

// ---- 게이트·풀 어텐션 준비의 뒤 ----
__global__ void gate_bwd_k(const float* dG, const float* O, const float* T0, int ldT, int R, int nq, int hd, float* dOut, float* dT0, uint16_t* dT0b) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = nq * hd;
  if (q >= (long long)R * W) return;
  const int r = (int)(q / W), c = (int)(q % W), h = c / hd, d = c % hd;
  const long long gi = (long long)r * ldT + h * 2 * hd + hd + d;
  const float s = sigm(T0[gi]);
  dOut[q] = dG[q] * s;
  const float v = dG[q] * O[q] * s * (1.f - s);
  if (dT0b) dT0b[gi] = net::f2bf(v);
  else dT0[gi] = v;
}
void gate_bwd(const float* dG, const float* O, const float* T0, int ldT, int R, int nq, int hd, float* dOut, float* dT0, cudaStream_t st, uint16_t* dT0b) {
  gate_bwd_k<<<nb((long long)R * nq * hd), 256, 0, st>>>(dG, O, T0, ldT, R, nq, hd, dOut, dT0, dT0b);
  KCK();
}
constexpr int MAXHD = 256;
// 워프 = (행, 머리). RoPE 의 전치 → 머리 RMSNorm(1 + w) 의 뒤. bug 2: RoPE 전치 대신 앞 회전을 다시 씀
__global__ void full_prep_bwd_k(const float* T0, int ldT, int R, int n, int pos0, int nq, int nkv, int hd, int rot, float theta, float eps, const float* qn,
                                const float* kn, const float* dQ, const float* dKc, const float* dVc, int Lc, float* dT0, float* qnt, float* knt, int bug,
                                const int* poff, uint16_t* dT0b) {
  __shared__ float sg[8][MAXHD];
  const int w = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31, wl = threadIdx.x / 32;
  const int nh = nq + nkv;
  if (w >= R * nh) return;
  const int r = w / nh, h = w % nh, b = r / n, t = r % n, pos = (poff ? poff[b] : pos0) + t, kpos = pos0 + t;
  const bool isq = h < nq;
  const int kh = h - nq;
  const long long so = (long long)r * ldT + (isq ? h * 2 * hd : nq * 2 * hd + kh * hd);
  const float* dy = isq ? dQ + (long long)r * nq * hd + h * hd : dKc + ((long long)b * Lc + kpos) * nkv * hd + kh * hd;
  for (int d = lane; d < hd; d += 32) sg[wl][d] = dy[d];
  __syncwarp();
  const int half = rot / 2;
  float dy0[MAXHD / 32];
  int ii = 0;
  for (int d = lane; d < hd; d += 32, ++ii) {
    float v = sg[wl][d];
    if (d < rot) {
      const int i = d % half;
      const float inv = 1.f / powf(theta, (float)(2 * i) / (float)rot);
      const float fr = (float)pos * inv, c = cosf(fr), s = sinf(fr);
      if (bug == 2) v = v * c + (d < half ? -sg[wl][d + half] : sg[wl][d - half]) * s;
      else v = d < half ? sg[wl][d] * c + sg[wl][d + half] * s : sg[wl][d] * c - sg[wl][d - half] * s;
    }
    dy0[ii] = v;
  }
  const float* nw = isq ? qn : kn;
  float s = 0.f, gx = 0.f;
  ii = 0;
  for (int d = lane; d < hd; d += 32, ++ii) { const float x = T0[so + d]; s = s + x * x; gx = gx + dy0[ii] * (1.f + nw[d]) * x; }
  s = wsum(s); gx = wsum(gx);
  const float rs = 1.f / sqrtf(s / (float)hd + eps), c3 = rs * rs * rs * gx / (float)hd;
  float* wt = isq ? qnt + ((long long)r * nq + h) * hd : knt + ((long long)r * nkv + kh) * hd;
  ii = 0;
  for (int d = lane; d < hd; d += 32, ++ii) {
    const float x = T0[so + d];
    const float v = rs * dy0[ii] * (1.f + nw[d]) - x * c3;
    if (dT0b) dT0b[so + d] = net::f2bf(v);
    else dT0[so + d] = v;
    wt[d] = dy0[ii] * x * rs;
  }
  if (!isq) {
    const float* dv = dVc + ((long long)b * Lc + kpos) * nkv * hd + kh * hd;
    const long long oo = (long long)r * ldT + nq * 2 * hd + nkv * hd + kh * hd;
    for (int d = lane; d < hd; d += 32) {
      if (dT0b) dT0b[oo + d] = net::f2bf(dv[d]);
      else dT0[oo + d] = dv[d];
    }
  }
}
void full_prep_bwd(const float* T0, int ldT, int R, int n, int pos0, int nq, int nkv, int hd, int rot, float theta, float eps, const float* qn,
                   const float* kn, const float* dQ, const float* dKc, const float* dVc, int Lc, float* dT0, float* qnt, float* knt, int bug,
                   cudaStream_t st, const int* poff, uint16_t* dT0b) {
  full_prep_bwd_k<<<nb((long long)R * (nq + nkv), 8), 256, 0, st>>>(T0, ldT, R, n, pos0, nq, nkv, hd, rot, theta, eps, qn, kn, dQ, dKc, dVc, Lc, dT0, qnt,
                                                                   knt, bug, poff, dT0b);
  KCK();
}

// ---- DeltaNet 준비의 뒤: 워프 = (행, 머리) ----
__global__ void lin_prep_bwd_k(const float* T1, int ld1, const float* T0, int ld0, int boff, int R, int lh, int dk, const float* alog, const float* dtb,
                               const float* dQn, const float* dKn, const float* dG, const float* dBeta, float* dT1, float* dT0, float* alt, float* dtt,
                               uint16_t* dT0b) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (w >= R * lh) return;
  const int r = w / lh, h = w % lh;
  const float* q = T1 + (long long)r * ld1 + h * dk;
  const float* k = T1 + (long long)r * ld1 + lh * dk + h * dk;
  const float* dq = dQn + (long long)r * lh * dk + h * dk;
  const float* dkk = dKn + (long long)r * lh * dk + h * dk;
  float sq = 0.f, sk = 0.f, qd = 0.f, kd = 0.f;
  for (int d = lane; d < dk; d += 32) { sq = sq + q[d] * q[d]; sk = sk + k[d] * k[d]; qd = qd + q[d] * dq[d]; kd = kd + k[d] * dkk[d]; }
  sq = wsum(sq); sk = wsum(sk); qd = wsum(qd); kd = wsum(kd);
  const float iq = 1.f / sqrtf(sq + 1e-6f), ik = 1.f / sqrtf(sk + 1e-6f), sc = 1.f / sqrtf((float)dk);
  for (int d = lane; d < dk; d += 32) {
    dT1[(long long)r * ld1 + h * dk + d] = sc * (iq * dq[d] - q[d] * iq * iq * iq * qd);
    dT1[(long long)r * ld1 + lh * dk + h * dk + d] = ik * dkk[d] - k[d] * ik * ik * ik * kd;
  }
  if (lane == 0) {
    const float bb = T0[(long long)r * ld0 + boff + h], aa = T0[(long long)r * ld0 + boff + lh + h];
    const float be = sigm(bb);
    const float db = dBeta[(long long)r * lh + h] * be * (1.f - be);
    if (dT0b) dT0b[(long long)r * ld0 + boff + h] = net::f2bf(db);
    else dT0[(long long)r * ld0 + boff + h] = db;
    const float x = aa + dtb[h], ea = expf(alog[h]);
    const float sp = x > 20.f ? x : log1pf(expf(x)), dsp = x > 20.f ? 1.f : sigm(x);
    const float dg = dG[(long long)r * lh + h];
    const float dx = dg * (-ea) * dsp;
    if (dT0b) dT0b[(long long)r * ld0 + boff + lh + h] = net::f2bf(dx);
    else dT0[(long long)r * ld0 + boff + lh + h] = dx;
    alt[(long long)r * lh + h] = dg * (-ea * sp);
    dtt[(long long)r * lh + h] = dx;
  }
}
void lin_prep_bwd(const float* T1, int ld1, const float* T0, int ld0, int boff, int R, int lh, int dk, const float* alog, const float* dtb,
                  const float* dQn, const float* dKn, const float* dG, const float* dBeta, float* dT1, float* dT0, float* alt, float* dtt,
                  cudaStream_t st, uint16_t* dT0b) {
  lin_prep_bwd_k<<<nb((long long)R * lh * 32, 128), 128, 0, st>>>(T1, ld1, T0, ld0, boff, R, lh, dk, alog, dtb, dQn, dKn, dG, dBeta, dT1, dT0, alt, dtt,
                                                                   dT0b);
  KCK();
}

// ---- DeltaNet 앞 준비(학습, 융합 E): 인과 합성곱 + SiLU 를 이 커널 안에서(conv_k 없음) → q·k L2 정규화(·1/√dk), β = σ(b), g = −e^A·softplus(a + dt).
// 워프 = (행, 머리): q·k·v 채널 각 dk/dv 개를 레인이 나눠 합성곱(탭 K, 같은 식·같은 순서 = qk::conv_silu 와 비트 같음).
// T1(합성곱 출력, 뒤가 씀): keep 이면 q·k·v 전부, 아니면 v 만(재귀가 읽음) — 첫 앞 계산은 q·k 를 안 씀.
__device__ __forceinline__ float conv1(const float* T0, int ld0, long long rb, int t, int c, int K, const float* w) {
  float s = 0.f;
  for (int k = 0; k < K; ++k) {
    const int p = t - (K - 1) + k;
    if (p >= 0) s = s + w[c * K + k] * T0[(rb + p) * ld0 + c];
  }
  return silu(s);
}
__global__ void lin_prep_conv_k(const float* T0, int ld0, int n, int R, int lh, int dk, int dv, int K, const float* cw, int boff, const float* alog,
                                const float* dtb, int keep, float* T1, int ld1, float* Qn, float* Kn, float* G, float* Beta) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (w >= R * lh) return;
  const int r = w / lh, h = w % lh, t = r % n;
  const long long rb = (long long)(r - t);
  constexpr int MX = 8;   // dk ≤ 256, 정적 첨자(레지스터)
  float qv[MX], kv[MX];
  float sq = 0.f, sk = 0.f;
#pragma unroll
  for (int j = 0; j < MX; ++j) {
    const int d = lane + 32 * j;
    qv[j] = 0.f; kv[j] = 0.f;
    if (d < dk) {
      const int cq = h * dk + d, ck = lh * dk + h * dk + d;
      qv[j] = conv1(T0, ld0, rb, t, cq, K, cw);
      kv[j] = conv1(T0, ld0, rb, t, ck, K, cw);
      if (keep) { T1[(long long)r * ld1 + cq] = qv[j]; T1[(long long)r * ld1 + ck] = kv[j]; }
    }
  }
#pragma unroll
  for (int j = 0; j < MX; ++j) if (lane + 32 * j < dk) { sq = sq + qv[j] * qv[j]; sk = sk + kv[j] * kv[j]; }
  sq = wsum(sq); sk = wsum(sk);
  const float iq = 1.f / sqrtf(sq + 1e-6f), ik = 1.f / sqrtf(sk + 1e-6f);
  const float sc = 1.f / sqrtf((float)dk);
#pragma unroll
  for (int j = 0; j < MX; ++j) {
    const int d = lane + 32 * j;
    if (d < dk) {
      Qn[(long long)r * lh * dk + h * dk + d] = qv[j] * iq * sc;
      Kn[(long long)r * lh * dk + h * dk + d] = kv[j] * ik;
    }
  }
  for (int d = lane; d < dv; d += 32) {
    const int cv = 2 * lh * dk + h * dv + d;
    T1[(long long)r * ld1 + cv] = conv1(T0, ld0, rb, t, cv, K, cw);
  }
  if (lane == 0) {
    const float bb = T0[(long long)r * ld0 + boff + h], aa = T0[(long long)r * ld0 + boff + lh + h];
    const float x = aa + dtb[h];
    const float sp = x > 20.f ? x : log1pf(expf(x));
    Beta[(long long)r * lh + h] = sigm(bb);
    G[(long long)r * lh + h] = -expf(alog[h]) * sp;
  }
}
void lin_prep_conv(const float* T0, int ld0, int B, int n, int lh, int dk, int dv, int K, const float* cw, int boff, const float* alog, const float* dtb,
                   bool keep, float* T1, int ld1, float* Qn, float* Kn, float* G, float* Beta, cudaStream_t st) {
  if (dk > 256) { std::fprintf(stderr, "lin_prep_conv dk > 256\n"); std::abort(); }
  const int R = B * n;
  lin_prep_conv_k<<<nb((long long)R * lh * 32, 128), 128, 0, st>>>(T0, ld0, n, R, lh, dk, dv, K, cw, boff, alog, dtb, keep ? 1 : 0, T1, ld1, Qn, Kn, G, Beta);
  KCK();
}

// ---- 합성곱 뒤(융합 E): 스레드 = (채널, 판), 시간 차례로 미끄럼 창(X·dp 탭 K) — dp 버퍼 없음.
// pre = Σ w·x, dp = dT1·silu'(pre), dX[t] = Σ_k w_k·dp[t+K−1−k](예전 convx 와 같은 순서), 가중치 조각 part[판][c·K + k] = Σ_t dp[t]·x[t−K+1+k] → colred(판 순서)
constexpr int CSEG = 32;   // 시간 조각(앞뒤 K−1 겹침): 판 × 조각 × 채널 스레드. K 는 컴파일 상수(창 배열이 레지스터에 남게)
template <int K>
__global__ void conv_bwd_seq_k(const float* X, int ld, int n, int C, const float* w, const float* dT1, int ld1, uint16_t* dXb, float* part) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z, sg = blockIdx.y;
  if (c >= C) return;
  const int t0 = sg * CSEG, t1 = min(n, t0 + CSEG);
  float wk[K], xw[K], dpw[K], gw[K];
#pragma unroll
  for (int k = 0; k < K; ++k) { wk[k] = w[c * K + k]; xw[k] = 0.f; dpw[k] = 0.f; gw[k] = 0.f; }
  const long long rb = (long long)b * n;
  // 창: xw[K−1] = x[tc], dpw[K−1] = dp[tc]. x 는 t0 − (K−1) 부터, dp 는 [t0, t1 + K − 1) 만 셈(dX[t] 는 dp[t..t+K−1])
  for (int tc = t0 - (K - 1); tc < t1 + K - 1; ++tc) {
#pragma unroll
    for (int j = 0; j < K - 1; ++j) { xw[j] = xw[j + 1]; dpw[j] = dpw[j + 1]; }
    xw[K - 1] = (tc >= 0 && tc < n) ? X[(rb + tc) * ld + c] : 0.f;
    float dp = 0.f;
    if (tc >= t0 && tc < n) {
      float pre = 0.f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const int p = tc - (K - 1) + k;
        if (p >= 0) pre = pre + wk[k] * xw[k];
      }
      dp = dT1[(rb + tc) * ld1 + c] * dsilu(pre);
      if (tc < t1)
#pragma unroll
        for (int k = 0; k < K; ++k) {
          const int p = tc - (K - 1) + k;
          if (p >= 0) gw[k] = gw[k] + dp * xw[k];
        }
    }
    dpw[K - 1] = dp;
    const int t = tc - (K - 1);
    if (t >= t0 && t < t1) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const int o = t + (K - 1) - k;
        if (o < n) s = s + wk[k] * dpw[K - 1 - k];
      }
      dXb[(rb + t) * ld + c] = net::f2bf(s);
    }
  }
  const long long pi = (long long)b * gridDim.y + sg;
#pragma unroll
  for (int k = 0; k < K; ++k) part[pi * C * K + (long long)c * K + k] = gw[k];
}
int conv_nseg(int n) { return (n + CSEG - 1) / CSEG; }
void conv_bwd_seq(const float* X, int ld, int B, int n, int C, int K, const float* w, const float* dT1, int ld1, uint16_t* dXb, float* part, float* gw,
                  cudaStream_t st) {
  const int ns = conv_nseg(n);
  if (K == 4) conv_bwd_seq_k<4><<<dim3(nb(C, 128), ns, B), 128, 0, st>>>(X, ld, n, C, w, dT1, ld1, dXb, part);
  else { std::fprintf(stderr, "conv_bwd_seq: K %d (4 만)\n", K); std::abort(); }
  colsum(part, B * ns, C * K, C * K, part + (long long)B * ns * C * K, gw, false, st);
  KCK();
}

// ---- DeltaNet 재귀의 뒤 ----
// 블록 = (판, 머리, 열 조각 NS 개 중 하나). 공유: S(이전 상태, 줄 간격 DVS+1), dS, 벡터.
// 검문점: 앞 계산(qk::deltanet, ck 를 주면)이 C = 16 스텝마다 S 를 작업 공간에 써 둔다(hasck = 1). 없으면 이 커널이 먼저 앞으로 돌며 쓴다.
// 뒤: 구간(C 스텝)마다 검문점에서 다시 계산해 S_{t−1} 들을 전역에 두고 거꾸로.
constexpr int DNC = 16, DNS = 4;
long long dn_ws_floats(int B, int n, int lh, int dk, int dv) {
  const int nck = (n + DNC - 1) / DNC;
  return (long long)B * lh * (nck + DNC) * dk * dv + (long long)B * n * lh * DNS * (2 * dk + 2);
}
template <int DK, int DVS>
struct DnSm {
  static constexpr int SP = DVS + 1;
  static constexpr int FLOATS = 2 * DK * SP + 2 * DK + 6 * DVS + 256;
};
template <int DK, int DVS>
__device__ __forceinline__ void dn_colsum(const float* S, const float* kvec, float* red, float* out) {
  // out[j] = Σ_i S[i][j]·kvec[i]  (256 스레드, 열 j = tid % DVS, 조각 p = tid / DVS)
  constexpr int P = 256 / DVS, RP = (DK + P - 1) / P, SP = DVS + 1;
  const int tid = threadIdx.x, j = tid % DVS, p = tid / DVS;
  float s = 0.f;
  for (int i = p * RP; i < min(DK, (p + 1) * RP); ++i) s = s + S[i * SP + j] * kvec[i];
  red[p * DVS + j] = s;
  __syncthreads();
  if (tid < DVS) {
    float a = 0.f;
    for (int q = 0; q < P; ++q) a = a + red[q * DVS + tid];
    out[tid] = a;
  }
  __syncthreads();
}
__device__ __forceinline__ float dn_bsum(float v, float* red) {
  v = wsum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float a = 0.f;
  for (int k = 0; k < 8; ++k) a = a + red[k];
  __syncthreads();
  return a;
}
template <int DK, int DVS>
__global__ void __launch_bounds__(256) dn_bwd_k(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Bt, const float* dO,
                                                int n, int lh, int nodecay, int hasck, float* ws, float* dQh, float* dKh, float* dV, int lddv, float* dGh,
                                                float* dBh, long long hstride, int bug) {
  using SM = DnSm<DK, DVS>;
  constexpr int SP = SM::SP, NE = DK * DVS, DV = DNS * DVS;
  extern __shared__ float sm[];
  float* sS = sm;
  float* sD = sS + DK * SP;
  float* sq = sD + DK * SP;
  float* sk = sq + DK;
  float* sdo = sk + DK;
  float* sv = sdo + DVS;
  float* sdl = sv + DVS;
  float* smm = sdl + DVS;
  float* sdm = smm + DVS;
  float* sx = sdm + DVS;
  float* red = sx + DVS;   // 256
  const int blk = blockIdx.x, part = blk % DNS, bh = blk / DNS, b = bh / lh, h = bh % lh, j0 = part * DVS, tid = threadIdx.x;
  const int nck = (n + DNC - 1) / DNC;
  float* ck = ws + (long long)blk * (nck + DNC) * NE;
  float* sg = ck + (long long)nck * NE;
  const long long gs = hstride / DK;   // 행·머리 수
  auto ld_vec = [&](long long r) {
    for (int i = tid; i < DK; i += 256) { sq[i] = Qn[r * lh * DK + h * DK + i]; sk[i] = Kn[r * lh * DK + h * DK + i]; }
    for (int j = tid; j < DVS; j += 256) sv[j] = V[r * ldv + h * DV + j0 + j];
  };
  auto step = [&](long long r) {
    ld_vec(r);
    __syncthreads();
    const float a = nodecay ? 1.f : expf(G[r * lh + h]), be = Bt[r * lh + h];
    dn_colsum<DK, DVS>(sS, sk, red, smm);
    for (int j = tid; j < DVS; j += 256) sdl[j] = be * (sv[j] - a * smm[j]);
    __syncthreads();
    for (int e = tid; e < NE; e += 256) { const int i = e / DVS, j = e % DVS; sS[i * SP + j] = a * sS[i * SP + j] + sk[i] * sdl[j]; }
    __syncthreads();
  };
  if (!hasck) {
    for (int e = tid; e < NE; e += 256) sS[(e / DVS) * SP + e % DVS] = 0.f;
    __syncthreads();
    for (int t = 0; t < n; ++t) {
      if (t % DNC == 0)
        for (int e = tid; e < NE; e += 256) ck[(long long)(t / DNC) * NE + e] = sS[(e / DVS) * SP + e % DVS];
      step((long long)b * n + t);
    }
  }
  for (int e = tid; e < NE; e += 256) sD[(e / DVS) * SP + e % DVS] = 0.f;
  __syncthreads();
  for (int c = nck - 1; c >= 0; --c) {
    const int t0 = c * DNC, t1 = min(n, t0 + DNC);
    for (int e = tid; e < NE; e += 256) sS[(e / DVS) * SP + e % DVS] = ck[(long long)c * NE + e];
    __syncthreads();
    for (int t = t0; t < t1; ++t) {
      for (int e = tid; e < NE; e += 256) sg[(long long)(t - t0) * NE + e] = sS[(e / DVS) * SP + e % DVS];
      step((long long)b * n + t);
    }
    for (int t = t1 - 1; t >= t0; --t) {
      const long long r = (long long)b * n + t;
      for (int e = tid; e < NE; e += 256) sS[(e / DVS) * SP + e % DVS] = sg[(long long)(t - t0) * NE + e];   // S_{t−1}
      ld_vec(r);
      for (int j = tid; j < DVS; j += 256) sdo[j] = dO[r * lh * DV + h * DV + j0 + j];
      __syncthreads();
      const float a = nodecay ? 1.f : expf(G[r * lh + h]), be = Bt[r * lh + h];
      dn_colsum<DK, DVS>(sS, sk, red, smm);
      for (int j = tid; j < DVS; j += 256) { smm[j] = a * smm[j]; sdl[j] = be * (sv[j] - smm[j]); }
      for (int e = tid; e < NE; e += 256) { const int i = e / DVS, j = e % DVS; sD[i * SP + j] = sD[i * SP + j] + sq[i] * sdo[j]; }
      __syncthreads();
      dn_colsum<DK, DVS>(sD, sk, red, sx);   // dδ
      float dbp = 0.f;
      for (int j = tid; j < DVS; j += 256) {
        dV[r * lddv + h * DV + j0 + j] = be * sx[j];
        dbp = dbp + sx[j] * (sv[j] - smm[j]);
        sdm[j] = -be * sx[j];
      }
      const float dbeta = dn_bsum(dbp, red);
      {
        constexpr int P = 256 / DK, CP = (DVS + P - 1) / P;
        const int i = tid % DK, p = tid / DK;
        float a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f;
        for (int j = p * CP; j < min(DVS, (p + 1) * CP); ++j) {
          const float s = sS[i * SP + j];
          a1 = a1 + s * sdo[j];
          a2 = a2 + sdl[j] * sdo[j];
          a3 = a3 + sD[i * SP + j] * sdl[j];
          a4 = a4 + s * sdm[j];
        }
        red[(p * DK + i) * 4 + 0] = a1;   // red 는 256 칸 + 뒤 여유(아래 FLOATS 에 4·256 을 잡음)
        red[(p * DK + i) * 4 + 1] = a2;
        red[(p * DK + i) * 4 + 2] = a3;
        red[(p * DK + i) * 4 + 3] = a4;
        __syncthreads();
        if (tid < DK) {
          float dq = 0.f, dd = 0.f, k1 = 0.f, k2 = 0.f;
          for (int q = 0; q < P; ++q) {
            dq = dq + red[(q * DK + tid) * 4 + 0]; dd = dd + red[(q * DK + tid) * 4 + 1];
            k1 = k1 + red[(q * DK + tid) * 4 + 2]; k2 = k2 + red[(q * DK + tid) * 4 + 3];
          }
          dQh[part * hstride + r * lh * DK + h * DK + tid] = a * dq + sk[tid] * dd;
          dKh[part * hstride + r * lh * DK + h * DK + tid] = k1 + a * k2;
        }
        __syncthreads();
      }
      float dap = 0.f;
      for (int e = tid; e < NE; e += 256) {
        const int i = e / DVS, j = e % DVS;
        const float g = sD[i * SP + j] + sk[i] * sdm[j];
        dap = dap + g * sS[i * SP + j];
        sD[i * SP + j] = (bug == 3 ? 1.f : a) * g;
      }
      const float da = dn_bsum(dap, red);
      if (tid == 0) {
        dGh[part * gs + r * lh + h] = nodecay ? 0.f : da * a;
        dBh[part * gs + r * lh + h] = dbeta;
      }
    }
  }
}
__global__ void dn_comb_k(const float* dQh, const float* dKh, const float* dGh, const float* dBh, long long hs, long long gs, float* dQ, float* dK, float* dG,
                          float* dB) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hs) {
    float q = 0.f, k = 0.f;
    for (int p = 0; p < DNS; ++p) { q = q + dQh[p * hs + i]; k = k + dKh[p * hs + i]; }
    dQ[i] = q; dK[i] = k;
  }
  if (i < gs) {
    float g = 0.f, bb = 0.f;
    for (int p = 0; p < DNS; ++p) { g = g + dGh[p * gs + i]; bb = bb + dBh[p * gs + i]; }
    dG[i] = g; dB[i] = bb;
  }
}
void deltanet_bwd(const float* Qn, const float* Kn, const float* V, int ldv, const float* G, const float* Beta, const float* dO, int B, int n, int lh, int dk,
                  int dv, bool nodecay, bool hasck, float* ws, float* dQn, float* dKn, float* dV, int lddv, float* dG, float* dBeta, int bug, cudaStream_t st) {
  const int nck = (n + DNC - 1) / DNC;
  const long long R = (long long)B * n;
  float* hbuf = ws + (long long)B * lh * (nck + DNC) * dk * dv;
  const long long hs = R * lh * dk, gs = R * lh;
  float* dQh = hbuf;
  float* dKh = dQh + DNS * hs;
  float* dGh = dKh + DNS * hs;
  float* dBh = dGh + DNS * gs;
  const int nblk = B * lh * DNS;
  if (dk == 128 && dv == 128) {
    const size_t sm = sizeof(float) * (DnSm<128, 32>::FLOATS + 3 * 256 + 64);
    static bool once = [&] { cudaFuncSetAttribute(dn_bwd_k<128, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); return true; }();
    (void)once;
    dn_bwd_k<128, 32><<<nblk, 256, sm, st>>>(Qn, Kn, V, ldv, G, Beta, dO, n, lh, nodecay, hasck, ws, dQh, dKh, dV, lddv, dGh, dBh, hs, bug);
  } else if (dk == 16 && dv == 16) {
    const size_t sm = sizeof(float) * (DnSm<16, 4>::FLOATS + 3 * 256 + 64);
    dn_bwd_k<16, 4><<<nblk, 256, sm, st>>>(Qn, Kn, V, ldv, G, Beta, dO, n, lh, nodecay, hasck, ws, dQh, dKh, dV, lddv, dGh, dBh, hs, bug);
  } else { std::fprintf(stderr, "deltanet_bwd dk %d dv %d\n", dk, dv); std::abort(); }
  KCK();
  dn_comb_k<<<nb(hs), 256, 0, st>>>(dQh, dKh, dGh, dBh, hs, gs, dQn, dKn, dG, dBeta);
  KCK();
}

// ---- 게이트 RMSNorm 의 뒤: 워프 = (행, 머리) ----
__global__ void __launch_bounds__(256) gnorm_bwd_k(const float* O, const float* Z, int ldz, int R, int lh, int dv, const float* w, float eps, const float* dY,
                                                   float* dO, float* dZ, uint16_t* dZb, int lddz, float* part) {
  const int wl = threadIdx.x >> 5, lane = threadIdx.x & 31;
  PAcc A;
  A.zero();
  for (int vi = wl; vi < RPB; vi += 8) {
    const int wi = blockIdx.x * RPB + vi;
    if (wi >= R * lh) break;
    const int r = wi / lh, h = wi % lh;
    const long long oo = (long long)r * lh * dv + h * dv, zo = (long long)r * ldz + h * dv;
    float s = 0.f;
    for (int d = lane; d < dv; d += 32) s = s + O[oo + d] * O[oo + d];
    s = wsum(s);
    const float rs = 1.f / sqrtf(s / (float)dv + eps);
    float g = 0.f;
    for (int d = lane; d < dv; d += 32) g = g + dY[oo + d] * w[d] * silu(Z[zo + d]) * O[oo + d];
    g = wsum(g);
    const float c3 = rs * rs * rs * g / (float)dv;
#pragma unroll
    for (int j = 0; j < PJ; ++j) {
      const int d = lane + 32 * j;
      if (d >= dv) break;
      const float z = Z[zo + d], o = O[oo + d], dy = dY[oo + d];
      const float vz = dy * w[d] * o * rs * dsilu(z);
      if (dZb) dZb[(long long)r * lddz + h * dv + d] = net::f2bf(vz);
      else dZ[(long long)r * lddz + h * dv + d] = vz;
      dO[oo + d] = rs * dy * w[d] * silu(z) - o * c3;
      A.a[j] = A.a[j] + dy * o * rs * silu(z);
    }
  }
  pacc_flush(A, dv, part, dv);
}
void gnorm_bwd(const float* O, const float* Z, int ldz, int R, int lh, int dv, const float* w, float eps, const float* dY, float* dO, float* dZ, int lddz,
               float* npart, float* gw, cudaStream_t st, uint16_t* dZb) {
  const int nbk = (R * lh + RPB - 1) / RPB;
  gnorm_bwd_k<<<nbk, 256, 0, st>>>(O, Z, ldz, R, lh, dv, w, eps, dY, dO, dZ, dZb, lddz, npart);
  colsum(npart, nbk, dv, dv, npart + (long long)nbk * dv, gw, false, st);
  KCK();
}

}  // namespace tk
}  // namespace rvla
