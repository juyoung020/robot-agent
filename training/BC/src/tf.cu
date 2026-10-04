// 토큰마다 학생 신경망 — 설명은 include/tf.h. GEMM 은 training/RL/network/src/gemm.cuh 의 gemm2_k(읽기만, 템플릿을 여기서 실체화)를,
// 옵티마이저는 net::adam_step 을 쓴다. 이 파일의 커널: 키 유효, 토큰 놓기, LayerNorm 앞·뒤, 어텐션 앞·뒤(판·머리마다 블록 하나),
// flow 입력·손실·오일러, dW 조각 합, 열 합(LN·종류 임베딩 기울기). 모두 결정적(고정 순서 합, 부동소수 원자 없음).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "detmath.h"
#include "gemm.cuh"
#include "tf.h"

namespace tfm {

using namespace net;

#define TCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

static long long al64(long long o) { return (o + 63) / 64 * 64; }
static WT mkw(TfLayout& t, long long& o, int N, int K) {
  WT w{o, N, K};
  o = al64(o + (long long)N * K);
  t.weights.push_back(w);
  return w;
}
static long long mkv(TfLayout& t, long long& o, int n) {
  const long long r = o;
  o = al64(o + n);
  t.vecs.push_back({r, n});
  return r;
}

TfLayout tf_layout(const TfCfg& c) {
  TfLayout t;
  long long o = 0;
  const int d = c.d, d1 = d + 16;
  for (int g = 0; g < N_GRP; ++g) {
    const bool mlp = g == G_OBJ;
    t.g_w1[g] = mkw(t, o, mlp ? c.obj_hidden : d, kGrp[g].K);
    t.g_w2[g] = mlp ? mkw(t, o, d, c.obj_hidden + 16) : WT{0, 0, 0};
    t.g_type[g] = mkv(t, o, kGrp[g].n_type * d);
  }
  auto blk = [&](bool ex) {
    TfLayout::Blk b{};
    b.ln1g = mkv(t, o, d); b.ln1b = mkv(t, o, d);
    b.qkv = mkw(t, o, 3 * d, d1);
    b.kvp = ex ? mkw(t, o, 2 * d, d1) : WT{0, 0, 0};
    b.wo = mkw(t, o, d, d1);
    b.ln2g = mkv(t, o, d); b.ln2b = mkv(t, o, d);
    b.w1 = mkw(t, o, c.mlp, d1);
    b.w2 = mkw(t, o, d, c.mlp + 16);
    return b;
  };
  for (int l = 0; l < c.layers; ++l) t.blk.push_back(blk(false));
  t.lnf_g = mkv(t, o, d); t.lnf_b = mkv(t, o, d);
  const int KA = (c.A + TEMB + 1 + 15) / 16 * 16;
  t.e_in = mkw(t, o, d, KA);
  for (int l = 0; l < c.e_layers; ++l) t.eblk.push_back(blk(true));
  t.lne_g = mkv(t, o, d); t.lne_b = mkv(t, o, d);
  t.e_out = mkw(t, o, c.A, d1);
  t.total = o;
  return t;
}

// ---- GEMM 실행(net_kernels.cu 의 gemm_launch 와 같은 타일 고르기 — 그쪽은 파일 안 static 이라 여기서 다시) ----
template <bool AT, bool BT, int EPI, int BM, int BN, int WM, int WN>
static void g2(const GemmG& g, cudaStream_t st) {
  const GemmP& p = g.p[0];
  constexpr int smem = 2 * 2 * (G2Tile<BM, AT>::ELEMS + G2Tile<BN, BT>::ELEMS);
  if (smem > 48 * 1024) {
    static const bool ok = [] {
      TCK(cudaFuncSetAttribute(gemm2_k<AT, BT, EPI, BM, BN, WM, WN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      return true;
    }();
    (void)ok;
  }
  gemm2_k<AT, BT, EPI, BM, BN, WM, WN><<<dim3((p.N + BN - 1) / BN, (p.M + BM - 1) / BM, g.zper), (BM / WM) * (BN / WN) * 32, smem, st>>>(g);
}
template <bool AT, bool BT, int EPI>
static void gl(const GemmP& p, int gz, cudaStream_t st) {
  const GemmG g{{p, p}, gz};
  if (p.N <= 16) g2<AT, BT, EPI, 128, 16, 32, 16>(g, st);
  else if (p.M <= 64) g2<AT, BT, EPI, 64, 64, 32, 32>(g, st);
  else if (p.N >= 128) g2<AT, BT, EPI, 128, 128, 64, 32>(g, st);
  else g2<AT, BT, EPI, 128, 64, 32, 32>(g, st);
  TCK(cudaGetLastError());
}
// 앞: out[M][ldo] = X[M][K]·W[N][K]ᵀ. kind 0 bf16 선형, 1 bf16 ELU, 2 f32, 3 f32 +=
static void gfwd(const uint16_t* X, int ldx, int M, const uint16_t* W, const WT& w, void* out, int ldo, int kind, cudaStream_t st) {
  GemmP p{};
  p.A = X; p.lda = ldx; p.B = W + w.off; p.ldb = w.K; p.M = M; p.N = w.N; p.K = w.K; p.C = out; p.ldc = ldo;
  p.act = kind == 1 ? ACT_ELU : ACT_LIN;
  if (kind <= 1) gl<false, false, EPI_ACT_BF16>(p, 1, st);
  else if (kind == 2) gl<false, false, EPI_F32>(p, 1, st);
  else gl<false, false, EPI_ACC_F32>(p, 1, st);
}
// dX: out[M][Np] = dZ[M][w.N]·W[:, 0:Np]. kind 0 bf16, 2 f32, 3 f32 +=, 4 bf16 ⊙ elu'(Y)
static void gdx(const uint16_t* dZ, int M, const uint16_t* W, const WT& w, int Np, void* out, int ldo, int kind, const uint16_t* Y, int ldy,
                cudaStream_t st) {
  GemmP p{};
  p.A = dZ; p.lda = w.N; p.B = W + w.off; p.ldb = w.K; p.M = M; p.N = Np; p.K = w.N; p.C = out; p.ldc = ldo; p.act = ACT_LIN; p.Y = Y; p.ldy = ldy;
  if (kind == 0) gl<false, true, EPI_ACT_BF16>(p, 1, st);
  else if (kind == 2) gl<false, true, EPI_F32>(p, 1, st);
  else if (kind == 3) gl<false, true, EPI_ACC_F32>(p, 1, st);
  else gl<false, true, EPI_DACT_BF16>(p, 1, st);
}
// 조각 합: G[off + i] = Σ_z ws[z·n + i] (고정 순서)
__global__ void dw_red_k(const float* ws, int splits, long long n, float* g) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = 0.f;
  for (int z = 0; z < splits; ++z) s = s + ws[(long long)z * n + i];
  g[i] = s;
}
// dW = dZᵀ·X (행 M 개, split-K) → G
static void gdw(const uint16_t* dZ, const uint16_t* X, int ldx, int M, const WT& w, float* ws, int chunk, float* G, cudaStream_t st) {
  GemmP p{};
  p.A = dZ; p.lda = w.N; p.B = X; p.ldb = ldx; p.M = w.N; p.N = w.K; p.K = M; p.C = ws; p.ldc = w.K; p.kchunk = chunk;
  const int sp = (M + chunk - 1) / chunk;
  gl<true, true, EPI_SPLIT_F32>(p, sp, st);
  const long long n = (long long)w.N * w.K;
  dw_red_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(ws, sp, n, G + w.off);
  TCK(cudaGetLastError());
}

// ---- 키 유효: tv[b][t] ----
__global__ void tv_k(const uint32_t* obj_mask, const uint32_t* grp_off, int B, uint8_t* tv) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * L_TOK) return;
  const int b = q / L_TOK, t = q % L_TOK;
  int g = 0;
  while (g + 1 < N_GRP && t >= grp_tok0(g + 1)) ++g;
  bool v = !(grp_off && ((grp_off[b] >> g) & 1u));
  if (g == G_OBJ) v = v && ((obj_mask[b] >> (t - grp_tok0(G_OBJ))) & 1u);
  tv[q] = v ? 1 : 0;
}
NDEV int type_of(int g, int t) { return grp_ntype(g) == 2 ? (t >= grp_ntok(g) / 2 ? 1 : 0) : 0; }
// 토큰 놓기: X[b·L + tok0 + t][c] = E(묶음 순서)[b·n + t][c] + 종류[c]
struct GT { long long o[N_GRP]; };
__global__ void place_k(const float* E, const float* P, GT lay, int B, int d, float* X) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)B * L_TOK * d) return;
  const int c = (int)(q % d);
  const long long row = q / d;
  const int b = (int)(row / L_TOK), tk = (int)(row % L_TOK);
  int g = 0;
  while (g + 1 < N_GRP && tk >= grp_tok0(g + 1)) ++g;
  const int t = tk - grp_tok0(g);
  const float e = E[((long long)B * grp_tok0(g) + (long long)b * grp_ntok(g) + t) * d + c];
  X[q] = e + P[lay.o[g] + (long long)type_of(g, t) * d + c];
}
// 묶음 순서 bf16 dE = 표본 순서 dX0
__global__ void gather_dE_k(const float* dR, int B, int d, uint16_t* dE) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)B * L_TOK * d) return;
  const int c = (int)(q % d);
  const long long row = q / d;
  const int b = (int)(row / L_TOK), tk = (int)(row % L_TOK);
  int g = 0;
  while (g + 1 < N_GRP && tk >= grp_tok0(g + 1)) ++g;
  const int t = tk - grp_tok0(g);
  dE[((long long)B * grp_tok0(g) + (long long)b * grp_ntok(g) + t) * d + c] = f2bf(dR[q]);
}

// ---- LayerNorm(워프 하나 = 행 하나, 통계 FP32). 출력 bf16 [R][ldo](d 칸 = 1 은 미리 써 둠) ----
constexpr float LN_EPS = 1e-5f;
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
constexpr int MAXC = 32;   // d ≤ 1024
__global__ void ln_fwd_k(const float* X, int R, int d, const float* gm, const float* bt, uint16_t* out, int ldo, float* mu, float* rs) {
  const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (r >= R) return;
  const float* x = X + (long long)r * d;
  const int nc = d / 32;
  float v[MAXC];
  float s = 0.f;
  for (int i = 0; i < nc; ++i) { v[i] = x[lane + 32 * i]; s = s + v[i]; }
  const float m = wsum(s) / (float)d;
  float q = 0.f;
  for (int i = 0; i < nc; ++i) { const float e = v[i] - m; q = q + e * e; }
  const float var = wsum(q) / (float)d;
  const float rsd = 1.f / sqrtf(var + LN_EPS);
  for (int i = 0; i < nc; ++i) {
    const int c = lane + 32 * i;
    out[(long long)r * ldo + c] = f2bf((v[i] - m) * rsd * gm[c] + bt[c]);
  }
  if (lane == 0) { mu[r] = m; rs[r] = rsd; }
}
// dR[r] += rs·(g − mean(g) − x̂·mean(g·x̂)), g = dy·γ
__global__ void ln_bwd_k(const float* dy, const float* X, int R, int d, const float* gm, const float* mu, const float* rs, int bug, float* dR) {
  const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (r >= R) return;
  const int nc = d / 32;
  const float m = mu[r], rsd = rs[r];
  float gv[MAXC], xh[MAXC];
  float s1 = 0.f, s2 = 0.f;
  for (int i = 0; i < nc; ++i) {
    const int c = lane + 32 * i;
    xh[i] = (X[(long long)r * d + c] - m) * rsd;
    gv[i] = dy[(long long)r * d + c] * gm[c];
    s1 = s1 + gv[i];
    s2 = s2 + gv[i] * xh[i];
  }
  const float m1 = bug == 2 ? 0.f : wsum(s1) / (float)d, m2 = wsum(s2) / (float)d;
  for (int i = 0; i < nc; ++i) {
    const int c = lane + 32 * i;
    float* o = dR + (long long)r * d + c;
    *o = *o + rsd * (gv[i] - m1 - xh[i] * m2);
  }
}
// LN 변수 기울기 조각: 블록 = 행 256 개, 스레드 = 열. cp[0][blk][c] = Σ dy·x̂, cp[1][blk][c] = Σ dy
constexpr int CR = 256;
__global__ void lnp_part_k(const float* dy, const float* X, int R, int d, const float* mu, const float* rs, float* cp) {
  const int nb = gridDim.x;
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float a = 0.f, b = 0.f;
    const int r1 = min(R, (int)(blockIdx.x + 1) * CR);
    for (int r = blockIdx.x * CR; r < r1; ++r) {
      const float g = dy[(long long)r * d + c];
      a = a + g * ((X[(long long)r * d + c] - mu[r]) * rs[r]);
      b = b + g;
    }
    cp[(long long)blockIdx.x * d + c] = a;
    cp[((long long)nb + blockIdx.x) * d + c] = b;
  }
}
// 조각 합: out[c] = Σ_k cp[k][c] (+ 둘째 묶음은 out2)
__global__ void colred_k(const float* cp, int nb, int d, float* out, float* out2) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= d) return;
  float a = 0.f, b = 0.f;
  for (int k = 0; k < nb; ++k) a = a + cp[(long long)k * d + c];
  out[c] = a;
  if (out2) {
    for (int k = 0; k < nb; ++k) b = b + cp[((long long)nb + k) * d + c];
    out2[c] = b;
  }
}
// 종류 임베딩 기울기 조각: 묶음 g 종류 ty 의 토큰들(판 b, 토큰 t0..t1) — 블록 = 판 cb 개
__global__ void temb_part_k(const float* dR, int B, int d, int tk0, int t0, int t1, int cb, float* cp) {
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float a = 0.f;
    const int b1 = min(B, (int)(blockIdx.x + 1) * cb);
    for (int b = blockIdx.x * cb; b < b1; ++b)
      for (int t = t0; t < t1; ++t) a = a + dR[((long long)b * L_TOK + tk0 + t) * d + c];
    cp[(long long)blockIdx.x * d + c] = a;
  }
}
__global__ void f2b_k(const float* x, long long n, uint16_t* y) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = f2bf(x[i]);
}
// 줄 rows × ld 의 열 col = 1, 열 col+1..ld-1 = 0
__global__ void ones_col_k(uint16_t* buf, long long rows, int ld, int col) {
  const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  for (int c = col; c < ld; ++c) buf[r * ld + c] = c == col ? (uint16_t)0x3f80 : (uint16_t)0;
}

// ---- 어텐션: 블록 = (판, 머리), 워프 4 개. 키·값 = [구간 1 (n1, 가림 valid1) ; 구간 2 (n2)]. 공유 메모리 줄 간격 66 bf16(33 낱말 → 은행 충돌 없음) ----
struct AttnP {
  const uint16_t* q; int ldq, nq;
  const uint16_t *k1, *v1; int ld1, n1;
  const uint16_t *k2, *v2; int ld2, n2;
  const uint8_t* valid1;
  uint16_t* o; int ldo;
  float* lse;
  int heads; float scale; int nomask, noscale;
  const uint16_t* dO; int lddo;
  float* D;
  uint16_t* dq; int lddq;
  uint16_t *dk1, *dv1; int lddk1;
  uint16_t *dk2, *dv2; int lddk2;
};
constexpr int AW = 8, SL = 66;
__device__ __forceinline__ float bfw(uint32_t w, int hi) { return bf2f((uint16_t)(hi ? (w >> 16) : (w & 0xffffu))); }
// 행 64 bf16 → 공유(줄 간격 66) 32 비트 낱말로
__device__ __forceinline__ void ld_row(uint16_t* s, const uint16_t* g, int lane) {
  reinterpret_cast<uint32_t*>(s)[lane] = reinterpret_cast<const uint32_t*>(g)[lane];
}
__device__ __forceinline__ const uint16_t* krow(const AttnP& p, int b, int h, int j, bool val) {
  if (j < p.n1) return (val ? p.v1 : p.k1) + ((long long)b * p.n1 + j) * p.ld1 + h * DH;
  return (val ? p.v2 : p.k2) + ((long long)b * p.n2 + (j - p.n1)) * p.ld2 + h * DH;
}
__device__ __forceinline__ bool kvalid(const AttnP& p, int b, int j) { return p.nomask || j >= p.n1 || !p.valid1 || p.valid1[(long long)b * p.n1 + j]; }
// q(f32 64, 공유)·k(bf16 줄) 차례 합
__device__ __forceinline__ float dot_qk(const float* q, const uint16_t* k) {
  const uint32_t* kw = reinterpret_cast<const uint32_t*>(k);
  float s = 0.f;
#pragma unroll 8
  for (int e = 0; e < DH / 2; ++e) {
    const uint32_t w = kw[e];
    s = s + q[2 * e] * bfw(w, 0);
    s = s + q[2 * e + 1] * bfw(w, 1);
  }
  return s;
}
__global__ void __launch_bounds__(AW * 32) attn_fwd_k(const AttnP p) {
  extern __shared__ __align__(16) uint8_t asm_[];
  const int b = blockIdx.x / p.heads, h = blockIdx.x % p.heads, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nk = p.n1 + p.n2;
  uint16_t* sK = reinterpret_cast<uint16_t*>(asm_);
  uint16_t* sV = sK + nk * SL;
  float* sw = reinterpret_cast<float*>(sV + nk * SL + (nk & 1) * 2);
  float* qs = sw + warp * (DH + nk);
  float* ps = qs + DH;
  for (int j = warp; j < nk; j += AW) { ld_row(sK + j * SL, krow(p, b, h, j, false), lane); ld_row(sV + j * SL, krow(p, b, h, j, true), lane); }
  __syncthreads();
  for (int i = warp; i < p.nq; i += AW) {
    const uint32_t qw = reinterpret_cast<const uint32_t*>(p.q + ((long long)b * p.nq + i) * p.ldq + h * DH)[lane];
    qs[2 * lane] = bfw(qw, 0); qs[2 * lane + 1] = bfw(qw, 1);
    __syncwarp();
    float m = -INFINITY;
    for (int j = lane; j < nk; j += 32) {
      const float s = kvalid(p, b, j) ? dot_qk(qs, sK + j * SL) * p.scale : -INFINITY;
      ps[j] = s;
      m = fmaxf(m, s);
    }
    m = wmax(m);
    float sum = 0.f;
    for (int j = lane; j < nk; j += 32) {
      const float e = ps[j] > -INFINITY ? expf(ps[j] - m) : 0.f;
      ps[j] = e;
      sum = sum + e;
    }
    sum = wsum(sum);
    __syncwarp();
    float o0 = 0.f, o1 = 0.f;
    for (int j = 0; j < nk; ++j) {
      const uint32_t w = reinterpret_cast<const uint32_t*>(sV + j * SL)[lane];
      o0 = o0 + ps[j] * bfw(w, 0);
      o1 = o1 + ps[j] * bfw(w, 1);
    }
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    reinterpret_cast<uint32_t*>(p.o + ((long long)b * p.nq + i) * p.ldo + h * DH)[lane] = (uint32_t)f2bf(o0 * inv) | ((uint32_t)f2bf(o1 * inv) << 16);
    if (lane == 0) p.lse[((long long)b * p.heads + h) * p.nq + i] = sum > 0.f ? m + logf(sum) : 0.f;
    __syncwarp();
  }
}
// dQ (+ D_i = dO_i·O_i 를 전역에): 키·값 공유
__global__ void __launch_bounds__(AW * 32) attn_dq_k(const AttnP p) {
  extern __shared__ __align__(16) uint8_t asm_[];
  const int b = blockIdx.x / p.heads, h = blockIdx.x % p.heads, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nk = p.n1 + p.n2;
  uint16_t* sK = reinterpret_cast<uint16_t*>(asm_);
  uint16_t* sV = sK + nk * SL;
  float* sw = reinterpret_cast<float*>(sV + nk * SL + (nk & 1) * 2);
  float* qs = sw + warp * (2 * DH + nk);
  float* dos = qs + DH;
  float* ds = dos + DH;
  for (int j = warp; j < nk; j += AW) { ld_row(sK + j * SL, krow(p, b, h, j, false), lane); ld_row(sV + j * SL, krow(p, b, h, j, true), lane); }
  __syncthreads();
  const float sc = p.noscale ? 1.f : p.scale;
  for (int i = warp; i < p.nq; i += AW) {
    const long long qi = (long long)b * p.nq + i;
    const uint32_t qw = reinterpret_cast<const uint32_t*>(p.q + qi * p.ldq + h * DH)[lane];
    const uint32_t dw = reinterpret_cast<const uint32_t*>(p.dO + qi * p.lddo + h * DH)[lane];
    const uint32_t ow = reinterpret_cast<const uint32_t*>(p.o + qi * p.ldo + h * DH)[lane];
    qs[2 * lane] = bfw(qw, 0); qs[2 * lane + 1] = bfw(qw, 1);
    dos[2 * lane] = bfw(dw, 0); dos[2 * lane + 1] = bfw(dw, 1);
    const float Di = wsum(dos[2 * lane] * bfw(ow, 0) + dos[2 * lane + 1] * bfw(ow, 1));
    const float ls = p.lse[((long long)b * p.heads + h) * p.nq + i];
    __syncwarp();
    for (int j = lane; j < nk; j += 32) {
      float v = 0.f;
      if (kvalid(p, b, j)) {
        const float pr = expf(dot_qk(qs, sK + j * SL) * p.scale - ls);
        v = pr * (dot_qk(dos, sV + j * SL) - Di);
      }
      ds[j] = v;
    }
    __syncwarp();
    float a0 = 0.f, a1 = 0.f;
    for (int j = 0; j < nk; ++j) {
      const uint32_t w = reinterpret_cast<const uint32_t*>(sK + j * SL)[lane];
      a0 = a0 + ds[j] * bfw(w, 0);
      a1 = a1 + ds[j] * bfw(w, 1);
    }
    reinterpret_cast<uint32_t*>(p.dq + qi * p.lddq + h * DH)[lane] = (uint32_t)f2bf(a0 * sc) | ((uint32_t)f2bf(a1 * sc) << 16);
    if (lane == 0) p.D[((long long)b * p.heads + h) * p.nq + i] = Di;
    __syncwarp();
  }
}
// dK·dV: 질의·dO 공유, 워프 하나 = 키 하나
__global__ void __launch_bounds__(AW * 32) attn_dkv_k(const AttnP p) {
  extern __shared__ __align__(16) uint8_t asm_[];
  const int b = blockIdx.x / p.heads, h = blockIdx.x % p.heads, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nk = p.n1 + p.n2, nq = p.nq;
  uint16_t* sQ = reinterpret_cast<uint16_t*>(asm_);
  uint16_t* sD = sQ + nq * SL;
  float* sl = reinterpret_cast<float*>(sD + nq * SL + (nq & 1) * 2);   // lse[nq], D[nq]
  float* sDi = sl + nq;
  float* sw = sDi + nq;
  float* ks = sw + warp * (2 * DH + 2 * nq);
  float* vs = ks + DH;
  float* ps = vs + DH;
  float* ds = ps + nq;
  for (int i = warp; i < nq; i += AW) {
    const long long qi = (long long)b * nq + i;
    ld_row(sQ + i * SL, p.q + qi * p.ldq + h * DH, lane);
    ld_row(sD + i * SL, p.dO + qi * p.lddo + h * DH, lane);
  }
  for (int i = threadIdx.x; i < nq; i += AW * 32) {
    sl[i] = p.lse[((long long)b * p.heads + h) * nq + i];
    sDi[i] = p.D[((long long)b * p.heads + h) * nq + i];
  }
  __syncthreads();
  const float sc = p.noscale ? 1.f : p.scale;
  for (int j = warp; j < nk; j += AW) {
    const uint32_t kw = reinterpret_cast<const uint32_t*>(krow(p, b, h, j, false))[lane];
    const uint32_t vw = reinterpret_cast<const uint32_t*>(krow(p, b, h, j, true))[lane];
    ks[2 * lane] = bfw(kw, 0); ks[2 * lane + 1] = bfw(kw, 1);
    vs[2 * lane] = bfw(vw, 0); vs[2 * lane + 1] = bfw(vw, 1);
    const bool val = kvalid(p, b, j);
    __syncwarp();
    for (int i = lane; i < nq; i += 32) {
      float pr = 0.f, dv = 0.f;
      if (val) {
        pr = expf(dot_qk(ks, sQ + i * SL) * p.scale - sl[i]);
        dv = pr * (dot_qk(vs, sD + i * SL) - sDi[i]);
      }
      ps[i] = pr;
      ds[i] = dv;
    }
    __syncwarp();
    float v0 = 0.f, v1 = 0.f, k0 = 0.f, k1 = 0.f;
    for (int i = 0; i < nq; ++i) {
      const uint32_t dw = reinterpret_cast<const uint32_t*>(sD + i * SL)[lane];
      const uint32_t qw = reinterpret_cast<const uint32_t*>(sQ + i * SL)[lane];
      v0 = v0 + ps[i] * bfw(dw, 0); v1 = v1 + ps[i] * bfw(dw, 1);
      k0 = k0 + ds[i] * bfw(qw, 0); k1 = k1 + ds[i] * bfw(qw, 1);
    }
    uint16_t *dk, *dvp;
    if (j < p.n1) { const long long r = ((long long)b * p.n1 + j) * p.lddk1 + h * DH; dk = p.dk1 + r; dvp = p.dv1 + r; }
    else { const long long r = ((long long)b * p.n2 + (j - p.n1)) * p.lddk2 + h * DH; dk = p.dk2 + r; dvp = p.dv2 + r; }
    reinterpret_cast<uint32_t*>(dk)[lane] = (uint32_t)f2bf(k0 * sc) | ((uint32_t)f2bf(k1 * sc) << 16);
    reinterpret_cast<uint32_t*>(dvp)[lane] = (uint32_t)f2bf(v0) | ((uint32_t)f2bf(v1) << 16);
    __syncwarp();
  }
}
static size_t smem_fwd(int nk, int extra) { return (size_t)2 * nk * SL * 2 + 4 + (size_t)AW * (extra + nk) * 4; }
static void attn_fwd(const AttnP& p, int B, cudaStream_t st) {
  const int nk = p.n1 + p.n2;
  const size_t sm = smem_fwd(nk, DH);
  attn_fwd_k<<<B * p.heads, AW * 32, sm, st>>>(p);
  TCK(cudaGetLastError());
}
static void attn_bwd(const AttnP& p, int B, cudaStream_t st) {
  const int nk = p.n1 + p.n2;
  attn_dq_k<<<B * p.heads, AW * 32, smem_fwd(nk, 2 * DH), st>>>(p);
  const size_t sm2 = (size_t)2 * p.nq * SL * 2 + 4 + (size_t)2 * p.nq * 4 + (size_t)AW * (2 * DH + 2 * p.nq) * 4;
  attn_dkv_k<<<B * p.heads, AW * 32, sm2, st>>>(p);
  TCK(cudaGetLastError());
}

// ---- flow matching ----
__global__ void flow_in_k(const float* chunk, const float* cmask, const long long* iter, uint64_t seed, int B, int H, int A, int KA, uint16_t* ain,
                          float* tau, float* eps, float* u) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const uint64_t it = (uint64_t)iter[0];
  const float tt = u01(hash4(seed ^ 0xF10Full, it, (uint64_t)b, 0x7a75ull));
  tau[b] = tt;
  float te[TEMB];
  temb_f(tt, te);
  for (int h = 0; h < H; ++h) {
    uint16_t* row = ain + ((long long)b * H + h) * KA;
    const bool on = cmask[(long long)b * H + h] > 0.f;
    for (int k = 0; k < A; ++k) {
      const long long q = ((long long)b * H + h) * A + k;
      const float a = on ? chunk[q] : 0.f;
      const float e = gauss(hash4(seed ^ 0xF10Full, it, (uint64_t)b, 0x1000ull + (uint64_t)(h * A + k)));
      eps[q] = e;
      row[k] = f2bf(tt * e + (1.f - tt) * a);
      u[q] = e - a;
    }
    for (int k = 0; k < TEMB; ++k) row[A + k] = f2bf(te[k]);
  }
}
constexpr int LT = 256;
__global__ void __launch_bounds__(LT) loss_k(const float* vel, const float* u, const float* cmask, const uint32_t* adim, int B, int H, int A, int bug,
                                             uint16_t* dz, float* part) {
  __shared__ float sh[LT];
  const int r = blockIdx.x * LT + threadIdx.x;
  float q = 0.f;
  if (r < B * H) {
    const uint32_t am = adim[0];
    const float m = cmask[r], inv = 1.f / (float)B, two = bug == 4 ? 1.f : 2.f;
    for (int k = 0; k < A; ++k) {
      const float on = ((am >> k) & 1u) ? m : 0.f;
      const float dd = vel[(long long)r * A + k] - u[(long long)r * A + k];
      q = q + on * dd * dd;
      dz[(long long)r * A + k] = f2bf(two * on * dd * inv);
    }
  }
  sh[threadIdx.x] = q;
  __syncthreads();
  for (int s = LT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] = sh[threadIdx.x] + sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
__global__ void loss_red_k(const float* part, int nb, int B, float* out) {
  if (threadIdx.x != 0) return;
  float s = 0.f;
  for (int k = 0; k < nb; ++k) s = s + part[k];
  out[0] = s / (float)B;
}
__global__ void noise_k(const long long* key, int t, uint64_t seed, int B, int H, int A, float* x) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (long long)B * H * A) return;
  const int b = (int)(q / (H * A)), j = (int)(q % (H * A));
  x[q] = gauss(hash4(seed ^ 0x1AF5ull, (uint64_t)key[0], (uint64_t)t * 65536ull + (uint64_t)b, (uint64_t)j));
}
__global__ void ain_k(const float* x, float tau, int B, int H, int A, int KA, uint16_t* ain) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= B * H) return;
  float te[TEMB];
  temb_f(tau, te);
  uint16_t* row = ain + (long long)r * KA;
  for (int k = 0; k < A; ++k) row[k] = f2bf(x[(long long)r * A + k]);
  for (int k = 0; k < TEMB; ++k) row[A + k] = f2bf(te[k]);
}
__global__ void euler_k(const float* v, float dt, long long n, int A, const uint32_t* adim, int last, float* x, float* act) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n) return;
  x[q] = x[q] - dt * v[q];
  if (last) act[q] = ((adim[0] >> (q % A)) & 1u) ? x[q] : 0.f;
}

// ---------------------------------------------------------------------------------------------------------------------
template <class T_>
T_* Tf::alloc(size_t n) {
  void* p = nullptr;
  TCK(cudaMalloc(&p, sizeof(T_) * n + 16));
  TCK(cudaMemset(p, 0, sizeof(T_) * n + 16));
  allocs.push_back(p);
  bytes += sizeof(T_) * n;
  return reinterpret_cast<T_*>(p);
}
static void ones(uint16_t* buf, long long rows, int ld, int col) {
  ones_col_k<<<(unsigned)((rows + 255) / 256), 256>>>(buf, rows, ld, col);
  TCK(cudaGetLastError());
}

void Tf::init(const TfCfg& cfg) {
  c = cfg;
  if (c.d % 64 || c.d / DH != c.heads || c.d > 1024 || c.mlp % 16 || c.obj_hidden % 16 || c.A > MAX_A || c.A % 8 || c.dw_chunk % 32) {
    std::fprintf(stderr, "tf: bad config (d multiple of 64 = heads·64 ≤ 1024, mlp·obj_hidden multiples of 16, A multiple of 8 ≤ 16)\n");
    std::abort();
  }
  lay = tf_layout(c);
  d1 = c.d + 16;
  KA = (c.A + TEMB + 1 + 15) / 16 * 16;
  const int d = c.d, B = c.Bmax, H = c.H, Lb = c.layers, Le = c.e_layers;
  const long long R = (long long)B * L, RA = (long long)B * H;
  P = alloc<float>(lay.total); G = alloc<float>(lay.total); Am = alloc<float>(lay.total); Av = alloc<float>(lay.total);
  Pb = alloc<uint16_t>(lay.total);
  gn_part = alloc<float>(GN_BLOCKS);
  tv = alloc<uint8_t>(R);
  E = alloc<float>(R * d);
  oh = alloc<uint16_t>((size_t)B * 16 * (c.obj_hidden + 16));
  ones(oh, (long long)B * 16, c.obj_hidden + 16, c.obj_hidden);
  for (int k = 0; k <= 2 * Lb; ++k) Xs.push_back(alloc<float>(R * d));
  for (int l = 0; l < Lb; ++l) {
    A1.push_back(alloc<uint16_t>(R * d1)); ones(A1.back(), R, d1, d);
    A2.push_back(alloc<uint16_t>(R * d1)); ones(A2.back(), R, d1, d);
    QKV.push_back(alloc<uint16_t>(R * 3 * d));
    O.push_back(alloc<uint16_t>(R * d1)); ones(O.back(), R, d1, d);
    Hh.push_back(alloc<uint16_t>(R * (c.mlp + 16))); ones(Hh.back(), R, c.mlp + 16, c.mlp);
    mu1.push_back(alloc<float>(R)); rs1.push_back(alloc<float>(R)); mu2.push_back(alloc<float>(R)); rs2.push_back(alloc<float>(R));
    lse.push_back(alloc<float>((size_t)B * c.heads * L)); Dd.push_back(alloc<float>((size_t)B * c.heads * L));
  }
  muf = alloc<float>(R); rsf = alloc<float>(R);
  Pf = alloc<uint16_t>(R * d1); ones(Pf, R, d1, d);
  ain = alloc<uint16_t>(RA * KA); ones(ain, RA, KA, c.A + TEMB);
  tau = alloc<float>(B); eps = alloc<float>(RA * c.A); u = alloc<float>(RA * c.A);
  for (int k = 0; k <= 2 * Le; ++k) Xa.push_back(alloc<float>(RA * d));
  for (int l = 0; l < Le; ++l) {
    eA1.push_back(alloc<uint16_t>(RA * d1)); ones(eA1.back(), RA, d1, d);
    eA2.push_back(alloc<uint16_t>(RA * d1)); ones(eA2.back(), RA, d1, d);
    eQKV.push_back(alloc<uint16_t>(RA * 3 * d));
    ePKV.push_back(alloc<uint16_t>(R * 2 * d));
    eO.push_back(alloc<uint16_t>(RA * d1)); ones(eO.back(), RA, d1, d);
    eH.push_back(alloc<uint16_t>(RA * (c.mlp + 16))); ones(eH.back(), RA, c.mlp + 16, c.mlp);
    emu1.push_back(alloc<float>(RA)); ers1.push_back(alloc<float>(RA)); emu2.push_back(alloc<float>(RA)); ers2.push_back(alloc<float>(RA));
    else_.push_back(alloc<float>((size_t)B * c.heads * H)); eD.push_back(alloc<float>((size_t)B * c.heads * H));
  }
  muo = alloc<float>(RA); rso = alloc<float>(RA);
  Ao = alloc<uint16_t>(RA * d1); ones(Ao, RA, d1, d);
  vel = alloc<float>(RA * c.A);
  dR = alloc<float>(R * d); dT = alloc<float>(R * d);
  dRb = alloc<uint16_t>(R * d); dQKV = alloc<uint16_t>(R * 3 * d); dHh = alloc<uint16_t>(R * c.mlp); dOb = alloc<uint16_t>(R * d);
  dRa = alloc<float>(RA * d); dTa = alloc<float>(RA * d); dPf = alloc<float>(R * d);
  dRab = alloc<uint16_t>(RA * d); dQKVa = alloc<uint16_t>(RA * 3 * d); dPKV = alloc<uint16_t>(R * 2 * d); dHa = alloc<uint16_t>(RA * c.mlp);
  dOa = alloc<uint16_t>(RA * d); dz = alloc<uint16_t>(RA * c.A);
  dEb = alloc<uint16_t>(R * d); doh = alloc<uint16_t>((size_t)B * 16 * c.obj_hidden);
  long long wsn = 0;
  auto need = [&](const WT& w, long long rows) { const long long s = (rows + c.dw_chunk - 1) / c.dw_chunk; if (s * w.N * w.K > wsn) wsn = s * w.N * w.K; };
  for (int g = 0; g < N_GRP; ++g) { need(lay.g_w1[g], (long long)B * kGrp[g].n_tok); if (lay.g_w2[g].N) need(lay.g_w2[g], (long long)B * kGrp[g].n_tok); }
  for (auto& b : lay.blk) { need(b.qkv, R); need(b.wo, R); need(b.w1, R); need(b.w2, R); }
  for (auto& b : lay.eblk) { need(b.qkv, RA); need(b.kvp, R); need(b.wo, RA); need(b.w1, RA); need(b.w2, RA); }
  need(lay.e_in, RA); need(lay.e_out, RA);
  ws = alloc<float>(wsn);
  cpart = alloc<float>((size_t)2 * ((R + CR - 1) / CR + 1) * d);
  lpart = alloc<float>((RA + LT - 1) / LT + 1);
  loss_d = alloc<float>(4);
  // 어텐션 공유 메모리 한도(99 KB 까지)
  const int big = 99 * 1024;
  TCK(cudaFuncSetAttribute(attn_fwd_k, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  TCK(cudaFuncSetAttribute(attn_dq_k, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  TCK(cudaFuncSetAttribute(attn_dkv_k, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  if (smem_fwd(L + H, 2 * DH) > (size_t)big) { std::fprintf(stderr, "tf: attention shared memory too large\n"); std::abort(); }

  // 초기화(호스트, 결정적): 가중치 균등 ±gain·√(3/fan_in)(1 칸 뒤 0), 편향 0, LN γ 1 β 0, 종류 임베딩 ±0.02
  std::vector<float> hp(lay.total, 0.f);
  uint64_t s = c.seed * 0x9E3779B97F4A7C15ull + 0x7f4a7c15ull;
  auto fill = [&](const WT& w, int kreal, float gain) {
    const float a = gain * std::sqrt(3.f / (float)kreal);
    for (int n = 0; n < w.N; ++n)
      for (int k = 0; k < kreal; ++k) hp[w.off + (long long)n * w.K + k] = dm::rand_range(s, -a, a);
  };
  const float g2v = 1.41421356f, res = 1.f / std::sqrt(2.f * (float)(Lb + Le));
  for (int g = 0; g < N_GRP; ++g) {
    if (lay.g_w2[g].N) { fill(lay.g_w1[g], kGrp[g].k_real, g2v); fill(lay.g_w2[g], c.obj_hidden, 1.f); }
    else fill(lay.g_w1[g], kGrp[g].k_real, 1.f);
    for (int k = 0; k < kGrp[g].n_type * d; ++k) hp[lay.g_type[g] + k] = dm::rand_range(s, -0.02f, 0.02f);
  }
  auto fblk = [&](const TfLayout::Blk& b, bool ex) {
    for (int k = 0; k < d; ++k) { hp[b.ln1g + k] = 1.f; hp[b.ln2g + k] = 1.f; }
    fill(b.qkv, d, 1.f);
    if (ex) fill(b.kvp, d, 1.f);
    fill(b.wo, d, res);
    fill(b.w1, d, g2v);
    fill(b.w2, c.mlp, res);
  };
  for (auto& b : lay.blk) fblk(b, false);
  for (int k = 0; k < d; ++k) { hp[lay.lnf_g + k] = 1.f; hp[lay.lne_g + k] = 1.f; }
  fill(lay.e_in, c.A + TEMB, 1.f);
  for (auto& b : lay.eblk) fblk(b, true);
  fill(lay.e_out, d, 0.01f);
  TCK(cudaMemcpy(P, hp.data(), sizeof(float) * lay.total, cudaMemcpyHostToDevice));
  to_bf16(P, Pb, lay.total, 0);
  TCK(cudaDeviceSynchronize());
}

void Tf::free_all() {
  for (void* p : allocs) cudaFree(p);
  allocs.clear();
}

static unsigned nb_(long long n, int t = 256) { return (unsigned)((n + t - 1) / t); }

static AttnP self_attn(const Tf& t, int l, int B) {
  AttnP p{};
  const int d = t.c.d;
  p.q = t.QKV[l]; p.ldq = 3 * d; p.nq = t.L;
  p.k1 = t.QKV[l] + d; p.v1 = t.QKV[l] + 2 * d; p.ld1 = 3 * d; p.n1 = t.L;
  p.n2 = 0;
  p.valid1 = t.tv;
  p.o = t.O[l]; p.ldo = t.d1; p.lse = t.lse[l];
  p.heads = t.c.heads; p.scale = 1.f / std::sqrt((float)DH); p.nomask = t.bug == 3; p.noscale = t.bug == 1;
  p.D = t.Dd[l];
  (void)B;
  return p;
}
static AttnP ex_attn(const Tf& t, int l) {
  AttnP p{};
  const int d = t.c.d;
  p.q = t.eQKV[l]; p.ldq = 3 * d; p.nq = t.c.H;
  p.k1 = t.ePKV[l]; p.v1 = t.ePKV[l] + d; p.ld1 = 2 * d; p.n1 = t.L;
  p.k2 = t.eQKV[l] + d; p.v2 = t.eQKV[l] + 2 * d; p.ld2 = 3 * d; p.n2 = t.c.H;
  p.valid1 = t.tv;
  p.o = t.eO[l]; p.ldo = t.d1; p.lse = t.else_[l];
  p.heads = t.c.heads; p.scale = 1.f / std::sqrt((float)DH); p.nomask = t.bug == 3; p.noscale = t.bug == 1;
  p.D = t.eD[l];
  return p;
}

void Tf::forward_prefix(const TfIn& in, int B, cudaStream_t st) {
  const int d = c.d;
  const long long R = (long long)B * L;
  tv_k<<<nb_(R), 256, 0, st>>>(in.obj_mask, in.grp_off, B, tv);
  for (int g = 0; g < N_GRP; ++g) {
    const int rows = B * kGrp[g].n_tok;
    float* Eg = E + (long long)B * grp_tok0(g) * d;
    if (lay.g_w2[g].N) {
      gfwd(in.g[g], kGrp[g].K, rows, Pb, lay.g_w1[g], oh, c.obj_hidden + 16, 1, st);
      gfwd(oh, c.obj_hidden + 16, rows, Pb, lay.g_w2[g], Eg, d, 2, st);
    } else {
      gfwd(in.g[g], kGrp[g].K, rows, Pb, lay.g_w1[g], Eg, d, 2, st);
    }
  }
  GT gt;
  for (int g = 0; g < N_GRP; ++g) gt.o[g] = lay.g_type[g];
  place_k<<<nb_(R * d), 256, 0, st>>>(E, P, gt, B, d, Xs[0]);
  const size_t xb = sizeof(float) * R * d;
  for (int l = 0; l < c.layers; ++l) {
    const auto& w = lay.blk[l];
    ln_fwd_k<<<nb_(R, 8), 256, 0, st>>>(Xs[2 * l], (int)R, d, P + w.ln1g, P + w.ln1b, A1[l], d1, mu1[l], rs1[l]);
    gfwd(A1[l], d1, (int)R, Pb, w.qkv, QKV[l], 3 * d, 0, st);
    attn_fwd(self_attn(*this, l, B), B, st);
    TCK(cudaMemcpyAsync(Xs[2 * l + 1], Xs[2 * l], xb, cudaMemcpyDeviceToDevice, st));
    gfwd(O[l], d1, (int)R, Pb, w.wo, Xs[2 * l + 1], d, 3, st);
    ln_fwd_k<<<nb_(R, 8), 256, 0, st>>>(Xs[2 * l + 1], (int)R, d, P + w.ln2g, P + w.ln2b, A2[l], d1, mu2[l], rs2[l]);
    gfwd(A2[l], d1, (int)R, Pb, w.w1, Hh[l], c.mlp + 16, 1, st);
    TCK(cudaMemcpyAsync(Xs[2 * l + 2], Xs[2 * l + 1], xb, cudaMemcpyDeviceToDevice, st));
    gfwd(Hh[l], c.mlp + 16, (int)R, Pb, w.w2, Xs[2 * l + 2], d, 3, st);
  }
  ln_fwd_k<<<nb_(R, 8), 256, 0, st>>>(Xs[2 * c.layers], (int)R, d, P + lay.lnf_g, P + lay.lnf_b, Pf, d1, muf, rsf);
  TCK(cudaGetLastError());
}

void Tf::flow_inputs(const TfFlow& f, int B, cudaStream_t st) {
  flow_in_k<<<nb_(B, 128), 128, 0, st>>>(f.chunk, f.cmask, f.iter, f.seed, B, c.H, c.A, KA, ain, tau, eps, u);
  TCK(cudaGetLastError());
}

void Tf::expert_forward(int B, bool prefix_kv, cudaStream_t st) {
  const int d = c.d;
  const long long R = (long long)B * L, RA = (long long)B * c.H;
  const size_t xb = sizeof(float) * RA * d;
  gfwd(ain, KA, (int)RA, Pb, lay.e_in, Xa[0], d, 2, st);
  for (int l = 0; l < c.e_layers; ++l) {
    const auto& w = lay.eblk[l];
    ln_fwd_k<<<nb_(RA, 8), 256, 0, st>>>(Xa[2 * l], (int)RA, d, P + w.ln1g, P + w.ln1b, eA1[l], d1, emu1[l], ers1[l]);
    gfwd(eA1[l], d1, (int)RA, Pb, w.qkv, eQKV[l], 3 * d, 0, st);
    if (prefix_kv) gfwd(Pf, d1, (int)R, Pb, w.kvp, ePKV[l], 2 * d, 0, st);
    attn_fwd(ex_attn(*this, l), B, st);
    TCK(cudaMemcpyAsync(Xa[2 * l + 1], Xa[2 * l], xb, cudaMemcpyDeviceToDevice, st));
    gfwd(eO[l], d1, (int)RA, Pb, w.wo, Xa[2 * l + 1], d, 3, st);
    ln_fwd_k<<<nb_(RA, 8), 256, 0, st>>>(Xa[2 * l + 1], (int)RA, d, P + w.ln2g, P + w.ln2b, eA2[l], d1, emu2[l], ers2[l]);
    gfwd(eA2[l], d1, (int)RA, Pb, w.w1, eH[l], c.mlp + 16, 1, st);
    TCK(cudaMemcpyAsync(Xa[2 * l + 2], Xa[2 * l + 1], xb, cudaMemcpyDeviceToDevice, st));
    gfwd(eH[l], c.mlp + 16, (int)RA, Pb, w.w2, Xa[2 * l + 2], d, 3, st);
  }
  ln_fwd_k<<<nb_(RA, 8), 256, 0, st>>>(Xa[2 * c.e_layers], (int)RA, d, P + lay.lne_g, P + lay.lne_b, Ao, d1, muo, rso);
  gfwd(Ao, d1, (int)RA, Pb, lay.e_out, vel, c.A, 2, st);
  TCK(cudaGetLastError());
}

void Tf::loss(const TfFlow& f, int B, cudaStream_t st) {
  const int RA = B * c.H, nb = (RA + LT - 1) / LT;
  loss_k<<<nb, LT, 0, st>>>(vel, u, f.cmask, f.adim_mask, B, c.H, c.A, bug, dz, lpart);
  loss_red_k<<<1, 32, 0, st>>>(lpart, nb, B, loss_d);
  TCK(cudaGetLastError());
}

// LN 뒤 + γ·β 기울기
static void ln_back(Tf& t, const float* dy, const float* X, int R, long long gofs, long long bofs, const float* mu, const float* rs, float* dRr,
                    cudaStream_t st) {
  const int d = t.c.d;
  ln_bwd_k<<<nb_(R, 8), 256, 0, st>>>(dy, X, R, d, t.P + gofs, mu, rs, t.bug, dRr);
  const int nb = (R + CR - 1) / CR;
  lnp_part_k<<<nb, 256, 0, st>>>(dy, X, R, d, mu, rs, t.cpart);
  colred_k<<<nb_(d), 256, 0, st>>>(t.cpart, nb, d, t.G + gofs, t.G + bofs);
  TCK(cudaGetLastError());
}

// 한 블록 뒤(몸통·전문가 같은 꼴). Xin = 블록 입력 스냅숏, Xmid = 어텐션 뒤 스냅숏
struct BlkBufs {
  const float *Xin, *Xmid; uint16_t *A1, *A2, *Hh, *O; float *mu1, *rs1, *mu2, *rs2;
  float *dR, *dT; uint16_t *dRb, *dHh, *dOb, *dQKV; int R;
};
static void blk_mlp_back(Tf& t, const TfLayout::Blk& w, const BlkBufs& b, cudaStream_t st) {
  const int d = t.c.d, R = b.R, mp = t.c.mlp;
  f2b_k<<<nb_((long long)R * d), 256, 0, st>>>(b.dR, (long long)R * d, b.dRb);
  gdw(b.dRb, b.Hh, mp + 16, R, w.w2, t.ws, t.c.dw_chunk, t.G, st);
  gdx(b.dRb, R, t.Pb, w.w2, mp, b.dHh, mp, 4, b.Hh, mp + 16, st);
  gdw(b.dHh, b.A2, t.d1, R, w.w1, t.ws, t.c.dw_chunk, t.G, st);
  gdx(b.dHh, R, t.Pb, w.w1, d, b.dT, d, 2, nullptr, 0, st);
  ln_back(t, b.dT, b.Xmid, R, w.ln2g, w.ln2b, b.mu2, b.rs2, b.dR, st);
  f2b_k<<<nb_((long long)R * d), 256, 0, st>>>(b.dR, (long long)R * d, b.dRb);
  gdw(b.dRb, b.O, t.d1, R, w.wo, t.ws, t.c.dw_chunk, t.G, st);
  gdx(b.dRb, R, t.Pb, w.wo, d, b.dOb, d, 0, nullptr, 0, st);
}
static void blk_qkv_back(Tf& t, const TfLayout::Blk& w, const BlkBufs& b, cudaStream_t st) {
  const int d = t.c.d, R = b.R;
  gdw(b.dQKV, b.A1, t.d1, R, w.qkv, t.ws, t.c.dw_chunk, t.G, st);
  gdx(b.dQKV, R, t.Pb, w.qkv, d, b.dT, d, 2, nullptr, 0, st);
  ln_back(t, b.dT, b.Xin, R, w.ln1g, w.ln1b, b.mu1, b.rs1, b.dR, st);
}

void Tf::backward(const TfIn& in, int B, cudaStream_t st) {
  const int d = c.d;
  const int R = B * L, RA = B * c.H;
  // 머리
  gdw(dz, Ao, d1, RA, lay.e_out, ws, c.dw_chunk, G, st);
  gdx(dz, RA, Pb, lay.e_out, d, dTa, d, 2, nullptr, 0, st);
  TCK(cudaMemsetAsync(dRa, 0, sizeof(float) * (size_t)RA * d, st));
  ln_back(*this, dTa, Xa[2 * c.e_layers], RA, lay.lne_g, lay.lne_b, muo, rso, dRa, st);
  TCK(cudaMemsetAsync(dPf, 0, sizeof(float) * (size_t)R * d, st));
  for (int l = c.e_layers - 1; l >= 0; --l) {
    const auto& w = lay.eblk[l];
    BlkBufs b{Xa[2 * l], Xa[2 * l + 1], eA1[l], eA2[l], eH[l], eO[l], emu1[l], ers1[l], emu2[l], ers2[l], dRa, dTa, dRab, dHa, dOa, dQKVa, RA};
    blk_mlp_back(*this, w, b, st);
    AttnP p = ex_attn(*this, l);
    p.dO = dOa; p.lddo = d; p.dq = dQKVa; p.lddq = 3 * d;
    p.dk1 = dPKV; p.dv1 = dPKV + d; p.lddk1 = 2 * d;
    p.dk2 = dQKVa + d; p.dv2 = dQKVa + 2 * d; p.lddk2 = 3 * d;
    attn_bwd(p, B, st);
    if (tap && l == 0) {
      TCK(cudaMemcpyAsync(tap_edO, dOa, sizeof(uint16_t) * (size_t)RA * d, cudaMemcpyDeviceToDevice, st));
      TCK(cudaMemcpyAsync(tap_edQKV, dQKVa, sizeof(uint16_t) * (size_t)RA * 3 * d, cudaMemcpyDeviceToDevice, st));
      TCK(cudaMemcpyAsync(tap_edPKV, dPKV, sizeof(uint16_t) * (size_t)R * 2 * d, cudaMemcpyDeviceToDevice, st));
    }
    gdw(dPKV, Pf, d1, R, w.kvp, ws, c.dw_chunk, G, st);
    gdx(dPKV, R, Pb, w.kvp, d, dPf, d, 3, nullptr, 0, st);
    blk_qkv_back(*this, w, b, st);
  }
  f2b_k<<<nb_((long long)RA * d), 256, 0, st>>>(dRa, (long long)RA * d, dRab);
  gdw(dRab, ain, KA, RA, lay.e_in, ws, c.dw_chunk, G, st);
  // 몸통
  TCK(cudaMemsetAsync(dR, 0, sizeof(float) * (size_t)R * d, st));
  ln_back(*this, dPf, Xs[2 * c.layers], R, lay.lnf_g, lay.lnf_b, muf, rsf, dR, st);
  for (int l = c.layers - 1; l >= 0; --l) {
    const auto& w = lay.blk[l];
    BlkBufs b{Xs[2 * l], Xs[2 * l + 1], A1[l], A2[l], Hh[l], O[l], mu1[l], rs1[l], mu2[l], rs2[l], dR, dT, dRb, dHh, dOb, dQKV, R};
    blk_mlp_back(*this, w, b, st);
    AttnP p = self_attn(*this, l, B);
    p.dO = dOb; p.lddo = d; p.dq = dQKV; p.lddq = 3 * d;
    p.dk1 = dQKV + d; p.dv1 = dQKV + 2 * d; p.lddk1 = 3 * d;
    attn_bwd(p, B, st);
    if (tap && l == 0) {
      TCK(cudaMemcpyAsync(tap_dO, dOb, sizeof(uint16_t) * (size_t)R * d, cudaMemcpyDeviceToDevice, st));
      TCK(cudaMemcpyAsync(tap_dQKV, dQKV, sizeof(uint16_t) * (size_t)R * 3 * d, cudaMemcpyDeviceToDevice, st));
      TCK(cudaMemcpyAsync(tap_dRpre, dR, sizeof(float) * (size_t)R * d, cudaMemcpyDeviceToDevice, st));
    }
    blk_qkv_back(*this, w, b, st);
    if (tap && l == 0) TCK(cudaMemcpyAsync(tap_dT, dT, sizeof(float) * (size_t)R * d, cudaMemcpyDeviceToDevice, st));
  }
  // 묶음 임베딩
  for (int g = 0; g < N_GRP; ++g) {
    for (int ty = 0; ty < kGrp[g].n_type; ++ty) {
      const int n = kGrp[g].n_tok, t0 = kGrp[g].n_type == 2 ? ty * n / 2 : 0, t1 = kGrp[g].n_type == 2 ? (ty + 1) * n / 2 : n;
      const int cb = (t1 - t0) >= 256 ? 1 : 256 / (t1 - t0), nb = (B + cb - 1) / cb;
      temb_part_k<<<nb, 256, 0, st>>>(dR, B, d, grp_tok0(g), t0, t1, cb, cpart);
      colred_k<<<nb_(d), 256, 0, st>>>(cpart, nb, d, G + lay.g_type[g] + (long long)ty * d, nullptr);
    }
  }
  gather_dE_k<<<nb_((long long)R * d), 256, 0, st>>>(dR, B, d, dEb);
  for (int g = 0; g < N_GRP; ++g) {
    const int rows = B * kGrp[g].n_tok;
    const uint16_t* dEg = dEb + (long long)B * grp_tok0(g) * d;
    if (lay.g_w2[g].N) {
      gdw(dEg, oh, c.obj_hidden + 16, rows, lay.g_w2[g], ws, c.dw_chunk, G, st);
      gdx(dEg, rows, Pb, lay.g_w2[g], c.obj_hidden, doh, c.obj_hidden, 4, oh, c.obj_hidden + 16, st);
      gdw(doh, in.g[g], kGrp[g].K, rows, lay.g_w1[g], ws, c.dw_chunk, G, st);
    } else {
      gdw(dEg, in.g[g], kGrp[g].K, rows, lay.g_w1[g], ws, c.dw_chunk, G, st);
    }
  }
  TCK(cudaGetLastError());
}

void Tf::infer(const TfIn& in, int B, int steps, const uint32_t* adim_mask, const long long* key, int t, uint64_t seed, float* xbuf, float* act,
               cudaStream_t st) {
  forward_prefix(in, B, st);
  const long long n = (long long)B * c.H * c.A;
  noise_k<<<nb_(n), 256, 0, st>>>(key, t, seed, B, c.H, c.A, xbuf);
  for (int s = 0; s < steps; ++s) {
    ain_k<<<nb_((long long)B * c.H, 128), 128, 0, st>>>(xbuf, 1.f - (float)s / (float)steps, B, c.H, c.A, KA, ain);
    expert_forward(B, s == 0, st);
    euler_k<<<nb_(n), 256, 0, st>>>(vel, 1.f / (float)steps, n, c.A, adim_mask, s == steps - 1, xbuf, act);
  }
  TCK(cudaGetLastError());
}

}  // namespace tfm
