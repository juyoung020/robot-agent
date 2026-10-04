// V3 FP8 GEMM 시험(G6, 계획서 GPU_TRAINING.md 9절 V3·12절). sm_120a 로 빌드(BC CMake 의 fp8_verify).
//   fp8_verify probe             : 12절 열린 문제의 잰 값 — 블록당 공유 메모리 한도, 섞인 형식(e5m2 × e4m3) 한 명령, 8 비트 ldmatrix 전치 조각 배치
//   fp8_verify quant             : FP8 양자화(cuda_fp8 __nv_cvt_float_to_fp8, RNE·satfinite) 를 CPU 참조 반올림과 float 2³² 개 전부 비트 비교(E4M3, E5M2)
//   fp8_verify gemm [--negative] : GEMM V3 — ① 같은 양자화 입력의 FP64 와 비교(누산 오차만) ② 양자화 안 한 입력의 FP64 와 비교(양자화 오차)
//                                  각각 바닥(CPU 흉내)과 견줘 우리 오차 ≤ 2 × 바닥. --negative: 버그 셋(배율 하나 빼기, B 조각 반쪽 바꾸기, 0 쪽 자르기 양자화)
//   fp8_verify bench             : 같은 모양에서 FP8 대 16 비트 GEMM TFLOPS (인코더 모양 = tn_k, 학습 신경망 모양 = gemm8_k 대 gemm2_k)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "gemm_fp8.cuh"
#include "net_ops.h"

#define FCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(3); } } while (0)

using namespace f8;

// ---- CPU 참조 FP8 반올림(가장 가까운 짝수, satfinite: 넘으면 ±최댓값). 반환 = 바이트 ----------------------------------------
static uint8_t qref(float x, int f) {
  const int mb = f == E4M3 ? 3 : 2, bias = f == E4M3 ? 7 : 15, emin = 1 - bias;
  const double maxv = fmax_of(f);
  const uint8_t sg = std::signbit(x) ? 0x80 : 0;
  double a = std::fabs((double)x);
  if (std::isnan(x)) return sg | 0x7f;
  if (a > maxv) a = maxv;
  int e;
  std::frexp(a, &e);
  e -= 1;
  if (e < emin) e = emin;
  double qv = std::nearbyint(a / std::ldexp(1.0, e - mb)) * std::ldexp(1.0, e - mb);
  if (qv > maxv) qv = maxv;
  if (qv == 0.0) return sg;
  std::frexp(qv, &e);
  e -= 1;
  if (e < emin) return sg | (uint8_t)(int)(qv / std::ldexp(1.0, emin - mb));
  const int m = (int)((qv / std::ldexp(1.0, e) - 1.0) * (1 << mb));
  return sg | (uint8_t)(((e + bias) << mb) | m);
}
static double dqref(uint8_t b, int f) {
  const int mb = f == E4M3 ? 3 : 2, bias = f == E4M3 ? 7 : 15;
  const int s = b >> 7, e = (b >> mb) & ((1 << (7 - mb)) - 1), m = b & ((1 << mb) - 1);
  const double v = e == 0 ? std::ldexp((double)m, 1 - bias - mb) : std::ldexp(1.0 + (double)m / (1 << mb), e - bias);
  return s ? -v : v;
}
static double qd(float x, float inv, int f) { return dqref(qref(x * inv, f), f) / inv; }

// ---- quant: 2³² 전수 ----
__global__ void quant_all_k(uint32_t base, uint8_t* o4, uint8_t* o5, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = __uint_as_float(base + i);
  o4[i] = (uint8_t)q8<E4M3>(x);
  o5[i] = (uint8_t)q8<E5M2>(x);
}
static int run_quant() {
  const uint32_t CH = 1u << 28;
  uint8_t *d4, *d5;
  FCK(cudaMalloc(&d4, CH)); FCK(cudaMalloc(&d5, CH));
  std::vector<uint8_t> h4(CH), h5(CH);
  long long bad[2] = {0, 0}, nan_in = 0;
  uint32_t first_bad[2] = {0, 0};
  for (uint64_t base = 0; base < (1ull << 32); base += CH) {
    quant_all_k<<<CH / 256, 256>>>((uint32_t)base, d4, d5, CH);
    FCK(cudaMemcpy(h4.data(), d4, CH, cudaMemcpyDeviceToHost));
    FCK(cudaMemcpy(h5.data(), d5, CH, cudaMemcpyDeviceToHost));
#pragma omp parallel for reduction(+ : nan_in) schedule(static)
    for (long long i = 0; i < (long long)CH; ++i) {
      const uint32_t u = (uint32_t)base + (uint32_t)i;
      float x;
      std::memcpy(&x, &u, 4);
      if (std::isnan(x)) { ++nan_in; continue; }   // NaN 은 비교 밖(양쪽 다 NaN 부호만 다를 수 있음)
      for (int f = 0; f < 2; ++f) {
        const uint8_t g = f ? h5[i] : h4[i];
        if (g != qref(x, f)) {
#pragma omp critical
          {
            if (bad[f]++ == 0) first_bad[f] = u;
          }
        }
      }
    }
  }
  cudaFree(d4); cudaFree(d5);
  for (int f = 0; f < 2; ++f) {
    float x;
    std::memcpy(&x, &first_bad[f], 4);
    std::printf("quant %s: %lld of %llu non-NaN floats differ from CPU RNE/satfinite reference", f ? "E5M2" : "E4M3", bad[f], (1ull << 32) - nan_in);
    if (bad[f]) std::printf(" (first 0x%08x = %g: GPU vs CPU)", first_bad[f], x);
    std::printf("  %s\n", bad[f] ? "FAIL" : "ok");
  }
  return (bad[0] || bad[1]) ? 1 : 0;
}

// ---- probe(12절) ----
__global__ void ldsm_probe_k(uint32_t* o) {
  __shared__ __align__(16) uint8_t s[32 * 16];
  for (int i = threadIdx.x; i < 512; i += 32) s[i] = (uint8_t)(i & 255);
  __syncwarp();
  uint32_t r[4];
  ldsm8t2(r, s + (threadIdx.x & 15) * 16 + (threadIdx.x >> 4) * 256);
  for (int j = 0; j < 4; ++j) o[threadIdx.x * 4 + j] = r[j];
}
__global__ void mix_probe_k(const uint8_t* a, const uint8_t* b, float* d) {
  // 한 워프: A 16×32 (e5m2), B 8×32 (e4m3) — 조각 배치대로 레지스터에
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  uint32_t af[4], bf[2];
  auto rd = [](const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); };
  af[0] = rd(a + g * 32 + 4 * t); af[1] = rd(a + (g + 8) * 32 + 4 * t); af[2] = rd(a + g * 32 + 16 + 4 * t); af[3] = rd(a + (g + 8) * 32 + 16 + 4 * t);
  bf[0] = rd(b + g * 32 + 4 * t); bf[1] = rd(b + g * 32 + 16 + 4 * t);
  float acc[4] = {0, 0, 0, 0};
  mma8<E5M2, E4M3>(acc, af, bf);
  d[(g) * 8 + 2 * t] = acc[0]; d[(g) * 8 + 2 * t + 1] = acc[1]; d[(g + 8) * 8 + 2 * t] = acc[2]; d[(g + 8) * 8 + 2 * t + 1] = acc[3];
}
static int run_probe() {
  int optin = 0, sm = 0, blk = 0, cc0 = 0, cc1 = 0;
  FCK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  FCK(cudaDeviceGetAttribute(&sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  FCK(cudaDeviceGetAttribute(&blk, cudaDevAttrMaxSharedMemoryPerBlock, 0));
  FCK(cudaDeviceGetAttribute(&cc0, cudaDevAttrComputeCapabilityMajor, 0));
  FCK(cudaDeviceGetAttribute(&cc1, cudaDevAttrComputeCapabilityMinor, 0));
  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  std::printf("device cc %d.%d, runtime %d, driver %d\n", cc0, cc1, rt, drv);
  std::printf("shared memory: per block opt-in %d B (%.1f KB), per SM %d B (%.1f KB), per block default %d B\n", optin, optin / 1024.0, sm, sm / 1024.0, blk);
  // 섞인 형식: A e5m2, B e4m3 → CPU 와 비교(곱이 정확하고 합이 작아 FP32 에서 정확해야 함)
  std::mt19937 rng(7);
  std::vector<uint8_t> a(16 * 32), b(8 * 32);
  for (auto& v : a) { do v = (uint8_t)(rng() & 0xff); while (std::isnan(dqref(v, E5M2)) || std::isinf(dqref(v, E5M2)) || ((v & 0x7c) == 0x7c) || std::fabs(dqref(v, E5M2)) > 64); }
  for (auto& v : b) { do v = (uint8_t)(rng() & 0xff); while ((v & 0x7f) == 0x7f || std::fabs(dqref(v, E4M3)) > 64); }
  uint8_t *da, *db;
  float* dd;
  FCK(cudaMalloc(&da, a.size())); FCK(cudaMalloc(&db, b.size())); FCK(cudaMalloc(&dd, 128 * 4));
  FCK(cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice)); FCK(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice));
  mix_probe_k<<<1, 32>>>(da, db, dd);
  std::vector<float> d(128);
  FCK(cudaMemcpy(d.data(), dd, 512, cudaMemcpyDeviceToHost));
  double maxd = 0, maxv = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) {
      double s = 0;
      for (int k = 0; k < 32; ++k) s += dqref(a[m * 32 + k], E5M2) * dqref(b[n * 32 + k], E4M3);
      maxd = std::max(maxd, std::fabs(s - d[m * 8 + n]) / (std::fabs(s) + 1e-30));
      maxv = std::max(maxv, std::fabs(s));
    }
  std::printf("mixed e5m2 x e4m3 (mma.sync kind::f8f6f4 m16n8k32, one instruction): compiles for sm_120a, 16x8 outputs vs CPU exact: max rel diff %.3e (|max| %.1f)  %s\n",
              maxd, maxv, maxd < 1e-6 ? "ok" : "FAIL");
  // 8 비트 ldmatrix 전치: 메모리 [k][행] 바이트 = k·16 + 행 (k 0..31) → 레인 조각
  uint32_t* dl;
  FCK(cudaMalloc(&dl, 32 * 4 * 4));
  ldsm_probe_k<<<1, 32>>>(dl);
  std::vector<uint32_t> L(128);
  FCK(cudaMemcpy(L.data(), dl, 512, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int lane = 0; lane < 32; ++lane) {
    const int g = lane >> 2, t = lane & 3;
    for (int j = 0; j < 4; ++j)
      for (int e = 0; e < 4; ++e) {
        const int row = g + ((j & 1) ? 8 : 0), k = ((j & 2) ? 16 : 0) + 4 * t + e;   // 기대: A 조각 a0..a3 = (행, k)
        const int want = (k * 16 + row) & 255;
        bad += (int)((L[lane * 4 + j] >> (8 * e)) & 0xff) != want;
      }
  }
  std::printf("8-bit ldmatrix transpose (ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8): exists on sm_120a; fragment = mma A layout (a0..a3) from [k][row] bytes: %d of 512 bytes differ  %s\n",
              bad, bad ? "FAIL" : "ok");
  cudaFree(da); cudaFree(db); cudaFree(dd); cudaFree(dl);
  return (maxd < 1e-6 && !bad) ? 0 : 1;
}

// ---- GEMM V3 ----
static std::vector<uint16_t> rand_bf16(size_t n, double sd, uint32_t seed, double p_out = 1e-3, double out_mul = 20.0) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, sd);
  std::uniform_real_distribution<double> ud(0.0, 1.0);
  std::vector<uint16_t> v(n);
  for (auto& x : v) {
    double y = nd(rng);
    if (ud(rng) < p_out) y *= out_mul;
    x = net::f2bf((float)y);
  }
  return v;
}
// 논리 피연산자 X(i, k) (i < R, k < K) 를 저장 배치(T: [k][i], 아니면 [i][k]) 에서 읽음
struct Opnd {
  std::vector<uint16_t> h;
  int R, K;
  bool T;
  double at(int i, int k) const { return net::bf2f(T ? h[(size_t)k * R + i] : h[(size_t)i * K + k]); }
  long long ld() const { return T ? R : K; }
};
static float host_amax(const Opnd& o) {
  float m = 0;
  for (uint16_t v : o.h) m = std::max(m, std::fabs(net::bf2f(v)));
  return m;
}
// 0 쪽 자르기(FP32 누산 흉내 — network/README 의 텐서 코어 누산 모형: k 묶음 곱을 정확히 더하고 0 쪽으로 잘라 FP32)
static double rz_f32(double x) {
  const float f = (float)x;   // 가장 가까운 짝수
  if ((double)f == x || !std::isfinite(f)) return f;
  if (std::fabs((double)f) > std::fabs(x)) return std::nextafter(f, 0.f);
  return f;
}
struct Res { double eq, fq, e_raw, f_raw; };   // ① GPU·바닥, ② GPU·바닥 (상대 L2)
// C(m, n) = Σ_k A(m,k) B(n,k).  GPU 결과 G[m][n](분할 합 포함), 양자화 형식 fa·fb, 배율 inv, 누산 묶음 32, 분할 kchunk
static Res compare(const Opnd& A, const Opnd& B, int fa, int fb, const std::vector<double>& G, int kchunk) {
  const int M = A.R, N = B.R, K = A.K;
  const float ia = pow2_inv(host_amax(A), fa), ib = pow2_inv(host_amax(B), fb);
  std::vector<double> Aq((size_t)M * K), Bq((size_t)N * K), Ar((size_t)M * K), Br((size_t)N * K);
#pragma omp parallel for
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) { Ar[(size_t)i * K + k] = A.at(i, k); Aq[(size_t)i * K + k] = qd((float)A.at(i, k), ia, fa); }
#pragma omp parallel for
  for (int i = 0; i < N; ++i)
    for (int k = 0; k < K; ++k) { Br[(size_t)i * K + k] = B.at(i, k); Bq[(size_t)i * K + k] = qd((float)B.at(i, k), ib, fb); }
  const double deq = (1.0 / ia) * (1.0 / ib);
  double n_eq = 0, n_fq = 0, n_er = 0, n_fr = 0, d_q = 0, d_r = 0;
#pragma omp parallel for reduction(+ : n_eq, n_fq, n_er, n_fr, d_q, d_r) schedule(dynamic, 4)
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const double* a = &Aq[(size_t)m * K];
      const double* b = &Bq[(size_t)n * K];
      const double* ar = &Ar[(size_t)m * K];
      const double* br = &Br[(size_t)n * K];
      double rq = 0, rr = 0, emul = 0;
      for (int z0 = 0; z0 < K; z0 += kchunk) {   // 분할마다 FP32 누산(흉내) → 분할 합은 float 차례
        double acc = 0;
        for (int k0 = z0; k0 < std::min(K, z0 + kchunk); k0 += 32) {
          double s = 0;
          for (int k = k0; k < std::min(std::min(K, z0 + kchunk), k0 + 32); ++k) s += (a[k] * ia) * (b[k] * ib);   // 양자화 값(배율 공간)의 정확한 곱
          acc = rz_f32(acc + s);
        }
        emul = (double)(float)(emul + (double)(float)(acc * deq));
      }
      for (int k = 0; k < K; ++k) { rq += a[k] * b[k]; rr += ar[k] * br[k]; }
      const double g = G[(size_t)m * N + n];
      n_eq += (g - rq) * (g - rq); n_fq += (emul - rq) * (emul - rq);
      n_er += (g - rr) * (g - rr); n_fr += (rq - rr) * (rq - rr);
      d_q += rq * rq; d_r += rr * rr;
    }
  return {std::sqrt(n_eq / d_q), std::sqrt(n_fq / d_q), std::sqrt(n_er / d_r), std::sqrt(n_fr / d_r)};
}

template <int FA, int FB, bool AT, bool BT, int EPI>
static void launch8(const G8& g, int np, cudaStream_t st) {
  const net::GemmP& p = g.p[0];
  constexpr int BM = 128, BN = 128, WM = 64, WN = 32;
  gemm8_k<FA, FB, AT, BT, EPI, BM, BN, WM, WN><<<dim3((p.N + BN - 1) / BN, (p.M + BM - 1) / BM, g.zper * np), (BM / WM) * (BN / WN) * 32, 0, st>>>(g);
  FCK(cudaGetLastError());
}
// 학습 신경망 GEMM(gemm8_k): 한 경우를 GPU 로 돌려 결과(분할 합 포함, double)
struct Case { const char* name; int fa, fb; bool AT, BT; int M, N, K, kchunk; };
static std::vector<double> run_g8(const Case& c, const Opnd& A, const Opnd& B, int bug, float** amA_out = nullptr) {
  uint16_t *dA, *dB;
  float *dC, *am;
  const int splits = (c.K + c.kchunk - 1) / c.kchunk;
  FCK(cudaMalloc(&dA, A.h.size() * 2)); FCK(cudaMalloc(&dB, B.h.size() * 2));
  FCK(cudaMalloc(&dC, sizeof(float) * c.M * c.N * splits)); FCK(cudaMalloc(&am, 8));
  FCK(cudaMemcpy(dA, A.h.data(), A.h.size() * 2, cudaMemcpyHostToDevice)); FCK(cudaMemcpy(dB, B.h.data(), B.h.size() * 2, cudaMemcpyHostToDevice));
  FCK(cudaMemset(am, 0, 8));
  amax_bf16_impl_k<<<240, 256>>>(dA, A.T ? A.K : A.R, A.T ? A.R : A.K, A.ld(), am);
  amax_bf16_impl_k<<<240, 256>>>(dB, B.T ? B.K : B.R, B.T ? B.R : B.K, B.ld(), am + 1);
  G8 g{};
  net::GemmP& p = g.p[0];
  p.A = dA; p.lda = A.ld(); p.B = dB; p.ldb = B.ld(); p.M = c.M; p.N = c.N; p.K = c.K; p.C = dC; p.ldc = c.N; p.kchunk = c.kchunk;
  g.p[1] = p;
  g.amA[0] = g.amA[1] = am; g.amB[0] = g.amB[1] = am + 1;
  g.zper = splits;
  g.bug = bug == 1 ? 3 : bug == 3 ? 1 : bug;   // 시험 버그 1 = A 배율 빠뜨림(커널 3), 2 = B 조각 반쪽, 3 = 0 쪽 자르기(커널 1)
  const int ep = splits > 1 ? net::EPI_SPLIT_F32 : net::EPI_F32;
  auto L = [&](auto fa, auto fb) {
    constexpr int FA = decltype(fa)::value, FB = decltype(fb)::value;
    if (ep == net::EPI_F32) {
      if (!c.AT && !c.BT) launch8<FA, FB, false, false, net::EPI_F32>(g, 1, 0);
      else if (!c.AT && c.BT) launch8<FA, FB, false, true, net::EPI_F32>(g, 1, 0);
      else if (c.AT && !c.BT) launch8<FA, FB, true, false, net::EPI_F32>(g, 1, 0);
      else launch8<FA, FB, true, true, net::EPI_F32>(g, 1, 0);
    } else {
      if (c.AT && c.BT) launch8<FA, FB, true, true, net::EPI_SPLIT_F32>(g, 1, 0);
      else { std::fprintf(stderr, "split only TT\n"); std::exit(2); }
    }
  };
  using I4 = std::integral_constant<int, E4M3>;
  using I5 = std::integral_constant<int, E5M2>;
  if (c.fa == E4M3 && c.fb == E4M3) L(I4{}, I4{});
  else if (c.fa == E5M2 && c.fb == E4M3) L(I5{}, I4{});
  else L(I5{}, I5{});
  FCK(cudaDeviceSynchronize());
  std::vector<float> h((size_t)c.M * c.N * splits);
  FCK(cudaMemcpy(h.data(), dC, h.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<double> G((size_t)c.M * c.N, 0.0);
  for (size_t i = 0; i < G.size(); ++i) {
    float s = 0.f;
    for (int z = 0; z < splits; ++z) s = s + h[(size_t)z * c.M * c.N + i];
    G[i] = s;
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(am);
  (void)amA_out;
  return G;
}

// 인코더 경로(tn_k): GPU 가 행마다(A)·열마다(B) 2 의 거듭제곱 배율로 양자화 → tn_k → out = acc · sa[r] · sb[c]
__global__ void rowquant_k(const uint16_t* X, int rows, int K, int fmt, uint8_t* Q, float* deq) {
  const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, l = threadIdx.x & 31;
  if (r >= rows) return;
  float m = 0.f;
  for (int k = l; k < K; k += 32) m = fmaxf(m, fabsf(net::bf2f(X[(size_t)r * K + k])));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float inv = pow2_inv(m, fmt);
  for (int k = l; k < K; k += 32) {
    const float x = net::bf2f(X[(size_t)r * K + k]) * inv;
    Q[(size_t)r * K + k] = (uint8_t)(fmt == E4M3 ? q8<E4M3>(x) : q8<E5M2>(x));
  }
  if (l == 0) deq[r] = 1.f / inv;
}
struct EpiOut {
  float* C; int ldc; const float* sa; const float* sb;
  __device__ __forceinline__ void operator()(int r, int c, float x0, float x1) const {
    const float s = sa[r];
    *reinterpret_cast<float2*>(C + (size_t)r * ldc + c) = make_float2(x0 * s * sb[c], x1 * s * sb[c + 1]);
  }
};
template <int KIND>
static void launch_tn(const uint16_t* A16, int lda16, const uint16_t* B16, int ldb16, int M, int N, int K16, const EpiOut& e, cudaStream_t st) {
  constexpr int BM = 128, BN = 128, WM = 64, WN = 32, NST = 4;
  constexpr int smem = tn_smem<BM, BN>(NST);
  static bool once = [] { FCK(cudaFuncSetAttribute(tn_k<KIND, BM, BN, WM, WN, NST, EpiOut>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); return true; }();
  (void)once;
  tn_k<KIND, BM, BN, WM, WN, NST, EpiOut><<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), (BM / WM) * (BN / WN) * 32, smem, st>>>(A16, lda16, B16, ldb16, M, N, K16, e);
  FCK(cudaGetLastError());
}
static Res run_tn_case(int kind, int M, int N, int K, uint32_t seed) {
  const int fa = kind == TN_E5E4 ? E5M2 : E4M3, fb = E4M3;
  Opnd A{rand_bf16((size_t)M * K, 1.0, seed), M, K, false}, B{rand_bf16((size_t)N * K, 0.05, seed + 1), N, K, false};
  uint16_t *dA, *dB;
  uint8_t *qA, *qB;
  float *sa, *sb, *dC;
  FCK(cudaMalloc(&dA, A.h.size() * 2)); FCK(cudaMalloc(&dB, B.h.size() * 2)); FCK(cudaMalloc(&qA, A.h.size())); FCK(cudaMalloc(&qB, B.h.size()));
  FCK(cudaMalloc(&sa, M * 4)); FCK(cudaMalloc(&sb, N * 4)); FCK(cudaMalloc(&dC, (size_t)M * N * 4));
  FCK(cudaMemcpy(dA, A.h.data(), A.h.size() * 2, cudaMemcpyHostToDevice)); FCK(cudaMemcpy(dB, B.h.data(), B.h.size() * 2, cudaMemcpyHostToDevice));
  rowquant_k<<<(M + 7) / 8, 256>>>(dA, M, K, fa, qA, sa);
  rowquant_k<<<(N + 7) / 8, 256>>>(dB, N, K, fb, qB, sb);
  const EpiOut e{dC, N, sa, sb};
  if (kind == TN_E5E4) launch_tn<TN_E5E4>((const uint16_t*)qA, K / 2, (const uint16_t*)qB, K / 2, M, N, K / 2, e, 0);
  else launch_tn<TN_E4E4>((const uint16_t*)qA, K / 2, (const uint16_t*)qB, K / 2, M, N, K / 2, e, 0);
  FCK(cudaDeviceSynchronize());
  std::vector<float> h((size_t)M * N);
  FCK(cudaMemcpy(h.data(), dC, h.size() * 4, cudaMemcpyDeviceToHost));
  // CPU: 행·열 배율로 양자화한 FP64 (①), 원본 FP64 (②), 흉내(묶음 32 정확 합 → 0 쪽 자르기 FP32 누산 → 배율 곱)
  std::vector<double> Aq((size_t)M * K), Bq((size_t)N * K), sA(M), sB(N);
  auto quant_rows = [&](const Opnd& o, int f, std::vector<double>& q, std::vector<double>& s) {
#pragma omp parallel for
    for (int i = 0; i < o.R; ++i) {
      float m = 0;
      for (int k = 0; k < o.K; ++k) m = std::max(m, std::fabs((float)o.at(i, k)));
      const float inv = pow2_inv(m, f);
      s[i] = 1.0 / inv;
      for (int k = 0; k < o.K; ++k) q[(size_t)i * o.K + k] = dqref(qref((float)o.at(i, k) * inv, f), f);   // 배율 공간 값
    }
  };
  quant_rows(A, fa, Aq, sA);
  quant_rows(B, fb, Bq, sB);
  double n_eq = 0, n_fq = 0, n_er = 0, n_fr = 0, d_q = 0, d_r = 0;
#pragma omp parallel for reduction(+ : n_eq, n_fq, n_er, n_fr, d_q, d_r) schedule(dynamic, 4)
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double rq = 0, rr = 0, acc = 0;
      for (int k0 = 0; k0 < K; k0 += 32) {
        double s = 0;
        for (int k = k0; k < k0 + 32; ++k) s += Aq[(size_t)m * K + k] * Bq[(size_t)n * K + k];
        acc = rz_f32(acc + s);
        rq += s;
      }
      for (int k = 0; k < K; ++k) rr += A.at(m, k) * B.at(n, k);
      const double sc = sA[m] * sB[n];
      rq *= sc;
      const double emul = (double)(float)((double)(float)acc * sc);
      const double g = h[(size_t)m * N + n];
      n_eq += (g - rq) * (g - rq); n_fq += (emul - rq) * (emul - rq);
      n_er += (g - rr) * (g - rr); n_fr += (rq - rr) * (rq - rr);
      d_q += rq * rq; d_r += rr * rr;
    }
  cudaFree(dA); cudaFree(dB); cudaFree(qA); cudaFree(qB); cudaFree(sa); cudaFree(sb); cudaFree(dC);
  return {std::sqrt(n_eq / d_q), std::sqrt(n_fq / d_q), std::sqrt(n_er / d_r), std::sqrt(n_fr / d_r)};
}

static bool report(const char* name, const Res& r, bool expect_pass = true) {
  const double r1 = r.eq / r.fq, r2 = r.e_raw / r.f_raw;
  const bool ok = r1 <= 2.0 && r2 <= 2.0;
  std::printf("  %-44s (1) same-quantized FP64: GPU %.3e floor %.3e ratio %.2f | (2) unquantized FP64: GPU %.3e floor %.3e ratio %.2f  %s\n", name, r.eq, r.fq, r1, r.e_raw,
              r.f_raw, r2, expect_pass ? (ok ? "ok" : "FAIL") : (ok ? "PASSES (bad)" : "fails (good)"));
  return ok;
}

static int run_gemm(bool negative) {
  int fails = 0, total = 0;
  std::printf("V3 FP8 GEMM, criterion: our error <= 2 x floor (floor (1) = CPU emulation of FP32 accumulate [exact k32 sum, round toward zero], floor (2) = CPU FP8 quantization in FP64)\n");
  std::printf(" tn_k (encoder path: pre-quantized, per-row A / per-column B power-of-two scales, K-contiguous):\n");
  struct T { int kind; int M, N, K; const char* nm; };
  const T tc[] = {{TN_E4E4, 1024, 768, 768, "E4M3xE4M3 M1024 N768 K768 (qkv-like)"}, {TN_E4E4, 512, 768, 3072, "E4M3xE4M3 M512 N768 K3072 (fc2-like)"},
                  {TN_E5E4, 1024, 768, 768, "E5M2xE4M3 M1024 N768 K768 (mixed)"}};
  for (const T& t : tc) { ++total; fails += !report(t.nm, run_tn_case(t.kind, t.M, t.N, t.K, 11 + t.K)); }
  std::printf(" gemm8_k (training path: bf16 in, per-tensor power-of-two scales from device amax, quantize while loading):\n");
  const Case cs[] = {{"fwd  E4M3xE4M3 NN M4096 N256 K288", E4M3, E4M3, false, false, 4096, 256, 288, 288},
                     {"dgrad E5M2xE4M3 A·Bt M4096 N256 K256", E5M2, E4M3, false, true, 4096, 256, 256, 256},
                     {"wgrad E5M2xE4M3 At·Bt M256 N288 K4096 split 1024", E5M2, E4M3, true, true, 256, 288, 4096, 1024},
                     {"At·B E4M3xE4M3 M256 N256 K512", E4M3, E4M3, true, false, 256, 256, 512, 512}};
  for (const Case& c : cs) {
    Opnd A{rand_bf16((size_t)c.M * c.K, c.fa == E5M2 ? 1e-3 : 1.0, 100 + c.M + c.K), c.M, c.K, c.AT};
    Opnd B{rand_bf16((size_t)c.N * c.K, c.fa == E5M2 && !c.AT ? 0.05 : 1.0, 200 + c.N), c.N, c.K, c.BT};
    const auto G = run_g8(c, A, B, 0);
    ++total;
    fails += !report(c.name, compare(A, B, c.fa, c.fb, G, c.kchunk));
  }
  std::printf("V3 FP8 GEMM: %d / %d passed\n", total - fails, total);
  if (!negative) return fails;
  std::printf(" negative controls (each must fail):\n");
  int caught = 0;
  const char* nm[4] = {"", "bug 1: A dequant scale dropped", "bug 2: B fragment k-halves swapped", "bug 3: truncating (toward-zero) quantization"};
  for (int bug = 1; bug <= 3; ++bug) {
    const Case& c = cs[0];
    Opnd A{rand_bf16((size_t)c.M * c.K, 1.0, 100 + c.M + c.K), c.M, c.K, c.AT};
    Opnd B{rand_bf16((size_t)c.N * c.K, 1.0, 200 + c.N), c.N, c.K, c.BT};
    const auto G = run_g8(c, A, B, bug);
    caught += !report(nm[bug], compare(A, B, c.fa, c.fb, G, c.kchunk), false);
  }
  std::printf("V3 --negative: %d / 3 bugs caught  %s\n", caught, caught == 3 ? "PASS" : "FAIL");
  return fails + (caught != 3);
}

// ---- bench ----
static float time_ms(const std::function<void()>& f, int reps) {
  cudaEvent_t a, b;
  FCK(cudaEventCreate(&a)); FCK(cudaEventCreate(&b));
  f();
  FCK(cudaDeviceSynchronize());
  FCK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) f();
  FCK(cudaEventRecord(b));
  FCK(cudaEventSynchronize(b));
  float ms = 0;
  FCK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / reps;
}
static int run_bench() {
  std::printf("bench (CUDA events, GPU otherwise idle). TFLOPS = 2MNK / time. Peaks (whitepaper): FP16/BF16 f32-acc 87.9, FP8 f32-acc 175.8\n");
  std::printf(" encoder shapes (M = 512 images x 64 tokens = one update step): tn_k FP16 (same kernel structure as vit.cu vgemm) vs tn_k FP8 E4M3 (pre-quantized), both 128x128 tile, 4-stage cp.async\n");
  const int M = 32768;
  const int shp[4][2] = {{2304, 768}, {768, 768}, {3072, 768}, {768, 3072}};
  const char* nm[4] = {"qkv  N2304 K768", "proj N768  K768", "fc1  N3072 K768", "fc2  N768  K3072"};
  uint16_t *A, *B;
  float* C;
  float *sa, *sb;
  FCK(cudaMalloc(&A, (size_t)M * 3072 * 2)); FCK(cudaMalloc(&B, (size_t)3072 * 3072 * 2)); FCK(cudaMalloc(&C, (size_t)M * 3072 * 4));
  FCK(cudaMalloc(&sa, M * 4)); FCK(cudaMalloc(&sb, 3072 * 4));
  FCK(cudaMemset(A, 0x22, (size_t)M * 3072 * 2)); FCK(cudaMemset(B, 0x22, (size_t)3072 * 3072 * 2));
  const EpiOut e{C, 0, sa, sb};
  double t16 = 0, t8 = 0, fl = 0;
  for (int i = 0; i < 4; ++i) {
    const int N = shp[i][0], K = shp[i][1];
    EpiOut ee = e;
    ee.ldc = N;
    const float a = time_ms([&] { launch_tn<TN_F16>(A, K, B, K, M, N, K, ee, 0); }, 20);
    const float b = time_ms([&] { launch_tn<TN_E4E4>(A, K / 2, B, K / 2, M, N, K / 2, ee, 0); }, 20);
    const double f = 2.0 * M * N * K;
    t16 += a; t8 += b; fl += f;
    std::printf("  %s: FP16 %.3f ms %.1f TFLOPS | FP8 %.3f ms %.1f TFLOPS | x%.2f\n", nm[i], a, f / a / 1e9, b, f / b / 1e9, a / b);
  }
  std::printf("  one block's 4 GEMMs: FP16 %.3f ms (%.1f TFLOPS), FP8 %.3f ms (%.1f TFLOPS), x%.2f\n", t16, fl / t16 / 1e9, t8, fl / t8 / 1e9, t16 / t8);
  // 학습 신경망 모양의 상한: 앞 층 끝단이 FP8 사본을 이미 써 두었다고 치고(미리 양자화) 같은 tn_k 로
  std::printf(" training-net forward shapes, pre-quantized bound (tn_k FP16 vs FP8, 128x128 tile): M 65,536\n");
  for (int i = 0; i < 3; ++i) {
    const int K = i == 0 ? 288 : 272, N = i == 2 ? 128 : 256;
    EpiOut ee = e;
    ee.ldc = N;
    const float a = time_ms([&] { launch_tn<TN_F16>(A, K, B, K, 65536, N, K, ee, 0); }, 20);
    const float b = time_ms([&] { launch_tn<TN_E4E4>(A, K / 2, B, K / 2, 65536, N, K / 2, ee, 0); }, 20);
    const double f = 2.0 * 65536 * N * K;
    std::printf("  K%d N%d: FP16 %.3f ms %.1f TFLOPS | FP8 %.3f ms %.1f TFLOPS | x%.2f\n", K, N, a, f / a / 1e9, b, f / b / 1e9, a / b);
  }
  // 학습 신경망 모양: gemm2_k(bf16, net::gemm_fwd 등) 대 gemm8_k(읽으며 양자화) — 같은 끝단
  std::printf(" training-net shapes (RL teacher minibatch 65,536 rows): gemm2_k BF16 (net_ops, production) vs gemm8_k FP8 (bf16 in, quantize on load) + amax pass\n");
  const int MR = 65536;
  uint16_t *X, *W, *Y, *dZ;
  float *ws, *am;
  FCK(cudaMalloc(&X, (size_t)MR * 288 * 2)); FCK(cudaMalloc(&W, 288 * 288 * 2)); FCK(cudaMalloc(&Y, (size_t)MR * 288 * 2)); FCK(cudaMalloc(&dZ, (size_t)MR * 288 * 2));
  FCK(cudaMalloc(&ws, (size_t)64 * 288 * 288 * 4)); FCK(cudaMalloc(&am, 16));
  FCK(cudaMemset(X, 0x22, (size_t)MR * 288 * 2)); FCK(cudaMemset(W, 0x22, 288 * 288 * 2)); FCK(cudaMemset(dZ, 0x22, (size_t)MR * 288 * 2));
  FCK(cudaMemset(am, 0, 16));
  struct RS { const char* nm; int K, N; };
  const RS rs[3] = {{"A1 288->256", 288, 256}, {"A2 272->256", 272, 256}, {"A3 272->128", 272, 128}};
  for (const RS& r : rs) {
    const net::LayerDesc L{r.K, r.N, r.N + 16, r.N, net::ACT_ELU, 1.f};
    const double f = 2.0 * MR * r.N * r.K;
    const float tb = time_ms([&] { net::gemm_fwd(L, X, MR, W, Y, 0); }, 20);
    G8 g{};
    net::GemmP& p = g.p[0];
    p.A = X; p.lda = r.K; p.B = W; p.ldb = r.K; p.M = MR; p.N = r.N; p.K = r.K; p.C = Y; p.ldc = r.N + 16; p.act = net::ACT_ELU;
    g.p[1] = p; g.amA[0] = g.amA[1] = am; g.amB[0] = g.amB[1] = am + 1; g.zper = 1;
    const float t8g = time_ms([&] { launch8<E4M3, E4M3, false, false, net::EPI_ACT_BF16>(g, 1, 0); }, 20);
    const float ta = time_ms([&] { amax_bf16_impl_k<<<240, 256>>>(X, MR, r.K, r.K, am); amax_bf16_impl_k<<<240, 256>>>(W, r.N, r.K, r.K, am + 1); }, 20);
    std::printf("  fwd %s: BF16 %.3f ms %.1f TFLOPS | FP8 GEMM %.3f ms %.1f TFLOPS (x%.2f) | + amax %.3f ms -> x%.2f\n", r.nm, tb, f / tb / 1e9, t8g, f / t8g / 1e9, tb / t8g, ta,
                tb / (t8g + ta));
  }
  // dgrad·wgrad (A2 모양: N 256, K 272 → dX 256 칸)
  {
    const net::LayerDesc L{272, 256, 272, 256, net::ACT_ELU, 1.f};
    const double fd = 2.0 * MR * 256 * 256, fw = 2.0 * 256 * 272 * MR;
    const float tdb = time_ms([&] { net::gemm_dx_dact(L, dZ, MR, W, X, 256, Y, 0, 0); }, 20);
    G8 g{};
    net::GemmP& p = g.p[0];
    p.A = dZ; p.lda = 256; p.B = W; p.ldb = 272; p.M = MR; p.N = 256; p.K = 256; p.C = Y; p.ldc = 256; p.Y = X; p.ldy = 272;
    g.p[1] = p; g.amA[0] = g.amA[1] = am; g.amB[0] = g.amB[1] = am + 1; g.zper = 1;
    const float td8 = time_ms([&] { launch8<E5M2, E4M3, false, true, net::EPI_DACT_BF16>(g, 1, 0); }, 20);
    std::printf("  dgrad A2 (dZ 256 -> dX 256): BF16 %.3f ms %.1f TFLOPS | FP8 %.3f ms %.1f TFLOPS (x%.2f)\n", tdb, fd / tdb / 1e9, td8, fd / td8 / 1e9, tdb / td8);
    const int ch = 1024;
    const float twb = time_ms([&] { net::gemm_dw(L, dZ, X, MR, ws, ch, 0); }, 20);
    G8 w{};
    net::GemmP& q = w.p[0];
    q.A = dZ; q.lda = 256; q.B = X; q.ldb = 272; q.M = 256; q.N = 272; q.K = MR; q.C = ws; q.ldc = 272; q.kchunk = ch;
    w.p[1] = q; w.amA[0] = w.amA[1] = am; w.amB[0] = w.amB[1] = am + 1; w.zper = MR / ch;
    const float tw8 = time_ms([&] { launch8<E5M2, E4M3, true, true, net::EPI_SPLIT_F32>(w, 1, 0); }, 20);
    std::printf("  wgrad A2 (dW 256x272, K = 65,536 rows, split 1,024): BF16 %.3f ms %.1f TFLOPS | FP8 %.3f ms %.1f TFLOPS (x%.2f)\n", twb, fw / twb / 1e9, tw8, fw / tw8 / 1e9,
                twb / tw8);
  }
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(sa); cudaFree(sb); cudaFree(X); cudaFree(W); cudaFree(Y); cudaFree(dZ); cudaFree(ws); cudaFree(am);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: fp8_verify probe | quant | gemm [--negative] | bench\n"); return 2; }
  const std::string m = argv[1];
  const bool neg = argc > 2 && !std::strcmp(argv[2], "--negative");
  if (m == "probe") return run_probe();
  if (m == "quant") return run_quant();
  if (m == "gemm") return run_gemm(neg);
  if (m == "bench") return run_bench();
  std::fprintf(stderr, "unknown mode %s\n", m.c_str());
  return 2;
}
