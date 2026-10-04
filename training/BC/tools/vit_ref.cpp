// SigLIP 2 B/32-256 영상 탑 CPU 참조판(검증 9절, 얼린 인코더의 앞 계산만):
//   FP64 : 모두 double, 가중치 FP32 원본 → "정답"
//   EMUL : GPU 정밀도 처방 흉내 — GEMM 입력(패치·LN 출력·어텐션 출력·GELU 출력)과 가중치·qkv 를 FP16(set_half(false) 면 BF16) 으로, 끝 토큰 bf16, 잔차 X 는 float,
//          편향·LN·softmax·GELU 는 double(GPU 는 FP32 — bf16 반올림보다 훨씬 작은 차이) → 정답과의 차가 "바닥"
#include "vit_ref.h"

#include "fp8_ref.h"

#include <cmath>
#include <algorithm>
#include <cstring>

namespace vitref {

using namespace vit;

static inline float rbf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffffu) > 0x7f800000u) return f;
  x = (x + 0x7fffu + ((x >> 16) & 1u)) & 0xffff0000u;
  float o;
  std::memcpy(&o, &x, 4);
  return o;
}
// FP16 반올림(가장 가까운 짝수, 아래넘침 포함)
static inline float rh(float f) {
  if (!std::isfinite(f)) return f;
  const float a = std::fabs(f);
  if (a >= 65520.f) return std::copysign(INFINITY, f);
  int e;
  std::frexp(a, &e);
  const float ulp = std::ldexp(1.f, std::max(e - 11, -24));
  return std::copysign(std::nearbyint(a / ulp) * ulp, f);
}
static bool g_half = true;
void set_half(bool h) { g_half = h; }
static uint8_t g_f8[LAYERS] = {};
static uint8_t g_h16[LAYERS] = {};
static bool g_h16p = false;
void set_h16(const uint8_t* h, bool patch) { for (int l = 0; l < LAYERS; ++l) g_h16[l] = h ? h[l] : 0; g_h16p = patch; }
static bool g_hacc = false;
static bool g_p16 = true;   // EMUL: 어텐션 P 를 16 비트로(GPU Encoder::kern 1)
void set_attn_p16(bool on) { g_p16 = on; }
static int g_hpromo = 1;   // FP16 부분합을 FP32 로 옮기는 간격(k 32 단위) — GPU TN_HPROMO 와 같게
void set_hpromo(int p) { g_hpromo = p; }   // 다음 gemm 부름이 FP16 누산 흉내인가(forward 가 부르기 전에 정함)
void set_f8(const uint8_t* f8) { for (int l = 0; l < LAYERS; ++l) g_f8[l] = f8 ? f8[l] : 0; }
// EMUL FP8: 행 [T][K] 마다 amax(float) → 2 의 거듭제곱 배율 → E4M3 반올림한 값(GPU 의 ln8_k·rowq_k 와 같은 식)
static void q8rows(double* A, int T, int K) {
#pragma omp parallel for schedule(static)
  for (int t = 0; t < T; ++t) {
    float am = 0.f;
    for (int k = 0; k < K; ++k) am = std::fmax(am, std::fabs((float)A[(size_t)t * K + k]));
    const float inv = f8ref::pow2_inv(am, 0);
    for (int k = 0; k < K; ++k) A[(size_t)t * K + k] = f8ref::qd((float)A[(size_t)t * K + k], inv, 0);
  }
}
static std::vector<float> w8conv(const std::vector<float>& w, int N, int K) {
  std::vector<float> o(w.size());
  for (int n = 0; n < N; ++n) {
    float am = 0.f;
    for (int k = 0; k < K; ++k) am = std::fmax(am, std::fabs(w[(size_t)n * K + k]));
    const float inv = f8ref::pow2_inv(am, 0);
    for (int k = 0; k < K; ++k) o[(size_t)n * K + k] = (float)f8ref::qd(w[(size_t)n * K + k], inv, 0);
  }
  return o;
}
static inline float r16(float f) { return g_half ? rh(f) : rbf(f); }
static inline double q(int md, double v) { return md == EMUL ? (double)r16((float)v) : v; }

struct WCache { std::vector<float> w; };
static std::vector<float> wconv(int md, const std::vector<float>& w) {
  std::vector<float> o(w.size());
  for (size_t i = 0; i < w.size(); ++i) o[i] = md == EMUL ? r16(w[i]) : w[i];
  return o;
}
// Y[t][n] = Σ_k A[t][k] W[n][k] + b[n]
static void gemm(const double* A, int T, int K, const std::vector<float>& W, const std::vector<float>& b, int N, double* Y) {
#pragma omp parallel for schedule(static)
  for (int t = 0; t < T; ++t) {
    const double* a = A + (size_t)t * K;
    for (int n = 0; n < N; ++n) {
      const float* w = W.data() + (size_t)n * K;
      double s = 0.0;
      if (g_hacc) {   // FP16 누산 흉내: k16 정확 합을 FP16 부분합에 더해 반올림, k32 마다 FP32 누산기로
        float acc = 0.f;
        for (int k0 = 0; k0 < K; k0 += 32 * g_hpromo) {
          float h = 0.f;
          for (int k1 = k0; k1 < std::min(K, k0 + 32 * g_hpromo); k1 += 16) {
            double p = 0.0;
            for (int k = k1; k < k1 + 16; ++k) p += a[k] * (double)w[k];
            h = rh((float)((double)h + p));
          }
          acc = acc + h;
        }
        s = acc;
      } else {
        for (int k = 0; k < K; ++k) s += a[k] * (double)w[k];
      }
      Y[(size_t)t * N + n] = s + (double)b[n];
    }
  }
}
static void layernorm(int md, const double* X, int T, const std::vector<float>& g, const std::vector<float>& b, double* Y, bool out_bf16 = false) {
#pragma omp parallel for schedule(static)
  for (int t = 0; t < T; ++t) {
    const double* x = X + (size_t)t * D;
    double m = 0, v = 0;
    for (int c = 0; c < D; ++c) m += x[c];
    m /= D;
    for (int c = 0; c < D; ++c) v += (x[c] - m) * (x[c] - m);
    v /= D;
    const double r = 1.0 / std::sqrt(v + (double)LN_EPS);
    for (int c = 0; c < D; ++c) {
      const double y = (x[c] - m) * r * g[c] + b[c];
      Y[(size_t)t * D + c] = md != EMUL ? y : out_bf16 ? (double)rbf((float)y) : q(md, y);
    }
  }
}

void forward(int md, const HostWeights& hw, const std::vector<const uint8_t*>& imgs, int layers, std::vector<double>& out) {
  const int n = (int)imgs.size(), T = n * NTOK;
  std::vector<double> P((size_t)T * KP), X((size_t)T * D), Y((size_t)T * D), QKV((size_t)T * 3 * D), H((size_t)T * MLP), tmp((size_t)T * D);
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < NTOK; ++p)
      for (int k = 0; k < KP; ++k) {
        const int ch = k / (PATCH * PATCH), ky = (k % (PATCH * PATCH)) / PATCH, kx = k % PATCH;
        const int y = (p / GRID) * PATCH + ky, x = (p % GRID) * PATCH + kx;
        const uint8_t u = imgs[i][((size_t)y * IMG + x) * 3 + ch];
        double v;
        if (md == EMUL) { const float f = (float)u / 255.f; v = r16((f - 0.5f) / 0.5f); }
        else v = ((double)u / 255.0 - 0.5) / 0.5;
        P[((size_t)i * NTOK + p) * KP + k] = v;
      }
  g_hacc = md == EMUL && g_half && g_h16p;
  gemm(P.data(), T, KP, wconv(md, hw.patch_w), hw.patch_b, D, X.data());
  g_hacc = false;
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < D; ++c) {
      double& x = X[(size_t)t * D + c];
      x = x + hw.pos[(size_t)(t % NTOK) * D + c];
      if (md == EMUL) x = (float)x;
    }
  for (int l = 0; l < layers; ++l) {
    const auto& B = hw.blk[l];
    const uint8_t f8 = md == EMUL && g_half ? g_f8[l] : 0;
    const uint8_t h16 = md == EMUL && g_half ? (uint8_t)(g_h16[l] & ~f8) : 0;
    layernorm(f8 & F8_QKV ? FP64 : md, X.data(), T, B.ln1_g, B.ln1_b, Y.data());
    if (f8 & F8_QKV) { for (auto& v : Y) v = (float)v; q8rows(Y.data(), T, D); }   // GPU: LN 출력 FP32 → FP8
    g_hacc = h16 & F8_QKV;
    gemm(Y.data(), T, D, f8 & F8_QKV ? w8conv(B.qkv_w, 3 * D, D) : wconv(md, B.qkv_w), B.qkv_b, 3 * D, QKV.data());
    for (auto& v : QKV) v = q(md, v);
#pragma omp parallel for schedule(static)
    for (int ih = 0; ih < n * HEADS; ++ih) {
      const int i = ih / HEADS, h = ih % HEADS;
      for (int a = 0; a < NTOK; ++a) {
        const double* qa = QKV.data() + ((size_t)i * NTOK + a) * 3 * D + h * HD;
        double s[NTOK], mx = -1e300;
        for (int j = 0; j < NTOK; ++j) {
          const double* kj = QKV.data() + ((size_t)i * NTOK + j) * 3 * D + D + h * HD;
          double d = 0;
          for (int e = 0; e < HD; ++e) d += qa[e] * kj[e];
          s[j] = d * 0.125;
          mx = std::fmax(mx, s[j]);
        }
        double sum = 0;
        for (int j = 0; j < NTOK; ++j) {
          s[j] = std::exp(s[j] - mx);
          if (md == EMUL && g_p16) s[j] = r16((float)s[j]);   // GPU 텐서 코어 어텐션: P 를 16 비트로, 합은 반올림한 값의 합
          sum += s[j];
        }
        for (int e = 0; e < HD; ++e) {
          double o = 0;
          for (int j = 0; j < NTOK; ++j) o += s[j] * QKV[((size_t)i * NTOK + j) * 3 * D + 2 * D + h * HD + e];
          Y[((size_t)i * NTOK + a) * D + h * HD + e] = q(md, o / sum);
        }
      }
    }
    if (f8 & F8_PROJ) q8rows(Y.data(), T, D);   // 어텐션 출력(FP16) → FP8
    g_hacc = h16 & F8_PROJ;
    gemm(Y.data(), T, D, f8 & F8_PROJ ? w8conv(B.proj_w, D, D) : wconv(md, B.proj_w), B.proj_b, D, tmp.data());
    for (size_t k = 0; k < X.size(); ++k) { X[k] = X[k] + tmp[k]; if (md == EMUL) X[k] = (float)X[k]; }
    layernorm(f8 & F8_FC1 ? FP64 : md, X.data(), T, B.ln2_g, B.ln2_b, Y.data());
    if (f8 & F8_FC1) { for (auto& v : Y) v = (float)v; q8rows(Y.data(), T, D); }
    g_hacc = h16 & F8_FC1;
    gemm(Y.data(), T, D, f8 & F8_FC1 ? w8conv(B.fc1_w, MLP, D) : wconv(md, B.fc1_w), B.fc1_b, MLP, H.data());
    for (auto& v : H) {
      const double u = 0.7978845608028654 * (v + 0.044715 * v * v * v);
      v = q(md, 0.5 * v * (1.0 + std::tanh(u)));
    }
    if (f8 & F8_FC2) q8rows(H.data(), T, MLP);   // GELU 출력(FP16) → FP8
    g_hacc = h16 & F8_FC2;
    gemm(H.data(), T, MLP, f8 & F8_FC2 ? w8conv(B.fc2_w, D, MLP) : wconv(md, B.fc2_w), B.fc2_b, D, tmp.data());
    g_hacc = false;
    for (size_t k = 0; k < X.size(); ++k) { X[k] = X[k] + tmp[k]; if (md == EMUL) X[k] = (float)X[k]; }
  }
  out.assign((size_t)T * D, 0.0);
  layernorm(md, X.data(), T, hw.norm_g, hw.norm_b, out.data(), true);   // 학생 입력 토큰은 bf16
}

}  // namespace vitref
