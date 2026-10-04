// tf CPU 참조판 — tf.cu 의 계산을 같은 차례로 FP64(정답)·EMUL(바닥)로. 검증 전용.
#include "tf_ref.h"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace tfref {
using namespace tfm;

static int g_kblk = 16, g_chunk = 1024;
void tc_model(int kblk, int chunk) { g_kblk = kblk; g_chunk = chunk; }
static inline float rz(double x) {
  float f = (float)x;
  if (std::fabs((double)f) > std::fabs(x)) f = std::nextafter(f, 0.f);
  return f;
}
static inline double bfr(double x) { return (double)net::rbf((float)x); }
static inline double eluf(Mode md, double z) {
  if (md == FP64) return z > 0 ? z : std::expm1(z);
  const float f = (float)z;
  return (double)(f > 0.f ? f : std::expm1(f));
}
static inline double elug(double y) { return y > 0 ? 1.0 : y + 1.0; }

// 텐서 코어 합: Σ_k a[k·sa]·b[k·sb]
static inline double tdot(Mode md, const double* a, long long sa, const double* b, long long sb, int K) {
  if (md == FP64) {
    double s = 0;
    for (int k = 0; k < K; ++k) s += a[k * sa] * b[k * sb];
    return s;
  }
  float s = 0.f;
  for (int k0 = 0; k0 < K; k0 += g_kblk) {
    double t = s;
    for (int k = k0; k < K && k < k0 + g_kblk; ++k) t += a[k * sa] * b[k * sb];
    s = rz(t);
  }
  return s;
}
static inline double outv(Mode md, double z, int kind, double prev) {
  if (kind == 0) return md == EMUL ? bfr(z) : z;
  if (kind == 1) { const double e = eluf(md, z); return md == EMUL ? bfr(e) : e; }
  if (kind == 2) return md == EMUL ? (double)(float)z : z;
  return md == EMUL ? (double)((float)prev + (float)z) : prev + z;
}
void lin(Mode md, const double* X, int ldx, int M, const double* W, int N, int K, double* out, int ldo, int kind) {
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const double z = tdot(md, X + (long long)m * ldx, 1, W + (long long)n * K, 1, K);
      out[(long long)m * ldo + n] = outv(md, z, kind, out[(long long)m * ldo + n]);
    }
}
// dX = dZ[M][Nz]·W[:, 0:Np]  kind 0 bf16, 2 f32, 3 f32 +=, 4 bf16 ⊙ elu'(Y)
static void ldx(Mode md, const double* dZ, int M, const double* W, int Nz, int Kw, int Np, double* out, int ldo, int kind, const double* Y, int ldy) {
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; ++m)
    for (int j = 0; j < Np; ++j) {
      double z = tdot(md, dZ + (long long)m * Nz, 1, W + j, Kw, Nz);
      double* o = out + (long long)m * ldo + j;
      if (kind == 4) {
        z = md == EMUL ? (double)((float)z * (float)elug(Y[(long long)m * ldy + j])) : z * elug(Y[(long long)m * ldy + j]);
        *o = md == EMUL ? bfr(z) : z;
      } else *o = outv(md, z, kind, *o);
    }
}
// dW[N][K] = Σ_m dZ[m][n]·X[m][k] (EMUL: 조각 g_chunk 행, 조각 안 16 행 묶음 RZ, 조각 합 float 차례)
static void ldw(Mode md, const double* dZ, int M, int N, const double* X, int ldxx, int K, double* G) {
#pragma omp parallel for schedule(dynamic, 4)
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      if (md == FP64) {
        double s = 0;
        for (int m = 0; m < M; ++m) s += dZ[(long long)m * N + n] * X[(long long)m * ldxx + k];
        G[(long long)n * K + k] = s;
      } else {
        float tot = 0.f;
        for (int c0 = 0; c0 < M; c0 += g_chunk) {
          const int c1 = std::min(M, c0 + g_chunk);
          float a = 0.f;
          for (int m0 = c0; m0 < c1; m0 += g_kblk) {
            double t = a;
            for (int m = m0; m < c1 && m < m0 + g_kblk; ++m) t += dZ[(long long)m * N + n] * X[(long long)m * ldxx + k];
            a = rz(t);
          }
          tot = tot + a;
        }
        G[(long long)n * K + k] = tot;
      }
    }
}

constexpr double EPS = 1e-5;
void ln_fwd(Mode md, const double* X, int R, int d, const double* g, const double* b, double* out, int ldo, double* mu, double* rs) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < R; ++r) {
    const double* x = X + (long long)r * d;
    if (md == FP64) {
      double m = 0, v = 0;
      for (int c = 0; c < d; ++c) m += x[c];
      m /= d;
      for (int c = 0; c < d; ++c) v += (x[c] - m) * (x[c] - m);
      v /= d;
      const double s = 1.0 / std::sqrt(v + EPS);
      for (int c = 0; c < d; ++c) out[(long long)r * ldo + c] = (x[c] - m) * s * g[c] + b[c];
      if (mu) { mu[r] = m; rs[r] = s; }
    } else {
      float m = 0.f, v = 0.f;
      for (int c = 0; c < d; ++c) m = m + (float)x[c];
      m = m / (float)d;
      for (int c = 0; c < d; ++c) { const float e = (float)x[c] - m; v = v + e * e; }
      v = v / (float)d;
      const float s = 1.f / std::sqrt(v + 1e-5f);
      for (int c = 0; c < d; ++c) out[(long long)r * ldo + c] = bfr(((float)x[c] - m) * s * (float)g[c] + (float)b[c]);
      if (mu) { mu[r] = m; rs[r] = s; }
    }
  }
}
void ln_bwd(Mode md, const double* dy, const double* X, int R, int d, const double* g, double* dR, double* dg, double* db) {
  std::vector<double> mu(R), rs(R);
#pragma omp parallel for schedule(static)
  for (int r = 0; r < R; ++r) {
    const double* x = X + (long long)r * d;
    const double* y = dy + (long long)r * d;
    double* o = dR + (long long)r * d;
    if (md == FP64) {
      double m = 0, v = 0;
      for (int c = 0; c < d; ++c) m += x[c];
      m /= d;
      for (int c = 0; c < d; ++c) v += (x[c] - m) * (x[c] - m);
      v /= d;
      const double s = 1.0 / std::sqrt(v + EPS);
      mu[r] = m; rs[r] = s;
      double s1 = 0, s2 = 0;
      for (int c = 0; c < d; ++c) { const double gg = y[c] * g[c]; s1 += gg; s2 += gg * (x[c] - m) * s; }
      s1 /= d; s2 /= d;
      for (int c = 0; c < d; ++c) o[c] += s * (y[c] * g[c] - s1 - (x[c] - m) * s * s2);
    } else {
      float m = 0.f, v = 0.f;
      for (int c = 0; c < d; ++c) m = m + (float)x[c];
      m = m / (float)d;
      for (int c = 0; c < d; ++c) { const float e = (float)x[c] - m; v = v + e * e; }
      v = v / (float)d;
      const float s = 1.f / std::sqrt(v + 1e-5f);
      mu[r] = m; rs[r] = s;
      float s1 = 0.f, s2 = 0.f;
      for (int c = 0; c < d; ++c) { const float gg = (float)y[c] * (float)g[c]; s1 = s1 + gg; s2 = s2 + gg * (((float)x[c] - m) * s); }
      s1 = s1 / (float)d; s2 = s2 / (float)d;
      for (int c = 0; c < d; ++c) {
        const float xh = ((float)x[c] - m) * s;
        o[c] = (double)((float)o[c] + s * ((float)y[c] * (float)g[c] - s1 - xh * s2));
      }
    }
  }
  if (!dg) return;
  for (int c = 0; c < d; ++c) {
    if (md == FP64) {
      double a = 0, b = 0;
      for (int r = 0; r < R; ++r) { a += dy[(long long)r * d + c] * (X[(long long)r * d + c] - mu[r]) * rs[r]; b += dy[(long long)r * d + c]; }
      dg[c] = a; db[c] = b;
    } else {
      float ta = 0.f, tb = 0.f;
      for (int r0 = 0; r0 < R; r0 += 256) {
        float a = 0.f, b = 0.f;
        for (int r = r0; r < R && r < r0 + 256; ++r) {
          const float gy = (float)dy[(long long)r * d + c];
          a = a + gy * (((float)X[(long long)r * d + c] - (float)mu[r]) * (float)rs[r]);
          b = b + gy;
        }
        ta = ta + a; tb = tb + b;
      }
      dg[c] = ta; db[c] = tb;
    }
  }
}

const double kScale = 0.125;   // 1/√64
void attn_fwd(Mode md, const double* Q, const double* K, const double* Vv, const uint8_t* valid, int nq, int nk, double* O) {
  std::vector<double> p(nk);
  for (int i = 0; i < nq; ++i) {
    double m = -INFINITY;
    for (int j = 0; j < nk; ++j) {
      if (valid && !valid[j]) { p[j] = -INFINITY; continue; }
      double s = 0;
      if (md == FP64) for (int e = 0; e < 64; ++e) s += Q[i * 64 + e] * K[j * 64 + e];
      else { float f = 0.f; for (int e = 0; e < 64; ++e) f = f + (float)Q[i * 64 + e] * (float)K[j * 64 + e]; s = (double)(f * (float)kScale); }
      if (md == FP64) s *= kScale;
      p[j] = s;
      m = std::max(m, s);
    }
    double sum = 0;
    float sumf = 0.f;
    for (int j = 0; j < nk; ++j) {
      if (p[j] == -INFINITY) { p[j] = 0; continue; }
      if (md == FP64) { p[j] = std::exp(p[j] - m); sum += p[j]; }
      else { p[j] = (double)std::exp((float)p[j] - (float)m); sumf = sumf + (float)p[j]; }
    }
    for (int e = 0; e < 64; ++e) {
      if (md == FP64) {
        double o = 0;
        for (int j = 0; j < nk; ++j) o += p[j] * Vv[j * 64 + e];
        O[i * 64 + e] = sum > 0 ? o / sum : 0;
      } else {
        float o = 0.f;
        for (int j = 0; j < nk; ++j) o = o + (float)p[j] * (float)Vv[j * 64 + e];
        O[i * 64 + e] = bfr(sumf > 0.f ? o * (1.f / sumf) : 0.f);
      }
    }
  }
}
void attn_bwd(Mode md, const double* Q, const double* K, const double* Vv, const uint8_t* valid, const double* dO, int nq, int nk, double* dQ,
              double* dK, double* dV) {
  std::vector<double> P((size_t)nq * nk), dS((size_t)nq * nk);
  for (int i = 0; i < nq; ++i) {
    double m = -INFINITY;
    double* p = &P[(size_t)i * nk];
    for (int j = 0; j < nk; ++j) {
      if (valid && !valid[j]) { p[j] = -INFINITY; continue; }
      double s = 0;
      for (int e = 0; e < 64; ++e) s += Q[i * 64 + e] * K[j * 64 + e];
      p[j] = s * kScale;
      m = std::max(m, p[j]);
    }
    double sum = 0;
    for (int j = 0; j < nk; ++j) { p[j] = p[j] == -INFINITY ? 0 : std::exp(p[j] - m); sum += p[j]; }
    for (int j = 0; j < nk; ++j) p[j] /= sum;
    // O 를 같은 식으로 다시(D = dO·O)
    double D = 0;
    for (int e = 0; e < 64; ++e) {
      double o = 0;
      for (int j = 0; j < nk; ++j) o += p[j] * Vv[j * 64 + e];
      if (md == EMUL) o = bfr(o);   // GPU 는 저장된 bf16 O 로 D 를 셈
      D += dO[i * 64 + e] * o;
    }
    for (int j = 0; j < nk; ++j) {
      double dp = 0;
      for (int e = 0; e < 64; ++e) dp += dO[i * 64 + e] * Vv[j * 64 + e];
      dS[(size_t)i * nk + j] = p[j] * (dp - D);
    }
  }
  for (int i = 0; i < nq; ++i)
    for (int e = 0; e < 64; ++e) {
      double a = 0;
      for (int j = 0; j < nk; ++j) a += dS[(size_t)i * nk + j] * K[j * 64 + e];
      dQ[i * 64 + e] = md == EMUL ? bfr((float)(a * kScale)) : a * kScale;
    }
  for (int j = 0; j < nk; ++j)
    for (int e = 0; e < 64; ++e) {
      double a = 0, b = 0;
      for (int i = 0; i < nq; ++i) { a += dS[(size_t)i * nk + j] * Q[i * 64 + e]; b += P[(size_t)i * nk + j] * dO[i * 64 + e]; }
      dK[j * 64 + e] = md == EMUL ? bfr((float)(a * kScale)) : a * kScale;
      dV[j * 64 + e] = md == EMUL ? bfr((float)b) : b;
    }
}

// ---- 전체 ----
struct Ctx {
  Mode md;
  const TfCfg& c;
  const TfLayout& lay;
  V Pw;          // GEMM 가중치(EMUL: bf16 반올림)
  const V& P;    // 원본(LN·종류 임베딩)
  int d, d1, B, L, H, A, KA;
};
// 어텐션 층(판·머리마다) — 키 = [구간 1 (n1, valid) ; 구간 2 (n2)]
static void attn_layer(Ctx& x, const double* q, int ldq, int nq, const double* k1, const double* v1, int ld1, int n1, const double* k2, const double* v2,
                       int ld2, int n2, const std::vector<uint8_t>& tv, double* o, int ldo) {
  const int nk = n1 + n2;
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int b = 0; b < x.B; ++b)
    for (int h = 0; h < x.c.heads; ++h) {
      std::vector<double> Q((size_t)nq * 64), K((size_t)nk * 64), Vv((size_t)nk * 64), O((size_t)nq * 64);
      std::vector<uint8_t> val(nk, 1);
      for (int i = 0; i < nq; ++i) for (int e = 0; e < 64; ++e) Q[i * 64 + e] = q[((long long)b * nq + i) * ldq + h * 64 + e];
      for (int j = 0; j < nk; ++j)
        for (int e = 0; e < 64; ++e) {
          if (j < n1) { K[j * 64 + e] = k1[((long long)b * n1 + j) * ld1 + h * 64 + e]; Vv[j * 64 + e] = v1[((long long)b * n1 + j) * ld1 + h * 64 + e]; }
          else { K[j * 64 + e] = k2[((long long)b * n2 + j - n1) * ld2 + h * 64 + e]; Vv[j * 64 + e] = v2[((long long)b * n2 + j - n1) * ld2 + h * 64 + e]; }
        }
      for (int j = 0; j < n1; ++j) val[j] = tv[(size_t)b * n1 + j];
      attn_fwd(x.md, Q.data(), K.data(), Vv.data(), val.data(), nq, nk, O.data());
      for (int i = 0; i < nq; ++i) for (int e = 0; e < 64; ++e) o[((long long)b * nq + i) * ldo + h * 64 + e] = O[i * 64 + e];
    }
}
static void attn_layer_bwd(Ctx& x, const double* q, int ldq, int nq, const double* k1, const double* v1, int ld1, int n1, const double* k2, const double* v2,
                           int ld2, int n2, const std::vector<uint8_t>& tv, const double* dO, int lddo, double* dq, int lddq, double* dk1, double* dv1,
                           int lddk1, double* dk2, double* dv2, int lddk2) {
  const int nk = n1 + n2;
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int b = 0; b < x.B; ++b)
    for (int h = 0; h < x.c.heads; ++h) {
      std::vector<double> Q((size_t)nq * 64), K((size_t)nk * 64), Vv((size_t)nk * 64), G((size_t)nq * 64), dQ((size_t)nq * 64), dK((size_t)nk * 64),
          dV((size_t)nk * 64);
      std::vector<uint8_t> val(nk, 1);
      for (int i = 0; i < nq; ++i)
        for (int e = 0; e < 64; ++e) {
          Q[i * 64 + e] = q[((long long)b * nq + i) * ldq + h * 64 + e];
          G[i * 64 + e] = dO[((long long)b * nq + i) * lddo + h * 64 + e];
        }
      for (int j = 0; j < nk; ++j)
        for (int e = 0; e < 64; ++e) {
          if (j < n1) { K[j * 64 + e] = k1[((long long)b * n1 + j) * ld1 + h * 64 + e]; Vv[j * 64 + e] = v1[((long long)b * n1 + j) * ld1 + h * 64 + e]; }
          else { K[j * 64 + e] = k2[((long long)b * n2 + j - n1) * ld2 + h * 64 + e]; Vv[j * 64 + e] = v2[((long long)b * n2 + j - n1) * ld2 + h * 64 + e]; }
        }
      for (int j = 0; j < n1; ++j) val[j] = tv[(size_t)b * n1 + j];
      attn_bwd(x.md, Q.data(), K.data(), Vv.data(), val.data(), G.data(), nq, nk, dQ.data(), dK.data(), dV.data());
      for (int i = 0; i < nq; ++i) for (int e = 0; e < 64; ++e) dq[((long long)b * nq + i) * lddq + h * 64 + e] = dQ[i * 64 + e];
      for (int j = 0; j < nk; ++j)
        for (int e = 0; e < 64; ++e) {
          if (j < n1) { dk1[((long long)b * n1 + j) * lddk1 + h * 64 + e] = dK[j * 64 + e]; dv1[((long long)b * n1 + j) * lddk1 + h * 64 + e] = dV[j * 64 + e]; }
          else { dk2[((long long)b * n2 + j - n1) * lddk2 + h * 64 + e] = dK[j * 64 + e]; dv2[((long long)b * n2 + j - n1) * lddk2 + h * 64 + e] = dV[j * 64 + e]; }
        }
    }
}
static const double* Wp(Ctx& x, const WT& w) { return x.Pw.data() + w.off; }
static V buf(long long n, int ld = 0, int col = -1) {
  V v((size_t)n, 0.0);
  if (col >= 0) for (long long r = 0; r < n / ld; ++r) v[(size_t)(r * ld + col)] = 1.0;
  return v;
}
static void tobf(Ctx& x, V& v) { if (x.md == EMUL) for (auto& e : v) e = bfr(e); }

void run(Mode md, const TfCfg& c, const TfLayout& lay, const V& P, const In& in, Trace& tr, bool backward) {
  Ctx x{md, c, lay, P, P, c.d, c.d + 16, in.B, L_TOK, c.H, c.A, (c.A + TEMB + 1 + 15) / 16 * 16};
  if (md == EMUL) for (auto& e : x.Pw) e = bfr(e);
  const int d = x.d, d1 = x.d1, B = x.B, L = x.L, mp = c.mlp, hid = c.obj_hidden;
  const long long R = (long long)B * L, RA = (long long)B * x.H;
  auto fl = [&](double v) { return md == EMUL ? (double)(float)v : v; };
  // 키 유효
  std::vector<uint8_t> tv((size_t)R);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < L; ++t) {
      int g = 0;
      while (g + 1 < N_GRP && t >= grp_tok0(g + 1)) ++g;
      bool v = !(in.off.size() && ((in.off[b] >> g) & 1u));
      if (g == G_OBJ) v = v && ((in.obj[b] >> (t - grp_tok0(G_OBJ))) & 1u);
      tv[(size_t)b * L + t] = v;
    }
  // 임베딩
  V E((size_t)R * d);
  V oh = buf((long long)B * 16 * (hid + 16), hid + 16, hid);
  for (int g = 0; g < N_GRP; ++g) {
    const int rows = B * kGrp[g].n_tok;
    double* Eg = E.data() + (long long)B * grp_tok0(g) * d;
    if (lay.g_w2[g].N) {
      lin(md, in.g[g].data(), kGrp[g].K, rows, Wp(x, lay.g_w1[g]), hid, kGrp[g].K, oh.data(), hid + 16, 1);
      lin(md, oh.data(), hid + 16, rows, Wp(x, lay.g_w2[g]), d, hid + 16, Eg, d, 2);
    } else lin(md, in.g[g].data(), kGrp[g].K, rows, Wp(x, lay.g_w1[g]), d, kGrp[g].K, Eg, d, 2);
  }
  tr.X0.assign((size_t)R * d, 0);
  for (int b = 0; b < B; ++b)
    for (int tk = 0; tk < L; ++tk) {
      int g = 0;
      while (g + 1 < N_GRP && tk >= grp_tok0(g + 1)) ++g;
      const int t = tk - grp_tok0(g), ty = grp_type_of(g, t);
      for (int cc = 0; cc < d; ++cc) {
        const double e = E[((long long)B * grp_tok0(g) + (long long)b * kGrp[g].n_tok + t) * d + cc];
        const double te = P[lay.g_type[g] + (long long)ty * d + cc];
        tr.X0[((long long)b * L + tk) * d + cc] = md == EMUL ? (double)((float)e + (float)te) : e + te;
      }
    }
  // 몸통
  const int Lb = c.layers, Le = c.e_layers;
  tr.Xs.assign(2 * Lb + 1, V());
  tr.Xs[0] = tr.X0;
  tr.A1.assign(Lb, V()); tr.A2.assign(Lb, V()); tr.QKV.assign(Lb, V()); tr.O.assign(Lb, V()); tr.Hh.assign(Lb, V());
  const double* Pd = P.data();
  for (int l = 0; l < Lb; ++l) {
    const auto& w = lay.blk[l];
    tr.A1[l] = buf(R * d1, d1, d);
    ln_fwd(md, tr.Xs[2 * l].data(), (int)R, d, Pd + w.ln1g, Pd + w.ln1b, tr.A1[l].data(), d1, nullptr, nullptr);
    tr.QKV[l].assign((size_t)R * 3 * d, 0);
    lin(md, tr.A1[l].data(), d1, (int)R, Wp(x, w.qkv), 3 * d, d1, tr.QKV[l].data(), 3 * d, 0);
    tr.O[l] = buf(R * d1, d1, d);
    const double* q = tr.QKV[l].data();
    attn_layer(x, q, 3 * d, L, q + d, q + 2 * d, 3 * d, L, nullptr, nullptr, 0, 0, tv, tr.O[l].data(), d1);
    tr.Xs[2 * l + 1] = tr.Xs[2 * l];
    lin(md, tr.O[l].data(), d1, (int)R, Wp(x, w.wo), d, d1, tr.Xs[2 * l + 1].data(), d, 3);
    tr.A2[l] = buf(R * d1, d1, d);
    ln_fwd(md, tr.Xs[2 * l + 1].data(), (int)R, d, Pd + w.ln2g, Pd + w.ln2b, tr.A2[l].data(), d1, nullptr, nullptr);
    tr.Hh[l] = buf(R * (mp + 16), mp + 16, mp);
    lin(md, tr.A2[l].data(), d1, (int)R, Wp(x, w.w1), mp, d1, tr.Hh[l].data(), mp + 16, 1);
    tr.Xs[2 * l + 2] = tr.Xs[2 * l + 1];
    lin(md, tr.Hh[l].data(), mp + 16, (int)R, Wp(x, w.w2), d, mp + 16, tr.Xs[2 * l + 2].data(), d, 3);
  }
  tr.Pf = buf(R * d1, d1, d);
  ln_fwd(md, tr.Xs[2 * Lb].data(), (int)R, d, Pd + lay.lnf_g, Pd + lay.lnf_b, tr.Pf.data(), d1, nullptr, nullptr);
  // 전문가
  tr.Xa.assign(2 * Le + 1, V());
  tr.Xa[0].assign((size_t)RA * d, 0);
  lin(md, in.ain.data(), x.KA, (int)RA, Wp(x, lay.e_in), d, x.KA, tr.Xa[0].data(), d, 2);
  tr.eA1.assign(Le, V()); tr.eQKV.assign(Le, V()); tr.ePKV.assign(Le, V()); tr.eO.assign(Le, V()); tr.eH.assign(Le, V());
  std::vector<V> eA2(Le);
  for (int l = 0; l < Le; ++l) {
    const auto& w = lay.eblk[l];
    tr.eA1[l] = buf(RA * d1, d1, d);
    ln_fwd(md, tr.Xa[2 * l].data(), (int)RA, d, Pd + w.ln1g, Pd + w.ln1b, tr.eA1[l].data(), d1, nullptr, nullptr);
    tr.eQKV[l].assign((size_t)RA * 3 * d, 0);
    lin(md, tr.eA1[l].data(), d1, (int)RA, Wp(x, w.qkv), 3 * d, d1, tr.eQKV[l].data(), 3 * d, 0);
    tr.ePKV[l].assign((size_t)R * 2 * d, 0);
    lin(md, tr.Pf.data(), d1, (int)R, Wp(x, w.kvp), 2 * d, d1, tr.ePKV[l].data(), 2 * d, 0);
    tr.eO[l] = buf(RA * d1, d1, d);
    const double* q = tr.eQKV[l].data();
    const double* kv = tr.ePKV[l].data();
    attn_layer(x, q, 3 * d, x.H, kv, kv + d, 2 * d, L, q + d, q + 2 * d, 3 * d, x.H, tv, tr.eO[l].data(), d1);
    tr.Xa[2 * l + 1] = tr.Xa[2 * l];
    lin(md, tr.eO[l].data(), d1, (int)RA, Wp(x, w.wo), d, d1, tr.Xa[2 * l + 1].data(), d, 3);
    eA2[l] = buf(RA * d1, d1, d);
    ln_fwd(md, tr.Xa[2 * l + 1].data(), (int)RA, d, Pd + w.ln2g, Pd + w.ln2b, eA2[l].data(), d1, nullptr, nullptr);
    tr.eH[l] = buf(RA * (mp + 16), mp + 16, mp);
    lin(md, eA2[l].data(), d1, (int)RA, Wp(x, w.w1), mp, d1, tr.eH[l].data(), mp + 16, 1);
    tr.Xa[2 * l + 2] = tr.Xa[2 * l + 1];
    lin(md, tr.eH[l].data(), mp + 16, (int)RA, Wp(x, w.w2), d, mp + 16, tr.Xa[2 * l + 2].data(), d, 3);
  }
  tr.Ao = buf(RA * d1, d1, d);
  ln_fwd(md, tr.Xa[2 * Le].data(), (int)RA, d, Pd + lay.lne_g, Pd + lay.lne_b, tr.Ao.data(), d1, nullptr, nullptr);
  tr.vel.assign((size_t)RA * x.A, 0);
  lin(md, tr.Ao.data(), d1, (int)RA, Wp(x, lay.e_out), x.A, d1, tr.vel.data(), x.A, 2);
  // 손실
  tr.dz.assign((size_t)RA * x.A, 0);
  double ls = 0;
  float lsf = 0.f;
  for (long long r0 = 0; r0 < RA; r0 += 256) {   // GPU: 블록 256 행 부분합(블록 안 나무 합) → 차례 합
    float blk = 0.f;
    for (long long r = r0; r < RA && r < r0 + 256; ++r)
      for (int k = 0; k < x.A; ++k) {
        const double on = ((in.adim >> k) & 1u) ? in.cmask[(size_t)r] : 0.0;
        const double dd = tr.vel[(size_t)(r * x.A + k)] - in.u[(size_t)(r * x.A + k)];
        ls += on * dd * dd;
        blk = blk + (float)on * (float)dd * (float)dd;
        tr.dz[(size_t)(r * x.A + k)] = md == EMUL ? bfr(2.f * (float)on * (float)dd * (1.f / (float)B)) : 2.0 * on * dd / B;
      }
    lsf = lsf + blk;
  }
  tr.loss = md == EMUL ? (double)(lsf / (float)B) : ls / B;
  if (!backward) return;

  // ---- 뒤 ----
  tr.grad.assign((size_t)lay.total, 0.0);
  double* Gd = tr.grad.data();
  auto dw = [&](const V& dZ, int M, const WT& w, const double* X, int ldxx) { ldw(md, dZ.data(), M, w.N, X, ldxx, w.K, Gd + w.off); };
  auto lnb = [&](const V& dy, const V& X, int Rr, long long go, long long bo, V& dRr) { ln_bwd(md, dy.data(), X.data(), Rr, d, Pd + go, dRr.data(), Gd + go, Gd + bo); };
  V dTa((size_t)RA * d, 0), dRa((size_t)RA * d, 0), dPf((size_t)R * d, 0);
  dw(tr.dz, (int)RA, lay.e_out, tr.Ao.data(), d1);
  ldx(md, tr.dz.data(), (int)RA, Wp(x, lay.e_out), x.A, d1, d, dTa.data(), d, 2, nullptr, 0);
  lnb(dTa, tr.Xa[2 * Le], (int)RA, lay.lne_g, lay.lne_b, dRa);
  // 블록 뒤(공용)
  auto blk_back = [&](const TfLayout::Blk& w, const V& Xin, const V& Xmid, const V& A1v, const V& A2v, const V& Hv, const V& Ov, V& dRr, int Rr,
                      V& dOb) {
    V dRb = dRr;
    tobf(x, dRb);
    dw(dRb, Rr, w.w2, Hv.data(), mp + 16);
    V dH((size_t)Rr * mp);
    ldx(md, dRb.data(), Rr, Wp(x, w.w2), d, mp + 16, mp, dH.data(), mp, 4, Hv.data(), mp + 16);
    dw(dH, Rr, w.w1, A2v.data(), d1);
    V dT((size_t)Rr * d);
    ldx(md, dH.data(), Rr, Wp(x, w.w1), mp, d1, d, dT.data(), d, 2, nullptr, 0);
    lnb(dT, Xmid, Rr, w.ln2g, w.ln2b, dRr);
    dRb = dRr;
    tobf(x, dRb);
    dw(dRb, Rr, w.wo, Ov.data(), d1);
    dOb.assign((size_t)Rr * d, 0);
    ldx(md, dRb.data(), Rr, Wp(x, w.wo), d, d1, d, dOb.data(), d, 0, nullptr, 0);
    (void)Xin; (void)A1v;
  };
  auto qkv_back = [&](const TfLayout::Blk& w, const V& Xin, const V& A1v, const V& dQ, V& dRr, int Rr) {
    dw(dQ, Rr, w.qkv, A1v.data(), d1);
    V dT((size_t)Rr * d);
    ldx(md, dQ.data(), Rr, Wp(x, w.qkv), 3 * d, d1, d, dT.data(), d, 2, nullptr, 0);
    lnb(dT, Xin, Rr, w.ln1g, w.ln1b, dRr);
  };
  for (int l = Le - 1; l >= 0; --l) {
    const auto& w = lay.eblk[l];
    V dO;
    blk_back(w, tr.Xa[2 * l], tr.Xa[2 * l + 1], tr.eA1[l], eA2[l], tr.eH[l], tr.eO[l], dRa, (int)RA, dO);
    V dQ((size_t)RA * 3 * d, 0), dPKV((size_t)R * 2 * d, 0);
    const double* q = tr.eQKV[l].data();
    const double* kv = tr.ePKV[l].data();
    attn_layer_bwd(x, q, 3 * d, x.H, kv, kv + d, 2 * d, L, q + d, q + 2 * d, 3 * d, x.H, tv, dO.data(), d, dQ.data(), 3 * d, dPKV.data(),
                   dPKV.data() + d, 2 * d, dQ.data() + d, dQ.data() + 2 * d, 3 * d);
    dw(dPKV, (int)R, w.kvp, tr.Pf.data(), d1);
    ldx(md, dPKV.data(), (int)R, Wp(x, w.kvp), 2 * d, d1, d, dPf.data(), d, 3, nullptr, 0);
    qkv_back(w, tr.Xa[2 * l], tr.eA1[l], dQ, dRa, (int)RA);
  }
  {
    V dRb = dRa;
    tobf(x, dRb);
    dw(dRb, (int)RA, lay.e_in, in.ain.data(), x.KA);
  }
  V dR((size_t)R * d, 0);
  lnb(dPf, tr.Xs[2 * Lb], (int)R, lay.lnf_g, lay.lnf_b, dR);
  for (int l = Lb - 1; l >= 0; --l) {
    const auto& w = lay.blk[l];
    V dO;
    blk_back(w, tr.Xs[2 * l], tr.Xs[2 * l + 1], tr.A1[l], tr.A2[l], tr.Hh[l], tr.O[l], dR, (int)R, dO);
    V dQ((size_t)R * 3 * d, 0);
    const double* q = tr.QKV[l].data();
    attn_layer_bwd(x, q, 3 * d, L, q + d, q + 2 * d, 3 * d, L, nullptr, nullptr, 0, 0, tv, dO.data(), d, dQ.data(), 3 * d, dQ.data() + d, dQ.data() + 2 * d,
                   3 * d, nullptr, nullptr, 0);
    qkv_back(w, tr.Xs[2 * l], tr.A1[l], dQ, dR, (int)R);
  }
  tr.dX0 = dR;
  // 종류 임베딩
  for (int g = 0; g < N_GRP; ++g)
    for (int ty = 0; ty < kGrp[g].n_type; ++ty) {
      const int n = kGrp[g].n_tok, nt = kGrp[g].n_type, t0 = ty * n / nt, t1 = (ty + 1) * n / nt;
      const int cb = (t1 - t0) >= 256 ? 1 : 256 / (t1 - t0);
      for (int cc = 0; cc < d; ++cc) {
        double s = 0;
        float tot = 0.f;
        for (int b0 = 0; b0 < B; b0 += cb) {
          float a = 0.f;
          for (int b = b0; b < B && b < b0 + cb; ++b)
            for (int t = t0; t < t1; ++t) { const double v = dR[((long long)b * L + grp_tok0(g) + t) * d + cc]; s += v; a = a + (float)v; }
          tot = tot + a;
        }
        Gd[lay.g_type[g] + (long long)ty * d + cc] = md == EMUL ? (double)tot : s;
      }
    }
  // 묶음 임베딩
  V dE((size_t)R * d);
  for (int b = 0; b < B; ++b)
    for (int tk = 0; tk < L; ++tk) {
      int g = 0;
      while (g + 1 < N_GRP && tk >= grp_tok0(g + 1)) ++g;
      const int t = tk - grp_tok0(g);
      for (int cc = 0; cc < d; ++cc) {
        const double v = dR[((long long)b * L + tk) * d + cc];
        dE[((long long)B * grp_tok0(g) + (long long)b * kGrp[g].n_tok + t) * d + cc] = md == EMUL ? bfr(v) : v;
      }
    }
  for (int g = 0; g < N_GRP; ++g) {
    const int rows = B * kGrp[g].n_tok;
    V dEg(dE.begin() + (long long)B * grp_tok0(g) * d, dE.begin() + ((long long)B * grp_tok0(g) + rows) * d);
    if (lay.g_w2[g].N) {
      // OBJ 숨은 다시(앞과 같은 식)
      V ohg = buf((long long)rows * (hid + 16), hid + 16, hid);
      lin(md, in.g[g].data(), kGrp[g].K, rows, Wp(x, lay.g_w1[g]), hid, kGrp[g].K, ohg.data(), hid + 16, 1);
      dw(dEg, rows, lay.g_w2[g], ohg.data(), hid + 16);
      V dh((size_t)rows * hid);
      ldx(md, dEg.data(), rows, Wp(x, lay.g_w2[g]), d, hid + 16, hid, dh.data(), hid, 4, ohg.data(), hid + 16);
      dw(dh, rows, lay.g_w1[g], in.g[g].data(), kGrp[g].K);
    } else dw(dEg, rows, lay.g_w1[g], in.g[g].data(), kGrp[g].K);
  }
  (void)fl;
}

double loss_only(const TfCfg& c, const TfLayout& lay, const V& P, const In& in) {
  Trace tr;
  run(FP64, c, lay, P, in, tr, false);
  return tr.loss;
}

}  // namespace tfref
