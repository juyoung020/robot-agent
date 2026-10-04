// RecallVLA CPU 참조판 — 설명은 vref.h. GPU 코드와 따로 식에서 바로 짰다(같은 함수를 쓰지 않음).
#include "vref.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>

#include "net.h"
#include "tkern.cuh"

namespace vref {
using namespace rvla;
static Mode MD = FP64;
static bool envf(const char* n) { const char* e = getenv(n); return e && e[0] == '1'; }
static double Qb(double x) {
  if (MD == FP64) return x;
  return (double)net::bf2f(net::f2bf((float)x));
}
static void Qv(V& v) { if (MD == EMUL) for (auto& x : v) x = Qb(x); }
static double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double silu(double x) { return x * sigm(x); }
static double dsilu(double x) { const double s = sigm(x); return s * (1.0 + x * (1.0 - s)); }
static double gelu(double x) { const double u = 0.7978845608028654 * (x + 0.044715 * x * x * x); return 0.5 * x * (1.0 + std::tanh(u)); }
static double dgelu(double x) {
  const double u = 0.7978845608028654 * (x + 0.044715 * x * x * x), t = std::tanh(u);
  return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x);
}

// Y[M][N] = X[M][K]·W[N][K]ᵀ (W = 변수 배열 + off)
static V mm(const V& X, int M, int K, const V& W, long long off, int N) {
  V Y((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      const double* x = &X[(size_t)m * K];
      const double* w = &W[off + (size_t)n * K];
      for (int k = 0; k < K; ++k) s += x[k] * w[k];
      Y[(size_t)m * N + n] = s;
    }
  return Y;
}
static V mmdx(const V& dY, int M, int N, const V& W, long long off, int K) {
  V dX((size_t)M * K, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const double d = dY[(size_t)m * N + n];
      if (d == 0) continue;
      const double* w = &W[off + (size_t)n * K];
      for (int k = 0; k < K; ++k) dX[(size_t)m * K + k] += d * w[k];
    }
  return dX;
}
static void mmdw(const V& dY, int M, int N, const V& X, int K, V& G, long long off) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const double d = dY[(size_t)m * N + n];
      if (d == 0) continue;
      for (int k = 0; k < K; ++k) G[off + (size_t)n * K + k] += d * X[(size_t)m * K + k];
    }
}
static void addv(V& a, const V& b) { for (size_t i = 0; i < a.size(); ++i) a[i] += b[i]; }
static V qd(V v) { Qv(v); return v; }

// RMSNorm(1 + w 또는 w) 한 벡터
static void rms_f(const double* x, int D, const double* w, bool p1, double eps, double* y) {
  double s = 0;
  for (int d = 0; d < D; ++d) s += x[d] * x[d];
  const double r = 1.0 / std::sqrt(s / D + eps);
  for (int d = 0; d < D; ++d) y[d] = x[d] * r * (p1 ? 1.0 + w[d] : w[d]);
}
static void rms_b(const double* dy, const double* x, int D, const double* w, bool p1, double eps, double* dx, double* gw) {
  double s = 0, gx = 0;
  for (int d = 0; d < D; ++d) s += x[d] * x[d];
  const double r = 1.0 / std::sqrt(s / D + eps);
  for (int d = 0; d < D; ++d) gx += dy[d] * (p1 ? 1.0 + w[d] : w[d]) * x[d];
  for (int d = 0; d < D; ++d) {
    dx[d] += r * dy[d] * (p1 ? 1.0 + w[d] : w[d]) - x[d] * r * r * r * gx / D;
    gw[d] += dy[d] * x[d] * r;
  }
}
static void ln_f(const double* x, int D, const double* g, const double* b, double eps, double* y) {
  double m = 0;
  for (int d = 0; d < D; ++d) m += x[d];
  m /= D;
  double v = 0;
  for (int d = 0; d < D; ++d) v += (x[d] - m) * (x[d] - m);
  const double r = 1.0 / std::sqrt(v / D + eps);
  for (int d = 0; d < D; ++d) y[d] = (x[d] - m) * r * g[d] + b[d];
}
static void ln_b(const double* dy, const double* x, int D, const double* g, double eps, double* dx, double* gg, double* gb) {
  double m = 0;
  for (int d = 0; d < D; ++d) m += x[d];
  m /= D;
  double v = 0;
  for (int d = 0; d < D; ++d) v += (x[d] - m) * (x[d] - m);
  const double r = 1.0 / std::sqrt(v / D + eps);
  double s1 = 0, s2 = 0;
  for (int d = 0; d < D; ++d) { const double xh = (x[d] - m) * r, gd = dy[d] * g[d]; s1 += gd; s2 += gd * xh; }
  s1 /= D; s2 /= D;
  for (int d = 0; d < D; ++d) {
    const double xh = (x[d] - m) * r, gd = dy[d] * g[d];
    dx[d] += r * (gd - s1 - xh * s2);
    gg[d] += dy[d] * xh;
    gb[d] += dy[d];
  }
}
// RoPE(rotate_half, 앞 rot 차원) 한 머리
static void rope(double* x, int rot, double theta, int pos, bool inv) {
  const int h = rot / 2;
  for (int i = 0; i < h; ++i) {
    const float invf = 1.f / std::pow((float)theta, (float)(2 * i) / (float)rot);
    const double fr = (double)((float)pos * invf), c = std::cos(fr), s = std::sin(fr);
    const double a = x[i], b = x[i + h];
    if (!inv) { x[i] = a * c - b * s; x[i + h] = b * c + a * s; }
    else { x[i] = a * c + b * s; x[i + h] = b * c - a * s; }
  }
}
// 어텐션(한 판): 질의 nq 행 × 머리 Hq, 키 nk 행 × 머리 Hk, ok(t, j) 로 가림.
// EMUL: GPU 텐서 코어 판(flash.cu)과 같은 자리 반올림 — Q·K·V·dO 는 bf16, P = e^{S − lse}(정규화) 는 PV·dV 에서 bf16, dS 는 dQ·dK 에서 bf16,
// D = Σ dO·O(반올림 안 한 dO, O = Σ bf(P)·bf(V)).
struct AttIO { int nqr, nk, Hq, Hk, hd; std::function<bool(int, int)> ok; };
static void att_f(const AttIO& a, const double* Q, const double* K, const double* Vv, double* O, std::vector<double>& P) {
  const Mode m0 = MD;
  if (envf("VREF_ATT_EXACT")) MD = FP64;   // 진단: 어텐션 안 반올림 흉내 끄기
  P.assign((size_t)a.nqr * a.Hq * a.nk, 0.0);
  const double sc = 1.0 / std::sqrt((double)a.hd);
  for (int t = 0; t < a.nqr; ++t)
    for (int h = 0; h < a.Hq; ++h) {
      const int kh = h / (a.Hq / a.Hk);
      double mx = -1e300;
      std::vector<double> s(a.nk, -1e300);
      for (int j = 0; j < a.nk; ++j) {
        if (!a.ok(t, j)) continue;
        double d = 0;
        for (int e = 0; e < a.hd; ++e) d += Qb(Q[((size_t)t * a.Hq + h) * a.hd + e]) * Qb(K[((size_t)j * a.Hk + kh) * a.hd + e]);
        s[j] = d * sc;
        mx = std::max(mx, s[j]);
      }
      double z = 0;
      for (int j = 0; j < a.nk; ++j) if (a.ok(t, j)) z += std::exp(s[j] - mx);
      const double lse = mx + std::log(z);
      for (int e = 0; e < a.hd; ++e) O[((size_t)t * a.Hq + h) * a.hd + e] = 0;
      for (int j = 0; j < a.nk; ++j) {
        if (!a.ok(t, j)) continue;
        const double p = std::exp(s[j] - lse), pb = Qb(p);
        P[((size_t)t * a.Hq + h) * a.nk + j] = p;
        for (int e = 0; e < a.hd; ++e) O[((size_t)t * a.Hq + h) * a.hd + e] += pb * Qb(Vv[((size_t)j * a.Hk + kh) * a.hd + e]);
      }
    }
  MD = m0;
}
static void att_b(const AttIO& a, const double* Q, const double* K, const double* Vv, const std::vector<double>& P, const double* dO, double* dQ, double* dK,
                  double* dV) {
  const Mode m0 = MD;
  if (envf("VREF_ATT_EXACT")) MD = FP64;   // 진단: 어텐션 안 반올림 흉내 끄기
  const double sc = 1.0 / std::sqrt((double)a.hd);
  for (int t = 0; t < a.nqr; ++t)
    for (int h = 0; h < a.Hq; ++h) {
      const int kh = h / (a.Hq / a.Hk);
      const double* dor = dO + ((size_t)t * a.Hq + h) * a.hd;
      std::vector<double> dp(a.nk, 0.0), orow(a.hd, 0.0);
      double D = 0;
      for (int j = 0; j < a.nk; ++j) {
        const double p = P[((size_t)t * a.Hq + h) * a.nk + j];
        if (p == 0) continue;
        double s = 0;
        for (int e = 0; e < a.hd; ++e) s += Qb(dor[e]) * Qb(Vv[((size_t)j * a.Hk + kh) * a.hd + e]);
        dp[j] = s;
        if (MD == FP64) D += p * s;
        else for (int e = 0; e < a.hd; ++e) orow[e] += Qb(p) * Qb(Vv[((size_t)j * a.Hk + kh) * a.hd + e]);
      }
      if (MD == EMUL) for (int e = 0; e < a.hd; ++e) D += dor[e] * orow[e];
      for (int j = 0; j < a.nk; ++j) {
        const double p = P[((size_t)t * a.Hq + h) * a.nk + j];
        if (p == 0) continue;
        const double ds = Qb(p * (dp[j] - D)), pb = Qb(p);
        for (int e = 0; e < a.hd; ++e) {
          dQ[((size_t)t * a.Hq + h) * a.hd + e] += ds * sc * Qb(K[((size_t)j * a.Hk + kh) * a.hd + e]);
          if (dK) dK[((size_t)j * a.Hk + kh) * a.hd + e] += ds * sc * Qb(Q[((size_t)t * a.Hq + h) * a.hd + e]);
          if (dV) dV[((size_t)j * a.Hk + kh) * a.hd + e] += pb * Qb(dor[e]);
        }
      }
    }
  MD = m0;
}

void att_ref(bool emul, int nqr, int nk, int Hq, int Hk, int hd, const std::function<bool(int, int)>& ok, const double* Q, const double* K, const double* Vv,
             const double* dO, double* O, double* dQ, double* dK, double* dV) {
  const Mode m0 = MD;
  MD = emul ? EMUL : FP64;
  AttIO a{nqr, nk, Hq, Hk, hd, ok};
  std::vector<double> P;
  att_f(a, Q, K, Vv, O, P);
  att_b(a, Q, K, Vv, P, dO, dQ, dK, dV);
  MD = m0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Gated DeltaNet 한 열: 재귀 꼴(정답)과 덩이 꼴(GPU 와 같은 자리 반올림)
static double rbf(double x, bool em) { return em ? (double)net::bf2f(net::f2bf((float)x)) : x; }

void dn_rec_fwd(int n, int dk, int dv, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0, double* o,
                double* S1) {
  V S((size_t)dk * dv, 0.0);
  if (S0) S.assign(S0, S0 + (size_t)dk * dv);
  for (int t = 0; t < n; ++t) {
    const double a = std::exp(g[t]);
    for (auto& x : S) x *= a;
    for (int j = 0; j < dv; ++j) {
      double m = 0;
      for (int i = 0; i < dk; ++i) m += S[(size_t)i * dv + j] * k[(size_t)t * dk + i];
      const double d = be[t] * (v[(size_t)t * dv + j] - m);
      for (int i = 0; i < dk; ++i) S[(size_t)i * dv + j] += k[(size_t)t * dk + i] * d;
    }
    for (int j = 0; j < dv; ++j) {
      double s = 0;
      for (int i = 0; i < dk; ++i) s += S[(size_t)i * dv + j] * q[(size_t)t * dk + i];
      o[(size_t)t * dv + j] = s;
    }
  }
  if (S1) std::memcpy(S1, S.data(), sizeof(double) * dk * dv);
}
void dn_rec_bwd(int n, int dk, int dv, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0,
                const double* dO, const double* dS1, double* dq, double* dkk, double* dvv, double* dg, double* dbe, double* dS0) {
  const size_t NE = (size_t)dk * dv;
  std::vector<V> Sp(n + 1, V(NE, 0.0));   // Sp[t] = S_{t−1}(t 번째 스텝 전)
  if (S0) Sp[0].assign(S0, S0 + NE);
  std::vector<double> otmp((size_t)dv);
  for (int t = 0; t < n; ++t) {
    V S = Sp[t];
    const double a = std::exp(g[t]);
    for (auto& x : S) x *= a;
    for (int j = 0; j < dv; ++j) {
      double m = 0;
      for (int i = 0; i < dk; ++i) m += S[(size_t)i * dv + j] * k[(size_t)t * dk + i];
      const double d = be[t] * (v[(size_t)t * dv + j] - m);
      for (int i = 0; i < dk; ++i) S[(size_t)i * dv + j] += k[(size_t)t * dk + i] * d;
    }
    Sp[t + 1] = S;
  }
  V dS(NE, 0.0);
  if (dS1) dS.assign(dS1, dS1 + NE);
  for (int t = n - 1; t >= 0; --t) {
    const double a = std::exp(g[t]), b = be[t];
    const double* kk = k + (size_t)t * dk;
    const double* qq = q + (size_t)t * dk;
    const double* Sq = Sp[t].data();
    std::vector<double> mv(dv), del(dv), vv(dv), dd(dv), dm(dv);
    for (int j = 0; j < dv; ++j) {
      double s = 0;
      for (int i = 0; i < dk; ++i) s += a * Sq[(size_t)i * dv + j] * kk[i];
      mv[j] = s;
      vv[j] = v[(size_t)t * dv + j];
      del[j] = b * (vv[j] - s);
    }
    const double* doo = dO + (size_t)t * dv;
    for (int i = 0; i < dk; ++i) for (int j = 0; j < dv; ++j) dS[(size_t)i * dv + j] += qq[i] * doo[j];
    for (int i = 0; i < dk; ++i) {
      double s = 0;
      for (int j = 0; j < dv; ++j) s += (a * Sq[(size_t)i * dv + j] + kk[i] * del[j]) * doo[j];
      dq[(size_t)t * dk + i] = s;
    }
    double db = 0;
    for (int j = 0; j < dv; ++j) {
      double s = 0;
      for (int i = 0; i < dk; ++i) s += dS[(size_t)i * dv + j] * kk[i];
      dd[j] = s;
      dvv[(size_t)t * dv + j] = b * s;
      db += s * (vv[j] - mv[j]);
      dm[j] = -b * s;
    }
    dbe[t] = db;
    for (int i = 0; i < dk; ++i) {
      double s = 0;
      for (int j = 0; j < dv; ++j) s += dS[(size_t)i * dv + j] * del[j] + a * Sq[(size_t)i * dv + j] * dm[j];
      dkk[(size_t)t * dk + i] = s;
    }
    double da = 0;
    for (size_t e = 0; e < NE; ++e) {
      const int i = (int)(e / dv), j = (int)(e % dv);
      const double gg = dS[e] + kk[i] * dm[j];
      da += gg * Sq[e];
      dS[e] = a * gg;
    }
    dg[t] = da * a;
  }
  if (dS0) std::memcpy(dS0, dS.data(), sizeof(double) * NE);
}

// 덩이 하나의 앞 값(뒤에서도 다시 씀). 행 Cn 개(나머지 C − Cn 은 0 덧댐: k = v = q = β = g = 0 → 상태에 영향 없음)
struct DnChunk {
  int C, dk, dv;
  V Kb, Qb, Vb, gam, Ga, be, KK, A, T, T1b, T2b, Wb, U, P, Pb;
  void make(int c0, int Cn, bool em, const double* q, const double* k, const double* v, const double* g, const double* bt) {
    Kb.assign((size_t)C * dk, 0); Qb.assign((size_t)C * dk, 0); Vb.assign((size_t)C * dv, 0); gam.assign(C, 0); Ga.assign(C, 0); be.assign(C, 0);
    double acc = 0;
    for (int t = 0; t < C; ++t) {
      if (t < Cn) {
        for (int i = 0; i < dk; ++i) { Kb[(size_t)t * dk + i] = rbf(k[(size_t)(c0 + t) * dk + i], em); Qb[(size_t)t * dk + i] = rbf(q[(size_t)(c0 + t) * dk + i], em); }
        for (int j = 0; j < dv; ++j) Vb[(size_t)t * dv + j] = rbf(v[(size_t)(c0 + t) * dv + j], em);
        acc += g[c0 + t];
        be[t] = bt[c0 + t];
      }
      gam[t] = acc;
      Ga[t] = std::exp(acc);
    }
    KK.assign((size_t)C * C, 0); A.assign((size_t)C * C, 0); P.assign((size_t)C * C, 0); Pb.assign((size_t)C * C, 0);
    for (int t = 0; t < C; ++t)
      for (int s = 0; s <= t; ++s) {
        double a = 0, b = 0;
        for (int i = 0; i < dk; ++i) { a += Kb[(size_t)t * dk + i] * Kb[(size_t)s * dk + i]; b += Qb[(size_t)t * dk + i] * Kb[(size_t)s * dk + i]; }
        KK[(size_t)t * C + s] = a;
        const double M = std::exp(gam[t] - gam[s]);
        if (s < t) A[(size_t)t * C + s] = be[t] * a * M;
        P[(size_t)t * C + s] = b * M;
        Pb[(size_t)t * C + s] = rbf(b * M, em);
      }
    T.assign((size_t)C * C, 0);
    for (int u = 0; u < C; ++u) {
      T[(size_t)u * C + u] = 1;
      for (int t = u + 1; t < C; ++t) {
        double s = 0;
        for (int r = u; r < t; ++r) s += A[(size_t)t * C + r] * T[(size_t)r * C + u];
        T[(size_t)t * C + u] = -s;
      }
    }
    T1b.assign((size_t)C * C, 0); T2b.assign((size_t)C * C, 0);
    for (int t = 0; t < C; ++t)
      for (int s = 0; s <= t; ++s) {
        T1b[(size_t)t * C + s] = rbf(T[(size_t)t * C + s] * be[s], em);
        T2b[(size_t)t * C + s] = rbf(T[(size_t)t * C + s] * be[s] * Ga[s], em);
      }
    U.assign((size_t)C * dv, 0); Wb.assign((size_t)C * dk, 0);
    for (int t = 0; t < C; ++t) {
      for (int j = 0; j < dv; ++j) { double s = 0; for (int r = 0; r <= t; ++r) s += T1b[(size_t)t * C + r] * Vb[(size_t)r * dv + j]; U[(size_t)t * dv + j] = s; }
      for (int i = 0; i < dk; ++i) { double s = 0; for (int r = 0; r <= t; ++r) s += T2b[(size_t)t * C + r] * Kb[(size_t)r * dk + i]; Wb[(size_t)t * dk + i] = rbf(s, em); }
    }
  }
};
// 덩이 하나 앞: S(시작 상태, 제자리로 다음 상태) → o 행 Cn 개, Δ 반환
static void dn_chunk_step(const DnChunk& c, int Cn, bool em, V& S, double* o, V* Dout) {
  const int C = c.C, dk = c.dk, dv = c.dv;
  V Sb(S.size());
  for (size_t e = 0; e < S.size(); ++e) Sb[e] = rbf(S[e], em);
  V D((size_t)C * dv), Db((size_t)C * dv), Dh((size_t)C * dv);
  const double gC = c.gam[C - 1], GC = std::exp(gC);
  for (int t = 0; t < C; ++t)
    for (int j = 0; j < dv; ++j) {
      double s = 0;
      for (int i = 0; i < dk; ++i) s += c.Wb[(size_t)t * dk + i] * Sb[(size_t)i * dv + j];
      const double d = c.U[(size_t)t * dv + j] - s;
      D[(size_t)t * dv + j] = d;
      Db[(size_t)t * dv + j] = rbf(d, em);
      Dh[(size_t)t * dv + j] = rbf(d * std::exp(gC - c.gam[t]), em);
    }
  for (int t = 0; t < Cn; ++t)
    for (int j = 0; j < dv; ++j) {
      double s1 = 0, s2 = 0;
      for (int i = 0; i < dk; ++i) s1 += c.Qb[(size_t)t * dk + i] * Sb[(size_t)i * dv + j];
      for (int r = 0; r <= t; ++r) s2 += c.Pb[(size_t)t * C + r] * Db[(size_t)r * dv + j];
      o[(size_t)t * dv + j] = c.Ga[t] * s1 + s2;
    }
  for (int i = 0; i < dk; ++i)
    for (int j = 0; j < dv; ++j) {
      double s = 0;
      for (int r = 0; r < C; ++r) s += c.Kb[(size_t)r * dk + i] * Dh[(size_t)r * dv + j];
      S[(size_t)i * dv + j] = GC * S[(size_t)i * dv + j] + s;
    }
  if (Dout) *Dout = D;
}
void dn_chk_fwd(int n, int dk, int dv, int C, bool em, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0,
                double* o, double* S1) {
  V S((size_t)dk * dv, 0.0);
  if (S0) S.assign(S0, S0 + S.size());
  DnChunk c{C, dk, dv};
  for (int c0 = 0; c0 < n; c0 += C) {
    const int Cn = std::min(C, n - c0);
    c.make(c0, Cn, em, q, k, v, g, be);
    dn_chunk_step(c, Cn, em, S, o + (size_t)c0 * dv, nullptr);
  }
  if (S1) std::memcpy(S1, S.data(), sizeof(double) * S.size());
}
void dn_chk_bwd(int n, int dk, int dv, int C, bool em, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0,
                const double* dO, const double* dS1, double* dq, double* dkk, double* dvv, double* dg, double* dbe, double* dS0) {
  const size_t NE = (size_t)dk * dv;
  const int nck = (n + C - 1) / C;
  std::vector<V> Sc(nck + 1), Dl(nck);
  Sc[0].assign(NE, 0.0);
  if (S0) Sc[0].assign(S0, S0 + NE);
  std::vector<DnChunk> ch(nck, DnChunk{C, dk, dv});
  V otmp((size_t)C * dv);
  for (int c = 0; c < nck; ++c) {
    const int c0 = c * C, Cn = std::min(C, n - c0);
    ch[c].make(c0, Cn, em, q, k, v, g, be);
    Sc[c + 1] = Sc[c];
    dn_chunk_step(ch[c], Cn, em, Sc[c + 1], otmp.data(), &Dl[c]);
  }
  V dS(NE, 0.0);
  if (dS1) dS.assign(dS1, dS1 + NE);
  for (int ci = nck - 1; ci >= 0; --ci) {
    const DnChunk& c = ch[ci];
    const int c0 = ci * C, Cn = std::min(C, n - c0);
    const V& S = Sc[ci];
    const V& D = Dl[ci];
    const double gC = c.gam[C - 1], GC = std::exp(gC);
    V Sb(NE), dSb(NE);
    for (size_t e = 0; e < NE; ++e) { Sb[e] = rbf(S[e], em); dSb[e] = rbf(dS[e], em); }
    V dOb((size_t)C * dv, 0), dOg((size_t)C * dv, 0), Db((size_t)C * dv), Dh((size_t)C * dv), Dhu((size_t)C * dv);
    for (int t = 0; t < C; ++t)
      for (int j = 0; j < dv; ++j) {
        const double d = D[(size_t)t * dv + j], e = std::exp(gC - c.gam[t]);
        Db[(size_t)t * dv + j] = rbf(d, em);
        Dhu[(size_t)t * dv + j] = d * e;
        Dh[(size_t)t * dv + j] = rbf(d * e, em);
        if (t < Cn) {
          dOb[(size_t)t * dv + j] = rbf(dO[(size_t)(c0 + t) * dv + j], em);
          dOg[(size_t)t * dv + j] = rbf(c.Ga[t] * dO[(size_t)(c0 + t) * dv + j], em);
        }
      }
    // 상태 쪽: dΔ̂ = Kb dSb', dΔ = e^{γC−γ} dΔ̂ + Pbᵀ dOb, dS ← Γ_C dS + Qbᵀ dOg − Wbᵀ dΔb
    V dDh((size_t)C * dv), dD((size_t)C * dv), dDb((size_t)C * dv);
    for (int t = 0; t < C; ++t)
      for (int j = 0; j < dv; ++j) {
        double s = 0, p = 0;
        for (int i = 0; i < dk; ++i) s += c.Kb[(size_t)t * dk + i] * dSb[(size_t)i * dv + j];
        for (int r = t; r < C; ++r) p += c.Pb[(size_t)r * C + t] * dOb[(size_t)r * dv + j];
        dDh[(size_t)t * dv + j] = s;
        dD[(size_t)t * dv + j] = std::exp(gC - c.gam[t]) * s + p;
        dDb[(size_t)t * dv + j] = rbf(dD[(size_t)t * dv + j], em);
      }
    V dSn(NE);
    for (int i = 0; i < dk; ++i)
      for (int j = 0; j < dv; ++j) {
        double s = 0;
        for (int t = 0; t < C; ++t) s += c.Qb[(size_t)t * dk + i] * dOg[(size_t)t * dv + j] - c.Wb[(size_t)t * dk + i] * dDb[(size_t)t * dv + j];
        dSn[(size_t)i * dv + j] = GC * dS[(size_t)i * dv + j] + s;
      }
    // 기울기
    V dgam(C, 0.0), dbt(C, 0.0), dQ((size_t)C * dk, 0.0), dK((size_t)C * dk, 0.0), dWb((size_t)C * dk);
    {   // (d) Γ_C Σ dS∘bf(S)(GPU 는 bf16 검문점을 씀)
      double s = 0;
      for (size_t e = 0; e < NE; ++e) s += dS[e] * Sb[e];
      dgam[C - 1] += GC * s;
    }
    for (int t = 0; t < C; ++t) {
      double rho = 0;
      for (int j = 0; j < dv; ++j) rho += dDh[(size_t)t * dv + j] * Dhu[(size_t)t * dv + j];
      dgam[C - 1] += rho;
      dgam[t] -= rho;
      for (int i = 0; i < dk; ++i) {
        double a = 0, w = 0, kk = 0;
        for (int j = 0; j < dv; ++j) {
          a += dOb[(size_t)t * dv + j] * Sb[(size_t)i * dv + j];
          w += dDb[(size_t)t * dv + j] * Sb[(size_t)i * dv + j];
          kk += Dh[(size_t)t * dv + j] * dSb[(size_t)i * dv + j];
        }
        dQ[(size_t)t * dk + i] = c.Ga[t] * a;
        dWb[(size_t)t * dk + i] = rbf(-w, em);
        dK[(size_t)t * dk + i] = kk;
      }
      double qa = 0;
      for (int i = 0; i < dk; ++i) qa += c.Qb[(size_t)t * dk + i] * dQ[(size_t)t * dk + i];
      dgam[t] += qa;   // (a)
    }
    // dP = dOb Δbᵀ (s ≤ t)
    V dQKb((size_t)C * C, 0.0);
    for (int t = 0; t < C; ++t)
      for (int s = 0; s <= t; ++s) {
        double dp = 0;
        for (int j = 0; j < dv; ++j) dp += dOb[(size_t)t * dv + j] * Db[(size_t)s * dv + j];
        const double x = dp * c.P[(size_t)t * C + s];
        dgam[t] += x; dgam[s] -= x;
        dQKb[(size_t)t * C + s] = rbf(dp * std::exp(c.gam[t] - c.gam[s]), em);
      }
    for (int t = 0; t < C; ++t)
      for (int i = 0; i < dk; ++i) {
        double a = 0, b = 0;
        for (int s = 0; s <= t; ++s) a += dQKb[(size_t)t * C + s] * c.Kb[(size_t)s * dk + i];
        for (int r = t; r < C; ++r) b += dQKb[(size_t)r * C + t] * c.Qb[(size_t)r * dk + i];
        dQ[(size_t)t * dk + i] += a;
        dK[(size_t)t * dk + i] += b;
      }
    // dV = T1bᵀ dΔb, dT1 = dΔb Vbᵀ, dT2 = dWb Kbᵀ, dK += T2bᵀ dWb
    V dT((size_t)C * C, 0.0);
    for (int s = 0; s < C; ++s) {
      if (s < Cn)
        for (int j = 0; j < dv; ++j) {
          double a = 0;
          for (int t = s; t < C; ++t) a += c.T1b[(size_t)t * C + s] * dDb[(size_t)t * dv + j];
          dvv[(size_t)(c0 + s) * dv + j] = a;
        }
      for (int i = 0; i < dk; ++i) {
        double a = 0;
        for (int t = s; t < C; ++t) a += c.T2b[(size_t)t * C + s] * dWb[(size_t)t * dk + i];
        dK[(size_t)s * dk + i] += a;
      }
    }
    for (int t = 0; t < C; ++t)
      for (int s = 0; s <= t; ++s) {
        double d1 = 0, d2 = 0;
        for (int j = 0; j < dv; ++j) d1 += dDb[(size_t)t * dv + j] * c.Vb[(size_t)s * dv + j];
        for (int i = 0; i < dk; ++i) d2 += dWb[(size_t)t * dk + i] * c.Kb[(size_t)s * dk + i];
        const double Tt = c.T[(size_t)t * C + s];
        dbt[s] += d1 * Tt + d2 * Tt * c.Ga[s];
        dgam[s] += d2 * Tt * c.be[s] * c.Ga[s];
        if (s < t) dT[(size_t)t * C + s] = d1 * c.be[s] + d2 * c.be[s] * c.Ga[s];
      }
    // dA = −Tᵀ dT Tᵀ (s < t): X = Tbᵀ dTb, dA = −Xb Tbᵀ
    V Tb((size_t)C * C), dTb((size_t)C * C), Xb((size_t)C * C);
    for (size_t e = 0; e < Tb.size(); ++e) { Tb[e] = rbf(c.T[e], em); dTb[e] = rbf(dT[e], em); }
    for (int a = 0; a < C; ++a)
      for (int b = 0; b < C; ++b) {
        double x = 0;
        for (int r = 0; r < C; ++r) x += Tb[(size_t)r * C + a] * dTb[(size_t)r * C + b];
        Xb[(size_t)a * C + b] = rbf(x, em);
      }
    V dKKs((size_t)C * C, 0.0);
    for (int t = 0; t < C; ++t)
      for (int s = 0; s < t; ++s) {
        double x = 0;
        for (int r = 0; r < C; ++r) x += Xb[(size_t)t * C + r] * Tb[(size_t)s * C + r];
        const double dA = -x, M = std::exp(c.gam[t] - c.gam[s]);
        dbt[t] += dA * c.KK[(size_t)t * C + s] * M;
        const double y = dA * c.A[(size_t)t * C + s];
        dgam[t] += y; dgam[s] -= y;
        const double dkk2 = dA * c.be[t] * M;
        dKKs[(size_t)t * C + s] += dkk2;
        dKKs[(size_t)s * C + t] += dkk2;
      }
    for (int t = 0; t < C; ++t)
      for (int i = 0; i < dk; ++i) {
        double a = 0;
        for (int s = 0; s < C; ++s) a += rbf(dKKs[(size_t)t * C + s], em) * c.Kb[(size_t)s * dk + i];
        dK[(size_t)t * dk + i] += a;
      }
    // 내보내기: dg = 역 누적합
    double acc = 0;
    for (int t = C - 1; t >= 0; --t) {
      acc += dgam[t];
      if (t < Cn) { dg[c0 + t] = acc; dbe[c0 + t] = dbt[t]; }
    }
    for (int t = 0; t < Cn; ++t)
      for (int i = 0; i < dk; ++i) { dq[(size_t)(c0 + t) * dk + i] = dQ[(size_t)t * dk + i]; dkk[(size_t)(c0 + t) * dk + i] = dK[(size_t)t * dk + i]; }
    dS = dSn;
  }
  if (dS0) std::memcpy(dS0, dS.data(), sizeof(double) * NE);
}

// ---------------------------------------------------------------------------------------------------------------------
void run(Mode md, const Model& m, const Params& P, const In& in, Out& out, bool backward) {
  MD = md;
  const VCfg& c = m.c;
  const QCfg& Q = m.q.c;
  const QLayout& ql = m.q.lay;
  const int B = in.B, L = in.L, R = B * L, H = Q.H, D = c.vD, T = c.vT, Rv = B * c.cams * T, RA = B * c.Hc, Mt = in.Mt;
  const int nf = Q.n_full();
  out.g.qW.assign(P.qW.size(), 0.0); out.g.qV.assign(P.qV.size(), 0.0);
  out.g.aW.assign(P.aW.size(), 0.0); out.g.aV.assign(P.aV.size(), 0.0);
  V& gqW = out.g.qW; V& gqV = out.g.qV; V& gaW = out.g.aW; V& gaV = out.g.aV;
  const V& qW = P.qW; const V& qV = P.qV; const V& aW = P.aW; const V& aV = P.aV;
  const double eps = Q.eps;

  // ---- 영상 탑 ----
  struct VS { V x, a1, qkv, o, ao, P, xm, a2, h1, ag; std::vector<std::vector<double>> Pi; };
  std::vector<VS> vs(c.vL);
  V vx = mm(in.patches, Rv, c.vK, aW, m.v_patch.off, D);
  for (int r = 0; r < Rv; ++r) for (int d = 0; d < D; ++d) vx[(size_t)r * D + d] += aV[m.v_patchb + d] + aV[m.v_pos + (r % T) * D + d];
  const int vhd = D / c.vHeads;
  AttIO va{T, T, c.vHeads, c.vHeads, vhd, [](int, int) { return true; }};
  for (int l = 0; l < c.vL; ++l) {
    const auto& b = m.vb[l];
    VS& S = vs[l];
    S.x = vx;
    S.a1.resize((size_t)Rv * D);
    for (int r = 0; r < Rv; ++r) ln_f(&vx[(size_t)r * D], D, &aV[b.ln1g], &aV[b.ln1b], c.vEps, &S.a1[(size_t)r * D]);
    Qv(S.a1);
    S.qkv = mm(S.a1, Rv, D, aW, b.qkv.off, 3 * D);
    for (int r = 0; r < Rv; ++r) for (int k = 0; k < 3 * D; ++k) S.qkv[(size_t)r * 3 * D + k] += aV[b.qkvb + k];
    S.o.assign((size_t)Rv * D, 0.0);
    S.Pi.resize(Rv / T);
    for (int im = 0; im < Rv / T; ++im) {
      V q((size_t)T * D), k((size_t)T * D), v((size_t)T * D);
      for (int t = 0; t < T; ++t) for (int d = 0; d < D; ++d) {
        q[(size_t)t * D + d] = S.qkv[((size_t)im * T + t) * 3 * D + d];
        k[(size_t)t * D + d] = S.qkv[((size_t)im * T + t) * 3 * D + D + d];
        v[(size_t)t * D + d] = S.qkv[((size_t)im * T + t) * 3 * D + 2 * D + d];
      }
      att_f(va, q.data(), k.data(), v.data(), &S.o[(size_t)im * T * D], S.Pi[im]);
    }
    S.ao = qd(S.o);
    S.xm = mm(S.ao, Rv, D, aW, b.proj.off, D);
    for (int r = 0; r < Rv; ++r) for (int d = 0; d < D; ++d) S.xm[(size_t)r * D + d] += vx[(size_t)r * D + d] + aV[b.projb + d];
    S.a2.resize((size_t)Rv * D);
    for (int r = 0; r < Rv; ++r) ln_f(&S.xm[(size_t)r * D], D, &aV[b.ln2g], &aV[b.ln2b], c.vEps, &S.a2[(size_t)r * D]);
    Qv(S.a2);
    S.h1 = mm(S.a2, Rv, D, aW, b.fc1.off, c.vMLP);
    for (int r = 0; r < Rv; ++r) for (int k = 0; k < c.vMLP; ++k) S.h1[(size_t)r * c.vMLP + k] += aV[b.fc1b + k];
    S.ag.resize(S.h1.size());
    for (size_t i = 0; i < S.h1.size(); ++i) S.ag[i] = gelu(S.h1[i]);
    Qv(S.ag);
    vx = mm(S.ag, Rv, c.vMLP, aW, b.fc2.off, D);
    for (int r = 0; r < Rv; ++r) for (int d = 0; d < D; ++d) vx[(size_t)r * D + d] += S.xm[(size_t)r * D + d] + aV[b.fc2b + d];
  }
  const V vxL = vx;
  V tok((size_t)Rv * D);
  for (int r = 0; r < Rv; ++r) ln_f(&vx[(size_t)r * D], D, &aV[m.v_lnfg], &aV[m.v_lnfb], c.vEps, &tok[(size_t)r * D]);
  Qv(tok);
  V vOut = mm(tok, Rv, D, aW, m.v_proj.off, H);
  for (int r = 0; r < Rv; ++r) for (int k = 0; k < H; ++k) vOut[(size_t)r * H + k] += aV[m.v_projb + k];
  // ---- 묶음 ----
  V gOut[N_VG], gh1, gh;
  // 기억 경로: φ 를 [정밀 B·16 | 기억 B·nmax] 줄에 한꺼번에, 그다음 기억 요약 인코더
  const int nl = c.mem_lat, nmax = c.mem_nmax, MI = c.mem_I, nh = c.mem_heads, mhd = c.mem ? H / nh : 1;
  const int Rl = B * nl, Rm = B * nmax, r0 = B * 16, rows_all = r0 + Rm;
  const double rcs = 1.0 / (std::sqrt((double)mhd) * nh);
  V Xall, phi;
  struct MS { V zin, aq, Q, mk, KV, O, Ob, zx, as, QKV, Os, Osb, zs, am, gu, hh; std::vector<std::vector<double>> Pc, Ps; };
  std::vector<MS> ms(c.mem ? c.mem_blk : 0);
  V mz, drl, dex;
  if (c.mem) {
    Xall = in.prec;
    Xall.insert(Xall.end(), in.mem.begin(), in.mem.end());
    gh1 = mm(Xall, rows_all, MEM_K, aW, m.g_w1[VG_OBJ].off, c.obj_hid);
    gh.resize(gh1.size());
    for (size_t i = 0; i < gh1.size(); ++i) gh[i] = gelu(gh1[i]);
    Qv(gh);
    phi = mm(gh, rows_all, c.obj_hid, aW, m.g_w2.off, H);
    for (int r = 0; r < rows_all; ++r) for (int k = 0; k < H; ++k) phi[(size_t)r * H + k] += aV[m.g_b2 + k];
    gOut[VG_OBJ] = phi;   // 정밀 칸 = 앞 B·16 줄
    const double* phim = &phi[(size_t)r0 * H];
    V Iv = mm(in.instr, B, c.instr_k, aW, m.m_instr.off, H);
    mz.assign((size_t)Rl * H, 0.0);
    for (int r = 0; r < Rl; ++r) for (int k = 0; k < H; ++k) mz[(size_t)r * H + k] = aV[m.m_lat + (r % nl) * H + k] + Iv[(size_t)(r / nl) * H + k];
    out.rlog.assign((size_t)B * 2 * (nmax + 1), -1e30);
    for (int kb = 0; kb < c.mem_blk; ++kb) {
      const auto& e = m.mb[kb];
      MS& S = ms[kb];
      S.zin = mz;
      S.aq.resize((size_t)Rl * H);
      for (int r = 0; r < Rl; ++r) rms_f(&mz[(size_t)r * H], H, &aV[e.lnq], true, eps, &S.aq[(size_t)r * H]);
      Qv(S.aq);
      S.Q = mm(S.aq, Rl, H, aW, e.wq.off, H);
      S.mk.resize((size_t)Rm * H);
      for (int r = 0; r < Rm; ++r) rms_f(phim + (size_t)r * H, H, &aV[e.lnk], true, eps, &S.mk[(size_t)r * H]);
      Qv(S.mk);
      S.KV = mm(S.mk, Rm, H, aW, e.wkv.off, 2 * H);
      S.O.assign((size_t)Rl * H, 0.0);
      S.Pc.resize(B);
      for (int b = 0; b < B; ++b) {
        const int n = in.mem_n[b];
        V Kc((size_t)(n + 1) * H), Vc((size_t)(n + 1) * H);
        for (int j = 0; j < n; ++j) for (int d = 0; d < H; ++d) { Kc[(size_t)j * H + d] = S.KV[((size_t)b * nmax + j) * 2 * H + d]; Vc[(size_t)j * H + d] = S.KV[((size_t)b * nmax + j) * 2 * H + H + d]; }
        for (int d = 0; d < H; ++d) { Kc[(size_t)n * H + d] = aV[e.nullkv + d]; Vc[(size_t)n * H + d] = aV[e.nullkv + H + d]; }
        AttIO a{nl, n + 1, nh, nh, mhd, [](int, int) { return true; }};
        att_f(a, &S.Q[(size_t)b * nl * H], Kc.data(), Vc.data(), &S.O[(size_t)b * nl * H], S.Pc[b]);
        if (kb == c.mem_blk - 1)
          for (int r = 0; r < 2; ++r)
            for (int j = 0; j <= n; ++j) {
              const double* kk = j == 0 ? &aV[e.nullkv] : &S.KV[((size_t)b * nmax + j - 1) * 2 * H];
              double d = 0;
              for (int q = 0; q < H; ++q) d += S.Q[((size_t)b * nl + r) * H + q] * kk[q];
              out.rlog[((size_t)b * 2 + r) * (nmax + 1) + j] = rcs * d;
            }
      }
      S.Ob = qd(S.O);
      S.zx = mz;
      addv(S.zx, mm(S.Ob, Rl, H, aW, e.wo.off, H));
      S.as.resize((size_t)Rl * H);
      for (int r = 0; r < Rl; ++r) rms_f(&S.zx[(size_t)r * H], H, &aV[e.lns], true, eps, &S.as[(size_t)r * H]);
      Qv(S.as);
      S.QKV = mm(S.as, Rl, H, aW, e.sqkv.off, 3 * H);
      S.Os.assign((size_t)Rl * H, 0.0);
      S.Ps.resize(B);
      for (int b = 0; b < B; ++b) {
        V q((size_t)nl * H), k((size_t)nl * H), v((size_t)nl * H);
        for (int t = 0; t < nl; ++t) for (int d = 0; d < H; ++d) {
          q[(size_t)t * H + d] = S.QKV[((size_t)b * nl + t) * 3 * H + d];
          k[(size_t)t * H + d] = S.QKV[((size_t)b * nl + t) * 3 * H + H + d];
          v[(size_t)t * H + d] = S.QKV[((size_t)b * nl + t) * 3 * H + 2 * H + d];
        }
        AttIO a{nl, nl, nh, nh, mhd, [](int, int) { return true; }};
        att_f(a, q.data(), k.data(), v.data(), &S.Os[(size_t)b * nl * H], S.Ps[b]);
      }
      S.Osb = qd(S.Os);
      S.zs = S.zx;
      addv(S.zs, mm(S.Osb, Rl, H, aW, e.so.off, H));
      S.am.resize((size_t)Rl * H);
      for (int r = 0; r < Rl; ++r) rms_f(&S.zs[(size_t)r * H], H, &aV[e.lnm], true, eps, &S.am[(size_t)r * H]);
      Qv(S.am);
      S.gu = mm(S.am, Rl, H, aW, e.gu.off, 2 * MI);
      S.hh.resize((size_t)Rl * MI);
      for (int r = 0; r < Rl; ++r) for (int i = 0; i < MI; ++i) S.hh[(size_t)r * MI + i] = silu(S.gu[(size_t)r * 2 * MI + i]) * S.gu[(size_t)r * 2 * MI + MI + i];
      Qv(S.hh);
      mz = S.zs;
      addv(mz, mm(S.hh, Rl, MI, aW, e.dn.off, H));
    }
    gOut[VG_MEM].resize((size_t)Rl * H);
    for (int r = 0; r < Rl; ++r) rms_f(&mz[(size_t)r * H], H, &aV[m.m_lnout], true, eps, &gOut[VG_MEM][(size_t)r * H]);
    out.memtok = gOut[VG_MEM];
    // InfoNCE
    drl.assign((size_t)B * 2 * (nmax + 1), 0.0);
    double lr = 0;
    for (int b = 0; b < B; ++b) for (int r = 0; r < 2; ++r) {
      const int n = in.mem_n[b], t = in.rec_tgt[b * 2 + r];
      if (t < -1 || t >= n) continue;
      const double* l = &out.rlog[((size_t)b * 2 + r) * (nmax + 1)];
      double mx = -1e300;
      for (int j = 0; j <= n; ++j) mx = std::max(mx, l[j]);
      double z = 0;
      for (int j = 0; j <= n; ++j) z += std::exp(l[j] - mx);
      lr += mx + std::log(z) - l[t + 1];
      for (int j = 0; j <= n; ++j) drl[((size_t)b * 2 + r) * (nmax + 1) + j] = c.lam_rec / B * (std::exp(l[j] - mx) / z - (j == t + 1 ? 1.0 : 0.0));
    }
    out.lrec = lr / B;
    // 있음 머리
    dex.assign((size_t)B * 2, 0.0);
    double lx = 0;
    for (int q = 0; q < B * 2; ++q) {
      const double* t = &gOut[VG_MEM][((size_t)(q / 2) * nl + q % 2) * H];
      double v = aV[m.m_ex + H];
      for (int k = 0; k < H; ++k) v += aV[m.m_ex + k] * t[k];
      const int y = in.rec_tgt[q];
      if (y < -1) continue;
      const double yy = y >= 0 ? 1.0 : 0.0;
      lx += std::max(v, 0.0) - v * yy + std::log1p(std::exp(-std::fabs(v)));
      dex[q] = c.lam_ex / B * (1.0 / (1.0 + std::exp(-v)) - yy);
    }
    out.lex = lx / B;
  }
  for (int g = 0; g < N_VG; ++g) {
    if (!m.grp_on(g) || g == VG_MEM || (c.mem && g == VG_OBJ)) continue;
    const int rows = B * kVGrp[g].n_tok, K = kVGrp[g].K;
    if (g == VG_OBJ) {
      gh1 = mm(in.grp[g], rows, K, aW, m.g_w1[g].off, c.obj_hid);
      gh.resize(gh1.size());
      for (size_t i = 0; i < gh1.size(); ++i) gh[i] = gelu(gh1[i]);
      Qv(gh);
      gOut[g] = mm(gh, rows, c.obj_hid, aW, m.g_w2.off, H);
      for (int r = 0; r < rows; ++r) for (int k = 0; k < H; ++k) gOut[g][(size_t)r * H + k] += aV[m.g_b2 + k];
    } else gOut[g] = mm(in.grp[g], rows, K, aW, m.g_w1[g].off, H);
  }
  // ---- 열 ----
  V X((size_t)R * H, 0.0);
  for (int r = 0; r < R; ++r) {
    const int code = in.src[r], kind = (code >> 28) & 15, idx = code & 0x0fffffff;
    for (int k = 0; k < H; ++k) {
      double v = 0;
      if (kind == SK_TXT) v = qW[ql.emb.off + (size_t)idx * H + k];
      else if (kind == SK_IMG) v = vOut[(size_t)idx * H + k];
      else if (kind >= SK_GRP) v = gOut[kind - SK_GRP][(size_t)idx * H + k] + aV[m.g_type[kind - SK_GRP] + k];
      X[(size_t)r * H + k] = v;
    }
  }
  // ---- 몸통 ----
  struct LS {
    V x, a1, t0, xm, a2, gu, hh;
    V q, k, v, o, ag; std::vector<std::vector<double>> Pb;   // 풀
    V pre, t1, qn, kn, be, gg, ov;                            // 선형
  };
  std::vector<LS> ls(Q.layers);
  const int hd = Q.hd, nq = Q.nq, nkv = Q.nkv, kvw = Q.kvw(), qgw = Q.qg(), li = Q.lin_in(), la = Q.lin_all(), lh = Q.lh, dk = Q.dk, dv = Q.dv;
  std::vector<V> Kf(nf), Vf(nf);   // 풀 층 K·V(RoPE 뒤) [R][kvw]
  auto dn_gather = [&](const LS& S, int b, int h, int L_, int lh_, int dk_, int dv_, int li_, V& qs, V& ks, V& vs2, V& gs, V& bs) {
    qs.resize((size_t)L_ * dk_); ks.resize((size_t)L_ * dk_); vs2.resize((size_t)L_ * dv_); gs.resize(L_); bs.resize(L_);
    for (int t = 0; t < L_; ++t) {
      const size_t r = (size_t)b * L_ + t;
      for (int i = 0; i < dk_; ++i) { qs[(size_t)t * dk_ + i] = S.qn[(r * lh_ + h) * dk_ + i]; ks[(size_t)t * dk_ + i] = S.kn[(r * lh_ + h) * dk_ + i]; }
      for (int j = 0; j < dv_; ++j) vs2[(size_t)t * dv_ + j] = S.t1[r * li_ + 2 * lh_ * dk_ + h * dv_ + j];
      gs[t] = S.gg[r * lh_ + h]; bs[t] = S.be[r * lh_ + h];
    }
  };
  int fi = 0;
  for (int l = 0; l < Q.layers; ++l) {
    const auto& Ly = ql.l[l];
    LS& S = ls[l];
    S.x = X;
    S.a1.resize((size_t)R * H);
    for (int r = 0; r < R; ++r) rms_f(&X[(size_t)r * H], H, &qV[Ly.ln1], true, eps, &S.a1[(size_t)r * H]);
    Qv(S.a1);
    V mix;
    if (Q.full[l]) {
      const int W0 = qgw + 2 * kvw;
      S.t0 = mm(S.a1, R, H, qW, Ly.wqkv.off, W0);
      S.q.assign((size_t)R * nq * hd, 0); S.k.assign((size_t)R * kvw, 0); S.v.assign((size_t)R * kvw, 0);
      for (int r = 0; r < R; ++r) {
        const int t = r % L;
        for (int h = 0; h < nq; ++h) { rms_f(&S.t0[(size_t)r * W0 + h * 2 * hd], hd, &qV[Ly.qn], true, eps, &S.q[((size_t)r * nq + h) * hd]); rope(&S.q[((size_t)r * nq + h) * hd], Q.rot, Q.theta, t, false); }
        for (int h = 0; h < nkv; ++h) {
          rms_f(&S.t0[(size_t)r * W0 + qgw + h * hd], hd, &qV[Ly.kn], true, eps, &S.k[((size_t)r * nkv + h) * hd]);
          rope(&S.k[((size_t)r * nkv + h) * hd], Q.rot, Q.theta, t, false);
          for (int e = 0; e < hd; ++e) S.v[((size_t)r * nkv + h) * hd + e] = S.t0[(size_t)r * W0 + qgw + kvw + h * hd + e];
        }
      }
      S.o.assign((size_t)R * nq * hd, 0);
      S.Pb.resize(B);
      AttIO a{L, L, nq, nkv, hd, [](int t, int j) { return j <= t; }};
      for (int b = 0; b < B; ++b) att_f(a, &S.q[(size_t)b * L * nq * hd], &S.k[(size_t)b * L * kvw], &S.v[(size_t)b * L * kvw], &S.o[(size_t)b * L * nq * hd], S.Pb[b]);
      S.ag.resize((size_t)R * nq * hd);
      for (int r = 0; r < R; ++r) for (int h = 0; h < nq; ++h) for (int e = 0; e < hd; ++e)
        S.ag[((size_t)r * nq + h) * hd + e] = S.o[((size_t)r * nq + h) * hd + e] * sigm(S.t0[(size_t)r * W0 + h * 2 * hd + hd + e]);
      Qv(S.ag);
      mix = mm(S.ag, R, nq * hd, qW, Ly.wo.off, H);
      Kf[fi] = S.k; Vf[fi] = S.v;
      ++fi;
    } else {
      S.t0 = mm(S.a1, R, H, qW, Ly.win.off, la);
      S.pre.assign((size_t)R * li, 0); S.t1.assign((size_t)R * li, 0);
      for (int r = 0; r < R; ++r) {
        const int t = r % L;
        for (int ch = 0; ch < li; ++ch) {
          double s = 0;
          for (int kk = 0; kk < Q.conv; ++kk) { const int p = t - (Q.conv - 1) + kk; if (p >= 0) s += qV[Ly.convw + ch * Q.conv + kk] * S.t0[(size_t)(r - t + p) * la + ch]; }
          S.pre[(size_t)r * li + ch] = s;
          S.t1[(size_t)r * li + ch] = silu(s);
        }
      }
      S.qn.assign((size_t)R * lh * dk, 0); S.kn.assign((size_t)R * lh * dk, 0); S.be.assign((size_t)R * lh, 0); S.gg.assign((size_t)R * lh, 0);
      for (int r = 0; r < R; ++r) for (int h = 0; h < lh; ++h) {
        double sq = 0, sk = 0;
        for (int d = 0; d < dk; ++d) { const double a = S.t1[(size_t)r * li + h * dk + d], bb = S.t1[(size_t)r * li + lh * dk + h * dk + d]; sq += a * a; sk += bb * bb; }
        const double iq = 1.0 / std::sqrt(sq + 1e-6), ik = 1.0 / std::sqrt(sk + 1e-6);
        for (int d = 0; d < dk; ++d) {
          S.qn[((size_t)r * lh + h) * dk + d] = S.t1[(size_t)r * li + h * dk + d] * iq / std::sqrt((double)dk);
          S.kn[((size_t)r * lh + h) * dk + d] = S.t1[(size_t)r * li + lh * dk + h * dk + d] * ik;
        }
        S.be[(size_t)r * lh + h] = sigm(S.t0[(size_t)r * la + li + lh * dv + h]);
        const double x = S.t0[(size_t)r * la + li + lh * dv + lh + h] + qV[Ly.dtb + h];
        const double sp = x > 20 ? x : std::log1p(std::exp(x));
        S.gg[(size_t)r * lh + h] = -std::exp(qV[Ly.alog + h]) * sp;
      }
      S.ov.assign((size_t)R * lh * dv, 0);
      for (int b = 0; b < B; ++b) for (int h = 0; h < lh; ++h) {
        V qs, ks, vs2, gs, bs, os((size_t)L * dv);
        dn_gather(S, b, h, L, lh, dk, dv, li, qs, ks, vs2, gs, bs);
        if (MD == EMUL && !envf("VREF_DN_REC")) dn_chk_fwd(L, dk, dv, tk::dn_chunk(dk), true, qs.data(), ks.data(), vs2.data(), gs.data(), bs.data(), nullptr, os.data(), nullptr);
        else dn_rec_fwd(L, dk, dv, qs.data(), ks.data(), vs2.data(), gs.data(), bs.data(), nullptr, os.data(), nullptr);
        for (int t = 0; t < L; ++t) for (int j = 0; j < dv; ++j) S.ov[(((size_t)b * L + t) * lh + h) * dv + j] = os[(size_t)t * dv + j];
      }
      S.ag.resize((size_t)R * lh * dv);
      for (int r = 0; r < R; ++r) for (int h = 0; h < lh; ++h) {
        const double* o = &S.ov[((size_t)r * lh + h) * dv];
        double s = 0;
        for (int d = 0; d < dv; ++d) s += o[d] * o[d];
        const double rs = 1.0 / std::sqrt(s / dv + eps);
        for (int d = 0; d < dv; ++d) S.ag[((size_t)r * lh + h) * dv + d] = qV[Ly.gnw + d] * o[d] * rs * silu(S.t0[(size_t)r * la + li + h * dv + d]);
      }
      Qv(S.ag);
      mix = mm(S.ag, R, lh * dv, qW, Ly.wout.off, H);
    }
    S.xm = X;
    addv(S.xm, mix);
    S.a2.resize((size_t)R * H);
    for (int r = 0; r < R; ++r) rms_f(&S.xm[(size_t)r * H], H, &qV[Ly.ln2], true, eps, &S.a2[(size_t)r * H]);
    Qv(S.a2);
    S.gu = mm(S.a2, R, H, qW, Ly.wgu.off, 2 * Q.I);
    S.hh.resize((size_t)R * Q.I);
    for (int r = 0; r < R; ++r) for (int i = 0; i < Q.I; ++i) S.hh[(size_t)r * Q.I + i] = silu(S.gu[(size_t)r * 2 * Q.I + i]) * S.gu[(size_t)r * 2 * Q.I + Q.I + i];
    Qv(S.hh);
    X = S.xm;
    addv(X, mm(S.hh, R, Q.I, qW, Ly.wdn.off, H));
  }
  const V XL = X;
  V hn((size_t)R * H);
  for (int r = 0; r < R; ++r) rms_f(&X[(size_t)r * H], H, &qV[ql.lnf], true, eps, &hn[(size_t)r * H]);
  out.hidden = hn;
  // ---- 글 ----
  const int Vv = Q.vocab;
  V hnT((size_t)Mt * H), lg;
  for (int t = 0; t < Mt; ++t) for (int k = 0; k < H; ++k) hnT[(size_t)t * H + k] = hn[(size_t)in.tgt_row[t] * H + k];
  Qv(hnT);
  lg = mm(hnT, Mt, H, qW, ql.emb.off, Vv);
  double ltxt = 0;
  V dlg((size_t)Mt * Vv, 0.0);
  for (int t = 0; t < Mt; ++t) {
    double mx = -1e300;
    for (int v = 0; v < Vv; ++v) mx = std::max(mx, lg[(size_t)t * Vv + v]);
    double z = 0;
    for (int v = 0; v < Vv; ++v) z += std::exp(lg[(size_t)t * Vv + v] - mx);
    ltxt += in.tgt_w[t] * (mx + std::log(z) - lg[(size_t)t * Vv + in.tgt_id[t]]);
    for (int v = 0; v < Vv; ++v)
      dlg[(size_t)t * Vv + v] = c.lam_txt * in.tgt_w[t] * (std::exp(lg[(size_t)t * Vv + v] - mx) / z - (v == in.tgt_id[t] ? 1.0 : 0.0));
  }
  // ---- 전문가 ----
    const std::vector<V>& KfE = in.kvfix ? in.kvfix[0] : Kf;
  const std::vector<V>& VfE = in.kvfix ? in.kvfix[1] : Vf;
  const int De = c.De, eW0 = qgw + 2 * kvw;
  struct ES { V x, a1, t0, q, k, v, o, ag, xm, a2, gu, hh; std::vector<std::vector<double>> Pb; };
  std::vector<ES> es(nf);
  V ex = mm(in.ain, RA, c.kpad(), aW, m.e_in.off, De);
  for (int f = 0; f < nf; ++f) {
    const auto& e = m.eb[f];
    ES& S = es[f];
    S.x = ex;
    S.a1.resize((size_t)RA * De);
    for (int r = 0; r < RA; ++r) rms_f(&ex[(size_t)r * De], De, &aV[e.ln1], true, eps, &S.a1[(size_t)r * De]);
    Qv(S.a1);
    S.t0 = mm(S.a1, RA, De, aW, e.qkv.off, eW0);
    S.q.assign((size_t)RA * nq * hd, 0); S.k.assign((size_t)RA * kvw, 0); S.v.assign((size_t)RA * kvw, 0);
    for (int r = 0; r < RA; ++r) {
      const int b = r / c.Hc, t = r % c.Hc, pos = in.plen[b] + t;
      for (int h = 0; h < nq; ++h) { rms_f(&S.t0[(size_t)r * eW0 + h * 2 * hd], hd, &aV[e.qn], true, eps, &S.q[((size_t)r * nq + h) * hd]); rope(&S.q[((size_t)r * nq + h) * hd], Q.rot, Q.theta, pos, false); }
      for (int h = 0; h < nkv; ++h) {
        rms_f(&S.t0[(size_t)r * eW0 + qgw + h * hd], hd, &aV[e.kn], true, eps, &S.k[((size_t)r * nkv + h) * hd]);
        rope(&S.k[((size_t)r * nkv + h) * hd], Q.rot, Q.theta, pos, false);
        for (int d = 0; d < hd; ++d) S.v[((size_t)r * nkv + h) * hd + d] = S.t0[(size_t)r * eW0 + qgw + kvw + h * hd + d];
      }
    }
    S.o.assign((size_t)RA * nq * hd, 0);
    S.Pb.resize(B);
    for (int b = 0; b < B; ++b) {
      const int pl = in.plen[b];
      V K((size_t)(L + c.Hc) * kvw), Vx((size_t)(L + c.Hc) * kvw);
      for (int j = 0; j < L; ++j) for (int d = 0; d < kvw; ++d) { K[(size_t)j * kvw + d] = KfE[f][((size_t)b * L + j) * kvw + d]; Vx[(size_t)j * kvw + d] = VfE[f][((size_t)b * L + j) * kvw + d]; }
      for (int j = 0; j < c.Hc; ++j) for (int d = 0; d < kvw; ++d) { K[(size_t)(L + j) * kvw + d] = S.k[((size_t)b * c.Hc + j) * kvw + d]; Vx[(size_t)(L + j) * kvw + d] = S.v[((size_t)b * c.Hc + j) * kvw + d]; }
      AttIO a{c.Hc, L + c.Hc, nq, nkv, hd, [pl, L](int, int j) { return j < pl || j >= L; }};
      att_f(a, &S.q[(size_t)b * c.Hc * nq * hd], K.data(), Vx.data(), &S.o[(size_t)b * c.Hc * nq * hd], S.Pb[b]);
    }
    S.ag.resize((size_t)RA * nq * hd);
    for (int r = 0; r < RA; ++r) for (int h = 0; h < nq; ++h) for (int d = 0; d < hd; ++d)
      S.ag[((size_t)r * nq + h) * hd + d] = S.o[((size_t)r * nq + h) * hd + d] * sigm(S.t0[(size_t)r * eW0 + h * 2 * hd + hd + d]);
    Qv(S.ag);
    S.xm = ex;
    addv(S.xm, mm(S.ag, RA, nq * hd, aW, e.o.off, De));
    S.a2.resize((size_t)RA * De);
    for (int r = 0; r < RA; ++r) rms_f(&S.xm[(size_t)r * De], De, &aV[e.ln2], true, eps, &S.a2[(size_t)r * De]);
    Qv(S.a2);
    S.gu = mm(S.a2, RA, De, aW, e.gu.off, 2 * c.Ie);
    S.hh.resize((size_t)RA * c.Ie);
    for (int r = 0; r < RA; ++r) for (int i = 0; i < c.Ie; ++i) S.hh[(size_t)r * c.Ie + i] = silu(S.gu[(size_t)r * 2 * c.Ie + i]) * S.gu[(size_t)r * 2 * c.Ie + c.Ie + i];
    Qv(S.hh);
    ex = S.xm;
    addv(ex, mm(S.hh, RA, c.Ie, aW, e.dn.off, De));
  }
  V ao((size_t)RA * De);
  for (int r = 0; r < RA; ++r) rms_f(&ex[(size_t)r * De], De, &aV[m.e_lnf], true, eps, &ao[(size_t)r * De]);
  out.Kf = Kf; out.Vf = Vf;
  Qv(ao);
  out.vel = mm(ao, RA, De, aW, m.e_out.off, c.A);
  double lfm = 0;
  V dz((size_t)RA * c.A, 0.0);
  for (int r = 0; r < RA; ++r) for (int k = 0; k < c.A; ++k) {
    const double on = ((in.adim >> k) & 1u) ? in.cmask[r] : 0.0, d = out.vel[(size_t)r * c.A + k] - in.u[(size_t)r * c.A + k];
    lfm += on * d * d;
    dz[(size_t)r * c.A + k] = c.lam_fm * 2.0 * on * d / B;
  }
  lfm /= B;
  out.ltxt = ltxt; out.lfm = lfm; out.loss = c.lam_txt * ltxt + c.lam_fm * lfm + (c.mem ? c.lam_rec * out.lrec + c.lam_ex * out.lex : 0.0);
  if (!backward) return;

  // ================= 뒤 =================
  // 전문가
  Qv(dz);
  mmdw(dz, RA, c.A, ao, De, gaW, m.e_out.off);
  V dao = mmdx(dz, RA, c.A, aW, m.e_out.off, De);
  V edR((size_t)RA * De, 0.0);
  for (int r = 0; r < RA; ++r) rms_b(&dao[(size_t)r * De], &ex[(size_t)r * De], De, &aV[m.e_lnf], true, eps, &edR[(size_t)r * De], &gaV[m.e_lnf]);
  std::vector<V> dKx(nf, V((size_t)R * kvw, 0.0)), dVx(nf, V((size_t)R * kvw, 0.0));
  for (int f = nf - 1; f >= 0; --f) {
    const auto& e = m.eb[f];
    ES& S = es[f];
    // MLP
    V db = qd(edR);
    mmdw(db, RA, De, S.hh, c.Ie, gaW, e.dn.off);
    V dhh = mmdx(db, RA, De, aW, e.dn.off, c.Ie);
    V dgu((size_t)RA * 2 * c.Ie);
    for (int r = 0; r < RA; ++r) for (int i = 0; i < c.Ie; ++i) {
      const double g = S.gu[(size_t)r * 2 * c.Ie + i], u = S.gu[(size_t)r * 2 * c.Ie + c.Ie + i], d = dhh[(size_t)r * c.Ie + i];
      dgu[(size_t)r * 2 * c.Ie + i] = d * u * dsilu(g);
      dgu[(size_t)r * 2 * c.Ie + c.Ie + i] = d * silu(g);
    }
    Qv(dgu);
    mmdw(dgu, RA, 2 * c.Ie, S.a2, De, gaW, e.gu.off);
    V da2 = mmdx(dgu, RA, 2 * c.Ie, aW, e.gu.off, De);
    for (int r = 0; r < RA; ++r) rms_b(&da2[(size_t)r * De], &S.xm[(size_t)r * De], De, &aV[e.ln2], true, eps, &edR[(size_t)r * De], &gaV[e.ln2]);
    // 어텐션
    db = qd(edR);
    mmdw(db, RA, De, S.ag, nq * hd, gaW, e.o.off);
    V dag = mmdx(db, RA, De, aW, e.o.off, nq * hd);
    V dT0((size_t)RA * eW0, 0.0), dO((size_t)RA * nq * hd), dq((size_t)RA * nq * hd, 0.0), dkk((size_t)RA * kvw, 0.0), dvv((size_t)RA * kvw, 0.0);
    for (int r = 0; r < RA; ++r) for (int h = 0; h < nq; ++h) for (int d = 0; d < hd; ++d) {
      const double gv = S.t0[(size_t)r * eW0 + h * 2 * hd + hd + d], s = sigm(gv), g2 = dag[((size_t)r * nq + h) * hd + d];
      dO[((size_t)r * nq + h) * hd + d] = g2 * s;
      dT0[(size_t)r * eW0 + h * 2 * hd + hd + d] = g2 * S.o[((size_t)r * nq + h) * hd + d] * s * (1 - s);
    }
    for (int b = 0; b < B; ++b) {
      const int pl = in.plen[b];
      V K((size_t)(L + c.Hc) * kvw), Vx((size_t)(L + c.Hc) * kvw), dK((size_t)(L + c.Hc) * kvw, 0.0), dVv((size_t)(L + c.Hc) * kvw, 0.0);
      for (int j = 0; j < L; ++j) for (int d = 0; d < kvw; ++d) { K[(size_t)j * kvw + d] = Kf[f][((size_t)b * L + j) * kvw + d]; Vx[(size_t)j * kvw + d] = Vf[f][((size_t)b * L + j) * kvw + d]; }
      for (int j = 0; j < c.Hc; ++j) for (int d = 0; d < kvw; ++d) { K[(size_t)(L + j) * kvw + d] = S.k[((size_t)b * c.Hc + j) * kvw + d]; Vx[(size_t)(L + j) * kvw + d] = S.v[((size_t)b * c.Hc + j) * kvw + d]; }
      AttIO a{c.Hc, L + c.Hc, nq, nkv, hd, [pl, L](int, int j) { return j < pl || j >= L; }};
      att_b(a, &S.q[(size_t)b * c.Hc * nq * hd], K.data(), Vx.data(), S.Pb[b], &dO[(size_t)b * c.Hc * nq * hd], &dq[(size_t)b * c.Hc * nq * hd], dK.data(), dVv.data());
      for (int j = 0; j < c.Hc; ++j) for (int d = 0; d < kvw; ++d) { dkk[((size_t)b * c.Hc + j) * kvw + d] = dK[(size_t)(L + j) * kvw + d]; dvv[((size_t)b * c.Hc + j) * kvw + d] = dVv[(size_t)(L + j) * kvw + d]; }
      if (!c.ki) for (int j = 0; j < L; ++j) for (int d = 0; d < kvw; ++d) { dKx[f][((size_t)b * L + j) * kvw + d] = dK[(size_t)j * kvw + d]; dVx[f][((size_t)b * L + j) * kvw + d] = dVv[(size_t)j * kvw + d]; }
    }
    for (int r = 0; r < RA; ++r) {
      const int b = r / c.Hc, t = r % c.Hc, pos = in.plen[b] + t;
      for (int h = 0; h < nq; ++h) {
        rope(&dq[((size_t)r * nq + h) * hd], Q.rot, Q.theta, pos, true);
        rms_b(&dq[((size_t)r * nq + h) * hd], &S.t0[(size_t)r * eW0 + h * 2 * hd], hd, &aV[e.qn], true, eps, &dT0[(size_t)r * eW0 + h * 2 * hd], &gaV[e.qn]);
      }
      for (int h = 0; h < nkv; ++h) {
        rope(&dkk[((size_t)r * nkv + h) * hd], Q.rot, Q.theta, pos, true);
        rms_b(&dkk[((size_t)r * nkv + h) * hd], &S.t0[(size_t)r * eW0 + qgw + h * hd], hd, &aV[e.kn], true, eps, &dT0[(size_t)r * eW0 + qgw + h * hd], &gaV[e.kn]);
        for (int d = 0; d < hd; ++d) dT0[(size_t)r * eW0 + qgw + kvw + h * hd + d] = dvv[((size_t)r * nkv + h) * hd + d];
      }
    }
    Qv(dT0);
    mmdw(dT0, RA, eW0, S.a1, De, gaW, e.qkv.off);
    V da1 = mmdx(dT0, RA, eW0, aW, e.qkv.off, De);
    for (int r = 0; r < RA; ++r) rms_b(&da1[(size_t)r * De], &S.x[(size_t)r * De], De, &aV[e.ln1], true, eps, &edR[(size_t)r * De], &gaV[e.ln1]);
  }
  mmdw(qd(edR), RA, De, in.ain, c.kpad(), gaW, m.e_in.off);
  // 글 → 끝 RMSN
  Qv(dlg);
  V dhnT = mmdx(dlg, Mt, Vv, qW, ql.emb.off, H);
  mmdw(dlg, Mt, Vv, hnT, H, gqW, ql.emb.off);
  V dhn((size_t)R * H, 0.0);
  for (int t = 0; t < Mt; ++t) for (int k = 0; k < H; ++k) dhn[(size_t)in.tgt_row[t] * H + k] = dhnT[(size_t)t * H + k];
  V dR((size_t)R * H, 0.0);
  for (int r = 0; r < R; ++r) rms_b(&dhn[(size_t)r * H], &XL[(size_t)r * H], H, &qV[ql.lnf], true, eps, &dR[(size_t)r * H], &gqV[ql.lnf]);
  // 몸통 층
  fi = nf - 1;
  for (int l = Q.layers - 1; l >= 0; --l) {
    const auto& Ly = ql.l[l];
    LS& S = ls[l];
    V db = qd(dR);
    mmdw(db, R, H, S.hh, Q.I, gqW, Ly.wdn.off);
    V dhh = mmdx(db, R, H, qW, Ly.wdn.off, Q.I);
    V dgu((size_t)R * 2 * Q.I);
    for (int r = 0; r < R; ++r) for (int i = 0; i < Q.I; ++i) {
      const double g = S.gu[(size_t)r * 2 * Q.I + i], u = S.gu[(size_t)r * 2 * Q.I + Q.I + i], d = dhh[(size_t)r * Q.I + i];
      dgu[(size_t)r * 2 * Q.I + i] = d * u * dsilu(g);
      dgu[(size_t)r * 2 * Q.I + Q.I + i] = d * silu(g);
    }
    Qv(dgu);
    mmdw(dgu, R, 2 * Q.I, S.a2, H, gqW, Ly.wgu.off);
    V da2 = mmdx(dgu, R, 2 * Q.I, qW, Ly.wgu.off, H);
    for (int r = 0; r < R; ++r) rms_b(&da2[(size_t)r * H], &S.xm[(size_t)r * H], H, &qV[Ly.ln2], true, eps, &dR[(size_t)r * H], &gqV[Ly.ln2]);
    db = qd(dR);
    V da1;
    if (Q.full[l]) {
      const int W0 = qgw + 2 * kvw;
      mmdw(db, R, H, S.ag, nq * hd, gqW, Ly.wo.off);
      V dag = mmdx(db, R, H, qW, Ly.wo.off, nq * hd);
      V dT0((size_t)R * W0, 0.0), dO((size_t)R * nq * hd), dq((size_t)R * nq * hd, 0.0), dK((size_t)R * kvw, 0.0), dVv((size_t)R * kvw, 0.0);
      for (int r = 0; r < R; ++r) for (int h = 0; h < nq; ++h) for (int d = 0; d < hd; ++d) {
        const double s = sigm(S.t0[(size_t)r * W0 + h * 2 * hd + hd + d]), g2 = dag[((size_t)r * nq + h) * hd + d];
        dO[((size_t)r * nq + h) * hd + d] = g2 * s;
        dT0[(size_t)r * W0 + h * 2 * hd + hd + d] = g2 * S.o[((size_t)r * nq + h) * hd + d] * s * (1 - s);
      }
      AttIO a{L, L, nq, nkv, hd, [](int t, int j) { return j <= t; }};
      for (int b = 0; b < B; ++b)
        att_b(a, &S.q[(size_t)b * L * nq * hd], &S.k[(size_t)b * L * kvw], &S.v[(size_t)b * L * kvw], S.Pb[b], &dO[(size_t)b * L * nq * hd], &dq[(size_t)b * L * nq * hd],
              &dK[(size_t)b * L * kvw], &dVv[(size_t)b * L * kvw]);
      if (!c.ki) { addv(dK, dKx[fi]); addv(dVv, dVx[fi]); }
      for (int r = 0; r < R; ++r) {
        const int t = r % L;
        for (int h = 0; h < nq; ++h) {
          rope(&dq[((size_t)r * nq + h) * hd], Q.rot, Q.theta, t, true);
          rms_b(&dq[((size_t)r * nq + h) * hd], &S.t0[(size_t)r * W0 + h * 2 * hd], hd, &qV[Ly.qn], true, eps, &dT0[(size_t)r * W0 + h * 2 * hd], &gqV[Ly.qn]);
        }
        for (int h = 0; h < nkv; ++h) {
          rope(&dK[((size_t)r * nkv + h) * hd], Q.rot, Q.theta, t, true);
          rms_b(&dK[((size_t)r * nkv + h) * hd], &S.t0[(size_t)r * W0 + qgw + h * hd], hd, &qV[Ly.kn], true, eps, &dT0[(size_t)r * W0 + qgw + h * hd], &gqV[Ly.kn]);
          for (int d = 0; d < hd; ++d) dT0[(size_t)r * W0 + qgw + kvw + h * hd + d] = dVv[((size_t)r * nkv + h) * hd + d];
        }
      }
      Qv(dT0);
      mmdw(dT0, R, W0, S.a1, H, gqW, Ly.wqkv.off);
      da1 = mmdx(dT0, R, W0, qW, Ly.wqkv.off, H);
      --fi;
    } else {
      mmdw(db, R, H, S.ag, lh * dv, gqW, Ly.wout.off);
      V dag = mmdx(db, R, H, qW, Ly.wout.off, lh * dv);
      V dT0((size_t)R * la, 0.0), dov((size_t)R * lh * dv, 0.0), dt1((size_t)R * li, 0.0);
      // 게이트 RMSNorm
      for (int r = 0; r < R; ++r) for (int h = 0; h < lh; ++h) {
        const double* o = &S.ov[((size_t)r * lh + h) * dv];
        double s = 0, g = 0;
        for (int d = 0; d < dv; ++d) s += o[d] * o[d];
        const double rs = 1.0 / std::sqrt(s / dv + eps);
        for (int d = 0; d < dv; ++d) g += dag[((size_t)r * lh + h) * dv + d] * qV[Ly.gnw + d] * silu(S.t0[(size_t)r * la + li + h * dv + d]) * o[d];
        for (int d = 0; d < dv; ++d) {
          const double z = S.t0[(size_t)r * la + li + h * dv + d], dy = dag[((size_t)r * lh + h) * dv + d];
          dT0[(size_t)r * la + li + h * dv + d] = dy * qV[Ly.gnw + d] * o[d] * rs * dsilu(z);
          dov[((size_t)r * lh + h) * dv + d] = rs * dy * qV[Ly.gnw + d] * silu(z) - o[d] * rs * rs * rs * g / dv;
          gqV[Ly.gnw + d] += dy * o[d] * rs * silu(z);
        }
      }
      // 재귀
      V dqn((size_t)R * lh * dk, 0.0), dkn((size_t)R * lh * dk, 0.0), dbe((size_t)R * lh, 0.0), dgg((size_t)R * lh, 0.0);
      for (int b = 0; b < B; ++b) for (int h = 0; h < lh; ++h) {
        V qs, ks, vs2, gs, bs, dos((size_t)L * dv), dq((size_t)L * dk), dk2((size_t)L * dk), dv2((size_t)L * dv), dg2(L), db2(L);
        dn_gather(S, b, h, L, lh, dk, dv, li, qs, ks, vs2, gs, bs);
        for (int t = 0; t < L; ++t) for (int j = 0; j < dv; ++j) dos[(size_t)t * dv + j] = dov[(((size_t)b * L + t) * lh + h) * dv + j];
        if (MD == EMUL && !envf("VREF_DN_REC"))
          dn_chk_bwd(L, dk, dv, tk::dn_chunk(dk), true, qs.data(), ks.data(), vs2.data(), gs.data(), bs.data(), nullptr, dos.data(), nullptr, dq.data(), dk2.data(),
                     dv2.data(), dg2.data(), db2.data(), nullptr);
        else
          dn_rec_bwd(L, dk, dv, qs.data(), ks.data(), vs2.data(), gs.data(), bs.data(), nullptr, dos.data(), nullptr, dq.data(), dk2.data(), dv2.data(), dg2.data(),
                     db2.data(), nullptr);
        for (int t = 0; t < L; ++t) {
          const size_t r = (size_t)b * L + t;
          for (int i = 0; i < dk; ++i) { dqn[(r * lh + h) * dk + i] = dq[(size_t)t * dk + i]; dkn[(r * lh + h) * dk + i] = dk2[(size_t)t * dk + i]; }
          for (int j = 0; j < dv; ++j) dt1[r * li + 2 * lh * dk + h * dv + j] = dv2[(size_t)t * dv + j];
          dgg[r * lh + h] = dg2[t];
          dbe[r * lh + h] = db2[t];
        }
      }
      // 정규화·β·g
      for (int r = 0; r < R; ++r) for (int h = 0; h < lh; ++h) {
        const double* q = &S.t1[(size_t)r * li + h * dk];
        const double* k = &S.t1[(size_t)r * li + lh * dk + h * dk];
        const double* dq = &dqn[((size_t)r * lh + h) * dk];
        const double* dkk2 = &dkn[((size_t)r * lh + h) * dk];
        double sq = 0, sk = 0, qd2 = 0, kd = 0;
        for (int d = 0; d < dk; ++d) { sq += q[d] * q[d]; sk += k[d] * k[d]; qd2 += q[d] * dq[d]; kd += k[d] * dkk2[d]; }
        const double iq = 1.0 / std::sqrt(sq + 1e-6), ik = 1.0 / std::sqrt(sk + 1e-6), sc = 1.0 / std::sqrt((double)dk);
        for (int d = 0; d < dk; ++d) {
          dt1[(size_t)r * li + h * dk + d] = sc * (iq * dq[d] - q[d] * iq * iq * iq * qd2);
          dt1[(size_t)r * li + lh * dk + h * dk + d] = ik * dkk2[d] - k[d] * ik * ik * ik * kd;
        }
        const double be = S.be[(size_t)r * lh + h];
        dT0[(size_t)r * la + li + lh * dv + h] = dbe[(size_t)r * lh + h] * be * (1 - be);
        const double x = S.t0[(size_t)r * la + li + lh * dv + lh + h] + qV[Ly.dtb + h], ea = std::exp(qV[Ly.alog + h]);
        const double sp = x > 20 ? x : std::log1p(std::exp(x)), dsp = x > 20 ? 1.0 : sigm(x);
        const double dg = dgg[(size_t)r * lh + h], dx = dg * (-ea) * dsp;
        dT0[(size_t)r * la + li + lh * dv + lh + h] = dx;
        gqV[Ly.alog + h] += dg * (-ea * sp);
        gqV[Ly.dtb + h] += dx;
      }
      // 합성곱
      for (int r = 0; r < R; ++r) {
        const int t = r % L;
        for (int ch = 0; ch < li; ++ch) {
          const double dp = dt1[(size_t)r * li + ch] * dsilu(S.pre[(size_t)r * li + ch]);
          for (int kk = 0; kk < Q.conv; ++kk) {
            const int p = t - (Q.conv - 1) + kk;
            if (p < 0) continue;
            const size_t rp = (size_t)(r - t + p);
            dT0[rp * la + ch] += qV[Ly.convw + ch * Q.conv + kk] * dp;
            gqV[Ly.convw + ch * Q.conv + kk] += dp * S.t0[rp * la + ch];
          }
        }
      }
      Qv(dT0);
      mmdw(dT0, R, la, S.a1, H, gqW, Ly.win.off);
      da1 = mmdx(dT0, R, la, qW, Ly.win.off, H);
    }
    for (int r = 0; r < R; ++r) rms_b(&da1[(size_t)r * H], &S.x[(size_t)r * H], H, &qV[Ly.ln1], true, eps, &dR[(size_t)r * H], &gqV[Ly.ln1]);
  }
  // 열 흩기
  V dvOut((size_t)Rv * H, 0.0), dgOut[N_VG];
  for (int g = 0; g < N_VG; ++g) if (m.grp_on(g)) dgOut[g].assign((size_t)B * m.ntok(g) * H, 0.0);
  for (int r = 0; r < R; ++r) {
    const int code = in.src[r], kind = (code >> 28) & 15, idx = code & 0x0fffffff;
    for (int k = 0; k < H; ++k) {
      const double d = dR[(size_t)r * H + k];
      if (kind == SK_TXT) gqW[ql.emb.off + (size_t)idx * H + k] += d;
      else if (kind == SK_IMG) dvOut[(size_t)idx * H + k] = d;
      else if (kind >= SK_GRP) { dgOut[kind - SK_GRP][(size_t)idx * H + k] = d; gaV[m.g_type[kind - SK_GRP] + k] += d; }
    }
  }
  if (c.mem) {
    // 기억 요약 인코더 뒤
    const double* phim = &phi[(size_t)r0 * H];
    V dphim((size_t)Rm * H, 0.0), dz((size_t)Rl * H, 0.0);
    for (int q = 0; q < B * 2; ++q) {
      const size_t row = ((size_t)(q / 2) * nl + q % 2) * H;
      for (int k = 0; k < H; ++k) { gaV[m.m_ex + k] += dex[q] * gOut[VG_MEM][row + k]; dgOut[VG_MEM][row + k] += dex[q] * aV[m.m_ex + k]; }
      gaV[m.m_ex + H] += dex[q];
    }
    for (int r = 0; r < Rl; ++r) rms_b(&dgOut[VG_MEM][(size_t)r * H], &mz[(size_t)r * H], H, &aV[m.m_lnout], true, eps, &dz[(size_t)r * H], &gaV[m.m_lnout]);
    for (int kb = c.mem_blk - 1; kb >= 0; --kb) {
      const auto& e = m.mb[kb];
      MS& S = ms[kb];
      V db = qd(dz);
      mmdw(db, Rl, H, S.hh, MI, gaW, e.dn.off);
      V dhh = mmdx(db, Rl, H, aW, e.dn.off, MI);
      V dgu((size_t)Rl * 2 * MI);
      for (int r = 0; r < Rl; ++r) for (int i = 0; i < MI; ++i) {
        const double g = S.gu[(size_t)r * 2 * MI + i], u = S.gu[(size_t)r * 2 * MI + MI + i], d = dhh[(size_t)r * MI + i];
        dgu[(size_t)r * 2 * MI + i] = d * u * dsilu(g);
        dgu[(size_t)r * 2 * MI + MI + i] = d * silu(g);
      }
      Qv(dgu);
      mmdw(dgu, Rl, 2 * MI, S.am, H, gaW, e.gu.off);
      V da = mmdx(dgu, Rl, 2 * MI, aW, e.gu.off, H);
      for (int r = 0; r < Rl; ++r) rms_b(&da[(size_t)r * H], &S.zs[(size_t)r * H], H, &aV[e.lnm], true, eps, &dz[(size_t)r * H], &gaV[e.lnm]);
      // 자기 어텐션
      db = qd(dz);
      mmdw(db, Rl, H, S.Osb, H, gaW, e.so.off);
      V dOs = mmdx(db, Rl, H, aW, e.so.off, H);
      V dqkv((size_t)Rl * 3 * H, 0.0);
      for (int b = 0; b < B; ++b) {
        V q((size_t)nl * H), k((size_t)nl * H), v((size_t)nl * H), dq((size_t)nl * H, 0.0), dk2((size_t)nl * H, 0.0), dv2((size_t)nl * H, 0.0);
        for (int t = 0; t < nl; ++t) for (int d = 0; d < H; ++d) {
          q[(size_t)t * H + d] = S.QKV[((size_t)b * nl + t) * 3 * H + d];
          k[(size_t)t * H + d] = S.QKV[((size_t)b * nl + t) * 3 * H + H + d];
          v[(size_t)t * H + d] = S.QKV[((size_t)b * nl + t) * 3 * H + 2 * H + d];
        }
        AttIO a{nl, nl, nh, nh, mhd, [](int, int) { return true; }};
        att_b(a, q.data(), k.data(), v.data(), S.Ps[b], &dOs[(size_t)b * nl * H], dq.data(), dk2.data(), dv2.data());
        for (int t = 0; t < nl; ++t) for (int d = 0; d < H; ++d) {
          dqkv[((size_t)b * nl + t) * 3 * H + d] = dq[(size_t)t * H + d];
          dqkv[((size_t)b * nl + t) * 3 * H + H + d] = dk2[(size_t)t * H + d];
          dqkv[((size_t)b * nl + t) * 3 * H + 2 * H + d] = dv2[(size_t)t * H + d];
        }
      }
      Qv(dqkv);
      mmdw(dqkv, Rl, 3 * H, S.as, H, gaW, e.sqkv.off);
      da = mmdx(dqkv, Rl, 3 * H, aW, e.sqkv.off, H);
      for (int r = 0; r < Rl; ++r) rms_b(&da[(size_t)r * H], &S.zx[(size_t)r * H], H, &aV[e.lns], true, eps, &dz[(size_t)r * H], &gaV[e.lns]);
      // 교차 어텐션
      db = qd(dz);
      mmdw(db, Rl, H, S.Ob, H, gaW, e.wo.off);
      V dO = mmdx(db, Rl, H, aW, e.wo.off, H);
      V dQ((size_t)Rl * H, 0.0), dKV((size_t)Rm * 2 * H, 0.0);
      for (int b = 0; b < B; ++b) {
        const int n = in.mem_n[b];
        V Kc((size_t)(n + 1) * H), Vc((size_t)(n + 1) * H), dK((size_t)(n + 1) * H, 0.0), dVv((size_t)(n + 1) * H, 0.0);
        for (int j = 0; j < n; ++j) for (int d = 0; d < H; ++d) { Kc[(size_t)j * H + d] = S.KV[((size_t)b * nmax + j) * 2 * H + d]; Vc[(size_t)j * H + d] = S.KV[((size_t)b * nmax + j) * 2 * H + H + d]; }
        for (int d = 0; d < H; ++d) { Kc[(size_t)n * H + d] = aV[e.nullkv + d]; Vc[(size_t)n * H + d] = aV[e.nullkv + H + d]; }
        AttIO a{nl, n + 1, nh, nh, mhd, [](int, int) { return true; }};
        att_b(a, &S.Q[(size_t)b * nl * H], Kc.data(), Vc.data(), S.Pc[b], &dO[(size_t)b * nl * H], &dQ[(size_t)b * nl * H], dK.data(), dVv.data());
        for (int j = 0; j < n; ++j) for (int d = 0; d < H; ++d) { dKV[((size_t)b * nmax + j) * 2 * H + d] = dK[(size_t)j * H + d]; dKV[((size_t)b * nmax + j) * 2 * H + H + d] = dVv[(size_t)j * H + d]; }
        for (int d = 0; d < H; ++d) { gaV[e.nullkv + d] += dK[(size_t)n * H + d]; gaV[e.nullkv + H + d] += dVv[(size_t)n * H + d]; }
        if (kb == c.mem_blk - 1)
          for (int r = 0; r < 2; ++r) {
            const double* dr = &drl[((size_t)b * 2 + r) * (nmax + 1)];
            for (int j = 0; j <= n; ++j) {
              if (dr[j] == 0) continue;
              const double* kk = j == 0 ? &aV[e.nullkv] : &S.KV[((size_t)b * nmax + j - 1) * 2 * H];
              double* dkk = j == 0 ? &gaV[e.nullkv] : &dKV[((size_t)b * nmax + j - 1) * 2 * H];
              for (int q = 0; q < H; ++q) {
                dQ[((size_t)b * nl + r) * H + q] += rcs * dr[j] * kk[q];
                dkk[q] += rcs * dr[j] * S.Q[((size_t)b * nl + r) * H + q];
              }
            }
          }
      }
      Qv(dQ);
      mmdw(dQ, Rl, H, S.aq, H, gaW, e.wq.off);
      da = mmdx(dQ, Rl, H, aW, e.wq.off, H);
      for (int r = 0; r < Rl; ++r) rms_b(&da[(size_t)r * H], &S.zin[(size_t)r * H], H, &aV[e.lnq], true, eps, &dz[(size_t)r * H], &gaV[e.lnq]);
      Qv(dKV);
      mmdw(dKV, Rm, 2 * H, S.mk, H, gaW, e.wkv.off);
      V dmk = mmdx(dKV, Rm, 2 * H, aW, e.wkv.off, H);
      for (int r = 0; r < Rm; ++r) rms_b(&dmk[(size_t)r * H], phim + (size_t)r * H, H, &aV[e.lnk], true, eps, &dphim[(size_t)r * H], &gaV[e.lnk]);
    }
    V dI((size_t)B * H, 0.0);
    for (int r = 0; r < Rl; ++r) for (int k = 0; k < H; ++k) { gaV[m.m_lat + (r % nl) * H + k] += dz[(size_t)r * H + k]; dI[(size_t)(r / nl) * H + k] += dz[(size_t)r * H + k]; }
    Qv(dI);
    mmdw(dI, B, H, in.instr, c.instr_k, gaW, m.m_instr.off);
    // φ(정밀 + 기억 줄)
    V dphi = dgOut[VG_OBJ];
    dphi.insert(dphi.end(), dphim.begin(), dphim.end());
    for (int r = 0; r < rows_all; ++r) for (int k = 0; k < H; ++k) gaV[m.g_b2 + k] += dphi[(size_t)r * H + k];
    V dgb = qd(dphi);
    mmdw(dgb, rows_all, H, gh, c.obj_hid, gaW, m.g_w2.off);
    V dgh = mmdx(dgb, rows_all, H, aW, m.g_w2.off, c.obj_hid);
    for (size_t i = 0; i < dgh.size(); ++i) dgh[i] *= dgelu(gh1[i]);
    Qv(dgh);
    mmdw(dgh, rows_all, c.obj_hid, Xall, MEM_K, gaW, m.g_w1[VG_OBJ].off);
  }
  for (int g = 0; g < N_VG; ++g) {
    if (!m.grp_on(g) || g == VG_MEM || (c.mem && g == VG_OBJ)) continue;
    const int rows = B * kVGrp[g].n_tok, K = kVGrp[g].K;
    V dgb = qd(dgOut[g]);
    if (g == VG_OBJ) {
      for (int r = 0; r < rows; ++r) for (int k = 0; k < H; ++k) gaV[m.g_b2 + k] += dgOut[g][(size_t)r * H + k];
      mmdw(dgb, rows, H, gh, c.obj_hid, gaW, m.g_w2.off);
      V dgh = mmdx(dgb, rows, H, aW, m.g_w2.off, c.obj_hid);
      for (size_t i = 0; i < dgh.size(); ++i) dgh[i] *= dgelu(gh1[i]);
      Qv(dgh);
      mmdw(dgh, rows, c.obj_hid, in.grp[g], K, gaW, m.g_w1[g].off);
    } else mmdw(dgb, rows, H, in.grp[g], K, gaW, m.g_w1[g].off);
  }
  // 영상
  for (int r = 0; r < Rv; ++r) for (int k = 0; k < H; ++k) gaV[m.v_projb + k] += dvOut[(size_t)r * H + k];
  V dvb = qd(dvOut);
  mmdw(dvb, Rv, H, tok, D, gaW, m.v_proj.off);
  if (c.vis_train) {
    V dtok = mmdx(dvb, Rv, H, aW, m.v_proj.off, D);
    V vdR((size_t)Rv * D, 0.0);
    for (int r = 0; r < Rv; ++r) ln_b(&dtok[(size_t)r * D], &vxL[(size_t)r * D], D, &aV[m.v_lnfg], c.vEps, &vdR[(size_t)r * D], &gaV[m.v_lnfg], &gaV[m.v_lnfb]);
    for (int l = c.vL - 1; l >= 0; --l) {
      const auto& b = m.vb[l];
      VS& S = vs[l];
      for (int r = 0; r < Rv; ++r) for (int d = 0; d < D; ++d) gaV[b.fc2b + d] += vdR[(size_t)r * D + d];
      V db = qd(vdR);
      mmdw(db, Rv, D, S.ag, c.vMLP, gaW, b.fc2.off);
      V dh = mmdx(db, Rv, D, aW, b.fc2.off, c.vMLP);
      for (size_t i = 0; i < dh.size(); ++i) dh[i] *= dgelu(S.h1[i]);
      for (int r = 0; r < Rv; ++r) for (int k = 0; k < c.vMLP; ++k) gaV[b.fc1b + k] += dh[(size_t)r * c.vMLP + k];
      Qv(dh);
      mmdw(dh, Rv, c.vMLP, S.a2, D, gaW, b.fc1.off);
      V da = mmdx(dh, Rv, c.vMLP, aW, b.fc1.off, D);
      for (int r = 0; r < Rv; ++r) ln_b(&da[(size_t)r * D], &S.xm[(size_t)r * D], D, &aV[b.ln2g], c.vEps, &vdR[(size_t)r * D], &gaV[b.ln2g], &gaV[b.ln2b]);
      for (int r = 0; r < Rv; ++r) for (int d = 0; d < D; ++d) gaV[b.projb + d] += vdR[(size_t)r * D + d];
      db = qd(vdR);
      mmdw(db, Rv, D, S.ao, D, gaW, b.proj.off);
      V dO = mmdx(db, Rv, D, aW, b.proj.off, D);
      V dqkv((size_t)Rv * 3 * D, 0.0);
      for (int im = 0; im < Rv / T; ++im) {
        V q((size_t)T * D), k((size_t)T * D), v((size_t)T * D), dq((size_t)T * D, 0.0), dk2((size_t)T * D, 0.0), dv2((size_t)T * D, 0.0);
        for (int t = 0; t < T; ++t) for (int d = 0; d < D; ++d) {
          q[(size_t)t * D + d] = S.qkv[((size_t)im * T + t) * 3 * D + d];
          k[(size_t)t * D + d] = S.qkv[((size_t)im * T + t) * 3 * D + D + d];
          v[(size_t)t * D + d] = S.qkv[((size_t)im * T + t) * 3 * D + 2 * D + d];
        }
        att_b(va, q.data(), k.data(), v.data(), S.Pi[im], &dO[(size_t)im * T * D], dq.data(), dk2.data(), dv2.data());
        for (int t = 0; t < T; ++t) for (int d = 0; d < D; ++d) {
          dqkv[((size_t)im * T + t) * 3 * D + d] = dq[(size_t)t * D + d];
          dqkv[((size_t)im * T + t) * 3 * D + D + d] = dk2[(size_t)t * D + d];
          dqkv[((size_t)im * T + t) * 3 * D + 2 * D + d] = dv2[(size_t)t * D + d];
        }
      }
      for (int r = 0; r < Rv; ++r) for (int k = 0; k < 3 * D; ++k) gaV[b.qkvb + k] += dqkv[(size_t)r * 3 * D + k];
      Qv(dqkv);
      mmdw(dqkv, Rv, 3 * D, S.a1, D, gaW, b.qkv.off);
      da = mmdx(dqkv, Rv, 3 * D, aW, b.qkv.off, D);
      for (int r = 0; r < Rv; ++r) ln_b(&da[(size_t)r * D], &S.x[(size_t)r * D], D, &aV[b.ln1g], c.vEps, &vdR[(size_t)r * D], &gaV[b.ln1g], &gaV[b.ln1b]);
    }
    for (int r = 0; r < Rv; ++r) for (int d = 0; d < D; ++d) { gaV[m.v_patchb + d] += vdR[(size_t)r * D + d]; gaV[m.v_pos + (r % T) * D + d] += vdR[(size_t)r * D + d]; }
    mmdw(qd(vdR), Rv, D, in.patches, c.vK, gaW, m.v_patch.off);
  }
  // 행렬 기울기는 GPU 가 bf16 으로 저장
  Qv(gqW);
  Qv(gaW);
}

}  // namespace vref
