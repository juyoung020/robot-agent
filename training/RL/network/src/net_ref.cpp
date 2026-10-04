// CPU 참조판: FP64(정답) 와 EMUL(GPU 정밀도 처방 흉내 → 바닥). 행 단위 OpenMP. 학습 경로에는 들어가지 않는다.
#include "net_ref.h"

#include "fp8_ref.h"

#include <cmath>
#include <algorithm>
#include <cstring>

namespace netref {
using namespace net;

static inline double rnd(Mode md, double x, bool bf) {
  if (md == FP64) return x;
  const float f = (float)x;
  return bf ? (double)rbf(f) : (double)f;
}
static inline double elu_d(Mode md, double z) {
  if (md == FP64) return z > 0.0 ? z : std::expm1(z);
  const float f = (float)z;
  return (double)(f > 0.f ? f : std::expm1(f));
}
static inline double elu_g(double y) { return y > 0.0 ? 1.0 : y + 1.0; }

// 텐서 코어 FP32 누산 흉내(EMUL): k 16 개(mma 한 번)의 곱은 정확히 더하고, 누산기와 합친 결과를 0 쪽으로 잘라 FP32 로(가정 — 잰 비율로 확인, README).
static int g_kblk = 16, g_kchunk = 1024;
void set_tc_model(int kblock, int kchunk) { g_kblk = kblock; g_kchunk = kchunk; }
static int g_fp8 = 0;
void set_fp8(int mask) { g_fp8 = mask; }
static bool f8on(Mode md, const LayerDesc& L, int role) { return md == EMUL && L.fp8 && (g_fp8 >> role & 1); }
// 텐서(행 R × 열 C, 줄 간격 ld) 를 bf16 값 그대로 보고 amax → 배율 inv, 양자화한 값(배율 공간, 정수배 아님)을 q 에
static float f8quant(const double* X, long long R, int C, long long ld, int fmt, std::vector<double>& q) {
  float am = 0.f;
  for (long long r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) am = std::fmax(am, std::fabs((float)X[r * ld + c]));
  const float inv = f8ref::pow2_inv(am, fmt);
  q.assign((size_t)R * ld, 0.0);
#pragma omp parallel for schedule(static)
  for (long long r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) q[(size_t)(r * ld + c)] = f8ref::dq(f8ref::q((float)X[r * ld + c] * inv, fmt), fmt);
  return inv;
}
static inline float rz(double x) {
  float f = (float)x;
  if (std::fabs((double)f) > std::fabs(x)) f = std::nextafter(f, 0.f);
  return f;
}

void lin_fwd(Mode md, const double* X, int M, const LayerDesc& L, const double* W, double* out) {
  if (f8on(md, L, 0)) {   // FP8: E4M3 X × E4M3 W, k 32 묶음 RZ 누산, 배율 곱(float) → 끝단
    std::vector<double> xq, wq;
    const float ix = f8quant(X, M, L.K, L.K, 0, xq), iw = f8quant(W, L.N, L.K, L.K, 0, wq);
    const float deq = (1.f / ix) * (1.f / iw);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
      double* o = out + (size_t)m * L.ldo;
      for (int n = 0; n < L.N; ++n) {
        float s = 0.f;
        for (int k0 = 0; k0 < L.K; k0 += 32) {
          double b = s;
          for (int k = k0; k < L.K && k < k0 + 32; ++k) b += xq[(size_t)m * L.K + k] * wq[(size_t)n * L.K + k];
          s = rz(b);
        }
        const double z = (double)(s * deq);
        o[n] = L.act == ACT_ELU ? rnd(md, elu_d(md, z), true) : rnd(md, z, false);
      }
      for (int n = L.N; n < L.ldo; ++n) o[n] = n == L.N ? 1.0 : 0.0;
    }
    return;
  }
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; ++m) {
    const double* x = X + (size_t)m * L.K;
    double* o = out + (size_t)m * L.ldo;
    for (int n = 0; n < L.N; ++n) {
      const double* w = W + (size_t)n * L.K;
      double z;
      if (md == EMUL) {
        float s = 0.f;
        for (int k0 = 0; k0 < L.K; k0 += g_kblk) {
          double b = s;
          for (int k = k0; k < L.K && k < k0 + g_kblk; ++k) b += (double)(float)x[k] * (double)(float)w[k];
          s = rz(b);
        }
        z = s;
      } else {
        double s = 0.0;
        for (int k = 0; k < L.K; ++k) s += x[k] * w[k];
        z = s;
      }
      o[n] = L.act == ACT_ELU ? rnd(md, elu_d(md, z), true) : rnd(md, z, false);
    }
    for (int n = L.N; n < L.ldo; ++n) o[n] = n == L.N ? 1.0 : 0.0;
  }
}

void lin_dx(Mode md, const double* dZ, int M, const LayerDesc& L, const double* W, int Np, const double* Y, double* out, bool round_bf16) {
  if (f8on(md, L, 1)) {   // FP8 dgrad: E5M2 dZ × E4M3 W[:, 0:Np]
    std::vector<double> dq, wq;
    const float id = f8quant(dZ, M, L.N, L.N, 1, dq), iw = f8quant(W, L.N, Np, L.K, 0, wq);
    const float deq = (1.f / id) * (1.f / iw);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m)
      for (int j = 0; j < Np; ++j) {
        float s = 0.f;
        for (int n0 = 0; n0 < L.N; n0 += 32) {
          double b = s;
          for (int n = n0; n < L.N && n < n0 + 32; ++n) b += dq[(size_t)m * L.N + n] * wq[(size_t)n * L.K + j];
          s = rz(b);
        }
        double a = (double)(s * deq);
        if (Y) a = (double)((float)a * (float)elu_g(Y[(size_t)m * L.K + j]));
        out[(size_t)m * Np + j] = rnd(md, a, round_bf16);
      }
    return;
  }
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; ++m) {
    const double* d = dZ + (size_t)m * L.N;
    for (int j = 0; j < Np; ++j) {
      double a;
      if (md == EMUL) {
        float s = 0.f;
        for (int n0 = 0; n0 < L.N; n0 += g_kblk) {
          double b = s;
          for (int n = n0; n < L.N && n < n0 + g_kblk; ++n) b += (double)(float)d[n] * (double)(float)W[(size_t)n * L.K + j];
          s = rz(b);
        }
        a = s;
        if (Y) a = (double)((float)a * (float)elu_g(Y[(size_t)m * L.K + j]));
      } else {
        double s = 0.0;
        for (int n = 0; n < L.N; ++n) s += d[n] * W[(size_t)n * L.K + j];
        a = s;
        if (Y) a *= elu_g(Y[(size_t)m * L.K + j]);
      }
      out[(size_t)m * Np + j] = rnd(md, a, round_bf16);
    }
  }
}

void lin_dw(Mode md, const double* dZ, const double* X, int M, const LayerDesc& L, double* G) {
  if (f8on(md, L, 2)) {   // FP8 wgrad: E5M2 dZᵀ × E4M3 X, 조각(kchunk 행)마다 k 32 묶음 RZ, 배율 곱(float), 조각 합 float 차례
    std::vector<double> dq, xq;
    const float id = f8quant(dZ, M, L.N, L.N, 1, dq), ix = f8quant(X, M, L.K, L.K, 0, xq);
    const float deq = (1.f / id) * (1.f / ix);
#pragma omp parallel for schedule(dynamic, 4)
    for (int n = 0; n < L.N; ++n) {
      std::vector<float> accf(L.K), tot(L.K, 0.f);
      std::vector<double> blk(L.K);
      for (int c0 = 0; c0 < M; c0 += g_kchunk) {
        std::fill(accf.begin(), accf.end(), 0.f);
        for (int m0 = c0; m0 < M && m0 < c0 + g_kchunk; m0 += 32) {
          for (int k = 0; k < L.K; ++k) blk[k] = accf[k];
          for (int m = m0; m < M && m < m0 + 32 && m < c0 + g_kchunk; ++m) {
            const double d = dq[(size_t)m * L.N + n];
            if (d == 0.0) continue;
            for (int k = 0; k < L.K; ++k) blk[k] += d * xq[(size_t)m * L.K + k];
          }
          for (int k = 0; k < L.K; ++k) accf[k] = rz(blk[k]);
        }
        for (int k = 0; k < L.K; ++k) tot[k] = tot[k] + accf[k] * deq;
      }
      for (int k = 0; k < L.K; ++k) G[(size_t)n * L.K + k] = tot[k];
    }
    return;
  }
#pragma omp parallel for schedule(dynamic, 4)
  for (int n = 0; n < L.N; ++n) {
    std::vector<double> acc(L.K, 0.0), blk(L.K);
    std::vector<float> accf(L.K), tot(L.K, 0.f);
    if (md == EMUL) {   // 조각(kchunk 행)마다 k 16 묶음 RZ 누산, 조각 합은 차례로 float(GPU dw_reduce 와 같은 순서)
      for (int c0 = 0; c0 < M; c0 += g_kchunk) {
        std::fill(accf.begin(), accf.end(), 0.f);
        for (int m0 = c0; m0 < M && m0 < c0 + g_kchunk; m0 += g_kblk) {
          for (int k = 0; k < L.K; ++k) blk[k] = accf[k];
          for (int m = m0; m < M && m < m0 + g_kblk && m < c0 + g_kchunk; ++m) {
            const double d = (double)(float)dZ[(size_t)m * L.N + n];
            if (d == 0.0) continue;
            const double* x = X + (size_t)m * L.K;
            for (int k = 0; k < L.K; ++k) blk[k] += d * (double)(float)x[k];
          }
          for (int k = 0; k < L.K; ++k) accf[k] = rz(blk[k]);
        }
        for (int k = 0; k < L.K; ++k) tot[k] = tot[k] + accf[k];
      }
      for (int k = 0; k < L.K; ++k) G[(size_t)n * L.K + k] = tot[k];
    } else {
      for (int m = 0; m < M; ++m) {
        const double d = dZ[(size_t)m * L.N + n];
        if (d == 0.0) continue;
        const double* x = X + (size_t)m * L.K;
        for (int k = 0; k < L.K; ++k) acc[k] += d * x[k];
      }
      for (int k = 0; k < L.K; ++k) G[(size_t)n * L.K + k] = acc[k];
    }
  }
}

void pool_fwd(Mode md, const double* s2o, const uint32_t* mask, int M, double* x0, int* amax) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < S_H; ++c) {
      double sum = 0.0, mx = 0.0;
      float sumf = 0.f;
      int am = 255, n = 0;
      for (int s = 0; s < KSLOT; ++s) {
        if (!((mask[r] >> s) & 1u)) continue;
        const double y = s2o[((size_t)r * KSLOT + s) * S_H + c];
        sum += y;
        sumf = sumf + (float)y;
        if (am == 255 || y > mx) { mx = y; am = s; }
        ++n;
      }
      double mean = 0.0;
      if (n) mean = md == EMUL ? (double)(sumf / (float)n) : sum / n;
      x0[(size_t)r * X0_W + c] = rnd(md, mean, true);
      x0[(size_t)r * X0_W + S_H + c] = rnd(md, n ? mx : 0.0, true);
      amax[(size_t)r * S_H + c] = am;
    }
}

void pool_bwd(Mode md, const double* dpool, const double* s2o, const uint32_t* mask, const int* amax, int M, double* dzs2) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < M; ++r)
    for (int s = 0; s < KSLOT; ++s)
      for (int c = 0; c < S_H; ++c) {
        const size_t q = ((size_t)r * KSLOT + s) * S_H + c;
        double d = 0.0;
        if ((mask[r] >> s) & 1u) {
          const int n = __builtin_popcount(mask[r]);
          if (md == EMUL) {
            float f = (float)dpool[(size_t)r * POOL_W + c] / (float)n;
            if (amax[(size_t)r * S_H + c] == s) f = f + (float)dpool[(size_t)r * POOL_W + S_H + c];
            f = f * (float)elu_g(s2o[q]);
            d = f;
          } else {
            d = dpool[(size_t)r * POOL_W + c] / n;
            if (amax[(size_t)r * S_H + c] == s) d += dpool[(size_t)r * POOL_W + S_H + c];
            d *= elu_g(s2o[q]);
          }
        }
        dzs2[q] = rnd(md, d, true);
      }
}

void loss(Mode md, const double* mean, const double* val, const double* logstd, const Batch& b, const Hyper& h, double* dzA, double* dzC, double* dls,
          double* st) {
  const int M = b.M;
  double q[13] = {0};
  float qf[N_ACT] = {0.f};   // EMUL: log σ 기울기 합은 GPU 처럼 FP32 로 누산(행 순서대로 — GPU 는 블록 나무 합이라 오차가 이보다 작거나 같은 크기)
  for (int r = 0; r < M; ++r) {
    if (md == EMUL) {   // GPU ppo_loss_k 와 같은 float 식
      const float scale = 1.f / (float)M;
      const float A = (b.adv[r] - b.adv_mean) / (b.adv_std + 1e-8f);
      float z[N_ACT], inv[N_ACT], logp = 0.f;
      for (int k = 0; k < N_ACT; ++k) {
        const float ls = (float)logstd[k];
        inv[k] = std::exp(-ls);
        z[k] = (b.act[(size_t)r * N_ACT + k] - (float)mean[(size_t)r * N_ACT + k]) * inv[k];
        if ((h.act_mask >> k) & 1u) logp = logp + (-0.5f * z[k] * z[k] - ls - 0.5f * kLog2Pi);
      }
      const float lr_ = logp - b.oldlogp[r], ratio = std::exp(lr_);
      const float s1 = ratio * A, rc = std::fmin(std::fmax(ratio, 1.f - h.clip), 1.f + h.clip), s2 = rc * A;
      const float g = (s1 <= s2) ? -ratio * A : 0.f;
      for (int k = 0; k < N_ACT; ++k) {
        const bool on = (h.act_mask >> k) & 1u;
        const float mu = (float)mean[(size_t)r * N_ACT + k], ex = std::fabs(mu) - 1.f;
        const float db = (on && h.bound_coef != 0.f && ex > 0.f) ? h.bound_coef * 2.f * ex * (mu > 0.f ? 1.f : -1.f) : 0.f;
        if (db != 0.f) q[12] += (double)(h.bound_coef * ex * ex);
        const float d0 = scale * g * z[k] * inv[k];
        dzA[(size_t)r * N_ACT + k] = on ? rbf(db != 0.f ? d0 + scale * db : d0) : 0.0;
        qf[k] = qf[k] + (on ? scale * g * (z[k] * z[k] - 1.f) : 0.f);
      }
      const float v = (float)val[(size_t)r * 8], ov = b.oldv[r], R = b.ret[r];
      float vl, gv;
      if (h.vclip > 0.f) {
        const float dv = v - ov, vc = ov + std::fmin(std::fmax(dv, -h.vclip), h.vclip);
        const float l1 = (v - R) * (v - R), l2 = (vc - R) * (vc - R);
        vl = 0.5f * std::fmax(l1, l2);
        gv = (l1 >= l2) ? (v - R) : ((dv > -h.vclip && dv < h.vclip) ? (vc - R) : 0.f);
      } else { vl = 0.5f * (v - R) * (v - R); gv = v - R; }
      dzC[(size_t)r * 8] = rbf(scale * h.vf_coef * gv);
      for (int k = 1; k < 8; ++k) dzC[(size_t)r * 8 + k] = 0.0;
      q[8] += -std::fmin(s1, s2);
      q[9] += vl;
      q[10] += (ratio - 1.f) - lr_;
      q[11] += std::fabs(ratio - 1.f) > h.clip ? 1.0 : 0.0;
    } else {
      const double scale = 1.0 / M;
      const double A = ((double)b.adv[r] - b.adv_mean) / ((double)b.adv_std + 1e-8);
      double z[N_ACT], inv[N_ACT], logp = 0.0;
      for (int k = 0; k < N_ACT; ++k) {
        inv[k] = std::exp(-logstd[k]);
        z[k] = ((double)b.act[(size_t)r * N_ACT + k] - mean[(size_t)r * N_ACT + k]) * inv[k];
        if ((h.act_mask >> k) & 1u) logp += -0.5 * z[k] * z[k] - logstd[k] - 0.5 * 1.8378770664093453;
      }
      const double lr_ = logp - b.oldlogp[r], ratio = std::exp(lr_);
      const double s1 = ratio * A, rc = std::fmin(std::fmax(ratio, 1.0 - h.clip), 1.0 + h.clip), s2 = rc * A;
      const double g = (s1 <= s2) ? -ratio * A : 0.0;
      for (int k = 0; k < N_ACT; ++k) {
        const bool on = (h.act_mask >> k) & 1u;
        const double mu = mean[(size_t)r * N_ACT + k], ex = std::fabs(mu) - 1.0;
        const double db = (on && h.bound_coef != 0.f && ex > 0.0) ? h.bound_coef * 2.0 * ex * (mu > 0.0 ? 1.0 : -1.0) : 0.0;
        if (db != 0.0) q[12] += h.bound_coef * ex * ex;
        dzA[(size_t)r * N_ACT + k] = on ? scale * g * z[k] * inv[k] + scale * db : 0.0;
        q[k] += on ? scale * g * (z[k] * z[k] - 1.0) : 0.0;
      }
      const double v = val[(size_t)r * 8], ov = b.oldv[r], R = b.ret[r];
      double vl, gv;
      if (h.vclip > 0.f) {
        const double dv = v - ov, vc = ov + std::fmin(std::fmax(dv, -(double)h.vclip), (double)h.vclip);
        const double l1 = (v - R) * (v - R), l2 = (vc - R) * (vc - R);
        vl = 0.5 * std::fmax(l1, l2);
        gv = (l1 >= l2) ? (v - R) : ((dv > -h.vclip && dv < h.vclip) ? (vc - R) : 0.0);
      } else { vl = 0.5 * (v - R) * (v - R); gv = v - R; }
      dzC[(size_t)r * 8] = scale * h.vf_coef * gv;
      for (int k = 1; k < 8; ++k) dzC[(size_t)r * 8 + k] = 0.0;
      q[8] += -std::fmin(s1, s2);
      q[9] += vl;
      q[10] += (ratio - 1.0) - lr_;
      q[11] += std::fabs(ratio - 1.0) > h.clip ? 1.0 : 0.0;
    }
  }
  if (md == EMUL) for (int k = 0; k < N_ACT; ++k) q[k] = (double)(qf[k] - h.ent_coef) + h.ent_coef;   // 아래에서 빼는 ent_coef 도 FP32 로
  for (int k = 0; k < N_ACT; ++k) dls[k] = ((h.act_mask >> k) & 1u) ? q[k] - h.ent_coef : 0.0;
  st[0] = q[8] / M;
  st[1] = q[9] / M;
  st[2] = q[10] / M;
  st[3] = q[11] / M;
  st[4] = q[12] / M;   // 자르기 밖 평균 벌
}

static void run_impl(Mode md, const std::vector<double>& P, const Batch& b, const Hyper& h, Trace& tr, bool backward) {
  const ParamLayout lay = param_layout();
  const int M = b.M, MS = M * KSLOT;
  std::vector<double> W(P.size());
  for (size_t i = 0; i < P.size(); ++i) W[i] = md == EMUL ? (double)rbf((float)P[i]) : P[i];
  auto Wl = [&](int l) { return W.data() + lay.off[l]; };
  const double* logstd = P.data() + lay.logstd;

  tr.sin.assign(b.sin.size(), 0.0);
  for (size_t i = 0; i < b.sin.size(); ++i) tr.sin[i] = bf2f(b.sin[i]);
  tr.x0.assign(b.x0.size(), 0.0);
  for (size_t i = 0; i < b.x0.size(); ++i) tr.x0[i] = bf2f(b.x0[i]);
  tr.s1o.assign((size_t)MS * kLayers[L_S1].ldo, 0.0);
  tr.s2o.assign((size_t)MS * kLayers[L_S2].ldo, 0.0);
  lin_fwd(md, tr.sin.data(), MS, kLayers[L_S1], Wl(L_S1), tr.s1o.data());
  lin_fwd(md, tr.s1o.data(), MS, kLayers[L_S2], Wl(L_S2), tr.s2o.data());
  tr.amax.assign((size_t)M * S_H, 255);
  pool_fwd(md, tr.s2o.data(), b.mask.data(), M, tr.x0.data(), tr.amax.data());
  for (int head : {L_A1, L_C1}) {
    const double* X = tr.x0.data();
    for (int l = head; l < head + 4; ++l) {
      tr.h[l].assign((size_t)M * kLayers[l].ldo, 0.0);
      lin_fwd(md, X, M, kLayers[l], Wl(l), tr.h[l].data());
      X = tr.h[l].data();
    }
  }
  tr.mean = tr.h[L_A4];
  tr.val = tr.h[L_C4];
  for (int l = 0; l < N_LAYER; ++l) tr.dz[l].assign((size_t)(l <= L_S2 ? MS : M) * kLayers[l].N, 0.0);
  double dls[N_ACT], st[5];
  loss(md, tr.mean.data(), tr.val.data(), logstd, b, h, tr.dz[L_A4].data(), tr.dz[L_C4].data(), dls, st);
  tr.pg = st[0]; tr.vl = st[1]; tr.kl = st[2]; tr.clipfrac = st[3];
  tr.ent = 0.0;
  for (int k = 0; k < N_ACT; ++k) if ((h.act_mask >> k) & 1u) tr.ent += logstd[k] + 0.5 + 0.5 * 1.8378770664093453;
  tr.loss = tr.pg + h.vf_coef * tr.vl - h.ent_coef * tr.ent + st[4];
  if (!backward) return;
  tr.grad.assign(lay.total, 0.0);
  std::vector<double> dp[2];
  for (int ci = 0; ci < 2; ++ci) {
    const int head = ci ? L_C1 : L_A1;
    for (int l = head + 3; l > head; --l) {
      lin_dw(md, tr.dz[l].data(), tr.h[l - 1].data(), M, kLayers[l], tr.grad.data() + lay.off[l]);
      lin_dx(md, tr.dz[l].data(), M, kLayers[l], Wl(l), kLayers[l - 1].N, tr.h[l - 1].data(), tr.dz[l - 1].data(), true);
    }
    lin_dw(md, tr.dz[head].data(), tr.x0.data(), M, kLayers[head], tr.grad.data() + lay.off[head]);
    dp[ci].assign((size_t)M * POOL_W, 0.0);
    lin_dx(md, tr.dz[head].data(), M, kLayers[head], Wl(head), POOL_W, nullptr, dp[ci].data(), false);
  }
  tr.dpool.assign((size_t)M * POOL_W, 0.0);
  for (size_t i = 0; i < tr.dpool.size(); ++i) tr.dpool[i] = md == EMUL ? (double)((float)dp[0][i] + (float)dp[1][i]) : dp[0][i] + dp[1][i];
  pool_bwd(md, tr.dpool.data(), tr.s2o.data(), b.mask.data(), tr.amax.data(), M, tr.dz[L_S2].data());
  lin_dw(md, tr.dz[L_S2].data(), tr.s1o.data(), MS, kLayers[L_S2], tr.grad.data() + lay.off[L_S2]);
  lin_dx(md, tr.dz[L_S2].data(), MS, kLayers[L_S2], Wl(L_S2), kLayers[L_S1].N, tr.s1o.data(), tr.dz[L_S1].data(), true);
  lin_dw(md, tr.dz[L_S1].data(), tr.sin.data(), MS, kLayers[L_S1], tr.grad.data() + lay.off[L_S1]);
  for (int k = 0; k < N_ACT; ++k) tr.grad[lay.logstd + k] = dls[k];
}

void run(Mode md, const std::vector<float>& P, const Batch& b, const Hyper& h, Trace& tr, bool backward) {
  std::vector<double> Pd(P.begin(), P.end());
  run_impl(md, Pd, b, h, tr, backward);
}

void run_d(Mode md, const std::vector<double>& P, const Batch& b, const Hyper& h, Trace& tr, bool backward) { run_impl(md, P, b, h, tr, backward); }

double loss_only(const std::vector<double>& P, const Batch& b, const Hyper& h) {
  Trace tr;
  run_impl(FP64, P, b, h, tr, false);
  return tr.loss;
}

double adam(Mode md, std::vector<double>& P, const std::vector<double>& G, std::vector<double>& m, std::vector<double>& v, long long t, float lr,
            const net::AdamHyper& ah) {
  const size_t n = P.size();
  double ss = 0.0;
  for (size_t i = 0; i < n; ++i) ss += G[i] * G[i];
  const double gn = std::sqrt(ss);
  if (md == EMUL) {
    const float gnf = (float)gn;
    const float c = ah.max_norm > 0.f ? std::fmin(1.f, ah.max_norm / (gnf + 1e-6f)) : 1.f;
    const float bc1 = 1.f - std::pow(ah.b1, (float)t), bc2 = 1.f - std::pow(ah.b2, (float)t);
    for (size_t i = 0; i < n; ++i) {
      const float g = (float)G[i] * c;
      const float mi = ah.b1 * (float)m[i] + (1.f - ah.b1) * g;
      const float vi = ah.b2 * (float)v[i] + (1.f - ah.b2) * (g * g);
      m[i] = mi;
      v[i] = vi;
      P[i] = (double)((float)P[i] - lr * (mi / bc1) / (std::sqrt(vi / bc2) + ah.eps));
    }
  } else {
    const double c = ah.max_norm > 0.f ? std::fmin(1.0, ah.max_norm / (gn + 1e-6)) : 1.0;
    const double bc1 = 1.0 - std::pow((double)ah.b1, (double)t), bc2 = 1.0 - std::pow((double)ah.b2, (double)t);
    for (size_t i = 0; i < n; ++i) {
      const double g = G[i] * c;
      m[i] = ah.b1 * m[i] + (1.0 - ah.b1) * g;
      v[i] = ah.b2 * v[i] + (1.0 - ah.b2) * g * g;
      P[i] -= (double)lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + ah.eps);
    }
  }
  return gn;
}

}  // namespace netref
