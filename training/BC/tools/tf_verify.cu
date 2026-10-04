// tf(토큰마다 학생) 검증(계획서 9절): V4 연산 하나씩·V5 학습 한 스텝(CPU FP64 정답 + EMUL 바닥, GPU 오차 ≤ 2 × 바닥), FP64 대 유한 차분,
// 음성 대조(버그 넷 — 각각 실패해야 함), V6 그래프 == 즉시 실행, V7 같은 씨앗 비트 동일·다른 씨앗 다름, 속도.
//   tf_verify v5 | neg | fd | v6 | v7 | bench [B] | all       (통과 = 종료 코드 0)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "detmath.h"
#include "net_ref.h"
#include "tf.h"
#include "tf_ref.h"

using namespace tfm;
using tfref::V;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)
static const cudaStream_t ST = cudaStreamPerThread;

template <class T_> static std::vector<T_> dl(const T_* d, size_t n) {
  std::vector<T_> h(n);
  CK(cudaMemcpy(h.data(), d, sizeof(T_) * n, cudaMemcpyDeviceToHost));
  return h;
}
static V bfv(const std::vector<uint16_t>& h) { V v(h.size()); for (size_t i = 0; i < h.size(); ++i) v[i] = net::bf2f(h[i]); return v; }
static V fv(const std::vector<float>& h) { return V(h.begin(), h.end()); }
// 행 rows × ld 에서 앞 cols 칸만
static V cols(const V& a, long long rows, int ld, int nc) { V o; o.reserve((size_t)rows * nc); for (long long r = 0; r < rows; ++r) for (int c = 0; c < nc; ++c) o.push_back(a[(size_t)(r * ld + c)]); return o; }

static double rel(const V& g, const V& r) {
  double a = 0, b = 0;
  for (size_t i = 0; i < r.size(); ++i) { a += (g[i] - r[i]) * (g[i] - r[i]); b += r[i] * r[i]; }
  return b > 0 ? std::sqrt(a / b) : std::sqrt(a);
}
static int n_fail = 0, n_check = 0;
static bool check(const char* name, const V& gpu, const V& f64, const V& emu) {
  const double e = rel(gpu, f64), fl = rel(emu, f64);
  const bool ok = e <= 2.0 * fl + 1e-7;
  ++n_check;
  if (!ok) ++n_fail;
  std::printf("  %-28s gpu %.3e  floor %.3e  ratio %5.2f  %s\n", name, e, fl, fl > 0 ? e / fl : 0.0, ok ? "ok" : "FAIL");
  return ok;
}

// ---- 입력(결정적 난수) ----
struct HostIn {
  int B;
  std::vector<uint16_t> g[N_GRP];
  std::vector<uint32_t> obj, off;
  std::vector<float> chunk, cmask;
};
static HostIn make_in(int B, int H, int A, uint64_t seed) {
  HostIn h;
  h.B = B;
  uint64_t s = seed * 0x9E3779B97F4A7C15ull + 99;
  for (int g = 0; g < N_GRP; ++g) {
    const int n = B * kGrp[g].n_tok, K = kGrp[g].K, kr = kGrp[g].k_real;
    h.g[g].assign((size_t)n * K, 0);
    for (int r = 0; r < n; ++r)
      for (int k = 0; k < K; ++k) h.g[g][(size_t)r * K + k] = k < kr ? net::f2bf(dm::rand_range(s, -1.f, 1.f)) : k == kr ? (uint16_t)0x3f80 : 0;
  }
  h.obj.resize(B); h.off.resize(B);
  for (int b = 0; b < B; ++b) {
    const int ns = (int)(dm::rand01(s) * 17.f);
    h.obj[b] = ns >= 32 ? 0xffffffffu : (ns ? ((1u << ns) - 1u) : 0u);
    h.off[b] = 0;
  }
  if (B > 1) h.off[1] = (1u << G_WALL) | (1u << G_ROOM);
  if (B > 2) h.off[2] = (1u << G_OBJ) | (1u << G_IMG);
  h.chunk.resize((size_t)B * H * A); h.cmask.resize((size_t)B * H);
  for (auto& v : h.chunk) v = dm::rand_range(s, -1.f, 1.f);
  for (auto& v : h.cmask) v = dm::rand01(s) < 0.8f ? 1.f : 0.f;
  return h;
}
struct DevIn {
  uint16_t* g[N_GRP];
  uint32_t *obj, *off, *adim;
  float *chunk, *cmask;
  long long* iter;
  TfIn in;
  TfFlow fl;
  template <class T_> T_* up(const std::vector<T_>& v) { T_* d; CK(cudaMalloc(&d, sizeof(T_) * v.size() + 16)); CK(cudaMemcpy(d, v.data(), sizeof(T_) * v.size(), cudaMemcpyHostToDevice)); return d; }
  DevIn(const HostIn& h, uint32_t adim_v, long long it, uint64_t seed) {
    for (int g = 0; g < N_GRP; ++g) { this->g[g] = up(h.g[g]); in.g[g] = this->g[g]; }
    obj = up(h.obj); off = up(h.off);
    adim = up(std::vector<uint32_t>{adim_v});
    chunk = up(h.chunk); cmask = up(h.cmask);
    iter = up(std::vector<long long>{it});
    in.obj_mask = obj; in.grp_off = off;
    fl = TfFlow{chunk, cmask, adim, iter, seed};
  }
};
static void train_fwd_bwd(Tf& t, DevIn& di, int B) {
  t.forward_prefix(di.in, B, ST);
  t.flow_inputs(di.fl, B, ST);
  t.expert_forward(B, true, ST);
  t.loss(di.fl, B, ST);
  t.backward(di.in, B, ST);
}
static tfref::In ref_in(const Tf& t, const HostIn& h, uint32_t adim) {
  tfref::In r;
  r.B = h.B;
  for (int g = 0; g < N_GRP; ++g) r.g[g] = bfv(h.g[g]);
  r.obj = h.obj; r.off = h.off;
  const long long RA = (long long)h.B * t.c.H;
  r.ain = bfv(dl(t.ain, (size_t)RA * t.KA));
  r.u = fv(dl(t.u, (size_t)RA * t.c.A));
  r.cmask = h.cmask;
  r.adim = adim;
  return r;
}
static V pslice(const V& P, long long off, long long n) { return V(P.begin() + off, P.begin() + off + n); }

static bool g_def = false;   // --def: 기본 모양(d 256, 머리 4, 몸통 6 + 전문가 4 층, B 4)
static TfCfg cfg_v5() {
  TfCfg c;
  if (g_def) { c.Bmax = 4; c.dw_chunk = 256; c.seed = 3; return c; }
  c.d = 128; c.heads = 2; c.layers = 2; c.mlp = 512; c.e_layers = 2; c.H = 16; c.A = 8; c.obj_hidden = 128; c.Bmax = 8; c.dw_chunk = 256; c.seed = 3;
  return c;
}

// V4 + V5 (+ Adam 1·10 스텝). bug ≠ 0 이면 음성 대조(실패 수를 돌려줌)
static int run_v5(int bug, bool adam10) {
  const TfCfg c = cfg_v5();
  const int B = c.Bmax;
  const uint32_t adim = 0x3fu;   // 행동 6·7 고정(커리큘럼 마스크 시험)
  Tf t;
  t.init(c);
  t.bug = bug;
  const int d = c.d, d1 = d + 16;
  const long long R = (long long)B * L_TOK, RA = (long long)B * c.H;
  CK(cudaMalloc(&t.tap_dO, R * d * 2)); CK(cudaMalloc(&t.tap_dQKV, R * 3 * d * 2)); CK(cudaMalloc(&t.tap_dT, R * d * 4)); CK(cudaMalloc(&t.tap_dRpre, R * d * 4));
  CK(cudaMalloc(&t.tap_edO, RA * d * 2)); CK(cudaMalloc(&t.tap_edQKV, RA * 3 * d * 2)); CK(cudaMalloc(&t.tap_edPKV, R * 2 * d * 2));
  t.tap = true;
  HostIn h = make_in(B, c.H, c.A, 7);
  DevIn di(h, adim, 5, 1234);
  train_fwd_bwd(t, di, B);
  CK(cudaDeviceSynchronize());
  const int f0 = n_fail;
  tfref::tc_model(16, c.dw_chunk);
  const V P = fv(dl(t.P, t.lay.total));
  V Pw = P;   // EMUL 가중치(bf16)
  for (auto& e : Pw) e = net::rbf((float)e);
  tfref::In ri = ref_in(t, h, adim);
  std::printf("[V4] ops, layer 0 (same quantized inputs)\n");
  {
    // 임베딩·몸통 층 0 앞: 전체 앞을 돌려 첫 단계 값만 견줌(입력 = 같은 양자화 입력)
    tfref::Trace a, b;
    tfref::run(tfref::FP64, c, t.lay, P, ri, a, false);
    tfref::run(tfref::EMUL, c, t.lay, P, ri, b, false);
    check("embed X0", fv(dl(t.Xs[0], R * d)), a.X0, b.X0);
    // LN 앞 / QKV / 어텐션 앞 / 잔차 / MLP: GPU 입력에서 한 연산씩
    const V X0g = fv(dl(t.Xs[0], R * d));
    const auto& w = t.lay.blk[0];
    for (int md = 0; md < 2; ++md) (void)md;
    V lf((size_t)R * d1, 0), le((size_t)R * d1, 0);
    tfref::ln_fwd(tfref::FP64, X0g.data(), (int)R, d, P.data() + w.ln1g, P.data() + w.ln1b, lf.data(), d1, nullptr, nullptr);
    tfref::ln_fwd(tfref::EMUL, X0g.data(), (int)R, d, P.data() + w.ln1g, P.data() + w.ln1b, le.data(), d1, nullptr, nullptr);
    check("LN fwd", cols(bfv(dl(t.A1[0], R * d1)), R, d1, d), cols(lf, R, d1, d), cols(le, R, d1, d));
    const V A1g = bfv(dl(t.A1[0], R * d1));
    V qf((size_t)R * 3 * d), qe((size_t)R * 3 * d);
    tfref::lin(tfref::FP64, A1g.data(), d1, (int)R, P.data() + w.qkv.off, 3 * d, d1, qf.data(), 3 * d, 0);
    tfref::lin(tfref::EMUL, A1g.data(), d1, (int)R, Pw.data() + w.qkv.off, 3 * d, d1, qe.data(), 3 * d, 0);
    check("QKV linear", bfv(dl(t.QKV[0], R * 3 * d)), qf, qe);
    // 어텐션 앞(가림 있음)
    const V Q = bfv(dl(t.QKV[0], R * 3 * d));
    const std::vector<uint8_t> tv = dl(t.tv, R);
    auto attn_all = [&](tfref::Mode md, const V& q, int ldq, int nq, const V& kv1, int ld1, int ko1, int vo1, int n1, const V* kv2, int ld2, int ko2, int vo2,
                        int n2, V& o, int ldo) {
      const int nk = n1 + n2;
      for (int b = 0; b < B; ++b)
        for (int hh = 0; hh < c.heads; ++hh) {
          V Qh((size_t)nq * 64), K((size_t)nk * 64), Vv((size_t)nk * 64), Oh((size_t)nq * 64);
          std::vector<uint8_t> val(nk, 1);
          for (int i = 0; i < nq; ++i) for (int e = 0; e < 64; ++e) Qh[i * 64 + e] = q[((size_t)b * nq + i) * ldq + hh * 64 + e];
          for (int j = 0; j < nk; ++j)
            for (int e = 0; e < 64; ++e) {
              if (j < n1) { K[j * 64 + e] = kv1[((size_t)b * n1 + j) * ld1 + ko1 + hh * 64 + e]; Vv[j * 64 + e] = kv1[((size_t)b * n1 + j) * ld1 + vo1 + hh * 64 + e]; }
              else { K[j * 64 + e] = (*kv2)[((size_t)b * n2 + j - n1) * ld2 + ko2 + hh * 64 + e]; Vv[j * 64 + e] = (*kv2)[((size_t)b * n2 + j - n1) * ld2 + vo2 + hh * 64 + e]; }
            }
          for (int j = 0; j < n1; ++j) val[j] = tv[(size_t)b * L_TOK + j];
          tfref::attn_fwd(md, Qh.data(), K.data(), Vv.data(), val.data(), nq, nk, Oh.data());
          for (int i = 0; i < nq; ++i) for (int e = 0; e < 64; ++e) o[((size_t)b * nq + i) * ldo + hh * 64 + e] = Oh[i * 64 + e];
        }
    };
    V of((size_t)R * d), oe((size_t)R * d);
    attn_all(tfref::FP64, Q, 3 * d, L_TOK, Q, 3 * d, d, 2 * d, L_TOK, nullptr, 0, 0, 0, 0, of, d);
    attn_all(tfref::EMUL, Q, 3 * d, L_TOK, Q, 3 * d, d, 2 * d, L_TOK, nullptr, 0, 0, 0, 0, oe, d);
    check("attention fwd (masked)", cols(bfv(dl(t.O[0], R * d1)), R, d1, d), of, oe);
    // 잔차 + 출력 투영
    const V Og = bfv(dl(t.O[0], R * d1));
    V x1f = X0g, x1e = X0g;
    tfref::lin(tfref::FP64, Og.data(), d1, (int)R, P.data() + w.wo.off, d, d1, x1f.data(), d, 3);
    tfref::lin(tfref::EMUL, Og.data(), d1, (int)R, Pw.data() + w.wo.off, d, d1, x1e.data(), d, 3);
    check("out proj + residual", fv(dl(t.Xs[1], R * d)), x1f, x1e);
    // MLP
    const V A2g = bfv(dl(t.A2[0], R * d1));
    V hf((size_t)R * (c.mlp + 16), 0), he((size_t)R * (c.mlp + 16), 0);
    tfref::lin(tfref::FP64, A2g.data(), d1, (int)R, P.data() + w.w1.off, c.mlp, d1, hf.data(), c.mlp + 16, 1);
    tfref::lin(tfref::EMUL, A2g.data(), d1, (int)R, Pw.data() + w.w1.off, c.mlp, d1, he.data(), c.mlp + 16, 1);
    check("MLP fc1 ELU", cols(bfv(dl(t.Hh[0], R * (c.mlp + 16))), R, c.mlp + 16, c.mlp), cols(hf, R, c.mlp + 16, c.mlp), cols(he, R, c.mlp + 16, c.mlp));
    const V Hg = bfv(dl(t.Hh[0], R * (c.mlp + 16)));
    const V X1g = fv(dl(t.Xs[1], R * d));
    V x2f = X1g, x2e = X1g;
    tfref::lin(tfref::FP64, Hg.data(), c.mlp + 16, (int)R, P.data() + w.w2.off, d, c.mlp + 16, x2f.data(), d, 3);
    tfref::lin(tfref::EMUL, Hg.data(), c.mlp + 16, (int)R, Pw.data() + w.w2.off, d, c.mlp + 16, x2e.data(), d, 3);
    check("MLP fc2 + residual", fv(dl(t.Xs[2], R * d)), x2f, x2e);
    // 전문가 어텐션 앞(키 = prefix 150 가림 + 행동 16)
    const V eQ = bfv(dl(t.eQKV[0], RA * 3 * d)), eP = bfv(dl(t.ePKV[0], R * 2 * d));
    V eof((size_t)RA * d), eoe((size_t)RA * d);
    attn_all(tfref::FP64, eQ, 3 * d, c.H, eP, 2 * d, 0, d, L_TOK, &eQ, 3 * d, d, 2 * d, c.H, eof, d);
    attn_all(tfref::EMUL, eQ, 3 * d, c.H, eP, 2 * d, 0, d, L_TOK, &eQ, 3 * d, d, 2 * d, c.H, eoe, d);
    check("expert attention fwd", cols(bfv(dl(t.eO[0], RA * d1)), RA, d1, d), eof, eoe);
    // 어텐션 뒤(몸통 층 0): GPU dO·QKV → dQKV
    const V dOg = bfv(dl(t.tap_dO, R * d));
    auto attn_bwd_all = [&](tfref::Mode md, const V& q, int ldq, int nq, const V& kv1, int ld1, int ko1, int vo1, int n1, const V* kv2, int ld2, int ko2, int vo2,
                            int n2, const V& dO, V& dq, V& dk1, V& dv1, V& dk2, V& dv2) {
      const int nk = n1 + n2;
      for (int b = 0; b < B; ++b)
        for (int hh = 0; hh < c.heads; ++hh) {
          V Qh((size_t)nq * 64), K((size_t)nk * 64), Vv((size_t)nk * 64), G((size_t)nq * 64), dQ((size_t)nq * 64), dK((size_t)nk * 64), dV((size_t)nk * 64);
          std::vector<uint8_t> val(nk, 1);
          for (int i = 0; i < nq; ++i) for (int e = 0; e < 64; ++e) { Qh[i * 64 + e] = q[((size_t)b * nq + i) * ldq + hh * 64 + e]; G[i * 64 + e] = dO[((size_t)b * nq + i) * d + hh * 64 + e]; }
          for (int j = 0; j < nk; ++j)
            for (int e = 0; e < 64; ++e) {
              if (j < n1) { K[j * 64 + e] = kv1[((size_t)b * n1 + j) * ld1 + ko1 + hh * 64 + e]; Vv[j * 64 + e] = kv1[((size_t)b * n1 + j) * ld1 + vo1 + hh * 64 + e]; }
              else { K[j * 64 + e] = (*kv2)[((size_t)b * n2 + j - n1) * ld2 + ko2 + hh * 64 + e]; Vv[j * 64 + e] = (*kv2)[((size_t)b * n2 + j - n1) * ld2 + vo2 + hh * 64 + e]; }
            }
          for (int j = 0; j < n1; ++j) val[j] = tv[(size_t)b * L_TOK + j];
          tfref::attn_bwd(md, Qh.data(), K.data(), Vv.data(), val.data(), G.data(), nq, nk, dQ.data(), dK.data(), dV.data());
          for (int i = 0; i < nq; ++i) for (int e = 0; e < 64; ++e) dq[((size_t)b * nq + i) * d + hh * 64 + e] = dQ[i * 64 + e];
          for (int j = 0; j < nk; ++j)
            for (int e = 0; e < 64; ++e) {
              if (j < n1) { dk1[((size_t)b * n1 + j) * d + hh * 64 + e] = dK[j * 64 + e]; dv1[((size_t)b * n1 + j) * d + hh * 64 + e] = dV[j * 64 + e]; }
              else { dk2[((size_t)b * n2 + j - n1) * d + hh * 64 + e] = dK[j * 64 + e]; dv2[((size_t)b * n2 + j - n1) * d + hh * 64 + e] = dV[j * 64 + e]; }
            }
        }
    };
    {
      V dq[2], dk[2], dv[2], dum;
      for (int m = 0; m < 2; ++m) {
        dq[m].assign((size_t)R * d, 0); dk[m].assign((size_t)R * d, 0); dv[m].assign((size_t)R * d, 0);
        attn_bwd_all((tfref::Mode)m, Q, 3 * d, L_TOK, Q, 3 * d, d, 2 * d, L_TOK, nullptr, 0, 0, 0, 0, dOg, dq[m], dk[m], dv[m], dum, dum);
      }
      const V g = bfv(dl(t.tap_dQKV, R * 3 * d));
      V gq, gk, gv;
      for (long long r = 0; r < R; ++r) for (int e = 0; e < d; ++e) { gq.push_back(g[r * 3 * d + e]); gk.push_back(g[r * 3 * d + d + e]); gv.push_back(g[r * 3 * d + 2 * d + e]); }
      check("attention bwd dQ", gq, dq[0], dq[1]);
      check("attention bwd dK", gk, dk[0], dk[1]);
      check("attention bwd dV", gv, dv[0], dv[1]);
    }
    {
      const V edO = bfv(dl(t.tap_edO, RA * d));
      V dq[2], dk1[2], dv1[2], dk2[2], dv2[2];
      for (int m = 0; m < 2; ++m) {
        dq[m].assign((size_t)RA * d, 0); dk1[m].assign((size_t)R * d, 0); dv1[m].assign((size_t)R * d, 0); dk2[m].assign((size_t)RA * d, 0); dv2[m].assign((size_t)RA * d, 0);
        attn_bwd_all((tfref::Mode)m, eQ, 3 * d, c.H, eP, 2 * d, 0, d, L_TOK, &eQ, 3 * d, d, 2 * d, c.H, edO, dq[m], dk1[m], dv1[m], dk2[m], dv2[m]);
      }
      const V g = bfv(dl(t.tap_edQKV, RA * 3 * d)), gp = bfv(dl(t.tap_edPKV, R * 2 * d));
      V gq, gk2, gv2, gk1, gv1;
      for (long long r = 0; r < RA; ++r) for (int e = 0; e < d; ++e) { gq.push_back(g[r * 3 * d + e]); gk2.push_back(g[r * 3 * d + d + e]); gv2.push_back(g[r * 3 * d + 2 * d + e]); }
      for (long long r = 0; r < R; ++r) for (int e = 0; e < d; ++e) { gk1.push_back(gp[r * 2 * d + e]); gv1.push_back(gp[r * 2 * d + d + e]); }
      check("expert attn bwd dQ", gq, dq[0], dq[1]);
      check("expert attn bwd dK prefix", gk1, dk1[0], dk1[1]);
      check("expert attn bwd dV prefix", gv1, dv1[0], dv1[1]);
      check("expert attn bwd dK action", gk2, dk2[0], dk2[1]);
      check("expert attn bwd dV action", gv2, dv2[0], dv2[1]);
    }
    {  // LN 뒤(몸통 층 0 의 LN1): dR = dRpre + LNbwd(dT)
      const V dT = fv(dl(t.tap_dT, R * d)), pre = fv(dl(t.tap_dRpre, R * d));
      V rf = pre, re = pre;
      tfref::ln_bwd(tfref::FP64, dT.data(), X0g.data(), (int)R, d, P.data() + w.ln1g, rf.data(), nullptr, nullptr);
      tfref::ln_bwd(tfref::EMUL, dT.data(), X0g.data(), (int)R, d, P.data() + w.ln1g, re.data(), nullptr, nullptr);
      // 비교는 LN 몫(dR − dRpre)만 — 앞쪽 몫이 같은 값이라 상대 오차를 흐리지 않게
      const V gdr = fv(dl(t.dR, R * d));
      V a, b2, g2;
      for (size_t i = 0; i < pre.size(); ++i) { a.push_back(rf[i] - pre[i]); b2.push_back(re[i] - pre[i]); g2.push_back(gdr[i] - pre[i]); }
      check("LN bwd (dX)", g2, a, b2);
    }
  }
  std::printf("[V5] full train step\n");
  tfref::Trace a, b;
  tfref::run(tfref::FP64, c, t.lay, P, ri, a, true);
  tfref::run(tfref::EMUL, c, t.lay, P, ri, b, true);
  const float lg = dl(t.loss_d, 1)[0];
  std::printf("  loss gpu %.7f  fp64 %.7f  emul %.7f\n", lg, a.loss, b.loss);
  {   // 손실은 낱값 하나: EMUL 의 상대 오차는 표본 하나라 우연히 0 에 가까울 수 있다(2026-10-05, 목표 묶음 K 48 로 입력이 바뀌자 GPU 1.3e-6 대 EMUL 2.1e-7).
      // 바닥 = max(EMUL, FP32 합 반올림 한도 √(항 수)·2⁻²⁴) — 항 수 = 행동 칸 B·H·A. 벡터 값(velocity 등)은 그대로 EMUL 바닥
    const double n_terms = (double)RA * c.A, fl32 = std::sqrt(n_terms) * 5.960464477539063e-8;
    const double e = std::fabs((double)lg - a.loss) / std::fabs(a.loss), fe = std::fabs((double)b.loss - a.loss) / std::fabs(a.loss), fl = std::max(fe, fl32);
    const bool ok = e <= 2.0 * fl + 1e-7;
    ++n_check;
    if (!ok) ++n_fail;
    std::printf("  %-28s gpu %.3e  floor %.3e (emul %.3e, fp32 sum bound %.3e over %.0f terms)  ratio %5.2f  %s\n", "loss", e, fl, fe, fl32, n_terms, e / fl, ok ? "ok" : "FAIL");
  }
  check("prefix Pf", cols(bfv(dl(t.Pf, R * d1)), R, d1, d), cols(a.Pf, R, d1, d), cols(b.Pf, R, d1, d));
  check("velocity", fv(dl(t.vel, RA * c.A)), a.vel, b.vel);
  check("dX0 (embedding grad)", fv(dl(t.dR, R * d)), a.dX0, b.dX0);
  const V G = fv(dl(t.G, t.lay.total));
  int wi = 0;
  for (const auto& w : t.lay.weights) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "grad W%02d [%d x %d]", wi++, w.N, w.K);
    const long long n = (long long)w.N * w.K;
    check(nm, pslice(G, w.off, n), pslice(a.grad, w.off, n), pslice(b.grad, w.off, n));
  }
  int vi = 0;
  for (const auto& v : t.lay.vecs) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "grad vec%02d [%d]", vi++, v.second);
    check(nm, pslice(G, v.first, v.second), pslice(a.grad, v.first, v.second), pslice(b.grad, v.first, v.second));
  }
  // Adam 1 스텝(+ 10 스텝, 같은 미니배치·같은 τ/ε)
  {
    net::TrainState hs{};
    hs.lr = 1e-3f;
    net::TrainState* ts;
    CK(cudaMalloc(&ts, sizeof hs));
    CK(cudaMemcpy(ts, &hs, sizeof hs, cudaMemcpyHostToDevice));
    const net::AdamHyper ah{0.9f, 0.999f, 1e-8f, 1.0f};
    t.adam(ts, ah, ST);
    CK(cudaDeviceSynchronize());
    V Pf64 = P, Pem = P, m1(P.size(), 0), v1(P.size(), 0), m2(P.size(), 0), v2(P.size(), 0);
    netref::adam(netref::FP64, Pf64, a.grad, m1, v1, 1, 1e-3f, ah);
    netref::adam(netref::EMUL, Pem, b.grad, m2, v2, 1, 1e-3f, ah);
    auto delta = [&](const V& x) { V o(x.size()); for (size_t i = 0; i < x.size(); ++i) o[i] = x[i] - P[i]; return o; };
    check("Adam 1 step dP", delta(fv(dl(t.P, t.lay.total))), delta(Pf64), delta(Pem));
    if (adam10) {
      for (int k = 1; k < 10; ++k) {
        train_fwd_bwd(t, di, B);
        t.adam(ts, ah, ST);
        tfref::Trace a2, b2;
        tfref::run(tfref::FP64, c, t.lay, Pf64, ri, a2, true);
        netref::adam(netref::FP64, Pf64, a2.grad, m1, v1, k + 1, 1e-3f, ah);
        tfref::run(tfref::EMUL, c, t.lay, Pem, ri, b2, true);
        netref::adam(netref::EMUL, Pem, b2.grad, m2, v2, k + 1, 1e-3f, ah);
      }
      CK(cudaDeviceSynchronize());
      check("Adam 10 steps dP", delta(fv(dl(t.P, t.lay.total))), delta(Pf64), delta(Pem));
    }
    cudaFree(ts);
  }
  return n_fail - f0;
}

// FP64 참조판 대 중심 유한 차분(작은 설정, 입력은 호스트에서 바로)
static int run_fd() {
  TfCfg c;
  c.d = 64; c.heads = 1; c.layers = 1; c.mlp = 128; c.e_layers = 1; c.H = 4; c.A = 8; c.obj_hidden = 64; c.Bmax = 4; c.dw_chunk = 1024;
  const TfLayout lay = tf_layout(c);
  const int B = c.Bmax, KA = (c.A + TEMB + 1 + 15) / 16 * 16;
  HostIn h = make_in(B, c.H, c.A, 11);
  tfref::In ri;
  ri.B = B;
  for (int g = 0; g < N_GRP; ++g) ri.g[g] = bfv(h.g[g]);
  ri.obj = h.obj; ri.off = h.off; ri.cmask = h.cmask; ri.adim = 0x3fu;
  uint64_t s = 77;
  ri.ain.assign((size_t)B * c.H * KA, 0);
  for (int r = 0; r < B * c.H; ++r) for (int k = 0; k < KA; ++k) ri.ain[(size_t)r * KA + k] = k < c.A + TEMB ? net::rbf(dm::rand_range(s, -1.f, 1.f)) : k == c.A + TEMB;
  ri.u.resize((size_t)B * c.H * c.A);
  for (auto& v : ri.u) v = dm::rand_range(s, -1.5f, 1.5f);
  // 변수: GPU 와 같은 초기화 규칙을 흉내 낼 필요 없음 — 0 아닌 기울기가 나게 아무 값
  V P((size_t)lay.total, 0.0);
  for (auto& w : lay.weights) for (long long i = 0; i < (long long)w.N * w.K; ++i) P[w.off + i] = dm::rand_range(s, -1.f, 1.f) * std::sqrt(1.f / (float)w.K) * 1.5;
  for (auto& v : lay.vecs) for (int i = 0; i < v.second; ++i) P[v.first + i] = dm::rand_range(s, -0.3f, 0.3f);
  for (auto* bl : {&lay.blk, &lay.eblk}) for (auto& b : *bl) for (int i = 0; i < c.d; ++i) { P[b.ln1g + i] += 1.0; P[b.ln2g + i] += 1.0; }
  for (int i = 0; i < c.d; ++i) { P[lay.lnf_g + i] += 1.0; P[lay.lne_g + i] += 1.0; }
  tfref::Trace tr;
  tfref::run(tfref::FP64, c, lay, P, ri, tr, true);
  std::printf("[FD] FP64 reference vs central differences (loss %.6f)\n", tr.loss);
  double worst = 0;
  int n = 0, bad = 0;
  std::vector<std::pair<long long, int>> pick;
  for (auto& w : lay.weights) for (int q = 0; q < 4; ++q) pick.push_back({w.off + (long long)((dm::rand01(s) * w.N)) * w.K + (int)(dm::rand01(s) * (w.K - 15)), 0});
  for (auto& v : lay.vecs) for (int q = 0; q < 3; ++q) pick.push_back({v.first + (int)(dm::rand01(s) * v.second), 1});
  for (auto& pk : pick) {
    const long long i = pk.first;
    const double g = tr.grad[(size_t)i];
    double best = 1e30;
    for (double hh : {1e-4, 1e-5, 1e-6}) {
      V Pp = P, Pm = P;
      Pp[(size_t)i] += hh; Pm[(size_t)i] -= hh;
      const double fd = (tfref::loss_only(c, lay, Pp, ri) - tfref::loss_only(c, lay, Pm, ri)) / (2 * hh);
      const double e = std::fabs(fd - g) / std::max(1e-6, std::fabs(fd) + std::fabs(g));
      best = std::min(best, e);
      if (best < 1e-6) break;
    }
    ++n;
    worst = std::max(worst, best);
    if (best > 1e-4) { ++bad; std::printf("  param %lld grad %.6e worst rel %.3e\n", i, g, best); }
  }
  std::printf("  %d params, worst relative error %.3e  %s\n", n, worst, bad ? "FAIL" : "ok");
  ++n_check;
  if (bad) ++n_fail;
  return bad;
}

// 그래프 몸통: 학습 3 스텝 + 추론
struct Runner {
  Tf t;
  DevIn* di;
  net::TrainState* ts;
  net::AdamHyper ah{0.9f, 0.999f, 1e-8f, 1.0f};
  float *xb, *act;
  int B;
  void step() {
    train_fwd_bwd(t, *di, B);
    t.adam(ts, ah, ST);
    inc_k();
  }
  void inc_k();
  void body() {
    for (int k = 0; k < 3; ++k) step();
    t.infer(di->in, B, 5, di->adim, di->iter, 0, 99, xb, act, ST);
  }
};
__global__ void inc_iter_k(long long* it) { it[0] = it[0] + 1; }
void Runner::inc_k() { inc_iter_k<<<1, 1, 0, ST>>>(di->iter); }
static void setup(Runner& r, const TfCfg& c, const HostIn& h, uint64_t seed) {
  TfCfg cc = c;
  cc.seed = seed;
  r.t.init(cc);
  r.B = c.Bmax;
  r.di = new DevIn(h, 0x3fu, 0, 4321);
  net::TrainState hs{};
  hs.lr = 3e-4f;
  CK(cudaMalloc(&r.ts, sizeof hs));
  CK(cudaMemcpy(r.ts, &hs, sizeof hs, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&r.xb, sizeof(float) * r.B * c.H * c.A));
  CK(cudaMalloc(&r.act, sizeof(float) * r.B * c.H * c.A));
}
struct Snap { std::vector<float> P, m, v, G, act, vel, loss; };
static Snap snap(Runner& r) {
  CK(cudaDeviceSynchronize());
  const auto& t = r.t;
  const size_t na = (size_t)r.B * t.c.H * t.c.A;
  return Snap{dl(t.P, t.lay.total), dl(t.Am, t.lay.total), dl(t.Av, t.lay.total), dl(t.G, t.lay.total), dl(r.act, na), dl(t.vel, na), dl(t.loss_d, 1)};
}
static long long diff(const Snap& a, const Snap& b) {
  long long n = 0;
  auto cmp = [&](const std::vector<float>& x, const std::vector<float>& y) { for (size_t i = 0; i < x.size(); ++i) n += std::memcmp(&x[i], &y[i], 4) != 0; };
  cmp(a.P, b.P); cmp(a.m, b.m); cmp(a.v, b.v); cmp(a.G, b.G); cmp(a.act, b.act); cmp(a.vel, b.vel); cmp(a.loss, b.loss);
  return n;
}
static cudaGraphExec_t capture(Runner& r) {
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(ST, cudaStreamCaptureModeThreadLocal));
  r.body();
  CK(cudaStreamEndCapture(ST, &g));
  cudaGraphExec_t ex;
  CK(cudaGraphInstantiate(&ex, g, 0));
  size_t nn = 0;
  cudaGraphGetNodes(g, nullptr, &nn);
  std::printf("  graph nodes %zu\n", nn);
  cudaGraphDestroy(g);
  return ex;
}
static TfCfg cfg_def(int B) { TfCfg c; c.Bmax = B; return c; }

static int run_v6() {
  std::printf("[V6] graph == eager (default shapes d 256, 6 + 4 layers, B 16: 3 train steps + infer 5 Euler)\n");
  const TfCfg c = cfg_def(16);
  HostIn h = make_in(c.Bmax, c.H, c.A, 5);
  Runner a, b;
  setup(a, c, h, 1);
  setup(b, c, h, 1);
  a.body();
  const Snap sa = snap(a);
  cudaGraphExec_t g = capture(b);
  CK(cudaGraphLaunch(g, ST));
  const Snap sb = snap(b);
  const long long n = diff(sa, sb);
  int zero_fixed = 0;
  for (size_t i = 0; i < sa.act.size(); ++i) if ((i % c.A) >= 6 && sa.act[i] != 0.f) ++zero_fixed;
  std::printf("  words differing %lld, fixed action dims nonzero %d, loss %.6f  %s\n", n, zero_fixed, sa.loss[0], n == 0 && zero_fixed == 0 ? "ok" : "FAIL");
  ++n_check;
  if (n || zero_fixed) ++n_fail;
  return (n || zero_fixed) ? 1 : 0;
}
static int run_v7() {
  std::printf("[V7] determinism (graph, 3 train steps + infer)\n");
  const TfCfg c = cfg_def(16);
  HostIn h = make_in(c.Bmax, c.H, c.A, 5);
  Snap s[3];
  for (int k = 0; k < 3; ++k) {
    Runner r;
    setup(r, c, h, k < 2 ? 1 : 2);
    cudaGraphExec_t g = capture(r);
    CK(cudaGraphLaunch(g, ST));
    s[k] = snap(r);
    cudaGraphExecDestroy(g);
  }
  const long long same = diff(s[0], s[1]), other = diff(s[0], s[2]);
  std::printf("  same seed: %lld words differ, other seed: %lld words differ  %s\n", same, other, same == 0 && other > 0 ? "ok" : "FAIL");
  ++n_check;
  if (same || !other) ++n_fail;
  return (same || !other) ? 1 : 0;
}
static void bench(int B) {
  const TfCfg c = cfg_def(B);
  HostIn h = make_in(B, c.H, c.A, 5);
  Runner r;
  setup(r, c, h, 1);
  const long long nparam = r.t.lay.total;
  std::printf("[bench] B %d, L %d, d %d, layers %d + %d, params %lld, device bytes %.2f GB\n", B, L_TOK, c.d, c.layers, c.e_layers, nparam, r.t.bytes / 1e9);
  auto timeit = [&](const char* nm, auto fn, int reps) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(ST, cudaStreamCaptureModeThreadLocal));
    fn();
    CK(cudaStreamEndCapture(ST, &g));
    cudaGraphExec_t ex;
    CK(cudaGraphInstantiate(&ex, g, 0));
    CK(cudaGraphLaunch(ex, ST));
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, ST);
    for (int k = 0; k < reps; ++k) CK(cudaGraphLaunch(ex, ST));
    cudaEventRecord(e1, ST);
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    std::printf("  %-34s %.2f ms\n", nm, ms / reps);
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(g);
  };
  timeit("prefix forward", [&] { r.t.forward_prefix(r.di->in, B, ST); }, 10);
  timeit("train step (fwd + loss + bwd + Adam)", [&] { r.step(); }, 10);
  timeit("infer (prefix + 10 Euler)", [&] { r.t.infer(r.di->in, B, 10, r.di->adim, r.di->iter, 0, 99, r.xb, r.act, ST); }, 10);
}

int main(int argc, char** argv) {
  const std::string m = argc > 1 ? argv[1] : "all";
  for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--def") g_def = true;
  if (m == "v5" || m == "all") run_v5(0, true);
  if (m == "fd" || m == "all") run_fd();
  if (m == "neg" || m == "all") {
    const char* nm[5] = {"", "attention bwd without 1/sqrt(dh)", "LN bwd without mean term", "key mask ignored", "flow loss without factor 2"};
    int caught = 0;
    for (int bug = 1; bug <= 4; ++bug) {
      const int f0 = n_fail, c0 = n_check;
      std::printf("[negative %d: %s]\n", bug, nm[bug]);
      const int nf = run_v5(bug, false);
      n_fail = f0;   // 음성 대조의 실패는 통과로 셈
      n_check = c0;
      std::printf("  -> %d checks failed %s\n", nf, nf > 0 ? "(expected)" : "NOT CAUGHT");
      caught += nf > 0;
    }
    ++n_check;
    if (caught != 4) ++n_fail;
    std::printf("[negative] %d / 4 caught\n", caught);
  }
  if (m == "v6" || m == "all") run_v6();
  if (m == "v7" || m == "all") run_v7();
  if (m == "bench") bench(argc > 2 ? std::atoi(argv[2]) : 256);
  if (m != "bench") std::printf("== %d / %d checks passed\n", n_check - n_fail, n_check);
  return n_fail ? 1 : 0;
}
