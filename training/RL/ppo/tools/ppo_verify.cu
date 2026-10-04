// 검증 사다리 V4–V7 (계획서 GPU_TRAINING.md 9절).
//   ppo_verify v4 [--negative]   층 하나씩: GPU 가 실제로 받은 입력으로 CPU FP64(정답)·EMUL(바닥)을 돌려 GPU 출력과 비교. 기준 GPU 오차 ≤ 2 × 바닥
//   ppo_verify v5 [--negative]   학습 한 스텝: 손실·모든 기울기·Adam 1·10 스텝 갱신량을 CPU FP64 와 비교(같은 기준) + FP64 유한 차분으로 참조판 자체 확인
//   ppo_verify v6                그래프 = 즉시 실행(비트 동일, 3 바퀴 뒤 변수·Adam·롤아웃 버퍼·지도 토큰·기록)
//   ppo_verify v7                같은 씨앗 두 번 비트 동일, 다른 씨앗은 달라야 함
//   ppo_verify bench N T iters   그래프로 바퀴를 돌려 처리량(nsys 로 호스트 API 를 셀 때 쓰는 판)
// --negative: 일부러 버그를 넣는다(1 = 정책 사슬 A3→A2 의 ELU' 빠뜨림, 2 = 가치 기울기 부호). 반드시 실패해야 종료 코드 0.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "net_ref.h"
#include "trainer.h"

using namespace net;
using ppo::Trainer;

#define VCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

template <class T_>
static std::vector<T_> dl(const T_* d, size_t n) {
  std::vector<T_> h(n);
  VCK(cudaMemcpy(h.data(), d, sizeof(T_) * n, cudaMemcpyDeviceToHost));
  return h;
}
static std::vector<double> bfv(const std::vector<uint16_t>& v) {
  std::vector<double> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = bf2f(v[i]);
  return o;
}
static std::vector<double> fv(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

static PpoConfig small_cfg(uint64_t seed, int graphs) {
  PpoConfig c{};
  c.n_env = 512; c.horizon = 16; c.epochs = 2; c.minibatches = 4; c.stage = 1; c.use_map = 1; c.adaptive_lr = 0; c.use_graphs = graphs;
  c.log_ring = 8; c.dw_chunk = 1024; c.seed = seed;
  c.gamma = 0.99f; c.lam = 0.95f; c.clip = 0.2f; c.vclip = 0.2f; c.vf_coef = 0.5f; c.ent_coef = 0.001f;
  c.lr = 3e-4f; c.lr_min = 1e-5f; c.lr_max = 1e-3f; c.kl_target = 0.01f; c.max_grad_norm = 1.0f;
  c.adam_b1 = 0.9f; c.adam_b2 = 0.999f; c.adam_eps = 1e-8f; c.init_logstd = -0.5f; c.reward_scale = 1.f;
  c.act_dims = 2; c.shape_coef = 1.f; c.shape_near = 8.f; c.shape_aim = 1.f; c.shape_zone = 1.f; c.shape_v = 4.f; c.shape_w = 2.f;
  return c;
}

// ---- 비교 표 ----
struct Report {
  int fails = 0, n = 0;
  // rows×cols 부분(줄 간격 ld)만: GPU·FP64·EMUL
  void cmp(const char* name, const double* gpu, const double* f64, const double* emu, long long rows, int cols, int ld, double abs_floor = 1e-12) {
    double eg = 0, ee = 0, nr = 0, mg = 0;
    for (long long r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        const size_t i = (size_t)r * ld + c;
        const double a = gpu[i] - f64[i], b = emu[i] - f64[i];
        eg += a * a; ee += b * b; nr += f64[i] * f64[i];
        mg = std::fmax(mg, std::fabs(a));
      }
    const double den = std::sqrt(nr) > 0 ? std::sqrt(nr) : 1.0;
    const double rg = std::sqrt(eg) / den, re = std::sqrt(ee) / den;
    const bool ok = rg <= 2.0 * re + abs_floor;
    ++n;
    if (!ok) ++fails;
    std::printf("  %-26s GPU %.3e  floor(EMUL) %.3e  ratio %5.2f  max|GPU-FP64| %.2e  %s\n", name, rg, re, re > 0 ? rg / re : 0.0, mg, ok ? "ok" : "FAIL");
  }
};

// 롤아웃 몇 바퀴 → 미니배치 하나 모으기(+ oldlogp·oldv 흔들어 자르기가 일어나게)
static netref::Batch make_batch(Trainer& tr, netref::Hyper& hy) {
  for (int k = 0; k < 2; ++k) tr.iterate();
  VCK(cudaDeviceSynchronize());
  PpoLog L;
  while (tr.poll(&L)) {}
  tr.rollout_body();
  tr.gae();
  tr.gather(0, 0);
  VCK(cudaDeviceSynchronize());
  const int M = tr.MB;
  netref::Batch b;
  b.M = M;
  b.x0 = dl(tr.x0, (size_t)M * X0_W);
  b.sin = dl(tr.sin, (size_t)M * KSLOT * SLOT_IN);
  b.mask = dl(tr.mask, M);
  b.act = dl(tr.mb_act, (size_t)M * N_ACT);
  b.oldlogp = dl(tr.mb_oldlogp, M);
  b.oldv = dl(tr.mb_oldv, M);
  b.adv = dl(tr.mb_adv, M);
  b.ret = dl(tr.mb_ret, M);
  TrainState s = dl(tr.ts, 1)[0];
  b.adv_mean = s.adv_mean;
  b.adv_std = s.adv_std;
  uint64_t r = 99;
  for (int i = 0; i < M; ++i) {
    b.oldlogp[i] += 0.3f * (dm::rand01(r) + dm::rand01(r) + dm::rand01(r) - 1.5f);
    b.oldv[i] += 0.5f * (dm::rand01(r) - 0.5f);
  }
  VCK(cudaMemcpy(tr.mb_oldlogp, b.oldlogp.data(), sizeof(float) * M, cudaMemcpyHostToDevice));
  VCK(cudaMemcpy(tr.mb_oldv, b.oldv.data(), sizeof(float) * M, cudaMemcpyHostToDevice));
  hy = netref::Hyper{tr.cfg.clip, tr.cfg.vclip, tr.cfg.vf_coef, tr.cfg.ent_coef, tr.cfg.act_dims};
  long nslot = 0;
  for (int i = 0; i < M; ++i) nslot += __builtin_popcount(b.mask[i]);
  std::printf("batch: M=%d rows, mean filled map slots %.2f, adv mean %.3f std %.3f\n", M, (double)nslot / M, b.adv_mean, b.adv_std);
  return b;
}

static int run_v4(int bug) {
  std::printf("== V4: layer forward/backward vs CPU FP64 (floor = CPU bf16 emulation)%s\n", bug ? "  [NEGATIVE CONTROL]" : "");
  Trainer tr(small_cfg(11, 0));
  netref::Hyper hy;
  netref::Batch b = make_batch(tr, hy);
  tr.bug = bug;
  tr.forward(tr.MB);
  tr.loss(tr.MB);
  tr.backward(tr.MB);
  VCK(cudaDeviceSynchronize());
  const int M = b.M, MS = M * KSLOT;
  const ParamLayout lay = param_layout();
  const std::vector<double> W = bfv(dl(tr.Pb, lay.total));
  const std::vector<double> P = fv(dl(tr.P, lay.total));
  const std::vector<double> Gg = fv(dl(tr.G, lay.total));
  std::vector<double> h[N_LAYER], dz[N_LAYER];
  for (int l = 0; l < N_LAYER; ++l) {
    const LayerDesc& L = kLayers[l];
    const long long rows = l <= L_S2 ? MS : M;
    if (L.act == ACT_ELU) h[l] = bfv(dl(tr.ho[l], (size_t)rows * L.ldo));
    dz[l] = bfv(dl(tr.dz[l], (size_t)rows * L.N));
  }
  h[L_A4] = fv(dl(tr.mean, (size_t)M * N_ACT));
  h[L_C4] = fv(dl(tr.val, (size_t)M * 8));
  const std::vector<double> sin = bfv(b.sin), x0 = bfv(dl(tr.x0, (size_t)M * X0_W));
  const std::vector<uint8_t> am8 = dl(tr.amax, (size_t)M * S_H);
  std::vector<int> amax(am8.begin(), am8.end());
  const std::vector<double> dpool = fv(dl(tr.dpool, (size_t)M * POOL_W));
  Report R;
  auto in_of = [&](int l) -> const double* { return l == L_S1 ? sin.data() : (l == L_A1 || l == L_C1) ? x0.data() : h[l - 1].data(); };
  // 앞
  for (int l = 0; l < N_LAYER; ++l) {
    const LayerDesc& L = kLayers[l];
    const long long rows = l <= L_S2 ? MS : M;
    std::vector<double> f((size_t)rows * L.ldo), e((size_t)rows * L.ldo);
    netref::lin_fwd(netref::FP64, in_of(l), (int)rows, L, W.data() + lay.off[l], f.data());
    netref::lin_fwd(netref::EMUL, in_of(l), (int)rows, L, W.data() + lay.off[l], e.data());
    static const char* nm[N_LAYER] = {"S1", "S2", "A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4"};
    std::string s = std::string("fwd ") + nm[l];
    R.cmp(s.c_str(), h[l].data(), f.data(), e.data(), rows, l == L_C4 ? 1 : L.N, L.ldo);
  }
  {   // 집합
    std::vector<double> f(x0), e(x0);
    std::vector<int> a1(amax), a2(amax);
    netref::pool_fwd(netref::FP64, h[L_S2].data(), b.mask.data(), M, f.data(), a1.data());
    netref::pool_fwd(netref::EMUL, h[L_S2].data(), b.mask.data(), M, e.data(), a2.data());
    R.cmp("pool fwd (mean|max)", x0.data(), f.data(), e.data(), M, POOL_W, X0_W);
    long bad = 0;
    for (size_t i = 0; i < amax.size(); ++i) bad += amax[i] != a1[i];
    std::printf("  %-26s argmax differs %ld / %zu (ties only)\n", "pool argmax", bad, amax.size());
  }
  {   // 손실
    std::vector<double> fa((size_t)M * 8), fc((size_t)M * 8), ea((size_t)M * 8), ec((size_t)M * 8), fl(8), el(8), st(4);
    netref::loss(netref::FP64, h[L_A4].data(), h[L_C4].data(), P.data() + lay.logstd, b, hy, fa.data(), fc.data(), fl.data(), st.data());
    netref::loss(netref::EMUL, h[L_A4].data(), h[L_C4].data(), P.data() + lay.logstd, b, hy, ea.data(), ec.data(), el.data(), st.data());
    R.cmp("loss dmean (dZ A4)", dz[L_A4].data(), fa.data(), ea.data(), M, N_ACT, N_ACT);
    R.cmp("loss dvalue (dZ C4)", dz[L_C4].data(), fc.data(), ec.data(), M, 1, 8);
    R.cmp("loss dlogstd", Gg.data() + lay.logstd, fl.data(), el.data(), 1, N_ACT, N_ACT, 1e-9);
    std::printf("    (clip fraction in this batch %.3f)\n", st[3]);
  }
  // 뒤: dX(dZ 사슬), dpool, 집합 뒤, dW
  static const char* dxn[N_LAYER] = {"", "dx S2->S1", "", "dx A2->A1", "dx A3->A2", "dx A4->A3", "", "dx C2->C1", "dx C3->C2", "dx C4->C3"};
  for (int l : {L_A4, L_A3, L_A2, L_C4, L_C3, L_C2, L_S2}) {
    const LayerDesc& L = kLayers[l];
    const int Np = kLayers[l - 1].N;
    const long long rows = l <= L_S2 ? MS : M;
    std::vector<double> f((size_t)rows * Np), e((size_t)rows * Np);
    netref::lin_dx(netref::FP64, dz[l].data(), (int)rows, L, W.data() + lay.off[l], Np, h[l - 1].data(), f.data(), true);
    netref::lin_dx(netref::EMUL, dz[l].data(), (int)rows, L, W.data() + lay.off[l], Np, h[l - 1].data(), e.data(), true);
    R.cmp(dxn[l], dz[l - 1].data(), f.data(), e.data(), rows, Np, Np);
  }
  {
    std::vector<double> fa((size_t)M * POOL_W), fc(fa), ea(fa), ec(fa), f(fa), e(fa);
    netref::lin_dx(netref::FP64, dz[L_A1].data(), M, kLayers[L_A1], W.data() + lay.off[L_A1], POOL_W, nullptr, fa.data(), false);
    netref::lin_dx(netref::FP64, dz[L_C1].data(), M, kLayers[L_C1], W.data() + lay.off[L_C1], POOL_W, nullptr, fc.data(), false);
    netref::lin_dx(netref::EMUL, dz[L_A1].data(), M, kLayers[L_A1], W.data() + lay.off[L_A1], POOL_W, nullptr, ea.data(), false);
    netref::lin_dx(netref::EMUL, dz[L_C1].data(), M, kLayers[L_C1], W.data() + lay.off[L_C1], POOL_W, nullptr, ec.data(), false);
    for (size_t i = 0; i < f.size(); ++i) { f[i] = fa[i] + fc[i]; e[i] = (double)((float)ea[i] + (float)ec[i]); }
    R.cmp("dx A1+C1 -> dpool", dpool.data(), f.data(), e.data(), M, POOL_W, POOL_W);
    std::vector<double> g((size_t)MS * S_H), ge(g);
    netref::pool_bwd(netref::FP64, dpool.data(), h[L_S2].data(), b.mask.data(), amax.data(), M, g.data());
    netref::pool_bwd(netref::EMUL, dpool.data(), h[L_S2].data(), b.mask.data(), amax.data(), M, ge.data());
    R.cmp("pool bwd (dZ S2)", dz[L_S2].data(), g.data(), ge.data(), MS, S_H, S_H);
  }
  for (int l = 0; l < N_LAYER; ++l) {
    const LayerDesc& L = kLayers[l];
    const long long rows = l <= L_S2 ? MS : M;
    std::vector<double> f((size_t)L.N * L.K), e(f);
    netref::lin_dw(netref::FP64, dz[l].data(), in_of(l), (int)rows, L, f.data());
    netref::lin_dw(netref::EMUL, dz[l].data(), in_of(l), (int)rows, L, e.data());
    static const char* nm[N_LAYER] = {"dW S1", "dW S2", "dW A1", "dW A2", "dW A3", "dW A4", "dW C1", "dW C2", "dW C3", "dW C4"};
    R.cmp(nm[l], Gg.data() + lay.off[l], f.data(), e.data(), L.N, L.K, L.K);
  }
  std::printf("V4: %d / %d checks pass\n", R.n - R.fails, R.n);
  return R.fails;
}

static int run_v5(int bug) {
  std::printf("== V5: one training step vs CPU FP64 chain (floor = CPU bf16 emulation chain)%s\n", bug ? "  [NEGATIVE CONTROL]" : "");
  Trainer tr(small_cfg(23, 0));
  netref::Hyper hy;
  netref::Batch b = make_batch(tr, hy);
  tr.bug = bug;
  const ParamLayout lay = param_layout();
  const std::vector<float> P0 = dl(tr.P, lay.total);
  const std::vector<float> m0 = dl(tr.Am, lay.total), v0 = dl(tr.Av, lay.total);
  const TrainState s0 = dl(tr.ts, 1)[0];
  tr.forward(tr.MB);
  tr.loss(tr.MB);
  tr.backward(tr.MB);
  VCK(cudaDeviceSynchronize());
  const TrainState s1 = dl(tr.ts, 1)[0];
  const std::vector<double> Gg = fv(dl(tr.G, lay.total));
  const std::vector<double> mean = fv(dl(tr.mean, (size_t)b.M * N_ACT)), val = fv(dl(tr.val, (size_t)b.M * 8));
  netref::Trace tf, te;
  auto t0 = std::chrono::steady_clock::now();
  netref::run(netref::FP64, P0, b, hy, tf);
  netref::run(netref::EMUL, P0, b, hy, te);
  std::printf("CPU chains: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  Report R;
  R.cmp("policy mean", mean.data(), tf.mean.data(), te.mean.data(), b.M, N_ACT, N_ACT);
  R.cmp("value", val.data(), tf.val.data(), te.val.data(), b.M, 1, 8);
  {
    const double gp = s1.s_pg - s0.s_pg, gv = s1.s_vl - s0.s_vl, gk = s1.s_kl - s0.s_kl;
    const double a[3] = {gp, gv, gk}, f[3] = {tf.pg, tf.vl, tf.kl}, e[3] = {te.pg, te.vl, te.kl};
    R.cmp("loss pg | v | kl", a, f, e, 1, 3, 3, 1e-7);
    std::printf("    pg %.6f  v %.6f  kl %.6f  clipfrac %.3f (FP64)\n", tf.pg, tf.vl, tf.kl, tf.clipfrac);
  }
  static const char* nm[N_LAYER] = {"grad S1", "grad S2", "grad A1", "grad A2", "grad A3", "grad A4", "grad C1", "grad C2", "grad C3", "grad C4"};
  for (int l = 0; l < N_LAYER; ++l)
    R.cmp(nm[l], Gg.data() + lay.off[l], tf.grad.data() + lay.off[l], te.grad.data() + lay.off[l], 1, kLayers[l].N * kLayers[l].K, 0);
  R.cmp("grad logstd", Gg.data() + lay.logstd, tf.grad.data() + lay.logstd, te.grad.data() + lay.logstd, 1, N_ACT, 0, 1e-9);

  // Adam: GPU 의 m, v, t, lr 에서 시작해 1 스텝, 그 뒤 같은 미니배치로 9 스텝 더(앞·뒤를 다시)
  const net::AdamHyper ah{tr.cfg.adam_b1, tr.cfg.adam_b2, tr.cfg.adam_eps, tr.cfg.max_grad_norm};
  std::vector<double> Pf(P0.begin(), P0.end()), Pe(Pf), mf(m0.begin(), m0.end()), vf(v0.begin(), v0.end()), me(mf), ve(vf);
  long long t = s0.adam_t;
  std::vector<double> gf = tf.grad, ge = te.grad;
  for (int step = 1; step <= 10; ++step) {
    tr.optimizer();
    ++t;
    netref::adam(netref::FP64, Pf, gf, mf, vf, t, s0.lr, ah);
    netref::adam(netref::EMUL, Pe, ge, me, ve, t, s0.lr, ah);
    if (step == 1 || step == 10) {
      VCK(cudaDeviceSynchronize());
      const std::vector<float> Pg = dl(tr.P, lay.total);
      std::vector<double> dg(lay.total), df(lay.total), de(lay.total);
      for (long long i = 0; i < lay.total; ++i) { dg[i] = (double)Pg[i] - P0[i]; df[i] = Pf[i] - P0[i]; de[i] = Pe[i] - P0[i]; }
      for (int l = 0; l < N_LAYER; ++l) {
        char s[64];
        std::snprintf(s, sizeof s, "Adam %2d-step dP %s", step, nm[l] + 5);
        R.cmp(s, dg.data() + lay.off[l], df.data() + lay.off[l], de.data() + lay.off[l], 1, kLayers[l].N * kLayers[l].K, 0);
      }
      char s[64];
      std::snprintf(s, sizeof s, "Adam %2d-step dP logstd", step);
      R.cmp(s, dg.data() + lay.logstd, df.data() + lay.logstd, de.data() + lay.logstd, 1, N_ACT, 0, 1e-9);
    }
    if (step < 10) {
      tr.forward(tr.MB);
      tr.loss(tr.MB);
      tr.backward(tr.MB);
      netref::Trace a, c;
      netref::run_d(netref::FP64, Pf, b, hy, a);
      netref::run_d(netref::EMUL, Pe, b, hy, c);
      gf = a.grad;
      ge = c.grad;
    }
  }

  // 참조판 자체: FP64 기울기 대 FP64 중심 유한 차분(행 256 개 부분 미니배치)
  int fd_fail = 0;
  if (!bug) {
    netref::Batch sb = b;
    const int Ms = 256;
    sb.M = Ms;
    sb.x0.resize((size_t)Ms * X0_W); sb.sin.resize((size_t)Ms * KSLOT * SLOT_IN); sb.mask.resize(Ms);
    sb.act.resize((size_t)Ms * N_ACT); sb.oldlogp.resize(Ms); sb.oldv.resize(Ms); sb.adv.resize(Ms); sb.ret.resize(Ms);
    netref::Trace st;
    netref::run(netref::FP64, P0, sb, hy, st);
    std::vector<double> Pd(P0.begin(), P0.end());
    uint64_t rs = 5;
    int n = 0;
    double worst = 0;
    for (int l = 0; l <= N_LAYER; ++l) {
      for (int k = 0; k < (l == N_LAYER ? N_ACT : 12); ++k) {
        long long j;
        if (l == N_LAYER) j = lay.logstd + k;
        else {
          const LayerDesc& L = kLayers[l];
          const int nout = l == L_C4 ? 1 : L.N;
          j = lay.off[l] + (long long)(dm::rand01(rs) * nout) * L.K + (long long)(dm::rand01(rs) * (L.bias + 1));
        }
        const double h = 1e-5 * std::fmax(1.0, std::fabs(Pd[j]));
        const double p = Pd[j];
        Pd[j] = p + h;
        const double lp = netref::loss_only(Pd, sb, hy);
        Pd[j] = p - h;
        const double lm = netref::loss_only(Pd, sb, hy);
        Pd[j] = p;
        const double fd = (lp - lm) / (2 * h), g = st.grad[j];
        const double err = std::fabs(fd - g) / std::fmax(std::fabs(g), 1e-7);
        worst = std::fmax(worst, err);
        ++n;
        if (err > 1e-4) { ++fd_fail; std::printf("    FD mismatch param %lld (layer %d): analytic %.6e fd %.6e\n", j, l, g, fd); }
      }
    }
    std::printf("  %-26s %d params, worst rel err %.2e  %s\n", "FP64 grad vs finite diff", n, worst, fd_fail ? "FAIL" : "ok");
  }
  std::printf("V5: %d / %d checks pass%s\n", R.n - R.fails, R.n, fd_fail ? " (finite-difference check FAILED)" : "");
  return R.fails + fd_fail;
}

struct Snap {
  std::vector<float> P, m, v, obs, adv, val, act;
  std::vector<uint8_t> tok;
  std::vector<PpoLog> logs;
};
static Snap run_iters(PpoConfig c, int iters) {
  Trainer tr(c);
  Snap s;
  for (int k = 0; k < iters; ++k) {
    tr.iterate();
    VCK(cudaDeviceSynchronize());
    PpoLog L;
    while (tr.poll(&L)) { L.rollout_ms = L.update_ms = 0; s.logs.push_back(L); }
  }
  const long long n = tr.lay.total;
  s.P = dl(tr.P, n); s.m = dl(tr.Am, n); s.v = dl(tr.Av, n);
  s.obs = dl(tr.obs_buf, (size_t)(tr.T + 1) * N_OBS_G1 * tr.N);
  s.adv = dl(tr.adv_buf, (size_t)tr.T * tr.N);
  s.val = dl(tr.val_buf, (size_t)(tr.T + 1) * tr.N);
  s.act = dl(tr.act_buf, (size_t)tr.T * tr.N * N_ACT);
  s.tok = dl(reinterpret_cast<const uint8_t*>(tr.tok->at(0)), sizeof(gmap::MapTok) * (size_t)(tr.T + 1) * tr.N);
  return s;
}
static long diff_snap(const Snap& a, const Snap& b, bool verbose) {
  auto cnt = [](const void* x, const void* y, size_t nb) { long d = 0; const uint8_t *p = (const uint8_t*)x, *q = (const uint8_t*)y; for (size_t i = 0; i < nb; i += 4) d += std::memcmp(p + i, q + i, 4) != 0; return d; };
  struct { const char* n; long d; } t[] = {
      {"params", cnt(a.P.data(), b.P.data(), 4 * a.P.size())}, {"adam m", cnt(a.m.data(), b.m.data(), 4 * a.m.size())},
      {"adam v", cnt(a.v.data(), b.v.data(), 4 * a.v.size())}, {"obs buffer", cnt(a.obs.data(), b.obs.data(), 4 * a.obs.size())},
      {"advantages", cnt(a.adv.data(), b.adv.data(), 4 * a.adv.size())}, {"values", cnt(a.val.data(), b.val.data(), 4 * a.val.size())},
      {"actions", cnt(a.act.data(), b.act.data(), 4 * a.act.size())}, {"map tokens", cnt(a.tok.data(), b.tok.data(), a.tok.size())},
      {"logs", a.logs.size() == b.logs.size() ? cnt(a.logs.data(), b.logs.data(), sizeof(PpoLog) * a.logs.size()) : 1}};
  long tot = 0;
  for (auto& e : t) {
    tot += e.d;
    if (verbose) std::printf("  %-12s %ld differing words\n", e.n, e.d);
  }
  return tot;
}

static int run_v6() {
  std::printf("== V6: CUDA graphs == eager (bit-identical after 3 iterations)\n");
  const Snap g = run_iters(small_cfg(31, 1), 3), e = run_iters(small_cfg(31, 0), 3);
  const long d = diff_snap(g, e, true);
  std::printf("V6: %s (%ld differing words); last log: succ %.3f kl %.5f\n", d ? "FAIL" : "PASS", d, g.logs.back().succ, g.logs.back().kl);
  return d ? 1 : 0;
}
static int run_v7() {
  std::printf("== V7: determinism (same seed twice, graphs)\n");
  const Snap a = run_iters(small_cfg(41, 1), 4), b = run_iters(small_cfg(41, 1), 4), c = run_iters(small_cfg(42, 1), 4);
  const long d = diff_snap(a, b, true), dc = diff_snap(a, c, false);
  std::printf("V7: same seed %s (%ld differing words); different seed differs in %ld words (%s)\n", d ? "FAIL" : "PASS", d, dc, dc ? "ok" : "FAIL");
  return (d || !dc) ? 1 : 0;
}

static int run_bench(int N, int T, int iters, int mbs, int use_map) {
  PpoConfig c = small_cfg(7, 1);
  c.n_env = N; c.horizon = T; c.minibatches = mbs; c.epochs = 5; c.use_map = use_map; c.adaptive_lr = 1;
  Trainer tr(c);
  std::printf("bench: N=%d T=%d MB=%d epochs=%d, device bytes %.2f GB\n", N, T, tr.MB, c.epochs, ppo_device_bytes(&tr) / 1e9);
  tr.iterate();
  VCK(cudaDeviceSynchronize());
  PpoLog L;
  while (tr.poll(&L)) {}
  const auto t0 = std::chrono::steady_clock::now();
  int done = 0;
  double ro = 0, up = 0;
  while (done < iters) {
    while (tr.issued - tr.polled < 4 && tr.issued < iters + 1) tr.iterate();
    while (tr.poll(&L)) { ++done; ro += L.rollout_ms; up += L.update_ms; }
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("bench: %d iters in %.3f s wall -> %.3e env-steps/s; GPU rollout %.2f ms + update %.2f ms per iter (rollout %.0f%%)\n", iters, s,
              (double)N * T * iters / s, ro / iters, up / iters, 100.0 * ro / (ro + up));
  return 0;
}

// 체크포인트를 결정적 정책(σ → e^-12, 학습률 0)으로 돌려 성공률을 잰다:  ppo_verify eval <ckpt> <stage> [iters=10] [N=4096] [use_map=1]
static int run_eval(const char* path, int stage, int iters, int N, int use_map) {
  PpoConfig c = small_cfg(1234, 1);
  c.n_env = N; c.horizon = 64; c.minibatches = 4; c.epochs = 1; c.stage = stage; c.use_map = use_map; c.lr = 0.f; c.lr_max = 0.f; c.lr_min = 0.f;
  c.adaptive_lr = 0;
  Trainer tr(c);
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::perror(path); return 2; }
  std::vector<uint8_t> b(64 + sizeof(float) * 3 * tr.lay.total + sizeof(TrainState));
  const size_t got = std::fread(b.data(), 1, b.size(), f);
  std::fclose(f);
  float* P = reinterpret_cast<float*>(b.data() + 64);
  for (int k = 0; k < N_ACT; ++k) P[tr.lay.logstd + k] = -12.f;
  if (got != b.size() || ppo_load(&tr, b.data(), (int64_t)b.size())) { std::fprintf(stderr, "bad checkpoint\n"); return 2; }
  double s = 0, cl = 0, to = 0, ne = 0, len = 0, ret = 0;
  for (int k = 0; k < iters; ++k) {
    tr.eval_iterate();   // 갱신 없음(체크포인트의 학습률이 되살아나도 변수는 그대로)
    VCK(cudaDeviceSynchronize());
    if (std::getenv("EVAL_OBS")) {   // 진단: 관측 차원마다 판 평균 |값|
      const int N = tr.N; const std::vector<float> o = dl(tr.obs_buf + (size_t)tr.T * N_OBS_G1 * N, (size_t)N_OBS_G1 * N);
      std::printf("  obs iter %d:", k);
      for (int d = 0; d < N_OBS_G1; ++d) { double a = 0; for (int i = 0; i < N; ++i) a += std::fabs(o[(size_t)d * N + i]); std::printf(" %d:%.3g", d, a / N); }
      std::printf("\n");
    }

    PpoLog L;
    while (tr.poll(&L)) {
      if (std::getenv("EVAL_TRACE")) std::printf("  eval iter %d: episodes %.0f success %.4f collision %.4f timeout %.4f len %.1f\n", k, L.n_eps, L.succ, L.coll, L.tout, L.ep_len);
      if (k < 3) continue;   // 처음 바퀴들은 이전(무작위 시작) 에피소드가 섞여 있어 버림
      s += L.succ * L.n_eps; cl += L.coll * L.n_eps; to += L.tout * L.n_eps; ne += L.n_eps; len += L.ep_len * L.n_eps; ret += L.ep_ret * L.n_eps;
    }
  }
  // 충돌이 에피소드 몇 번째 스텝에서 나는가(스텝마다 동기 — 평가 전용): 0–2 스텝 = 리셋 자리에서 피할 수 없는 충돌 후보
  {
    std::vector<int> iv, done;
    std::vector<float> f;
    std::vector<uint64_t> rg;
    long hist[5] = {0, 0, 0, 0, 0}, ends[4] = {0, 0, 0, 0}, near_cup = 0, near_wall = 0, both = 0;
    for (int k = 0; k < 4 * tr.T; ++k) {
      const int t = k % tr.T;
      if (t == 0 && k == 0) { VCK(cudaMemcpy(tr.obs_buf, tr.obs_buf + (size_t)tr.T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
                              VCK(cudaMemcpy(tr.tok->at(0), tr.tok->at(tr.T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToDevice)); }
      tr.env->download(f, iv, rg);
      tr.rollout_step(t);
      VCK(cudaDeviceSynchronize());
      done = dl(tr.done_buf + (size_t)t * N, N);
      if (t == tr.T - 1) { VCK(cudaMemcpy(tr.obs_buf, tr.obs_buf + (size_t)tr.T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
                           VCK(cudaMemcpy(tr.tok->at(0), tr.tok->at(tr.T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToDevice)); }
      for (int i = 0; i < N; ++i) {
        if (!done[i]) continue;
        ++ends[done[i]];
        if (done[i] != env::kCollision) continue;
        const int st = iv[env::I_STEP * N + i];   // 이 스텝 전 에피소드 스텝 수
        {   // 스텝 전 상태로: 컵까지 중심 거리, 가장 가까운 벽까지(중심) 거리
          const float x = f[env::F_X * N + i], y = f[env::F_Y * N + i], tx = f[env::F_TX * N + i], ty = f[env::F_TY * N + i];
          const float hx = f[env::F_RHX * N + i], hy = f[env::F_RHY * N + i];
          const float dc = std::sqrt((tx - x) * (tx - x) + (ty - y) * (ty - y));
          const float dw = std::fmin(std::fmin(hx - x, x + hx), std::fmin(hy - y, y + hy));
          ++(dc < dw ? near_cup : near_wall);
          if (dw < 0.45f && dc < 0.6f) ++both;
        }
        ++hist[st == 0 ? 0 : st < 3 ? 1 : st < 10 ? 2 : st < 30 ? 3 : 4];
      }
    }
    std::printf("eval collisions by episode step: 0: %ld, 1-2: %ld, 3-9: %ld, 10-29: %ld, >=30: %ld (of %ld episodes: success %ld, collision %ld, timeout %ld)\n",
                hist[0], hist[1], hist[2], hist[3], hist[4], ends[1] + ends[2] + ends[3], ends[1], ends[2], ends[3]);
    std::printf("eval collisions: cup closer than wall %ld, wall closer %ld; cup within 0.6 m AND wall within 0.45 m %ld\n", near_cup, near_wall, both);
  }
  std::printf("eval %s stage A%d deterministic: %.0f episodes, success %.4f collision %.4f timeout %.4f, mean len %.1f return %.2f\n", path, stage, ne,
              s / ne, cl / ne, to / ne, len / ne, ret / ne);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: ppo_verify v4|v5|v6|v7|bench [--negative]\n"); return 2; }
  const std::string m = argv[1];
  bool neg = false;
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--negative")) neg = true;
  if (m == "bench") return run_bench(argc > 2 ? std::atoi(argv[2]) : 4096, argc > 3 ? std::atoi(argv[3]) : 64, argc > 4 ? std::atoi(argv[4]) : 20,
                                     argc > 5 ? std::atoi(argv[5]) : 4, argc > 6 ? std::atoi(argv[6]) : 1);
  if (m == "v4" || m == "v5") {
    if (!neg) return (m == "v4" ? run_v4(0) : run_v5(0)) ? 1 : 0;
    int caught = 0;
    for (int bug : {1, 2}) {
      const int f = m == "v4" ? run_v4(bug) : run_v5(bug);
      std::printf("negative control bug %d: %d failing checks (must be > 0)\n", bug, f);
      caught += f > 0;
    }
    return caught == 2 ? 0 : 1;
  }
  if (m == "v6") return run_v6();
  if (m == "eval" && argc > 3) return run_eval(argv[2], std::atoi(argv[3]), argc > 4 ? std::atoi(argv[4]) : 10, argc > 5 ? std::atoi(argv[5]) : 4096,
                                               argc > 6 ? std::atoi(argv[6]) : 1);
  if (m == "v7") return run_v7();
  return 2;
}
