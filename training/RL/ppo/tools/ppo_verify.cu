// 검증 사다리 V4–V7 (계획서 GPU_TRAINING.md 9절).
//   ppo_verify v4 [--negative]   층 하나씩: GPU 가 실제로 받은 입력으로 CPU FP64(정답)·EMUL(바닥)을 돌려 GPU 출력과 비교. 기준 GPU 오차 ≤ 2 × 바닥
//   (v4·v5 의 자료는 G3 설정(특권 목표·빈 지도) — 신경망 수치 검사의 기준. --g4data 면 G4 자료(지도 목표·처음 지도 섞음, 칸이 더 참):
//    1 스텝 검사는 같이 통과하나 Adam 10 스텝·FP64 유한 차분은 씨앗에 따라 흔들림(README))
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
#include <cstddef>
#include <string>
#include <vector>

#include "net_ref.h"
#include "obs.h"
#include "trainer.h"
#include "slot_fused.cuh"

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

static bool g_g4data = false;   // v4/v5 --g4data: G4 자료(목표는 지도에서, 처음 지도 섞음)로. v6/v7 은 늘 G4 자료
static bool g_a2 = false;       // --a2: A2 환경(가구) + 지도 토큰에 안 본 곳 광선(use_map 2). v4/v5 는 --g4data 와 같이
static bool g_act8 = false;     // --act8: 행동 8 모두 학습(act_mask 0xff — 팔·그리퍼 포함, VLA_INPUT 5절)
static bool g_aug = false;      // --aug: 학습 때 흔들기 켬(속도 잡음·직전 명령·이름 흔들기·칸 지우기·지도 끄기, obs.h ObsAug)
static bool g_beh = false;      // --beh: BEHAVIOR 장면 묶음을 만들어 환경·지도에 붙임(상자 방 단계면 결과 비트가 같아야 함). 단계 3 이면 늘 켬
static int g_stage = -1;        // --stage S: v6/v7 의 환경 단계(−1 = 설정 그대로). 3 = BEHAVIOR B1–B3(장면 묶음 ~/ra_b1k)
static bool g_curr = false;     // --curr: v6/v7 에 장치 커리큘럼(문턱 0): A1 → env 3(장치에서 환경 바꾸기) → env 3 다른 B 섞음(같은 환경) → A2 → env 3
static PpoConfig small_cfg(uint64_t seed, int graphs, bool g4 = true) {
  PpoConfig c{};
  c.n_env = 512; c.horizon = 16; c.epochs = 2; c.minibatches = 4; c.stage = 1; c.use_map = 1; c.adaptive_lr = 0; c.use_graphs = graphs;
  c.log_ring = 8; c.dw_chunk = 1024; c.seed = seed;
  c.gamma = 0.99f; c.lam = 0.95f; c.clip = 0.2f; c.vclip = 0.2f; c.vf_coef = 0.5f; c.ent_coef = 0.001f;
  c.lr = 3e-4f; c.lr_min = 1e-5f; c.lr_max = 1e-3f; c.kl_target = 0.01f; c.max_grad_norm = 1.0f;
  c.adam_b1 = 0.9f; c.adam_b2 = 0.999f; c.adam_eps = 1e-8f; c.init_logstd = -0.5f; c.reward_scale = 1.f;
  c.act_dims = 2; c.shape_coef = 1.f; c.shape_near = 8.f; c.shape_aim = 1.f; c.shape_zone = 1.f; c.shape_v = 4.f; c.shape_w = 2.f;
  // G4 경로를 지나게: 목표는 지도에서, 처음 지도 C0·C1·C2 섞음
  c.goal_from_map = 1; c.map_p0 = 0.3f; c.map_p1 = 0.4f; c.map_kmin = 3; c.map_kmax = 6; c.map_reveal_r = 1.5f;
  c.bound_coef = 0.5f;   // 자르기 밖 평균 벌(G4)도 검사에 넣음 — V4/V5 의 흔든 미니배치에서 |μ| > 1 인 행이 있게 아래에서 평균 출력을 키움
  if (!g4) { c.goal_from_map = 0; c.map_p0 = 0.f; c.map_p1 = 0.f; }   // G3 설정(특권 목표, 빈 지도) — V4/V5 기준 자료
  else if (g_a2) { c.stage = 2; c.use_map = 2; }
  if (g_act8) c.act_mask = 0xffu;
  if (g_stage >= 0) c.stage = g_stage;
  if (g_curr) { c.stage = 1; c.beh = 1; c.use_map = 2; c.env_stages = (1u << 1) | (1u << 2) | (1u << 3); }
  if (std::getenv("PPO_ALLFAM")) { c.beh = 1; c.env_stages = (uint32_t)std::strtoul(std::getenv("PPO_ALLFAM"), nullptr, 0) | (1u << c.stage); }   // 측정: 환경 커널 무리 셋을 모두 띄움(장치 단계 판의 빈 커널 비용)
  if (g_beh || c.stage >= 3) {   // BEHAVIOR: 기본 섞음(B1 0.34·B2 0.33·B3), 모든 장면, 학습 인스턴스
    c.beh = 1;
    c.bcurr = PpoBCurr{0.34f, 0.33f, 0u, 0, 0.f, 0, 0, 0};
    if (c.stage >= 3) c.use_map = 2;
  }
  if (g_aug) {
    c.aug_on = 1; c.aug_vel_sigma = 0.05f; c.aug_prev_drop = 0.1f; c.aug_prev_sigma = 0.05f;
    c.aug_p_erase = 0.1f; c.aug_p_syn = 0.2f; c.aug_p_hyper = 0.1f; c.aug_p_wrong = 0.05f; c.aug_p_slot_drop = 0.1f; c.aug_p_map_off = 0.05f;
  }
  c.fp8 = std::getenv("NET_FP8") ? std::atoi(std::getenv("NET_FP8")) : 0;   // G6: 몸통 층 FP8 비트(1 앞, 2 dgrad, 4 wgrad)
  netref::set_fp8(c.fp8);                                                    // CPU 흉내(바닥)도 같은 FP8 처방
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
  {   // 정책 평균 층(A4)을 롤아웃 전에 키워 |μ| > 1 인 행이 생기게(자르기 밖 평균 벌이 검사에 들어가도록). 초기 A4 는 gain 0.01 이라 |μ| ≪ 1
    const ParamLayout lay = param_layout();
    std::vector<float> P = dl(tr.P, lay.total);
    for (long long j = lay.off[L_A4]; j < lay.off[L_A4] + (long long)kLayers[L_A4].N * kLayers[L_A4].K; ++j) P[j] *= 60.f;
    VCK(cudaMemcpy(tr.P, P.data(), sizeof(float) * lay.total, cudaMemcpyHostToDevice));
    to_bf16(tr.P, tr.Pb, lay.total, 0);
  }
  for (int k = 0; k < 2; ++k) tr.iterate();
  VCK(cudaDeviceSynchronize());
  PpoLog L;
  while (tr.poll(&L)) {}
  tr.rollout_body();
  tr.gae();
  tr.keep_slot_bufs = true;   // sin·dZ S2·dZ S1 을 전역에도(층마다 CPU 와 비교)
  tr.gather(0, 0);
  VCK(cudaDeviceSynchronize());
  const int M = tr.MB;
  netref::Batch b;
  b.M = M;
  b.x0 = dl(tr.x0, (size_t)M * X0_W);
  b.sin = dl(tr.sin_full(M), (size_t)M * KSLOT * SLOT_IN);   // 줄인 칸 줄을 펼친 값(펼치기 = 묶음 커널과 같은 식)
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
  hy = netref::Hyper{tr.cfg.clip, tr.cfg.vclip, tr.cfg.vf_coef, tr.cfg.ent_coef, dl(tr.ts, 1)[0].act_mask, tr.cfg.bound_coef};
  {
    tr.forward(M);
    VCK(cudaDeviceSynchronize());
    const std::vector<float> mu = dl(tr.mean, (size_t)M * N_ACT);
    long nb = 0;
    for (int i = 0; i < M; ++i) for (int k = 0; k < tr.cfg.act_dims; ++k) nb += std::fabs(mu[(size_t)i * N_ACT + k]) > 1.f;
    std::printf("batch: policy mean beyond the action clip (|mu| > 1, bound loss coef %.2f active): %.3f of entries\n", tr.cfg.bound_coef,
                (double)nb / ((double)M * tr.cfg.act_dims));
  }
  long nslot = 0;
  for (int i = 0; i < M; ++i) nslot += __builtin_popcount(b.mask[i]);
  std::printf("batch: M=%d rows, mean filled map slots %.2f, adv mean %.3f std %.3f\n", M, (double)nslot / M, b.adv_mean, b.adv_std);
  return b;
}

static int run_v4(int bug) {
  std::printf("== V4: layer forward/backward vs CPU FP64 (floor = CPU bf16 emulation)%s\n", bug ? "  [NEGATIVE CONTROL]" : "");
  Trainer tr(small_cfg(11, 0, g_g4data));
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
    std::vector<double> fa((size_t)M * 8), fc((size_t)M * 8), ea((size_t)M * 8), ec((size_t)M * 8), fl(8), el(8), st(5);
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
  Trainer tr(small_cfg(std::getenv("V5_SEED") ? (uint64_t)std::atoll(std::getenv("V5_SEED")) : 23, 0, g_g4data));
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
        // 중심 차분. 어긋나면 h 를 1/10, 1/100 로 다시(손실의 꺾인 곳 — 집합 최댓값이 바뀌는 자리, PPO·가치 자르기 — 을 ±h 가 건너면
        // 차분이 틀림. 진짜 기울기 버그는 h 를 줄여도 남는다). 가장 작은 오차를 씀
        const double g = st.grad[j], p = Pd[j];
        double err = 1e30, fd = 0;
        for (int r = 0; r < 3 && err > 1e-4; ++r) {
          const double h = 1e-5 * std::fmax(1.0, std::fabs(p)) * (r == 0 ? 1.0 : r == 1 ? 0.1 : 0.01);
          Pd[j] = p + h;
          const double lp = netref::loss_only(Pd, sb, hy);
          Pd[j] = p - h;
          const double lm = netref::loss_only(Pd, sb, hy);
          Pd[j] = p;
          const double f = (lp - lm) / (2 * h), e = std::fabs(f - g) / std::fmax(std::fabs(g), 1e-7);
          if (e < err) { err = e; fd = f; }
          if (r > 0 && e <= 1e-4) std::printf("    FD param %lld (layer %d): ok at h/%d, rel err %.2e (at h the difference crossed a kink)\n", j, l, r == 1 ? 10 : 100, e);
        }
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
  std::vector<unsigned long long> tab;
  std::vector<PpoLog> logs;
};
static void snap_fill(Trainer& tr, Snap& s);
static long diff_snap(const Snap& a, const Snap& b, bool verbose);
// --curr 의 단계 표(문턱 0, 창 1 → 바퀴마다 넘어감): 환경 바꾸기 셋(장치), 같은 환경 B 섞음 바꾸기 하나
static void set_test_curr(Trainer& tr) {
  PpoCurrStage st[5] = {};
  const PpoBCurr b1{1.f, 0.f, 0u, 0, 0.f, 0, 0, 0}, b23{0.f, 0.5f, 0u, 0, 0.f, 0, 0, 0};
  st[0] = PpoCurrStage{1, 0.3f, 0.4f, 0.f, -1, 0u, 0, PpoBCurr{}};
  st[1] = PpoCurrStage{3, 1.0f, 0.f, 0.f, -1, 0u, 1, b1};
  st[2] = PpoCurrStage{3, 0.1f, 0.1f, 0.f, -1, 0u, 1, b23};
  st[3] = PpoCurrStage{2, 0.3f, 0.4f, 0.f, -1, 0u, 0, PpoBCurr{}};
  st[4] = PpoCurrStage{3, 0.3f, 0.4f, 2.f, -1, 0u, 1, PpoBCurr{0.34f, 0.33f, 0u, 0, 0.f, 0, 0, 0}};
  const uint32_t all = (1u << tr.scenes->host.nsc) - 1u;
  for (auto& q : st) if (q.b_set) q.bcurr.scene_mask = all;
  if (ppo_curr_set(&tr, st, 5, 1, 0) != 0) { std::fprintf(stderr, "curr_set failed\n"); std::exit(2); }
}
static Snap run_iters(PpoConfig c, int iters) {
  Trainer tr(c);
  if (g_curr) set_test_curr(tr);
  Snap s;
  for (int k = 0; k < iters; ++k) {
    tr.iterate();
    VCK(cudaDeviceSynchronize());
    PpoLog L;
    while (tr.poll(&L)) { L.rollout_ms = L.update_ms = 0; s.logs.push_back(L); }
  }
  snap_fill(tr, s);
  return s;
}
static void snap_fill(Trainer& tr, Snap& s) {
  const long long n = tr.lay.total;
  s.P = dl(tr.P, n); s.m = dl(tr.Am, n); s.v = dl(tr.Av, n);
  s.obs = dl(tr.obs_buf, (size_t)(tr.T + 1) * N_OBS_G1 * tr.N);
  s.adv = dl(tr.adv_buf, (size_t)tr.T * tr.N);
  s.val = dl(tr.val_buf, (size_t)(tr.T + 1) * tr.N);
  s.act = dl(tr.act_buf, (size_t)tr.T * tr.N * N_ACT);
  s.tok = dl(reinterpret_cast<const uint8_t*>(tr.tok->at(0)), sizeof(gmap::MapTok) * (size_t)(tr.T + 1) * tr.N);
  s.tab = dl(tr.tab, (size_t)3 * 2 * 10 * 6);
}

// 장치 단계 바꾸기 == 다시 만들기(예전 ppo_set_stage): 같은 설정으로 2 바퀴 → (가) request_stage(장치 값, 다음 롤아웃 앞에서 적용)
// (나) 동기 + make_env(환경·지도 새로 만들기) + 그래프 다시 잡기 → 둘 다 4 바퀴 더. 변수·Adam·버퍼·토큰·기록이 비트로 같아야 한다
static int run_switch(int N, int T, bool neg) {
  std::printf("== switch: device set_stage == recreate (env + map + rollout tail), N %d T %d\n", N, T);
  const int cs[][2] = {{1, 3}, {3, 2}, {2, 1}, {3, 3}, {0, 3}, {3, 0}};
  int fails = 0;
  for (const auto& q : cs) {
    PpoConfig c = small_cfg(53, 1);
    c.n_env = N; c.horizon = T; c.stage = q[0]; c.use_map = 2; c.beh = 1;
    c.bcurr = PpoBCurr{0.34f, 0.33f, 0u, 0, 0.f, 0, 0, 0};
    c.env_stages = (1u << q[0]) | (1u << q[1]);
    Snap r[2];
    for (int dev = 0; dev < 2; ++dev) {
      Trainer tr(c);
      PpoLog L;
      for (int k = 0; k < 2; ++k) { tr.iterate(); VCK(cudaDeviceSynchronize()); while (tr.poll(&L)) {} }
      if (dev && neg) {   // 음성 대조: 환경만 다시 시작(지도·롤아웃 끝 줄은 그대로) — 반드시 달라야 함
        if (tr.env->request_stage(q[1], tr.env_seed(q[1])) != 0) return 1;
      } else if (dev) {
        if (tr.request_stage(q[1]) != 0) { std::printf("  request_stage failed\n"); return 1; }
      } else {
        VCK(cudaDeviceSynchronize());
        tr.make_env(q[1]);
        VCK(cudaDeviceSynchronize());
        tr.capture();
      }
      for (int k = 0; k < 4; ++k) {
        tr.iterate();
        VCK(cudaDeviceSynchronize());
        while (tr.poll(&L)) { L.rollout_ms = L.update_ms = 0; r[dev].logs.push_back(L); }
      }
      snap_fill(tr, r[dev]);
    }
    const long d = diff_snap(r[0], r[1], false);
    const PpoLog& L = r[1].logs.back();
    std::printf("  env %d -> %d: %s (%ld differing words; last log stage %d, eps %.0f, B1/B2/B3 %.0f/%.0f/%.0f)\n", q[0], q[1], d ? "FAIL" : "same", d, L.stage, L.n_eps,
                L.n_b[0], L.n_b[1], L.n_b[2]);
    if (d) { if (!neg) diff_snap(r[0], r[1], true); ++fails; }
  }
  if (neg) {
    std::printf("switch negative control (map/tail not restarted): %d / 6 cases differ (must be 6)\n", fails);
    return fails == 6 ? 0 : 1;
  }
  std::printf("switch: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
static long diff_snap(const Snap& a, const Snap& b, bool verbose) {
  auto cnt = [](const void* x, const void* y, size_t nb) { long d = 0; const uint8_t *p = (const uint8_t*)x, *q = (const uint8_t*)y; for (size_t i = 0; i < nb; i += 4) d += std::memcmp(p + i, q + i, 4) != 0; return d; };
  struct { const char* n; long d; } t[] = {
      {"params", cnt(a.P.data(), b.P.data(), 4 * a.P.size())}, {"adam m", cnt(a.m.data(), b.m.data(), 4 * a.m.size())},
      {"adam v", cnt(a.v.data(), b.v.data(), 4 * a.v.size())}, {"obs buffer", cnt(a.obs.data(), b.obs.data(), 4 * a.obs.size())},
      {"advantages", cnt(a.adv.data(), b.adv.data(), 4 * a.adv.size())}, {"values", cnt(a.val.data(), b.val.data(), 4 * a.val.size())},
      {"actions", cnt(a.act.data(), b.act.data(), 4 * a.act.size())}, {"map tokens", cnt(a.tok.data(), b.tok.data(), a.tok.size())},
      {"logs", a.logs.size() == b.logs.size() ? cnt(a.logs.data(), b.logs.data(), sizeof(PpoLog) * a.logs.size()) : 1},
      {"episode table", cnt(a.tab.data(), b.tab.data(), 8 * a.tab.size())}};
  long tot = 0;
  for (auto& e : t) {
    tot += e.d;
    if (verbose) std::printf("  %-12s %ld differing words\n", e.n, e.d);
  }
  return tot;
}

static int run_v6() {
  std::printf("== V6: CUDA graphs == eager (bit-identical after 10 iterations)\n");
  const Snap g = run_iters(small_cfg(31, 1), 10), e = run_iters(small_cfg(31, 0), 10);   // 10 바퀴 = 160 스텝: 시간 끝 에피소드까지 지남
  const long d = diff_snap(g, e, true);
  const PpoLog& L = g.logs.back();
  double n0 = 0, n1 = 0, n2 = 0, ne = 0;
  for (const PpoLog& q : g.logs) { n0 += q.n_c[0]; n1 += q.n_c[1]; n2 += q.n_c[2]; ne += q.n_eps; }
  unsigned long long tn = 0;
  for (size_t k = 0; k < g.tab.size(); k += 6) tn += g.tab[k];
  if (g_curr) { std::printf("  env stage per iteration:"); for (const PpoLog& q : g.logs) std::printf(" %d", q.stage); std::printf("\n"); }
  std::printf("V6: %s (%ld differing words); last log: succ %.3f kl %.5f goal known %.3f; episodes %.0f, by first map C0/C1/C2 %.0f/%.0f/%.0f, table %llu\n",
              d ? "FAIL" : "PASS", d, L.succ, L.kl, L.goal_known, ne, n0, n1, n2, tn);
  return d ? 1 : 0;
}
static int run_v7() {
  std::printf("== V7: determinism (same seed twice, graphs)\n");
  const Snap a = run_iters(small_cfg(41, 1), 10), b = run_iters(small_cfg(41, 1), 10), c = run_iters(small_cfg(42, 1), 10);
  const long d = diff_snap(a, b, true), dc = diff_snap(a, c, false);
  std::printf("V7: same seed %s (%ld differing words); different seed differs in %ld words (%s)\n", d ? "FAIL" : "PASS", d, dc, dc ? "ok" : "FAIL");
  return (d || !dc) ? 1 : 0;
}

// 비트 동일 비교(빌드 사이): 학습 크기로 몇 바퀴 돈 뒤 변수·Adam·버퍼·기록을 파일로. 두 빌드의 파일을 cmp 로 견준다(갱신 최적화 전후).
//   ppo_verify snap <out> [N=4096] [T=64] [iters=6] [stage=1] [use_map=1]
static int run_snap(const char* out, int N, int T, int iters, int stage, int use_map) {
  PpoConfig c = small_cfg(7, 1);
  c.n_env = N; c.horizon = T; c.minibatches = 4; c.epochs = 5; c.use_map = use_map; c.adaptive_lr = 1; c.stage = stage;
  if (stage >= 3 && !c.beh) { c.beh = 1; c.bcurr = PpoBCurr{0.34f, 0.33f, 0u, 0, 0.f, 0, 0, 0}; }
  const Snap s = run_iters(c, iters);
  FILE* f = std::fopen(out, "wb");
  if (!f) { std::perror(out); return 2; }
  uint64_t h = 1469598103934665603ull;
  auto put = [&](const void* p, size_t nb) {
    std::fwrite(p, 1, nb, f);
    for (size_t i = 0; i < nb; ++i) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
  };
  put(s.P.data(), 4 * s.P.size()); put(s.m.data(), 4 * s.m.size()); put(s.v.data(), 4 * s.v.size());
  put(s.obs.data(), 4 * s.obs.size()); put(s.adv.data(), 4 * s.adv.size()); put(s.val.data(), 4 * s.val.size()); put(s.act.data(), 4 * s.act.size());
  put(s.tok.data(), s.tok.size()); put(s.tab.data(), 8 * s.tab.size());
  // 옛 배치 해시(빌드 사이 견주기): 기록은 E2 앞 PpoLog 칸(goal_known 까지)만 — 같은 상자 방 판이면 E2 앞 기준 해시와 같아야 한다
  uint64_t hl = h;
  for (const PpoLog& q : s.logs)
    for (size_t i = 0; i < offsetof(PpoLog, n_b); ++i) hl = (hl ^ ((const uint8_t*)&q)[i]) * 1099511628211ull;
  put(s.logs.data(), sizeof(PpoLog) * s.logs.size());
  std::fclose(f);
  const PpoLog& L = s.logs.back();
  std::printf("snap %s: N %d T %d iters %d A%d use_map %d%s -> FNV-1a %016llx  legacy-layout %016llx (last log: kl %.6f succ %.4f ep_ret %.4f, B1/B2/B3 eps %.0f/%.0f/%.0f)\n",
              out, N, T, iters, stage, use_map, c.beh ? " beh" : "", (unsigned long long)h, (unsigned long long)hl, L.kl, L.succ, L.ep_ret, L.n_b[0], L.n_b[1], L.n_b[2]);
  return 0;
}

#ifdef SK_PROF
namespace net { void slot_prof_print(); }
#endif
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
#ifdef SK_PROF
  net::slot_prof_print();   // 칸 MLP 묶음 커널 구간(측정 빌드 -DSK_PROF 만)
#endif
  return 0;
}

// 체크포인트를 결정적 정책(σ → e^-12, 학습률 0)으로 돌려 성공률을 잰다(계획서 5.6: 처음 완성도로 나눔).
//   ppo_verify eval <ckpt> <stage> [iters=10] [N=4096] [use_map=1] [goal_from_map=1] [p0 p1 kmin kmax]
// 처음 완성도 = 판 시작 때 미리 확정한 참 물체 비율(9 개 중). 칸: 0 %, 0–30 %(1–2 개), 30–70 %(3–6), 70–100 %(7–8), 100 %(9)
struct CapHook {   // 충돌 다시 돌리기: 서브스텝마다 자세를 남김(마지막 = 충돌한 자세)
  env::Core* out;
  __host__ __device__ void operator()(const env::Core& c) const { *out = c; }
};
static int run_eval(const char* path, int stage, int iters, int N, int use_map, int goal, const float* mc) {
  PpoConfig c = small_cfg(1234, 1);
  c.n_env = N; c.horizon = 64; c.minibatches = 4; c.epochs = 1; c.stage = stage; c.use_map = use_map; c.lr = 0.f; c.lr_max = 0.f; c.lr_min = 0.f;
  c.adaptive_lr = 0; c.goal_from_map = goal;
  c.map_p0 = mc[0]; c.map_p1 = mc[1]; c.map_kmin = (int)mc[2]; c.map_kmax = (int)mc[3]; c.map_reveal_r = 1.5f;
  Trainer tr(c);
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::perror(path); return 2; }
  std::vector<uint8_t> b(64 + sizeof(float) * 3 * tr.lay.total + sizeof(TrainState));
  const size_t got = std::fread(b.data(), 1, b.size(), f);
  std::fclose(f);
  float* P = reinterpret_cast<float*>(b.data() + 64);
  for (int k = 0; k < N_ACT; ++k) P[tr.lay.logstd + k] = -12.f;
  if (got != b.size() || ppo_load(&tr, b.data(), (int64_t)b.size())) { std::fprintf(stderr, "bad checkpoint\n"); return 2; }
  std::printf("eval %s: A%d use_map %d goal_from_map %d, first map C0 %.2f C1 %.2f (k %d..%d) C2 %.2f, N %d, %d iters (first 3 dropped)\n", path, stage, use_map,
              goal, mc[0], mc[1], (int)mc[2], (int)mc[3], 1.f - mc[0] - mc[1], N, iters);
  double s = 0, cl = 0, to = 0, ne = 0, len = 0, ret = 0, gk = 0;
  int nk = 0;
  for (int k = 0; k < iters; ++k) {
    tr.eval_iterate();   // 갱신 없음(체크포인트의 학습률이 되살아나도 변수는 그대로)
    VCK(cudaDeviceSynchronize());
    if (k == 2) VCK(cudaMemset(tr.tab, 0, sizeof(unsigned long long) * 3 * 2 * 10 * 6));   // 처음 바퀴들(무작위 시작 판)은 버림
    PpoLog L;
    while (tr.poll(&L)) {
      if (std::getenv("EVAL_TRACE")) std::printf("  eval iter %d: episodes %.0f success %.4f collision %.4f timeout %.4f len %.1f goal known %.3f\n", k, L.n_eps, L.succ, L.coll, L.tout, L.ep_len, L.goal_known);
      if (k < 3) continue;
      s += L.succ * L.n_eps; cl += L.coll * L.n_eps; to += L.tout * L.n_eps; ne += L.n_eps; len += L.ep_len * L.n_eps; ret += L.ep_ret * L.n_eps;
      gk += L.goal_known; ++nk;
    }
  }
  std::printf("eval overall: %.0f episodes, success %.4f collision %.4f timeout %.4f, mean len %.1f return %.2f; goal known at rollout end %.3f\n", ne, s / ne,
              cl / ne, to / ne, len / ne, ret / ne, nk ? gk / nk : 0.0);
  {   // 처음 완성도 칸(5.6)
    const std::vector<unsigned long long> T = dl(tr.tab, (size_t)3 * 2 * 10 * 6);
    auto bucket = [](int cnum) { return cnum == 0 ? 0 : cnum <= 2 ? 1 : cnum <= 6 ? 2 : cnum <= 8 ? 3 : 4; };
    static const char* bn[5] = {"0 %", "0-30 %", "30-70 %", "70-100 %", "100 %"};
    double A[3][5][6] = {};   // [전체·컵 모름·컵 앎][칸][량]
    for (int st = 0; st < 3; ++st)
      for (int g = 0; g < 2; ++g)
        for (int cn = 0; cn < 10; ++cn)
          for (int q = 0; q < 6; ++q) {
            const double v = (double)T[(size_t)((st * 2 + g) * 10 + cn) * 6 + q];
            A[0][bucket(cn)][q] += v;
            A[1 + g][bucket(cn)][q] += v;
          }
    static const char* gn[3] = {"all", "cup NOT pre-confirmed", "cup pre-confirmed"};
    for (int g = 0; g < 3; ++g) {
      std::printf("eval by initial completeness (%s):\n  %-9s %8s %8s %8s %8s %10s %12s\n", gn[g], "bucket", "episodes", "success", "collide", "timeout",
                  "steps(all)", "steps(succ)");
      for (int k = 0; k < 5; ++k) {
        const double n = A[g][k][0];
        if (n == 0) { std::printf("  %-9s %8s\n", bn[k], "-"); continue; }
        std::printf("  %-9s %8.0f %8.4f %8.4f %8.4f %10.1f %12.1f\n", bn[k], n, A[g][k][1] / n, A[g][k][2] / n, A[g][k][3] / n, A[g][k][4] / n,
                    A[g][k][1] > 0 ? A[g][k][5] / A[g][k][1] : 0.0);
      }
    }
  }
  // 충돌 다시 보기(스텝마다 동기 — 평가 전용): 에피소드 몇 번째 스텝, 컵·벽, 충돌한 서브스텝 자세에서 방 밖으로 나간 모서리(앞·뒤),
  // 그때 속도·각속도, 목표를 알았나
  {
    std::vector<int> iv, done;
    std::vector<float> fs, act;
    std::vector<uint64_t> rg;
    long hist[5] = {0, 0, 0, 0, 0}, ends[4] = {0, 0, 0, 0}, near_cup = 0, near_wall = 0, both = 0;
    long kind[6] = {0, 0, 0, 0, 0, 0};   // 컵, 벽 앞 모서리(앞으로), 벽 앞 모서리(뒤로·제자리), 벽 뒤 모서리(뒤로), 벽 뒤 모서리(앞으로·제자리), 가구(A2)
    long known_at = 0, turning = 0, cmd_back = 0, zone_close = 0;
    long sat_all[2] = {0, 0}, sat_hit[2] = {0, 0}, n_all = 0;   // 정책 평균 |μ| > 1(행동 자르기 밖) — 모든 스텝 / 충돌 스텝
    double mu_hit[2] = {0, 0};
    double sum_v = 0, sum_w = 0, sum_minray = 0, sum_surf = 0;
    for (int k = 0; k < 4 * tr.T; ++k) {
      const int t = k % tr.T;
      if (t == 0 && k == 0) { VCK(cudaMemcpy(tr.obs_buf, tr.obs_buf + (size_t)tr.T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
                              VCK(cudaMemcpy(tr.obs_rows, tr.obs_rows + (size_t)tr.T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
                              VCK(cudaMemcpy(tr.tok->at(0), tr.tok->at(tr.T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToDevice)); }
      tr.env->download(fs, iv, rg);
      const std::vector<gmap::MapTok> tk = dl(tr.tok->at(t), N);
      const std::vector<float> ob = dl(tr.obs_buf + (size_t)t * N_OBS_G1 * N, (size_t)N_OBS_G1 * N);
      tr.rollout_step(t);
      VCK(cudaDeviceSynchronize());
      done = dl(tr.done_buf + (size_t)t * N, N);
      act = dl(tr.act_env, (size_t)N_ACT * N);
      const std::vector<float> mu = dl(tr.mean, (size_t)N * N_ACT);
      for (int i = 0; i < N; ++i) for (int q = 0; q < 2; ++q) sat_all[q] += std::fabs(mu[(size_t)i * N_ACT + q]) > 1.f;
      n_all += N;
      if (t == tr.T - 1) { VCK(cudaMemcpy(tr.obs_buf, tr.obs_buf + (size_t)tr.T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
                           VCK(cudaMemcpy(tr.obs_rows, tr.obs_rows + (size_t)tr.T * N_OBS_G1 * N, sizeof(float) * N_OBS_G1 * N, cudaMemcpyDeviceToDevice));
                           VCK(cudaMemcpy(tr.tok->at(0), tr.tok->at(tr.T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToDevice)); }
      for (int i = 0; i < N; ++i) {
        if (!done[i]) continue;
        ++ends[done[i]];
        if (done[i] != env::kCollision) continue;
        const int st = iv[env::I_STEP * N + i];   // 이 스텝 전 에피소드 스텝 수
        const float x = fs[env::F_X * N + i], y = fs[env::F_Y * N + i], tx = fs[env::F_TX * N + i], ty = fs[env::F_TY * N + i];
        const float hx = fs[env::F_RHX * N + i], hy = fs[env::F_RHY * N + i];
        const float dc = std::sqrt((tx - x) * (tx - x) + (ty - y) * (ty - y));
        const float dw = std::fmin(std::fmin(hx - x, x + hx), std::fmin(hy - y, y + hy));
        ++(dc < dw ? near_cup : near_wall);
        if (dw < 0.45f && dc < 0.6f) ++both;
        ++hist[st == 0 ? 0 : st < 3 ? 1 : st < 10 ? 2 : st < 30 ? 3 : 4];
        // 같은 스텝을 호스트에서 다시(env.h 같은 소스): 충돌한 서브스텝 자세
        env::Core cc;
        env::Soa hs{fs.data(), iv.data(), rg.data(), N};
        env::load(hs, i, cc);
        float a[env::N_ACT];
        for (int q = 0; q < env::N_ACT; ++q) a[q] = act[(size_t)q * N + i];
        env::Core hit = cc;
        env::StepOut so;
        env::step_core(cc, a, so, false, CapHook{&hit});
        float sn, cs;
        dm::sincosf_d(hit.yaw, &sn, &cs);
        int worst = -1;
        float wv = 0.f;
        for (int q = 0; q < 4; ++q) {
          const float lx = (q & 1) ? env::K::half_len : -env::K::half_len, ly = (q & 2) ? env::K::half_wid : -env::K::half_wid;
          const float wx = hit.x + cs * lx - sn * ly, wy = hit.y + sn * lx + cs * ly;
          const float over = std::fmax(std::fmax(wx - hit.rhx, -hit.rhx - wx), std::fmax(wy - hit.rhy, -hit.rhy - wy));
          if (over > 0.f && (worst < 0 || over > wv)) { worst = q; wv = over; }
        }
        const bool fwd = hit.v > 0.02f, back = hit.v < -0.02f;
        bool furn = false;   // A2: 가구 상자에 닿음(방 밖 모서리가 없을 때)
        for (int q = 0; q < hit.nf && worst < 0; ++q) {
          float b4[4];
          for (int z = 0; z < 4; ++z) b4[z] = hit.fu.box(q, z);
          furn = furn || env::body_hits_box(hit.x, hit.y, sn, cs, b4);
        }
        if (worst < 0) ++kind[furn ? 5 : 0];
        else if (worst & 1) ++kind[fwd ? 1 : 2];
        else ++kind[back ? 3 : 4];
        turning += std::fabs(hit.w) > 0.3f;
        cmd_back += a[0] < 0.f;
        known_at += obsv::goal_known(tk[i]);
        for (int q = 0; q < 2; ++q) { const float m_ = mu[(size_t)i * N_ACT + q]; sat_hit[q] += std::fabs(m_) > 1.f; mu_hit[q] += std::fabs(m_); }
        sum_v += hit.v; sum_w += std::fabs(hit.w);
        float mr = 4.f;
        for (int q = 0; q < env::N_RAYS; ++q) mr = std::fmin(mr, 4.f * ob[(size_t)(env::N_BODY + q) * N + i]);
        sum_minray += mr;
        const float surf = ob[(size_t)(env::N_BODY + env::N_RAYS + 5) * N + i];
        sum_surf += surf;
        zone_close += surf < 0.9f;
      }
    }
    const long nc = ends[2] > 0 ? ends[2] : 1;
    std::printf("eval collisions by episode step: 0: %ld, 1-2: %ld, 3-9: %ld, 10-29: %ld, >=30: %ld (of %ld episodes: success %ld, collision %ld, timeout %ld)\n",
                hist[0], hist[1], hist[2], hist[3], hist[4], ends[1] + ends[2] + ends[3], ends[1], ends[2], ends[3]);
    std::printf("eval collisions: cup closer than wall %ld, wall closer %ld; cup within 0.6 m AND wall within 0.45 m %ld\n", near_cup, near_wall, both);
    std::printf("eval collision replay: furniture %ld | cup %ld | front corner out: moving fwd %ld, not fwd %ld | rear corner out: reversing %ld, not reversing %ld\n",
                kind[5], kind[0], kind[1], kind[2], kind[3], kind[4]);
    std::printf("  at hit: mean v %.3f m/s, mean |w| %.3f rad/s, |w| > 0.3: %ld, commanded v < 0: %ld, goal known: %ld; previous obs: min wall ray %.3f m, cup surface %.3f m (< 0.9 m: %ld)\n",
                sum_v / nc, sum_w / nc, turning, cmd_back, known_at, sum_minray / nc, sum_surf / nc, zone_close);
    std::printf("  policy mean beyond the action clip |mu| > 1: all steps vx %.3f wz %.3f; collision steps vx %ld wz %ld of %ld (mean |mu| vx %.2f wz %.2f)\n",
                (double)sat_all[0] / n_all, (double)sat_all[1] / n_all, sat_hit[0], sat_hit[1], ends[2], mu_hit[0] / nc, mu_hit[1] / nc);
  }
  return 0;
}

// 관측 모으기 CPU == GPU(v2: 정규화·얼린 표·흔들기·행동 8): 롤아웃 끝 스텝 T 의 assemble 결과(X0·칸 줄·채운 칸 비트)를
// 같은 G1 관측·지도 토큰·표·열쇠로 CPU 가 다시 만들어 비트 비교. --negative: CPU 쪽 흔들기 열쇠의 스텝을 하나 틀리게(반드시 달라야 함)
// ---- 칸 줄 304 칸이 뜻한 자리에서 왔는가(VLA_INPUT 3절: [숫자 33 | 이름 표 행 128 | 생김새 표 행 128 | 1 | 0 × 14]) ----
// 실제 롤아웃 토큰에 칸마다 이름·생김새 표 행을 덮어씀(첫 행·끝 행·무작위·없음 섞음). 흔들기 끔. 다섯 길을 칸마다 뜻한 값과 견줌:
//   CPU assemble 304 줄 / CPU 줄인 줄 + slot_col / GPU assemble 304 줄 / GPU 줄인 줄 + slot_expand 커널 / GPU 줄인 줄 + 묶음 커널 펼치기(SlotTab 타일)
__global__ void slotcols_asm_k(const float* obs_row, const gmap::MapTok* tok, int N, uint16_t* x0, uint16_t* sin, uint16_t* sc, obsv::VecTab vt) {
  const int e = blockIdx.x * 8 + threadIdx.x / 16, lane = threadIdx.x % 16;
  if (e >= N) return;
  obsv::assemble(obs_row + (size_t)e * N_OBS_G1, 1, 0, tok[e], x0 + (size_t)e * X0_W, sin + (size_t)e * KSLOT * SLOT_IN, 1, 0, lane, 16, vt);
  __syncwarp();
  obsv::assemble(obs_row + (size_t)e * N_OBS_G1, 1, 0, tok[e], x0 + (size_t)e * X0_W, sc + (size_t)e * KSLOT * SLOT_C, 1, 0, lane, 16, vt, nullptr, 0, 0, true);
}
__global__ void __launch_bounds__(256) slotcols_tile_k(net::SlotTab g, int rows, uint16_t* out) {
  __shared__ __align__(128) uint16_t sC[net::SK_R * SLOT_C];
  __shared__ __align__(128) uint16_t sIn[net::SK_R * net::SK_LI];
  const int r0 = blockIdx.x * net::SK_R, tid = threadIdx.x;
  g.stage(sC, r0, rows, tid);
  net::cp_commit();
  net::cp_wait<0>();
  __syncthreads();
  net::SlotTab::Pre pre;
  g.issue(pre, sC, tid);
  g.finish(sIn, pre, sC, tid);
  __syncthreads();
  for (int q = tid; q < net::SK_R * SLOT_IN; q += 256) {
    const int r = q / SLOT_IN, c = q % SLOT_IN;
    if (r0 + r < rows) out[(size_t)(r0 + r) * SLOT_IN + c] = sIn[r * net::SK_LI + c];
  }
}
static int run_slotcols() {
  PpoConfig c = small_cfg(5, 0);
  ppo::Trainer tr(c);
  for (int k = 0; k < 2; ++k) tr.iterate();
  VCK(cudaDeviceSynchronize());
  PpoLog L;
  while (tr.poll(&L)) {}
  const int N = tr.N, T = tr.T, rows = N * KSLOT;
  const std::vector<float> obs = dl(tr.obs_rows + (size_t)T * N_OBS_G1 * N, (size_t)N_OBS_G1 * N);
  std::vector<gmap::MapTok> tok(N);
  VCK(cudaMemcpy(tok.data(), tr.tok->at(T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToHost));
  const obsv::VecTab vh = tr.vt.host();
  const int nn = vh.n_name, na = vh.n_app;
  // 표 행 고르기: 첫·끝 행을 자주, 나머지는 무작위, 가끔 없음(−1)
  uint64_t rs = 12345;
  long nfirst = 0, nlast = 0, nnone = 0, nlive = 0;
  for (int e = 0; e < N; ++e)
    for (int b = 0; b < KSLOT; ++b) {
      const int k = (int)(dm::rand01(rs) * 8);
      tok[e].name_id[b] = (int16_t)(k == 0 ? 0 : k == 1 ? nn - 1 : k == 2 ? -1 : (int)(dm::rand01(rs) * nn) % nn);
      const int ka = (int)(dm::rand01(rs) * 8);
      tok[e].app_id[b] = (int16_t)(ka == 0 ? 0 : ka == 1 ? na - 1 : ka == 2 ? -1 : (int)(dm::rand01(rs) * na) % na);
    }
  // 뜻한 값
  std::vector<uint16_t> want((size_t)rows * SLOT_IN, 0);
  std::vector<uint32_t> mks(N);
  {
    std::vector<uint16_t> x0(X0_W);
    for (int e = 0; e < N; ++e) {
      const uint32_t mk = obsv::assemble(obs.data() + (size_t)e * N_OBS_G1, 1, 0, tok[e], x0.data(), nullptr, 1, 0, 0, 1, vh);
      mks[e] = mk;
      for (int b = 0; b < KSLOT; ++b) {
        if (!((mk >> b) & 1u)) continue;
        ++nlive;
        uint16_t* w = &want[((size_t)e * KSLOT + b) * SLOT_IN];
        const int n = tok[e].name_id[b], a = tok[e].app_id[b];
        nfirst += (n == 0) + (a == 0); nlast += (n == nn - 1) + (a == na - 1); nnone += (n < 0) + (a < 0);
        for (int q = 0; q < SLOT_VALS; ++q) w[q] = f2bf(obsv::slot_num(tok[e], b, q));
        for (int q = 0; q < VEC_D; ++q) {
          w[SLOT_NAME + q] = n >= 0 ? vh.name[(size_t)n * VEC_D + q] : (uint16_t)0;
          w[SLOT_APP + q] = a >= 0 ? vh.app[(size_t)a * VEC_D + q] : (uint16_t)0;
        }
        w[SLOT_BIAS] = f2bf(1.f);
      }
    }
  }
  // CPU 두 길
  std::vector<uint16_t> cfull((size_t)rows * SLOT_IN, 0), cc((size_t)rows * SLOT_C, 0), cexp((size_t)rows * SLOT_IN, 0);
  {
    std::vector<uint16_t> x0(X0_W);
    for (int e = 0; e < N; ++e) {
      obsv::assemble(obs.data() + (size_t)e * N_OBS_G1, 1, 0, tok[e], x0.data(), &cfull[(size_t)e * KSLOT * SLOT_IN], 1, 0, 0, 1, vh);
      obsv::assemble(obs.data() + (size_t)e * N_OBS_G1, 1, 0, tok[e], x0.data(), &cc[(size_t)e * KSLOT * SLOT_C], 1, 0, 0, 1, vh, nullptr, 0, 0, true);
    }
    for (long long r = 0; r < rows; ++r)
      for (int q = 0; q < SLOT_IN; ++q) cexp[(size_t)r * SLOT_IN + q] = slot_col(&cc[(size_t)r * SLOT_C], vh.name, vh.app, q);
  }
  // GPU 세 길
  gmap::MapTok* dtok = nullptr;
  float* dobs = nullptr;
  uint16_t *dx0 = nullptr, *dsin = nullptr, *dsc = nullptr, *dexp = nullptr, *dtile = nullptr;
  VCK(cudaMalloc(&dtok, sizeof(gmap::MapTok) * N));
  VCK(cudaMalloc(&dobs, sizeof(float) * obs.size()));
  VCK(cudaMalloc(&dx0, 2ull * N * X0_W));
  VCK(cudaMalloc(&dsin, 2ull * rows * SLOT_IN));
  VCK(cudaMalloc(&dsc, 2ull * rows * SLOT_C));
  VCK(cudaMalloc(&dexp, 2ull * rows * SLOT_IN));
  VCK(cudaMalloc(&dtile, 2ull * rows * SLOT_IN));
  VCK(cudaMemcpy(dtok, tok.data(), sizeof(gmap::MapTok) * N, cudaMemcpyHostToDevice));
  VCK(cudaMemcpy(dobs, obs.data(), sizeof(float) * obs.size(), cudaMemcpyHostToDevice));
  slotcols_asm_k<<<(N + 7) / 8, 128>>>(dobs, dtok, N, dx0, dsin, dsc, tr.vt.dev());
  const net::SlotC sc{dsc, tr.vt.d_name, tr.vt.d_app};
  net::slot_expand(sc, rows, dexp, 0);
  slotcols_tile_k<<<(rows + net::SK_R - 1) / net::SK_R, 256>>>(net::SlotTab{dsc, tr.vt.d_name, tr.vt.d_app}, rows, dtile);
  VCK(cudaDeviceSynchronize());
  const std::vector<uint16_t> gfull = dl(dsin, (size_t)rows * SLOT_IN), gexp = dl(dexp, (size_t)rows * SLOT_IN), gtile = dl(dtile, (size_t)rows * SLOT_IN);
  for (void* p : {(void*)dtok, (void*)dobs, (void*)dx0, (void*)dsin, (void*)dsc, (void*)dexp, (void*)dtile}) cudaFree(p);
  std::printf("== slotcols: %d envs, %ld live slots (name/app rows: first %ld, last %ld, none %ld; tables %d / %d rows)\n", N, nlive, nfirst, nlast, nnone, nn, na);
  int fails = 0;
  auto cmp = [&](const char* nm, const std::vector<uint16_t>& got) {
    long bad[5] = {0, 0, 0, 0, 0};   // 숫자, 이름, 생김새, 편향, 0 꼬리
    int firstc = -1;
    for (size_t i = 0; i < want.size(); ++i)
      if (got[i] != want[i]) {
        const int q = (int)(i % SLOT_IN);
        ++bad[q < SLOT_NAME ? 0 : q < SLOT_APP ? 1 : q < SLOT_BIAS ? 2 : q == SLOT_BIAS ? 3 : 4];
        if (firstc < 0) firstc = q;
      }
    const long tot = bad[0] + bad[1] + bad[2] + bad[3] + bad[4];
    fails += tot != 0;
    std::printf("  %-34s %8ld wrong words (nums %ld, name %ld, app %ld, bias %ld, pad %ld)%s%s\n", nm, tot, bad[0], bad[1], bad[2], bad[3], bad[4],
                tot ? "  first bad col " : "", tot ? std::to_string(firstc).c_str() : "");
  };
  cmp("CPU assemble 304", cfull);
  cmp("CPU compact + slot_col", cexp);
  cmp("GPU assemble 304", gfull);
  cmp("GPU compact + slot_expand", gexp);
  cmp("GPU compact + SlotTab tile (fused)", gtile);
  std::printf("slotcols: %s (%d / 5 paths wrong)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}

static int run_obs(bool neg) {
  PpoConfig c = small_cfg(5, 0);
  ppo::Trainer tr(c);
  for (int k = 0; k < 2; ++k) tr.iterate();
  VCK(cudaDeviceSynchronize());
  PpoLog L;
  while (tr.poll(&L)) {}
  tr.rollout_body();   // 끝 스텝 T 의 assemble 이 x0·sin·mask 에 남음(갱신 전)
  VCK(cudaDeviceSynchronize());
  const int N = tr.N, T = tr.T;
  const std::vector<float> rows = dl(tr.obs_rows + (size_t)T * N_OBS_G1 * N, (size_t)N_OBS_G1 * N);
  std::vector<gmap::MapTok> tok(N);
  VCK(cudaMemcpy(tok.data(), tr.tok->at(T), sizeof(gmap::MapTok) * N, cudaMemcpyDeviceToHost));
  const std::vector<uint16_t> gx0 = dl(tr.x0, (size_t)N * X0_W), gsin = dl(tr.sin_full(N), (size_t)N * KSLOT * SLOT_IN);
  const std::vector<uint32_t> gmk = dl(tr.mask, N);
  const TrainState s = dl(tr.ts, 1)[0];
  obsv::ObsAug a = dl(tr.aug_d, 1)[0];
  const obsv::VecTab vt = tr.vt.host();
  std::vector<uint16_t> x0((size_t)X0_W, 0), sn((size_t)KSLOT * SLOT_IN, 0);
  long bad = 0, nslot = 0, nonzero_name = 0;
  for (int e = 0; e < N; ++e) {
    std::fill(x0.begin(), x0.end(), 0);
    const uint64_t k1 = ((uint64_t)(neg ? T - 1 : T) << 32) | (uint64_t)e;
    const uint32_t mk = obsv::assemble(rows.data() + (size_t)e * N_OBS_G1, 1, 0, tok[e], x0.data(), sn.data(), tr.cfg.use_map, tr.cfg.goal_from_map, 0, 1, vt, &a,
                                       (uint64_t)s.iter, k1);
    for (int q = X0_OBS; q < X0_W; ++q) if (x0[q] != gx0[(size_t)e * X0_W + q]) { if (bad < 5 && !neg) std::printf("  x0 env %d col %d cpu %04x gpu %04x raw %.4f\n", e, q, x0[q], gx0[(size_t)e * X0_W + q], q - X0_OBS < N_OBS_G1 ? rows[(size_t)e * N_OBS_G1 + q - X0_OBS] : 0.f); ++bad; }   // 집합 칸 [0, 128) 은 칸 MLP 뒤 커널이 씀
    for (int q = 0; q < KSLOT * SLOT_IN; ++q) if (sn[q] != gsin[(size_t)e * KSLOT * SLOT_IN + q] && bad < 8 && !neg) std::printf("  sin env %d slot %d col %d cpu %04x gpu %04x\n", e, q / SLOT_IN, q % SLOT_IN, sn[q], gsin[(size_t)e * KSLOT * SLOT_IN + q]);
    for (int q = 0; q < KSLOT * SLOT_IN; ++q) bad += sn[q] != gsin[(size_t)e * KSLOT * SLOT_IN + q];
    bad += mk != gmk[e];
    nslot += __builtin_popcount(mk);
    for (int b = 0; b < KSLOT; ++b)
      if ((mk >> b) & 1u)
        for (int q = net::SLOT_NAME; q < net::SLOT_APP; ++q) if (sn[(size_t)b * SLOT_IN + q]) { ++nonzero_name; break; }
  }
  std::printf("obs assemble CPU vs GPU (N %d, step %d, iter %lld, use_map %d, aug %d, act_mask 0x%x): %ld words differ; filled slots %.2f/env, slots with a name vector %.3f\n",
              N, T, (long long)s.iter, tr.cfg.use_map, a.on, s.act_mask, bad, (double)nslot / N, nslot ? (double)nonzero_name / nslot : 0.0);
  if (neg) {
    std::printf("negative control (wrong augmentation key on CPU): %s\n", bad > 0 ? "differs (expected)" : "IDENTICAL (bad)");
    return bad > 0 ? 0 : 1;
  }
  return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: ppo_verify v4|v5|v6|v7|bench [--negative]\n"); return 2; }
  const std::string m = argv[1];
  bool neg = false;
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--negative")) neg = true;
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--g4data")) g_g4data = true;
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--a2")) g_a2 = true;
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--act8")) g_act8 = true;
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--aug")) g_aug = true;
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--beh")) g_beh = true;
  for (int a = 2; a + 1 < argc; ++a) if (!std::strcmp(argv[a], "--stage")) g_stage = std::atoi(argv[a + 1]);
  for (int a = 2; a < argc; ++a) if (!std::strcmp(argv[a], "--curr")) g_curr = true;
  if (m == "switch") return run_switch(argc > 2 && argv[2][0] != '-' ? std::atoi(argv[2]) : 1024, argc > 3 && argv[3][0] != '-' ? std::atoi(argv[3]) : 32, neg);
  if (m == "obs") return run_obs(neg);
  if (m == "slotcols") return run_slotcols();
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
  if (m == "eval" && argc > 3) {
    float mc[4] = {0.f, 0.f, 3.f, 6.f};
    for (int k = 0; k < 4; ++k) if (argc > 8 + k) mc[k] = (float)std::atof(argv[8 + k]);
    return run_eval(argv[2], std::atoi(argv[3]), argc > 4 ? std::atoi(argv[4]) : 10, argc > 5 ? std::atoi(argv[5]) : 4096, argc > 6 ? std::atoi(argv[6]) : 1,
                    argc > 7 ? std::atoi(argv[7]) : 1, mc);
  }
  if (m == "v7") return run_v7();
  if (m == "snap" && argc > 2)
    return run_snap(argv[2], argc > 3 ? std::atoi(argv[3]) : 4096, argc > 4 ? std::atoi(argv[4]) : 64, argc > 5 ? std::atoi(argv[5]) : 6,
                    argc > 6 ? std::atoi(argv[6]) : 1, argc > 7 ? std::atoi(argv[7]) : 1);
  return 2;
}
