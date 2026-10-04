// BC 학생 학습기 검증(계획서 9절) — 층 함수는 RL network 의 CPU 참조판(net_ref: FP64 정답, EMUL 바닥)을 쓰고, BC 손실·정책 사슬 조립·학생 입력은 여기서.
//   bc_verify v5 [--negative] [--teacher CKPT]  : 학습 한 스텝(손실·평균·dZ 모든 층·dpool·기울기 6 텐서) + Adam 1·10 스텝 + FP64 대 유한 차분
//                                               + 학생 입력 CPU == GPU 비트 + 기록 = 본 것(롤아웃 때 학생 입력과 저장 표본에서 다시 만든 입력이 비트 같음)
//   bc_verify v6                                : 그래프 == 즉시 실행(기록·학습·DAgger 롤아웃 섞어서) 비트 동일
//   bc_verify v7                                : 같은 씨앗 두 번 비트 동일 / 다른 씨앗은 달라야
//   bc_verify bench [N] [T] [mb] [K]            : 처리량
// 규칙(9절): GPU 오차(GPU − FP64) ≤ 2 × 바닥(EMUL − FP64), 상대 L2, 텐서마다. 통과 = 종료 코드 0. --negative 는 버그를 넣어 반드시 실패해야 0.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bc.h"
#include "net_ref.h"

using namespace bc;
using namespace net;

#define VCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(3); } } while (0)

template <class T_>
static std::vector<T_> dl(const T_* d, size_t n) {
  std::vector<T_> h(n);
  VCK(cudaMemcpy(h.data(), d, sizeof(T_) * n, cudaMemcpyDeviceToHost));
  return h;
}
static std::vector<double> bfv(const std::vector<uint16_t>& v) { std::vector<double> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = bf2f(v[i]); return o; }
static std::vector<double> fv(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

static std::string g_teacher = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/ra_ppoout/g5/t4/on_s1/ckpt_final.bin";

static BcConfig small_cfg(uint64_t seed, int graphs) {
  BcConfig c{};
  c.n_env = 512; c.horizon = 16; c.stage = 2; c.use_map = 2; c.teacher_use_map = 2; c.student_goal = 0;
  c.use_graphs = graphs; c.log_ring = 16; c.mb = 1024; c.upd_steps = 4; c.dw_chunk = 1024; c.store_render = 1;
  c.cap = 512 * 16 * 3; c.seed = seed; c.env_seed = seed + 100;
  c.lr = 1e-3f; c.adam_b1 = 0.9f; c.adam_b2 = 0.999f; c.adam_eps = 1e-8f; c.max_grad_norm = 1.0f;
  c.map_p0 = 0.2f; c.map_p1 = 0.6f; c.map_kmin = 1; c.map_kmax = 8; c.map_reveal_r = 1.5f;
  return c;
}
static void load_teacher(void* b) {
  if (bc_load_teacher(b, g_teacher.c_str()) != 0) std::fprintf(stderr, "note: teacher %s not loaded (labels from a zero teacher)\n", g_teacher.c_str());
}

// ---- CPU 참조판: 칸 MLP → 집합 → 정책 사슬 A1..A4 → MSE → 뒤 ----
struct Ref {
  std::vector<double> sin, s1o, s2o, x0, h[N_LAYER], dz[N_LAYER], dpool, grad;
  std::vector<int> amax;
  double loss = 0;
};
static void ref_run(netref::Mode md, const std::vector<double>& P, const std::vector<uint16_t>& x0b, const std::vector<uint16_t>& sinb,
                    const std::vector<uint32_t>& mask, const std::vector<float>& lab, int M, Ref& r, bool bwd) {
  const ParamLayout lay = param_layout();
  const int MS = M * KSLOT;
  std::vector<double> W(P.size());
  for (size_t i = 0; i < P.size(); ++i) W[i] = md == netref::EMUL ? (double)rbf((float)P[i]) : P[i];
  auto Wl = [&](int l) { return W.data() + lay.off[l]; };
  r.sin = bfv(sinb);
  r.x0 = bfv(x0b);
  r.s1o.assign((size_t)MS * kLayers[L_S1].ldo, 0.0);
  r.s2o.assign((size_t)MS * kLayers[L_S2].ldo, 0.0);
  netref::lin_fwd(md, r.sin.data(), MS, kLayers[L_S1], Wl(L_S1), r.s1o.data());
  netref::lin_fwd(md, r.s1o.data(), MS, kLayers[L_S2], Wl(L_S2), r.s2o.data());
  r.amax.assign((size_t)M * S_H, 255);
  netref::pool_fwd(md, r.s2o.data(), mask.data(), M, r.x0.data(), r.amax.data());
  const double* X = r.x0.data();
  for (int l = L_A1; l <= L_A4; ++l) {
    r.h[l].assign((size_t)M * kLayers[l].ldo, 0.0);
    netref::lin_fwd(md, X, M, kLayers[l], Wl(l), r.h[l].data());
    X = r.h[l].data();
  }
  r.dz[L_A4].assign((size_t)M * N_ACT, 0.0);
  double L = 0.0;
  float Lf = 0.f;
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < N_LAB; ++k) {
      if (md == netref::EMUL) {
        const float d = (float)r.h[L_A4][(size_t)m * N_ACT + k] - lab[(size_t)m * N_LAB + k];
        Lf = Lf + d * d;
        r.dz[L_A4][(size_t)m * N_ACT + k] = rbf(2.f * d * (1.f / (float)M));
      } else {
        const double d = r.h[L_A4][(size_t)m * N_ACT + k] - (double)lab[(size_t)m * N_LAB + k];
        L += d * d;
        r.dz[L_A4][(size_t)m * N_ACT + k] = 2.0 * d / M;
      }
    }
  r.loss = md == netref::EMUL ? (double)(Lf / (float)M) : L / M;
  if (!bwd) return;
  r.grad.assign(lay.total, 0.0);
  for (int l = L_A4; l > L_A1; --l) {
    r.dz[l - 1].assign((size_t)M * kLayers[l - 1].N, 0.0);
    netref::lin_dw(md, r.dz[l].data(), r.h[l - 1].data(), M, kLayers[l], r.grad.data() + lay.off[l]);
    netref::lin_dx(md, r.dz[l].data(), M, kLayers[l], Wl(l), kLayers[l - 1].N, r.h[l - 1].data(), r.dz[l - 1].data(), true);
  }
  netref::lin_dw(md, r.dz[L_A1].data(), r.x0.data(), M, kLayers[L_A1], r.grad.data() + lay.off[L_A1]);
  r.dpool.assign((size_t)M * POOL_W, 0.0);
  netref::lin_dx(md, r.dz[L_A1].data(), M, kLayers[L_A1], Wl(L_A1), POOL_W, nullptr, r.dpool.data(), false);
  r.dz[L_S2].assign((size_t)MS * S_H, 0.0);
  netref::pool_bwd(md, r.dpool.data(), r.s2o.data(), mask.data(), r.amax.data(), M, r.dz[L_S2].data());
  netref::lin_dw(md, r.dz[L_S2].data(), r.s1o.data(), MS, kLayers[L_S2], r.grad.data() + lay.off[L_S2]);
  r.dz[L_S1].assign((size_t)MS * S_H, 0.0);
  netref::lin_dx(md, r.dz[L_S2].data(), MS, kLayers[L_S2], Wl(L_S2), kLayers[L_S1].N, r.s1o.data(), r.dz[L_S1].data(), true);
  netref::lin_dw(md, r.dz[L_S1].data(), r.sin.data(), MS, kLayers[L_S1], r.grad.data() + lay.off[L_S1]);
}

static double rel(const double* a, const double* b, size_t n, size_t sa = 1, size_t sb = 1, size_t w = 1, size_t ld = 1) {
  // a, b: n 행 × w 열(줄 간격 ld) 비교
  (void)sa; (void)sb;
  double num = 0, den = 0;
  for (size_t r = 0; r < n; ++r)
    for (size_t c = 0; c < w; ++c) {
      const double x = a[r * ld + c], y = b[r * ld + c];
      num += (x - y) * (x - y);
      den += y * y;
    }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}
struct Check {
  int fails = 0, total = 0;
  void cmp(const char* name, const std::vector<double>& gpu, const std::vector<double>& f64, const std::vector<double>& emul, size_t rows, size_t w, size_t ld) {
    const double eg = rel(gpu.data(), f64.data(), rows, 1, 1, w, ld), ef = rel(emul.data(), f64.data(), rows, 1, 1, w, ld);
    const bool ok = eg <= 2.0 * ef + 1e-12;
    ++total;
    if (!ok) ++fails;
    std::printf("  %-22s GPU %.3e  floor(EMUL) %.3e  ratio %6.2f  %s\n", name, eg, ef, ef > 0 ? eg / ef : 0.0, ok ? "ok" : "FAIL");
  }
};
static std::vector<double> seg(const std::vector<double>& v, long long off, long long n) { return std::vector<double>(v.begin() + off, v.begin() + off + n); }

static int run_v5(int bug) {
  BcConfig c = small_cfg(23, 0);
  Bc b(c);
  b.bug = bug;
  b.keep_slot_bufs = true;
  load_teacher(&b);
  const ParamLayout lay = b.lay;
  const int N = b.N, M = b.MB;
  int fails = 0, total = 0;
  // 자료: 교사 기록 2 롤아웃 + 학생이 움직이며 기록 1 롤아웃(DAgger 자리)
  bc_set_mode(&b, 0, 1); b.launch(0); b.launch(0);
  bc_set_mode(&b, 1, 1);
  const long long cur_before = dl(b.data_d, 1)[0].cursor;
  b.launch(0);
  VCK(cudaDeviceSynchronize());
  { BcLog L; while (b.poll(&L)) {} }
  const Data dd = dl(b.data_d, 1)[0];
  std::printf("v5: N %d T %d, data count %lld, minibatch %d, bug %d\n", N, b.T, dd.count, M, bug);

  // (1) 기록 = 본 것: 마지막 스텝(t = T−1)의 저장 표본으로 CPU 가 다시 만든 학생 입력 == 롤아웃 때 GPU 학생 입력(비트), 라벨 == clamp(교사 μ)
  {
    const int t = b.T - 1;
    const auto x0g = dl(b.ns.x0, (size_t)N * X0_W), sing = dl(b.ns.sin, (size_t)N * KSLOT * SLOT_IN);
    const auto mg = dl(b.ns.mask, N);
    const auto mT = dl(b.nt.mean, (size_t)N * N_ACT);
    long long diff = 0, ldiff = 0;
    std::vector<uint16_t> obs(env::N_OBS), x0c(X0_W), sinc(KSLOT * SLOT_IN);
    for (int i = 0; i < N; ++i) {
      const long long slot = (cur_before + (long long)t * N + i) % b.cap;
      VCK(cudaMemcpy(obs.data(), b.d_obs + slot * env::N_OBS, 2 * env::N_OBS, cudaMemcpyDeviceToHost));
      gmap::MapTok tk;
      VCK(cudaMemcpy(&tk, b.d_tok + slot, sizeof tk, cudaMemcpyDeviceToHost));
      float lab[N_LAB];
      VCK(cudaMemcpy(lab, b.d_lab + slot * N_LAB, sizeof lab, cudaMemcpyDeviceToHost));
      float of[env::N_OBS];
      for (int k = 0; k < env::N_OBS; ++k) of[k] = bf2f(obs[k]);
      std::fill(x0c.begin(), x0c.end(), 0);
      const uint32_t mk = assemble_student(of, 1, 0, tk, x0c.data(), sinc.data(), c.use_map, c.student_goal, 0, 1);
      for (int k = X0_OBS; k < X0_W; ++k) diff += x0c[k] != x0g[(size_t)i * X0_W + k];
      for (int k = 0; k < KSLOT * SLOT_IN; ++k) diff += sinc[k] != sing[(size_t)i * KSLOT * SLOT_IN + k];
      diff += mk != mg[i];
      for (int k = 0; k < N_LAB; ++k) ldiff += lab[k] != std::fmin(std::fmax(mT[(size_t)i * N_ACT + k], -1.f), 1.f);
    }
    const bool ok = diff == 0 && ldiff == 0;
    ++total; fails += !ok;
    std::printf("  record == seen: student input rebuilt from %d stored samples vs rollout input: %lld words differ; labels vs clamp(teacher mu): %lld  %s\n", N, diff, ldiff,
                ok ? "ok" : "FAIL");
  }

  // (2) 학습 한 스텝
  const long long iter0 = dl(b.ts, 1)[0].iter;
  b.gather();
  b.forward(b.Pb, b.ns, M);
  b.loss(M);
  b.backward(M);
  VCK(cudaDeviceSynchronize());
  const auto x0b = dl(b.ns.x0, (size_t)M * X0_W), sinb = dl(b.ns.sin, (size_t)M * KSLOT * SLOT_IN);
  const auto mask = dl(b.ns.mask, M);
  const auto lab = dl(b.lab_mb, (size_t)M * N_LAB);
  // 학생 입력: CPU 가 같은 표본 번호(장치 열쇠를 호스트에서)로 만든 입력 == GPU 모으기(비트)
  {
    long long diff = 0;
    std::vector<uint16_t> obs(env::N_OBS), x0c(X0_W), sinc(KSLOT * SLOT_IN);
    for (int r = 0; r < M; ++r) {
      const long long idx = (long long)(hash4(c.seed, (uint64_t)iter0, (uint64_t)r, 0x42434d42ull) % (uint64_t)dd.count);
      VCK(cudaMemcpy(obs.data(), b.d_obs + idx * env::N_OBS, 2 * env::N_OBS, cudaMemcpyDeviceToHost));
      gmap::MapTok tk;
      VCK(cudaMemcpy(&tk, b.d_tok + idx, sizeof tk, cudaMemcpyDeviceToHost));
      float of[env::N_OBS];
      for (int k = 0; k < env::N_OBS; ++k) of[k] = bf2f(obs[k]);
      const uint32_t mk = assemble_student(of, 1, 0, tk, x0c.data(), sinc.data(), c.use_map, c.student_goal, 0, 1);
      for (int k = X0_OBS; k < X0_W; ++k) diff += x0c[k] != x0b[(size_t)r * X0_W + k];
      for (int k = 0; k < KSLOT * SLOT_IN; ++k) diff += sinc[k] != sinb[(size_t)r * KSLOT * SLOT_IN + k];
      diff += mk != mask[r];
    }
    ++total; fails += diff != 0;
    std::printf("  gather: student input CPU == GPU over %d rows: %lld words differ  %s\n", M, diff, diff == 0 ? "ok" : "FAIL");
  }
  const auto Pf = dl(b.P, lay.total);
  const std::vector<double> P0(Pf.begin(), Pf.end());
  netref::set_tc_model(16, c.dw_chunk);
  Ref R64, Rem;
  ref_run(netref::FP64, P0, x0b, sinb, mask, lab, M, R64, true);
  ref_run(netref::EMUL, P0, x0b, sinb, mask, lab, M, Rem, true);
  const double Lg = dl(b.ts, 1)[0].s_pg;
  {
    const double eg = std::fabs(Lg - R64.loss) / R64.loss, ef = std::fabs(Rem.loss - R64.loss) / R64.loss;
    const bool ok = eg <= 2.0 * ef + 2e-6;   // 스칼라 합: GPU 는 나무 합, EMUL 은 차례 합 → 반올림 1e-6 수준 여유
    ++total; fails += !ok;
    std::printf("  %-22s GPU %.6f FP64 %.6f EMUL %.6f  rel GPU %.2e floor %.2e  %s\n", "loss (MSE)", Lg, R64.loss, Rem.loss, eg, ef, ok ? "ok" : "FAIL");
  }
  Check ck;
  ck.cmp("mean A4 (f32)", fv(dl(b.ns.mean, (size_t)M * N_ACT)), R64.h[L_A4], Rem.h[L_A4], M, N_LAB, N_ACT);
  for (int l : {L_A1, L_A2, L_A3}) {
    char nm[32];
    std::snprintf(nm, sizeof nm, "fwd A%d (bf16)", l - L_A1 + 1);
    ck.cmp(nm, bfv(dl(b.ns.ho[l], (size_t)M * kLayers[l].ldo)), R64.h[l], Rem.h[l], M, kLayers[l].N, kLayers[l].ldo);
  }
  ck.cmp("fwd pool (bf16)", bfv(dl(b.ns.x0, (size_t)M * X0_W)), R64.x0, Rem.x0, M, POOL_W, X0_W);
  for (int l : {L_A4, L_A3, L_A2, L_A1}) {
    char nm[32];
    std::snprintf(nm, sizeof nm, "dZ A%d (bf16)", l - L_A1 + 1);
    ck.cmp(nm, bfv(dl(b.ns.dz[l], (size_t)M * kLayers[l].N)), R64.dz[l], Rem.dz[l], M, kLayers[l].N, kLayers[l].N);
  }
  ck.cmp("dpool (f32)", fv(dl(b.ns.dpool, (size_t)M * POOL_W)), R64.dpool, Rem.dpool, M, POOL_W, POOL_W);
  ck.cmp("dZ S2 (bf16)", bfv(dl(b.ns.dz[L_S2], (size_t)M * KSLOT * S_H)), R64.dz[L_S2], Rem.dz[L_S2], (size_t)M * KSLOT, S_H, S_H);
  ck.cmp("dZ S1 (bf16)", bfv(dl(b.ns.dz[L_S1], (size_t)M * KSLOT * S_H)), R64.dz[L_S1], Rem.dz[L_S1], (size_t)M * KSLOT, S_H, S_H);
  const auto Gg = fv(dl(b.G, lay.total));
  for (int l : {L_S1, L_S2, L_A1, L_A2, L_A3, L_A4}) {
    const long long n = (long long)kLayers[l].N * kLayers[l].K;
    char nm[32];
    std::snprintf(nm, sizeof nm, "grad W %s", l == L_S1 ? "S1" : l == L_S2 ? "S2" : l == L_A1 ? "A1" : l == L_A2 ? "A2" : l == L_A3 ? "A3" : "A4");
    ck.cmp(nm, seg(Gg, lay.off[l], n), seg(R64.grad, lay.off[l], n), seg(Rem.grad, lay.off[l], n), (size_t)n, 1, 1);
  }
  // (3) Adam 1 스텝: 갱신량 ΔP
  const AdamHyper ah{c.adam_b1, c.adam_b2, c.adam_eps, c.max_grad_norm};
  b.optimizer();
  VCK(cudaDeviceSynchronize());
  {
    const auto P1 = dl(b.P, lay.total);
    std::vector<double> d64 = P0, dem = P0, m64(lay.total, 0), v64(lay.total, 0), mem(lay.total, 0), vem(lay.total, 0);
    netref::adam(netref::FP64, d64, R64.grad, m64, v64, 1, c.lr, ah);
    netref::adam(netref::EMUL, dem, Rem.grad, mem, vem, 1, c.lr, ah);
    std::vector<double> dg(lay.total), df(lay.total), de(lay.total);
    for (long long i = 0; i < lay.total; ++i) { dg[i] = (double)P1[i] - P0[i]; df[i] = d64[i] - P0[i]; de[i] = dem[i] - P0[i]; }
    ck.cmp("Adam 1 step dP", dg, df, de, (size_t)lay.total, 1, 1);
  }
  // (4) Adam 10 스텝(같은 미니배치): 처음 변수·Adam 상태로 되돌린 뒤
  {
    VCK(cudaMemcpy(b.P, Pf.data(), sizeof(float) * lay.total, cudaMemcpyHostToDevice));
    VCK(cudaMemset(b.Am, 0, sizeof(float) * lay.total));
    VCK(cudaMemset(b.Av, 0, sizeof(float) * lay.total));
    TrainState s = dl(b.ts, 1)[0];
    s.adam_t = 0;
    VCK(cudaMemcpy(b.ts, &s, sizeof s, cudaMemcpyHostToDevice));
    to_bf16(b.P, b.Pb, lay.total, 0);
    for (int k = 0; k < 10; ++k) { b.forward(b.Pb, b.ns, M); b.loss(M); b.backward(M); b.optimizer(); }
    VCK(cudaDeviceSynchronize());
    const auto P10 = dl(b.P, lay.total);
    std::vector<double> p64 = P0, pem = P0, m64(lay.total, 0), v64(lay.total, 0), mem(lay.total, 0), vem(lay.total, 0);
    for (int k = 0; k < 10; ++k) {
      Ref a, e;
      ref_run(netref::FP64, p64, x0b, sinb, mask, lab, M, a, true);
      netref::adam(netref::FP64, p64, a.grad, m64, v64, k + 1, c.lr, ah);
      ref_run(netref::EMUL, pem, x0b, sinb, mask, lab, M, e, true);
      netref::adam(netref::EMUL, pem, e.grad, mem, vem, k + 1, c.lr, ah);
    }
    std::vector<double> dg(lay.total), df(lay.total), de(lay.total);
    for (long long i = 0; i < lay.total; ++i) { dg[i] = (double)P10[i] - P0[i]; df[i] = p64[i] - P0[i]; de[i] = pem[i] - P0[i]; }
    ck.cmp("Adam 10 steps dP", dg, df, de, (size_t)lay.total, 1, 1);
  }
  fails += ck.fails; total += ck.total;
  // (5) FP64 참조판 대 FP64 중심 유한 차분(변수 128 개, 행 256 — 꺾인 곳(최댓값 집합·ELU)을 건너면 h 를 1/10·1/100 로 다시 재서 가장 작은 오차)
  {
    const int Mf = 256;
    std::vector<uint16_t> x0s(x0b.begin(), x0b.begin() + (size_t)Mf * X0_W), sins(sinb.begin(), sinb.begin() + (size_t)Mf * KSLOT * SLOT_IN);
    std::vector<uint32_t> ms(mask.begin(), mask.begin() + Mf);
    std::vector<float> ls(lab.begin(), lab.begin() + (size_t)Mf * N_LAB);
    Ref g;
    ref_run(netref::FP64, P0, x0s, sins, ms, ls, Mf, g, true);
    double gs = 0;
    for (int l : {L_S1, L_S2, L_A1, L_A2, L_A3, L_A4})
      for (long long q = 0; q < (long long)kLayers[l].N * kLayers[l].K; ++q) gs = std::fmax(gs, std::fabs(g.grad[lay.off[l] + q]));
    uint64_t s = 99;
    double worst = 0;
    int nz = 0;
    const int layers[6] = {L_S1, L_S2, L_A1, L_A2, L_A3, L_A4};
    for (int j = 0; j < 128; ++j) {
      const int l = layers[j % 6];
      s = mix64(s + 1);
      const int n = (int)(s % (uint64_t)(l == L_A4 ? N_LAB : kLayers[l].N)), k = (int)((s >> 20) % (uint64_t)(kLayers[l].bias + 1));   // 입력 칸 + 편향 칸
      const long long q = lay.off[l] + (long long)n * kLayers[l].K + k;
      if (g.grad[q] != 0) ++nz;
      double best = 1e30;
      for (double h : {1e-4, 1e-5, 1e-6}) {
        std::vector<double> Pp = P0, Pm = P0;
        Pp[q] += h; Pm[q] -= h;
        Ref a, bb;
        ref_run(netref::FP64, Pp, x0s, sins, ms, ls, Mf, a, false);
        ref_run(netref::FP64, Pm, x0s, sins, ms, ls, Mf, bb, false);
        const double fd = (a.loss - bb.loss) / (2 * h);
        const double e = std::fabs(fd - g.grad[q]) / (std::fabs(g.grad[q]) + 1e-3 * gs);
        best = std::fmin(best, e);
        if (best < 1e-6) break;
      }
      worst = std::fmax(worst, best);
    }
    const bool ok = worst < 1e-4;
    ++total; fails += !ok;
    std::printf("  FP64 ref vs finite diff: 128 params (%d nonzero grads), worst rel err %.2e  %s\n", nz, worst, ok ? "ok" : "FAIL");
  }
  std::printf("v5: %d / %d checks passed\n", total - fails, total);
  return fails;
}

// ---- V6 / V7: 기록(교사) → 갱신 → DAgger 롤아웃(학생 + 기록) → 갱신 의 결과 전체를 비트로 ----
struct Snap {
  std::vector<float> P, Am, Av, lab;
  std::vector<uint16_t> obs;
  std::vector<gmap::MapTok> tok;
  std::vector<uint32_t> meta;
  std::vector<RenderState> rs;
  std::vector<unsigned long long> tab;
  std::vector<BcLog> logs;
};
static Snap run_seq(BcConfig c) {
  Bc b(c);
  load_teacher(&b);
  Snap s;
  auto drain = [&] { BcLog L; while (b.poll(&L)) { L.gpu_ms = 0; s.logs.push_back(L); } };
  bc_set_mode(&b, 0, 1);
  for (int k = 0; k < 2; ++k) { b.launch(0); drain(); }
  for (int k = 0; k < 3; ++k) { b.launch(1); drain(); }
  bc_set_mode(&b, 1, 1);
  b.launch(0);
  for (int k = 0; k < 2; ++k) { b.launch(1); drain(); }
  bc_set_mode(&b, 1, 0);
  b.launch(0);
  VCK(cudaDeviceSynchronize());
  drain();
  const long long n = b.lay.total, cap = b.cap;
  s.P = dl(b.P, n); s.Am = dl(b.Am, n); s.Av = dl(b.Av, n);
  s.lab = dl(b.d_lab, (size_t)cap * N_LAB); s.obs = dl(b.d_obs, (size_t)cap * env::N_OBS); s.tok = dl(b.d_tok, (size_t)cap); s.meta = dl(b.d_meta, (size_t)cap);
  s.rs = dl(b.d_rs, (size_t)cap);
  s.tab = dl(b.tab, (size_t)3 * 2 * 10 * 6);
  return s;
}
static long cnt(const void* a, const void* b, size_t bytes) {
  const uint32_t* x = (const uint32_t*)a; const uint32_t* y = (const uint32_t*)b;
  long d = 0;
  for (size_t i = 0; i < bytes / 4; ++i) d += x[i] != y[i];
  return d;
}
static long diff_snap(const Snap& a, const Snap& b, bool verbose) {
  struct { const char* n; long d; } r[] = {
      {"params", cnt(a.P.data(), b.P.data(), 4 * a.P.size())}, {"adam m", cnt(a.Am.data(), b.Am.data(), 4 * a.Am.size())},
      {"adam v", cnt(a.Av.data(), b.Av.data(), 4 * a.Av.size())}, {"labels", cnt(a.lab.data(), b.lab.data(), 4 * a.lab.size())},
      {"obs", cnt(a.obs.data(), b.obs.data(), 2 * a.obs.size())}, {"map tokens", cnt(a.tok.data(), b.tok.data(), sizeof(gmap::MapTok) * a.tok.size())},
      {"meta", cnt(a.meta.data(), b.meta.data(), 4 * a.meta.size())}, {"render state", cnt(a.rs.data(), b.rs.data(), sizeof(RenderState) * a.rs.size())},
      {"episode table", cnt(a.tab.data(), b.tab.data(), 8 * a.tab.size())},
      {"logs", a.logs.size() == b.logs.size() ? cnt(a.logs.data(), b.logs.data(), sizeof(BcLog) * a.logs.size()) : 999999}};
  long t = 0;
  for (auto& x : r) { if (verbose) std::printf("  %-14s %ld words differ\n", x.n, x.d); t += x.d; }
  return t;
}
static int run_v6() {
  const Snap g = run_seq(small_cfg(5, 1)), e = run_seq(small_cfg(5, 0));
  std::printf("v6 graph vs eager (record 2, update 3x%d, DAgger rollout 1, update 2, eval rollout 1):\n", small_cfg(5, 1).upd_steps);
  const long d = diff_snap(g, e, true);
  unsigned long long ne = 0;
  for (size_t k = 0; k < g.tab.size(); k += 6) ne += g.tab[k];
  std::printf("v6: %ld words differ, %llu episodes in table, %zu logs  %s\n", d, ne, g.logs.size(), d == 0 ? "PASS" : "FAIL");
  return d != 0;
}
static int run_v7() {
  const Snap a = run_seq(small_cfg(7, 1)), b = run_seq(small_cfg(7, 1)), c = run_seq(small_cfg(8, 1));
  const long same = diff_snap(a, b, false), other = diff_snap(a, c, false);
  std::printf("v7: same seed %ld words differ (must be 0), other seed %ld (must be > 0)  %s\n", same, other, same == 0 && other > 0 ? "PASS" : "FAIL");
  return !(same == 0 && other > 0);
}

static int run_bench(int N, int T, int mb, int K) {
  BcConfig c = small_cfg(1, 1);
  c.n_env = N; c.horizon = T; c.mb = mb; c.upd_steps = K; c.cap = (long long)N * T * 2;
  Bc b(c);
  load_teacher(&b);
  auto timed = [&](int kind, int reps) {
    BcLog L;
    double ms = 0;
    int n = 0;
    for (int k = 0; k < reps + 1; ++k) b.launch(kind);
    VCK(cudaDeviceSynchronize());
    while (b.poll(&L)) { if (n++ > 0) ms += L.gpu_ms; }
    return ms / reps;
  };
  bc_set_mode(&b, 0, 1);
  const double r_rec = timed(0, 5);
  bc_set_mode(&b, 1, 0);
  const double r_st = timed(0, 5);
  const double u = timed(1, 5);
  std::printf("bench N %d T %d: rollout teacher+record %.2f ms (%.3g env-step/s), student acts %.2f ms; update %d steps x mb %d: %.2f ms (%.3f ms/step, %.3g samples/s); device %.2f GB\n",
              N, T, r_rec, (double)N * T / (r_rec * 1e-3), r_st, K, mb, u, u / K, (double)mb * K / (u * 1e-3), bc_device_bytes(&b) / 1e9);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: bc_verify v5 [--negative] | v6 | v7 | bench [N T mb K]  [--teacher CKPT]\n"); return 2; }
  bool neg = false;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--negative") neg = true;
    else if (a == "--teacher" && i + 1 < argc) g_teacher = argv[++i];
    else pos.push_back(a);
  }
  const std::string m = pos[0];
  if (m == "v5") {
    if (!neg) return run_v5(0) ? 1 : 0;
    const int f1 = run_v5(1), f2 = run_v5(2);
    std::printf("v5 --negative: bug 1 (MSE grad factor 2 dropped) %d failed checks, bug 2 (ELU' A3->A2 dropped) %d  %s\n", f1, f2, f1 > 0 && f2 > 0 ? "PASS (both fail)" : "FAIL");
    return f1 > 0 && f2 > 0 ? 0 : 1;
  }
  if (m == "v6") return run_v6();
  if (m == "v7") return run_v7();
  if (m == "bench")
    return run_bench(pos.size() > 1 ? std::atoi(pos[1].c_str()) : 4096, pos.size() > 2 ? std::atoi(pos[2].c_str()) : 64, pos.size() > 3 ? std::atoi(pos[3].c_str()) : 8192,
                     pos.size() > 4 ? std::atoi(pos[4].c_str()) : 50);
  return 2;
}
