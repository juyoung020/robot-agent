// BC 학생 학습기 검증(계획서 9절) — 층 함수는 RL network 의 CPU 참조판(net_ref: FP64 정답, EMUL 바닥)을 쓰고, BC 손실(MSE·flow)·학생 사슬 조립
// (칸 MLP → 집합 → [P1 영상 칸 → 이어 붙이기] → A1–A3 → A4 또는 E1–E3)·학생 입력·청크 라벨은 여기서.
//   bc_verify v5 [--lite] [--negative] [--teacher CKPT] : 학습 한 스텝(손실·출력·dZ 모든 층·dpool·기울기 모든 텐서) + Adam 1·10 스텝 + FP64 대 유한 차분
//                                                        + 학생 입력·청크 라벨·flow 입력 CPU == GPU + 기록 = 본 것(학생 입력, 다시 렌더한 영상 토큰)
//   bc_verify v6 [--lite]                                : 그래프 == 즉시 실행(기록·학습·DAgger 롤아웃 섞어서, 영상 학생은 렌더·인코더 포함) 비트 동일
//   bc_verify v7 [--lite]                                : 같은 씨앗 두 번 비트 동일 / 다른 씨앗은 달라야
//   bc_verify bench [N] [T] [mb] [K] [--lite]            : 처리량(영상 학생은 렌더·인코딩·학습 나눠서). BC_OVERLAP_PROBE=1 이면 두 스트림 겹치기 시험도(README 점검 절)
// 기본은 영상 학생(영상 + 글 + flow 청크 16), --lite 는 student-lite(MSE, 영상·글 없음).
// 규칙(9절): GPU 오차(GPU − FP64) ≤ 2 × 바닥(EMUL − FP64), 상대 L2, 텐서마다. 통과 = 종료 코드 0. --negative 는 버그를 넣어 반드시 실패해야 0.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bc.h"
#include "bc_render.h"
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
static std::string g_text = "";
static bool g_lite = false;
static bool g_act8 = false;    // --act8: 행동 8 모두 학습(act_mask 0xff, VLA_INPUT 5절)
static bool g_arch1 = false;   // --arch1: 토큰마다 학생(tf.h). 신경망 수치는 tf_verify, 여기는 입력·자료·그래프·결정성
static bool g_aug = false;     // --aug: 학생 입력 흔들기 켬(v6/v7)
static bool g_txt = false;     // --txt: 영상 없는 학생에 글(지시) 칸 — BEHAVIOR(--stage 3)의 pnp 지시 행 확인용
static bool g_gdrop = false;   // --gdrop: 학생 목표 표시 감추기 0.5(obs.h p_goal_drop)
static int g_stage = -1;        // --stage 3: BEHAVIOR 집(영상 없는 학생만 — --lite)
static bool g_raug = false;    // --raug: 렌더 흔들기(색·조명·노출 0.8) + 팀 기본 섞기 0.25

static BcConfig small_cfg(uint64_t seed, int graphs) {
  BcConfig c{};
  c.n_env = 512; c.horizon = 16; c.stage = 2; c.use_map = 2; c.teacher_use_map = 2; c.student_goal = 0;
  c.use_graphs = graphs; c.log_ring = 16; c.mb = 1024; c.upd_steps = 4; c.dw_chunk = 1024; c.store_render = 1;
  c.cap = 512 * 16 * 3; c.seed = seed; c.env_seed = seed + 100;
  c.lr = 1e-3f; c.adam_b1 = 0.9f; c.adam_b2 = 0.999f; c.adam_eps = 1e-8f; c.max_grad_norm = 1.0f;
  c.map_p0 = 0.2f; c.map_p1 = 0.6f; c.map_kmin = 1; c.map_kmax = 8; c.map_reveal_r = 1.5f;
  if (!g_lite) {
    c.vision = 1; c.head = 1; c.chunk = 16; c.flow_steps = 4; c.text = 1; c.render_profile = 1; c.render_batch = 256; c.img_dim = IMG_D;
    c.n_env = 256; c.mb = 256; c.cap = 256 * 16 * 3; c.upd_steps = 2;
  }
  if (g_act8) c.act_mask = 0xffu;
  if (g_gdrop) c.goal_drop = 0.5f;
  if (g_txt) c.text = 1;
  if (g_stage >= 0) c.stage = g_stage;
  if (g_raug) { c.render_aug = 1; c.ra_color = 0.8f; c.ra_light = 0.8f; c.ra_expo = 0.8f; c.render_team_mix = 0.25f; }
  if (g_arch1 && !g_lite) { c.arch = 1; c.tf_d = 128; c.tf_layers = 2; c.tf_heads = 2; c.tf_mlp = 256; c.tf_elayers = 2; }
  if (g_aug) {
    c.aug_on = 1; c.aug_vel_sigma = 0.05f; c.aug_prev_drop = 0.1f; c.aug_prev_sigma = 0.05f;
    c.aug_p_erase = 0.1f; c.aug_p_syn = 0.2f; c.aug_p_hyper = 0.1f; c.aug_p_wrong = 0.05f; c.aug_p_slot_drop = 0.1f; c.aug_p_map_off = 0.05f;
  }
  c.vit_prec = std::getenv("BC_VIT_PREC") ? std::atoi(std::getenv("BC_VIT_PREC")) : 0;   // 인코더 정밀도(G6): 0 FP16·FP32 누산, 1 FP16 누산, 2 FP8
  c.fp8 = std::getenv("NET_FP8") ? std::atoi(std::getenv("NET_FP8")) : 0;   // G6: 몸통 층 FP8 비트(1 앞, 2 dgrad, 4 wgrad)
  netref::set_fp8(c.fp8);                                                    // CPU 흉내(바닥)도 같은 FP8 처방
  return c;
}
static void load_aux(Bc* b) {
  if (bc_load_teacher(b, g_teacher.c_str()) != 0) {   // v2 관측과 맞는 교사가 없으면 무작위 교사(PPO 와 같은 초기화 규칙, 씨앗 고정) — 라벨이 0 이 아니게
    std::fprintf(stderr, "note: teacher %s not loaded (v1 checkpoint or missing) - using a random teacher (seed 77)\n", g_teacher.c_str());
    std::vector<float> hp(b->lay.total, 0.f);
    uint64_t sd = 77;
    for (int l = 0; l < N_LAYER; ++l) {
      const LayerDesc& L = kLayers[l];
      const float a = (l == L_A4 ? 1.0f : L.gain) * std::sqrt(3.f / (float)L.bias);
      for (int n = 0; n < L.N; ++n) for (int k = 0; k < L.bias; ++k) hp[b->lay.off[l] + (size_t)n * L.K + k] = dm::rand_range(sd, -a, a);
    }
    float* tmp = nullptr;
    VCK(cudaMalloc(&tmp, sizeof(float) * hp.size()));
    VCK(cudaMemcpy(tmp, hp.data(), sizeof(float) * hp.size(), cudaMemcpyHostToDevice));
    to_bf16(tmp, b->PbT, (long long)hp.size(), 0);
    VCK(cudaDeviceSynchronize());
    cudaFree(tmp);
  }
  if (b->sn.text && !g_text.empty()) {
    const int k = bc_load_text_table(b, g_text.c_str());
    if (k < 1) { std::fprintf(stderr, "text table %s not loaded (%d)\n", g_text.c_str(), k); std::exit(2); }
  }
}

// ---- CPU 참조판: 학생 사슬 전체 ----
struct In {   // GPU 가 만든 같은 양자화 입력
  int M = 0;
  std::vector<uint16_t> x0, sin, tok, txt, ein;   // x0 [M][288], sin, 영상 토큰 [M×128][784], 글 [M][768], E1 입력의 x_τ·시간 [M][64]
  std::vector<uint32_t> mask;
  std::vector<float> lab, u, fm;   // MSE 라벨 [M][8] / flow 목표 [M][128]·가림 [M][16]
  uint32_t amask = 0x3u;           // 학습하는 행동 비트(장치 값)
};
struct Ref {
  std::vector<double> sin, s1o, s2o, x0, x0e, tok, h[SL_N], out, dz[SL_N], dpool, grad;
  std::vector<int> amax;
  double loss = 0;
};
static void ref_run(netref::Mode md, const StudentNet& sn, const std::vector<double>& P, const In& in, Ref& r, bool bwd) {
  const int M = in.M, MS = M * KSLOT, MT = M * IMG_TOK;
  const bool em = md == netref::EMUL;
  std::vector<double> W(P.size());
  for (size_t i = 0; i < P.size(); ++i) W[i] = em ? (double)rbf((float)P[i]) : P[i];
  auto Wl = [&](int l) { return W.data() + sn.off[l]; };
  const LayerDesc* L = sn.L;
  r.sin = bfv(in.sin);
  r.x0 = bfv(in.x0);
  r.s1o.assign((size_t)MS * L[SL_S1].ldo, 0.0);
  r.s2o.assign((size_t)MS * L[SL_S2].ldo, 0.0);
  netref::lin_fwd(md, r.sin.data(), MS, L[SL_S1], Wl(SL_S1), r.s1o.data());
  netref::lin_fwd(md, r.s1o.data(), MS, L[SL_S2], Wl(SL_S2), r.s2o.data());
  r.amax.assign((size_t)M * S_H, 255);
  netref::pool_fwd(md, r.s2o.data(), in.mask.data(), M, r.x0.data(), r.amax.data());
  const double* X = r.x0.data();
  if (sn.ext()) {
    if (sn.vision) {
      r.tok = bfv(in.tok);
      r.h[SL_P1].assign((size_t)MT * IMG_D, 0.0);
      netref::lin_fwd(md, r.tok.data(), MT, L[SL_P1], Wl(SL_P1), r.h[SL_P1].data());
    }
    r.x0e.assign((size_t)M * sn.k1, 0.0);
    for (int m = 0; m < M; ++m) {
      for (int c = 0; c < X0_W; ++c) r.x0e[(size_t)m * sn.k1 + c] = r.x0[(size_t)m * X0_W + c];
      if (sn.vision) for (int c = 0; c < IMG_W; ++c) r.x0e[(size_t)m * sn.k1 + sn.x_img + c] = r.h[SL_P1][(size_t)m * IMG_W + c];
      if (sn.text) for (int c = 0; c < TXT_W; ++c) r.x0e[(size_t)m * sn.k1 + sn.x_txt + c] = bf2f(in.txt[(size_t)m * TXT_W + c]);
    }
    X = r.x0e.data();
  }
  for (int l : {SL_A1, SL_A2, SL_A3}) {
    r.h[l].assign((size_t)M * L[l].ldo, 0.0);
    netref::lin_fwd(md, X, M, L[l], Wl(l), r.h[l].data());
    X = r.h[l].data();
  }
  double Ls = 0.0;
  float Lf = 0.f;
  if (!sn.head) {
    r.out.assign((size_t)M * N_ACT, 0.0);
    netref::lin_fwd(md, r.h[SL_A3].data(), M, L[SL_A4], Wl(SL_A4), r.out.data());
    r.dz[SL_A4].assign((size_t)M * N_ACT, 0.0);
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < N_LAB; ++k) {
        if (!((in.amask >> k) & 1u)) continue;
        if (em) {
          const float d = (float)r.out[(size_t)m * N_ACT + k] - in.lab[(size_t)m * N_LAB + k];
          Lf = Lf + d * d;
          r.dz[SL_A4][(size_t)m * N_ACT + k] = rbf(2.f * d * (1.f / (float)M));
        } else {
          const double d = r.out[(size_t)m * N_ACT + k] - (double)in.lab[(size_t)m * N_LAB + k];
          Ls += d * d;
          r.dz[SL_A4][(size_t)m * N_ACT + k] = 2.0 * d / M;
        }
      }
  } else {
    for (int m = 0; m < M; ++m)
      for (int c = 0; c < FLOW_W + TEMB; ++c) r.h[SL_A3][(size_t)m * E_IN + E_X + c] = bf2f(in.ein[(size_t)m * (FLOW_W + TEMB) + c]);
    for (int l : {SL_E1, SL_E2}) {
      r.h[l].assign((size_t)M * L[l].ldo, 0.0);
      netref::lin_fwd(md, l == SL_E1 ? r.h[SL_A3].data() : r.h[SL_E1].data(), M, L[l], Wl(l), r.h[l].data());
    }
    r.out.assign((size_t)M * FLOW_W, 0.0);
    netref::lin_fwd(md, r.h[SL_E2].data(), M, L[SL_E3], Wl(SL_E3), r.out.data());
    r.dz[SL_E3].assign((size_t)M * FLOW_W, 0.0);
    for (int m = 0; m < M; ++m)
      for (int j = 0; j < FLOW_W; ++j) {
        const float mk = ((in.amask >> (j % N_LAB)) & 1u) ? in.fm[(size_t)m * MAX_H + j / N_LAB] : 0.f;
        if (em) {
          const float d = (float)r.out[(size_t)m * FLOW_W + j] - in.u[(size_t)m * FLOW_W + j];
          Lf = Lf + mk * d * d;
          r.dz[SL_E3][(size_t)m * FLOW_W + j] = rbf(2.f * mk * d * (1.f / (float)M));
        } else {
          const double d = r.out[(size_t)m * FLOW_W + j] - (double)in.u[(size_t)m * FLOW_W + j];
          Ls += mk * d * d;
          r.dz[SL_E3][(size_t)m * FLOW_W + j] = 2.0 * mk * d / M;
        }
      }
  }
  r.loss = em ? (double)(Lf / (float)M) : Ls / M;
  if (!bwd) return;
  r.grad.assign(sn.total, 0.0);
  auto G = [&](int l) { return r.grad.data() + sn.off[l]; };
  if (!sn.head) {
    netref::lin_dw(md, r.dz[SL_A4].data(), r.h[SL_A3].data(), M, L[SL_A4], G(SL_A4));
    r.dz[SL_A3].assign((size_t)M * L[SL_A3].N, 0.0);
    netref::lin_dx(md, r.dz[SL_A4].data(), M, L[SL_A4], Wl(SL_A4), L[SL_A3].N, r.h[SL_A3].data(), r.dz[SL_A3].data(), true);
  } else {
    const int ch[4] = {SL_E3, SL_E2, SL_E1, SL_A3};
    for (int q = 0; q < 3; ++q) {
      const int l = ch[q], p = ch[q + 1];
      netref::lin_dw(md, r.dz[l].data(), r.h[p].data(), M, L[l], G(l));
      r.dz[p].assign((size_t)M * L[p].N, 0.0);
      netref::lin_dx(md, r.dz[l].data(), M, L[l], Wl(l), L[p].N, r.h[p].data(), r.dz[p].data(), true);
    }
  }
  for (int l : {SL_A3, SL_A2}) {
    const int p = l - 1;
    netref::lin_dw(md, r.dz[l].data(), r.h[p].data(), M, L[l], G(l));
    r.dz[p].assign((size_t)M * L[p].N, 0.0);
    netref::lin_dx(md, r.dz[l].data(), M, L[l], Wl(l), L[p].N, r.h[p].data(), r.dz[p].data(), true);
  }
  const double* X1 = sn.ext() ? r.x0e.data() : r.x0.data();
  netref::lin_dw(md, r.dz[SL_A1].data(), X1, M, L[SL_A1], G(SL_A1));
  r.dpool.assign((size_t)M * POOL_W, 0.0);
  netref::lin_dx(md, r.dz[SL_A1].data(), M, L[SL_A1], Wl(SL_A1), POOL_W, nullptr, r.dpool.data(), false);
  if (sn.vision) {
    r.dz[SL_P1].assign((size_t)M * IMG_W, 0.0);
    netref::lin_dx(md, r.dz[SL_A1].data(), M, L[SL_A1], Wl(SL_A1) + sn.x_img, IMG_W, r.x0e.data() + sn.x_img, r.dz[SL_P1].data(), true);
    netref::lin_dw(md, r.dz[SL_P1].data(), r.tok.data(), MT, L[SL_P1], G(SL_P1));
  }
  r.dz[SL_S2].assign((size_t)MS * S_H, 0.0);
  netref::pool_bwd(md, r.dpool.data(), r.s2o.data(), in.mask.data(), r.amax.data(), M, r.dz[SL_S2].data());
  netref::lin_dw(md, r.dz[SL_S2].data(), r.s1o.data(), MS, L[SL_S2], G(SL_S2));
  r.dz[SL_S1].assign((size_t)MS * S_H, 0.0);
  netref::lin_dx(md, r.dz[SL_S2].data(), MS, L[SL_S2], Wl(SL_S2), L[SL_S1].N, r.s1o.data(), r.dz[SL_S1].data(), true);
  netref::lin_dw(md, r.dz[SL_S1].data(), r.sin.data(), MS, L[SL_S1], G(SL_S1));
}

static double rel(const double* a, const double* b, size_t n, size_t w, size_t lda, size_t ldb) {
  double num = 0, den = 0;
  for (size_t r = 0; r < n; ++r)
    for (size_t c = 0; c < w; ++c) {
      const double x = a[r * lda + c], y = b[r * ldb + c];
      num += (x - y) * (x - y);
      den += y * y;
    }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}
struct Check {
  int fails = 0, total = 0;
  void cmp(const char* name, const std::vector<double>& gpu, const std::vector<double>& f64, const std::vector<double>& emul, size_t rows, size_t w, size_t ldg,
           size_t ldr = 0) {
    if (!ldr) ldr = ldg;
    const double eg = rel(gpu.data(), f64.data(), rows, w, ldg, ldr), ef = rel(emul.data(), f64.data(), rows, w, ldr, ldr);
    const bool ok = eg <= 2.0 * ef + 1e-12;
    ++total;
    if (!ok) ++fails;
    std::printf("  %-22s GPU %.3e  floor(EMUL) %.3e  ratio %6.2f  %s\n", name, eg, ef, ef > 0 ? eg / ef : 0.0, ok ? "ok" : "FAIL");
  }
};
static std::vector<double> seg(const std::vector<double>& v, long long off, long long n) { return std::vector<double>(v.begin() + off, v.begin() + off + n); }
static const char* lname(int l) {
  static const char* n[SL_N] = {"S1", "S2", "P1", "A1", "A2", "A3", "A4", "E1", "E2", "E3"};
  return n[l];
}

// GPU 학습 스텝 한 번의 입력(같은 양자화 입력)을 모은다
static In grab_inputs(Bc& b, int M) {
  In in;
  in.M = M;
  in.x0 = dl(b.sb.x0, (size_t)M * X0_W);
  in.sin = dl(b.sb.sin, (size_t)M * KSLOT * SLOT_IN);
  in.mask = dl(b.sb.mask, M);
  in.amask = dl(b.amask_d, 1)[0];
  if (b.sn.vision) in.tok = dl(b.sb.tok, (size_t)M * IMG_TOK * vit::TOK_LD);
  if (b.sn.text) {
    const auto x0e = dl(b.sb.x0e, (size_t)M * b.sn.k1);
    in.txt.resize((size_t)M * TXT_W);
    for (int m = 0; m < M; ++m)
      for (int c = 0; c < TXT_W; ++c) in.txt[(size_t)m * TXT_W + c] = x0e[(size_t)m * b.sn.k1 + b.sn.x_txt + c];
  }
  if (b.sn.arch == 1) { in.fm = dl(b.sb.fm, (size_t)M * MAX_H); return in; }
  if (!b.sn.head) in.lab = dl(b.lab_mb, (size_t)M * N_LAB);
  else {
    const auto h3 = dl(b.sb.h[SL_A3], (size_t)M * E_IN);
    in.ein.resize((size_t)M * (FLOW_W + TEMB));
    for (int m = 0; m < M; ++m)
      for (int c = 0; c < FLOW_W + TEMB; ++c) in.ein[(size_t)m * (FLOW_W + TEMB) + c] = h3[(size_t)m * E_IN + E_X + c];
    in.u = dl(b.sb.xf, (size_t)M * FLOW_W);
    in.fm = dl(b.sb.fm, (size_t)M * MAX_H);
  }
  return in;
}

static int run_v5(int bug) {
  BcConfig c = small_cfg(23, 0);
  Bc b(c);
  b.bug = bug;
  b.keep_slot_bufs = true;
  load_aux(&b);
  const StudentNet& sn = b.sn;
  const int N = b.N, M = b.MB;
  int fails = 0, total = 0;
  // 자료: 교사 기록 2 롤아웃 + 학생이 움직이며 기록 1 롤아웃(DAgger 자리)
  bc_set_mode(&b, 0, 1); b.launch(0); b.launch(0);
  bc_set_mode(&b, 1, 1);
  const Data dd_before = dl(b.data_d, 1)[0];
  const long long cur_before = dd_before.cursor;
  b.launch(0);
  VCK(cudaDeviceSynchronize());
  { BcLog L; while (b.poll(&L)) {} }
  const Data dd = dl(b.data_d, 1)[0];
  std::printf("v5 (%s): N %d T %d, data count %lld, minibatch %d, student params %lld, bug %d\n", g_lite ? "student-lite MSE" : g_arch1 ? "token student (arch 1): images + text + body/object/space tokens, flow expert" : "image student: vision + text + flow chunk",
              N, b.T, dd.count, M, (long long)bc_num_params(&b), bug);

  // (1) 기록 = 본 것: 마지막 스텝(t = T−1)의 저장 표본으로 CPU 가 다시 만든 학생 입력 == 롤아웃 때 GPU 학생 입력(비트), 라벨 == clamp(교사 μ),
  //     영상: 저장된 렌더 상태를 같은 묶음 자리로 다시 렌더·인코딩한 토큰 == 롤아웃 때 토큰(비트), 지시 번호 == 해시(저장 에피소드 번호)
  {
    const int t = b.T - 1;
    const auto x0g = dl(b.sb.x0, (size_t)N * X0_W), sing = dl(b.sb.sin, (size_t)N * KSLOT * SLOT_IN);
    const auto mg = dl(b.sb.mask, N);
    const auto mT = dl(b.nt.mean, (size_t)N * N_ACT);
    long long diff = 0, ldiff = 0, tdiff = 0;
    std::vector<uint16_t> obs(env::N_OBS), x0c(X0_W), sinc(KSLOT * SLOT_IN);
    const auto tidg = (sn.text || sn.arch == 1) ? dl(b.sb.tid, N) : std::vector<int>();
    const uint32_t am = dl(b.amask_d, 1)[0];
    const TxtSel tsel = dl(b.tsel_d, 1)[0];
    const obsv::ObsAug aug = dl(b.aug_d, 1)[0];   // 롤아웃 학생 입력과 같은 흔들기·열쇠(롤아웃 번호, 스텝·판) — 목표 표시 감추기 포함
    long long n_gdrop = 0;
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
      const uint64_t ak0 = (uint64_t)dd_before.rollouts, ak1 = ((uint64_t)t << 32) | (uint64_t)i;
      const uint32_t mk = assemble_student(of, 1, 0, tk, x0c.data(), sinc.data(), c.use_map, c.student_goal, 0, 1, b.vt.host(), &aug, ak0, ak1);
      n_gdrop += obsv::goal_drop(&aug, ak0, ak1);
      for (int k = X0_OBS; k < X0_W; ++k) diff += x0c[k] != x0g[(size_t)i * X0_W + k];
      for (int k = 0; k < KSLOT * SLOT_IN; ++k) diff += sinc[k] != sing[(size_t)i * KSLOT * SLOT_IN + k];
      diff += mk != mg[i];
      for (int k = 0; k < N_LAB; ++k) ldiff += lab[k] != (((am >> k) & 1u) ? std::fmin(std::fmax(mT[(size_t)i * N_ACT + k], -1.f), 1.f) : 0.f);
      if (sn.text || sn.arch == 1) {
        uint32_t ep;
        VCK(cudaMemcpy(&ep, b.d_epi + slot, 4, cudaMemcpyDeviceToHost));
        tdiff += tidg[i] != text_row_tok(tk, ep, tsel, 0);
      }
    }
    const bool ok = diff == 0 && ldiff == 0 && tdiff == 0;
    ++total; fails += !ok;
    std::printf("  record == seen: student input rebuilt from %d stored samples vs rollout input: %lld words differ; labels vs clamp(teacher mu): %lld; text ids: %lld; "
                "goal flags hidden in %lld samples (p %.2f)  %s\n", N, diff, ldiff, tdiff, n_gdrop, aug.p_goal_drop, ok ? "ok" : "FAIL");
    if (sn.vision) {
      const long long s0 = (cur_before + (long long)t * N) % b.cap;
      if (s0 + N > b.cap) { std::printf("  (render-state slots wrap; skipped)\n"); }
      else {
        const auto tok_roll = dl(b.sb.tok, (size_t)N * IMG_TOK * vit::TOK_LD);
        b.vis_encode(b.d_rs + s0, N);
        VCK(cudaDeviceSynchronize());
        const auto tok_re = dl(b.sb.tok, (size_t)N * IMG_TOK * vit::TOK_LD);
        long long d = 0;
        for (size_t k = 0; k < tok_re.size(); ++k) d += tok_re[k] != tok_roll[k];
        ++total; fails += d != 0;
        std::printf("  record == seen (images): re-render + re-encode of %d stored render states == rollout tokens: %lld of %zu words differ  %s\n", N, d, tok_re.size(),
                    d == 0 ? "ok" : "FAIL");
      }
    }
  }

  // (2) 학습 한 스텝
  const long long iter0 = dl(b.ts, 1)[0].iter;
  b.gather();
  b.student_trunk(M, b.rs_mb);
  b.head_forward(M);
  b.loss(M);
  b.backward(M);
  VCK(cudaDeviceSynchronize());
  const In in = grab_inputs(b, M);
  // 학생 입력·청크 라벨·flow 입력: CPU 가 같은 표본 번호(장치 열쇠를 호스트에서)로 만든 값 == GPU
  {
    long long diff = 0, cdiff = 0, rsd = 0;
    std::vector<uint16_t> obs(env::N_OBS), x0c(X0_W), sinc(KSLOT * SLOT_IN);
    const auto chunk = sn.head ? dl(b.chunk_mb, (size_t)M * FLOW_W) : std::vector<float>();
    const auto rsg = sn.vision ? dl(b.rs_mb, M) : std::vector<RenderState>();
    const auto meta = dl(b.d_meta, b.cap), epi = dl(b.d_epi, b.cap);
    const auto labs = dl(b.d_lab, (size_t)b.cap * N_LAB);
    int nvalid = 0;
    for (int r = 0; r < M; ++r) {
      const long long idx = (long long)(hash4(c.seed, (uint64_t)iter0, (uint64_t)r, 0x42434d42ull) % (uint64_t)dd.count);
      VCK(cudaMemcpy(obs.data(), b.d_obs + idx * env::N_OBS, 2 * env::N_OBS, cudaMemcpyDeviceToHost));
      gmap::MapTok tk;
      VCK(cudaMemcpy(&tk, b.d_tok + idx, sizeof tk, cudaMemcpyDeviceToHost));
      float of[env::N_OBS];
      for (int k = 0; k < env::N_OBS; ++k) of[k] = bf2f(obs[k]);
      const obsv::ObsAug ag = dl(b.aug_d, 1)[0];   // 모으기 열쇠 = (갱신 스텝 | 1 << 63, 행) — gather_k 와 같음
      const uint32_t mk = assemble_student(of, 1, 0, tk, x0c.data(), sinc.data(), c.use_map, c.student_goal, 0, 1, b.vt.host(), &ag,
                                           (uint64_t)iter0 | 0x8000000000000000ull, (uint64_t)r);
      for (int k = X0_OBS; k < X0_W; ++k) diff += x0c[k] != in.x0[(size_t)r * X0_W + k];
      for (int k = 0; k < KSLOT * SLOT_IN; ++k) diff += sinc[k] != in.sin[(size_t)r * KSLOT * SLOT_IN + k];
      diff += mk != in.mask[r];
      if (sn.head) {
        for (int h = 0; h < MAX_H; ++h) {
          const long long sh = (idx + (long long)h * N) % b.cap;
          const bool ok = h < sn.H && epi[sh] == epi[idx] && (meta[sh] & 0xffffu) == ((meta[idx] & 0xffffu) + (uint32_t)h);
          nvalid += ok;
          cdiff += in.fm[(size_t)r * MAX_H + h] != (ok ? 1.f : 0.f);
          for (int k = 0; k < N_LAB; ++k) cdiff += chunk[((size_t)r * MAX_H + h) * N_LAB + k] != (ok ? labs[sh * N_LAB + k] : 0.f);
        }
      }
      if (sn.vision) {
        RenderState rr;
        VCK(cudaMemcpy(&rr, b.d_rs + idx, sizeof rr, cudaMemcpyDeviceToHost));
        rsd += std::memcmp(&rr, &rsg[r], sizeof rr) != 0;
      }
    }
    const bool ok = diff == 0 && cdiff == 0 && rsd == 0;
    ++total; fails += !ok;
    std::printf("  gather: student input CPU == GPU over %d rows: %lld words differ; chunk labels+mask %lld (valid chunk steps %.1f %%); render states %lld  %s\n", M, diff,
                cdiff, 100.0 * nvalid / (M * (double)MAX_H), rsd, ok ? "ok" : "FAIL");
  }
  if (sn.arch == 1) {   // 토큰마다 학생: 묶음 입력 줄 = X0·지시 표에서 옮긴 값(CPU 가 같은 규칙으로 == GPU), 손실 유한. 신경망 수치는 tf_verify
    const auto txt = dl(b.txt, (size_t)MAX_TXT * TXT_W);
    const auto tid = dl(b.sb.tid, M);
    long long d = 0;
    const uint16_t one = 0x3f80;
    for (int r = 0; r < M; ++r) {
      const uint16_t* x = in.x0.data() + (size_t)r * X0_W;
      auto chk = [&](int g, int K, int kreal, auto val) {
        const auto rows = dl(b.tg[g] + (size_t)r * K, (size_t)K);
        for (int k = 0; k < K; ++k) d += rows[k] != (k < kreal ? (uint16_t)val(k) : k == kreal ? one : (uint16_t)0);
      };
      chk(tfm::G_ARM, 48, 42, [&](int k) { return x[X0_OBS + k]; });
      chk(tfm::G_BASE, 16, 3, [&](int k) { return x[X0_OBS + 42 + k]; });
      chk(tfm::G_GOAL, 16, 15, [&](int k) { return k < 11 ? x[X0_OBS + 45 + k] : x[X0_WAY + k - 11]; });
      chk(tfm::G_WALL, 80, 64, [&](int k) { return k < 56 ? x[X0_OBS + N_OBS_G1 + k] : x[X0_FRONT + k - 56]; });
      chk(tfm::G_ROOM, 16, 10, [&](int k) { return x[X0_OBS + N_OBS_G1 + 56 + k]; });
      chk(tfm::G_TXT, 144, 128, [&](int k) { return txt[(size_t)tid[r] * TXT_W + k]; });
    }
    const float L = dl(b.tf.loss_d, 1)[0];
    const bool ok = d == 0 && std::isfinite(L) && L > 0.f;
    ++total; fails += !ok;
    std::printf("  tf token rows (ARM, BASE, GOAL, WALL, ROOM, TXT) CPU == GPU over %d rows: %lld words differ; flow loss %.6f (finite)  %s\n", M, d, L, ok ? "ok" : "FAIL");
    std::printf("V5 (arch 1 inputs): %d / %d checks pass (transformer numerics: tf_verify)\n", total - fails, total);
    return fails;
  }
  if (sn.head) {   // flow 입력: x_τ(bf16)·목표 u(f32)·시간 임베딩(bf16) — 초월 함수(log, cos, sin, pow)의 CPU·GPU 차는 1 ulp 안
    const auto chunk = dl(b.chunk_mb, (size_t)M * FLOW_W);
    long long bad = 0;
    double worst_u = 0;
    for (int r = 0; r < M; ++r) {
      const float tau = u01(hash4(c.seed ^ 0xF10Full, (uint64_t)iter0, (uint64_t)r, 0x7a75ull));
      for (int j = 0; j < FLOW_W; ++j) {
        const float a = in.fm[(size_t)r * MAX_H + j / N_LAB] > 0.f ? chunk[(size_t)r * FLOW_W + j] : 0.f;
        const float eps = gauss(hash4(c.seed ^ 0xF10Full, (uint64_t)iter0, (uint64_t)r, 0x1000ull + (uint64_t)j));
        const float xt = tau * eps + (1.f - tau) * a, uu = eps - a;
        const int dx = std::abs((int)f2bf(xt) - (int)in.ein[(size_t)r * (FLOW_W + TEMB) + j]);
        bad += dx > 1;
        worst_u = std::fmax(worst_u, std::fabs((double)uu - in.u[(size_t)r * FLOW_W + j]));
      }
      uint16_t te[TEMB];
      temb_write(tau, te);
      for (int k = 0; k < TEMB; ++k) bad += std::abs((int)te[k] - (int)in.ein[(size_t)r * (FLOW_W + TEMB) + FLOW_W + k]) > 1;
    }
    const bool ok = bad == 0 && worst_u < 1e-5;
    ++total; fails += !ok;
    std::printf("  flow inputs CPU == GPU: x_tau / time embedding > 1 bf16 ulp apart: %lld; target u = eps - a worst abs diff %.2e (glibc vs CUDA logf/cosf ulps)  %s\n", bad, worst_u, ok ? "ok" : "FAIL");
  }
  const auto Pf = dl(b.P, sn.total);
  const std::vector<double> P0(Pf.begin(), Pf.end());
  netref::set_tc_model(16, c.dw_chunk);
  Ref R64, Rem;
  ref_run(netref::FP64, sn, P0, in, R64, true);
  ref_run(netref::EMUL, sn, P0, in, Rem, true);
  const double Lg = dl(b.ts, 1)[0].s_pg;
  {
    const double eg = std::fabs(Lg - R64.loss) / R64.loss, ef = std::fabs(Rem.loss - R64.loss) / R64.loss;
    const bool ok = eg <= 2.0 * ef + 2e-6;   // 스칼라 합: GPU 는 나무 합, EMUL 은 차례 합 → 반올림 1e-6 수준 여유
    ++total; fails += !ok;
    std::printf("  %-22s GPU %.6f FP64 %.6f EMUL %.6f  rel GPU %.2e floor %.2e  %s\n", sn.head ? "loss (flow)" : "loss (MSE)", Lg, R64.loss, Rem.loss, eg, ef, ok ? "ok" : "FAIL");
  }
  Check ck;
  const int ow = sn.head ? FLOW_W : N_ACT;
  ck.cmp(sn.head ? "out E3 v (f32)" : "mean A4 (f32)", fv(dl(b.sb.out, (size_t)M * ow)), R64.out, Rem.out, M, sn.head ? FLOW_W : N_LAB, ow);
  std::vector<int> hidden = {SL_A1, SL_A2, SL_A3};
  if (sn.head) { hidden.push_back(SL_E1); hidden.push_back(SL_E2); }
  for (int l : hidden) {
    char nm[32];
    std::snprintf(nm, sizeof nm, "fwd %s (bf16)", lname(l));
    ck.cmp(nm, bfv(dl(b.sb.h[l], (size_t)M * sn.L[l].ldo)), R64.h[l], Rem.h[l], M, sn.L[l].N, sn.L[l].ldo);
  }
  if (sn.vision) ck.cmp("fwd P1 (bf16)", bfv(dl(b.sb.imgf, (size_t)M * IMG_W)), R64.h[SL_P1], Rem.h[SL_P1], M, IMG_W, IMG_W);
  ck.cmp("fwd pool (bf16)", bfv(dl(b.sb.x0, (size_t)M * X0_W)), R64.x0, Rem.x0, M, POOL_W, X0_W);
  std::vector<int> dzs = sn.head ? std::vector<int>{SL_E3, SL_E2, SL_E1, SL_A3, SL_A2, SL_A1} : std::vector<int>{SL_A4, SL_A3, SL_A2, SL_A1};
  if (sn.vision) dzs.push_back(SL_P1);
  for (int l : dzs) {
    char nm[32];
    std::snprintf(nm, sizeof nm, "dZ %s (bf16)", lname(l));
    const size_t rows = l == SL_P1 ? (size_t)M : (size_t)M, w = l == SL_P1 ? IMG_W : sn.L[l].N;
    ck.cmp(nm, bfv(dl(b.sb.dz[l], rows * w)), R64.dz[l], Rem.dz[l], rows, w, w);
  }
  ck.cmp("dpool (f32)", fv(dl(b.sb.dpool, (size_t)M * POOL_W)), R64.dpool, Rem.dpool, M, POOL_W, POOL_W);
  ck.cmp("dZ S2 (bf16)", bfv(dl(b.sb.dz[SL_S2], (size_t)M * KSLOT * S_H)), R64.dz[SL_S2], Rem.dz[SL_S2], (size_t)M * KSLOT, S_H, S_H);
  ck.cmp("dZ S1 (bf16)", bfv(dl(b.sb.dz[SL_S1], (size_t)M * KSLOT * S_H)), R64.dz[SL_S1], Rem.dz[SL_S1], (size_t)M * KSLOT, S_H, S_H);
  const auto Gg = fv(dl(b.G, sn.total));
  for (int l = 0; l < SL_N; ++l) {
    if (!sn.on[l]) continue;
    const long long n = (long long)sn.L[l].N * sn.L[l].K;
    char nm[32];
    std::snprintf(nm, sizeof nm, "grad W %s", lname(l));
    ck.cmp(nm, seg(Gg, sn.off[l], n), seg(R64.grad, sn.off[l], n), seg(Rem.grad, sn.off[l], n), (size_t)n, 1, 1);
  }
  // (3) Adam 1 스텝: 갱신량 ΔP
  const AdamHyper ah{c.adam_b1, c.adam_b2, c.adam_eps, c.max_grad_norm};
  const long long NP = sn.total;
  b.optimizer();
  VCK(cudaDeviceSynchronize());
  {
    const auto P1 = dl(b.P, NP);
    std::vector<double> d64 = P0, dem = P0, m64(NP, 0), v64(NP, 0), mem(NP, 0), vem(NP, 0);
    netref::adam(netref::FP64, d64, R64.grad, m64, v64, 1, c.lr, ah);
    netref::adam(netref::EMUL, dem, Rem.grad, mem, vem, 1, c.lr, ah);
    std::vector<double> dg(NP), df(NP), de(NP);
    for (long long i = 0; i < NP; ++i) { dg[i] = (double)P1[i] - P0[i]; df[i] = d64[i] - P0[i]; de[i] = dem[i] - P0[i]; }
    ck.cmp("Adam 1 step dP", dg, df, de, (size_t)NP, 1, 1);
  }
  // (4) Adam 10 스텝(같은 미니배치·같은 영상 토큰·같은 flow 입력): 처음 변수·Adam 상태로 되돌린 뒤 몸통부터 다시
  {
    VCK(cudaMemcpy(b.P, Pf.data(), sizeof(float) * NP, cudaMemcpyHostToDevice));
    VCK(cudaMemset(b.Am, 0, sizeof(float) * NP));
    VCK(cudaMemset(b.Av, 0, sizeof(float) * NP));
    TrainState s = dl(b.ts, 1)[0];
    s.adam_t = 0;
    s.iter = iter0;   // flow τ·ε 가 같도록(같은 열쇠)
    VCK(cudaMemcpy(b.ts, &s, sizeof s, cudaMemcpyHostToDevice));
    to_bf16(b.P, b.Pb, NP, 0);
    for (int k = 0; k < 10; ++k) {
      b.student_trunk(M, b.rs_mb);
      b.head_forward(M);
      b.loss(M);
      b.backward(M);
      b.optimizer();
      TrainState s2 = dl(b.ts, 1)[0];   // 같은 열쇠로 되돌림(loss 가 iter 를 올림)
      s2.iter = iter0;
      VCK(cudaMemcpy(b.ts, &s2, sizeof s2, cudaMemcpyHostToDevice));
    }
    VCK(cudaDeviceSynchronize());
    const auto P10 = dl(b.P, NP);
    std::vector<double> p64 = P0, pem = P0, m64(NP, 0), v64(NP, 0), mem(NP, 0), vem(NP, 0);
    for (int k = 0; k < 10; ++k) {
      Ref a, e;
      ref_run(netref::FP64, sn, p64, in, a, true);
      netref::adam(netref::FP64, p64, a.grad, m64, v64, k + 1, c.lr, ah);
      ref_run(netref::EMUL, sn, pem, in, e, true);
      netref::adam(netref::EMUL, pem, e.grad, mem, vem, k + 1, c.lr, ah);
    }
    std::vector<double> dg(NP), df(NP), de(NP);
    for (long long i = 0; i < NP; ++i) { dg[i] = (double)P10[i] - P0[i]; df[i] = p64[i] - P0[i]; de[i] = pem[i] - P0[i]; }
    ck.cmp("Adam 10 steps dP", dg, df, de, (size_t)NP, 1, 1);
  }
  fails += ck.fails; total += ck.total;
  // (5) FP64 참조판 대 FP64 중심 유한 차분(층마다 고루 변수 192 개, 행 64 — 꺾인 곳(최댓값 집합·ELU)을 건너면 h 를 1/10·1/100 로 다시 재서 가장 작은 오차)
  {
    const int Mf = 64;
    In s = in;
    s.M = Mf;
    s.x0.resize((size_t)Mf * X0_W); s.sin.resize((size_t)Mf * KSLOT * SLOT_IN); s.mask.resize(Mf);
    if (sn.vision) s.tok.resize((size_t)Mf * IMG_TOK * vit::TOK_LD);
    if (sn.text) s.txt.resize((size_t)Mf * TXT_W);
    if (!sn.head) s.lab.resize((size_t)Mf * N_LAB);
    else { s.ein.resize((size_t)Mf * (FLOW_W + TEMB)); s.u.resize((size_t)Mf * FLOW_W); s.fm.resize((size_t)Mf * MAX_H); }
    Ref g;
    ref_run(netref::FP64, sn, P0, s, g, true);
    std::vector<int> layers;
    for (int l = 0; l < SL_N; ++l) if (sn.on[l]) layers.push_back(l);
    double gs = 0;
    for (int l : layers)
      for (long long q = 0; q < (long long)sn.L[l].N * sn.L[l].K; ++q) gs = std::fmax(gs, std::fabs(g.grad[sn.off[l] + q]));
    uint64_t sd = 99;
    double worst = 0;
    int nz = 0, np = 0;
    for (int j = 0; j < 192; ++j) {
      const int l = layers[j % layers.size()];
      sd = mix64(sd + 1);
      const LayerDesc& L = sn.L[l];
      int ncols = L.bias + 1, k;   // 입력 칸 + 편향 칸
      if (l == SL_A1) ncols = L.K;   // 영상·글 칸(편향 뒤)까지
      if (l == SL_E1) ncols = E_T + TEMB;
      k = (int)((sd >> 20) % (uint64_t)ncols);
      const int n = (int)(sd % (uint64_t)(l == SL_A4 ? N_LAB : L.N));
      const long long q = sn.off[l] + (long long)n * L.K + k;
      ++np;
      if (g.grad[q] != 0) ++nz;
      double best = 1e30;
      for (double h : {1e-4, 1e-5, 1e-6}) {
        std::vector<double> Pp = P0, Pm = P0;
        Pp[q] += h; Pm[q] -= h;
        Ref a, bb;
        ref_run(netref::FP64, sn, Pp, s, a, false);
        ref_run(netref::FP64, sn, Pm, s, bb, false);
        const double fd = (a.loss - bb.loss) / (2 * h);
        const double e = std::fabs(fd - g.grad[q]) / (std::fabs(g.grad[q]) + 1e-3 * gs);
        best = std::fmin(best, e);
        if (best < 1e-6) break;
      }
      worst = std::fmax(worst, best);
    }
    const bool ok = worst < 1e-4;
    ++total; fails += !ok;
    std::printf("  FP64 ref vs finite diff: %d params over %zu layers (%d nonzero grads), worst rel err %.2e  %s\n", np, layers.size(), nz, worst, ok ? "ok" : "FAIL");
  }
  std::printf("v5: %d / %d checks passed\n", total - fails, total);
  return fails;
}

// ---- V6 / V7: 기록(교사) → 갱신 → DAgger 롤아웃(학생 + 기록) → 갱신 → 평가 롤아웃 의 결과 전체를 비트로 ----
struct Snap {
  std::vector<float> P, Am, Av, lab;
  std::vector<uint16_t> obs;
  std::vector<gmap::MapTok> tok;
  std::vector<uint32_t> meta, epi;
  std::vector<RenderState> rs;
  std::vector<unsigned long long> tab;
  std::vector<BcLog> logs;
};
// reseed: 0 = 씨앗 바꾸기 없음(예전 순서), 1 = 기록·갱신 뒤 장치 씨앗 바꾸기(bc_reset_env), 2 = 같은 자리에서 예전 판(동기 + make_env + 그래프 다시 잡기)
static Snap run_seq(BcConfig c, int reseed = 0) {
  Bc b(c);
  load_aux(&b);
  Snap s;
  auto drain = [&] { BcLog L; while (b.poll(&L)) { L.gpu_ms = 0; s.logs.push_back(L); } };
  bc_set_mode(&b, 0, 1);
  for (int k = 0; k < 2; ++k) { b.launch(0); drain(); }
  for (int k = 0; k < 3; ++k) { b.launch(1); drain(); }
  if (reseed) {
    VCK(cudaDeviceSynchronize());
    drain();
    if (reseed == 1) { if (bc_reset_env(&b, c.env_seed + 77) != 0) std::exit(2); }
    else { b.make_env(c.env_seed + 77); VCK(cudaDeviceSynchronize()); if (c.use_graphs) b.capture(); }
  }
  bc_set_mode(&b, 1, 1);
  b.launch(0);
  for (int k = 0; k < 2; ++k) { b.launch(1); drain(); }
  bc_set_mode(&b, 1, 0);
  b.launch(0);
  VCK(cudaDeviceSynchronize());
  drain();
  const bool t1 = b.sn.arch == 1;
  const long long n = t1 ? b.tf.lay.total : b.sn.total, cap = b.cap;
  s.P = dl(t1 ? b.tf.P : b.P, n); s.Am = dl(t1 ? b.tf.Am : b.Am, n); s.Av = dl(t1 ? b.tf.Av : b.Av, n);
  s.lab = dl(b.d_lab, (size_t)cap * N_LAB); s.obs = dl(b.d_obs, (size_t)cap * env::N_OBS); s.tok = dl(b.d_tok, (size_t)cap); s.meta = dl(b.d_meta, (size_t)cap);
  s.epi = dl(b.d_epi, (size_t)cap);
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
      {"meta", cnt(a.meta.data(), b.meta.data(), 4 * a.meta.size())}, {"episode ids", cnt(a.epi.data(), b.epi.data(), 4 * a.epi.size())},
      {"render state", cnt(a.rs.data(), b.rs.data(), sizeof(RenderState) * a.rs.size())},
      {"episode table", cnt(a.tab.data(), b.tab.data(), 8 * a.tab.size())},
      {"logs", a.logs.size() == b.logs.size() ? cnt(a.logs.data(), b.logs.data(), sizeof(BcLog) * a.logs.size()) : 999999}};
  long t = 0;
  for (auto& x : r) { if (verbose) std::printf("  %-14s %ld words differ\n", x.n, x.d); t += x.d; }
  return t;
}
static int run_v6() {
  const Snap g = run_seq(small_cfg(5, 1)), e = run_seq(small_cfg(5, 0));
  std::printf("v6 graph vs eager (%s; record 2, update 3x%d, DAgger rollout 1, update 2, eval rollout 1):\n", g_lite ? "lite" : g_arch1 ? "token student (arch 1)" : "image student", small_cfg(5, 1).upd_steps);
  const long d = diff_snap(g, e, true);
  unsigned long long ne = 0;
  for (size_t k = 0; k < g.tab.size(); k += 6) ne += g.tab[k];
  std::printf("v6: %ld words differ, %llu episodes in table, %zu logs  %s\n", d, ne, g.logs.size(), d == 0 ? "PASS" : "FAIL");
  return d != 0;
}
// 장치 씨앗 바꾸기 == 다시 만들기(예전 bc_reset_env): 교사 기록 2 → 갱신 3 → 씨앗 바꾸기 → DAgger 롤아웃·갱신·평가 롤아웃 — 모든 자료·변수·기록이 비트로 같아야
static int run_reseed() {
  const Snap a = run_seq(small_cfg(5, 1), 1), b = run_seq(small_cfg(5, 1), 2), z = run_seq(small_cfg(5, 1), 0);
  std::printf("reseed (%s): device reseed vs recreate:\n", g_lite ? "lite" : g_arch1 ? "token student (arch 1)" : "image student");
  const long d = diff_snap(a, b, true), dz = diff_snap(a, z, false);
  std::printf("reseed: %ld words differ (must be 0); vs no reseed %ld (must be > 0)  %s\n", d, dz, d == 0 && dz > 0 ? "PASS" : "FAIL");
  return !(d == 0 && dz > 0);
}
static int run_v7() {
  const Snap a = run_seq(small_cfg(7, 1)), b = run_seq(small_cfg(7, 1)), c = run_seq(small_cfg(8, 1));
  const long same = diff_snap(a, b, false), other = diff_snap(a, c, false);
  std::printf("v7 (%s): same seed %ld words differ (must be 0), other seed %ld (must be > 0)  %s\n", g_lite ? "lite" : "image student", same, other,
              same == 0 && other > 0 ? "PASS" : "FAIL");
  return !(same == 0 && other > 0);
}

static int run_bench(int N, int T, int mb, int K) {
  BcConfig c = small_cfg(1, 1);
  c.n_env = N; c.horizon = T; c.mb = mb; c.upd_steps = K; c.cap = (long long)N * T * 2;
  Bc b(c);
  load_aux(&b);
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
  const double r_rec = timed(0, 3);
  bc_set_mode(&b, 1, 0);
  const double r_st = timed(0, 2);
  const double u = timed(1, 3);
  std::printf("bench vit_prec %d (0 FP16/FP32-acc, 1 FP16-acc, 2 FP8)\n", c.vit_prec);
  std::printf("bench %s N %d T %d: rollout teacher+record %.2f ms (%.3g env-step/s), student acts %.2f ms (%.3g env-step/s); update %d steps x mb %d: %.2f ms (%.3f ms/step, %.3g samples/s); device %.2f GB\n",
              g_lite ? "lite" : "image", N, T, r_rec, (double)N * T / (r_rec * 1e-3), r_st, (double)N * T / (r_st * 1e-3), K, mb, u, u / K, (double)mb * K / (u * 1e-3),
              bc_device_bytes(&b) / 1e9);
  if (b.sn.vision) {   // 나눠 재기(즉시 실행, 이벤트): 렌더만, 렌더 + 패치 + 인코더
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto ev = [&](auto&& f, int reps) {
      f();
      cudaEventRecord(e0, 0);
      for (int r = 0; r < reps; ++r) f();
      cudaEventRecord(e1, 0);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      return ms / reps;
    };
    b.gather();
    const float t_vis = ev([&] { b.vis_encode(b.rs_mb, mb); }, 3);
    const float t_rnd = ev([&] { for (int e = 0; e < mb; e += bcr::batch(b.rnd)) bcr::render(b.rnd, b.rs_mb + e, std::min(bcr::batch(b.rnd), mb - e), 0); }, 3);
    std::printf("  per update step (mb %d): render %.2f ms, patchify+encoder %.2f ms, student fwd/bwd/Adam + gather %.2f ms (by subtraction)\n", mb, t_rnd, t_vis - t_rnd,
                u / K - t_vis);
    const float t_vr = ev([&] { b.vis_encode(b.rs_roll, N); }, 2);
    std::printf("  per rollout step (N %d): render+encode %.2f ms of %.2f ms\n", N, t_vr, r_st / T);
    if (getenv("BC_OVERLAP_PROBE")) {   // 렌더(CUDA 코어)와 인코더(텐서 코어)를 두 스트림에 같이 띄우면 얼마나 겹치나(값은 버림 — 시간만)
      cudaStream_t sa, sb2;
      cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&sb2, cudaStreamNonBlocking);
      cudaEvent_t a0, a1, b1;
      cudaEventCreate(&a0); cudaEventCreate(&a1); cudaEventCreate(&b1);
      const int E = bcr::batch(b.rnd);
      auto rend = [&](cudaStream_t s) { for (int e = 0; e < N; e += E) bcr::render(b.rnd, b.rs_roll + e, std::min(E, N - e), s); };
      auto encd = [&](cudaStream_t s) { b.enc.run(2 * N, b.sb.tok, s); };
      auto tm = [&](auto&& f) { f(); VCK(cudaDeviceSynchronize()); cudaEventRecord(a0, sa); cudaStreamWaitEvent(sb2, a0, 0); f(); cudaEventRecord(a1, sa); cudaEventRecord(b1, sb2);
                                cudaStreamWaitEvent(sa, b1, 0); cudaEventRecord(a1, sa); cudaEventSynchronize(a1); float ms = 0; cudaEventElapsedTime(&ms, a0, a1); return ms; };
      const float tr = tm([&] { rend(sa); }), te = tm([&] { encd(sb2); }), tc = tm([&] { rend(sa); encd(sb2); });
      std::printf("  overlap probe (N %d): render %.2f ms, encoder %.2f ms, sum %.2f, both on two streams %.2f ms (hidden %.0f %% of render)\n", N, tr, te, tr + te, tc,
                  100.0 * (tr + te - tc) / tr);
      // 학생 갱신(영상 빼고)과 다음 미니배치 인코더를 같이 띄우면
      b.vis_skip = true;
      auto upd = [&](cudaStream_t) { b.update_step(); };   // 스레드 기본 스트림(per-thread)
      auto enc2 = [&](cudaStream_t s) { b.enc.run(2 * mb, b.sb.tok, s); };
      auto tm2 = [&](auto&& f) { f(); VCK(cudaDeviceSynchronize()); cudaEventRecord(a0, cudaStreamPerThread); cudaStreamWaitEvent(sb2, a0, 0); f();
                                 cudaEventRecord(b1, sb2); cudaStreamWaitEvent(cudaStreamPerThread, b1, 0); cudaEventRecord(a1, cudaStreamPerThread); cudaEventSynchronize(a1);
                                 float ms = 0; cudaEventElapsedTime(&ms, a0, a1); return ms; };
      const float ts_ = tm2([&] { upd(0); }), te2 = tm2([&] { enc2(sb2); }), tc2 = tm2([&] { upd(0); enc2(sb2); });
      b.vis_skip = false;
      std::printf("  overlap probe (mb %d): student step %.2f ms, encoder %.2f ms, sum %.2f, both %.2f ms\n", mb, ts_, te2, ts_ + te2, tc2);
    }
  }
  return 0;
}

// 렌더 흔들기·섞기의 인코더 쪽 효과(VLA_INPUT 6절, README "싼 설정 대 팀 기본 코사인 0.72"): 교사 기록 1 롤아웃의 렌더 상태 128 개를
// (A) 싼 설정 (B) 팀 기본 (C) 싼 설정 + 흔들기 (D) 팀 기본 + 같은 흔들기 로 그려 얼린 인코더 패치 토큰(128 × 768 / 상태)의 토큰별 코사인을 낸다
static void cos_stats(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, long long ntok, double& mean, double& p01) {
  std::vector<double> cs((size_t)ntok);
  for (long long t = 0; t < ntok; ++t) {
    double xy = 0, xx = 0, yy = 0;
    for (int k = 0; k < vit::D; ++k) {
      const double x = bf2f(a[(size_t)t * vit::TOK_LD + k]), y = bf2f(b[(size_t)t * vit::TOK_LD + k]);
      xy += x * y; xx += x * x; yy += y * y;
    }
    cs[(size_t)t] = xy / std::sqrt(std::fmax(xx * yy, 1e-30));
  }
  mean = 0;
  for (double c : cs) mean += c;
  mean /= (double)ntok;
  std::sort(cs.begin(), cs.end());
  p01 = cs[(size_t)(0.01 * (double)ntok)];
}
static int run_rquality(float color, float light, float expo) {
  BcConfig c = small_cfg(31, 0);
  Bc b(c);
  load_aux(&b);
  bc_set_mode(&b, 0, 1);
  b.launch(0);
  VCK(cudaDeviceSynchronize());
  { BcLog L; while (b.poll(&L)) {} }
  const int E = 128;
  const long long s0 = (dl(b.data_d, 1)[0].cursor + (long long)b.cap - (long long)b.N) % b.cap;   // 마지막 스텝 표본들(판 0..127)
  const RenderState* rs = b.d_rs + s0;
  const bcr::RenderAug off{0, 0.f, 0.f, 0.f}, on{1, color, light, expo};
  std::vector<uint16_t> tk[4];
  const int prof[4] = {bcr::CHEAP, bcr::TEAM_DEFAULT, bcr::CHEAP, bcr::TEAM_DEFAULT};
  for (int q = 0; q < 4; ++q) {
    bcr::Renderer* R = bcr::create(E, prof[q], q < 2 ? &off : &on);
    bcr::render(R, rs, E, cudaStreamPerThread);
    b.enc.patchify(bcr::rgb(R, 0), bcr::rgb(R, 1), E, 0, cudaStreamPerThread);
    b.enc.run(2 * E, b.sb.tok, cudaStreamPerThread);
    VCK(cudaDeviceSynchronize());
    tk[q] = dl(b.sb.tok, (size_t)E * IMG_TOK * vit::TOK_LD);
    bcr::destroy(R);
  }
  const long long nt = (long long)E * IMG_TOK;
  const char* nm[4] = {"cheap", "team default", "cheap + aug", "team default + aug"};
  const int pr[5][2] = {{0, 1}, {0, 2}, {2, 3}, {1, 3}, {2, 1}};
  std::printf("render quality (%d render states x 2 cameras, frozen SigLIP 2 patch tokens, aug color %.2f light %.2f expo %.2f):\n", E, color, light, expo);
  for (auto& p : pr) {
    double m, lo;
    cos_stats(tk[p[0]], tk[p[1]], nt, m, lo);
    std::printf("  %-20s vs %-20s token cosine mean %.4f, low 1 %% %.4f\n", nm[p[0]], nm[p[1]], m, lo);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: bc_verify v5 [--negative] | v6 | v7 | bench [N T mb K]  [--lite] [--teacher CKPT] [--text F32]\n"); return 2; }
  bool neg = false;
  std::vector<std::string> pos;
  {
    const char* src = __FILE__;
    std::string d(src);
    d = d.substr(0, d.rfind('/'));
    g_text = d + "/../data/instr_a2.f32";
  }
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--negative") neg = true;
    else if (a == "--lite") g_lite = true;
    else if (a == "--act8") g_act8 = true;
    else if (a == "--arch1") g_arch1 = true;
    else if (a == "--aug") g_aug = true;
    else if (a == "--raug") g_raug = true;
    else if (a == "--gdrop") g_gdrop = true;
    else if (a == "--txt") g_txt = true;
    else if (a == "--stage" && i + 1 < argc) g_stage = std::atoi(argv[++i]);
    else if (a == "--teacher" && i + 1 < argc) g_teacher = argv[++i];
    else if (a == "--text" && i + 1 < argc) g_text = argv[++i];
    else pos.push_back(a);
  }
  const std::string m = pos[0];
  if (m == "v5") {
    if (!neg) return run_v5(0) ? 1 : 0;
    std::vector<int> bugs = g_lite ? std::vector<int>{1, 2} : std::vector<int>{2, 3, 4, 5};
    const char* names[6] = {"", "MSE grad factor 2 dropped", "ELU' A3->A2 dropped", "flow-loss grad factor 2 dropped", "ELU' of P1 (image dX) dropped",
                            "ELU' E1->trunk dropped"};
    bool all = true;
    std::vector<int> f;
    for (int bg : bugs) f.push_back(run_v5(bg));
    for (size_t k = 0; k < bugs.size(); ++k) {
      std::printf("v5 --negative: bug %d (%s): %d failed checks\n", bugs[k], names[bugs[k]], f[k]);
      all = all && f[k] > 0;
    }
    std::printf("v5 --negative: %s\n", all ? "PASS (every bug fails)" : "FAIL");
    return all ? 0 : 1;
  }
  if (m == "v6") return run_v6();
  if (m == "reseed") return run_reseed();
  if (m == "rquality") return run_rquality(argc > 2 ? (float)std::atof(argv[2]) : 0.8f, argc > 3 ? (float)std::atof(argv[3]) : 0.8f, argc > 4 ? (float)std::atof(argv[4]) : 0.8f);
  if (m == "v7") return run_v7();
  if (m == "bench")
    return run_bench(pos.size() > 1 ? std::atoi(pos[1].c_str()) : 4096, pos.size() > 2 ? std::atoi(pos[2].c_str()) : 64, pos.size() > 3 ? std::atoi(pos[3].c_str()) : 8192,
                     pos.size() > 4 ? std::atoi(pos[4].c_str()) : 50);
  return 2;
}
