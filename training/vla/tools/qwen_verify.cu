// Qwen3.5 몸통 검증(MAPVLA_SPEC M1): HF FP32 기준값(tools/qwen_ref.py 덤프)과 층별 코사인, 탐욕 디코딩 토큰, 음성 대조, V6(그래프 = 즉시), V7(결정성), 속도.
//   qwen_verify cos  MODEL REF [--neg]     층별·끝 코사인(토큰마다, 평균·하위 1 %·최저). 통과: 모든 층 평균 ≥ 0.999·하위 1 % ≥ 0.99. --neg: 버그 1–4 가 실패해야 통과
//   qwen_verify gen  MODEL REF [N]         탐욕 N 토큰(기본 24)이 HF 와 같은지, 글로 풀어 보임
//   qwen_verify rope MODEL REF           RoPE 위치별 검사(tools/qwen_rope_ref.py 덤프): 우리 정규화 + RoPE 커널 대 HF, 음성 대조, 끝에서 끝 위치별 오차
//   qwen_verify v67  MODEL                 같은 입력 두 번 비트 동일, CUDA 그래프 == 즉시
//   qwen_verify bench MODEL B L            prefix 앞 계산 시간·장치 메모리, 디코딩 토큰/s
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "qkern.cuh"
#include "qwen.h"

using namespace rvla;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <class T>
static bool rd(const std::string& p, std::vector<T>& v) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const size_t n = (size_t)f.tellg();
  v.resize(n / sizeof(T));
  f.seekg(0);
  f.read((char*)v.data(), n);
  return true;
}
static int n_prompts(const std::string& ref) {
  int i = 0;
  std::vector<int> v;
  while (rd(ref + "/p" + std::to_string(i) + "_ids.i32", v)) ++i;
  return i;
}

struct Stat { double mean, p1, mn; size_t n; };
static Stat stat(std::vector<double> c) {
  std::sort(c.begin(), c.end());
  double s = 0;
  for (double x : c) s += x;
  Stat r{s / c.size(), c[(size_t)std::floor(0.01 * (c.size() - 1))], c[0], c.size()};
  return r;
}

static bool run_cos(Qwen& m, const std::string& ref, bool verbose) {
  const int np = n_prompts(ref), H = m.c.H, NL = m.c.layers + 2;
  std::vector<std::vector<double>> cs(NL);
  int* dids;
  float *taps, *hid;
  CK(cudaMalloc(&dids, sizeof(int) * 4096));
  CK(cudaMalloc(&taps, sizeof(float) * (size_t)NL * 1024 * H));
  CK(cudaMalloc(&hid, sizeof(float) * (size_t)1024 * H));
  for (int p = 0; p < np; ++p) {
    std::vector<int> ids;
    rd(ref + "/p" + std::to_string(p) + "_ids.i32", ids);
    const int n = (int)ids.size();
    CK(cudaMemcpy(dids, ids.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    m.forward(dids, nullptr, 1, n, 0, nullptr, hid, taps, 0);
    std::vector<float> g((size_t)NL * n * H);
    CK(cudaMemcpy(g.data(), taps, sizeof(float) * g.size(), cudaMemcpyDeviceToHost));
    for (int l = 0; l < NL; ++l) {
      std::vector<float> r;
      rd(ref + "/p" + std::to_string(p) + "_h" + std::to_string(l) + ".f32", r);
      for (int t = 0; t < n; ++t) {
        double ab = 0, aa = 0, bb = 0;
        for (int c = 0; c < H; ++c) {
          const double a = g[((size_t)l * n + t) * H + c], b = r[(size_t)t * H + c];
          ab += a * b; aa += a * a; bb += b * b;
        }
        cs[l].push_back(ab / std::sqrt(aa * bb + 1e-300));
      }
    }
  }
  bool ok = true;
  double wm = 1, wp = 1;
  for (int l = 0; l < NL; ++l) {
    Stat s = stat(cs[l]);
    const bool pass = s.mean >= 0.999 && s.p1 >= 0.99;
    ok = ok && pass;
    wm = std::min(wm, s.mean); wp = std::min(wp, s.p1);
    if (verbose) std::printf("  %-8s tokens %zu  cos mean %.6f  bottom1%% %.6f  min %.6f %s\n",
                             l == 0 ? "embed" : l == NL - 1 ? "final" : ("layer" + std::to_string(l)).c_str(), s.n, s.mean, s.p1, s.mn, pass ? "" : "FAIL");
  }
  std::printf("[cos] bug %d: worst layer mean %.6f, worst bottom1%% %.6f -> %s\n", m.bug, wm, wp, ok ? "PASS" : "FAIL");
  cudaFree(dids); cudaFree(taps); cudaFree(hid);
  return ok;
}


// RoPE 검사(코사인 기준이 θ × 10 을 놓친 것을 보완): HF 의 풀 층 q_proj·k_proj 출력을 우리 q·k 머리 정규화 + RoPE 커널(qk::full_prep)에 넣어
// 위치마다 HF 의 RoPE 뒤 q·k 와 비교(머리 벡터마다 상대 L2, 위치 최대 600). 음성 대조(θ × 1.01·× 10·1e4, 회전 32 차원, 위치 + 1)는 실패해야 함.
// 끝에서 끝: 우리 앞 계산 전체의 RoPE 뒤 q·k 대 HF(bf16 GEMM 잡음 포함)를 위치 구간별로 — 위치가 커질수록 오차가 자라지 않는지.
static bool run_rope(Qwen& m, const std::string& ref) {
  int ns = 0;
  { std::vector<int> v; while (rd(ref + "/r" + std::to_string(ns) + "_ids.i32", v)) ++ns; }
  if (!ns) { std::printf("[rope] no r*_ids.i32 in %s (run tools/qwen_rope_ref.py)\n", ref.c_str()); return false; }
  const int nq = m.c.nq, nkv = m.c.nkv, hd = m.c.hd, qg = m.c.qg(), kvw = m.c.kvw(), ldT = qg + 2 * kvw, nf = m.c.n_full();
  std::vector<int> fl;
  for (int i = 0; i < m.c.layers; ++i) if (m.c.full[i]) fl.push_back(i);
  float *T0, *Q, *K, *V, *taps, *hid;
  int* dids;
  CK(cudaMalloc(&T0, sizeof(float) * 1024 * ldT)); CK(cudaMalloc(&Q, sizeof(float) * 1024 * nq * hd)); CK(cudaMalloc(&K, sizeof(float) * 1024 * kvw));
  CK(cudaMalloc(&V, sizeof(float) * 1024 * kvw)); CK(cudaMalloc(&taps, sizeof(float) * (size_t)nf * 1024 * (nq * hd + kvw)));
  CK(cudaMalloc(&hid, sizeof(float) * 1024 * m.c.H)); CK(cudaMalloc(&dids, sizeof(int) * 1024));
  struct Var { const char* nm; float theta; int rot, dp; };
  const Var vars[] = {{"ours", m.c.theta, m.c.rot, 0}, {"theta x1.01", m.c.theta * 1.01f, m.c.rot, 0}, {"theta x10", m.c.theta * 10.f, m.c.rot, 0},
                      {"theta 1e4", 1e4f, m.c.rot, 0}, {"rotary 32 dims", m.c.theta, m.c.rot / 2, 0}, {"position + 1", m.c.theta, m.c.rot, 1}};
  bool ok = true;
  for (const Var& vr : vars) {
    double worst = 0, wq = 0, wk = 0;
    int wpos = -1;
    for (int si = 0; si < ns; ++si) {
      std::vector<int> ids;
      rd(ref + "/r" + std::to_string(si) + "_ids.i32", ids);
      const int n = (int)ids.size();
      for (int f = 0; f < nf; ++f) {
        std::vector<float> pq, pk, rq, rk;
        const std::string b = ref + "/r" + std::to_string(si) + "_";
        rd(b + "pq" + std::to_string(f) + ".f32", pq); rd(b + "pk" + std::to_string(f) + ".f32", pk);
        rd(b + "rq" + std::to_string(f) + ".f32", rq); rd(b + "rk" + std::to_string(f) + ".f32", rk);
        std::vector<float> t0((size_t)n * ldT, 0.f);
        for (int t = 0; t < n; ++t) {
          std::memcpy(&t0[(size_t)t * ldT], &pq[(size_t)t * qg], sizeof(float) * qg);
          std::memcpy(&t0[(size_t)t * ldT + qg], &pk[(size_t)t * kvw], sizeof(float) * kvw);
        }
        CK(cudaMemcpy(T0, t0.data(), sizeof(float) * t0.size(), cudaMemcpyHostToDevice));
        std::vector<int> poff(1, vr.dp);
        int* dpo = nullptr;
        if (vr.dp) { CK(cudaMalloc(&dpo, sizeof(int))); CK(cudaMemcpy(dpo, poff.data(), sizeof(int), cudaMemcpyHostToDevice)); }
        const auto& L = m.lay.l[fl[f]];
        qk::full_prep(T0, ldT, n, 1, n, 0, nq, nkv, hd, vr.rot, vr.theta, m.c.eps, m.Pv + L.qn, m.Pv + L.kn, false, Q, K, V, n, 0, dpo);
        std::vector<float> oq((size_t)n * nq * hd), ok2((size_t)n * kvw);
        CK(cudaMemcpy(oq.data(), Q, sizeof(float) * oq.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ok2.data(), K, sizeof(float) * ok2.size(), cudaMemcpyDeviceToHost));
        if (dpo) cudaFree(dpo);
        for (int t = 0; t < n; ++t) {
          for (int h = 0; h < nq + nkv; ++h) {
            const float* a = h < nq ? &oq[((size_t)t * nq + h) * hd] : &ok2[(size_t)t * kvw + (h - nq) * hd];
            const float* r = h < nq ? &rq[((size_t)t * nq + h) * hd] : &rk[(size_t)t * kvw + (h - nq) * hd];
            double e = 0, z = 0;
            for (int d = 0; d < hd; ++d) { e += (double)(a[d] - r[d]) * (a[d] - r[d]); z += (double)r[d] * r[d]; }
            const double rel = std::sqrt(e / std::max(z, 1e-30));
            if (rel > worst) { worst = rel; wpos = t; }
            if (h < nq) wq = std::max(wq, rel); else wk = std::max(wk, rel);
          }
        }
      }
    }
    const bool pass = worst <= 2e-4;
    const bool want = vr.nm == std::string("ours");
    std::printf("[rope] isolated norm+RoPE, %-15s worst per-head rel err %.2e (q %.2e, k %.2e) at position %d -> %s%s\n", vr.nm, worst, wq, wk, wpos,
                pass ? "PASS" : "FAIL", want ? "" : (pass ? "  (negative NOT caught)" : "  (negative caught)"));
    ok = ok && (want ? pass : !pass);
  }
  // 끝에서 끝: 우리 전체 앞 계산(bf16 GEMM 입력)의 RoPE 뒤 q·k
  m.qk_taps = taps;
  std::vector<double> bin_e(4, 0.0), bin_n(4, 0.0);
  double e2e_worst = 0;
  for (int si = 0; si < ns; ++si) {
    std::vector<int> ids;
    rd(ref + "/r" + std::to_string(si) + "_ids.i32", ids);
    const int n = (int)ids.size();
    CK(cudaMemcpy(dids, ids.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    m.forward(dids, nullptr, 1, n, 0, nullptr, hid, nullptr, 0);
    CK(cudaDeviceSynchronize());
    const size_t per = (size_t)n * (nq * hd + kvw);
    std::vector<float> tp(per * nf);
    CK(cudaMemcpy(tp.data(), taps, sizeof(float) * tp.size(), cudaMemcpyDeviceToHost));
    for (int f = 0; f < nf; ++f) {
      std::vector<float> rq, rk;
      const std::string b = ref + "/r" + std::to_string(si) + "_";
      rd(b + "rq" + std::to_string(f) + ".f32", rq); rd(b + "rk" + std::to_string(f) + ".f32", rk);
      for (int t = 0; t < n; ++t)
        for (int h = 0; h < nq + nkv; ++h) {
          const float* a = h < nq ? &tp[f * per + ((size_t)t * nq + h) * hd] : &tp[f * per + (size_t)n * nq * hd + (size_t)t * kvw + (h - nq) * hd];
          const float* r = h < nq ? &rq[((size_t)t * nq + h) * hd] : &rk[(size_t)t * kvw + (h - nq) * hd];
          double e = 0, z = 0;
          for (int d = 0; d < hd; ++d) { e += (double)(a[d] - r[d]) * (a[d] - r[d]); z += (double)r[d] * r[d]; }
          const double rel = std::sqrt(e / std::max(z, 1e-30));
          const int bin = t < 64 ? 0 : t < 192 ? 1 : t < 384 ? 2 : 3;
          bin_e[bin] += rel; bin_n[bin] += 1;
          e2e_worst = std::max(e2e_worst, rel);
        }
    }
  }
  m.qk_taps = nullptr;
  const double m0 = bin_e[0] / std::max(1.0, bin_n[0]), m3 = bin_e[3] / std::max(1.0, bin_n[3]);
  const bool e2e_ok = e2e_worst < 0.05 && m3 < 3 * m0;
  std::printf("[rope] end-to-end q/k after RoPE vs HF (bf16 GEMM noise included), mean per-head rel err by position: [0,64) %.2e  [64,192) %.2e  [192,384) %.2e  [384,600) %.2e, worst %.2e -> %s (rule: worst < 0.05, last bin < 3 x first)\n",
              m0, bin_e[1] / std::max(1.0, bin_n[1]), bin_e[2] / std::max(1.0, bin_n[2]), m3, e2e_worst, e2e_ok ? "PASS" : "FAIL");
  ok = ok && e2e_ok;
  std::printf("[rope] %s\n", ok ? "PASS" : "FAIL");
  cudaFree(T0); cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(taps); cudaFree(hid); cudaFree(dids);
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: qwen_verify cos|gen|v67|bench MODEL [REF] ...\n"); return 2; }
  const std::string cmd = argv[1], mdir = argv[2];
  Qwen m;
  std::string err;
  const int rmax = cmd == "bench" ? std::atoi(argv[3]) * std::atoi(argv[4]) : 1024;
  if (!m.load(mdir, rmax, &err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 2; }
  std::printf("model %s: H %d I %d layers %d (full %d), vocab %d, weights %.1f MB bf16 + %.2f MB f32, device %.2f GB\n", mdir.c_str(), m.c.H, m.c.I,
              m.c.layers, m.c.n_full(), m.c.vocab, m.lay.nW * 2 / 1e6, m.lay.nV * 4 / 1e6, m.bytes / 1e9);
  if (cmd == "cos") {
    const std::string ref = argv[3];
    const bool neg = argc > 4 && std::string(argv[4]) == "--neg";
    bool ok = run_cos(m, ref, true);
    if (neg) {
      const char* nm[] = {"", "RoPE theta 1e4 (old default)", "MLP pre-RMSNorm skipped", "DeltaNet decay skipped", "q/k norm skipped", "RoPE theta x10 (weak, reported)"};
      int caught = 0;
      for (int b = 1; b <= 5; ++b) {
        m.bug = b;
        std::printf("negative %d (%s): ", b, nm[b]);
        const bool p = run_cos(m, ref, false);
        if (b <= 4) caught += p ? 0 : 1;
      }
      m.bug = 0;
      std::printf("[neg] caught %d / 4\n", caught);
      ok = ok && caught == 4;
    }
    return ok ? 0 : 1;
  }
  if (cmd == "rope") return run_rope(m, argv[3]) ? 0 : 1;
  if (cmd == "gen") {
    const std::string ref = argv[3];
    const int N = argc > 4 ? std::atoi(argv[4]) : 24;
    Vocab voc;
    const bool hv = voc.load(ref + "/vocab.bin");
    const int np = n_prompts(ref);
    int same = 0, tf_all = 0, tf_tot = 0;
    double worst_all = 0;
    for (int p = 0; p < np; ++p) {
      std::vector<int> ids, gref;
      rd(ref + "/p" + std::to_string(p) + "_ids.i32", ids);
      rd(ref + "/p" + std::to_string(p) + "_gen.i32", gref);
      double ms = 0;
      auto out = generate(m, {ids}, N, -1, 0.f, 1.f, 0, &ms);
      int k = 0;
      while (k < (int)out[0].size() && k < (int)gref.size() && out[0][k] == gref[k]) ++k;
      const bool eq = k == std::min<int>(N, gref.size());
      // HF 로짓으로 갈림 자리의 차(HF 가 고른 토큰 − 우리 토큰). 작으면 거의 같은 값끼리의 갈림
      std::vector<float> glog;
      const bool hl = rd(ref + "/p" + std::to_string(p) + "_glog.f32", glog);
      const size_t V = m.c.vocab;
      // 교사 강요: 프롬프트 + HF 토큰으로 한 번 앞 → 스텝마다 우리 argmax == HF 토큰?
      std::vector<int> full = ids;
      full.insert(full.end(), gref.begin(), gref.end() - 1);
      const int nf = (int)full.size(), ng = (int)gref.size();
      int *dd, *am;
      float *hh, *lg;
      CK(cudaMalloc(&dd, sizeof(int) * nf)); CK(cudaMalloc(&am, sizeof(int) * ng));
      CK(cudaMalloc(&hh, sizeof(float) * nf * m.c.H)); CK(cudaMalloc(&lg, sizeof(float) * V * ng));
      CK(cudaMemcpy(dd, full.data(), sizeof(int) * nf, cudaMemcpyHostToDevice));
      m.forward(dd, nullptr, 1, nf, 0, nullptr, hh, nullptr, 0);
      m.logits(hh + (size_t)(nf - ng) * m.c.H, ng, lg, 0);
      std::vector<float> ol(V * ng);
      CK(cudaMemcpy(ol.data(), lg, sizeof(float) * ol.size(), cudaMemcpyDeviceToHost));
      int tf = 0;
      double worst_gap = 0;
      for (int s2 = 0; s2 < ng; ++s2) {
        const float* l = ol.data() + V * s2;
        int a = 0;
        for (size_t v = 1; v < V; ++v) if (l[v] > l[a]) a = (int)v;
        if (a == gref[s2]) { ++tf; continue; }
        const double gap = hl ? glog[V * s2 + gref[s2]] - glog[V * s2 + a] : -1;
        worst_gap = std::max(worst_gap, gap);
        std::printf("   teacher-forced step %d: ours %d vs HF %d, HF logit gap %.4f, our gap %.4f\n", s2, a, gref[s2], gap, l[a] - l[gref[s2]]);
      }
      cudaFree(dd); cudaFree(am); cudaFree(hh); cudaFree(lg);
      tf_all += tf; tf_tot += ng;
      worst_all = std::max(worst_all, worst_gap);
      same += eq;
      std::printf("prompt %d (%zu tok): free-run match %d / %zu %s, teacher-forced %d / %d, %.2f ms/tok\n", p, ids.size(), k, gref.size(), eq ? "SAME" : "DIFF",
                  tf, ng, ms);
      if (!eq && hl) std::printf("   first free-run divergence at step %d: HF logit gap (HF token - ours) %.4f\n", k, glog[V * k + gref[k]] - glog[V * k + out[0][k]]);
      if (hv) std::printf("   ours: %s\n   HF  : %s\n", voc.decode(out[0]).c_str(), voc.decode(gref).c_str());
      std::printf("   ids:"); for (int x : out[0]) std::printf(" %d", x); std::printf("\n");
    }
    const bool okg = worst_all < 0.05;
    std::printf("[gen] free-run identical %d / %d prompts, teacher-forced argmax = HF %d / %d steps, largest HF logit gap at a mismatch %.4f -> %s (rule: every mismatch is a near-tie, gap < 0.05)\n", same, np, tf_all, tf_tot, worst_all, okg ? "PASS" : "FAIL");
    return okg ? 0 : 1;
  }
  if (cmd == "v67") {
    const int B = 4, n = 200;
    std::vector<int> ids(B * n);
    for (int i = 0; i < B * n; ++i) ids[i] = (i * 7919 + 13) % m.c.vocab;
    int* dids;
    float *h0, *h1, *h2;
    CK(cudaMalloc(&dids, sizeof(int) * B * n));
    CK(cudaMalloc(&h0, sizeof(float) * B * n * m.c.H)); CK(cudaMalloc(&h1, sizeof(float) * B * n * m.c.H)); CK(cudaMalloc(&h2, sizeof(float) * B * n * m.c.H));
    CK(cudaMemcpy(dids, ids.data(), sizeof(int) * B * n, cudaMemcpyHostToDevice));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    m.forward(dids, nullptr, B, n, 0, nullptr, h0, nullptr, st);
    m.forward(dids, nullptr, B, n, 0, nullptr, h1, nullptr, st);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    m.forward(dids, nullptr, B, n, 0, nullptr, h2, nullptr, st);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaMemset(h2, 0, sizeof(float) * B * n * m.c.H));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    const size_t N = (size_t)B * n * m.c.H;
    std::vector<uint32_t> a(N), b(N), c(N);
    CK(cudaMemcpy(a.data(), h0, N * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), h1, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(), h2, N * 4, cudaMemcpyDeviceToHost));
    size_t d7 = 0, d6 = 0;
    for (size_t i = 0; i < N; ++i) { d7 += a[i] != b[i]; d6 += a[i] != c[i]; }
    // 묶음 무관: 열 1 개만 따로 == 묶음 안 열
    m.forward(dids + n, nullptr, 1, n, 0, nullptr, h1, nullptr, st);
    CK(cudaStreamSynchronize(st));
    std::vector<uint32_t> s1((size_t)n * m.c.H);
    CK(cudaMemcpy(s1.data(), h1, s1.size() * 4, cudaMemcpyDeviceToHost));
    size_t db = 0;
    for (size_t i = 0; i < s1.size(); ++i) db += s1[i] != a[(size_t)n * m.c.H + i];
    std::printf("[v7] same input twice: %zu / %zu words differ\n[v6] graph vs eager: %zu differ\n[batch] row alone vs in batch of %d: %zu differ\n", d7, N, d6, B, db);
    return (d7 == 0 && d6 == 0) ? 0 : 1;
  }
  if (cmd == "bench") {
    const int B = std::atoi(argv[3]), n = std::atoi(argv[4]);
    std::vector<int> ids((size_t)B * n);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int)((i * 7919 + 13) % m.c.vocab);
    int* dids;
    float* h;
    CK(cudaMalloc(&dids, sizeof(int) * ids.size()));
    CK(cudaMalloc(&h, sizeof(float) * ids.size() * m.c.H));
    CK(cudaMemcpy(dids, ids.data(), sizeof(int) * ids.size(), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int w = 0; w < 2; ++w) m.forward(dids, nullptr, B, n, 0, nullptr, h, nullptr, 0);
    const int reps = 5;
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) m.forward(dids, nullptr, B, n, 0, nullptr, h, nullptr, 0);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= reps;
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    const double fl = 2.0 * (double)(m.lay.nW - (long long)m.c.vocab * m.c.H) * B * n;
    std::printf("[bench] prefix B %d L %d: %.2f ms (%.3f ms/sample), GEMM-FLOP rate %.1f TFLOPS, model+work %.2f GB, device used %.2f GB\n", B, n, ms, ms / B,
                fl / (ms * 1e-3) / 1e12, m.bytes / 1e9, (tot - fr) / 1e9);
    std::vector<int> pr(64);
    for (int i = 0; i < 64; ++i) pr[i] = ids[i];
    double mpt = 0;
    generate(m, {pr}, 64, -1, 0.f, 1.f, 0, &mpt);
    std::printf("[bench] greedy decode B 1: %.2f ms/token = %.0f tok/s\n", mpt, 1000.0 / mpt);
    if (B >= 8) {
      std::vector<std::vector<int>> prs(8, pr);
      generate(m, prs, 64, -1, 0.f, 1.f, 0, &mpt);
      std::printf("[bench] greedy decode B 8: %.2f ms/step = %.0f tok/s total\n", mpt, 8000.0 / mpt);
    }
    return 0;
  }
  return 2;
}
