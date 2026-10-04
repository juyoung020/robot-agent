// 영상 경로 검증·측정(G5 영상 학생): 렌더 그래프 잡기(작업 1), 얼린 SigLIP 2 인코더(작업 2), 속도·품질.
//   vit_verify render-graph [E]       : 팀 RenderBatch 를 그래프로 — (a) 작업 공간 없이 바로 잡기(실패해야 함, 자식 과정에서)
//                                       (b) 한 번 그린 뒤 잡기: 그래프 == 즉시 실행 비트, 두 설정 시간(즉시 / 그래프)
//   vit_verify dump DIR [n]           : A2 판 n 개(싼 설정)를 그려 DIR/imgs.u8 (n×2 장 [256][256][3]) + 몇 장 PPM — Python 기준값 입력
//   vit_verify enc DIR [--negative]   : GPU 토큰 대 (1) PyTorch FP32 기준값(DIR/ref_tok.f32, tools/siglip_ref.py) 코사인, (2) CPU FP64·EMUL 참조판
//                                       (GPU 오차 ≤ 2 × 바닥), (3) 두 번 비트 같음·묶음 크기 무관, --negative 는 버그 3 개가 각각 실패해야 0
//   vit_verify bench [n_img]          : 인코더 처리량(즉시·그래프, 단계별), 렌더 256² 처리량
//   vit_verify quality [n]            : 같은 상태를 싼 설정·팀 기본으로 → 화소 차, 패치 토큰 코사인(싼 설정을 쓰는 값)
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bc_data.h"
#include "bc_render.h"
#include "env_api.h"
#include "vit.h"
#include "vit_ref.h"

#define VCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(3); } } while (0)

static const cudaStream_t ST = cudaStreamPerThread;

__global__ void rs_from_env_k(env::Soa s, int n, bc::RenderState* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bc::RenderState r;
  bc::render_state(s, i, r);
  out[i] = r;
}
// A2 판 n 개의 렌더 상태(리셋 직후 + 무작위 행동 몇 스텝 — 방·가구·컵·자세가 판마다 다름)
static bc::RenderState* a2_states(int n, uint64_t seed, int steps = 6) {
  env::DeviceEnv de(n, 2, seed);
  float *act, *obs, *rew;
  int* done;
  VCK(cudaMalloc(&act, sizeof(float) * 8 * n)); VCK(cudaMalloc(&obs, sizeof(float) * env::N_OBS * n));
  VCK(cudaMalloc(&rew, sizeof(float) * n)); VCK(cudaMalloc(&done, sizeof(int) * n));
  std::vector<float> ha((size_t)8 * n, 0.f);
  uint64_t s = seed;
  for (int k = 0; k < steps; ++k) {
    for (int i = 0; i < n; ++i) { s = net::mix64(s + i); ha[i] = 0.6f; ha[(size_t)n + i] = ((double)(s >> 11) / 9007199254740992.0) * 2.0 - 1.0; }
    VCK(cudaMemcpy(act, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
    de.step(act, obs, rew, done);
  }
  bc::RenderState* rs;
  VCK(cudaMalloc(&rs, sizeof(bc::RenderState) * n));
  rs_from_env_k<<<(n + 127) / 128, 128>>>(de.soa(), n, rs);
  VCK(cudaDeviceSynchronize());
  cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
  return rs;
}

// 판 n 개를 chunk 단위로 그려 u8 영상 [n][2][256][256][3] 으로(호스트)
static std::vector<uint8_t> render_host(bcr::Renderer* R, const bc::RenderState* rs, int n) {
  const size_t im = (size_t)bcr::RES * bcr::RES * 3;
  std::vector<uint8_t> out((size_t)n * 2 * im);
  const int E = bcr::batch(R);
  for (int e0 = 0; e0 < n; e0 += E) {
    const int m = std::min(E, n - e0);
    bcr::render(R, rs + e0, m, ST);
    VCK(cudaStreamSynchronize(ST));
    for (int c = 0; c < 2; ++c)
      for (int e = 0; e < m; ++e) VCK(cudaMemcpy(out.data() + ((size_t)(e0 + e) * 2 + c) * im, bcr::rgb(R, c) + (size_t)e * im, im, cudaMemcpyDeviceToHost));
  }
  return out;
}
static void write_ppm(const std::string& p, const uint8_t* img) {
  FILE* f = std::fopen(p.c_str(), "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%d %d\n255\n", bcr::RES, bcr::RES);
  std::fwrite(img, 1, (size_t)bcr::RES * bcr::RES * 3, f);
  std::fclose(f);
}
static bool read_file(const std::string& p, std::vector<uint8_t>& v) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  v.resize(n);
  const bool ok = std::fread(v.data(), 1, n, f) == (size_t)n;
  std::fclose(f);
  return ok;
}

// ---- 작업 1: 렌더 그래프 ----
static int run_render_graph(int E) {
  int fails = 0;
  // (a) 작업 공간 없이 바로 잡기 — 자식 과정(RCK 가 exit 함)
  {
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
      const int e = bcr::capture_cold_probe(64, bcr::CHEAP);
      std::printf("  cold capture returned %d (%s)\n", e, cudaGetErrorString((cudaError_t)e));
      std::fflush(stdout);
      _exit(e == 0 ? 0 : 7);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    const int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    std::printf("render-graph (a) capture without prior render (Scratch::ensure -> cudaMalloc inside capture): child exit %d -> %s\n", code,
                code != 0 ? "fails as expected" : "captured (unexpected)");
  }
  // (b) 한 번 그린 뒤: 그래프 == 즉시 실행(비트), 시간
  bc::RenderState* rs = a2_states(E, 4242);
  for (int prof : {bcr::CHEAP, bcr::TEAM_DEFAULT}) {
    bcr::Renderer* R = bcr::create(E, prof);
    const size_t im = (size_t)bcr::RES * bcr::RES * 3 * E;
    std::vector<uint8_t> eager[2], graph[2];
    bcr::render(R, rs, E, ST);
    VCK(cudaStreamSynchronize(ST));
    for (int c = 0; c < 2; ++c) { eager[c].resize(im); VCK(cudaMemcpy(eager[c].data(), bcr::rgb(R, c), im, cudaMemcpyDeviceToHost)); }
    VCK(cudaMemset((void*)bcr::rgb(R, 0), 0, im));
    VCK(cudaMemset((void*)bcr::rgb(R, 1), 0, im));
    cudaGraph_t g;
    VCK(cudaStreamBeginCapture(ST, cudaStreamCaptureModeThreadLocal));
    bcr::render(R, rs, E, ST);
    VCK(cudaStreamEndCapture(ST, &g));
    size_t nn = 0;
    cudaGraphGetNodes(g, nullptr, &nn);
    cudaGraphExec_t ex;
    VCK(cudaGraphInstantiate(&ex, g, 0));
    VCK(cudaGraphLaunch(ex, ST));
    VCK(cudaStreamSynchronize(ST));
    long diff = 0;
    for (int c = 0; c < 2; ++c) {
      graph[c].resize(im);
      VCK(cudaMemcpy(graph[c].data(), bcr::rgb(R, c), im, cudaMemcpyDeviceToHost));
      for (size_t k = 0; k < im; ++k) diff += graph[c][k] != eager[c][k];
    }
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const int reps = 20;
    float te = 0, tg = 0;
    cudaEventRecord(a, ST);
    for (int r = 0; r < reps; ++r) bcr::render(R, rs, E, ST);
    cudaEventRecord(b, ST);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&te, a, b);
    cudaEventRecord(a, ST);
    for (int r = 0; r < reps; ++r) cudaGraphLaunch(ex, ST);
    cudaEventRecord(b, ST);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&tg, a, b);
    te /= reps; tg /= reps;
    const bool ok = diff == 0;
    fails += !ok;
    std::printf("render-graph (b) %s E %d, 2 cams 256^2: graph %zu nodes, graph vs eager %ld bytes differ %s | eager %.2f ms, graph %.2f ms per batch -> %.0f env-frames/s; mem ~%.2f GB\n",
                prof == bcr::CHEAP ? "cheap" : "team-default", E, nn, diff, ok ? "ok" : "FAIL", te, tg, E / (tg * 1e-3), bcr::bytes(R) / 1e9);
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(g);
    bcr::destroy(R);
  }
  cudaFree(rs);
  return fails;
}

// ---- 덤프(Python 기준값 입력) ----
static int run_dump(const std::string& dir, int n) {
  bc::RenderState* rs = a2_states(n, 9001);
  bcr::Renderer* R = bcr::create(std::min(n, 256), bcr::CHEAP);
  const auto imgs = render_host(R, rs, n);
  FILE* f = std::fopen((dir + "/imgs.u8").c_str(), "wb");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", dir.c_str()); return 1; }
  std::fwrite(imgs.data(), 1, imgs.size(), f);
  std::fclose(f);
  const size_t im = (size_t)bcr::RES * bcr::RES * 3;
  for (int e = 0; e < std::min(n, 6); ++e)
    for (int c = 0; c < 2; ++c) write_ppm(dir + "/e" + std::to_string(e) + "_c" + std::to_string(c) + ".ppm", imgs.data() + ((size_t)e * 2 + c) * im);
  std::printf("dump: %d states x 2 cams -> %s/imgs.u8 (%zu bytes)\n", n, dir.c_str(), imgs.size());
  bcr::destroy(R);
  cudaFree(rs);
  return 0;
}

// ---- GPU 인코딩 도우미: 호스트 영상 → 토큰(double [n_img × 64][768]) ----
static std::vector<double> gpu_encode(vit::Encoder& enc, const std::vector<uint8_t>& imgs, int n_img, int layers = vit::LAYERS) {
  const size_t im = (size_t)vit::IMG * vit::IMG * 3;
  uint8_t* d;
  VCK(cudaMalloc(&d, im * n_img));
  VCK(cudaMemcpy(d, imgs.data(), im * n_img, cudaMemcpyHostToDevice));
  // 영상 j = 판 j/2 의 카메라 j%2: 카메라 0 줄 = 짝수 영상, 카메라 1 = 홀수 → patchify 는 [E][256][256][3] 카메라 배열 둘을 받으므로 펼쳐 넣는다
  uint8_t *c0, *c1;
  const int ne = (n_img + 1) / 2;
  VCK(cudaMalloc(&c0, im * ne)); VCK(cudaMalloc(&c1, im * ne));
  VCK(cudaMemset(c1, 0, im * ne));
  for (int j = 0; j < n_img; ++j) VCK(cudaMemcpy((j % 2 ? c1 : c0) + (size_t)(j / 2) * im, d + (size_t)j * im, im, cudaMemcpyDeviceToDevice));
  uint16_t* tok;
  VCK(cudaMalloc(&tok, (size_t)ne * 2 * vit::NTOK * vit::TOK_LD * 2));
  vit::init_token_buffer(tok, (long long)ne * 2 * vit::NTOK, ST);
  enc.patchify(c0, c1, ne, 0, ST);
  enc.run(ne * 2, tok, ST, layers);
  VCK(cudaStreamSynchronize(ST));
  std::vector<uint16_t> h((size_t)n_img * vit::NTOK * vit::TOK_LD);
  VCK(cudaMemcpy(h.data(), tok, h.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<double> o((size_t)n_img * vit::NTOK * vit::D);
  for (size_t t = 0; t < (size_t)n_img * vit::NTOK; ++t)
    for (int c = 0; c < vit::D; ++c) o[t * vit::D + c] = net::bf2f(h[t * vit::TOK_LD + c]);
  cudaFree(d); cudaFree(c0); cudaFree(c1); cudaFree(tok);
  return o;
}
struct CosStat { double mean, p1, min; };
static CosStat cos_stats(const std::vector<double>& a, const std::vector<double>& b, size_t rows, int w) {
  std::vector<double> c(rows);
  for (size_t r = 0; r < rows; ++r) {
    double ab = 0, aa = 0, bb = 0;
    for (int k = 0; k < w; ++k) { const double x = a[r * w + k], y = b[r * w + k]; ab += x * y; aa += x * x; bb += y * y; }
    c[r] = ab / std::sqrt(aa * bb + 1e-300);
  }
  double m = 0;
  for (double v : c) m += v;
  std::vector<double> s = c;
  std::sort(s.begin(), s.end());
  return {m / rows, s[(size_t)(0.01 * (rows - 1))], s[0]};
}
static double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
  double n = 0, d = 0;
  for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (a[i] - b[i]); d += b[i] * b[i]; }
  return std::sqrt(n / d);
}

static int run_enc(const std::string& dir, bool negative) {
  vit::HostWeights hw;
  std::string wp;
  if (!vit::load_weights("", hw, &wp)) return 2;
  std::printf("enc: weights %s\n", wp.c_str());
  std::vector<uint8_t> imgs, refb;
  if (!read_file(dir + "/imgs.u8", imgs)) { std::fprintf(stderr, "no %s/imgs.u8 (run dump first)\n", dir.c_str()); return 2; }
  const size_t im = (size_t)vit::IMG * vit::IMG * 3;
  const int n_img = (int)(imgs.size() / im);
  const bool have_ref = read_file(dir + "/ref_tok.f32", refb);
  vit::Encoder enc;
  const bool half = std::getenv("VIT_BF16") == nullptr;
  const char* f8s = std::getenv("VIT_FP8");
  if (f8s && !vit::parse_f8(f8s, enc.f8)) { std::fprintf(stderr, "bad VIT_FP8 '%s'\n", f8s); return 2; }
  if (const char* hs = std::getenv("VIT_F16ACC")) {
    if (!vit::parse_f8(hs, enc.h16)) { std::fprintf(stderr, "bad VIT_F16ACC '%s'\n", hs); return 2; }
    enc.h16_patch = std::getenv("VIT_F16ACC_PATCH") != nullptr;
    std::printf("FP16-accumulate table %s, patch %d\n", hs, (int)enc.h16_patch);
  }
  enc.init(hw, n_img + 1, half);
  vitref::set_half(half);
  vitref::set_f8(enc.f8);
  vitref::set_h16(enc.h16, enc.h16_patch);
  vitref::set_hpromo(vit::h16_promo());
  std::printf("enc: GEMM operands %s, FP8 table (qkv1 proj2 fc1 4 fc2 8 per layer):", half ? "FP16" : "BF16");
  for (int l = 0; l < vit::LAYERS; ++l) std::printf(" %x", enc.f8[l]);
  std::printf("\n");
  int fails = 0, total = 0;
  const auto G = gpu_encode(enc, imgs, n_img);
  // (1) PyTorch FP32 기준값
  if (have_ref) {
    std::vector<double> R(refb.size() / 4);
    for (size_t i = 0; i < R.size(); ++i) { float f; std::memcpy(&f, refb.data() + 4 * i, 4); R[i] = f; }
    const auto cs = cos_stats(G, R, (size_t)n_img * vit::NTOK, vit::D);
    const bool ok = cs.mean >= 0.999 && cs.p1 >= 0.99;
    ++total; fails += !ok;
    std::printf("  GPU vs PyTorch FP32 (open_clip, %d images x 64 patch tokens): cos mean %.6f p1 %.6f min %.6f  rel L2 %.3e  %s (target mean >= 0.999, p1 >= 0.99)\n",
                n_img, cs.mean, cs.p1, cs.min, rel_l2(G, R), ok ? "ok" : "FAIL");
    int n99 = 0, n9 = 0, worst = 0;
    double wc = 2, wn = 0, mn = 0;
    for (size_t r = 0; r < (size_t)n_img * vit::NTOK; ++r) {
      double ab = 0, aa = 0, bb = 0;
      for (int k = 0; k < vit::D; ++k) { ab += G[r * vit::D + k] * R[r * vit::D + k]; aa += G[r * vit::D + k] * G[r * vit::D + k]; bb += R[r * vit::D + k] * R[r * vit::D + k]; }
      const double c = ab / std::sqrt(aa * bb);
      n99 += c < 0.99; n9 += c < 0.9;
      mn += std::sqrt(bb);
      if (c < wc) { wc = c; worst = (int)r; wn = std::sqrt(bb); }
    }
    std::printf("    tokens with cos < 0.99: %d, < 0.9: %d of %d; worst = image %d patch %d (ref token norm %.2f, mean norm %.2f)\n", n99, n9, n_img * vit::NTOK, worst / 64, worst % 64, wn,
                mn / (n_img * vit::NTOK));
  } else {
    std::printf("  (no %s/ref_tok.f32 — run tools/siglip_ref.py)\n", dir.c_str());
  }
  if (std::getenv("VIT_QUICK")) return fails;   // 혼합 정밀도 찾기용: PyTorch 기준값 비교만
  // (2) CPU FP64 / EMUL (영상 4 장): 2 층 뒤와 12 층 뒤
  {
    const int nr = std::min(n_img, 4);
    std::vector<const uint8_t*> ptr;
    for (int j = 0; j < nr; ++j) ptr.push_back(imgs.data() + (size_t)j * im);
    for (int L : {2, vit::LAYERS}) {
      std::vector<double> f64, emu;
      vitref::forward(vitref::FP64, hw, ptr, L, f64);
      vitref::forward(vitref::EMUL, hw, ptr, L, emu);
      const auto g = gpu_encode(enc, std::vector<uint8_t>(imgs.begin(), imgs.begin() + (size_t)nr * im), nr, L);
      const double eg = rel_l2(g, f64), ef = rel_l2(emu, f64);
      const bool ok = eg <= 2.0 * ef;
      ++total; fails += !ok;
      const auto cs = cos_stats(g, f64, (size_t)nr * vit::NTOK, vit::D);
      std::printf("  tokens after %2d blocks: GPU vs FP64 rel %.3e, floor(EMUL) %.3e, ratio %.2f  (cos mean %.6f)  %s\n", L, eg, ef, eg / ef, cs.mean, ok ? "ok" : "FAIL");
      if (L == vit::LAYERS && have_ref) {
        std::vector<double> R((size_t)nr * vit::NTOK * vit::D);
        for (size_t i = 0; i < R.size(); ++i) { float fl; std::memcpy(&fl, refb.data() + 4 * i, 4); R[i] = fl; }
        const auto c2 = cos_stats(f64, R, (size_t)nr * vit::NTOK, vit::D);
        const bool ok2 = c2.mean > 0.99999;
        ++total; fails += !ok2;
        std::printf("  CPU FP64 reference vs PyTorch FP32 (architecture check): cos mean %.8f min %.8f rel %.3e  %s\n", c2.mean, c2.min, rel_l2(f64, R), ok2 ? "ok" : "FAIL");
      }
    }
  }
  // (3) 결정성·묶음 크기 무관
  {
    const auto G2 = gpu_encode(enc, imgs, n_img);
    long d = 0;
    for (size_t i = 0; i < G.size(); ++i) d += G[i] != G2[i];
    const int ns = std::min(n_img, 6);
    const auto Gs = gpu_encode(enc, std::vector<uint8_t>(imgs.begin(), imgs.begin() + (size_t)ns * im), ns);
    long ds = 0;
    for (size_t i = 0; i < Gs.size(); ++i) ds += Gs[i] != G[i];
    const bool ok = d == 0 && ds == 0;
    ++total; fails += !ok;
    std::printf("  determinism: same input twice %ld values differ; first %d images alone vs inside batch %d: %ld differ  %s\n", d, ns, n_img, ds, ok ? "ok" : "FAIL");
  }
  std::printf("enc: %d / %d checks passed\n", total - fails, total);
  if (!negative) return fails;
  // 음성 대조: 버그마다 (2)의 12 층 비교가 실패해야 함
  int caught = 0;
  const int nr = std::min(n_img, 2);
  std::vector<const uint8_t*> ptr;
  for (int j = 0; j < nr; ++j) ptr.push_back(imgs.data() + (size_t)j * im);
  std::vector<double> f64, emu;
  vitref::forward(vitref::FP64, hw, ptr, vit::LAYERS, f64);
  vitref::forward(vitref::EMUL, hw, ptr, vit::LAYERS, emu);
  const double ef = rel_l2(emu, f64);
  const char* names[5] = {"", "attention scale 1/8 dropped", "LayerNorm mean not subtracted", "position embedding dropped", "FP8 activation row scale dropped"};
  bool any8 = false;
  for (int l = 0; l < vit::LAYERS; ++l) any8 = any8 || enc.f8[l];
  const int nbug = any8 ? 4 : 3;
  for (int bug = 1; bug <= nbug; ++bug) {
    enc.bug = bug;
    const auto g = gpu_encode(enc, std::vector<uint8_t>(imgs.begin(), imgs.begin() + (size_t)nr * im), nr);
    const double eg = rel_l2(g, f64);
    const bool fail = !(eg <= 2.0 * ef);
    caught += fail;
    std::printf("  negative bug %d (%s): GPU vs FP64 rel %.3e vs floor %.3e -> %s\n", bug, names[bug], eg, ef, fail ? "fails (good)" : "PASSES (bad)");
  }
  enc.bug = 0;
  std::printf("enc --negative: %d / %d bugs caught  %s\n", caught, nbug, caught == nbug ? "PASS" : "FAIL");
  return caught == nbug ? 0 : 1;
}

// ---- 속도 ----
static int run_bench(int n_img) {
  vit::HostWeights hw;
  if (!vit::load_weights("", hw)) return 2;
  vit::Encoder enc;
  const char* f8s = std::getenv("VIT_FP8");
  if (f8s && !vit::parse_f8(f8s, enc.f8)) { std::fprintf(stderr, "bad VIT_FP8 '%s'\n", f8s); return 2; }
  if (const char* hs = std::getenv("VIT_F16ACC")) {
    if (!vit::parse_f8(hs, enc.h16)) { std::fprintf(stderr, "bad VIT_F16ACC '%s'\n", hs); return 2; }
    enc.h16_patch = std::getenv("VIT_F16ACC_PATCH") != nullptr;
    std::printf("FP16-accumulate table %s, patch %d\n", hs, (int)enc.h16_patch);
  }
  enc.init(hw, n_img);
  std::printf("bench: VIT_FP8=%s\n", f8s ? f8s : "(none: FP16)");
  uint16_t* tok;
  VCK(cudaMalloc(&tok, (size_t)n_img * vit::NTOK * vit::TOK_LD * 2));
  vit::init_token_buffer(tok, (long long)n_img * vit::NTOK, ST);
  const int ne = n_img / 2;
  bc::RenderState* rs = a2_states(256, 31);
  bcr::Renderer* R = bcr::create(256, bcr::CHEAP);
  bcr::render(R, rs, 256, ST);
  for (int e0 = 0; e0 < ne; e0 += 256) enc.patchify(bcr::rgb(R, 0), bcr::rgb(R, 1), std::min(256, ne - e0), e0, ST);
  VCK(cudaStreamSynchronize(ST));
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  auto timeit = [&](auto&& f, int reps) {
    f();
    cudaEventRecord(a, ST);
    for (int r = 0; r < reps; ++r) f();
    cudaEventRecord(b, ST);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
  };
  // 그래프: (렌더 + patchify) 묶음 + 인코더 전체
  cudaGraph_t g;
  VCK(cudaStreamBeginCapture(ST, cudaStreamCaptureModeThreadLocal));
  enc.run(n_img, tok, ST);
  VCK(cudaStreamEndCapture(ST, &g));
  cudaGraphExec_t ex;
  VCK(cudaGraphInstantiate(&ex, g, 0));
  const float t_eager = timeit([&] { enc.run(n_img, tok, ST); }, 5);
  const float t_graph = timeit([&] { cudaGraphLaunch(ex, ST); }, 5);
  const double gflop_img = 2.0 * vit::NTOK * ((double)vit::KP * vit::D + vit::LAYERS * (4.0 * vit::D * vit::D + 2.0 * vit::D * vit::MLP)) / 1e9 +
                           2.0 * 2 * vit::LAYERS * vit::HEADS * (double)vit::NTOK * vit::NTOK * vit::HD / 1e9;
  std::printf("encoder %d images (= %d samples x 2 cams): eager %.2f ms, graph %.2f ms -> %.3f ms/image, %.0f images/s, %.1f TFLOPS (%.2f GFLOP/image)\n", n_img, ne, t_eager,
              t_graph, t_graph / n_img, n_img / (t_graph * 1e-3), gflop_img * n_img / (t_graph * 1e-3) / 1e3, gflop_img);
  const float t_block = timeit([&] { enc.block(0, n_img, ST); }, 5);
  const float t_patch = timeit([&] { enc.patchify(bcr::rgb(R, 0), bcr::rgb(R, 1), std::min(256, ne), 0, ST); }, 5);
  std::printf("  one block %.2f ms (x12 = %.1f ms), patchify 256 samples %.3f ms, device: weights %.0f MB, work %.2f GB\n", t_block, 12 * t_block, t_patch, enc.W.bytes / 1e6,
              enc.bytes / 1e9);
  // 렌더 256²
  for (int prof : {bcr::CHEAP, bcr::TEAM_DEFAULT}) {
    bcr::Renderer* Rp = bcr::create(256, prof);
    const float t = timeit([&] { bcr::render(Rp, rs, 256, ST); }, 10);
    std::printf("render %s 256 envs x 2 cams 256^2: %.2f ms -> %.0f env-frames/s (%.1f us/sample)\n", prof == bcr::CHEAP ? "cheap" : "team-default", t, 256 / (t * 1e-3),
                t * 1e3 / 256);
    bcr::destroy(Rp);
  }
  bcr::destroy(R);
  cudaFree(rs);
  cudaFree(tok);
  return 0;
}

// ---- 품질: 싼 설정 대 팀 기본 ----
static int run_quality(int n, const std::string& dir) {
  vit::HostWeights hw;
  if (!vit::load_weights("", hw)) return 2;
  bc::RenderState* rs = a2_states(n, 555);
  bcr::Renderer* Rc = bcr::create(n, bcr::CHEAP);
  const auto ic = render_host(Rc, rs, n);
  bcr::destroy(Rc);
  bcr::Renderer* Rd = bcr::create(n, bcr::TEAM_DEFAULT);
  const auto id = render_host(Rd, rs, n);
  bcr::destroy(Rd);
  double mad = 0;
  for (size_t k = 0; k < ic.size(); ++k) mad += std::abs((int)ic[k] - (int)id[k]);
  mad /= ic.size();
  vit::Encoder enc;
  enc.init(hw, 2 * n);
  const auto tc = gpu_encode(enc, ic, 2 * n), td = gpu_encode(enc, id, 2 * n);
  const auto cs = cos_stats(tc, td, (size_t)2 * n * vit::NTOK, vit::D);
  // 같은 설정 안의 다른 상태끼리(비교 기준): 판 e 대 판 e+1
  std::vector<double> tsh(tc.size());
  const size_t per = (size_t)2 * vit::NTOK * vit::D;
  for (int e = 0; e < n; ++e) std::copy(tc.begin() + ((e + 1) % n) * per, tc.begin() + ((e + 1) % n + 1) * per, tsh.begin() + e * per);
  const auto cso = cos_stats(tc, tsh, (size_t)2 * n * vit::NTOK, vit::D);
  std::printf("quality %d states: mean |pixel diff| cheap vs team-default %.2f / 255; patch-token cos cheap vs default mean %.4f p1 %.4f (different states, same setting: %.4f)\n", n, mad,
              cs.mean, cs.p1, cso.mean);
  if (!dir.empty()) {
    const size_t im = (size_t)bcr::RES * bcr::RES * 3;
    for (int e = 0; e < std::min(n, 4); ++e)
      for (int c = 0; c < 2; ++c) {
        write_ppm(dir + "/q_cheap_e" + std::to_string(e) + "_c" + std::to_string(c) + ".ppm", ic.data() + ((size_t)e * 2 + c) * im);
        write_ppm(dir + "/q_default_e" + std::to_string(e) + "_c" + std::to_string(c) + ".ppm", id.data() + ((size_t)e * 2 + c) * im);
      }
  }
  cudaFree(rs);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: vit_verify render-graph [E] | dump DIR [n] | enc DIR [--negative] | bench [n_img] | quality [n] [DIR]\n"); return 2; }
  const std::string m = argv[1];
  bool neg = false;
  std::vector<std::string> pos;
  for (int i = 2; i < argc; ++i) { if (!std::strcmp(argv[i], "--negative")) neg = true; else pos.push_back(argv[i]); }
  if (m == "render-graph") return run_render_graph(pos.size() > 0 ? std::atoi(pos[0].c_str()) : 256);
  if (m == "dump") return run_dump(pos.at(0), pos.size() > 1 ? std::atoi(pos[1].c_str()) : 16);
  if (m == "enc") return run_enc(pos.at(0), neg);
  if (m == "bench") return run_bench(pos.size() > 0 ? std::atoi(pos[0].c_str()) : 1024);
  if (m == "quality") return run_quality(pos.size() > 0 ? std::atoi(pos[0].c_str()) : 64, pos.size() > 1 ? pos[1] : "");
  return 2;
}
