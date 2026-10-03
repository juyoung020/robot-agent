// 단계별 시간 표: 영상 쪽(720×720 장치 RGB 한 장, 물체 1–8 개: submit CPU, 자르기 커널·엔진 GPU, 끝까지 벽시계) +
// 라벨 찾기(설정별 µs/물체, 1위 일치 = 전부 훑기와).
//   sgclip_bench [--engine PLAN] [--labels DIR] [--index DIR] [--queries Q.f32] [--iters 50]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "sgclip.h"

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

int main(int argc, char** argv) {
  std::string engine, labels, index, queries;
  int iters = 50;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--engine") engine = argv[i + 1];
    else if (k == "--labels") labels = argv[i + 1];
    else if (k == "--index") index = argv[i + 1];
    else if (k == "--queries") queries = argv[i + 1];
    else if (k == "--iters") iters = std::atoi(argv[i + 1]);
  }
  if (!engine.empty()) {
    sgc_config c;
    sgc_default_config(&c);
    c.engine = engine.c_str();
    char err[256] = {0};
    sgc_encoder* e = sgc_create(&c, err, sizeof(err));
    if (!e) { std::printf("create: %s\n", err); return 1; }
    const int W = 720, H = 720;
    std::vector<uint8_t> img(size_t(W) * H * 3);
    std::mt19937 rng(1);
    for (auto& v : img) v = uint8_t(rng());
    uint8_t* d;
    cudaMalloc(&d, img.size());
    cudaMemcpy(d, img.data(), img.size(), cudaMemcpyHostToDevice);
    const int mw = 180, mh = 180, words = (mw * mh + 31) / 32;
    std::vector<uint32_t> bits(size_t(words) * 8);
    for (auto& w : bits) w = rng();
    sgc_frame fr{d, 1, int64_t(W) * 3, 3, W, H, mw, mh, 4.f, 4.f, 0.f, 0.f, bits.data()};
    std::vector<sgc_item> items;
    for (int k = 0; k < 8; ++k) items.push_back(sgc_item{uint32_t(k), k, {50.f + 60 * k, 100.f, 200.f + 60 * k, 260.f}, 1});
    std::printf("| batch | submit CPU us | crop GPU ms | net GPU ms | end-to-end wall ms | ms / object |\n|---|---|---|---|---|---|\n");
    for (int nb : {1, 2, 4, 8}) {
      double sub = 0, crop = 0, net = 0, wall = 0;
      std::vector<sgc_result> o(8);
      for (int it = 0; it < iters + 5; ++it) {
        const auto t0 = Clock::now();
        sgc_submit(e, 0, &fr, items.data(), nb);
        sgc_poll(e, o.data(), 8, 1);
        const auto t1 = Clock::now();
        sgc_timing t;
        sgc_get_timing(e, &t);
        if (it < 5) continue;
        sub += t.submit_us; crop += t.crop_ms; net += t.net_ms; wall += us(t0, t1) / 1000;
      }
      std::printf("| %d | %.0f | %.3f | %.3f | %.3f | %.3f |\n", nb, sub / iters, crop / iters, net / iters, wall / iters, wall / iters / nb);
    }
    sgc_timing t;
    sgc_get_timing(e, &t);
    std::printf("engine %s: S %d, G %d, device %.0f MB\n\n", engine.c_str(), t.input_size, t.grid, t.device_bytes / 1048576.0);
    cudaFree(d);
    sgc_destroy(e);
  }
  if (!labels.empty()) {
    char err[256] = {0};
    if (!index.empty()) std::filesystem::create_directories(index);
    const auto t0 = Clock::now();
    sgc_labels* L = sgc_labels_open(labels.c_str(), index.empty() ? nullptr : index.c_str(), err, sizeof(err));
    if (!L) { std::printf("labels: %s\n", err); return 1; }
    std::printf("labels %s: %d rows, open %.0f ms, simd %s\n", sgc_labels_name(L), sgc_labels_count(L), us(t0, Clock::now()) / 1000, sgc_simd());
    std::vector<float> Q;
    if (!queries.empty()) {
      std::ifstream f(queries, std::ios::binary | std::ios::ate);
      Q.resize(size_t(f.tellg()) / 4);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(Q.data()), std::streamsize(Q.size() * 4));
    }
    const int nq = int(Q.size() / SGC_DIM);
    if (!nq) { std::printf("no queries\n"); return 0; }
    for (int i = 0; i < nq; ++i) {
      float* q = &Q[size_t(i) * SGC_DIM];
      double s = 0;
      for (int d = 0; d < SGC_DIM; ++d) s += double(q[d]) * q[d];
      for (int d = 0; d < SGC_DIM; ++d) q[d] /= float(std::sqrt(s));
    }
    sgc_lookup_params pe;
    sgc_default_lookup(&pe);
    pe.exact = 1;
    std::vector<int> ex(nq);
    double use = 0;
    for (int i = 0; i < nq; ++i) {
      sgc_hit h[1];
      const auto a = Clock::now();
      sgc_labels_lookup(L, &Q[size_t(i) * SGC_DIM], 1, h, &pe);
      use += us(a, Clock::now());
      ex[i] = h[0].row;
    }
    std::printf("| method | us / object | top-1 = exact |\n|---|---|---|\n| exact 768-d FP16 (all %d rows) | %.0f | 1 |\n", sgc_labels_count(L),
                use / nq);
    for (auto [np, rr, pf] : std::vector<std::tuple<int, int, int>>{{4, 32, 0}, {8, 32, 0}, {8, 32, 256}, {8, 64, 0}, {16, 64, 0}, {32, 128, 0}}) {
      sgc_lookup_params p;
      sgc_default_lookup(&p);
      p.nprobe = np;
      p.rerank = rr;
      p.prefilter = pf;
      int ag = 0;
      double t = 0;
      for (int r = 0; r < 3; ++r)
        for (int i = 0; i < nq; ++i) {
          sgc_hit h[5];
          const auto a = Clock::now();
          sgc_labels_lookup(L, &Q[size_t(i) * SGC_DIM], 5, h, &p);
          t += us(a, Clock::now());
          if (r == 0) ag += h[0].row == ex[i];
        }
      std::printf("| IVF %d probes%s -> 128-d FP16 top %d -> 768-d | %.1f | %.3f |\n", np, pf ? " -> 128-bit Hamming 256" : "", rr, t / (3 * nq), double(ag) / nq);
    }
    double tn = 0;
    for (int i = 0; i < nq; ++i) {
      sgc_names n;
      const auto a = Clock::now();
      sgc_labels_names(L, &Q[size_t(i) * SGC_DIM], &n, nullptr);
      tn += us(a, Clock::now());
    }
    std::printf("| names (default + top-16 roll-up) | %.1f | |\n", tn / nq);
    sgc_labels_close(L);
  }
  return 0;
}
