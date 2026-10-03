// 엔진(TensorRT FP16, LayerNorm·GELU FP32) + 자르기 커널 끝까지 = PyTorch FP32 인가(tools/make_parity.py 의 평가 crop).
//   test_sgclip_encoder ENGINE PARITY_BIN
// 확인: 코사인 평균 ≥ 0.999, 하위 1 % ≥ 0.99 (최악 값도 찍음), 배치 1·3·8 이 같은 답(graph 켜고 끔 사이도), 비동기 고리(칸 2개)
// 결과가 빠짐없이 id 대로 돌아옴. 단계 시간(자르기·엔진 GPU ms, submit CPU µs)도 찍는다.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#include "sgclip.h"

struct Rec {
  int w, h;
  std::vector<uint8_t> rgb;
  float box[4];
  std::vector<uint32_t> bits;
  std::vector<float> ref;
};

static double cosine(const float* a, const float* b) {
  double s = 0, na = 0, nb = 0;
  for (int d = 0; d < SGC_DIM; ++d) s += double(a[d]) * b[d], na += double(a[d]) * a[d], nb += double(b[d]) * b[d];
  return s / std::sqrt(na * nb);
}

int main(int argc, char** argv) {
  if (argc < 3 || !std::filesystem::exists(argv[1]) || !std::filesystem::exists(argv[2])) {
    std::printf("skip: engine or parity file missing\n");
    return 77;
  }
  std::ifstream f(argv[2], std::ios::binary);
  int32_t hdr[2];
  f.read(reinterpret_cast<char*>(hdr), 8);
  std::vector<Rec> R(hdr[1]);
  for (Rec& r : R) {
    f.read(reinterpret_cast<char*>(&r.w), 4);
    f.read(reinterpret_cast<char*>(&r.h), 4);
    r.rgb.resize(size_t(r.w) * r.h * 3);
    f.read(reinterpret_cast<char*>(r.rgb.data()), std::streamsize(r.rgb.size()));
    f.read(reinterpret_cast<char*>(r.box), 16);
    r.bits.resize((size_t(r.w) * r.h + 31) / 32);
    f.read(reinterpret_cast<char*>(r.bits.data()), std::streamsize(r.bits.size() * 4));
    r.ref.resize(SGC_DIM);
    f.read(reinterpret_cast<char*>(r.ref.data()), SGC_DIM * 4);
  }
  if (!f) { std::printf("bad parity file\n"); return 1; }
  int fails = 0;
  for (int graph = 1; graph >= 0; --graph) {
    sgc_config c;
    sgc_default_config(&c);
    c.engine = argv[1];
    c.use_graph = graph;
    char err[256] = {0};
    sgc_encoder* e = sgc_create(&c, err, sizeof(err));
    if (!e) { std::printf("create: %s\n", err); return 1; }
    std::map<uint32_t, std::vector<float>> got;
    std::vector<sgc_result> out(16);
    // 물체마다 따로 영상(평가 crop) → 한 번에 1 개씩 제출(영상이 다름), 고리 2 칸 비동기
    double sub_us = 0;
    int n_sub = 0;
    for (int pass = 0; pass < 2; ++pass)
      for (size_t i = 0; i < R.size(); ++i) {
        const Rec& r = R[i];
        sgc_frame fr{r.rgb.data(), 0, int64_t(r.w) * 3, 3, r.w, r.h, r.w, r.h, 1.f, 1.f, 0.f, 0.f, r.bits.data()};
        sgc_item it{uint32_t(i + 1000 * pass), 0, {r.box[0], r.box[1], r.box[2], r.box[3]}, float(i)};
        int k;
        while ((k = sgc_submit(e, double(i), &fr, &it, 1)) == 0) {
          const int m = sgc_poll(e, out.data(), int(out.size()), 1);
          for (int j = 0; j < m; ++j) got[out[j].id].assign(out[j].emb, out[j].emb + SGC_DIM);
        }
        if (k < 0) { std::printf("submit failed\n"); return 1; }
        sgc_timing t;
        sgc_get_timing(e, &t);
        sub_us += t.submit_us;
        ++n_sub;
        const int m = sgc_poll(e, out.data(), int(out.size()), 0);
        for (int j = 0; j < m; ++j) got[out[j].id].assign(out[j].emb, out[j].emb + SGC_DIM);
      }
    for (int m; (m = sgc_poll(e, out.data(), int(out.size()), 1)) > 0;)
      for (int j = 0; j < m; ++j) got[out[j].id].assign(out[j].emb, out[j].emb + SGC_DIM);
    std::vector<double> cs;
    for (size_t i = 0; i < R.size(); ++i) {
      if (!got.count(uint32_t(i)) || !got.count(uint32_t(i + 1000))) { ++fails; continue; }
      cs.push_back(cosine(got[uint32_t(i)].data(), R[i].ref.data()));
      if (cosine(got[uint32_t(i)].data(), got[uint32_t(i + 1000)].data()) < 0.99999) ++fails;   // 같은 입력 → 같은 답
    }
    std::sort(cs.begin(), cs.end());
    double mean = 0;
    for (double v : cs) mean += v;
    mean /= std::max<size_t>(1, cs.size());
    const double p1 = cs.empty() ? 0 : cs[cs.size() / 100];
    const bool ok = cs.size() == R.size() && mean >= 0.999 && p1 >= 0.99;
    fails += !ok;
    sgc_timing t;
    sgc_get_timing(e, &t);
    std::printf("graph %d: %zu crops vs PyTorch FP32: cosine mean %.5f, p1 %.5f, min %.5f -> %s | submit %.0f us avg, last crop %.3f ms net %.3f ms, "
                "S %d G %d, device %.0f MB\n",
                graph, cs.size(), mean, p1, cs.empty() ? 0 : cs[0], ok ? "ok" : "FAIL", sub_us / n_sub, t.crop_ms, t.net_ms, t.input_size, t.grid,
                t.device_bytes / 1048576.0);
    // 배치: 같은 영상에 물체 1·3·8 개 — 같은 물체는 배치와 상관없이 같은 답
    const Rec& r = R[0];
    sgc_frame fr{r.rgb.data(), 0, int64_t(r.w) * 3, 3, r.w, r.h, r.w, r.h, 1.f, 1.f, 0.f, 0.f, r.bits.data()};
    std::vector<sgc_item> items;
    for (int k = 0; k < 8; ++k) {
      const float s = 1.f - 0.05f * k;
      const float cx = 0.5f * (r.box[0] + r.box[2]), cy = 0.5f * (r.box[1] + r.box[3]);
      const float hw = 0.5f * (r.box[2] - r.box[0]) * s, hh = 0.5f * (r.box[3] - r.box[1]) * s;
      items.push_back(sgc_item{uint32_t(k), 0, {cx - hw, cy - hh, cx + hw, cy + hh}, 0});
    }
    std::map<int, std::vector<std::vector<float>>> by;
    for (int nb : {1, 3, 8, 1, 8}) {
      sgc_submit(e, 0, &fr, items.data(), nb);
      std::vector<sgc_result> o(8);
      const int m = sgc_poll(e, o.data(), 8, 1);
      for (int j = 0; j < m; ++j) {
        auto& v = by[int(o[j].id)];
        v.emplace_back(o[j].emb, o[j].emb + SGC_DIM);
      }
    }
    double worst = 1;
    for (auto& [id, vs] : by)
      for (auto& v : vs) worst = std::min(worst, cosine(v.data(), vs[0].data()));
    std::printf("graph %d: batch 1/3/8 consistency worst cosine %.6f -> %s\n", graph, worst, worst > 0.9995 ? "ok" : "FAIL");
    fails += worst <= 0.9995;   // 배치 칸 엔진은 프로필마다 다른 FP16 전술(코사인 0.9999 안팎)
    // 묶음 크기별 시간(같은 영상, 장치 영상처럼 한 번 올려 둠)
    for (int nb : {1, 8}) {
      double crop = 0, net = 0, sub = 0;
      for (int it = 0; it < 30; ++it) {
        sgc_submit(e, 0, &fr, items.data(), nb);
        sgc_get_timing(e, &t);
        sub += t.submit_us;
        std::vector<sgc_result> o(8);
        sgc_poll(e, o.data(), 8, 1);
        sgc_get_timing(e, &t);
        if (it >= 10) crop += t.crop_ms, net += t.net_ms;
      }
      std::printf("graph %d batch %d: crop %.3f ms, net %.3f ms (GPU, mean of 20), submit %.0f us (host frame upload incl.)\n", graph, nb, crop / 20,
                  net / 20, sub / 30);
    }
    sgc_destroy(e);
  }
  return fails ? 1 : 0;
}
