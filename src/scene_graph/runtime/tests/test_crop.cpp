// best view 장치 자르기·구름 점 색 모으기 시험(작은 GPU 시험): sgrt_crop::run/gather 가 scenemap 호스트 식과 같은지, 시간.
//   비교: 상자만 잘라 내려받기 vs 온 영상(720×720×4) 내려받기. 그리고 scenemap 에 자르기 함수로 붙인 keyframe 시간.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../src/crop.hpp"
#include "scenemap.h"
#include "scenemap/bestview.hpp"

static double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
  constexpr int W = 720, H = 720, PS = 4;
  std::vector<uint8_t> img(size_t(W) * H * PS);
  std::mt19937 rng(1);
  for (auto& v : img) v = uint8_t(rng());
  uint8_t* d_img = nullptr;
  if (cudaMalloc(&d_img, img.size()) != cudaSuccess) {
    std::printf("no GPU\n");
    return 1;
  }
  cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice);
  // 상자 6 개: 작은 것(그대로), 큰 것(256 으로 줄임), 가장자리
  const float boxes[6][4] = {{300, 300, 340, 340}, {420, 300, 480, 380}, {60, 420, 660, 600},
                             {0, 0, 720, 200},     {500, 220, 600, 300}, {690, 690, 720, 720}};
  std::vector<sm_crop_req> rq(6), rh(6);
  std::vector<std::vector<uint8_t>> bd(6), bh(6);
  for (int k = 0; k < 6; ++k) {
    int32_t box[4], w, h;
    scenemap::cropGeometry(boxes[k], W, H, 0.1f, 256, box, &w, &h);
    bd[k].assign(size_t(w) * h * 3, 0);
    bh[k].assign(size_t(w) * h * 3, 1);
    rq[k] = sm_crop_req{box[0], box[1], box[2], box[3], w, h, bd[k].data()};
    rh[k] = rq[k];
    rh[k].dst = bh[k].data();
    scenemap::cropRgbHost(img.data(), int64_t(W) * PS, PS, rh[k]);
  }
  sgrt_crop::Gpu* g = sgrt_crop::create();
  int fail = 0;
  if (sgrt_crop::run(g, d_img, int64_t(W) * PS, PS, rq.data(), 6) != 0) ++fail;
  size_t bytes = 0;
  for (int k = 0; k < 6; ++k) {
    bytes += bd[k].size();
    if (bd[k] != bh[k]) {
      ++fail;
      std::printf("  FAIL crop %d (%dx%d) differs from host\n", k, rq[k].out_w, rq[k].out_h);
    }
  }
  const int N = 200;
  double t0 = nowMs();
  for (int i = 0; i < N; ++i) sgrt_crop::run(g, d_img, int64_t(W) * PS, PS, rq.data(), 6);
  const double crop_ms = (nowMs() - t0) / N;
  t0 = nowMs();
  for (int i = 0; i < N; ++i) sgrt_crop::run(g, d_img, int64_t(W) * PS, PS, rq.data(), 1);
  const double crop1_ms = (nowMs() - t0) / N;
  std::vector<uint8_t> full(img.size());
  t0 = nowMs();
  for (int i = 0; i < N; ++i) cudaMemcpy(full.data(), d_img, full.size(), cudaMemcpyDeviceToHost);
  const double full_ms = (nowMs() - t0) / N;
  std::printf("[crop] 6 상자(%zu B) 장치 자르기+내려받기 %.3f ms, 1 상자 %.3f ms, 온 영상(%zu B) 내려받기 %.3f ms, 호스트와 같음 %s\n",
              bytes, crop_ms, crop1_ms, full.size(), full_ms, fail ? "아님" : "예");
  // 구름 점 색 모으기: 화소 3000 개(영상 밖 좌표 포함) — 장치 = 호스트, 시간
  {
    const int n = 3000;
    std::vector<int32_t> xy(2 * n);
    for (int i = 0; i < n; ++i) { xy[2 * i] = int(rng() % (W + 20)) - 10; xy[2 * i + 1] = int(rng() % (H + 20)) - 10; }
    std::vector<uint8_t> gd(3 * n, 0), gh(3 * n, 1);
    scenemap::gatherRgbHost(img.data(), int64_t(W) * PS, PS, W, H, xy.data(), n, gh.data());
    if (sgrt_crop::gather(g, d_img, int64_t(W) * PS, PS, W, H, xy.data(), n, gd.data()) != 0 || gd != gh) {
      ++fail;
      std::printf("  FAIL gather differs from host\n");
    }
    t0 = nowMs();
    for (int i = 0; i < N; ++i) sgrt_crop::gather(g, d_img, int64_t(W) * PS, PS, W, H, xy.data(), n, gd.data());
    std::printf("[gather] 점 %d 개 색(장치 모으기 + 올림·내림) %.3f ms, 호스트와 같음 %s\n", n, (nowMs() - t0) / N, gd == gh ? "예" : "아님");
  }
  sgrt_crop::destroy(g);
  cudaFree(d_img);
  std::printf(fail ? "FAILED %d\n" : "ok\n", fail);
  return fail ? 1 : 0;
}
