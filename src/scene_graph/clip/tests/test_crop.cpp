// 자르기 커널 = CPU 기준(같은 식) 인가: 무작위 영상(RGB·RGBA, 행 여백), 영상 밖으로 나간 상자, 검출기 격자 마스크.
// FP32 출력은 5e-4(GPU FMA 묶음 반올림 차이 — 8 비트 화소 한 단계 7.8e-3 보다 훨씬 작음), FP16 출력은 FP16 반올림(4e-3) 안. 마스크 비율은 같아야 함. 시간도 찍는다.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "crop.hpp"

using namespace sgclip;

int main() {
  std::mt19937 rng(7);
  int fails = 0;
  for (int ps : {3, 4}) {
    const int W = 720, H = 540, S = 256, G = 8, n = 8;
    const int64_t rs = int64_t(W) * ps + 16;
    std::vector<uint8_t> img(size_t(rs) * H);
    for (auto& v : img) v = uint8_t(rng() & 255);
    MaskGeom mg{W / 4, H / 4, (W / 4 * (H / 4) + 31) / 32, 4.f, 4.f, 0.f, 0.f};
    std::vector<uint32_t> bits(size_t(mg.words) * n);
    for (auto& w : bits) w = rng();
    std::vector<CropJob> jobs(n);
    std::uniform_real_distribution<float> U(-80, 760);
    for (int i = 0; i < n; ++i) {
      float b[4] = {U(rng), U(rng), 0, 0};
      b[2] = b[0] + 10 + std::abs(U(rng)) * 0.5f;
      b[3] = b[1] + 10 + std::abs(U(rng)) * 0.5f;
      squareBox(b, 0.1f, &jobs[i].bx, &jobs[i].by, &jobs[i].side);
      jobs[i].mask_word0 = i == 3 ? -1 : i * mg.words;
    }
    std::vector<float> ref(size_t(n) * 3 * S * S), wref(size_t(n) * G * G);
    cropHost(img.data(), rs, ps, W, H, jobs.data(), n, bits.data(), mg, S, G, ref.data(), wref.data());
    uint8_t* d_img;
    CropJob* d_jobs;
    uint32_t* d_bits;
    void* d_out;
    float* d_w;
    cudaMalloc(&d_img, img.size());
    cudaMalloc(&d_jobs, sizeof(CropJob) * n);
    cudaMalloc(&d_bits, bits.size() * 4);
    cudaMalloc(&d_out, ref.size() * 4);
    cudaMalloc(&d_w, wref.size() * 4);
    cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jobs, jobs.data(), sizeof(CropJob) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bits, bits.data(), bits.size() * 4, cudaMemcpyHostToDevice);
    for (int half = 0; half < 2; ++half) {
      launchCrop(d_img, rs, ps, W, H, d_jobs, n, d_bits, mg, S, G, half, d_out, d_w, nullptr);
      cudaDeviceSynchronize();
      if (cudaGetLastError() != cudaSuccess) { std::printf("kernel error\n"); return 1; }
      std::vector<float> out(ref.size()), w(wref.size());
      if (half) {
        std::vector<__half> h(ref.size());
        cudaMemcpy(h.data(), d_out, h.size() * 2, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h.size(); ++i) out[i] = __half2float(h[i]);
      } else {
        cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost);
      }
      cudaMemcpy(w.data(), d_w, w.size() * 4, cudaMemcpyDeviceToHost);
      double md = 0, mw = 0;
      for (size_t i = 0; i < out.size(); ++i) md = std::max(md, double(std::abs(out[i] - ref[i])));
      for (size_t i = 0; i < w.size(); ++i) mw = std::max(mw, double(std::abs(w[i] - wref[i])));
      const bool ok = md < (half ? 4e-3 : 5e-4) && mw < 1e-6;
      // 시간
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      cudaEventRecord(a);
      for (int r = 0; r < 100; ++r) launchCrop(d_img, rs, ps, W, H, d_jobs, n, d_bits, mg, S, G, half, d_out, d_w, nullptr);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      std::printf("pix_stride %d %s: max |d| img %.2e wpatch %.2e -> %s, %d objects %.1f us/launch\n", ps, half ? "fp16" : "fp32", md, mw,
                  ok ? "ok" : "FAIL", n, ms * 10);
      fails += !ok;
    }
    cudaFree(d_img); cudaFree(d_jobs); cudaFree(d_bits); cudaFree(d_out); cudaFree(d_w);
  }
  return fails ? 1 : 0;
}
