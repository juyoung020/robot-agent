// best view RGB 자르기(src/crop.hpp).
#include <cuda_runtime.h>

#include <cstring>

#include "crop.hpp"

namespace sgrt_crop {
namespace {

constexpr int kBatch = 32;
struct Req {
  int x0, y0, bw, bh, ow, oh;
  int64_t off;   // 출력 버퍼 안 바이트 위치
};
struct Batch {
  Req r[kBatch];
};

// 출력 화소 (i, j) = 원 화소 [x0 + i·bw/ow, x0 + (i+1)·bw/ow) × [..] 평균(최소 1 화소) — bestview.cpp cropRgbHost 와 같음
__global__ void cropKernel(const uint8_t* __restrict__ src, int64_t rs, int ps, Batch b, uint8_t* __restrict__ out) {
  const Req q = b.r[blockIdx.z];
  const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= q.ow || j >= q.oh) return;
  const int ya = q.y0 + j * q.bh / q.oh, yb = max(ya + 1, q.y0 + (j + 1) * q.bh / q.oh);
  const int xa = q.x0 + i * q.bw / q.ow, xb = max(xa + 1, q.x0 + (i + 1) * q.bw / q.ow);
  unsigned s0 = 0, s1 = 0, s2 = 0;
  for (int y = ya; y < yb; ++y) {
    const uint8_t* p = src + y * rs + int64_t(xa) * ps;
    for (int x = xa; x < xb; ++x, p += ps) { s0 += p[0]; s1 += p[1]; s2 += p[2]; }
  }
  const unsigned n = unsigned((yb - ya) * (xb - xa));
  uint8_t* d = out + q.off + (int64_t(j) * q.ow + i) * 3;
  d[0] = uint8_t((s0 + n / 2) / n);
  d[1] = uint8_t((s1 + n / 2) / n);
  d[2] = uint8_t((s2 + n / 2) / n);
}

__global__ void gatherKernel(const uint8_t* __restrict__ src, int64_t rs, int ps, int w, int h, const int32_t* __restrict__ xy, int n,
                             uint8_t* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int x = min(max(xy[2 * i], 0), w - 1), y = min(max(xy[2 * i + 1], 0), h - 1);
  const uint8_t* p = src + y * rs + int64_t(x) * ps;
  out[3 * i] = p[0];
  out[3 * i + 1] = p[1];
  out[3 * i + 2] = p[2];
}

}  // namespace

struct Gpu {
  cudaStream_t st = nullptr;
  uint8_t* d_out = nullptr;
  uint8_t* h_out = nullptr;   // 고정(pinned) 메모리
  size_t cap = 0;
  // gather: 화소 좌표 올림 + 색 내림(고정 메모리), 점 수 기준
  int32_t* d_xy = nullptr;
  int32_t* h_xy = nullptr;
  uint8_t* d_rgb = nullptr;
  uint8_t* h_rgb = nullptr;
  size_t gcap = 0;
};

Gpu* create() {
  auto* g = new Gpu();
  if (cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) != cudaSuccess) {
    delete g;
    return nullptr;
  }
  return g;
}

void destroy(Gpu* g) {
  if (!g) return;
  cudaFree(g->d_out);
  cudaFreeHost(g->h_out);
  cudaFree(g->d_xy);
  cudaFreeHost(g->h_xy);
  cudaFree(g->d_rgb);
  cudaFreeHost(g->h_rgb);
  cudaStreamDestroy(g->st);
  delete g;
}

int run(Gpu* g, const uint8_t* src, int64_t rs, int ps, const sm_crop_req* reqs, int n) {
  if (!g || !src || n <= 0) return -1;
  size_t total = 0;
  for (int k = 0; k < n; ++k) total += size_t(reqs[k].out_w) * reqs[k].out_h * 3;
  if (total > g->cap) {
    cudaFree(g->d_out);
    cudaFreeHost(g->h_out);
    g->d_out = g->h_out = nullptr;
    g->cap = 0;
    const size_t cap = total + total / 2;
    if (cudaMalloc(&g->d_out, cap) != cudaSuccess || cudaMallocHost(&g->h_out, cap) != cudaSuccess) return -2;
    g->cap = cap;
  }
  size_t off = 0;
  for (int k0 = 0; k0 < n; k0 += kBatch) {
    Batch b{};
    const int m = n - k0 < kBatch ? n - k0 : kBatch;
    int mw = 1, mh = 1;
    for (int k = 0; k < m; ++k) {
      const sm_crop_req& r = reqs[k0 + k];
      b.r[k] = Req{r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0, r.out_w, r.out_h, int64_t(off)};
      off += size_t(r.out_w) * r.out_h * 3;
      mw = r.out_w > mw ? r.out_w : mw;
      mh = r.out_h > mh ? r.out_h : mh;
    }
    const dim3 blk(16, 16), grd((mw + 15) / 16, (mh + 15) / 16, m);
    cropKernel<<<grd, blk, 0, g->st>>>(src, rs, ps, b, g->d_out);
  }
  cudaMemcpyAsync(g->h_out, g->d_out, total, cudaMemcpyDeviceToHost, g->st);
  if (cudaStreamSynchronize(g->st) != cudaSuccess) return -3;
  off = 0;
  for (int k = 0; k < n; ++k) {
    const size_t sz = size_t(reqs[k].out_w) * reqs[k].out_h * 3;
    std::memcpy(reqs[k].dst, g->h_out + off, sz);
    off += sz;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

int gather(Gpu* g, const uint8_t* src, int64_t rs, int ps, int w, int h, const int32_t* xy, int n, uint8_t* rgb) {
  if (!g || !src || n <= 0) return -1;
  if (size_t(n) > g->gcap) {
    cudaFree(g->d_xy);
    cudaFreeHost(g->h_xy);
    cudaFree(g->d_rgb);
    cudaFreeHost(g->h_rgb);
    g->d_xy = g->h_xy = nullptr;
    g->d_rgb = g->h_rgb = nullptr;
    g->gcap = 0;
    const size_t cap = size_t(n) + size_t(n) / 2 + 1024;
    if (cudaMalloc(&g->d_xy, cap * 8) != cudaSuccess || cudaMallocHost(&g->h_xy, cap * 8) != cudaSuccess ||
        cudaMalloc(&g->d_rgb, cap * 3) != cudaSuccess || cudaMallocHost(&g->h_rgb, cap * 3) != cudaSuccess)
      return -2;
    g->gcap = cap;
  }
  std::memcpy(g->h_xy, xy, size_t(n) * 8);
  cudaMemcpyAsync(g->d_xy, g->h_xy, size_t(n) * 8, cudaMemcpyHostToDevice, g->st);
  gatherKernel<<<(n + 255) / 256, 256, 0, g->st>>>(src, rs, ps, w, h, g->d_xy, n, g->d_rgb);
  cudaMemcpyAsync(g->h_rgb, g->d_rgb, size_t(n) * 3, cudaMemcpyDeviceToHost, g->st);
  if (cudaStreamSynchronize(g->st) != cudaSuccess) return -3;
  std::memcpy(rgb, g->h_rgb, size_t(n) * 3);
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

}  // namespace sgrt_crop
