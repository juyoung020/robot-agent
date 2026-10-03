// CLIP 입력 만들기 커널(crop.hpp). 블록 [0, P): 출력 화소(256 스레드 = 화소 256 개, 채널 3), [P, P + G²): 마스크 칸 하나.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "crop.hpp"

namespace sgclip {

void squareBox(const float b[4], float margin, float* bx, float* by, float* side) {
  const float cx = 0.5f * (b[0] + b[2]), cy = 0.5f * (b[1] + b[3]);
  const float s = std::max(std::max(b[2] - b[0], b[3] - b[1]), 1.f) * (1.f + margin);
  *bx = cx - 0.5f * s;
  *by = cy - 0.5f * s;
  *side = s;
}

namespace {

__host__ __device__ inline float tap(const uint8_t* src, int64_t rs, int ps, int W, int H, int x, int y, int c) {
  if (x < 0 || y < 0 || x >= W || y >= H) return 0.5f;
  return src[int64_t(y) * rs + int64_t(x) * ps + c] * (1.f / 255.f);
}

__host__ __device__ inline void pixel(const uint8_t* src, int64_t rs, int ps, int W, int H, const CropJob& j, int S, int ox, int oy,
                                      float out[3]) {
  const float k = j.side / float(S);
  const float xs = j.bx + (ox + 0.5f) * k - 0.5f, ys = j.by + (oy + 0.5f) * k - 0.5f;
  const float fx0 = floorf(xs), fy0 = floorf(ys);
  const int x0 = int(fx0), y0 = int(fy0);
  const float fx = xs - fx0, fy = ys - fy0;
  for (int c = 0; c < 3; ++c) {
    const float a = tap(src, rs, ps, W, H, x0, y0, c), b = tap(src, rs, ps, W, H, x0 + 1, y0, c);
    const float d = tap(src, rs, ps, W, H, x0, y0 + 1, c), e = tap(src, rs, ps, W, H, x0 + 1, y0 + 1, c);
    const float v = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
    out[c] = (v - 0.5f) * 2.f;
  }
}

// 출력 화소 (ox, oy) 의 마스크 값 0/1
__host__ __device__ inline int maskAt(int W, int H, const CropJob& j, const uint32_t* bits, const MaskGeom& mg, int S, int ox, int oy) {
  if (j.mask_word0 < 0) return 1;
  const float k = j.side / float(S);
  const int xi = int(floorf(j.bx + (ox + 0.5f) * k)), yi = int(floorf(j.by + (oy + 0.5f) * k));   // 가장 가까운 원본 화소
  if (xi < 0 || yi < 0 || xi >= W || yi >= H) return 0;
  const int ci = int(floorf((xi + 0.5f - mg.ox) / mg.sx)), cj = int(floorf((yi + 0.5f - mg.oy) / mg.sy));
  if (ci < 0 || cj < 0 || ci >= mg.w || cj >= mg.h) return 0;
  const int cell = cj * mg.w + ci;
  return (bits[j.mask_word0 + (cell >> 5)] >> (cell & 31)) & 1u;
}

__global__ void cropKernel(const uint8_t* __restrict__ src, int64_t rs, int ps, int W, int H, const CropJob* __restrict__ jobs,
                           const uint32_t* __restrict__ bits, MaskGeom mg, int S, int G, int half, void* img, float* wpatch) {
  const int item = blockIdx.y;
  const CropJob j = jobs[item];
  const int P = (S * S + 255) / 256;
  if (int(blockIdx.x) < P) {
    const int p = blockIdx.x * 256 + threadIdx.x;
    if (p >= S * S) return;
    const int ox = p % S, oy = p / S;
    float v[3];
    pixel(src, rs, ps, W, H, j, S, ox, oy, v);
    const size_t base = size_t(item) * 3 * S * S + p;
    if (half) {
      __half* o = static_cast<__half*>(img);
      for (int c = 0; c < 3; ++c) o[base + size_t(c) * S * S] = __float2half_rn(v[c]);
    } else {
      float* o = static_cast<float*>(img);
      for (int c = 0; c < 3; ++c) o[base + size_t(c) * S * S] = v[c];
    }
    return;
  }
  // 마스크 칸 하나: (S/G)² 화소를 256 스레드가 나눠 셈
  const int cell = blockIdx.x - P;
  const int cs = S / G, cx = cell % G, cy = cell / G;
  int cnt = 0;
  for (int t = threadIdx.x; t < cs * cs; t += 256) cnt += maskAt(W, H, j, bits, mg, S, cx * cs + t % cs, cy * cs + t / cs);
  for (int o = 16; o > 0; o >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, o);
  __shared__ int sh[8];
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = cnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (int k = 0; k < 8; ++k) s += sh[k];
    wpatch[item * G * G + cell] = float(s) / float(cs * cs);
  }
}

}  // namespace

void launchCrop(const uint8_t* src, int64_t rs, int ps, int W, int H, const CropJob* jobs, int n, const uint32_t* bits, const MaskGeom& mg,
                int S, int G, bool half, void* img, float* wpatch, void* stream) {
  if (n <= 0) return;
  const int P = (S * S + 255) / 256;
  dim3 grid(P + G * G, n);
  cropKernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, rs, ps, W, H, jobs, bits, mg, S, G, half ? 1 : 0, img, wpatch);
}

void cropHost(const uint8_t* src, int64_t rs, int ps, int W, int H, const CropJob* jobs, int n, const uint32_t* bits, const MaskGeom& mg,
              int S, int G, float* img, float* wpatch) {
  const int cs = S / G;
  for (int i = 0; i < n; ++i) {
    for (int oy = 0; oy < S; ++oy)
      for (int ox = 0; ox < S; ++ox) {
        float v[3];
        pixel(src, rs, ps, W, H, jobs[i], S, ox, oy, v);
        for (int c = 0; c < 3; ++c) img[((size_t(i) * 3 + c) * S + oy) * S + ox] = v[c];
      }
    for (int cell = 0; cell < G * G; ++cell) {
      int cnt = 0;
      const int cx = cell % G, cy = cell / G;
      for (int t = 0; t < cs * cs; ++t) cnt += maskAt(W, H, jobs[i], bits, mg, S, cx * cs + t % cs, cy * cs + t / cs);
      wpatch[i * G * G + cell] = float(cnt) / float(cs * cs);
    }
  }
}

}  // namespace sgclip
