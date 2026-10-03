// 물체별 best view 자르기(include/scenemap/bestview.hpp). 장치 쪽 같은 계산은 runtime/src/crop.cu.
#include "scenemap/bestview.hpp"

#include <algorithm>
#include <cmath>

namespace scenemap {

bool cropGeometry(const float b[4], int img_w, int img_h, float margin, int max_side, int32_t box[4], int32_t* out_w,
                  int32_t* out_h) {
  const float mx = margin * (b[2] - b[0]), my = margin * (b[3] - b[1]);
  box[0] = std::clamp(int(std::floor(b[0] - mx)), 0, img_w);
  box[1] = std::clamp(int(std::floor(b[1] - my)), 0, img_h);
  box[2] = std::clamp(int(std::ceil(b[2] + mx)), 0, img_w);
  box[3] = std::clamp(int(std::ceil(b[3] + my)), 0, img_h);
  const int bw = box[2] - box[0], bh = box[3] - box[1];
  if (bw <= 0 || bh <= 0) return false;
  const double s = std::min(1.0, double(max_side) / std::max(bw, bh));
  *out_w = std::max(1, int(std::lround(bw * s)));
  *out_h = std::max(1, int(std::lround(bh * s)));
  return true;
}

// 출력 화소 i 가 덮는 원 화소 [a, b): a = x0 + i·bw/ow, b = x0 + (i+1)·bw/ow (최소 1 화소). crop.cu 와 같은 식.
void cropRgbHost(const uint8_t* src, int64_t rs, int ps, const sm_crop_req& r) {
  const int bw = r.x1 - r.x0, bh = r.y1 - r.y0;
  for (int j = 0; j < r.out_h; ++j) {
    const int ya = r.y0 + j * bh / r.out_h, yb = std::max(ya + 1, r.y0 + (j + 1) * bh / r.out_h);
    for (int i = 0; i < r.out_w; ++i) {
      const int xa = r.x0 + i * bw / r.out_w, xb = std::max(xa + 1, r.x0 + (i + 1) * bw / r.out_w);
      uint32_t s[3] = {0, 0, 0};
      for (int y = ya; y < yb; ++y) {
        const uint8_t* p = src + y * rs + int64_t(xa) * ps;
        for (int x = xa; x < xb; ++x, p += ps) { s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; }
      }
      const uint32_t n = uint32_t((yb - ya) * (xb - xa));
      uint8_t* d = r.dst + (size_t(j) * r.out_w + i) * 3;
      for (int c = 0; c < 3; ++c) d[c] = uint8_t((s[c] + n / 2) / n);
    }
  }
}

void cropDepthMm(const float* dm, int dw, int dh, int img_w, int img_h, const int32_t box[4], int ow, int oh, uint16_t* dst) {
  const double sx = img_w > 0 ? double(dw) / img_w : 1.0, sy = img_h > 0 ? double(dh) / img_h : 1.0;
  const double bw = box[2] - box[0], bh = box[3] - box[1];
  for (int j = 0; j < oh; ++j) {
    const int v = std::clamp(int((box[1] + (j + 0.5) * bh / oh) * sy), 0, dh - 1);
    for (int i = 0; i < ow; ++i) {
      const int u = std::clamp(int((box[0] + (i + 0.5) * bw / ow) * sx), 0, dw - 1);
      const float z = dm[size_t(v) * dw + u];
      dst[size_t(j) * ow + i] = (z > 0 && z < 65.535f) ? uint16_t(std::lround(z * 1000.f)) : 0;
    }
  }
}

void cropMask(const sm_detections* d, int k, const int32_t box[4], int ow, int oh, std::vector<uint8_t>* out) {
  out->assign(size_t(ow) * oh, 0);
  if (!d || !d->mask_bits || d->mask_w <= 0 || d->mask_h <= 0) return;
  const size_t words = (size_t(d->mask_w) * d->mask_h + 31) / 32;
  const uint32_t* bits = d->mask_bits + size_t(k) * words;
  const double bw = box[2] - box[0], bh = box[3] - box[1];
  for (int j = 0; j < oh; ++j) {
    const double y = box[1] + (j + 0.5) * bh / oh;
    const int mj = int(std::floor((y - d->mask_oy) / d->mask_sy));
    if (mj < 0 || mj >= d->mask_h) continue;
    for (int i = 0; i < ow; ++i) {
      const double x = box[0] + (i + 0.5) * bw / ow;
      const int mi = int(std::floor((x - d->mask_ox) / d->mask_sx));
      if (mi < 0 || mi >= d->mask_w) continue;
      const size_t c = size_t(mj) * d->mask_w + mi;
      if ((bits[c >> 5] >> (c & 31)) & 1u) (*out)[size_t(j) * ow + i] = 255;
    }
  }
}

void gatherRgbHost(const uint8_t* src, int64_t rs, int ps, int w, int h, const int32_t* xy, int n, uint8_t* rgb) {
  for (int i = 0; i < n; ++i) {
    const int x = std::clamp(xy[2 * i], 0, w - 1), y = std::clamp(xy[2 * i + 1], 0, h - 1);
    const uint8_t* p = src + y * rs + int64_t(x) * ps;
    rgb[3 * i] = p[0];
    rgb[3 * i + 1] = p[1];
    rgb[3 * i + 2] = p[2];
  }
}

}  // namespace scenemap
