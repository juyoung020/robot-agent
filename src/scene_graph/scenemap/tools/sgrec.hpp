// sgrt 기록(SGRT_RECORD, runtime/src/sgrt.cpp 의 형식) 읽기 — sm_bench·stage_bench 가 같이 씀.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace sgrec {
struct Rec {
  char tag;
  double stamp;
  std::vector<float> f;       // P: proprio, I: 깊이
  double g[3];                // G
  double scan[5];             // L: angle_min, angle_inc, time_inc, range_min, range_max (f = 거리)
  int w = 0, h = 0;
  double K[4];
  std::vector<uint8_t> rgba;  // I: RGBA(호스트 자르기용)
  // 검출
  int n = 0, img_w = 0, img_h = 0, mask_w = 0, mask_h = 0;
  float msx = 0, msy = 0, mox = 0, moy = 0;
  std::vector<int32_t> cls;
  std::vector<float> score, box;
  std::vector<uint32_t> bits;
};

template <class T>
inline bool rd(FILE* f, T* v) { return std::fread(v, sizeof(T), 1, f) == 1; }

inline bool load(const char* path, std::vector<Rec>* out, int max_frames) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t ver;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "SGRC", 4) || !rd(f, &ver) || ver != 1) return false;
  int frames = 0;
  for (;;) {
    const int t = std::fgetc(f);
    if (t == EOF) break;
    Rec r;
    r.tag = char(t);
    if (!rd(f, &r.stamp)) break;
    if (t == 'P') {
      int32_t n;
      if (!rd(f, &n)) break;
      r.f.resize(n);
      if (std::fread(r.f.data(), 4, n, f) != size_t(n)) break;
      if (max_frames > 0 && frames >= max_frames) break;
      ++frames;
    } else if (t == 'G') {
      if (std::fread(r.g, 8, 3, f) != 3) break;
    } else if (t == 'L') {   // 2D 라이다 스캔(10-06)
      int32_t n;
      if (!rd(f, &n) || std::fread(r.scan, 8, 5, f) != 5) break;
      r.f.resize(size_t(n));
      if (std::fread(r.f.data(), 4, size_t(n), f) != size_t(n)) break;
    } else if (t == 'I') {
      int32_t w, h;
      if (!rd(f, &w) || !rd(f, &h) || std::fread(r.K, 8, 4, f) != 4) break;
      r.w = w; r.h = h;
      r.f.resize(size_t(w) * h);
      if (std::fread(r.f.data(), 4, r.f.size(), f) != r.f.size()) break;
      const int has = std::fgetc(f);
      if (has == 1) {
        std::vector<uint8_t> rgb(size_t(w) * h * 3);
        if (std::fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) break;
        r.rgba.resize(size_t(w) * h * 4);
        for (size_t i = 0; i < size_t(w) * h; ++i) {
          r.rgba[4 * i] = rgb[3 * i]; r.rgba[4 * i + 1] = rgb[3 * i + 1]; r.rgba[4 * i + 2] = rgb[3 * i + 2]; r.rgba[4 * i + 3] = 255;
        }
      }
      int32_t n, iw, ih, mw, mh;
      if (!rd(f, &n) || !rd(f, &iw) || !rd(f, &ih) || !rd(f, &mw) || !rd(f, &mh) || !rd(f, &r.msx) || !rd(f, &r.msy) ||
          !rd(f, &r.mox) || !rd(f, &r.moy))
        break;
      r.n = n; r.img_w = iw; r.img_h = ih; r.mask_w = mw; r.mask_h = mh;
      if (n) {
        const size_t words = (size_t(mw) * mh + 31) / 32;
        r.cls.resize(n); r.score.resize(n); r.box.resize(4 * size_t(n)); r.bits.resize(words * n);
        if (std::fread(r.cls.data(), 4, n, f) != size_t(n) || std::fread(r.score.data(), 4, n, f) != size_t(n) ||
            std::fread(r.box.data(), 4, r.box.size(), f) != r.box.size() || std::fread(r.bits.data(), 4, r.bits.size(), f) != r.bits.size())
          break;
      }
    } else {
      std::fprintf(stderr, "bad tag %d\n", t);
      break;
    }
    out->push_back(std::move(r));
  }
  std::fclose(f);
  return !out->empty();
}
}  // namespace sgrec
