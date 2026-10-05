// sgrt 기록(SGRT_RECORD, 형식은 runtime/src/sgrt.cpp 머리말) 읽기 — 이 하나를 sm_bench·sgrt_replay·og2sg 가 같이 씀.
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

// 레코드를 하나씩 읽는다(기록은 판마다 수 GB — 통째로 올리지 않음)
class Reader {
 public:
  Reader() = default;
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  ~Reader() { if (f_) std::fclose(f_); }
  bool open(const char* path) {
    f_ = std::fopen(path, "rb");
    if (!f_) return false;
    char magic[4];
    uint32_t ver;
    return std::fread(magic, 1, 4, f_) == 4 && !std::memcmp(magic, "SGRC", 4) && rd(f_, &ver) && ver == 1;
  }
  // 다음 레코드(끝·잘림이면 false)
  bool next(Rec* r) {
    FILE* f = f_;
    const int t = std::fgetc(f);
    if (t == EOF) return false;
    r->tag = char(t);
    if (!rd(f, &r->stamp)) return false;
    if (t == 'P') {
      int32_t n;
      if (!rd(f, &n)) return false;
      r->f.resize(size_t(n));
      return std::fread(r->f.data(), 4, size_t(n), f) == size_t(n);
    }
    if (t == 'G') return std::fread(r->g, 8, 3, f) == 3;
    if (t == 'L') {   // 2D 라이다 스캔
      int32_t n;
      if (!rd(f, &n) || std::fread(r->scan, 8, 5, f) != 5) return false;
      r->f.resize(size_t(n));
      return std::fread(r->f.data(), 4, size_t(n), f) == size_t(n);
    }
    if (t != 'I') { std::fprintf(stderr, "sgrec: bad tag %d\n", t); return false; }
    int32_t w, h;
    if (!rd(f, &w) || !rd(f, &h) || std::fread(r->K, 8, 4, f) != 4) return false;
    r->w = w; r->h = h;
    r->f.resize(size_t(w) * h);
    if (std::fread(r->f.data(), 4, r->f.size(), f) != r->f.size()) return false;
    r->rgba.clear();
    if (std::fgetc(f) == 1) {
      rgb_.resize(size_t(w) * h * 3);
      if (std::fread(rgb_.data(), 1, rgb_.size(), f) != rgb_.size()) return false;
      r->rgba.resize(size_t(w) * h * 4);
      for (size_t i = 0; i < size_t(w) * h; ++i) {
        r->rgba[4 * i] = rgb_[3 * i]; r->rgba[4 * i + 1] = rgb_[3 * i + 1]; r->rgba[4 * i + 2] = rgb_[3 * i + 2]; r->rgba[4 * i + 3] = 255;
      }
    }
    int32_t n, iw, ih, mw, mh;
    if (!rd(f, &n) || !rd(f, &iw) || !rd(f, &ih) || !rd(f, &mw) || !rd(f, &mh) || !rd(f, &r->msx) || !rd(f, &r->msy) ||
        !rd(f, &r->mox) || !rd(f, &r->moy))
      return false;
    r->n = n; r->img_w = iw; r->img_h = ih; r->mask_w = mw; r->mask_h = mh;
    const size_t words = (size_t(mw) * mh + 31) / 32;
    r->cls.resize(size_t(n)); r->score.resize(size_t(n)); r->box.resize(4 * size_t(n)); r->bits.resize(words * size_t(n));
    return !n || (std::fread(r->cls.data(), 4, size_t(n), f) == size_t(n) && std::fread(r->score.data(), 4, size_t(n), f) == size_t(n) &&
                  std::fread(r->box.data(), 4, r->box.size(), f) == r->box.size() && std::fread(r->bits.data(), 4, r->bits.size(), f) == r->bits.size());
  }

 private:
  FILE* f_ = nullptr;
  std::vector<uint8_t> rgb_;
};

// 통째로 읽기(작은 기록·시험용). max_frames > 0 이면 'P' 그만큼에서 멈춤
inline bool load(const char* path, std::vector<Rec>* out, int max_frames) {
  Reader rd;
  if (!rd.open(path)) return false;
  int frames = 0;
  Rec r;
  while (rd.next(&r)) {
    if (r.tag == 'P' && max_frames > 0 && frames++ >= max_frames) break;
    out->push_back(r);
  }
  return !out->empty();
}
}  // namespace sgrec
