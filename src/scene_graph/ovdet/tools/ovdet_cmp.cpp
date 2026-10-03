// 검출기 비교(실제 영상): 엔진 여러 개를 같은 프레임들에 돌려 프레임당 검출 수·서로 다른 이름 수·ms 를 찍고, 프레임마다
// 마스크·상자를 칠한 그림(PPM)을 쓴다. 프롬프트는 엔진 어휘 전부(NULL).
//   ovdet_cmp <out_dir> <w> <h> <frames.rgb ...> -- <engine.plan ...>
//   frames.rgb: w×h×3 RGB u8 원 영상. 그림: <out_dir>/<frame 이름>__<engine 이름>.ppm
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "ovdet.h"

static std::string base(const std::string& p) {
  std::string b = p.substr(p.find_last_of('/') + 1);
  const size_t d = b.find_last_of('.');
  return d == std::string::npos ? b : b.substr(0, d);
}

// 3×5 숫자 글꼴 없이: 이름 대신 색으로 구분하고 표준 출력에 이름표를 남김
static void draw(std::vector<uint8_t>& im, int w, int h, const sm_detections* d) {
  static const uint8_t pal[8][3] = {{230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200},
                                    {245, 130, 48}, {145, 30, 180}, {70, 240, 240}, {240, 50, 230}};
  const size_t words = (size_t(d->mask_w) * d->mask_h + 31) / 32;
  for (int k = 0; k < d->n; ++k) {
    const uint8_t* c = pal[k % 8];
    const uint32_t* bits = d->mask_bits + k * words;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const int i = int((x + 0.5f - d->mask_ox) / d->mask_sx), j = int((y + 0.5f - d->mask_oy) / d->mask_sy);
        if (i < 0 || j < 0 || i >= d->mask_w || j >= d->mask_h) continue;
        const size_t cc = size_t(j) * d->mask_w + i;
        if (!((bits[cc >> 5] >> (cc & 31)) & 1u)) continue;
        uint8_t* p = &im[(size_t(y) * w + x) * 3];
        for (int q = 0; q < 3; ++q) p[q] = uint8_t((p[q] + c[q]) / 2);
      }
    const float* b = d->box + 4 * k;
    const int x0 = std::clamp(int(b[0]), 0, w - 1), x1 = std::clamp(int(b[2]) - 1, 0, w - 1);
    const int y0 = std::clamp(int(b[1]), 0, h - 1), y1 = std::clamp(int(b[3]) - 1, 0, h - 1);
    for (int t = 0; t < 2; ++t) {
      for (int x = x0; x <= x1; ++x) {
        std::memcpy(&im[(size_t(std::min(y0 + t, h - 1)) * w + x) * 3], c, 3);
        std::memcpy(&im[(size_t(std::max(y1 - t, 0)) * w + x) * 3], c, 3);
      }
      for (int y = y0; y <= y1; ++y) {
        std::memcpy(&im[(size_t(y) * w + std::min(x0 + t, w - 1)) * 3], c, 3);
        std::memcpy(&im[(size_t(y) * w + std::max(x1 - t, 0)) * 3], c, 3);
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: ovdet_cmp <out_dir> <w> <h> <frames.rgb ...> -- <engine.plan ...>\n");
    return 2;
  }
  const std::string out = argv[1];
  const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
  std::vector<std::string> frames, engines;
  bool eng = false;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--")) { eng = true; continue; }
    (eng ? engines : frames).push_back(argv[i]);
  }
  std::vector<std::vector<uint8_t>> img(frames.size());
  for (size_t f = 0; f < frames.size(); ++f) {
    img[f].resize(size_t(W) * H * 3);
    FILE* fp = std::fopen(frames[f].c_str(), "rb");
    if (!fp || std::fread(img[f].data(), 1, img[f].size(), fp) != img[f].size()) {
      std::fprintf(stderr, "cannot read %s\n", frames[f].c_str());
      return 1;
    }
    std::fclose(fp);
  }
  std::printf("| engine | input | vocab | det/frame | distinct names | total ms/frame | net ms |\n|---|---|---|---|---|---|---|\n");
  std::string details;
  for (const std::string& e : engines) {
    OvdConfig cfg;
    ovd_default_config(&cfg);
    const std::string names = e + ".names.txt";
    cfg.seg_engine = e.c_str();
    cfg.names = names.c_str();
    char err[512];
    OvdHandle* h = ovd_create(&cfg, err, sizeof err);
    if (!h) { std::fprintf(stderr, "%s: %s\n", e.c_str(), err); continue; }
    std::set<int> cls;
    double tot = 0, net = 0;
    int ndet = 0, nt = 0, grid = 0;
    for (size_t f = 0; f < frames.size(); ++f) {
      OvdImage im = {0.0, 0, img[f].data(), H, W, (long long)W * 3, 3, 0, 0};
      OvdTiming t;
      const sm_detections* d = nullptr;
      for (int rep = 0; rep < 6; ++rep) {   // 첫 번은 데움, 뒤 5 번 시간 평균
        d = ovd_detect(h, &im, &t);
        if (!d) break;
        if (rep) { tot += t.total_ms; net += t.net_ms; ++nt; }
      }
      if (!d) { std::fprintf(stderr, "detect: %s\n", ovd_last_error(h)); break; }
      grid = d->mask_w;
      ndet += d->n;
      details += base(frames[f]) + " " + base(e) + ":";
      for (int k = 0; k < d->n; ++k) {
        cls.insert(d->cls[k]);
        char buf[160];
        std::snprintf(buf, sizeof buf, " %s(%.2f)", ovd_vocab_name(h, d->cls[k]), d->score[k]);
        details += buf;
      }
      details += "\n";
      std::vector<uint8_t> a = img[f];
      draw(a, W, H, d);
      const std::string p = out + "/" + base(frames[f]) + "__" + base(e) + ".ppm";
      FILE* o = std::fopen(p.c_str(), "wb");
      std::fprintf(o, "P6\n%d %d\n255\n", W, H);
      std::fwrite(a.data(), 1, a.size(), o);
      std::fclose(o);
    }
    std::printf("| %s | grid %d | %d | %.2f | %zu | %.2f | %.2f |\n", base(e).c_str(), grid, ovd_vocab_size(h), double(ndet) / frames.size(),
                cls.size(), tot / std::max(1, nt), net / std::max(1, nt));
    ovd_destroy(h);
  }
  std::printf("\n%s", details.c_str());
  return 0;
}
