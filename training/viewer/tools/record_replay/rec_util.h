// 기록 도구 공통(학습기와 무관한 호스트 코드): 작은 JSON 조립, 기준 JPEG 부호기(카메라 영상 칸), 실행 폴더 경로·판 번호.
// .trp 바이트는 trpc(Rust trainfmt 의 C ABI)가 쓴다 — 형식 정의는 trainfmt 한 곳.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "trpc/trpc.h"

namespace rec {

// ---------------------------------------------------------------- JSON
inline std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[48];
  std::snprintf(b, sizeof b, "%.7g", v);
  return b;
}
inline std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += c;
  }
  return o + "\"";
}
struct Obj {   // {"k": v, ...} 를 차례대로
  std::string s = "{";
  Obj& raw(const char* k, const std::string& v) { if (s.size() > 1) s += ','; s += jstr(k) + ':' + v; return *this; }
  Obj& num(const char* k, double v) { return raw(k, jnum(v)); }
  Obj& str(const char* k, const std::string& v) { return raw(k, jstr(v)); }
  Obj& b(const char* k, bool v) { return raw(k, v ? "true" : "false"); }
  std::string done() const { return s + "}"; }
};
inline std::string arr3(const float* p) { return "[" + jnum(p[0]) + "," + jnum(p[1]) + "," + jnum(p[2]) + "]"; }

// ---------------------------------------------------------------- 경로
inline void mkdirs(const std::string& p) {
  std::string cur;
  for (size_t i = 0; i <= p.size(); ++i) {
    if (i == p.size() || p[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); }
    if (i < p.size()) cur += p[i];
  }
}
// 줄기 폴더: s_<split>/ (TRAIN_VIEWER 3.1 — 평가 판은 본 집계에 섞지 않는다)
struct Out {
  std::string run, split, dir, rep, eps;
  Out(const std::string& run_, const std::string& split_) : run(run_), split(split_) {
    dir = split.empty() || split == "main" ? run : run + "/s_" + split;
    rep = dir + "/replays";
    eps = split.empty() || split == "main" ? run + "/episodes.jsonl" : dir + "/episodes_" + split + ".jsonl";
    mkdirs(rep);
  }
  // 재개해도 판 번호를 이어 센다(파일 이름이 겹치지 않게 — 교훈 14): 지금 있는 줄 수
  long next_ep() const {
    FILE* f = std::fopen(eps.c_str(), "rb");
    if (!f) return 0;
    long n = 0;
    int c;
    while ((c = std::fgetc(f)) != EOF) n += c == '\n';
    std::fclose(f);
    return n;
  }
  void append_episode(const std::string& line) const {
    FILE* f = std::fopen(eps.c_str(), "ab");
    if (!f) return;
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }
};

// ---------------------------------------------------------------- 기준(baseline) JPEG 부호기, 4:4:4, 표준 허프만 표(ITU T.81 부록 K)
namespace jpeg {
static const uint8_t ZZ[64] = {0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
                               35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
static const uint8_t QL[64] = {16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                               18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
static const uint8_t QC[64] = {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
                               99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};
static const uint8_t DCL_B[16] = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t DCC_B[16] = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
static const uint8_t DC_V[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const uint8_t ACL_B[16] = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
static const uint8_t ACL_V[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1,
    0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
    0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
    0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
    0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
static const uint8_t ACC_B[16] = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
static const uint8_t ACC_V[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1,
    0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36,
    0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
    0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

struct Huff { uint16_t code[256]; uint8_t size[256]; };
inline Huff make(const uint8_t bits[16], const uint8_t* vals) {
  Huff h{};
  int k = 0;
  uint16_t code = 0;
  for (int l = 1; l <= 16; ++l) {
    for (int i = 0; i < bits[l - 1]; ++i, ++k) { h.code[vals[k]] = code++; h.size[vals[k]] = (uint8_t)l; }
    code <<= 1;
  }
  return h;
}
struct Bits {
  std::vector<uint8_t>& o;
  uint32_t acc = 0;
  int n = 0;
  void put(uint32_t v, int len) {
    for (int i = len - 1; i >= 0; --i) {
      acc = (acc << 1) | ((v >> i) & 1);
      if (++n == 8) { o.push_back((uint8_t)acc); if ((uint8_t)acc == 0xFF) o.push_back(0); acc = 0; n = 0; }
    }
  }
  void flush() { while (n) put(1, 1); }
};
inline void seg(std::vector<uint8_t>& o, uint8_t m, const std::vector<uint8_t>& body) {
  o.push_back(0xFF); o.push_back(m); o.push_back((uint8_t)((body.size() + 2) >> 8)); o.push_back((uint8_t)(body.size() + 2));
  o.insert(o.end(), body.begin(), body.end());
}
// rgb: [h][w][3] u8. w·h 가 8 의 배수가 아니면 가장자리 화소를 되풀이한다
inline std::vector<uint8_t> encode(const uint8_t* rgb, int w, int h, int quality = 80) {
  int sc = quality < 50 ? 5000 / quality : 200 - 2 * quality;
  uint8_t ql[64], qc[64];
  for (int i = 0; i < 64; ++i) {
    ql[i] = (uint8_t)std::min(255, std::max(1, (QL[i] * sc + 50) / 100));
    qc[i] = (uint8_t)std::min(255, std::max(1, (QC[i] * sc + 50) / 100));
  }
  static Huff hdl = make(DCL_B, DC_V), hdc = make(DCC_B, DC_V), hal = make(ACL_B, ACL_V), hac = make(ACC_B, ACC_V);
  std::vector<uint8_t> o = {0xFF, 0xD8};
  seg(o, 0xE0, {'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0});
  std::vector<uint8_t> q = {0};
  for (int i = 0; i < 64; ++i) q.push_back(ql[ZZ[i]]);
  q.push_back(1);
  for (int i = 0; i < 64; ++i) q.push_back(qc[ZZ[i]]);
  seg(o, 0xDB, q);
  seg(o, 0xC0, {8, (uint8_t)(h >> 8), (uint8_t)h, (uint8_t)(w >> 8), (uint8_t)w, 3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1});
  std::vector<uint8_t> d;
  auto tab = [&](uint8_t cls_id, const uint8_t bits[16], const uint8_t* vals, int nv) { d.push_back(cls_id); d.insert(d.end(), bits, bits + 16); d.insert(d.end(), vals, vals + nv); };
  tab(0x00, DCL_B, DC_V, 12); tab(0x10, ACL_B, ACL_V, 162); tab(0x01, DCC_B, DC_V, 12); tab(0x11, ACC_B, ACC_V, 162);
  seg(o, 0xC4, d);
  seg(o, 0xDA, {3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0});
  static float C[8][8];
  static bool init = false;
  if (!init) { for (int u = 0; u < 8; ++u) for (int x = 0; x < 8; ++x) C[u][x] = (u ? 0.5f : 0.35355339f) * std::cos((2 * x + 1) * u * 3.14159265f / 16.f); init = true; }
  Bits bw{o};
  int pdc[3] = {0, 0, 0};
  auto cat = [](int v) { int a = v < 0 ? -v : v, n = 0; while (a) { ++n; a >>= 1; } return n; };
  for (int by = 0; by < h; by += 8)
    for (int bx = 0; bx < w; bx += 8) {
      float blk[3][64];
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
          const int sx = std::min(bx + x, w - 1), sy = std::min(by + y, h - 1);
          const uint8_t* p = rgb + ((size_t)sy * w + sx) * 3;
          const float r = p[0], g = p[1], b = p[2];
          blk[0][y * 8 + x] = 0.299f * r + 0.587f * g + 0.114f * b - 128.f;
          blk[1][y * 8 + x] = -0.168736f * r - 0.331264f * g + 0.5f * b;
          blk[2][y * 8 + x] = 0.5f * r - 0.418688f * g - 0.081312f * b;
        }
      for (int c = 0; c < 3; ++c) {
        float tmp[64], F[64];
        for (int y = 0; y < 8; ++y) for (int u = 0; u < 8; ++u) { float s = 0; for (int x = 0; x < 8; ++x) s += C[u][x] * blk[c][y * 8 + x]; tmp[y * 8 + u] = s; }
        for (int v = 0; v < 8; ++v) for (int u = 0; u < 8; ++u) { float s = 0; for (int y = 0; y < 8; ++y) s += C[v][y] * tmp[y * 8 + u]; F[v * 8 + u] = s; }
        const uint8_t* qt = c ? qc : ql;
        int z[64];
        for (int k = 0; k < 64; ++k) z[k] = (int)std::lround(F[ZZ[k]] / qt[ZZ[k]]);
        const Huff& hd = c ? hdc : hdl;
        const Huff& ha = c ? hac : hal;
        const int diff = z[0] - pdc[c];
        pdc[c] = z[0];
        int n = cat(diff);
        bw.put(hd.code[n], hd.size[n]);
        if (n) bw.put(diff < 0 ? (uint32_t)(diff - 1) & ((1u << n) - 1) : (uint32_t)diff, n);
        int run = 0;
        for (int k = 1; k < 64; ++k) {
          if (!z[k]) { ++run; continue; }
          while (run > 15) { bw.put(ha.code[0xF0], ha.size[0xF0]); run -= 16; }
          n = cat(z[k]);
          const int sym = (run << 4) | n;
          bw.put(ha.code[sym], ha.size[sym]);
          bw.put(z[k] < 0 ? (uint32_t)(z[k] - 1) & ((1u << n) - 1) : (uint32_t)z[k], n);
          run = 0;
        }
        if (run) bw.put(ha.code[0], ha.size[0]);
      }
    }
  bw.flush();
  o.push_back(0xFF); o.push_back(0xD9);
  return o;
}
}  // namespace jpeg

}  // namespace rec
