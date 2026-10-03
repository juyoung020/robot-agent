// 최소 PNG 쓰기(include/scenemap/png.hpp). 덩어리: 서명, IHDR, IDAT 하나, IEND. CRC 는 zlib crc32.
#include "scenemap/png.hpp"

#include <zlib.h>

#include <vector>

namespace scenemap {
namespace {

void put32(std::string& o, uint32_t v) {
  o += char(v >> 24);
  o += char(v >> 16);
  o += char(v >> 8);
  o += char(v);
}

void chunk(std::string& o, const char* type, const uint8_t* d, size_t n) {
  put32(o, uint32_t(n));
  const size_t at = o.size();
  o.append(type, 4);
  o.append(reinterpret_cast<const char*>(d), n);
  put32(o, uint32_t(crc32(0, reinterpret_cast<const Bytef*>(o.data() + at), uInt(n + 4))));
}

// rows: h 줄, 줄마다 stride 바이트(빅 엔디언 표본). bpp = 화소 바이트(Sub 필터 거리)
std::string encode(const uint8_t* rows, int w, int h, int bpp, int bit_depth, int ctype, int level) {
  if (w <= 0 || h <= 0) return {};
  const size_t stride = size_t(w) * bpp;
  std::vector<uint8_t> f(size_t(h) * (stride + 1));
  for (int y = 0; y < h; ++y) {
    const uint8_t* in = rows + y * stride;
    uint8_t* out = &f[y * (stride + 1)];
    out[0] = 1;   // Sub
    for (size_t i = 0; i < stride; ++i) out[1 + i] = uint8_t(in[i] - (i >= size_t(bpp) ? in[i - bpp] : 0));
  }
  uLongf zn = compressBound(uLong(f.size()));
  std::vector<uint8_t> z(zn);
  if (compress2(z.data(), &zn, f.data(), uLong(f.size()), level) != Z_OK) return {};
  std::string o("\x89PNG\r\n\x1a\n", 8);
  uint8_t ih[13];
  for (int k = 0; k < 4; ++k) { ih[k] = uint8_t(uint32_t(w) >> (24 - 8 * k)); ih[4 + k] = uint8_t(uint32_t(h) >> (24 - 8 * k)); }
  ih[8] = uint8_t(bit_depth);
  ih[9] = uint8_t(ctype);
  ih[10] = ih[11] = ih[12] = 0;
  chunk(o, "IHDR", ih, 13);
  chunk(o, "IDAT", z.data(), zn);
  chunk(o, "IEND", nullptr, 0);
  return o;
}

}  // namespace

std::string pngRgb8(const uint8_t* rgb, int w, int h, int level) { return encode(rgb, w, h, 3, 8, 2, level); }

std::string pngGray8(const uint8_t* v, int w, int h, int level) { return encode(v, w, h, 1, 8, 0, level); }

std::string pngGray16(const uint16_t* v, int w, int h, int level) {
  std::vector<uint8_t> be(size_t(w) * h * 2);
  for (size_t i = 0; i < size_t(w) * h; ++i) {
    be[2 * i] = uint8_t(v[i] >> 8);
    be[2 * i + 1] = uint8_t(v[i]);
  }
  return encode(be.data(), w, h, 2, 16, 0, level);
}

}  // namespace scenemap
